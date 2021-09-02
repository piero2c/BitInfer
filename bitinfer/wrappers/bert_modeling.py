'''
    Wrappers for Bert submodules adapted from
    https://github.com/huggingface/transformers/blob/master/src/transformers/models/bert/modeling_bert.py.
    This implementation is far from optimal and will likely be rewritten in a more concise way in the future (using Flax maybe?)
'''

import torch
import math
from torch import nn
from transformers.models.bert.modeling_bert import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions
)


class BitAdditiveBias(nn.Module):
    ''' Wraps BertEmbeddings, Linear and LayerNorm layers '''

    def __init__(self, module, param_name, expand_seq_axis=False):
        super().__init__()
        self.module = module
        self.param_name = param_name
        self.expand_seq_axis = expand_seq_axis

    def forward(self, *args, offsets=None, **kwargs):
        assert not self.module.training

        output = self.module(*args, **kwargs)

        if offsets is not None and self.param_name in offsets:
            bias = offsets[self.param_name]

            if self.expand_seq_axis:
                bias = bias.resize(-1, 1, bias.shape[0])

            return output + bias

        return output


class BitSelfAttention(nn.Module):
    ''' Wraps BertSelfAttention Layer.
        The code is mainly copied from HF/transformers, altough the decoder
        and crossatention support is removed. '''

    def __init__(self, module, prefix=''):
        super().__init__()

        module.query, module.key, module.value = (
            BitAdditiveBias(module.query, prefix + '.query.bias'),
            BitAdditiveBias(module.key, prefix + '.key.bias'),
            BitAdditiveBias(module.value, prefix + '.value.bias')
        )
        self.module = module

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        offsets=None
    ):
        mixed_query_layer = self.module.query(hidden_states,
                                              offsets=offsets)

        if encoder_hidden_states is not None or self.module.is_decoder:
            raise NotImplementedError

        if past_key_value is not None:
            key_layer = self.module.transpose_for_scores(
                self.module.key(hidden_states, offsets=offsets)
            )
            value_layer = self.module.transpose_for_scores(
                self.module.value(hidden_states, offsets=offsets)
            )
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.module.transpose_for_scores(
                self.module.key(hidden_states, offsets=offsets)
            )
            value_layer = self.module.transpose_for_scores(
                self.module.value(hidden_states, offsets=offsets)
            )

        query_layer = self.module.transpose_for_scores(mixed_query_layer)
        attention_scores = torch.matmul(query_layer,
                                        key_layer.transpose(-1, -2))

        if self.module.position_embedding_type == 'relative_key' or\
           self.module.position_embedding_type == 'relative_key_query':
            seq_length = hidden_states.size()[1]
            position_ids_l = torch.arange(seq_length, dtype=torch.long,
                                          device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(seq_length, dtype=torch.long,
                                          device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r
            positional_embedding = self.module.distance_embedding(
                distance + self.module.max_position_embeddings - 1
            )
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)

            if self.module.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores

            elif self.module.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.module.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.module.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.module.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.module.is_decoder:
            outputs = outputs + (past_key_value,)

        return outputs


class BitSelfOutput(nn.Module):
    ''' Wraps BertSelfOutput '''
    def __init__(self, module, prefix=''):
        super().__init__()
        module.dense = BitAdditiveBias(
            module.dense, prefix + '.dense.bias'
        )
        module.LayerNorm = BitAdditiveBias(
            module.LayerNorm, prefix + '.LayerNorm.bias'
        )
        self.module = module

    def forward(self, hidden_states, input_tensor, offsets=None):
        hidden_states = self.module.dense(hidden_states, offsets=offsets)
        hidden_states = self.module.dropout(hidden_states)
        hidden_states = self.module.LayerNorm(hidden_states + input_tensor,
                                              offsets=offsets)
        return hidden_states


class BitAttention(nn.Module):
    ''' Wraps BertAttention '''
    def __init__(self, module, prefix=''):
        super().__init__()
        module.self = BitSelfAttention(module.self, prefix=prefix + '.self')
        module.output = BitSelfOutput(module.output, prefix=prefix + '.output')
        self.module = module

    def forward(self,
                hidden_states,
                attention_mask=None,
                head_mask=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                past_key_value=None,
                output_attentions=False,
                offsets=None):
        self_outputs = self.module.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
            offsets=offsets
        )
        attention_output = self.module.output(
            self_outputs[0], hidden_states, offsets=offsets
        )
        outputs = (attention_output,) + self_outputs[1:]
        return outputs


class BitIntermediate(nn.Module):
    ''' Wraps BertIntermediate '''
    def __init__(self, module, prefix=''):
        super().__init__()
        module.dense = BitAdditiveBias(
            module.dense, prefix + '.intermediate.dense.bias'
        )
        self.module = module

    def forward(self, hidden_states, offsets=None):
        hidden_states = self.module.dense(hidden_states, offsets=offsets)
        hidden_states = self.module.intermediate_act_fn(hidden_states)
        return hidden_states


class BitOutput(nn.Module):
    ''' Wraps BertOutput '''
    def __init__(self, module, prefix=''):
        super().__init__()
        module.dense = BitAdditiveBias(
            module.dense, prefix + '.output.dense.bias'
        )
        module.LayerNorm = BitAdditiveBias(
            module.LayerNorm, prefix + '.output.LayerNorm.bias'
        )
        self.module = module

    def forward(self, hidden_states, input_tensor, offsets=None):
        hidden_states = self.module.dense(hidden_states, offsets=offsets)
        hidden_states = self.module.dropout(hidden_states)
        hidden_states = self.module.LayerNorm(
            hidden_states + input_tensor, offsets=offsets
        )
        return hidden_states


class BitLayer(nn.Module):
    ''' Wraps BertLayer '''
    def __init__(self, module, prefix=''):
        super().__init__()
        module.attention = BitAttention(module.attention, prefix=prefix + '.attention')
        module.intermediate = BitIntermediate(module.intermediate, prefix=prefix)
        module.output = BitOutput(module.output, prefix=prefix)
        self.module = module

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        offsets=None
    ):
        if self.module.is_decoder:
            raise NotImplementedError

        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.module.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
            offsets=offsets
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]

        if self.module.chunk_size_feed_forward > 0:
            raise NotImplementedError

        layer_output = self.feed_forward_chunk(
            attention_output, offsets=offsets
        )
        outputs = (layer_output,) + outputs

        return outputs

    def feed_forward_chunk(self, attention_output, **kwargs):
        if 'offsets' in kwargs:
            offsets = kwargs['offsets']

        intermediate_output = self.module.intermediate(
            attention_output, offsets=offsets
        )
        layer_output = self.module.output(
            intermediate_output, attention_output, offsets=offsets
        )
        return layer_output


class BitEncoder(nn.Module):
    ''' Wraps BertEncoder '''
    def __init__(self, module):
        super().__init__()
        for i, layer in enumerate(module.layer):
            module.layer[i] = BitLayer(layer, prefix=f'bert.encoder.layer.{i}')
        self.module = module

    def forward(self,
                hidden_states,
                attention_mask=None,
                head_mask=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                past_key_values=None,
                use_cache=None,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
                offsets=None):
        assert not self.module.training

        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = None

        next_decoder_cache = () if use_cache else None
        for i, layer_module in enumerate(self.module.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                layer_head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                past_key_value,
                output_attentions,
                offsets=offsets
            )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


class BitPooler(nn.Module):
    ''' Wraps BertPooler '''
    def __init__(self, module):
        super().__init__()
        module.dense = BitAdditiveBias(module.dense, 'bert.pooler.dense.bias')
        self.module = module

    def forward(self, hidden_states, offsets=None):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.module.dense(first_token_tensor, offsets=offsets)
        pooled_output = self.module.activation(pooled_output)
        return pooled_output


class BitModel(nn.Module):
    ''' Wraps BertModel '''
    def __init__(self, module):
        super().__init__()

        module.embeddings = BitAdditiveBias(module.embeddings,
                                            'bert.embeddings.LayerNorm.bias')
        module.encoder = BitEncoder(module.encoder)
        module.pooler = BitPooler(module.pooler)

        self.module = module

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        offsets=None
    ):
        assert not self.module.config.is_decoder

        output_attentions = output_attentions if output_attentions is not None else self.module.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.module.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.module.config.use_return_dict

        use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        if token_type_ids is None:
            if hasattr(self.module.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.module.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        extended_attention_mask: torch.Tensor = self.module.get_extended_attention_mask(attention_mask, input_shape, device)

        if self.module.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.module.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        head_mask = self.module.get_head_mask(head_mask, self.module.config.num_hidden_layers)

        embedding_output = self.module.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
            offsets=offsets
        )
        encoder_outputs = self.module.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            offsets=offsets
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.module.pooler(sequence_output, offsets=offsets) if self.module.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )
