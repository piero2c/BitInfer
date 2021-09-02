import torch
import pytest
from transformers import BertModel, BertConfig
from bitinfer import helpers
from copy import deepcopy


@pytest.fixture
def bert_model():
    config = BertConfig(num_hidden_layers=2, hidden_size=24)
    return BertModel(config)


def test_get_trainable_parameters(bert_model):
    for _, param_tensor in bert_model.named_parameters():
        param_tensor.requires_grad = False

    assert set(helpers.get_trainable_parameters(bert_model).keys()) == set()

    bert_model.base_model.embeddings.word_embeddings.weight.requires_grad = True
    bert_model.base_model.pooler.dense.weight.requires_grad = True

    assert set(helpers.get_trainable_parameters(bert_model).keys()) ==\
           {'embeddings.word_embeddings.weight', 'pooler.dense.weight'}


def test_freeze(bert_model):
    model = bert_model

    helpers.freeze_parameters(model, 'all')
    free_params = set(helpers.get_trainable_parameters(model).keys())
    assert all('bias' in par_name for par_name in free_params)

    helpers.freeze_parameters(model, 'query')
    free_params = set(helpers.get_trainable_parameters(model).keys())
    assert all('query.bias' in par_name for par_name in free_params)

    helpers.freeze_parameters(model, ['query', 'key'])
    free_params = set(helpers.get_trainable_parameters(model).keys())
    assert all('query.bias' in par_name or 'key.bias' in par_name
               for par_name in free_params)


def test_get_offsets(bert_model):
    original_model = deepcopy(bert_model)

    helpers.freeze_parameters(bert_model, learnable_biases='pooler')
    bert_model.base_model.pooler.dense.bias.data = torch.randn_like(
        bert_model.base_model.pooler.dense.bias
    )

    sd = helpers.get_offsets(original_model, bert_model)

    assert set(sd.keys()) == {'classifier', 'offsets'}
    assert set(sd['offsets'].keys()) == {'pooler.dense.bias'}
