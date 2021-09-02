from copy import deepcopy
import base64
from pathlib import Path
import torch
from io import BytesIO
import pytest
from transformers import (
    BertForSequenceClassification, BertConfig, BertTokenizer
)

from bitinfer.inference import TorchDynamicInferenceSession
from bitinfer import helpers

texts = ['Just a text', 'Another one']
text_pairs = [('The cat sat on the mat', 'The mat sat on the cat'),
              ('The cat sat on the mat', 'The kitty sat on the mat')]

dummy_tokenizer = BertTokenizer(Path(__file__).parent / 'dummy_tokenizer/vocab.txt')


@pytest.fixture
def bert_model():
    config = BertConfig(num_hidden_layers=2, hidden_size=24, num_labels=2)
    return BertForSequenceClassification(config).eval()


@pytest.fixture
def inference_session(bert_model):
    return TorchDynamicInferenceSession(deepcopy(bert_model), dummy_tokenizer)


@pytest.fixture
def finetuned_model_1(bert_model):
    helpers.freeze_parameters(bert_model, learnable_biases=['pooler'])

    bert_model.base_model.pooler.dense.bias.data = torch.randn_like(
        bert_model.base_model.pooler.dense.bias.data
    )

    return bert_model


@pytest.fixture
def finetuned_model_2(bert_model):
    helpers.freeze_parameters(bert_model, learnable_biases=['query'])

    encoder_layers = bert_model.base_model.encoder.layer

    for layer in encoder_layers:
        query_bias = layer.attention.self.query.bias
        query_bias.data = torch.randn_like(query_bias)

    return bert_model


@pytest.fixture
def finetuned_model_list(finetuned_model_1, finetuned_model_2):
    return [finetuned_model_1, finetuned_model_2]


def test_predict_from_hash(inference_session, bert_model, finetuned_model_list):
    for ft_model in finetuned_model_list:
        buffer = BytesIO()
        torch.save(helpers.get_offsets(bert_model, ft_model), buffer)
        buffer.seek(0)

        param_hash = base64.b64encode(buffer.read())
        pred = inference_session.predict(texts, param_hash=param_hash)
        assert isinstance(pred, torch.Tensor) and pred.shape == (len(texts), 2)

        pred = inference_session.predict(text_pairs, param_hash=param_hash)
        assert isinstance(pred, torch.Tensor) and pred.shape == (len(texts), 2)


def test_predict_from_sd(inference_session, bert_model, finetuned_model_list):
    for ft_model in finetuned_model_list:
        param_sd = helpers.get_offsets(bert_model, ft_model)

        pred = inference_session.predict(texts, param_dict=param_sd)
        assert isinstance(pred, torch.Tensor) and pred.shape == (len(texts), 2)

        pred = inference_session.predict(text_pairs, param_dict=param_sd)
        assert isinstance(pred, torch.Tensor) and pred.shape == (len(texts), 2)
