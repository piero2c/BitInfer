from io import BytesIO
from collections import OrderedDict
from typing import Union, Dict, List, Tuple
from base64 import b64decode
import torch
import torch.nn.functional as F
from transformers import (
    BertForSequenceClassification,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedTokenizer
)

from .wrappers.bert_modeling import BitModel


class TorchDynamicInferenceSession():
    def __init__(self, base_model: Union[BertForSequenceClassification, str],
                 tokenizer: Union[PreTrainedTokenizer, str] = None,
                 device='cpu'):

        if isinstance(base_model, str):
            assert tokenizer is None
            tokenizer = AutoTokenizer.from_pretrained(base_model)
            base_model = AutoModelForSequenceClassification.from_pretrained(base_model)
        else:
            if isinstance(tokenizer, str):
                tokenizer = AutoTokenizer.from_pretrained(tokenizer)
            assert tokenizer is not None

        self.base_model = BitModel(
            base_model.base_model.eval().to(device)
        )
        self.tokenizer = tokenizer
        self.device = device

    def predict(self, texts: Union[List[str], List[Tuple[str, str]]],
                param_dict: Union[Dict, OrderedDict] = None,
                param_hash: Union[str, bytes] = None,
                param_path: str = None):
        if sum(int(m is not None) for m in [param_dict, param_hash, param_path]) != 1:
            raise ValueError(
                'Please provide only one of [param_dict, param_hash, param_path].'
            )

        if param_hash is not None:
            param_dict = torch.load(BytesIO(b64decode(param_hash)))

        if param_path is not None:
            param_dict = torch.load(open(param_path, 'rb'))

        assert 'classifier' in param_dict and 'offsets' in param_dict
        assert 'classifier.weight' in param_dict['classifier']
        assert 'classifier.bias' in param_dict['classifier']

        # Checks for text pairs
        if all(isinstance(item, tuple) for item in texts):
            inputs = self.tokenizer(*zip(*texts), truncation=True,
                                    padding=True, return_tensors='pt')
        else:
            inputs = self.tokenizer(texts, truncation=True,
                                    padding=True, return_tensors='pt')

        enc_inputs = self.base_model(**inputs.to(self.device),
                                     offsets=param_dict['offsets'])

        # Returns logits
        return F.linear(enc_inputs.pooler_output,
                        param_dict['classifier']['classifier.weight'],
                        param_dict['classifier']['classifier.bias'])

    def predict_multiple(self, input_dict: Dict):
        # Runs multiple bitfit models in a single batch
        # Expects a dict w/ model names, params and texts
        # {'model1': {'params': '', 'texts': []},
        #  'model2': {'params': '', 'texts': []},
        #  ...,
        #  'modeln': {'params': '', 'texts': []}
        #  }
        raise NotImplementedError


class OnnxInferenceSession():
    def __init__(self) -> None:
        ...

    def predict(self, texts: List[str],
                param_dict: Union[Dict, OrderedDict] = None,
                param_hash: Union[str, bytes] = None,
                param_path: str = None):
        raise NotImplementedError

    def predict_multiple(self, input_dict: Dict):
        # Expects a dict w/ model names, params and texts
        # {'model1': {'params': {}, 'texts': []},
        #  'model2': {'params': {}, 'texts': []},
        #  ...,
        #  'modeln': {'params': {}, 'texts': []}
        #  }
        raise NotImplementedError
