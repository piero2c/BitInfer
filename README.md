# BitInfer

This package provides an easy and efficient way to deploy a large pool of custom fine-tuned BERT models with BitFit ([Zaken et. al, arXiv:2106.10199](https://arxiv.org/abs/2106.10199)) using a single Pytorch model instance.

## Usage

**Training and saving a bitfit model**

```python
from bitinfer import helpers
from transformers import BertForSequenceClassification
from copy import deepcopy

# Loads a Bert model for text-classification
finetuned_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)

# We'll need the original model weights later
original_model = deepcopy(model)

# Freezes all model parameters except for query and output bias vectors
helpers.freeze_parameters(finetuned_model, learnable_biases=['query', 'output'])

# Train the model as usual
...
```

Saving the bias vector offsets and classification head takes less than 300Kb of space.

```python
helpers.save_bitfit(original_model, finetuned_model, 'sentiment_analysis.pt')
os.path.getsize('sentiment_analysis.pt')  # 208Kb

# Alternatively, we can also store the model using a base64 hash
model_hash = base64.b64encode(open('sentiment_analysis.pt', 'rb').read())  # 313Kb
```

**Serving multiple bitfit models**

```python
from bitinfer.inference import TorchDynamicInferenceSession

sess = TorchDynamicInferenceSession('bert-base-uncased', device='cpu')

# Let's load three bitfit models finetuned on `bert-base-uncased`
sentiment_analysis_hash = b'UEsDBAAACAg...'
sentence_similarity_model = 'my_model2.pt'
my_nli_model_sd = torch.load('my_nli_model.pt')
```

Since the bitfit parameters are lightweight (< 1MB), we can serve multiple models in real-time with no speed penalty.

```python
%%time
sess.predict(['This movie sucks ;(', 'Loved it!!!'],
             param_hash=sentiment_analysis_hash)
# Wall time: 31 ms

%%time
sess.predict([('The cat sat on the mat', 'Matt sat on the mat'),
              ('The mat sat on the cat', 'Mat matt sat mat')],
             param_path=sentence_similarity_model)
# Wall time: 25 ms
```

## Planned features
* ONNX support
* Single batch multi-model inference (CPU/GPU)

