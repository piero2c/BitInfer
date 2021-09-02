import torch
from typing import OrderedDict, Union, List, Dict


def freeze_parameters(hf_model: torch.nn.Module,
                      learnable_biases: Union[str, List[str]] = 'all'):
    learnable_biases = ['bias'] if learnable_biases == 'all' else learnable_biases

    if not isinstance(learnable_biases, list):
        learnable_biases = [learnable_biases]

    for par_name, par_tensor in hf_model.base_model.named_parameters():
        par_tensor.requires_grad = any(
            ('bias' in par_name and kw in par_name) for kw in learnable_biases
        )

    return hf_model


def get_trainable_parameters(hf_model: torch.nn.Module):
    return {
        par_name: par_tensor
        for par_name, par_tensor in hf_model.named_parameters()
        if par_tensor.requires_grad
    }


def get_offsets(base_model: Union[torch.nn.Module, OrderedDict, Dict],
                finetuned_model: Union[torch.nn.Module, OrderedDict, Dict]):
    if isinstance(finetuned_model, torch.nn.Module):
        finetuned_model = get_trainable_parameters(finetuned_model)

    if isinstance(base_model, torch.nn.Module):
        base_model = base_model.state_dict()

    return {
        'offsets': {
            param_name: param_tensor - base_model[param_name]
            for param_name, param_tensor in finetuned_model.items()
            if 'classifier' not in param_name
        },
        'classifier': {
            param_name: param_tensor
            for param_name, param_tensor in finetuned_model.items()
            if 'classifier' in param_name
        }
    }


def save_bitfit(base_model: Union[torch.nn.Module, OrderedDict, Dict],
                finetuned_model: Union[torch.nn.Module, OrderedDict, Dict],
                path: str):
    torch.save(get_offsets(base_model, finetuned_model), path)
