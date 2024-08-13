import logging
import os
import time

import portalocker
import torch
import wandb

# A logger for this file
logging.basicConfig(
    format=(
        "[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] " "%(message)s"
    ),
    level=logging.INFO,
)

def count_params(m):
    """
    Count number of parameters in model.
    """
    # count total
    total_params = sum(p.numel() for p in m.parameters())

    return total_params

def conv_outdim(i_dim, k, padding=0, stride=1, dilation=1):
    """Return the dimension after applying a convolution along one axis"""
    return int(1 + (i_dim + 2 * padding - dilation * (k - 1) - 1) / stride)

def selectt(embedding_layer, x, use_index_select):
    """Use index select instead of default forward to possible speed up embedding."""
    if use_index_select:
        out = embedding_layer.weight.index_select(0, x.view(-1))
        # handle reshaping x to 1-d and output back to N-d
        return out.view(x.shape + (-1,))
    else:
        return embedding_layer(x)

def _step_to_range(delta, num_steps):
    """Range of `num_steps` integers with distance `delta` centered around zero."""
    return delta * torch.arange(-num_steps // 2, num_steps // 2)

def interleave(*args):
    return [val for pair in zip(*args) for val in pair]

def load_checkpoint(model_load_name: str):
    logging.info(f"Loading model with name {model_load_name}")
    
    checkpoint = torch.load(os.path.join('nethack_files', model_load_name), map_location="cpu")

    model_state_dict_keys = list(checkpoint['model_state_dict'].keys())
    for key in model_state_dict_keys:
        if key.startswith('module'):
            checkpoint['model_state_dict'][key[7:]] = checkpoint['model_state_dict'][key]
            del checkpoint['model_state_dict'][key]

    return checkpoint

