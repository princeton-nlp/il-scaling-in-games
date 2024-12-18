import math

import torch.nn as nn

from il_scale.utils.atari_conf import OBS_SHAPE
from il_scale.utils.model_utils import conv_outdim

FLOP_TO_STR = {
    int(1e13): "1e13",
    int(2e13): "2e13",
    int(5e13): "5e13",
    int(1e14): "1e14",
    int(2e14): "2e14",
    int(5e14): "5e14",
    int(1e15): "1e15",
    int(2e15): "2e15",
    int(5e15): "5e15",
    int(1e16): "1e16",
    int(2e16): "2e16",
    int(5e16): "5e16",
    int(1e17): "1e17",
    int(2e17): "2e17",
    int(5e17): "5e17",
    int(1e18): "1e18",
}


class FLOPCounter:
    """
    Assume backward pass is 2x FLOPS of forward pass.

    Ignored FLOPs:
        - MaxPool
        - Nonlinearities
        - Biases
    """

    def __init__(self):
        self.h, self.w = OBS_SHAPE[:2]

    def count_flops(self, model: nn.Module, samples: int = 1):
        """ """
        flops = 0
        h_out, w_out = self.h, self.w

        modules = model.modules()

        # first count all flops
        for m in modules:
            flops += self._flops_in_module(m, h_out, w_out)
            h_out, w_out = self._maybe_update_obs_shape(m, h_out, w_out)

        forward_flops = flops
        backward_flops = 2 * forward_flops

        return {
            "forward_flops": forward_flops * samples,
            "backward_flops": backward_flops * samples,
            "total_flops": (forward_flops + backward_flops) * samples,
        }

    def _flops_in_module(
        self, m: nn.Module, h_out: int, w_out: int, ignore_embs: bool = False
    ):
        if isinstance(m, nn.Conv2d):
            flops = self.conv_layer(
                m.kernel_size[0], m.in_channels, m.out_channels, h_out, w_out
            )

        elif isinstance(m, nn.Linear):
            flops = self.dense_layer(m.in_features, m.out_features)

        elif isinstance(m, nn.LSTM):
            flops = self.lstm_layer(m.input_size, m.hidden_size)

        elif isinstance(m, nn.Embedding) and not ignore_embs:
            flops = self.embedding_layer(m.num_embeddings, m.embedding_dim)

        else:
            flops = 0

        return flops

    def _maybe_update_obs_shape(self, m: nn.Module, h_out: int, w_out: int):
        if isinstance(m, nn.Conv2d):
            h_out = conv_outdim(h_out, k=m.kernel_size[0], stride=m.stride[0])
            w_out = conv_outdim(w_out, k=m.kernel_size[0], stride=m.stride[0])

        return h_out, w_out

    def embedding_layer(self, num_embeddings: int, embedding_dim: int):
        """ """
        return 2 * num_embeddings * embedding_dim * self.h * self.w

    def conv_layer(
        self, kernel_size: int, c_in: int, c_out: int, h_out: int, w_out: int
    ):
        """
        Conv layers takes 2 x h_out x w_out per parameter.
        """
        params = kernel_size**2 * c_in
        return 2 * h_out * w_out * c_out * params

    def dense_layer(self, in_features: int, out_features: int):
        """ """
        return 2 * in_features * out_features

    def lstm_layer(self, input_size: int, hidden_size: int):
        """ """
        # count i_t, f_t, g_t, o_t
        i_flops = 2 * input_size * hidden_size + 2 * hidden_size * hidden_size
        f_flops = i_flops
        g_flops = i_flops
        o_flops = i_flops

        # count c_t
        c_flops = 3 * hidden_size

        # count h_t
        h_flops = hidden_size

        return i_flops + f_flops + g_flops + o_flops + c_flops + h_flops
