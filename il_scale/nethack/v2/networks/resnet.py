import math

import torch.nn as nn
import torch
from omegaconf import DictConfig

from il_scale.nethack.v2.utils.model import selectt

class CharColorEncoderResnet(nn.Module):
    """
    Inspired by network from IMPALA https://arxiv.org/pdf/1802.01561.pdf
    """
    def __init__(
        self, 
        obs_shape,
        cfg: DictConfig
    ):
        super(CharColorEncoderResnet, self).__init__()

        self.cfg = cfg
        self.resnet_scale_channels = self.cfg.network.resnet_scale_channels
        self.hdim = self.cfg.network.resnet_hdim
        self.h, self.w = obs_shape

        self.blocks = []

        self.conv_params = [
            ([cfg.network.char_edim * cfg.network.obs_frame_stack + cfg.network.color_edim * cfg.network.obs_frame_stack, 16 * self.resnet_scale_channels, cfg.network.obs_kernel_size, cfg.network.resnet_num_blocks] if not self.cfg.network.add_char_color else [cfg.network.char_edim * cfg.network.obs_frame_stack, 16, cfg.network.obs_kernel_size, cfg.network.resnet_num_blocks]),
            [16 * self.resnet_scale_channels, 32 * self.resnet_scale_channels, cfg.network.obs_kernel_size, cfg.network.resnet_num_blocks],
            [32 * self.resnet_scale_channels, 32 * self.resnet_scale_channels, cfg.network.obs_kernel_size, cfg.network.resnet_num_blocks]
        ]

        print('resnet scale channels', self.resnet_scale_channels)
        self.conv_params = self.conv_params[:self.cfg.network.obs_conv_blocks]

        for (
            in_channels,
            out_channels,
            filter_size,
            num_res_blocks
        ) in self.conv_params:
            block = []
            # Downsample
            block.append(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    filter_size,
                    stride=1,
                    padding=(filter_size // 2)
                )
            )

            if self.cfg.network.fix_initialization:
                print('fixing resnet first conv initialization ...')
                block[-1].weight.data *= 1.0 / block[-1].weight.norm(
                    dim=tuple(range(1, block[-1].weight.data.ndim)), p=2, keepdim=True
                )

                block[-1].bias.data *= 0

            block.append(
                nn.MaxPool2d(
                    kernel_size=3,
                    stride=2
                )
            )
            self.h = math.floor((self.h - 1 * (3 - 1) - 1)/2 + 1) # from PyTorch Docs
            self.w = math.floor((self.w - 1 * (3 - 1) - 1)/2 + 1) # from PyTorch Docs

            # Residual block(s)
            for _ in range(num_res_blocks):
                block.append(ResBlock(cfg, out_channels, out_channels, filter_size, self.cfg.network.resnet_num_layers))
            self.blocks.append(nn.Sequential(*block))

        self.conv_net = nn.Sequential(*self.blocks)
        self.out_size = self.h * self.w * out_channels

        print('resnet out size', self.out_size)
        if self.cfg.network.add_norm_after_linear:
            print('Adding norm resnet linears ... ')
            fc_layers = [nn.Linear(self.out_size, self.cfg.network.resnet_hdim), nn.LayerNorm(self.cfg.network.resnet_hdim), nn.ELU(inplace=True)]
            for _ in range(self.cfg.network.resnet_num_fc_layers - 1):
                fc_layers.append(nn.Linear(self.cfg.network.resnet_hdim, self.cfg.network.resnet_hdim))
                fc_layers.append(nn.LayerNorm(self.cfg.network.resnet_hdim))
                fc_layers.append(nn.ELU(inplace=True))
            self.fc_head = nn.Sequential(*fc_layers)
        else:
            fc_layers = [nn.Linear(self.out_size, self.cfg.network.resnet_hdim), nn.ELU(inplace=True)]
            for _ in range(self.cfg.network.resnet_num_fc_layers - 1):
                fc_layers.append(nn.Linear(self.cfg.network.resnet_hdim, self.cfg.network.resnet_hdim))
                fc_layers.append(nn.ELU(inplace=True))
            self.fc_head = nn.Sequential(*fc_layers)

        self.char_embeddings = nn.Embedding(256, self.cfg.network.char_edim)
        self.color_embeddings = nn.Embedding(128, self.cfg.network.color_edim)

        if self.cfg.network.fix_initialization:
            self.apply(self._init_weights)

    def _init_weights(self, module, scale: float = 1.0, bias: bool = True):
        if isinstance(module, nn.Linear):
            print('fixing resnet linear initialization ...')
            module.weight.data *= scale / module.weight.norm(dim=1, p=2, keepdim=True)

            if bias:
                module.bias.data *= 0

    def forward(self, chars, colors):
        chars, colors = self._embed(chars, colors) # 21 x 80
        if self.cfg.network.add_char_color:
            x = chars + colors
            x = x.permute(0, 1, 4, 2, 3).flatten(1, 2).contiguous()
        else:
            x = self._stack(chars, colors)
        x = self.conv_net(x)
        x = x.view(-1, self.out_size)
        x = self.fc_head(x)
        return x

    def _embed(self, chars, colors):
        chars = selectt(self.char_embeddings, chars.long(), True)
        colors = selectt(self.color_embeddings, colors.long(), True)
        return chars, colors

    def _stack(self, chars, colors):
        obs = torch.cat([chars, colors], dim=-1)
        return obs.permute(0, 1, 4, 2, 3).flatten(1, 2).contiguous()

class ResBlock(nn.Module):
    def __init__(self, cfg: DictConfig, in_channels: int, out_channels: int, filter_size: int, num_layers: int):
        super(ResBlock, self).__init__()

        self.cfg = cfg

        layers = []
        for _ in range(num_layers):
            layers.append(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    filter_size,
                    stride=1,
                    padding=(filter_size // 2)
                )
            )
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ELU(inplace=True))

        self.net = nn.Sequential(*layers)

        if self.cfg.network.fix_initialization:
            self.apply(self._init_weights)

    def _init_weights(self, module, scale: float = 1.0, bias: bool = True):
        if isinstance(module, nn.Conv2d):
            print('fixing resblock initialization ...')
            # NOTE: from https://github.com/openai/Video-Pre-Training/blob/077ba2b9885ff696051df8348dc760d9699139ca/lib/util.py#L68-L73
            # and https://github.com/openai/Video-Pre-Training/blob/077ba2b9885ff696051df8348dc760d9699139ca/lib/impala_cnn.py#L164-L168
            # and https://github.com/openai/Video-Pre-Training/blob/077ba2b9885ff696051df8348dc760d9699139ca/lib/impala_cnn.py#L105
            scale = math.sqrt(self.cfg.network.obs_conv_blocks) / math.sqrt(self.cfg.network.resnet_num_blocks)

            module.weight.data *= scale / module.weight.norm(
                dim=tuple(range(1, module.weight.data.ndim)), p=2, keepdim=True
            )

            # Init Bias
            if bias:
                module.bias.data *= 0

    def forward(self, x):
        return self.net(x) + x