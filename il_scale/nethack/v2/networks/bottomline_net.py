
from nle import nethack
from torch import nn
import torch
from omegaconf import DictConfig

from il_scale.nethack.v2.utils.model import conv_outdim

class BottomLineEncoder(nn.Module):
    """
    Adapted from https://github.com/dungeonsdatasubmission/dungeonsdata-neurips2022/blob/67139262966aa11555cf7aca15723375b36fbe42/experiment_code/hackrl/models/offline_chaotic_dwarf.py
    """
    def __init__(self, cfg: DictConfig, hdim: int = 128):
        super(BottomLineEncoder, self).__init__()

        self.cfg = cfg

        print('bottomline hidden dim', hdim)
        self.conv_layers = []
        w = nethack.NLE_TERM_CO * 2
        for in_ch, out_ch, filter, stride in [[2, 32, 8, 4], [32, 64, 4, 1]]:
            self.conv_layers.append(nn.Conv1d(in_ch, out_ch, filter, stride=stride))
            self.conv_layers.append(nn.ELU(inplace=True))
            w = conv_outdim(w, filter, padding=0, stride=stride)

        self.hdim = hdim

        self.out_dim = w * out_ch
        self.conv_net = nn.Sequential(*self.conv_layers)

        print('bottom line out dim', self.out_dim)
        if self.cfg.network.add_norm_after_linear:
            print('Adding norm in bottomline ... ')
            self.fwd_net = nn.Sequential(
                nn.Linear(self.out_dim, self.hdim),
                nn.LayerNorm(self.hdim),
                nn.ELU(),
                nn.Linear(self.hdim, self.hdim),
                nn.LayerNorm(self.hdim),
                nn.ELU(),
            )
        else:
            self.fwd_net = nn.Sequential(
                nn.Linear(self.out_dim, self.hdim),
                nn.ELU(),
                nn.Linear(self.hdim, self.hdim),
                nn.ELU(),
            )

        if self.cfg.network.fix_initialization:
            self.apply(self._init_weights)

    def _init_weights(self, module, scale: float = 1.0, bias: bool = True):
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv1d):
            print('fixing bottom line initialization ...')
            module.weight.data *= scale / module.weight.norm(dim=1, p=2, keepdim=True)

            if bias:
                module.bias.data *= 0


    def forward(self, bottom_lines):
        B, D = bottom_lines.shape
        # ASCII 32: ' ', ASCII [33-128]: visible characters
        chars_normalised = (bottom_lines - 32) / 96

        # ASCII [45-57]: -./01234556789
        numbers_mask = (bottom_lines > 44) * (bottom_lines < 58)
        digits_normalised = numbers_mask * (bottom_lines - 47) / 10 # why subtract 47 here and not 48?

        # Put in different channels & conv (B, 2, D)
        x = torch.stack([chars_normalised, digits_normalised], dim=1)
        return self.fwd_net(self.conv_net(x).view(B, -1))