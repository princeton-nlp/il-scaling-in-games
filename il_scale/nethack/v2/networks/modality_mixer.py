
from nle import nethack
from torch import nn
import torch.nn.functional as F
from omegaconf import DictConfig

class ModalityMixer(nn.Module):
    """
    Policy head.
    """
    def __init__(self, cfg: DictConfig, out_dim: int):
        super(ModalityMixer, self).__init__()
        
        self.cfg = cfg

        if self.cfg.network.add_norm_after_linear:
            print('Adding norm in modality mixer ... ')
            fc_layers = [nn.Linear(out_dim, self.cfg.network.hdim), nn.LayerNorm(self.cfg.network.hdim), nn.ReLU()]
            for _ in range(self.cfg.network.policy_num_fc_layers - 1):
                fc_layers.append(nn.Linear(self.cfg.network.hdim, self.cfg.network.hdim))
                fc_layers.append(nn.LayerNorm(self.cfg.network.hdim))
                fc_layers.append(nn.ReLU())
            self.fc = nn.Sequential(*fc_layers)
        else:
            fc_layers = [nn.Linear(out_dim, self.cfg.network.hdim), nn.ReLU()]
            for _ in range(self.cfg.network.policy_num_fc_layers - 1):
                fc_layers.append(nn.Linear(self.cfg.network.hdim, self.cfg.network.hdim))
                fc_layers.append(nn.ReLU())
            self.fc = nn.Sequential(*fc_layers)

        if self.cfg.network.fix_initialization:
            self.apply(self._init_weights)

    def _init_weights(self, module, scale: float = 1.0, bias: bool = True):
        if isinstance(module, nn.Linear):
            print('fixing modality mixer initialization ...')
            module.weight.data *= scale / module.weight.norm(dim=1, p=2, keepdim=True)

            if bias:
                module.bias.data *= 0

    def forward(self, x):
        return self.fc(x)
