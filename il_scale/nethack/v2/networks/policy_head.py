
from nle import nethack
from torch import nn
import torch.nn.functional as F
from omegaconf import DictConfig

class PolicyHead(nn.Module):
    """
    Policy head.
    """
    def __init__(self, cfg: DictConfig, num_actions: int):
        super(PolicyHead, self).__init__()

        self.cfg = cfg
        self.policy = nn.Linear(self.cfg.network.hdim, num_actions)

        if self.cfg.network.fix_initialization:
            self.apply(self._init_weights)

    def _init_weights(self, module, scale: float = 1.0, bias: bool = True):
        if isinstance(module, nn.Linear):
            print('fixing policy head initialization ...')
            module.weight.data *= scale / module.weight.norm(dim=1, p=2, keepdim=True)

            if bias:
                module.bias.data *= 0

    def forward(self, x):
        return self.policy(x)