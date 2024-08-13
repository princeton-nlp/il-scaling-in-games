
from nle import nethack
from torch import nn
import torch.nn.functional as F
from omegaconf import DictConfig

class TopLineEncoder(nn.Module):
    """
    This class uses a one-hot encoding of the ASCII characters
    as features that get fed into an MLP. 
    Adapted from https://github.com/dungeonsdatasubmission/dungeonsdata-neurips2022/blob/67139262966aa11555cf7aca15723375b36fbe42/experiment_code/hackrl/models/offline_chaotic_dwarf.py
    """
    def __init__(self, cfg: DictConfig, msg_hdim: int):
        super(TopLineEncoder, self).__init__()

        self.cfg = cfg
        
        self.msg_hdim = msg_hdim
        self.i_dim = nethack.NLE_TERM_CO * 256

        print('msg hdim', msg_hdim)
        if self.cfg.network.add_norm_after_linear:
            print('Adding norm in topline ... ')
            self.msg_fwd = nn.Sequential(
                nn.Linear(self.i_dim, self.msg_hdim), # NOTE: Jens: this first layer is pretty much an embedding layer
                nn.LayerNorm(self.msg_hdim),
                nn.ELU(inplace=True),
                nn.Linear(self.msg_hdim, self.msg_hdim),
                nn.LayerNorm(self.msg_hdim),
                nn.ELU(inplace=True),
            )

        else:
            self.msg_fwd = nn.Sequential(
                nn.Linear(self.i_dim, self.msg_hdim), # NOTE: Jens: this first layer is pretty much an embedding layer
                nn.ELU(inplace=True),
                nn.Linear(self.msg_hdim, self.msg_hdim),
                nn.ELU(inplace=True),
            )

        # TODO: consider changing this to character embeddings with CNN

        if self.cfg.network.fix_initialization:
            self.apply(self._init_weights)

    def _init_weights(self, module, scale: float = 1.0, bias: bool = True):
        if isinstance(module, nn.Linear):
            print('fixing topline initialization ...')
            module.weight.data *= scale / module.weight.norm(dim=1, p=2, keepdim=True)

            if bias:
                module.bias.data *= 0

    def forward(self, message):
        # Characters start at 33 in ASCII and go to 128. 96 = 128 - 32
        message_normed = (
            F.one_hot((message).long(), 256).reshape(-1, self.i_dim).float()
        )
        return self.msg_fwd(message_normed)