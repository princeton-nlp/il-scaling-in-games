
from nle import nethack
from nle.nethack import INV_SIZE
import torch
from torch import nn
import torch.nn.functional as F
from transformers import BertConfig, BertModel
from omegaconf import DictConfig

from il_scale.nethack.v2.utils.model import selectt

class InventoryNet(nn.Module):
    """
    Encodes the inventory strings.
    """
    def __init__(self, inv_edim: int, inv_hdim: int):
        super(InventoryNet, self).__init__()
        
        self.inv_edim = inv_edim
        self.inv_hdim = inv_hdim

        self.emb = nn.Embedding(nethack.MAX_GLYPH + 1, self.inv_edim)
        self.mlp = nn.Sequential(
            nn.Linear(INV_SIZE[0] * self.inv_edim, self.inv_hdim),
            nn.LayerNorm(self.inv_hdim),
            nn.ELU(),
            nn.Linear(self.inv_hdim, self.inv_hdim)
        )

    def forward(self, inv_glyphs: torch.Tensor):
        B = inv_glyphs.shape[0]
        inv_emb = selectt(self.emb, inv_glyphs, True)
        inv_emb = inv_emb.view(B, INV_SIZE[0] * self.inv_edim)
        inv_rep = self.mlp(inv_emb)

        return inv_rep
        