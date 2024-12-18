import random
import os
import logging

import numpy as np
import torch
import wandb


def setup_wandb(name: str, cfg) -> None:
    wandb.init(
        entity="princeton-nlp",
        project="il-scale",
        sync_tensorboard=True,
        name=name,
        config=cfg,
        mode=cfg["wandb_mode"],
    )


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
