import logging
from typing import Union
import os

from nle import nethack
import wandb
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from omegaconf import DictConfig

from il_scale.nethack.v2.logger import Logger
from il_scale.nethack.v2.networks.policy_net import PolicyNet


class Agent():
    def __init__(self, cfg: DictConfig, logger: Logger):
        self.cfg = cfg
        self.logger = logger

        # Assign initial dummy rank and world_size
        self.rank = 0
        self.world_size = 1
        self.ddp = False

    def predict(self, batch, inference_params=None):
        return self.model(batch, inference_params=inference_params)

    def move_to_ddp(self, rank: int, world_size: int, find_unused_parameters: bool = False):
        self.rank = rank
        self.world_size = world_size
        self.ddp = True
        self.model = DDP(self.model, device_ids=[rank], find_unused_parameters=find_unused_parameters)
    
    def load(self, state_dict):
        self.model.load_state_dict(state_dict)

    def to(self, device):
        self.model.to(device)

    def parameters(self):
        return self.model.parameters()

    def eval(self):
        self.model.eval()

    def train(self):
        self.model.train()

    def state_dict(self):
        return self.model.module.state_dict() if self.ddp else self.model.state_dict()

    def save(self, chkpt_name: str):
        checkpointpath = os.path.join(self.logger.rundir, f'{chkpt_name}.tar')
        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "flags": vars(self.cfg)
            },
            checkpointpath,
        )
        wandb.save(checkpointpath)

    def get_running_std(self):
        if self.ddp:
            return self.model.module.get_running_std()
        else:
            return self.model.get_running_std()

    def construct_model(self, load_config = None):
        cfg = self.cfg if not load_config else load_config
        self.model = PolicyNet(cfg)