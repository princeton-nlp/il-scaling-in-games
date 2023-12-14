import configargparse
import logging

import torch

from il_scale.nethack.utils import assign_free_gpus


class Config:
    def __init__(self, flags: configargparse.Namespace):
        # Transfer params from flags
        for key, value in flags.__dict__.items():
            self.__dict__[key] = value

    def set_device(self):
        if not self.disable_cuda and torch.cuda.is_available():
            assign_free_gpus(max_gpus=self.num_gpus, logger=logging)
            self.device = torch.device("cuda")
            logging.info("Using CUDA.")
        else:
            self.device = torch.device("cpu")
            logging.info("Not using CUDA.")
