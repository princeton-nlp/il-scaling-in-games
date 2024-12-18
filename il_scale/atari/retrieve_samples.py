import time
import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import os
from collections import defaultdict

import torch.nn as nn
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
from torch import multiprocessing as mp
import torch

from il_scale.utils.setup_utils import setup_wandb, set_seeds
from il_scale.networks.atari_networks import SimpleCNN
from il_scale.data.parquet_dataset import ParquetDataset
from il_scale.trainers.trainer import Trainer
from il_scale.utils.model_utils import count_params

# A logger for this file
log = logging.getLogger(__name__)

FLOPS = [
    "1e13",
    "2e13",
    "5e13",
    "1e14",
    "2e14",
    "5e14",
    "1e15",
    "2e15",
    "5e15",
    "1e16",
    "2e16",
    "5e16",
    "1e17",
    "2e17",
    "5e17",
    "1e18",
]


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    log.info(OmegaConf.to_yaml(cfg))
    set_seeds(cfg.atari.seed)

    # save_results_path = f"data_objects/flop_params_to_samples_{cfg.atari.name}.tar"
    # if os.path.exists(save_results_path):
    #     log.info("File already exists, exiting ...")
    #     exit(0)
    # else:
    #     flop_params_to_samples = defaultdict(dict)

    flop_params_to_samples = defaultdict(dict)

    api = wandb.Api()
    runs = api.runs("princeton-nlp/il-scale")
    for run in runs:
        if cfg.exp.iso_flop_tag in run.tags:
            for flop in FLOPS:
                if os.path.exists(f"models/{run.id}/flops_{flop}.tar"):
                    log.info(f"Loading models/{run.id}/flops_{flop}.tar ...")
                    chkpt = torch.load(
                        f"models/{run.id}/flops_{flop}.tar", map_location="cpu"
                    )
                    num_params = chkpt["num_params"]
                    num_samples = chkpt["num_samples"]
                    breakpoint()

                    flop_params_to_samples[flop][num_params] = num_samples
                    log.info(flop)
                else:
                    break

    breakpoint()

    # # save
    # if not os.path.exists("data_objects"):
    #     os.makedirs(f"data_objects", exist_ok=False)

    # log.info("Saving flop_params_to_samples to file ...")
    # torch.save(flop_params_to_samples, save_results_path)


if __name__ == "__main__":
    main()
