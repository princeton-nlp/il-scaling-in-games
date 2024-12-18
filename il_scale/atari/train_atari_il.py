import time
import logging
from concurrent.futures import ThreadPoolExecutor
import os

import torch
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf

from il_scale.utils.setup_utils import setup_wandb, set_seeds
from il_scale.networks.atari_networks import SimpleCNN
from il_scale.data.parquet_dataset import ParquetDataset
from il_scale.trainers.trainer import Trainer

# A logger for this file
log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    log.info(OmegaConf.to_yaml(cfg))
    set_seeds(cfg.atari.seed)

    wandb_conf = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    setup_wandb(cfg.exp.wandb_name, wandb_conf)

    if cfg.atari.name == "Breakout":
        n = 4
    elif cfg.atari.name == "Boxing":
        n = 18
    elif cfg.atari.name == "Qbert":
        n = 6
    elif cfg.atari.name == "BankHeist":
        n = 18
    elif cfg.atari.name == "BattleZone":
        n = 18
    elif cfg.atari.name == "Phoenix":
        n = 8
    elif cfg.atari.name == "NameThisGame":
        n = 6
    elif cfg.atari.name == "Gopher":
        n = 8
    elif cfg.atari.name == "SpaceInvaders":
        n = 6
    elif cfg.atari.name == "DoubleDunk":
        n = 18
    elif cfg.atari.name == "BeamRider":
        n = 9
    model = SimpleCNN(w_scale=cfg.atari.w_scale, n=n).to(cfg.atari.device)
    tp = ThreadPoolExecutor(max_workers=30)
    train_gids = torch.load(f"data_objects/{cfg.atari.name}_data_split.tar")[
        "train_gids"
    ]
    dataset = ParquetDataset(
        dataset_root=f"datasets/{cfg.atari.name}",
        threadpool=tp,
        seq_length=cfg.atari.seq_length,
        batch_size=cfg.atari.batch_size,
        gameids=train_gids,
    )
    trainer = Trainer(model, dataset, cfg.atari)
    trainer.train()


if __name__ == "__main__":
    main()
