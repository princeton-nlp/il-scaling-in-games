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


def mask_labels_from_gameids(gameids: torch.tensor, labels: torch.tensor):
    mask = (gameids == 0).to(labels.device)
    new_labels = labels.clone()
    new_labels.masked_fill_(mask, -100)
    return new_labels


@torch.no_grad()
def evaluate(model, cfg):
    criterion = nn.CrossEntropyLoss(reduction="sum")

    dev_gids = torch.load(f"data_objects/{cfg.atari.name}_data_split.tar")["dev_gids"]
    tp = ThreadPoolExecutor(max_workers=30)
    dataset = ParquetDataset(
        dataset_root=f"datasets/{cfg.atari.name}",
        threadpool=tp,
        seq_length=cfg.atari.seq_length,
        batch_size=cfg.atari.batch_size,
        gameids=dev_gids,
    )

    losses = []
    total_samples = 0

    for batch in dataset:
        for key in batch:
            batch[key] = torch.from_numpy(batch[key]).to(cfg.atari.device)

        labels = mask_labels_from_gameids(batch["gameids"], batch["actions"].long())

        logits = model(batch["states"].float())
        B, T = logits.shape[:2]
        logits = logits.view(B * T, -1)
        labels = labels.view(B * T)
        loss = criterion(logits, labels)

        losses.append(loss)
        total_samples += (labels != -100).sum().item()

    return torch.sum(torch.stack(losses)) / total_samples


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    log.info(OmegaConf.to_yaml(cfg))
    set_seeds(cfg.atari.seed)

    save_results_path = f"data_objects/eval_loss_{cfg.atari.name}.tar"
    if os.path.exists(save_results_path):
        log.info("Loading eval loss from file ...")
        loss_all = torch.load(save_results_path)
    else:
        loss_all = defaultdict(dict)

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

                    if flop in loss_all and num_params in loss_all[flop]:
                        log.info(
                            f"already evaluated {float(flop):e} with params {num_params:e}, skipping."
                        )
                        continue

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
                    model = SimpleCNN(w_scale=run.config["atari"]["w_scale"], n=n).to(
                        cfg.atari.device
                    )
                    model.load_state_dict(chkpt["model_state_dict"])
                    eval_loss = evaluate(model, cfg)

                    loss_all[flop][num_params] = eval_loss
                    log.info(flop)
                    log.info(f"{count_params(model)[0]:e}")
                else:
                    break

    # save
    if not os.path.exists("data_objects"):
        os.makedirs(f"data_objects", exist_ok=False)

    log.info("Saving eval loss to file ...")
    torch.save(loss_all, save_results_path)


if __name__ == "__main__":
    main()
