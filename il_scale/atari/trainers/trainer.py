import time
import logging
from typing import Union
import os

import torch
import torch.nn as nn
import wandb

from il_scale.data.parquet_dataset import ParquetDataset
from il_scale.utils.flop import FLOPCounter, FLOP_TO_STR
from il_scale.utils.model_utils import count_params

log = logging.getLogger(__name__)


def mask_labels_from_gameids(gameids: torch.tensor, labels: torch.tensor):
    mask = (gameids == 0).to(labels.device)
    new_labels = labels.clone()
    new_labels.masked_fill_(mask, -100)
    return new_labels


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        dataset: ParquetDataset,
        cfg,
    ):
        self.model = model
        self.dataset = dataset
        self.cfg = cfg

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.lr)

        self.is_flop_saved = {int(k): False for k in cfg.flops_to_save}
        self.flop_counter = FLOPCounter()
        self.flops_per_sample = self.flop_counter.count_flops(self.model)["total_flops"]

    def train(self):
        start_time = time.time()
        grad_steps = 0
        total_flops = 0
        total_samples = 0
        stats = {"loss": [], "top_1": [], "top_2": [], "top_3": []}
        self.optimizer.zero_grad()

        for batch in self.dataset:
            # NOTE: we compute this here before it's moved to the GPU
            batch_samples = sum([1 for gid in batch["gameids"].flatten() if gid != 0])

            for key in batch:
                batch[key] = torch.from_numpy(batch[key]).to(self.cfg.device)

            labels = mask_labels_from_gameids(batch["gameids"], batch["actions"].long())

            logits = self.model(batch["states"].float())

            B, T = logits.shape[:2]
            logits = logits.view(B * T, -1)

            labels = labels.view(B * T)
            loss = self.criterion(logits, labels)

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            # top-k
            _, top_k = torch.topk(logits, k=3, dim=1)
            labels = labels.view(-1, 1)  # make labels broadcastable
            top_1 = torch.sum(top_k[:, :1] == labels) / batch_samples
            top_2 = torch.sum(top_k[:, :2] == labels) / batch_samples
            top_3 = torch.sum(top_k[:, :3] == labels) / batch_samples

            stats["loss"].append(loss)
            stats["top_1"].append(top_1)
            stats["top_2"].append(top_2)
            stats["top_3"].append(top_3)

            grad_steps += 1
            total_flops += self.flops_per_sample * batch_samples
            total_samples += batch_samples

            self.maybe_save_flops(total_flops, grad_steps * batch_samples, loss)

            if grad_steps % self.cfg.log_freq == 0:
                wandb_stats = {
                    "loss": torch.mean(torch.stack(stats["loss"])).item(),
                    "top_1": torch.mean(torch.stack(stats["top_1"])).item(),
                    "top_2": torch.mean(torch.stack(stats["top_2"])).item(),
                    "top_3": torch.mean(torch.stack(stats["top_3"])).item(),
                    "fps": self.cfg.log_freq
                    * batch_samples
                    / (time.time() - start_time),
                    "samples": grad_steps * batch_samples,
                }
                stats = {"loss": [], "top_1": [], "top_2": [], "top_3": []}
                log.info("Logging to wandb ...")
                wandb.log(wandb_stats)
                start_time = time.time()

            if self.stop_training(total_samples, total_flops):
                log.info("Done training.")
                return

        log.info("Ran out of samples. Stopping ...")

    def maybe_save_flops(self, flops: int, samples: int, loss):
        model_folder = os.path.join("models", wandb.run.id)
        if not os.path.exists(model_folder):
            os.makedirs(model_folder, exist_ok=False)

        for flop in self.is_flop_saved.keys():
            if flops >= flop and not self.is_flop_saved[flop]:
                path = os.path.join(
                    "models", wandb.run.id, f"flops_{FLOP_TO_STR[flop]}.tar"
                )
                torch.save(
                    {
                        "model_state_dict": self.model.state_dict(),
                        "flops": flops,
                        "num_params": count_params(self.model)[0],
                        "num_samples": samples,
                        "last_loss": loss.item(),
                    },
                    path,
                )
                wandb.save(path)
                self.is_flop_saved[flop] = True
                log.info(f"Saved model with {flop:e} flops.")

    def stop_training(self, total_samples: int, total_flops: float):
        if all([v for v in self.is_flop_saved.values()]):
            log.info("All flops saved.")
            return True
        elif total_samples >= self.cfg.max_samples:
            log.info("Max samples reached.")
            return True
        elif total_flops >= self.cfg.max_flops:
            log.info("Max flops reached.")
            return True

        return False
