import logging
import os
import time

import wandb
import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR
from transformers import get_linear_schedule_with_warmup, get_constant_schedule_with_warmup
from omegaconf import DictConfig, OmegaConf

from il_scale.nethack.v2.logger import Logger
from il_scale.nethack.v2.data.tty_data import TTYData
from il_scale.nethack.v2.utils.setup import DDPUtil
from il_scale.nethack.v2.utils.model import load_checkpoint

# A logger for this file
logging.basicConfig(
    format=(
        "[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] " "%(message)s"
    ),
    level=logging.INFO,
)

class FakeScheduler():
    def __init__(self, lr:float):
        self.lr = lr

    def step(self):
        pass

    def get_last_lr(self):
        return [self.lr]

class Trainer():
    def __init__(
        self, 
        cfg: DictConfig, 
        logger: Logger, 
        data: TTYData, 
        ddp_util: DDPUtil
    ):
        self.cfg = cfg
        self.logger = logger
        self.data = data
        self.ddp_util = ddp_util

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self._get_model().parameters(), 
            lr=self.cfg.optimizer.lr
        )
        logging.info(f'LR: {self.cfg.optimizer.lr}')

        self.total_steps = self.cfg.trainer.total_samples / (self.data.train_batch_size * self.data.train_seq_len * self.cfg.setup.num_gpus * self.cfg.trainer.gradient_acc)
        post_warmup_steps = self.total_steps - self.cfg.optimizer.optim_warmup_steps
        schedule_total_steps = self.cfg.optimizer.optim_warmup_steps + post_warmup_steps * 1/(1 - self.cfg.optimizer.lr_end_fraction)
        
        if self.cfg.optimizer.scheduler_type == 'linear':
            print('Using linear lr schedule ...')
            self.scheduler = get_linear_schedule_with_warmup(self.optimizer, self.cfg.optimizer.optim_warmup_steps, schedule_total_steps)
        
        elif self.cfg.optimizer.scheduler_type == 'constant':
            print('Using constant lr schedule ...')
            self.scheduler = get_constant_schedule_with_warmup(
                self.optimizer, 
                0.0
            )

        elif self.cfg.optimizer.scheduler_type == "cosine":
            print('Using cosine lr schedule ...')
            scheduler1 = get_constant_schedule_with_warmup(self.optimizer, self.cfg.optimizer.optim_warmup_steps)
            scheduler2 = CosineAnnealingLR(self.optimizer, T_max=post_warmup_steps, eta_min=self.cfg.optimizer.lr_end_fraction * self.cfg.optimizer.lr)
            self.scheduler = SequentialLR(self.optimizer, schedulers=[scheduler1, scheduler2], milestones=[self.cfg.optimizer.optim_warmup_steps])

        self.use_amp = self.cfg.setup.use_amp
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        self.train_min_loss = 1e10
        self.time_budget = self.cfg.trainer.train_time_budget * 60 * 60 # convert to seconds
        self.saved_chkpts = {0}

        self._maybe_resume()

    ###### INTERFACE ######

    def train(self):
        raise NotImplementedError()

    ###### PRIVATE ######

    def _reset(self):
        self.num_samples = 0
        self.train_min_loss = 1e10

    def _stop_condition(self):
        if time.time() - self.logger.start_time > self.time_budget:
            logging.info(f'Running out of time ...')
            return True
        elif self.logger.grad_steps > self.total_steps:
            logging.info(f'Running out of grad steps ...')
            return True
        else:
            return False
        
    def _get_total_samples(self):
        return self.logger.tot_samples + self.logger.log_samples * self.ddp_util.world_size

    def _save_chkpts(self, dev_metrics: dict, model_name: str = "model"):
        # Save if dev loss improves
        if dev_metrics['dev_loss'] < self.dev_min_loss:
            self.dev_min_loss = dev_metrics['dev_loss']
            # dev loss improved so reset patience
            self.patience = 0
            self._save(f"{model_name}_loss.tar")
        else:
            # dev loss didn't improve so increase patience
            self.patience += 1

        if dev_metrics['num_samples'] // self.cfg.trainer.chkpt_freq not in self.saved_chkpts:
            chkpt_num = dev_metrics['num_samples'] // self.cfg.trainer.chkpt_freq
            self._save(f"{model_name}_{chkpt_num}.tar")
            self.saved_chkpts.add(chkpt_num)

        # Save latest checkpoint always
        self._save(f"{model_name}_latest.tar")

    def _save(self, chkpt_name: str):
        checkpointpath = os.path.join(self.logger.rundir, chkpt_name)
        model = self._get_model()
        logging.info("Saving checkpoint to %s", checkpointpath)
        wandb_conf = OmegaConf.to_container(self.cfg, resolve=True, throw_on_missing=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "scaler_state_dict": self.scaler.state_dict(),
                "flags": wandb_conf,
                "num_samples": self.logger.tot_samples + self.logger.log_samples * self.ddp_util.world_size,
                "gradient_steps": self.logger.grad_steps,
                "train_min_loss": self.train_min_loss
            },
            checkpointpath,
        )
        wandb.save(checkpointpath)

    def _maybe_resume(self):
        if self.cfg.setup.wandb_id:
            wandb_id = self.cfg.setup.wandb_id
            logging.info(f"Resuming state from wandb_id {wandb_id}")

            # Get checkpoint
            checkpoint = load_checkpoint(self.cfg.setup.model_load_name, wandb_id, overwrite=False)

            # Load weights for agents
            self._load_weights(checkpoint['model_state_dict'])
            logging.info(f"Loaded weights!")

            # Load trainer states
            logging.info(f"Loading optimizer ...")
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            if 'scaler_state_dict' in checkpoint:
                self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            self.logger.grad_steps = checkpoint['gradient_steps']
            self.logger.tot_samples = checkpoint['num_samples']
            self.train_min_loss = checkpoint['train_min_loss']
        else:
            logging.info("No wandb_id specified to resume from.")

    def _get_model(self):
        raise NotImplementedError()