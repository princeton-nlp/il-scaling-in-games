import logging
from concurrent.futures import ThreadPoolExecutor
import os

import wandb
import torch
from torch import nn
from omegaconf import DictConfig

from il_scale.nethack.v2.trainers.trainer import Trainer
from il_scale.nethack.v2.agent import Agent
from il_scale.nethack.v2.data.tty_data import TTYData
from il_scale.nethack.v2.utils.setup import DDPUtil
from il_scale.nethack.v2.utils.model import count_params
from il_scale.nethack.v2.logger import Logger

# A logger for this file
logging.basicConfig(
    format=(
        "[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] " "%(message)s"
    ),
    level=logging.INFO,
)

class BCTrainer(Trainer):
    def __init__(
        self, 
        cfg: DictConfig, 
        logger: Logger, 
        agent: Agent, 
        data: TTYData,
        ddp_util: DDPUtil
    ):
        self.agent = agent
        super(BCTrainer, self).__init__(cfg, logger, data, ddp_util)

        self.num_model_params = count_params(agent.model)

    ###### INTERFACE ######

    def train(self):
        self._reset()
        self.logger.start()
        self.agent.train()

        max_workers = self.cfg.data.workers
        with ThreadPoolExecutor(max_workers=max_workers) as tp:
            # Retrieve training data
            train_data = self.data.get_train_dataloader(tp, self.ddp_util.rank, self.ddp_util.world_size)

            # Start training loop
            logging.info(f"Processing {len(train_data.gameids)} gameids.")
            for i, batch in enumerate(train_data, 1):

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.use_amp):
                    agent_outputs = self.agent.predict(batch)
                
                    # Reshape logits
                    T, B = agent_outputs['policy_logits'].shape[:2]
                    logits = agent_outputs['policy_logits'].view(B * T, -1)

                    # Loss and gradients
                    labels = batch['labels'].contiguous().view(B * T)
                    loss = self.criterion(logits, labels) / self.cfg.trainer.gradient_acc

                self.scaler.scale(loss).backward()

                self.logger.update_metrics(B, T, loss * self.cfg.trainer.gradient_acc, logits, labels, labels.shape[0], batch, i)
                self.logger.sample_step(labels.shape[0])

                if i % self.cfg.trainer.gradient_acc != 0:
                    continue

                torch.nn.utils.clip_grad_norm_(self._get_model().parameters(), self.cfg.trainer.clip)
                self.scaler.step(self.optimizer)
                self.scheduler.step()

                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self.logger.gradient_step()

                if self.logger.grad_steps % self.cfg.trainer.log_freq == 0:
                    print('logging!')
                    self.logger.log_train(self.ddp_util.rank, self.scheduler.get_last_lr()[0])
                    self.logger.reset()
                
                # save checkpoint regularly
                if self.ddp_util.rank == 0 and self.logger.grad_steps % (self.cfg.trainer.log_freq * 20) == 0:
                    self._save("model_latest.tar")

                # save checkpoint every 1B
                if self.ddp_util.rank == 0 and self._get_total_samples() // self.cfg.trainer.chkpt_freq not in self.saved_chkpts:
                    chkpt_num = self._get_total_samples() // self.cfg.trainer.chkpt_freq
                    self._save(f"model_{chkpt_num}.tar")
                    self.saved_chkpts.add(chkpt_num)

                # Stop training if we have seen enough samples
                if self._stop_condition():
                    if not self.logger.just_reset:
                        self.logger.log_train(self.ddp_util.rank)
                        self.logger.reset()

                    break

        logging.info("Done training")

    ###### PRIVATE ######

    def _get_model(self):
        return self.agent.model.module if self.agent.ddp else self.agent.model
            
    def _load_weights(self, state_dict):
        self.agent.load(state_dict)