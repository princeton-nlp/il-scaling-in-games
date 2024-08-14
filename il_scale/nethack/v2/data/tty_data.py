import os
import math
import glob

import torch
import nle.dataset as nld
import wandb
from omegaconf import DictConfig

import il_scale.nethack.v2.utils.constants as CONSTANTS
from il_scale.nethack.v2.data.ttyrec_data_loader import TtyrecDataLoader

class TTYData:
    def __init__(
        self, 
        cfg: DictConfig
    ):
        self.cfg = cfg
        print(f'Using dataset: {self.cfg.dataset_name}')

        self.dbfilename = CONSTANTS.DBFILENAME

        # NOTE: These games act weird in nld-nao, just 
        # leave them out.
        self.nld_nao_exclude_games = {
            4131490, 4969574, 4974128, 4984208, 
            4993467, 4997147, 4998268, 5001964, 
            5003497, 5010205, 5010250
        }

        # Load train & dev ids
        self.gameids = self._get_gameids()
        self.train_gameids = self.gameids['train_gameids']

        # Train dataloader params
        self.train_batch_size = min(self.cfg.batch_size, len(self.train_gameids))
        self.train_seq_len = self.cfg.unroll_length

    def _get_gameids(self):
        subselect_sql = "SELECT gameid FROM games WHERE maxlvl >=?"
        subselect_sql_args = (self.cfg.min_dlvl_reached,)
        data = TtyrecDataLoader(
            self.cfg.env,
            torch.device('cpu'),
            dataset_name=self.cfg.dataset_name,
            batch_size=1,
            seq_length=1,
            dbfilename=self.dbfilename,
            threadpool=None,
            shuffle=True,
            subselect_sql=subselect_sql,
            subselect_sql_args=subselect_sql_args,
            use_role=self.cfg.use_bl_role,
            obs_frame_stack=self.cfg.obs_frame_stack,
            use_inventory=self.cfg.use_inventory
        )
    
        train_gameids = data.gameids
        dev_gameids = []
        
        return { 
            "train_gameids": train_gameids,
            "dev_gameids": dev_gameids
        }

    def get_train_dataloader(self, tp = None, rank: int = 0, world_size: int = 1, device = None):
        data_chunk = math.ceil(len(self.train_gameids)/world_size) # spreads a bit uneven but probably fine
        return TtyrecDataLoader(
            self.cfg.env,
            rank if not device else device,
            dataset_name=self.cfg.dataset_name,
            batch_size=self.train_batch_size,
            seq_length=self.train_seq_len,
            dbfilename=self.dbfilename,
            threadpool=tp,
            shuffle=True,
            gameids=self.train_gameids[rank * data_chunk: (rank + 1) * data_chunk],
            loop_forever=True,
            use_role=self.cfg.use_bl_role,
            obs_frame_stack=self.cfg.obs_frame_stack,
            use_inventory=self.cfg.use_inventory
        )

    def mask_labels_from_gameids(self, gameids: torch.tensor, labels: torch.tensor):
        mask = (gameids == 0).to(labels.device)
        new_labels = labels.clone()
        new_labels.masked_fill_(mask, -100)
        return new_labels

    def mask_labels_from_dlvls(self, dlvls: torch.tensor, labels: torch.tensor):
        dlvl_mask = (dlvls > self.max_label_level)
        new_labels = labels.clone()
        new_labels.masked_fill_(dlvl_mask, -100)
        return new_labels