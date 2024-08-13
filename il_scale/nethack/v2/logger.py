import os
import time
import logging
from collections import Counter, defaultdict
import random

import wandb
import torch
import torch.distributed as dist
from nle import nethack
import numpy as np
import scipy.stats as stats
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf

from il_scale.nethack.v2.utils.logging import update_mean

# A logger for this file
logging.basicConfig(
    format=(
        "[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] " "%(message)s"
    ),
    level=logging.INFO,
)

class Logger():
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

        # Wandb settings
        self.group_name = ""

        # Global metrics
        self.grad_steps = 0
        self.tot_samples = 0

        # Shared train/dev metrics
        self.acc = 0
        self.loss = 0
        self.top_1 = 0
        self.top_3 = 0
        self.top_5 = 0
        self.top_10 = 0
        self.log_samples = 0
        self.lvl_accs = dict()
        self.act_freqs = dict()
        self.inv_preds = defaultdict(list)
        self.err_examples = []
        self.first_batch_acc = 0
        self.per_act_recall = dict()
        self.per_act_recall_top_3 = dict()
        self.per_act_recall_top_5 = dict()
        self.per_act_recall_top_10 = dict()
        self.per_act_errs = defaultdict(int)
        self.per_act_log_probs_T = defaultdict(int)
        self.per_act_log_probs_t = defaultdict(list)
        self.just_reset = True

        # update config if necessary, needs to happen for all processes
        # if self.cfg.setup.wandb_id:
        #     run_file = os.path.join('princeton-nlp', 'nethack', self.cfg.setup.wandb_id)
        #     run = wandb.Api().run(run_file)
        #     self.cfg.update(run.config)

    ###### INTERFACE ######

    def shutdown(self):
        wandb.finish()

    def setup(self):
        self._init_wandb()

        if self.cfg.setup.wandb_mode == "disabled":
            self.rundir = os.path.join(
                self.cfg.trainer.save_dir, "torchbeast-%s" % time.strftime("%Y%m%d-%H%M%S")
            )
            if not os.path.exists(self.rundir):
                os.makedirs(self.rundir)
                logging.info("Saving stuff to %s", self.rundir)
            self.log_dir = self.rundir
        else:
            self.rundir = wandb.run.dir
            self.log_dir = os.path.join('logs', wandb.run.id)
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)

    def gradient_step(self):
        self.grad_steps += 1

    def sample_step(self, samples: int):
        self.log_samples += samples

    def log_train(self, rank: int, lr: int = None):
        metrics = dict()

        # Log non-synchronized metrics
        metrics['train_loss'] = self.loss.item() 
        metrics['train_acc'] = self.acc.item()
        metrics['train_top_1'] = self.top_1.item()
        metrics['train_top_3'] = self.top_3.item()
        metrics['train_top_5'] = self.top_5.item()
        metrics['train_top_10'] = self.top_10.item()
        metrics['grad_step'] = self.grad_steps
        metrics['lr'] = lr

        # Synchronize some of the metrics
        dist.barrier()
        train_sps = self.log_samples / (time.time() - self.log_time)
        train_log_samples_tensor = torch.tensor([self.log_samples], device=rank)
        sps_tensor = torch.tensor([train_sps], device=rank)
        dist.all_reduce(sps_tensor, dist.ReduceOp.SUM)
        dist.all_reduce(train_log_samples_tensor, dist.ReduceOp.SUM)

        # Update total number of samples in training
        self.tot_samples += train_log_samples_tensor.item()

        # Add synchronized metrics
        metrics['train_sps'] = sps_tensor.item()
        metrics['num_samples'] = self.tot_samples

        logging.info(f"Train loss: {metrics['train_loss']}")
        logging.info(f"Train acc: {metrics['train_acc']}")
        logging.info(f"Train SPS: {train_sps:.2f}")
        logging.info(f"Total Train SPS: {metrics['train_sps']}")

        if rank == 0:
            metrics = self._add_group_name(metrics)
            wandb.log(metrics)

    def _compute_recall(self, raw_recall):
        for act in raw_recall:
            tp = raw_recall[act]['tp']
            fn = raw_recall[act]['fn']
            raw_recall[act] = tp/(tp + fn)

    def reset(self):
        self.acc = 0
        self.loss = 0
        self.top_1 = 0
        self.top_3 = 0
        self.top_5 = 0
        self.top_10 = 0
        self.log_samples = 0
        self.log_time = time.time()
        self.lvl_accs = dict()
        self.act_freqs = Counter()
        self.inv_preds = defaultdict(list)
        self.err_examples = []
        self.first_batch_acc = 0
        self.per_act_recall = dict()
        self.per_act_recall_top_3 = dict()
        self.per_act_recall_top_5 = dict()
        self.per_act_recall_top_10 = dict()
        self.per_act_errs = defaultdict(int)
        self.per_act_log_probs_T = defaultdict(int)
        self.per_act_log_probs_t = defaultdict(list)

        self.just_reset = True

    def update_metrics(self, B: int, T: int, loss: torch.tensor, logits: torch.tensor, labels: torch.tensor, 
                        batch_samples: int, batch: torch.tensor, i: int, policy_dists = None, compute_lvl_accs: bool = False,
                        compute_act_freq: bool = False, compute_inv_acc: bool = False,
                        dump_err_examples: bool = False, compute_first_batch_acc: bool = False,
                        compute_per_act_recall: bool = False, compute_per_act_errs: bool = False,
                        compute_per_act_log_probs: bool = False):
        """
        blstats: T x B
        """
        self.just_reset = False
        
        blstats = batch['blstats']
        preds = torch.argmax(logits, dim=1)
        # Loss
        self.loss = update_mean(self.loss, self.log_samples, loss, batch_samples)
        
        # Accuracy
        acc = torch.sum(preds == labels)/batch_samples
        self.acc = update_mean(self.acc, self.log_samples, acc, batch_samples)

        if compute_first_batch_acc and i == 0:
            first_batch_acc = torch.sum(preds == labels)/batch_samples
            self.first_batch_acc = update_mean(self.first_batch_acc, self.log_samples, first_batch_acc, batch_samples)

        _, top_k = torch.topk(logits, k=10, dim=1)
        labels = labels.view(-1, 1)  # make labels broadcastable

        # Top-1
        top_1 = torch.sum(top_k[:, :1] == labels)/batch_samples
        self.top_1 = update_mean(self.top_1, self.log_samples, top_1, batch_samples)

        # Top-3
        top_3 = torch.sum(top_k[:, :3] == labels)/batch_samples
        self.top_3 = update_mean(self.top_3, self.log_samples, top_3, batch_samples)

        # Top - 5
        top_5 = torch.sum(top_k[:, :5] == labels)/batch_samples
        self.top_5 = update_mean(self.top_5, self.log_samples, top_5, batch_samples)

        # Top - 10
        top_10 = torch.sum(top_k[:, :10] == labels)/batch_samples
        self.top_10 = update_mean(self.top_10, self.log_samples, top_10, batch_samples)

        # Lvl accs, switch from T x B to B x T
        if compute_lvl_accs:
            dlvls = blstats[..., nethack.NLE_BL_DLEVEL].transpose(0, 1).contiguous()
            unique_lvls = set(dlvls.flatten().tolist())
            for lvl in unique_lvls:
                dlvl_mask = (dlvls == lvl).flatten().tolist()
                for valid, label, pred in zip(dlvl_mask, labels, preds):
                    if valid and label != -100:
                        if lvl in self.lvl_accs:
                            self.lvl_accs[lvl]["total"] += 1
                            self.lvl_accs[lvl]["correct"] += int(label == pred)
                        else:
                            self.lvl_accs[lvl] = dict()
                            self.lvl_accs[lvl]["total"] = 1
                            self.lvl_accs[lvl]["correct"] = int(label == pred)

        if compute_act_freq:
            self.act_freqs += Counter(labels.flatten().tolist())

        if compute_inv_acc:
            re_preds = preds.view(T, B)
            re_labels = labels.view(T, B)
            for b in range(B):
                for t in range(T):
                    m_ascii = batch['tty_chars'][t][b][0, :].tolist()
                    m = "".join([chr(char) for char in m_ascii])
                    next_true_act = nethack.ACTIONS[re_labels[t][b].item()]
                    next_pred_act = nethack.ACTIONS[re_preds[t][b].item()]
                    result = int(next_true_act == next_pred_act)
                    if m.startswith('What do you want to eat'):
                        self.inv_preds['eat'].append(result)
                    elif m.startswith('What do you want to throw'):
                        self.inv_preds['throw'].append(result)
                    elif m.startswith('What do you want to wear'):
                        self.inv_preds['wear'].append(result)
                    elif m.startswith('What do you want to drink'):
                        self.inv_preds['drink'].append(result)
                    elif m.startswith('What do you want to drop'):
                        self.inv_preds['drop'].append(result)
                    elif m.startswith('What do you want to wield'):
                        self.inv_preds['wield'].append(result)
                    elif m.startswith('What do you want to use or apply'):
                        self.inv_preds['apply'].append(result)
                    elif m.startswith('What do you want to dip'):
                        self.inv_preds['dip'].append(result)
                    elif m.startswith('What do you want to write with'):
                        self.inv_preds['engrave'].append(result)
                    elif m.startswith('What do you want to ready'):
                        self.inv_preds['ready'].append(result)
                    elif m.startswith('What do you want to read'):
                        self.inv_preds['read'].append(result)

        if dump_err_examples:
            re_preds = preds.view(T, B)
            re_labels = labels.view(T, B)
            for b in range(B):
                for t in range(T):
                    true_act = nethack.ACTIONS[re_labels[t][b].item()]
                    pred_act = nethack.ACTIONS[re_preds[t][b].item()]
                    if true_act != pred_act:
                        self.err_examples.append({
                            'state': batch['tty_chars'][t][b],
                            'label': true_act,
                            'pred': pred_act
                        })

        if compute_per_act_errs:
            re_preds = preds.view(T, B)
            re_labels = labels.view(T, B)
            for b in range(B):
                for t in range(T):
                    true_act = nethack.ACTIONS[re_labels[t][b].item()]
                    pred_act = nethack.ACTIONS[re_preds[t][b].item()]
                    if true_act != pred_act:
                        self.per_act_errs[true_act] += 1

        if compute_per_act_log_probs:
            re_labels = labels.view(T, B)
            probs = F.log_softmax(logits.view(T, B, -1), dim=-1)
            for b in range(B):
                for t in range(T):
                    act = nethack.ACTIONS[re_labels[t][b].item()]
                    a_prob = probs[t][b][re_labels[t][b].item()].item()
                    self.per_act_log_probs_T[act] += a_prob
                    self.per_act_log_probs_t[act].append(a_prob)

        if compute_per_act_recall:
            re_preds = preds.view(T, B)
            re_labels = labels.view(T, B)
            re_topk = top_k.view(T, B, 10)
            for b in range(B):
                for t in range(T):
                    true_act_idx = re_labels[t][b].item()
                    true_act = nethack.ACTIONS[true_act_idx]
                    pred_act = nethack.ACTIONS[re_preds[t][b].item()]

                    if true_act not in self.per_act_recall:
                        self.per_act_recall[true_act] = { 'tp': 0, 'fn': 0 }
                    if true_act not in self.per_act_recall_top_3:
                        self.per_act_recall_top_3[true_act] = { 'tp': 0, 'fn': 0 }
                    if true_act not in self.per_act_recall_top_5:
                        self.per_act_recall_top_5[true_act] = { 'tp': 0, 'fn': 0 }
                    if true_act not in self.per_act_recall_top_10:
                        self.per_act_recall_top_10[true_act] = { 'tp': 0, 'fn': 0 }

                    # Top-1
                    if true_act != pred_act:
                        self.per_act_recall[true_act]['fn'] += 1
                    else:
                        self.per_act_recall[true_act]['tp'] += 1

                    # Top-3
                    if true_act_idx not in re_topk[t][b][:3]:
                        self.per_act_recall_top_3[true_act]['fn'] += 1
                    else:
                        self.per_act_recall_top_3[true_act]['tp'] += 1

                    # Top-5
                    if true_act_idx not in re_topk[t][b][:5]:
                        self.per_act_recall_top_5[true_act]['fn'] += 1
                    else:
                        self.per_act_recall_top_5[true_act]['tp'] += 1

                    # Top-10
                    if true_act_idx not in re_topk[t][b][:10]:
                        self.per_act_recall_top_10[true_act]['fn'] += 1
                    else:
                        self.per_act_recall_top_10[true_act]['tp'] += 1


    def start(self):
        self.start_time = time.time()
        self.log_time = time.time()
        # Reset everything
        if not self.cfg.setup.wandb_id: # don't overwrite continuation values
            self.grad_steps = 0
            self.tot_samples = 0
        self.reset()

    def set_group_name(self, name: str):
        self.group_name = name

    ###### PRIVATE ######

    def _add_group_name(self, metrics: dict):
        new_metrics = dict()
        if self.group_name != "":
            keys = list(metrics.keys())
            for key in keys:
                new_key = f'{self.group_name}{key}'
                new_metrics[new_key] = metrics[key]
        else:
            new_metrics = metrics

        return new_metrics

    def _init_wandb(self):
        wandb_conf = OmegaConf.to_container(self.cfg, resolve=True, throw_on_missing=True)
        if self.cfg.setup.wandb_id:
            wandb.init(
                project='nethack', 
                resume='must', 
                id=self.cfg.setup.wandb_id, 
                mode=self.cfg.setup.wandb_mode
            )
        else:
            wandb.init(
                project='nethack', 
                config=wandb_conf,
                mode=self.cfg.setup.wandb_mode, 
                name=self.cfg.setup.wandb_name, 
                tags=([] if not self.cfg.setup.wandb_tag else [self.cfg.setup.wandb_tag])
            )