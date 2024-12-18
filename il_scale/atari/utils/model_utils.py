import os

import torch.nn as nn
import wandb
from stable_baselines3 import A2C, PPO


def create_model(cfg, env):
    if cfg.algo == "ppo":
        model = PPO(
            "CnnPolicy",
            env,
            verbose=1,
            tensorboard_log=f"runs/{wandb.run.id}",
            batch_size=cfg.batch_size,
            ent_coef=cfg.ent_coef,
            n_epochs=cfg.n_epochs,
            n_steps=cfg.n_steps,
            learning_rate=cfg.lr,
        )
    elif cfg.algo == "a2c":
        model = A2C(
            "CnnPolicy",
            env,
            verbose=1,
            tensorboard_log=f"runs/{wandb.run.id}",
            learning_rate=cfg.lr,
            vf_coef=cfg.vf_coef,
            ent_coef=cfg.ent_coef,
        )
    else:
        raise ValueError(f"Unknown algorithm: {cfg.algo}")

    return model


def load_model(cfg):
    if not os.path.exists(cfg.model_load_path):
        model_name = cfg.model_load_path.split("/")[-1]
        model_folder = cfg.model_load_path.split("/")[-2]
        download_wandb(model_name, model_folder, cfg.wandb_load_id)

    if cfg.algo == "ppo":
        model = PPO.load(cfg.model_load_path[:-4])
    elif cfg.algo == "a2c":
        model = A2C.load(cfg.model_load_path[:-4])
    else:
        raise ValueError(f"Unknown algorithm: {cfg.algo}")

    return model


def download_wandb(f_name: str, t_name: str, wandb_id: str):
    """
    Download model weights named f_name for given wandb run into t_name folder.
    """
    api = wandb.Api()
    run = api.run(f"princeton-nlp/il-scale/{wandb_id}")
    return run.file(f_name).download(root=t_name, replace=True)


def conv_outdim(i_dim, k, padding=0, stride=1, dilation=1):
    """Return the dimension after applying a convolution along one axis"""
    return int(1 + (i_dim + 2 * padding - dilation * (k - 1) - 1) / stride)


def count_params(m: nn.Module):
    total_params = sum(p.numel() for p in m.parameters())
    total_trainable_params = sum(p.numel() for p in m.parameters() if p.requires_grad)

    return total_params, total_trainable_params
