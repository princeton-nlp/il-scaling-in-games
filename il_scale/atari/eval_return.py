import time
import logging
from concurrent.futures import ThreadPoolExecutor
import os
from collections import defaultdict

import torch.nn as nn
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, VecMonitor
import torch
import numpy as np

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
def evaluate(model, env, cfg):
    returns = []
    episodes = 0
    obs = env.reset()
    obs = np.expand_dims(obs, axis=0)
    while True:
        obs = torch.from_numpy(obs).to(cfg.atari.device)
        logits = model(obs.float())
        action = torch.distributions.Categorical(logits=logits).sample()
        obs, rewards, dones, info = env.step([action.cpu().item()])
        obs = np.expand_dims(obs, axis=0)

        if dones[0] and "episode" in info[0].keys():
            episodes += 1

            returns.append(info[0]["episode"]["r"])

            if episodes >= cfg.eval.num_episodes:
                return sum(returns) / len(returns)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    log.info(OmegaConf.to_yaml(cfg))
    set_seeds(cfg.atari.seed)

    save_results_path = (
        f"data_objects/eval_return_{cfg.atari.name}_{cfg.exp.train_seed}.tar"
    )
    if os.path.exists(save_results_path):
        log.info("Loading eval return from file ...")
        return_all = torch.load(save_results_path)
    else:
        return_all = defaultdict(dict)

    vec_env = make_atari_env(
        f"{cfg.atari.name}Deterministic-v0", n_envs=1, seed=cfg.atari.seed
    )
    vec_env = VecFrameStack(vec_env, n_stack=4)

    api = wandb.Api()
    runs = api.runs("princeton-nlp/il-scale")
    for run in runs:
        if cfg.exp.iso_flop_tag in run.tags:
            for flop in FLOPS:
                if os.path.exists(f"models/{run.id}/flops_{flop}.tar"):
                    log.info(f"Loading models/{run.id}/flops_{flop}.tar ...")
                    log.info("Run name: " + run.name)
                    chkpt = torch.load(
                        f"models/{run.id}/flops_{flop}.tar", map_location="cpu"
                    )
                    num_params = chkpt["num_params"]

                    if flop in return_all and num_params in return_all[flop]:
                        log.info(
                            f"already evaluated {flop:e} with params {num_params:e}, skipping."
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
                    eval_return = evaluate(model, vec_env, cfg)

                    return_all[flop][num_params] = eval_return
                    log.info(flop)
                    log.info(f"{count_params(model)[0]:e}")

                    # save
                    if not os.path.exists("data_objects"):
                        os.makedirs(f"data_objects", exist_ok=False)

                    log.info("Saving eval return to file ...")
                    torch.save(return_all, save_results_path)
                else:
                    break

    # save
    if not os.path.exists("data_objects"):
        os.makedirs(f"data_objects", exist_ok=False)

    log.info("Saving eval return to file ...")
    torch.save(return_all, save_results_path)


if __name__ == "__main__":
    main()
