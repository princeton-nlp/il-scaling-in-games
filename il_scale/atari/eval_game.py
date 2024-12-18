import logging
import os
import time

import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, VecMonitor
from stable_baselines3.common.evaluation import evaluate_policy
import pyarrow as pa
import pyarrow.parquet as pq
import scipy.stats as stats
import numpy as np

from il_scale.utils.setup_utils import setup_wandb
from il_scale.utils.model_utils import load_model

from il_scale.data.atari_schemas import OBS_SCHEMA

# A logger for this file
log = logging.getLogger(__name__)


def eval_model(env, model, max_episodes):
    obs = env.reset()
    scores = []
    episodes = 0
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)

        for i in range(env.num_envs):
            if dones[i] and "episode" in info[i].keys():
                scores.append(info[i]["episode"]["r"])
                print("ended with score", info[i]["episode"]["r"])
                episodes += 1
                if episodes >= max_episodes:
                    return scores


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    log.info(OmegaConf.to_yaml(cfg))

    wandb_conf = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    # NOTE: Frame skip = 4, repeat action probability = 0.25
    vec_env = make_atari_env(
        f"{cfg.atari.name}Deterministic-v0", n_envs=1, seed=cfg.atari.seed
    )
    vec_env = VecFrameStack(vec_env, n_stack=4)

    model = load_model(cfg.atari)

    scores = eval_model(vec_env, model, cfg.atari.max_episodes)

    print("expert score:", np.mean(scores))
    print(
        "ci",
        stats.t.interval(
            0.95, len(scores) - 1, loc=np.mean(scores), scale=stats.sem(scores)
        ),
    )


if __name__ == "__main__":
    main()
