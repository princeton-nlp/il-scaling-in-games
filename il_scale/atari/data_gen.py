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
import numpy as np

from il_scale.atari.utils.setup_utils import setup_wandb
from il_scale.atari.utils.model_utils import load_model

from il_scale.atari.data.atari_schemas import OBS_SCHEMA

# A logger for this file
log = logging.getLogger(__name__)


class DataGenerator:
    def __init__(self, env, model, dataset_path: str) -> None:
        self.env = env
        self.model = model
        self.dataset_path = dataset_path

        self.meta_schema = pa.schema([("score", pa.int32()), ("steps", pa.uint32())])

    def generate(self, max_episodes: int, seed: int):
        states = [[] for _ in range(self.env.num_envs)]
        actions = [[] for _ in range(self.env.num_envs)]
        scores = [0 for _ in range(self.env.num_envs)]

        episodes = 0
        steps = 0
        ents = []

        obs = self.env.reset()
        for i in range(self.env.num_envs):
            states[i].append(obs[i].flatten())
        while True:
            ents.append(self.model.policy.get_distribution(self.model.policy.obs_to_tensor(obs)[0]).entropy().item())

            action, _states = self.model.predict(obs, deterministic=True)
            obs, rewards, dones, info = self.env.step(action)
            steps += 1

            for i in range(self.env.num_envs):
                actions[i].append(action[i])
                scores[i] += rewards[i]
                if dones[i] and "episode" in info[i].keys():
                    # save data
                    obs_table = pa.table([states[i], actions[i]], schema=OBS_SCHEMA)

                    meta_table = pa.table(
                        [
                            [info[i]["episode"]["r"]],
                            [steps],
                        ],
                        schema=self.meta_schema,
                    )

                    gid = seed * max_episodes + episodes
                    rollout_folder = os.path.join(self.dataset_path, str(gid))
                    if not os.path.exists(rollout_folder):
                        os.makedirs(rollout_folder, exist_ok=False)

                    rollout_path = os.path.join(
                        self.dataset_path, str(gid), "rollout.parquet"
                    )
                    meta_path = os.path.join(
                        self.dataset_path, str(gid), "metadata.parquet"
                    )

                    pq.write_table(obs_table, rollout_path)
                    pq.write_table(meta_table, meta_path)

                    states[i] = []
                    actions[i] = []
                    scores[i] = 0
                    steps = 0

                    episodes += 1
                    if episodes >= max_episodes:
                        return ents

                states[i].append(obs[i].flatten())


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    log.info(OmegaConf.to_yaml(cfg))

    wandb_conf = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    setup_wandb(cfg.exp.wandb_name, wandb_conf)

    # NOTE: Frame skip = 4, repeat action probability = 0.25
    vec_env = make_atari_env(
        f"{cfg.atari.name}Deterministic-v0", n_envs=1, seed=cfg.atari.seed
    )
    vec_env = VecFrameStack(vec_env, n_stack=4)

    model = load_model(cfg.atari)

    # create dataset folder
    if not os.path.exists(cfg.atari.dataset_path):
        os.makedirs(cfg.atari.dataset_path, exist_ok=False)

    data_generator = DataGenerator(vec_env, model, cfg.atari.dataset_path)
    # data_generator.generate(max_episodes=cfg.atari.max_episodes, seed=cfg.atari.seed)
    ents = data_generator.generate(max_episodes=10, seed=cfg.atari.seed)
    print(f'Avg. Entropy: {np.mean(ents)}')


if __name__ == "__main__":
    main()
