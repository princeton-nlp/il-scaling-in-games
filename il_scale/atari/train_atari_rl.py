import logging

import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from wandb.integration.sb3 import WandbCallback

from il_scale.atari.utils.setup_utils import setup_wandb, set_seeds
from il_scale.atari.utils.model_utils import create_model, count_params

# A logger for this file
log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    log.info(OmegaConf.to_yaml(cfg))
    set_seeds(cfg.atari.seed)

    wandb_conf = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    setup_wandb(cfg.exp.wandb_name, wandb_conf)

    # NOTE: Frame skip = 4, repeat action probability = 0.25
    vec_env = make_atari_env(
        f"{cfg.atari.name}Deterministic-v0", n_envs=8, seed=cfg.atari.seed
    )
    vec_env = VecFrameStack(vec_env, n_stack=4)

    model = create_model(cfg.atari, vec_env)

    model.learn(
        total_timesteps=cfg.atari.total_timesteps,
        callback=WandbCallback(
            verbose=2,
            model_save_freq=cfg.atari.save_freq,
            model_save_path=f"./models/{wandb.run.id}",
        ),
        progress_bar=True,
    )


if __name__ == "__main__":
    main()
