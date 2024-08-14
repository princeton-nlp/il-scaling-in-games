# Standard library imports
import os
import traceback
import logging

# Necessary for multithreading.
os.environ["OMP_NUM_THREADS"] = "1"

# Third party imports
from torch import multiprocessing as mp
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf

# Local application imports
from il_scale.nethack.v2.utils.setup import DDPUtil, set_device
from il_scale.nethack.v2.utils.model import count_params
from il_scale.nethack.v2.agent import Agent
from il_scale.nethack.v2.data.tty_data import TTYData
from il_scale.nethack.v2.trainers.bc_trainer import BCTrainer
from il_scale.nethack.v2.logger import Logger

logging.basicConfig(
    format=(
        "[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] " "%(message)s"
    ),
    level=logging.INFO,
)

def train(
    rank: int,
    world_size: int,
    ddp_util: DDPUtil,
    cfg: DictConfig,
    data: TTYData,
):
    # Only setup logger in rank 0 process
    logger = Logger(cfg)
    if rank == 0:
        logger.setup()

    # Setup DDP
    ddp_util.setup(rank, world_size)

    # Create agents
    agent = Agent(cfg, logger)
    agent.construct_model()

    # Log model size to wandb
    total_params = count_params(agent.model)
    if rank == 0:
        wandb.log({"num_params": total_params})
    logging.info("Created model with {} total params.".format(total_params))

    # Move to GPU
    agent.to(rank)

    # Create trainer
    bc_trainer = BCTrainer(cfg, logger, agent, data, ddp_util)

    # Move to DDP
    agent.move_to_ddp(rank, world_size, find_unused_parameters=cfg.network.core_mode!="mamba")

    # Start training
    bc_trainer.train()

    if rank == 0:
        logger.shutdown()


@hydra.main(version_base=None, config_path="../../conf", config_name="nethack_config")
def main(cfg: DictConfig) -> None:
    try:
        logging.info(OmegaConf.to_yaml(cfg))
        set_device(cfg)

        data = TTYData(cfg.data)

        ddp_util = DDPUtil()

        mp.spawn(
            train,
            args=(cfg.setup.num_gpus, ddp_util, cfg, data),
            nprocs=cfg.setup.num_gpus,
            join=True,
        )

    except Exception:
        DDPUtil.cleanup()
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
