import logging
import os
import logging
import socket
from typing import Tuple
import subprocess
import time

import torch
from omegaconf import DictConfig, OmegaConf
import wandb

import torch
import torch.distributed as dist
import gym

# A logger for this file
logging.basicConfig(
    format=(
        "[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] " "%(message)s"
    ),
    level=logging.INFO,
)

class DDPUtil():
    def __init__(self):
        # Let os find unused port
        # From: https://stackoverflow.com/questions/1365265/on-localhost-how-do-i-pick-a-free-port-number
        sock = socket.socket()
        sock.bind(('', 0))
        self.PORT = str(sock.getsockname()[1])
        logging.info(f"Found port {self.PORT}.")

        self.HOST = 'localhost'
    
    def setup(self, rank: int, world_size: int):
        logging.info('Setting up ...')
        os.environ['MASTER_ADDR'] = self.HOST
        os.environ['MASTER_PORT'] = self.PORT
        logging.info(f"Port set: {self.PORT}")

        # Initialize the process group
        dist.init_process_group('nccl', rank=rank, world_size=world_size)

        # Make sure all allocations happen on one device. CUDA OOM without this on rank 0.
        torch.cuda.set_device(rank) 

        # Save rank and world_size
        self.rank = rank 
        self.world_size = world_size
        
        logging.info('Done setting up.')

    @staticmethod
    def cleanup():
        dist.destroy_process_group()

def set_device(cfg: DictConfig):
    if not cfg.setup.disable_cuda and torch.cuda.is_available():
        assign_free_gpus(max_gpus=cfg.setup.num_gpus, logger=logging)
        # Set device to CUDA
        OmegaConf.set_struct(cfg, False)
        cfg.device = "cuda"
        OmegaConf.set_struct(cfg, True)
        logging.info("Using CUDA.")
    else:
        # Set device to CPU
        OmegaConf.set_struct(cfg, False)
        cfg.device = "cpu"
        OmegaConf.set_struct(cfg, True)
        logging.info("Not using CUDA.")


def assign_free_gpus(
    threshold_vram_usage: int = 1000,
    max_gpus: int = 1,
    wait: bool = False,
    sleep_time: int = 10,
    logger=None,
):
    """
    Assigns free gpus to the current process via the CUDA_AVAILABLE_DEVICES env variable
    This function should be called after all imports,
    in case you are setting CUDA_AVAILABLE_DEVICES elsewhere

    Borrowed and fixed from https://gist.github.com/afspies/7e211b83ca5a8902849b05ded9a10696

    Args:
        threshold_vram_usage (int, optional): A GPU is considered free if the vram usage is below the threshold
                                              Defaults to 1500 (MiB).
        max_gpus (int, optional): Max GPUs is the maximum number of gpus to assign.
                                  Defaults to 2.
        wait (bool, optional): Whether to wait until a GPU is free. Default False.
        sleep_time (int, optional): Sleep time (in seconds) to wait before checking GPUs, if wait=True. Default 10.
    """

    def _check():
        # Get the list of GPUs via nvidia-smi
        smi_query_result = subprocess.check_output(
            "nvidia-smi -q -d Memory | grep -A4 GPU", shell=True
        )
        # Extract the usage information
        gpu_info = smi_query_result.decode("utf-8").split("\n")
        gpu_info = list(filter(lambda info: "Used" in info, gpu_info))
        gpu_info = [
            int(x.split(":")[1].replace("MiB", "").strip()) for x in gpu_info
        ]  # Remove garbage
        # Keep gpus under threshold only
        free_gpus = [
            str(i) for i, mem in enumerate(gpu_info) if mem < threshold_vram_usage
        ]
        free_gpus = free_gpus[: min(max_gpus, len(free_gpus))]
        gpus_to_use = ",".join(free_gpus)
        return gpus_to_use

    while True:
        gpus_to_use = _check()
        if gpus_to_use or not wait:
            break
        print(f"No free GPUs found, retrying in {sleep_time}s")
        time.sleep(sleep_time)

    if not gpus_to_use:
        raise RuntimeError("No free GPUs found")
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus_to_use
    logger.info(f"Available GPUS: {gpus_to_use}")


def create_env(
    name, 
    *args, 
    observation_keys: Tuple[str] = (
        "glyphs", "blstats", "inv_glyphs",
        "message", "inv_oclasses", "inv_letters", 
        "inv_strs", "tty_chars", "tty_colors", 
        "tty_cursor", "internal"
        ),
    **kwargs
):
    env = gym.make(
        name, 
        observation_keys=observation_keys,
        *args, 
        **kwargs
    )

    return env