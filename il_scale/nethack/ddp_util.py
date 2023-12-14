import os
import logging
import socket

import torch
import torch.distributed as dist


class DDPUtil:
    def __init__(self):
        # Let os find unused port
        # From: https://stackoverflow.com/questions/1365265/on-localhost-how-do-i-pick-a-free-port-number
        sock = socket.socket()
        sock.bind(("", 0))
        self.PORT = str(sock.getsockname()[1])
        logging.info(f"Found port {self.PORT}.")

        self.HOST = "localhost"

    def setup(self, rank: int, world_size: int):
        logging.info("Setting up ...")
        os.environ["MASTER_ADDR"] = self.HOST
        os.environ["MASTER_PORT"] = self.PORT
        logging.info(f"Port set: {self.PORT}")

        # Initialize the process group
        dist.init_process_group("nccl", rank=rank, world_size=world_size)

        # Make sure all allocations happen on one device. CUDA OOM without this on rank 0.
        torch.cuda.set_device(rank)

        # Save rank and world_size
        self.rank = rank
        self.world_size = world_size

        logging.info("Done setting up.")

    @staticmethod
    def cleanup():
        dist.destroy_process_group()
