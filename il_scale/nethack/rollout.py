import logging
import os
import fcntl
import random
import configargparse
from typing import List

# Necessary for multithreading.
os.environ["OMP_NUM_THREADS"] = "1"

import torch
from torch import multiprocessing as mp
from nle import nethack
import numpy as np
import scipy.stats as stats

from il_scale.nethack.ddp_util import DDPUtil
from il_scale.nethack.config import Config
from il_scale.nethack.agent import Agent
from il_scale.nethack.utils import (
    create_env,
)
from il_scale.nethack.resetting_env import ResettingEnvironment

mp.set_sharing_strategy(
    "file_system"
)  # see https://github.com/pytorch/pytorch/issues/11201

logging.basicConfig(
    format=(
        "[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] " "%(message)s"
    ),
    level=logging.INFO,
)


class RolloutDashboard:
    """
    Class to keep track of any state or analytics
    during an individual rollout.
    """

    def __init__(self):
        self.steps = 0
        self.max_dlvl = 1

    def get_metrics(self):
        """
        Return all tracked metrics.
        """
        metrics = dict()
        metrics["steps"] = self.steps
        metrics["max_dlvl"] = self.max_dlvl

        return metrics

    def step(self, observation: dict):
        """
        Perform any state/analytics updates.
        """
        # Maybe update max dlvl achieved
        dungeon_level = observation["blstats"][0][0][nethack.NLE_BL_DLEVEL].item()
        if dungeon_level > self.max_dlvl:
            self.max_dlvl = dungeon_level

        # Increase step counter
        self.steps += 1


class Rollout:
    """
    A class used to rollout trained models on the NLE environment.
    """

    def __init__(self, config: Config):
        self.config = config

        # Load trained model config
        self.model_flags = torch.load("nethack_files/model_flags.tar")

    def _agent_setup(self):
        """
        Construct agent and load in weights.
        """
        # Construct agent
        agent = Agent(self.config, None)
        agent.construct_model(self.model_flags)

        # Load checkpoint & weights
        checkpoint = torch.load("nethack_files/model_115.tar", map_location='cpu')
        agent.load(checkpoint["model_state_dict"])

        # Put agent in eval model
        agent.model.eval()

        return agent

    def _get_base_path(self):
        """
        Construct path for main folder where all rollout data
        (returns, ttyrecs, etc.) will be saved.
        """
        # Extract relevant rollout params
        t = self.config.temperature
        topp = self.config.top_p
        topk = self.config.top_k
        steps = self.config.max_episode_steps
        penalty = self.config.rollout_penalty_step
        config_folder_name = (
            f"temp_{t}_topp_{topp}_topk_{topk}_steps_{steps}_penalty_{penalty}"
        )

        run_folder_name = "model_115"

        base_path = os.path.join(
            self.config.test_savedir,
            run_folder_name,
            self.config.rollout_character,
            self.config.sampling_type,
            config_folder_name,
        )

        # Create path if it doesn't exist
        if not os.path.exists(base_path):
            os.makedirs(base_path)

        return base_path

    def _submit_actor(self, ctx, seed: int, idx: int):
        """
        Submit and return actor idx with given seed.
        """
        actor = ctx.Process(
            target=self._single_rollout,
            args=(seed, idx),
            name="Actor-%i" % idx,
        )
        actor.start()

        return actor

    def _get_seeds(self):
        """
        Generate num_rollouts number of seeds.
        """
        return random.sample(list(range(int(1e6))), self.config.num_rollouts)

    def _spawn_rollouts(self):
        """
        Spawn flags.num_actors number of parallel actors to perform rollouts.
        """
        # Make sure we don't copy agent memory
        self.agent.model.share_memory()

        # Spawn actors
        actor_processes = []
        seeds = self._get_seeds()

        # Get context
        ctx = mp.get_context("fork")

        # Spawn first set of actors
        for i in range(self.config.num_actors):
            actor = self._submit_actor(ctx, seeds[i], i)
            actor_processes.append(actor)
        i += 1

        # Keep spawning new processes as old ones finish
        while len(actor_processes) < self.config.num_rollouts:
            if not self.done_q.empty():
                print(self.done_q.get())
                actor = self._submit_actor(ctx, seeds[i], i)
                actor_processes.append(actor)
                i += 1

        # Wait for all actors to finish
        for actor in actor_processes:
            actor.join()

    def _get_ttyrec_folder(self, seed: int, actor_num: int):
        """
        Construct and return folder path to save rollout ttyrecs.
        """
        ttyrec_save_folder = os.path.join(
            self.base_path, "rollouts", f"rollout-{actor_num}-seed-{seed}"
        )
        if not os.path.exists(ttyrec_save_folder) and self.config.save_ttyrec_every:
            os.makedirs(ttyrec_save_folder)

        return ttyrec_save_folder

    def _setup_env(
        self,
        ttyrec_save_folder: str,
        seed: int,
        device: torch.device = torch.device("cpu"),
    ):
        """
        All logic related to setting up the appropriate NLE environment.
        """
        # Setup environment
        if self.config.env == "NetHackChallenge-v0":
            env = create_env(
                self.config.env,
                save_ttyrec_every=self.config.save_ttyrec_every,
                savedir=ttyrec_save_folder,  # will only save here if save_ttyrec_every is nonzero
                penalty_time=0.0,
                penalty_step=self.config.rollout_penalty_step,
                max_episode_steps=self.config.max_episode_steps,
                no_progress_timeout=150,
                character=self.config.rollout_character
            )
            logging.info(f"Rolling out with {self.config.rollout_character} ...")
        else:
            env = create_env(
                self.config.env,
                save_ttyrec_every=self.config.save_ttyrec_every,
                savedir=ttyrec_save_folder,  # will only save here if save_ttyrec_every is nonzero
                penalty_time=0.0,
                penalty_step=self.config.rollout_penalty_step,
                max_episode_steps=self.config.max_episode_steps
            )

        # Set seed
        if self.config.env != "NetHackChallenge-v0":
            env.seed(seed, seed)

        env_keys = ("tty_chars", "tty_colors", "tty_cursor", "blstats")

        env = ResettingEnvironment(
            env,
            num_lagged_actions=self.model_flags.lagged_actions,
            env_keys=env_keys,
            device=device,
        )

        return env

    @torch.no_grad()
    def _single_rollout(
        self, seed: int, actor_num: int, device: torch.device = torch.device("cpu")
    ):
        """
        Rollout and log relevant objects (observations, actions, returns).
        """
        ttyrec_save_folder = self._get_ttyrec_folder(seed, actor_num)

        env = self._setup_env(ttyrec_save_folder, seed, device)

        observation = env.initial()
        observation["prev_action"] = observation["last_action"]  # key name conversion
        agent_state = self.agent.initial_state(batch_size=1)

        frame_stack_chars = torch.zeros(
            (
                1,
                self.model_flags.obs_frame_stack - 1,
                nethack.nethack.TERMINAL_SHAPE[0],
                nethack.nethack.TERMINAL_SHAPE[1],
            )
        ).to(device)
        frame_stack_colors = frame_stack_chars.clone()
        # Zeros are unseen in training, add 32 to make it like end of game frame
        if self.model_flags.obs_frame_stack > 1:
            frame_stack_chars += 32

        dashboard = RolloutDashboard()

        while dashboard.steps < self.config.max_episode_steps:
            # Stack frames
            observation["tty_chars"] = torch.cat(
                [frame_stack_chars, observation["tty_chars"]], dim=1
            ).unsqueeze(1)
            observation["tty_colors"] = torch.cat(
                [frame_stack_colors, observation["tty_colors"]], dim=1
            ).unsqueeze(1)

            # Update frame stack
            if self.model_flags.obs_frame_stack > 1:
                frame_stack_chars = observation["tty_chars"][
                    :, 0, -(self.model_flags.obs_frame_stack - 1) :
                ].clone()
                frame_stack_colors = observation["tty_colors"][
                    :, 0, -(self.model_flags.obs_frame_stack - 1) :
                ].clone()

            observation["done"] = observation["done"].bool()

            # Forward
            policy_outputs, agent_state = self.agent.predict(observation, agent_state)

            # Step through env
            observation = env.step(policy_outputs["action"])
            observation["prev_action"] = observation[
                "last_action"
            ]  # key name conversion

            # Update dashboard
            dashboard.step(observation)

            # Check if rollout is done
            if observation["done"].item():
                logging.info("Reached done signal.")
                self._wrap_up_rollout(observation, dashboard, ttyrec_save_folder)
                break
        else:
            logging.info("Cutting episode short ...")
            # Episode might not have finished
            self._wrap_up_rollout(observation, dashboard, ttyrec_save_folder)

        env.close()

    def _wrap_up_rollout(
        self, observation, dashboard: RolloutDashboard, ttyrec_save_folder: str
    ):
        """
        Do any final logging/saving/etc. that needs to happen
        when the game ends.
        """
        metrics = dashboard.get_metrics()
        metrics["episode_return"] = observation["episode_return"].item()

        logging.info(
            "Episode ended after %d steps. Return: %.1f",
            observation["episode_step"].item(),
            observation["episode_return"].item(),
        )
        logging.info(f"{metrics}")

        # log to file
        self._save_metric_to_file(metrics["steps"], "episode_lengths.txt")
        self._save_metric_to_file(observation["episode_return"].item(), "returns.txt")
        self._save_metric_to_file(metrics["max_dlvl"], "dungeon_levels.txt")

        if self.config.save_ttyrec_every:
            np.save(
                os.path.join(ttyrec_save_folder, "act_history.npy"),
                dashboard.act_history,
            )

        self.metrics_q.put(metrics)

        if self.done_q:
            self.done_q.put("done!")

    def _save_metric_to_file(self, metric: int, file_name: str):
        """
        Writes given metric to file_name in base_path folder.
        """
        # Construct path
        path = os.path.join(self.base_path, file_name)

        # Write when file is available
        with open(path, "a") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            f.write(f"{metric}\n")
            fcntl.flock(f, fcntl.LOCK_UN)

    def rollout_cpu(self):
        """
        Rollout trained model ~flags.num_rollouts number of times on CPU.
        """
        self.agent = self._agent_setup()
        self.base_path = self._get_base_path()

        self.metrics_q = mp.Manager().Queue()
        self.done_q = mp.Manager().Queue()
        self._spawn_rollouts()

        self._post_process()

    def rollout_gpu(self):
        """
        Rollout trained model ~flags.num_rollouts number of times on GPU.
        """
        self.agent = self._agent_setup()
        self.base_path = self._get_base_path()

        ddp_util = DDPUtil()

        seeds = self._get_seeds()

        self.metrics_q = mp.Manager().Queue()
        self.done_q = mp.Manager().Queue()
        mp.spawn(
            self._rollout_chunk_gpu,
            args=(self.config.num_gpus, ddp_util, seeds),
            nprocs=self.config.num_gpus,
            join=True,
        )

        self._post_process()

    def _rollout_chunk_gpu(
        self, rank: int, world_size: int, ddp_util: DDPUtil, seeds: List[int]
    ):
        """
        TODO
        """
        ddp_util.setup(rank, world_size)
        self.agent.to(rank)
        self.agent.move_to_ddp(rank, world_size)
        seeds = seeds[
            rank * len(seeds) // world_size : (rank + 1) * len(seeds) // world_size
        ]

        for idx, seed in enumerate(seeds):
            self._single_rollout(seed, idx, rank)

    def _post_process(self):
        """
        Compute and save final metrics.
        """
        returns = []
        episode_lens = []
        while not self.metrics_q.empty():
            metrics = self.metrics_q.get()

            # returns
            returns.append(metrics["episode_return"])

            # episode lens
            episode_lens.append(metrics["steps"])

        logging.info(f"Avg. return: {np.mean(returns)}")
        logging.info(f"95% CI: {str(stats.t.interval(0.95, len(returns)-1, loc=np.mean(returns), scale=stats.sem(returns)))}")


def main(flags):
    config = Config(flags)

    rollout = Rollout(config)
    if flags.use_gpu:
        rollout.rollout_gpu()
    else:
        rollout.rollout_cpu()


if __name__ == "__main__":
    parser = configargparse.ArgumentParser()
    parser.add_argument(
        "--parameter_file",
        type=str,
        is_config_file=True,
        help="Parameter file (yaml) to read in all arguments.",
    )
    parser.add_argument("--test_savedir", type=str, default="logs")
    parser.add_argument("--num_actors", type=int, default=1)
    parser.add_argument("--model_load_name", type=str, default="nethack_files/model_115.tar")
    parser.add_argument("--sampling_type", type=str, default="softmax")
    parser.add_argument("--top_p", type=float, default=0.90)
    parser.add_argument("--top_k", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--save_ttyrec_every", type=int, default=0)
    parser.add_argument("--rollout_penalty_step", type=float, default=0.0)
    parser.add_argument("--max_episode_steps", type=int, default=100000)
    parser.add_argument("--rollout_character", type=str, default="@")
    parser.add_argument("--env", type=str, default="NetHackChallenge-v0")
    parser.add_argument("--num_rollouts", type=int, default=1000)
    parser.add_argument("--use_gpu", action="store_true")
    parser.add_argument("--num_gpus", type=int, default=1)

    flags = parser.parse_args()

    main(flags)
