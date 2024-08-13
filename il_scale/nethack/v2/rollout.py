import logging
import os
from collections import defaultdict, Counter
import fcntl
import random
import configargparse
from typing import List
from functools import reduce

# Necessary for multithreading.
os.environ["OMP_NUM_THREADS"] = "1"

import torch
from torch import multiprocessing as mp
from torch.distributions import Categorical
from nle import nethack
import numpy as np
import scipy.stats as stats
from mamba_ssm.utils.generation import InferenceParams
import hydra
from omegaconf import DictConfig, OmegaConf

from il_scale.nethack.v2.utils.setup import DDPUtil, create_env
from il_scale.nethack.v2.utils.model import load_checkpoint
from il_scale.nethack.v2.utils.sokoban_wrapper import TaskRewardsInfoWrapper
from il_scale.nethack.v2.agent import Agent
from il_scale.nethack.v2.resetting_env import ResettingEnvironment

mp.set_sharing_strategy('file_system') # see https://github.com/pytorch/pytorch/issues/11201

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
        self.temp_act_dict = defaultdict(int)
        self.act_history = []
        self.ppl = 0
        self.top_k_count = defaultdict(int)
    
    def avg_ppl(self):
        """
        Return average perplexity across time steps.
        """
        return self.ppl/self.steps

    def get_metrics(self):
        """
        Return all tracked metrics.
        """
        metrics = dict()
        metrics["avg_ppl"] = self.avg_ppl().item()
        metrics["act_counts"] = self.temp_act_dict
        metrics["steps"] = self.steps
        metrics["max_dlvl"] = self.max_dlvl
        metrics["top_k_count"] = self.top_k_count

        return metrics
    
    def step(self, observation: dict, policy_outputs: torch.tensor, action: int):
        """
        Perform any state/analytics updates.
        """
        # Count actions
        self.temp_act_dict[action] += 1

        # Build action history
        self.act_history.append(action)

        # Update perplexity sum
        self.ppl += torch.exp(Categorical(logits=policy_outputs["policy_logits"].flatten()).entropy())

        # Maybe update max dlvl achieved
        dungeon_level = observation["blstats"][0][0][nethack.NLE_BL_DLEVEL].item()
        if dungeon_level > self.max_dlvl:
            self.max_dlvl = dungeon_level

        # Update top-k count
        p_logits_sort = sorted([(idx, l) for idx, l in enumerate(policy_outputs["policy_logits"].flatten().tolist())], key=lambda x: x[-1], reverse=True)
        for rank, (idx, _) in enumerate(p_logits_sort):
            if action == idx:
                self.top_k_count[rank + 1] += 1
                break

        # Increase step counter
        self.steps += 1

class Rollout:
    """
    A class used to rollout trained models on the NLE environment.
    """
    def __init__(self, config: DictConfig):
        self.config = config
        self.model_flags = self.config

    def _agent_setup(self):
        """
        Construct agent and load in weights.
        """
        # Construct agent
        agent = Agent(self.config, None)
        agent.construct_model(self.model_flags)

        # Load checkpoint & weights
        checkpoint = load_checkpoint(self.config.rollout.model_load_name)
        agent.load(checkpoint["model_state_dict"])

        # Put agent in eval model
        agent.model.eval()

        return agent

    def _get_base_path(self):
        """
        Construct path for main folder where all rollout data 
        (returns, ttyrecs, etc.) will be saved.
        """
        run_name = self.config.rollout.model_load_name

        # Extract relevant rollout params
        t = self.config.rollout.temperature
        topp = self.config.rollout.top_p
        topk = self.config.rollout.top_k
        steps = self.config.rollout.max_episode_steps
        penalty = self.config.rollout.rollout_penalty_step
        tag = self.config.rollout.rollout_tag
        config_folder_name = f"temp_{t}_topp_{topp}_topk_{topk}_steps_{steps}_penalty_{penalty}_tag_{tag}"
        run_folder_name = run_name

        base_path = os.path.join(
            self.config.rollout.rollout_save_dir, 
            run_folder_name, 
            self.config.rollout.rollout_character, 
            self.config.rollout.sampling_type, 
            config_folder_name
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
            args=(
                seed,
                idx
            ),
            name="Actor-%i" % idx,
        )
        actor.start()

        return actor

    def _get_seeds(self):
        """
        Generate num_rollouts number of seeds.
        """
        return random.sample(list(range(int(1e6))), self.config.rollout.num_rollouts)

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
        for i in range(self.config.rollout.num_actors):
            actor = self._submit_actor(ctx, seeds[i], i)
            actor_processes.append(actor)
        i += 1

        # Keep spawning new processes as old ones finish
        while len(actor_processes) < self.config.rollout.num_rollouts:
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
        ttyrec_save_folder = os.path.join(self.base_path, "rollouts", f"rollout-{actor_num}-seed-{seed}")
        if not os.path.exists(ttyrec_save_folder) and self.config.rollout.save_ttyrec_every:
            os.makedirs(ttyrec_save_folder)

        return ttyrec_save_folder

    def _setup_env(self, ttyrec_save_folder: str, seed: int, device: torch.device = torch.device('cpu'), sokoban_state: str = None):
        """
        All logic related to setting up the appropriate NLE environment.
        """

        # Setup environment
        if self.config.rollout.env == 'NetHackChallenge-v0':
            gym_env = create_env(
                self.config.rollout.env, 
                save_ttyrec_every=self.config.rollout.save_ttyrec_every,
                savedir=ttyrec_save_folder, # will only save here if save_ttyrec_every is nonzero
                penalty_time=0.0,
                penalty_step=self.config.rollout.rollout_penalty_step,
                max_episode_steps=self.config.rollout.max_episode_steps,
                no_progress_timeout=10_000,
                character=self.config.rollout.rollout_character,
            )
            logging.info(f"Rolling out with {self.config.rollout.rollout_character} ...")
            logging.info(f"Sokoban state: {sokoban_state}")
        else:
            gym_env = create_env(
                self.config.rollout.env, 
                save_ttyrec_every=self.config.rollout.save_ttyrec_every,
                savedir=ttyrec_save_folder, # will only save here if save_ttyrec_every is nonzero
                penalty_time=0.0,
                penalty_step=self.config.rollout.rollout_penalty_step,
                max_episode_steps=self.config.rollout.max_episode_steps,
            )

        # Set seed
        if self.config.rollout.env != 'NetHackChallenge-v0':
            logging.info(f'Seeding with seed: {seed}')
            gym_env.seed(seed, seed)

        gym_env = TaskRewardsInfoWrapper(gym_env)

        env_keys = ("tty_chars", "tty_colors", "tty_cursor", "blstats", "glyphs", "inv_glyphs", "message")
        env = ResettingEnvironment(
            gym_env, 
            num_lagged_actions=0,
            env_keys=env_keys,
            device=device
        )

        return env

    @torch.no_grad()
    def _single_rollout(self, seed: int, actor_num: int, device: torch.device = torch.device('cpu'), sokoban_state: str = None):
        """
        Rollout and log relevant objects (observations, actions, returns).
        """
        inference_params = InferenceParams(max_seqlen=self.config.rollout.max_seqlen, max_batch_size=1)
        ttyrec_save_folder = self._get_ttyrec_folder(seed, actor_num)

        env = self._setup_env(ttyrec_save_folder, seed, device, sokoban_state=sokoban_state)

        observation = env.initial()
        observation["prev_action"] = observation["last_action"] # key name conversion
        
        if sokoban_state is not None:
            # mask out the welcome back message
            observation["tty_chars"][..., 0, :] = 32
            observation["tty_colors"][..., 0, :] = 0

        frame_stack_chars = torch.zeros((1, self.model_flags.network.obs_frame_stack - 1, nethack.nethack.TERMINAL_SHAPE[0], nethack.nethack.TERMINAL_SHAPE[1])).to(device)
        frame_stack_colors = frame_stack_chars.clone()
        # Zeros are unseen in training, add 32 to make it like end of game frame
        if self.model_flags.network.obs_frame_stack > 1:
            frame_stack_chars += 32

        dashboard = RolloutDashboard()

        while dashboard.steps < self.config.rollout.max_episode_steps:
            # Stack frames
            observation["tty_chars"] = torch.cat([frame_stack_chars, observation["tty_chars"]], dim=1).unsqueeze(1)
            observation["tty_colors"] = torch.cat([frame_stack_colors, observation["tty_colors"]], dim=1).unsqueeze(1)

            # Update frame stack
            if self.model_flags.network.obs_frame_stack > 1:
                frame_stack_chars = observation["tty_chars"][:, 0, -(self.model_flags.network.obs_frame_stack - 1):].clone()
                frame_stack_colors = observation["tty_colors"][:, 0, -(self.model_flags.network.obs_frame_stack - 1):].clone()

            observation["done"] = observation["done"].bool()

            # Forward
            policy_outputs = self.agent.predict(observation, inference_params=inference_params)

            observation, info = env.step(policy_outputs["action"])
            inference_params.seqlen_offset += 1

            observation["prev_action"] = observation["last_action"] # key name conversion

            # Update dashboard
            dashboard.step(observation, policy_outputs, policy_outputs["action"].item())

            # Check if rollout is done
            if observation["done"].item():
                logging.info("Reached done signal.")
                self._wrap_up_rollout(observation, dashboard, ttyrec_save_folder, info)
                break
        else: 
            logging.info("Cutting episode short ...")
            # Episode might not have finished
            self._wrap_up_rollout(observation, dashboard, ttyrec_save_folder, info)

        env.close()

    def _wrap_up_rollout(self, observation, dashboard: RolloutDashboard, ttyrec_save_folder: str, info):
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
        logging.info(f'{metrics}')

        # log to file
        self._save_metric_to_file(metrics["steps"], "episode_lengths.txt")
        self._save_metric_to_file(observation["episode_return"].item(), "returns.txt")
        self._save_metric_to_file(metrics["max_dlvl"], "dungeon_levels.txt")
        self._save_metric_to_file(info["episode_extra_stats"]["sokobanfillpit_score"], "sokobanfillpit_score.txt")
        self._save_metric_to_file(info["episode_extra_stats"]["sokobansolvedlevels_score"], "sokobansolvedlevels_score.txt")
        self._save_metric_to_file(info["episode_extra_stats"]["sokoban_reached_score"], "sokoban_reached_score.txt")

        if self.config.rollout.save_ttyrec_every:
            np.save(os.path.join(ttyrec_save_folder, "act_history.npy"), dashboard.act_history)

        self.metrics_q.put(metrics)

        if self.done_q:
            self.done_q.put('done!')

    def _save_metric_to_file(self, metric: float, file_name: str):
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
            args=(self.config.rollout.num_gpus, ddp_util, seeds),
            nprocs=self.config.rollout.num_gpus,
            join=True
        )

        self._post_process()

    def _rollout_chunk_gpu(self, rank: int, world_size: int, ddp_util: DDPUtil, seeds: List[int]):
        """
        TODO
        """
        ddp_util.setup(rank, world_size)
        self.agent.to(rank)
        self.agent.move_to_ddp(rank, world_size)
        seeds = seeds[rank * len(seeds)//world_size:(rank + 1) * len(seeds)//world_size]

        for idx, seed in enumerate(seeds):
            self._single_rollout(seed, idx, rank)

    def _post_process(self):
        """
        Compute and save final metrics.
        """
        # Action frequencies
        act_freq_path = os.path.join(self.base_path, "act_freq.tar")
        act_freqs = []
        act_all_keys = set()

        # Perplexity
        ppl_path = os.path.join(self.base_path, "ppl.tar")
        ppl = 0

        # Top-k counts
        top_k_counts_path = os.path.join(self.base_path, "top_k_counts.tar")
        top_k_counts = []
        
        total_steps = 0

        returns = []
        episode_lens = []
        while not self.metrics_q.empty():
            metrics = self.metrics_q.get()

            # returns
            returns.append(metrics["episode_return"])

            # episode lens
            episode_lens.append(metrics["steps"])
            
            # Total number of actions in episode
            total_num_actions = np.sum(list(metrics["act_counts"].values()))

            # Normalize
            for key in metrics["act_counts"].keys():
                metrics["act_counts"][key] /= total_num_actions

            # Collect all action frequency dicts
            act_freqs.append(metrics["act_counts"])

            # Keep track of all actions seen across episodes
            act_all_keys = act_all_keys.union(set(metrics["act_counts"].keys()))

            # Update avg perplexity
            ppl = (ppl * total_steps + metrics["avg_ppl"] * metrics["steps"])/(total_steps + metrics["steps"])

            # Add to top-k counts array
            top_k_counts.append(metrics["top_k_count"])

            total_steps += metrics["steps"]

        # Aggregate action frequency dictionaries
        agg_act_freq = Counter()
        for key in act_all_keys:
            obs = [act_freq[key] if key in act_freq else 0.0 for act_freq in act_freqs]
            agg_act_freq[key] = (np.average(obs), stats.sem(obs))

        # Save
        torch.save(agg_act_freq, act_freq_path)
        torch.save({"avg_ppl": ppl}, ppl_path)
        torch.save({"top_k_counts": top_k_counts}, top_k_counts_path)

        # Write stats
        self._save_metric_to_file(np.mean(returns), "return_stats.txt")
        self._save_metric_to_file(np.median(returns), "return_stats.txt")
        self._save_metric_to_file(stats.sem(returns), "return_stats.txt")
        self._save_metric_to_file(np.max(returns), "return_stats.txt")
        self._save_metric_to_file(np.min(returns), "return_stats.txt")

        self._save_metric_to_file(np.mean(episode_lens), "episode_length_stats.txt")
        self._save_metric_to_file(np.median(episode_lens), "episode_length_stats.txt")
        self._save_metric_to_file(stats.sem(episode_lens), "episode_length_stats.txt")
        self._save_metric_to_file(np.max(episode_lens), "episode_length_stats.txt")
        self._save_metric_to_file(np.min(episode_lens), "episode_length_stats.txt")

@hydra.main(version_base=None, config_path="../../../conf", config_name="nethack_config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    rollout = Rollout(cfg)
    if cfg.rollout.use_gpu:
        rollout.rollout_gpu()
    else:
        rollout.rollout_cpu()

if __name__ == "__main__":
    main()