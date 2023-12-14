import logging

try:
    import torch
    from torch import multiprocessing as mp
    from torch import nn
    from torch.nn import functional as F
except ImportError:
    logging.exception(
        "PyTorch not found. Please install the agent dependencies with "
        '`pip install "nle[agent]"`'
    )

import gym  # noqa: E402


def _format_observations(
    observation,
    keys=("tty_chars", "tty_colors", "tty_cursor", "blstats"),
    device: torch.device = torch.device("cpu"),
):
    observations = {}
    for key in keys:
        entry = observation[key]
        entry = torch.from_numpy(entry).to(device)
        entry = entry.view((1, 1) + entry.shape)  # (...) -> (T,B,...).
        observations[key] = entry
    return observations


class ResettingEnvironment:
    """Turns a Gym environment into something that can be step()ed indefinitely."""

    def __init__(
        self,
        gym_env,
        num_lagged_actions=5,
        env_keys=None,
        device: torch.device = torch.device("cpu"),
    ):
        self.device = device
        self.gym_env = gym_env
        self.episode_return = None
        self.episode_step = None
        self.num_actions = self.gym_env.action_space.n
        self.num_lagged_actions = num_lagged_actions
        self.lagged_actions = (self.num_actions + 1) * torch.ones(
            1, self.num_lagged_actions, dtype=torch.int64
        )
        self.env_keys = (
            tuple(gym_env.observation_space.spaces.keys())
            if env_keys is None
            else env_keys
        )

    def initial(self):
        initial_reward = torch.zeros(1, 1)
        # This supports only single-tensor actions ATM.
        initial_last_action = torch.zeros(1, 1, dtype=torch.int64)
        self.episode_return = torch.zeros(1, 1)
        self.episode_step = torch.zeros(1, 1, dtype=torch.int32)
        initial_done = torch.ones(1, 1, dtype=torch.uint8)
        self.lagged_actions = (self.num_actions + 1) * torch.ones(
            1, self.num_lagged_actions, dtype=torch.int64
        )
        self.last_game_end_type = None

        result = _format_observations(self.gym_env.reset(), self.env_keys, self.device)
        result.update(
            reward=initial_reward.to(self.device),
            done=initial_done.to(self.device),
            episode_return=self.episode_return,
            episode_step=self.episode_step,
            last_action=initial_last_action.to(self.device),
            lagged_actions=self.lagged_actions,
        )
        return result

    def _update_lagged_actions(self, action):
        if self.num_lagged_actions > 0:
            self.lagged_actions[:, :-1] = self.lagged_actions[:, 1:].clone()
            self.lagged_actions[:, -1] = action

    def step(self, action: torch.tensor):
        observation, reward, done, unused_info = self.gym_env.step(action.item())

        self._update_lagged_actions(action)
        self.episode_step += 1
        self.episode_return += reward
        episode_step = self.episode_step
        episode_return = self.episode_return
        if done:
            # Get cause of death category
            self.last_game_end_type = self.gym_env.nethack.how_done()
            observation = self.gym_env.reset()
            self.episode_return = torch.zeros(1, 1)
            self.episode_step = torch.zeros(1, 1, dtype=torch.int32)
            self.lagged_actions = (self.num_actions + 1) * torch.ones(
                1, self.num_lagged_actions, dtype=torch.int64
            )

        result = _format_observations(observation, self.env_keys, self.device)

        reward = torch.tensor(reward).view(1, 1)
        done = torch.tensor(done).view(1, 1)

        result.update(
            reward=reward.to(self.device),
            done=done.to(self.device),
            episode_return=episode_return,
            episode_step=episode_step,
            last_action=action.to(self.device),
            lagged_actions=self.lagged_actions,
        )
        return result

    def close(self):
        self.gym_env.close()
