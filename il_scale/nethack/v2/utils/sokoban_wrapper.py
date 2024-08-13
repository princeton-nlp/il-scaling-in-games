import re

import gym
from nle import nethack
import numpy as np

S_PIT = nethack.GLYPH_CMAP_OFF + 52
S_HOLE = nethack.GLYPH_CMAP_OFF + 54

class Score:
    def __init__(self):
        self.score = 0
        # convert name to snake_case
        # https://stackoverflow.com/questions/1175208/elegant-python-function-to-convert-camelcase-to-snake-case
        self.name = re.sub("(?!^)([A-Z]+)", r"_\1", self.__class__.__name__).lower()

    def reset_score(self):
        self.score = 0

class SokobanfillpitScore(Score):
    """
    This task requires the agent to put the boulders inside wholes for sokoban.
    We count each successful boulder moved into a whole as a total reward.
    """

    def reward(self, env, last_observation, observation, end_status):
        # the score counts how many pits we fill
        char_array = [chr(i) for i in observation[env._message_index]]
        message = "".join(char_array)

        if message.startswith("The boulder fills a pit.") or message.startswith(
            "The boulder falls into and plugs a hole in the floor!"
        ):
            reward = 1
        else:
            reward = 0
        self.score += reward

        return reward


class SokobansolvedlevelsScore(Score):
    def __init__(self):
        super().__init__()
        self.sokoban_levels = {}
        # self.glyph_translator = {str(getattr(SS, s)): s for s in vars(SS)}

    def reward(self, env, last_observation, observation, end_status):
        glyphs = observation[env._glyph_index]
        blstats = observation[env._blstats_index]

        dungeon_num = blstats[nethack.NLE_BL_DNUM]
        dungeon_level = blstats[nethack.NLE_BL_DLEVEL]

        # for debugging, TODO: remove
        # named_glyphs = list(map(lambda x: self.glyph_translator.get(str(x)), glyphs.flatten()))
        # named_glyphs = np.array(named_glyphs).reshape(glyphs.shape)

        # TODO: this isn't error proof, one could create more holes / pits with a wand of digging
        # Note: we can't just check if the agent reached stairs because it can go out of the pits
        # this isn't "solving"
        if (dungeon_num, dungeon_level) == (4, 4):
            pits = np.isin(glyphs, [S_PIT]).sum()
            key = (dungeon_num, dungeon_level)
            self.sokoban_levels[key] = pits
        elif (dungeon_num, dungeon_level) in [(4, 3), (4, 2), (4, 1)]:
            holes = np.isin(glyphs, [S_HOLE]).sum()
            key = (dungeon_num, dungeon_level)
            self.sokoban_levels[key] = holes

        self.score = 0
        for pits in self.sokoban_levels.values():
            # when all pits are filled we assume that sokoban level is solved
            if pits == 0:
                self.score += 1

    def reset_score(self):
        super().reset_score()
        self.sokoban_levels = {}


class SokobanReachedScore(Score):
    def __init__(self):
        super().__init__()
        self.reached_levels = {}
        
    def reward(self, env, last_observation, observation, end_status):
        blstats = observation[env._blstats_index]

        dungeon_num = blstats[nethack.NLE_BL_DNUM]
        dungeon_level = blstats[nethack.NLE_BL_DLEVEL]

        key = (dungeon_num, dungeon_level)
        self.reached_levels[key] = 1

        self.score = 0
        for reached in self.reached_levels.keys():
            if reached in [(4, 4), (4, 3), (4, 2), (4, 1)]:
                self.score += 1
        
    def reset_score(self):
        super().reset_score()
        self.sokoban_levels = {}

class TaskRewardsInfoWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)

        self.tasks = [
            SokobanfillpitScore(),
            SokobansolvedlevelsScore(),
            SokobanReachedScore()
        ]

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)

        for task in self.tasks:
            task.reset_score()

        return obs

    def step(self, action):
        # use tuple and copy to avoid shallow copy (`last_observation` would be the same as `observation`)
        last_observation = tuple(a.copy() for a in self.env.unwrapped.last_observation)
        obs, reward, done, info = self.env.step(action)
        observation = tuple(a.copy() for a in self.env.unwrapped.last_observation)
        end_status = info["end_status"]

        if done:
            info["episode_extra_stats"] = self.add_more_stats(info)

        # we will accumulate rewards for each step and log them when done signal appears
        for task in self.tasks:
            task.reward(self.env.unwrapped, last_observation, observation, end_status)

        return obs, reward, done, info

    def add_more_stats(self, info):
        extra_stats = info.get("episode_extra_stats", {})
        new_extra_stats = {task.name: task.score for task in self.tasks}
        return {**extra_stats, **new_extra_stats}