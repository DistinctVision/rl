import typing as tp

import numpy as np

import gym.spaces
from rlgym.utils.gamestates import GameState
from rlgym.utils import ActionParser
from rlgym.utils import common_values

from daft_quick_nick.game_data import ModelDataProvider


class GymActionParser(ActionParser):

    def __init__(self, data_provider: ModelDataProvider):
        self.data_provider = data_provider

    def get_action_space(self) -> gym.spaces.Space:
        return gym.spaces.Discrete(n=self.data_provider.num_actions)

    def parse_actions(self, actions: tp.List[tp.List[int]], state: GameState) -> np.ndarray:
        action_arr = []
        for action_idx in actions:
            action = self.data_provider.action_lookup_table[action_idx]
            [throttle, steer, pitch, yaw, roll, jump, boost, handbrake] = action
            action_data = [throttle, steer, yaw, pitch, roll, jump, boost, handbrake]
            action_arr.append(action_data)
        return np.array(action_arr, dtype=np.float32)
