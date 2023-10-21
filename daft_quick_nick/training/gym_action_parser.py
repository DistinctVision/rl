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
        return gym.spaces.Box(-1, 1, shape=(common_values.NUM_ACTIONS,))

    def parse_actions(self, actions: tp.List[int], state: GameState) -> np.ndarray:
        action_arr = []
        for action_idx in actions:
            action = self.data_provider.action_lookup_table[action_idx]
            [throttle, pitch, steer_or_yaw, roll, jump, boost, handbrake] = action
            action_data = [boost or throttle, steer_or_yaw, steer_or_yaw, pitch, roll, jump, boost, handbrake]
            action_arr.append(action_data)
        return np.array(action_arr, dtype=np.float32)
