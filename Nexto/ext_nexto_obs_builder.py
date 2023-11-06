import typing as tp

from dataclasses import dataclass

import numpy as np
import torch

from rlgym_compat.game_state import GameState, PlayerData
from rlgym.utils.obs_builders import ObsBuilder
from Nexto.nexto_obs import NextoObsBuilder
from ppocket_rocket.game_data import ModelDataProvider
from ppocket_rocket.training import GymObsBuilder


@dataclass
class ExtNextoObsData:
    gym_obs: torch.Tensor
    nexto_obs: tp.Any


class ExtNextoObsBuilder(ObsBuilder):
    
    def __init__(self, model_data_provider: ModelDataProvider(),
                 orange_mirror: bool = True,
                 double_mirror: bool = False):
        super().__init__()
        self.next_obs_builder = NextoObsBuilder()
        self.gym_obs_builder = GymObsBuilder(model_data_provider=model_data_provider,
                                             orange_mirror=orange_mirror, double_mirror=double_mirror)

    def reset(self, initial_state: GameState):
        self.next_obs_builder.reset(initial_state)
        self.gym_obs_builder.reset(initial_state)

    def build_obs(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> ExtNextoObsData:
        gym_obs = self.gym_obs_builder.build_obs(player, state, previous_action)
        next_obs = self.next_obs_builder.build_obs(player, state, previous_action)
        return ExtNextoObsData(gym_obs, next_obs)
