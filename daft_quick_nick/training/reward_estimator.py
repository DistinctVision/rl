import typing as tp

import numpy as np

from rlgym.utils.reward_functions import RewardFunction
from rlgym.utils.gamestates import GameState, PlayerData
from daft_quick_nick.util.constans import CAR_MAX_SPEED, BALL_RADIUS


class RewardEstimator(RewardFunction):
    
    def __init__(self, reward_decay: float) -> None:
        super().__init__()
        self.reward_decay = reward_decay
        
    def reset(self, initial_state: GameState):
        ...

    def pre_step(self, state: GameState):
        ...

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        
        dist = np.linalg.norm(player.car_data.position - state.ball.position) - BALL_RADIUS
        liu_dist = np.exp(-0.5 * dist / CAR_MAX_SPEED)
        
        return liu_dist

    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        return self.get_reward(player, state, previous_action)
        