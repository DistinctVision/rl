import typing as tp

import numpy as np

from rlgym.utils.reward_functions import RewardFunction
from rlgym.utils.gamestates import GameState, PlayerData
from daft_quick_nick.utils.constans import CAR_MAX_SPEED, BALL_RADIUS


class RewardEstimator(RewardFunction):
    
    def __init__(self) -> None:
        super().__init__()
        self.prev_distances: tp.Dict[int, float] = {}
        
    def reset(self, initial_state: GameState):
        self.prev_distances.clear()
        for player in initial_state.players:
            dis = np.linalg.norm(player.car_data.position - initial_state.ball.position)
            self.prev_distances[player.car_id] = dis

    def pre_step(self, state: GameState):
        ...

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        
        dis = np.linalg.norm(player.car_data.position - state.ball.position)
        delta_dis = self.prev_distances[player.car_id] - dis
        self.prev_distances[player.car_id] = dis
        
        return delta_dis / BALL_RADIUS

    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        return self.get_reward(player, state, previous_action)
        