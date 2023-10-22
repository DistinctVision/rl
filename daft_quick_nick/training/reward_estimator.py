import typing as tp

import numpy as np

from rlgym.utils.reward_functions import RewardFunction
from rlgym.utils.gamestates import GameState, PlayerData
from util.constans import CAR_MAX_SPEED, BALL_RADIUS


class RewardEstimator(RewardFunction):
    
    def __init__(self, reward_decay: float) -> None:
        super().__init__()
        self.reward_decay = reward_decay

    def pre_step(self, state: GameState):
        """
        Function to pre-compute values each step. This function is called only once each step, before get_reward is
        called for each player.

        :param state: The current state of the game.
        """
        ...

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        """
        Function to compute the reward for a player. This function is given a player argument, and it is expected that
        the reward returned by this function will be for that player.

        :param player: Player to compute the reward for.
        :param state: The current state of the game.
        :param previous_action: The action taken at the previous environment step.

        :return: A reward for the player provided.
        """
        
        dist = np.linalg.norm(player.car_data.position - state.ball.position) - BALL_RADIUS
        liu_dist = np.exp(-0.5 * dist / CAR_MAX_SPEED)
        
        return liu_dist

    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        """
        Function to compute the reward for a player at the final step of an episode. This will be called only once, when
        it is determined that the current state is a terminal one. This may be useful for sparse reward signals that only
        produce a value at the final step of an environment. By default, the regular get_reward is used.

        :param player: Player to compute the reward for.
        :param state: The current state of the game.
        :param previous_action: The action taken at the previous environment step.

        :return: A reward for the player provided.
        """
        return self.get_reward(player, state, previous_action)
        