import typing as tp
from pathlib import Path

import math
from copy import deepcopy

import numpy as np

from rlgym.utils import math as rl_math

from rlgym.utils.reward_functions import RewardFunction
from rlgym.utils.gamestates import GameState, PlayerData
from rlgym.utils.common_values import CEILING_Z, BLUE_GOAL_BACK, \
    BLUE_GOAL_CENTER, ORANGE_GOAL_BACK, ORANGE_GOAL_CENTER, BALL_RADIUS, \
        BLUE_TEAM, ORANGE_TEAM


def signed_sqrt(x: float) -> float:
    if x < 0.0:
        return - math.sqrt(- x)
    return math.sqrt(x)


class ClosestToBallReward(RewardFunction):
    
    def __init__(self, value: float = 1.0):
        super().__init__()
        self.value = value
        self.cur_distances: tp.Dict[int, float] = {}
        
    def reset(self, initial_state: GameState):
        self.cur_distances = {}
        for player in initial_state.players:
            dis = np.linalg.norm(player.car_data.position - initial_state.ball.position)
            self.cur_distances[player.car_id] = dis

    def pre_step(self, state: GameState):
        self.prev_distances = self.cur_distances
        self.cur_distances = {}
        for player in state.players:
            dis = np.linalg.norm(player.car_data.position - state.ball.position)
            self.cur_distances[player.car_id] = dis
        
    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        enemy = None
        for car in state.players:
            if car.car_id != player.car_id:
                enemy = car
                break
        if self.cur_distances[player.car_id] < self.cur_distances[enemy.car_id]:
            return self.value
        return -self.value
    
    
class LastTouchReward(RewardFunction):
    
    def __init__(self, value: float = 1.0):
        super().__init__()
        self.value = value
        self.last_touch_car_id: tp.Optional[int] = None
        
    def reset(self, initial_state: GameState):
        self.last_touch_car_id: tp.Optional[int] = None

    def pre_step(self, state: GameState):
        for player in state.players:
            if player.ball_touched:
                self.last_touch_car_id = player.car_id
                break
        
    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        if player.car_id == self.last_touch_car_id:
            return self.value
        return 0
    
    
class DistBallToGoalReward(RewardFunction):
    ORANGE_GOAL = (np.array(ORANGE_GOAL_BACK) + np.array(ORANGE_GOAL_CENTER)) / 2
    BLUE_GOAL = (np.array(BLUE_GOAL_BACK) + np.array(BLUE_GOAL_CENTER)) / 2
    
    def __init__(self, scale: float = 1e3 / BALL_RADIUS):
        super().__init__()
        self.scale = scale
        self.prev_dis_to_orange_goal = 0.0
        self.prev_dis_to_blue_goal = 0.0
        self.cur_dis_to_orange_goal = 0.0
        self.cur_dis_to_blue_goal = 0.0

    def reset(self, initial_state: GameState):
        self.prev_dis_to_orange_goal = 0.0
        self.prev_dis_to_blue_goal = 0.0
        self.cur_dis_to_orange_goal = np.linalg.norm(initial_state.ball.position - self.ORANGE_GOAL) * self.scale
        self.cur_dis_to_blue_goal = np.linalg.norm(initial_state.ball.position - self.BLUE_GOAL) * self.scale

    def pre_step(self, state: GameState):
        self.prev_dis_to_orange_goal = self.cur_dis_to_orange_goal
        self.prev_dis_to_blue_goal = self.cur_dis_to_blue_goal
        self.cur_dis_to_orange_goal = np.linalg.norm(state.ball.position - self.ORANGE_GOAL) * self.scale
        self.cur_dis_to_blue_goal = np.linalg.norm(state.ball.position - self.BLUE_GOAL) * self.scale
        
    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        delta_blue_dis = (self.cur_dis_to_blue_goal - self.prev_dis_to_blue_goal) / max(self.cur_dis_to_blue_goal, 1)
        delta_orange_dis = (self.cur_dis_to_orange_goal - self.prev_dis_to_orange_goal) / max(self.cur_dis_to_orange_goal, 1)
        if player.team_num == BLUE_TEAM:
            return (delta_blue_dis - delta_orange_dis)
        return (delta_orange_dis - delta_blue_dis)


class AlignBallGoal(RewardFunction):
    def __init__(self, defense=0.01, offense=0.01):
        super().__init__()
        self.defense = defense
        self.offense = offense

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        ball = state.ball.position
        pos = player.car_data.position
        protecc = np.array(BLUE_GOAL_BACK)
        attacc = np.array(ORANGE_GOAL_BACK)
        if player.team_num == ORANGE_TEAM:
            protecc, attacc = attacc, protecc
            
        delta_ball_pos = ball - pos

        # Align player->ball and net->player vectors
        defensive_reward = self.defense * rl_math.cosine_similarity(delta_ball_pos, pos - protecc)

        # Align player->ball and player->net vectors
        offensive_reward = self.offense * rl_math.cosine_similarity(delta_ball_pos, attacc - pos)

        return defensive_reward + offensive_reward
    
    
class SaveBoostReward(RewardFunction):
    
    def __init__(self, scale: float = 1e-3):
        self.scale = scale
    
    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        # 1 reward for each frame with 100 boost, sqrt because 0->20 makes bigger difference than 80->100
        return np.sqrt(player.boost_amount) * self.scale
    
    
class TouchBallReward(RewardFunction):
    
    def __init__(self, value: float = 10.0):
        self.value = value
        
    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        if not player.ball_touched:
            return 0.0
        height_weight = 1 + (state.ball.position[2] - BALL_RADIUS) / CEILING_Z
        return self.value * height_weight
    
    
class DemolutionReward(RewardFunction):
    
    def __init__(self, value: float = 1e2):
        self.value = value
        self.prev = {
            'orange': False,
            'blue': False
        }
        self.cur = deepcopy(self.prev)
        
    def reset(self, initial_state: GameState):
        self.prev = {
            'orange': False,
            'blue': False
        }
        self.cur = deepcopy(self.prev)

    def pre_step(self, state: GameState):
        self.prev = deepcopy(self.cur)
        for player in state.players:
            team_name = 'orange' if player.team_num == ORANGE_TEAM else 'blue'
            self.cur[team_name] = player.is_demoed

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        score = 0
        if player.team_num == ORANGE_TEAM:
            player_team = 'orange'
            enemy_team = 'blue'
        else:
            player_team = 'blue'
            enemy_team = 'orange'
        if self.cur[player_team] and self.cur[player_team] != self.prev[player_team]:
            score -= self.value
        if self.cur[enemy_team] and self.cur[enemy_team] != self.prev[enemy_team]:
            score += self.value
            
        return score
            
    
    
class GoalReward(RewardFunction):
    
    def __init__(self, discount_factor: float, penalty: float = 1.0, goal_reward: float = 10.0):
        self.discount_factor = discount_factor
        self.penalty = penalty
        self.goal_reward = goal_reward
        self.prev_scores = {
            'orange': 0,
            'blue': 0
        }
        self.cur_scores = deepcopy(self.prev_scores)
        self.prev_rewards = {
            'orange': 0,
            'blue': 0
        }
        self.cur_rewards = deepcopy(self.prev_rewards)
        
    def reset(self, initial_state: GameState):
        self.prev_scores = {
            'orange': initial_state.orange_score,
            'blue': initial_state.blue_score
        }
        self.cur_scores = deepcopy(self.prev_scores)
        self.prev_rewards = {
            'orange': 0,
            'blue': 0
        }
        self.cur_rewards = deepcopy(self.prev_rewards)

    def pre_step(self, state: GameState):
        self.prev_scores = deepcopy(self.cur_scores)
        self.cur_scores = {
            'orange': state.orange_score,
            'blue': state.blue_score
        }
        self.prev_rewards = deepcopy(self.cur_rewards)
        self.cur_rewards = {
            team_name: reward * self.discount_factor
            for team_name, reward in self.prev_rewards.items()
        }
        
    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        (player_team_name, enemy_team_name) = ('blue', 'orange') if player.team_num == BLUE_TEAM else ('orange', 'blue')
        if self.cur_scores[player_team_name] > self.prev_scores[player_team_name]:
            return self.goal_reward - self.cur_rewards[player_team_name] * 2
        elif self.cur_scores[enemy_team_name] > self.prev_scores[enemy_team_name]:
            return self.cur_rewards[player_team_name] * 2 - self.goal_reward
        self.cur_rewards[player_team_name] -= self.penalty
        return - self.penalty
                
    
class ConstantReward(RewardFunction):
    def __init__(self, value: float = 1e-2):
        super().__init__()
        self.value = value
    
    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        return - self.value
    

class GeneralReward(RewardFunction):
    
    def __init__(self, discount_factor: float):
        super().__init__()
        self.rewards: tp.Dict[str, RewardFunction] = {
            'closest_to_ball': ClosestToBallReward(),
            'last_touch': LastTouchReward(),
            'goal_reward': GoalReward(discount_factor=discount_factor)
        }
        
    def reset(self, initial_state: GameState):
        for reward in self.rewards.values():
            reward.reset(initial_state)

    def pre_step(self, state: GameState):
        for reward in self.rewards.values():
            reward.pre_step(state)

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> tp.Dict[str, float]:
        rewards = {reward_name: reward_estimator.get_reward(player, state, previous_action)
                   for reward_name, reward_estimator in self.rewards.items()}
        return rewards
