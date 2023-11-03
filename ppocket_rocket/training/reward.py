import typing as tp
from pathlib import Path

import math

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


class DistToBallReward(RewardFunction):
    
    def __init__(self, scale: float = 1.0 / BALL_RADIUS):
        super().__init__()
        self.scale = scale
        self.prev_distances: tp.Dict[int, float] = {}
        self.cur_distances: tp.Dict[int, float] = {}
        
    def reset(self, initial_state: GameState):
        self.prev_distances = {}
        self.cur_distances = {}
        for player in initial_state.players:
            dis = np.linalg.norm(player.car_data.position - initial_state.ball.position)
            self.cur_distances[player.car_id] = dis * self.scale

    def pre_step(self, state: GameState):
        self.prev_distances = self.cur_distances
        self.cur_distances = {}
        for player in state.players:
            dis = np.linalg.norm(player.car_data.position - state.ball.position)
            self.cur_distances[player.car_id] = dis * self.scale
        
    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        enemy = None
        for car in state.players:
            if car.car_id != player.car_id:
                enemy = car
                break
        prev_delta_dis = self.prev_distances[enemy.car_id] - self.prev_distances[player.car_id]
        cur_delta_dis = self.cur_distances[enemy.car_id] - self.cur_distances[player.car_id]
        
        time_delta_dis = cur_delta_dis - prev_delta_dis
        return signed_sqrt(time_delta_dis)
    
    
class DistBallToGoalReward(RewardFunction):
    ORANGE_GOAL = (np.array(ORANGE_GOAL_BACK) + np.array(ORANGE_GOAL_CENTER)) / 2
    BLUE_GOAL = (np.array(BLUE_GOAL_BACK) + np.array(BLUE_GOAL_CENTER)) / 2
    
    def __init__(self, scale: float = 10.0 / BALL_RADIUS):
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
        delta_blue_dis = (self.cur_dis_to_blue_goal - self.prev_dis_to_blue_goal)
        delta_orange_dis = (self.cur_dis_to_orange_goal - self.prev_dis_to_orange_goal)
        if player.team_num == BLUE_TEAM:
            return signed_sqrt(delta_blue_dis - delta_orange_dis)
        return signed_sqrt(delta_orange_dis - delta_blue_dis)


class AlignBallGoal(RewardFunction):
    def __init__(self, defense=10.0, offense=10.0):
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
    
    def __init__(self, scale: float = 0.1):
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
    
    
class GoalReward(RewardFunction):
    
    def __init__(self, value: float = 1000.0):
        self.value = value
        
    def reset(self, initial_state: GameState):
        self.blue_score = initial_state.blue_score
        self.orange_score = initial_state.orange_score

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        if state.blue_score > self.blue_score:
            self.blue_score = state.blue_score
            if player.team_num == BLUE_TEAM:
                return self.value
            return - self.value
        if state.orange_score > self.orange_score:
            self.orange_score = state.orange_score
            if player.team_num == BLUE_TEAM:
                return - self.value
            return self.value
        return 0
    

class GeneralReward(RewardFunction):
    
    def __init__(self):
        super().__init__()
        self.rewards: tp.Dict[str, RewardFunction] = {
            'dist_to_ball': DistToBallReward(),
            'dist_ball_to_goal': DistBallToGoalReward(),
            'align_ball': AlignBallGoal(),
            'save_boost': SaveBoostReward(),
            'touch_ball': TouchBallReward(),
            'goal': GoalReward()
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
