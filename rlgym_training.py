import  typing as tp
from pathlib import Path
import yaml
from collections import deque

import numpy as np
import torch

from rlgym.envs import Match
from rlgym.utils.reward_functions import DefaultReward
from rlgym.utils.obs_builders import DefaultObs
from rlgym.utils.state_setters import DefaultState
from rlgym.utils.action_parsers import DefaultAction
from rlgym.utils.terminal_conditions.common_conditions import TimeoutCondition, GoalScoredCondition
from rlgym.gamelaunch import LaunchPreference
from rlgym_tools.sb3_utils import SB3MultipleInstanceEnv

from daft_quick_nick.game_data import ModelDataProvider
from daft_quick_nick.training import Trainer, ReplayBuffer, EpisodeDataRecorder
from daft_quick_nick.training import GymActionParser, GymObsBuilder, Trainer, RewardEstimator
from daft_quick_nick.training import RandomBallGameState


def fix_data(data) -> tp.List[torch.Tensor]:
    """
    This is a workaround to fix the bug inside sb3. Just reformat output data.
    """
    
    if isinstance(data, torch.Tensor):
        return [data]
    out  = []
    for d in data:
        if len(d) == 0:
            continue
        if isinstance(d, torch.Tensor):
            out.append(d)
            continue
        assert isinstance(d,  list)
        for e in d:
            assert isinstance(e,  torch.Tensor)
            out.append(e)
    return out


def rlgym_training(num_instances: int):
    cfg = yaml.safe_load(open(Path('daft_quick_nick') / 'cfg.yaml', 'r'))
    replay_buffer_cfg = dict(cfg['replay_buffer'])
    min_rp_data_size = int(replay_buffer_cfg['min_buffer_size'])
    
    model_data_provider = ModelDataProvider()
    action_parser = GymActionParser(model_data_provider)
    obs_builder = GymObsBuilder(model_data_provider)
    reward_estimator = RewardEstimator(float(cfg['model']['reward_decay']))
    replay_buffer = ReplayBuffer()
    trainer = Trainer(cfg, replay_buffer)
        
    num_cars = 2
    
    ep_data_recorders = [EpisodeDataRecorder(trainer) for _ in range(num_cars * num_instances)]

    def get_match():
        return Match(
            reward_function=reward_estimator,
            terminal_conditions=[TimeoutCondition(30 * 10), GoalScoredCondition()],
            obs_builder=obs_builder,
            state_setter=RandomBallGameState(),
            action_parser=action_parser,
            game_speed=100, tick_skip=12, spawn_opponents=True, team_size=1
        )

    env = SB3MultipleInstanceEnv(match_func_or_matches=get_match,
                                 num_instances=num_instances, wait_time=20)

    last_rewards = deque(maxlen=100)
    ep_counter = 0
    default_action_index = model_data_provider.default_action_index
    
    train_freq = int(cfg['training']['train_freq'])
    data_counter = 0
    
    while True:
        obs = env.reset()
        obs = fix_data(obs)
        done = np.array([False for _ in  range(num_cars * num_instances)], dtype=bool)
        
        steps = 0
        ep_rewards = np.zeros((num_instances * num_cars), dtype=float)
        
        while not done.all():
            actions = [ep_data_recorder.get_action(car_obs) if not car_done else default_action_index
                       for ep_data_recorder, car_obs, car_done in zip(ep_data_recorders, obs, done)]
            env.step_async(actions)
            
            if len(replay_buffer) > min_rp_data_size:
                data_counter += num_instances * num_cars
                if data_counter >= train_freq:
                    trainer.train_step()
                    data_counter = data_counter % train_freq
            
            next_obs, rewards, next_done, gameinfo = env.step_wait()
            next_obs = fix_data(next_obs)

            for car_idx in range(num_instances * num_cars):
                if done[car_idx]:
                    continue
                ep_data_recorders[car_idx].record(obs[car_idx], actions[car_idx], 
                                                  rewards[car_idx], next_done[car_idx])
            
            ep_rewards += rewards
            obs = next_obs
            done = np.logical_or(done, next_done)
            steps += 1
        
        for reward in ep_rewards:
            last_rewards.append(reward)
        last_mean_reward = sum(last_rewards) / len(last_rewards)
        trainer.add_metric_value('reward', last_mean_reward)
        
        ep_counter += 1
        
        rewards_str = ', '.join([f"{float(reward):.2f}" for reward in ep_rewards])

        print(f'Episode: {ep_counter} | Replay buffer size: {len(replay_buffer)} | Mean rewards: {last_mean_reward:.2f} '\
              f'| Episode Rewards: {rewards_str}')
    
    
if __name__ == '__main__':
    from argparse import ArgumentParser
    
    args_parser = ArgumentParser()
    args_parser.add_argument('-n', '--num_instances', type=int, default=1)
    args = args_parser.parse_args()
    
    rlgym_training(args.num_instances)
