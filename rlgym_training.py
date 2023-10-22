import  typing as tp
from pathlib import Path
import yaml
from collections import deque

import numpy as np

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


def rlgym_training(num_instances: int):
    cfg = yaml.safe_load(open(Path('daft_quick_nick') / 'cfg.yaml', 'r'))
    replay_buffer_cfg = dict(cfg['replay_buffer'])
    min_rp_data_size = int(replay_buffer_cfg['min_buffer_size'])

    action_parser = GymActionParser()
    obs_builder = GymObsBuilder()
    reward_estimator = RewardEstimator(float(cfg['model']['reward_decay']))
    replay_buffer = ReplayBuffer()
    trainer = Trainer(cfg, replay_buffer)
    
    ep_data_recorders = [EpisodeDataRecorder(trainer) for _ in range(num_instances)]

    def get_match():
        return Match(
            reward_function=reward_estimator,
            terminal_conditions=[TimeoutCondition(30 * 10), GoalScoredCondition()],
            obs_builder=obs_builder,
            state_setter=DefaultState(),
            action_parser=action_parser,
            game_speed=100, tick_skip=12, spawn_opponents=True, team_size=1
        )

    env = SB3MultipleInstanceEnv(match_func_or_matches=get_match,
                                 num_instances=num_instances, wait_time=20)

    last_rewards = deque(maxlen=100)
    ep_counter = 0

    while True:
        obs = env.reset()
        done = np.array([False for _ in range(num_instances)], dtype=bool)
        
        steps = 0
        ep_rewards = np.zeros((num_instances,), dtype=float)
        
        while not done.all():
            actions = [ep_data_recorders[env_idx].get_action(obs[env_idx]) if not done[env_idx] else None
                       for env_idx in range(num_instances)]
            env.step_async(actions)
            
            if len(replay_buffer) > min_rp_data_size:
                trainer.train_step()
            
            next_obs, rewards, next_done, gameinfo = env.step_wait()

            for env_idx in range(num_instances):
                if done[env_idx]:
                    continue
                ep_data_recorders[env_idx].record(obs[env_idx], actions[env_idx], rewards[env_idx], next_done[env_idx])
            
            ep_rewards += rewards
            obs = next_obs
            done = next_done
            steps += 1
        
        for reward in ep_rewards:
            last_rewards.append(reward)
        last_mean_reward = sum(last_rewards) / len(last_rewards)
        trainer.add_metric_value('reward', last_mean_reward)
        
        ep_counter += 1

        print(f'Episode: {ep_counter} | Replay buffer size: {len(replay_buffer)} | Mean rewards: {last_mean_reward:.2f} '\
              f'| Episode Rewards: {", ".join([f"{reward:.2f}" for reward in last_rewards])}')
    
    
if __name__ == '__main__':
    from argparse import ArgumentParser
    
    args_parser = ArgumentParser()
    args_parser.add_argument('-n', '--num_instances', type=int, default=1)
    args = args_parser.parse_args()
    
    rlgym_training(args.num_instances)
