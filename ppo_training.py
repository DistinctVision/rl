import  typing as tp
from pathlib import Path
import math
import yaml
from collections import deque

import numpy as np
import torch

from rlgym.envs import Match
from rlgym.utils.terminal_conditions.common_conditions import TimeoutCondition, GoalScoredCondition
from rlgym_tools.sb3_utils import SB3MultipleInstanceEnv

from daft_quick_nick.game_data import ModelDataProvider
from daft_quick_nick.training.ppo_trainer import PpoTrainer
from daft_quick_nick.training.rollout import RolloutBuffer, RolloutDataset
from daft_quick_nick.actor_critic_policy import ActorCriticPolicy
from daft_quick_nick.training import GymActionParser, GymObsBuilder, RewardEstimator
from daft_quick_nick.training.state_setter import NectoStateSetter


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


def ppo_training(num_of_env_instances: int):
    device = 'cuda'
    
    cfg = yaml.safe_load(open(Path('daft_quick_nick') / 'ppo_cfg.yaml', 'r'))
    rollout_cfg = dict(cfg['rollout'])
    batch_size = int(cfg['training']['batch_size'])
    rollout_max_buffer_size = int(rollout_cfg['max_buffer_size'])
    target_data_size = rollout_max_buffer_size
    rollout_max_buffer_size = math.ceil(rollout_max_buffer_size / num_of_env_instances)
    rollout_cfg['max_buffer_size'] = rollout_max_buffer_size + 1
    
    model_data_provider = ModelDataProvider()
    action_parser = GymActionParser(model_data_provider)
    obs_builder = GymObsBuilder(model_data_provider, use_mirror=False)
    reward_estimator = RewardEstimator()
    trainer = PpoTrainer(cfg, device)
    actor_critic_policy = trainer.models
    
    rollout_buffers: tp.List[RolloutBuffer] = []
        
    num_cars = 2

    def get_match():
        return Match(
            reward_function=reward_estimator,
            terminal_conditions=[TimeoutCondition(60 * 10), GoalScoredCondition()],
            obs_builder=obs_builder,
            state_setter=NectoStateSetter(),
            action_parser=action_parser,
            game_speed=100, tick_skip=12, spawn_opponents=True, team_size=1
        )

    env = SB3MultipleInstanceEnv(match_func_or_matches=get_match,
                                 num_instances=num_of_env_instances, wait_time=20)

    last_rewards = deque(maxlen=50)
    ep_counter = 0
    
    while True:
        cur_rollout_buffers = [RolloutBuffer(rollout_cfg, actor_critic_policy.value_net)
                               for _ in range(num_of_env_instances * num_cars)]
        for cur_rollout_buffer in cur_rollout_buffers:
            cur_rollout_buffer.start()
        
        obs = env.reset()
        obs = fix_data(obs)
        dones = np.array([False for _ in  range(num_cars * num_of_env_instances)], dtype=bool)
        
        steps = 0
        ep_rewards = np.zeros((num_of_env_instances * num_cars), dtype=float)
        
        while not dones.all():
            action_dists = [actor_critic_policy.get_action_dist(obs_tensor)
                            for obs_tensor in obs]
            actions = [int(action_dict.sample()) for action_dict in action_dists]
            env.step_async(actions)
            next_obs, rewards, next_dones, gameinfo = env.step_wait()
            next_obs = fix_data(next_obs)

            for car_idx in range(num_of_env_instances * num_cars):
                if dones[car_idx]:
                    continue
                action_log_prob = float(action_dists[car_idx].log_prob(torch.tensor(actions[car_idx])))
                cur_rollout_buffers[car_idx].add(obs[car_idx],
                                                 actions[car_idx], action_log_prob,
                                                 rewards[car_idx])
            
            if steps >= rollout_max_buffer_size:
                next_dones_ = []
                for next_done, state in zip(next_dones, gameinfo):
                    if not next_done:
                        state['TimeLimit.truncated'] = True
                    next_dones_.append(True)
                next_dones = np.array(next_dones_, dtype=bool)
            
            for car_idx in range(num_of_env_instances * num_cars):
                if not next_dones[car_idx]:
                    continue
                state = gameinfo[car_idx]
                cur_rollout_buffers[car_idx].finish(next_obs[car_idx], truncated=state['TimeLimit.truncated'])
            
            ep_rewards += rewards
            obs = next_obs
            dones = np.logical_or(dones, next_dones)
            steps += 1
        
        for reward in ep_rewards:
            last_rewards.append(reward)
        last_mean_reward = sum(last_rewards) / len(last_rewards)
    
        trainer.set_ext_values(mean_reward=last_mean_reward)
        
        ep_counter += 1
        
        assert all([cur_rollout_buffer.is_finished for cur_rollout_buffer in cur_rollout_buffers])
        rollout_buffers += cur_rollout_buffers
        
        data_size = sum([len(rollout_buffer) for rollout_buffer in rollout_buffers])
        ep_rewards_str = ', '.join([f'{reward:.2f}' for reward in ep_rewards])
        print(f'Episode: {ep_counter} | Rollout buffer size: {data_size} | Mean rewards: {last_mean_reward:.2f} |'\
            f'Episode Rewards: {ep_rewards_str}')
        
        while data_size >= target_data_size:
            dataset, rollout_buffers = RolloutDataset.collect_data(target_data_size, batch_size, rollout_buffers)
            trainer.train(dataset)
            data_size = sum([len(rollout_buffer) for rollout_buffer in rollout_buffers])
    
    
if __name__ == '__main__':
    from argparse import ArgumentParser
    
    args_parser = ArgumentParser()
    args_parser.add_argument('-n', '--num_instances', type=int, default=1)
    args = args_parser.parse_args()
    
    ppo_training(args.num_instances)
