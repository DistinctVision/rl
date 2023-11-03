import  typing as tp
from pathlib import Path
import math
import yaml
from collections import deque

import numpy as np
import torch

import plotly.express as px

from rlgym.envs import Match
from rlgym.utils.terminal_conditions.common_conditions import TimeoutCondition, GoalScoredCondition
from rlgym_tools.sb3_utils import SB3MultipleInstanceEnv

from ppocket_rocket.game_data import ModelDataProvider
from ppocket_rocket.training.ppo_trainer import PpoTrainer
from ppocket_rocket.training.rollout import RolloutBuffer, RolloutDataset
from ppocket_rocket.training import GymActionParser, GymObsBuilder, GeneralReward
from ppocket_rocket.training.state_setter import GeneralStateSetter


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


def save_histograms(save_folder_path: tp.Union[Path, str], mem_rewards_values: tp.Dict[str, deque], prep_name: str):
    save_folder_path = Path(save_folder_path)
    reward_names: tp.List[str] = []
    reward_means: tp.List[float] = []
    for reward_name, rewards_values in mem_rewards_values.items():
        if len(rewards_values) == 0:
            continue
        fig = px.histogram(x=rewards_values)
        fig.write_image(save_folder_path / f'{prep_name}_{reward_name}.png')
        reward_names.append(reward_name)
        reward_means.append(sum(rewards_values) / len(rewards_values))
        
    fig = px.bar({'names': reward_names, 'means': reward_means}, x='names', y='means')
    fig.write_image(save_folder_path / f'{prep_name}_bar_mean_rewards.png')
    


def ppo_training(num_of_env_instances: int):
    device = 'cuda'
    
    cfg = yaml.safe_load(open(Path('ppocket_rocket') / 'ppo_cfg.yaml', 'r'))
    rollout_cfg = dict(cfg['rollout'])
    training_cfg = cfg['training']
    batch_size = int(training_cfg['batch_size'])
    sequence_size = int(cfg['model']['sequence_size'])
    
    rollout_max_buffer_size = int(rollout_cfg['max_buffer_size'])
    target_data_size = rollout_max_buffer_size
    rollout_max_buffer_size = math.ceil(rollout_max_buffer_size / num_of_env_instances)
    rollout_cfg['max_buffer_size'] = rollout_max_buffer_size + 1
    
    model_data_provider = ModelDataProvider()
    action_parser = GymActionParser(model_data_provider)
    obs_builder = GymObsBuilder(model_data_provider, orange_mirror=True)
    general_reward = GeneralReward()
    trainer = PpoTrainer(cfg, device)
    actor_critic_policy = trainer.models
    
    blue_mem_reward_values = {reward_name: deque(maxlen=10_000) for reward_name in general_reward.rewards}
    orange_mem_reward_values = {reward_name: deque(maxlen=10_000) for reward_name in general_reward.rewards}
    
    rollout_buffers: tp.List[RolloutBuffer] = []
        
    num_cars = 2

    def get_match():
        return Match(
            reward_function=general_reward,
            terminal_conditions=[TimeoutCondition(60 * 10), GoalScoredCondition()],
            obs_builder=obs_builder,
            state_setter=GeneralStateSetter(dict(cfg['replays'])),
            action_parser=action_parser,
            game_speed=100, tick_skip=12, spawn_opponents=True, team_size=1
        )

    env = SB3MultipleInstanceEnv(match_func_or_matches=get_match,
                                 num_instances=num_of_env_instances, wait_time=30)

    last_rewards = deque(maxlen=50)
    ep_counter = 0
    
    while True:
        cur_rollout_buffers = [RolloutBuffer(rollout_cfg, actor_critic_policy.value_net, sequence_size)
                               for _ in range(num_of_env_instances * num_cars)]
        for cur_rollout_buffer in cur_rollout_buffers:
            cur_rollout_buffer.start()
        
        obs = env.reset()
        obs = fix_data(obs)
        dones = np.array([False for _ in  range(num_cars * num_of_env_instances)], dtype=bool)
        
        steps = 0
        ep_rewards = np.zeros((num_of_env_instances * num_cars), dtype=float)
        
        while not dones.all():
            action_dists = [actor_critic_policy.get_action_dist(cur_rollout_buffer.new_state(obs_tensor))
                            for cur_rollout_buffer, obs_tensor in zip(cur_rollout_buffers, obs)]
            actions = [int(action_dict.sample()) for action_dict in action_dists]
            env.step_async(actions)
            next_obs, splitted_rewards, next_dones, gameinfo = env.step_wait()
            next_obs = fix_data(next_obs)
            
            rewards = []
            for env_idx, env_splitted_rewards in enumerate(splitted_rewards):
                mem_reward_values = blue_mem_reward_values if env_idx % 2 == 0 else orange_mem_reward_values
                reward = 0.0
                for reward_name, reward_value in env_splitted_rewards.items():
                    reward += reward_value
                    mem_reward_values[reward_name].append(reward_value)
                rewards.append(reward)

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
            dataset, rollout_buffers = RolloutDataset.collect_data(target_data_size, batch_size, sequence_size,
                                                                   rollout_buffers)
            trainer.train(dataset)
            data_size = sum([len(rollout_buffer) for rollout_buffer in rollout_buffers])
            
        if ep_counter % int(training_cfg['save']['save_hist_every_n_step']) == 0:
            save_histograms(trainer.log_writer.output_plot_folder, blue_mem_reward_values, 'blue')
            save_histograms(trainer.log_writer.output_plot_folder, orange_mem_reward_values, 'orange')
    
    
if __name__ == '__main__':
    from argparse import ArgumentParser
    
    args_parser = ArgumentParser()
    args_parser.add_argument('-n', '--num_instances', type=int, default=1)
    args = args_parser.parse_args()
    
    ppo_training(args.num_instances)
