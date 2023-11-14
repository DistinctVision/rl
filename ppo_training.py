import  typing as tp
from pathlib import Path
import math
import random
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
from Nexto.ext_nexto_obs_builder import ExtNextoObsBuilder, ExtNextoObsData
from Nexto.agent import Agent as NextoAgent


def fix_data(data) -> tp.List[tp.Union[torch.Tensor, ExtNextoObsData]]:
    """
    This is a workaround to fix the bug inside sb3. Just reformat output data.
    """
    
    if isinstance(data, (torch.Tensor, ExtNextoObsData)):
        return [data]
    out  = []
    for d in data:
        if isinstance(d, (torch.Tensor, ExtNextoObsData)):
            out.append(d)
            continue
        if len(d) == 0:
            continue
        assert isinstance(d,  list)
        for e in d:
            assert isinstance(e, (torch.Tensor, ExtNextoObsData))
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
    


def ppo_training(num_of_env_instances: int, tag: str):
    device = 'cuda'
    
    cfg = yaml.safe_load(open(Path('ppocket_rocket') / 'ppo_cfg.yaml', 'r'))
    rollout_cfg = dict(cfg['rollout'])
    training_cfg = cfg['training']
    batch_size = int(training_cfg['batch_size'])
    sequence_size = int(cfg['model']['sequence_size'])
    discount_factor = float(cfg['rollout']['discount_factor'])
    
    rollout_max_buffer_size = int(rollout_cfg['max_buffer_size'])
    target_data_size = rollout_max_buffer_size
    rollout_max_buffer_size = math.ceil(rollout_max_buffer_size / num_of_env_instances)
    rollout_cfg['max_buffer_size'] = rollout_max_buffer_size + 1
    
    model_data_provider = ModelDataProvider()
    action_parser = GymActionParser(model_data_provider)
    general_reward = GeneralReward(discount_factor=discount_factor)
    trainer = PpoTrainer(cfg, device, tag)
    actor_critic_policy = trainer.models
    
    # obs_builder = GymObsBuilder(model_data_provider, orange_mirror=True)
    ext_obs_builder = ExtNextoObsBuilder(model_data_provider, orange_mirror=True)
    nexto_agent = NextoAgent(result_as_index=True)
    
    blue_mem_reward_values = {reward_name: deque(maxlen=10_000) for reward_name in general_reward.rewards}
    orange_mem_reward_values = {reward_name: deque(maxlen=10_000) for reward_name in general_reward.rewards}
    
    rollout_buffers: tp.List[RolloutBuffer] = []
        
    num_cars = 2

    def get_match():
        return Match(
            reward_function=general_reward,
            terminal_conditions=[TimeoutCondition(90 * 12), GoalScoredCondition()],
            obs_builder=ext_obs_builder,
            state_setter=GeneralStateSetter(dict(cfg['replays'])),
            action_parser=action_parser,
            game_speed=100, tick_skip=8, spawn_opponents=True, team_size=1
        )

    env = SB3MultipleInstanceEnv(match_func_or_matches=get_match,
                                 num_instances=num_of_env_instances, wait_time=30)

    last_rewards = deque(maxlen=200)
    ep_counter = 0
    
    while True:
        cur_rollout_buffers: tp.List[tp.Optional[RolloutBuffer]] = []
        
        for _ in range(num_of_env_instances):
            # has_nexto = np.random.choice([False, False, True])
            # if has_nexto:
            #     nexto_team = np.random.choice([0, 1])
            #     if nexto_team == 0:
            #         cur_rollout_buffers.append(None)
            #         cur_rollout_buffers.append(RolloutBuffer(rollout_cfg, actor_critic_policy.value_net,
            #                                                  sequence_size))
            #     else:
            #         cur_rollout_buffers.append(RolloutBuffer(rollout_cfg, actor_critic_policy.value_net,
            #                                                  sequence_size))
            #         cur_rollout_buffers.append(None)
            # else:
            cur_rollout_buffers.append(RolloutBuffer(rollout_cfg, actor_critic_policy.value_net, sequence_size))
            cur_rollout_buffers.append(RolloutBuffer(rollout_cfg, actor_critic_policy.value_net, sequence_size))
                
        nexto_betas: tp.List[tp.Optional[float]] = []
        for cur_rollout_buffer in cur_rollout_buffers:
            if cur_rollout_buffer is not None:
                cur_rollout_buffer.start()
                nexto_betas.append(None)
            else:
                nexto_betas.append(random.uniform(0, 1))
        
        obs = env.reset()
        obs = fix_data(obs)
        
        nexto_obs = [c_obs.nexto_obs for c_obs in obs]
        obs = [c_obs.gym_obs for c_obs in obs]
        
        dones = np.array([False for _ in  range(num_cars * num_of_env_instances)], dtype=bool)
        
        steps = 0
        ep_rewards = [0 if cur_rollout_buffer is not None else None for cur_rollout_buffer in cur_rollout_buffers]
        
        while not dones.all():
            action_dists: tp.List[torch.distributions.Categorical] = []
            actions: tp.List[int] = []
            for cur_rollout_buffer, obs_tensor, nexto_state, nexto_beta in \
                            zip(cur_rollout_buffers, obs, nexto_obs, nexto_betas):
                if cur_rollout_buffer is not None:
                    action_dist = actor_critic_policy.get_action_dist(cur_rollout_buffer.new_state(obs_tensor))
                    action = int(action_dist.sample())
                    action_dists.append(action_dist)
                    actions.append(action)
                else:
                    action = nexto_agent.act(nexto_state, nexto_beta)
                    action_dists.append(None)
                    actions.append(action)
                    
            env.step_async(actions)
            next_obs, splitted_rewards, next_dones, gameinfo = env.step_wait()
            next_obs = fix_data(next_obs)
        
            next_nexto_obs = [c_obs.nexto_obs for c_obs in next_obs]
            next_obs = [c_obs.gym_obs for c_obs in next_obs]
            
            rewards = []
            for env_idx, (env_splitted_rewards, cur_rollout_buffer) in \
                            enumerate(zip(splitted_rewards, cur_rollout_buffers)):
                if cur_rollout_buffer is None:
                    rewards.append(None)
                    continue
                mem_reward_values = blue_mem_reward_values if env_idx % 2 == 0 else orange_mem_reward_values
                reward = 0.0
                for reward_name, reward_value in env_splitted_rewards.items():
                    reward += reward_value
                    mem_reward_values[reward_name].append(reward_value)
                rewards.append(reward)

            for car_idx in range(num_of_env_instances * num_cars):
                cur_rollout_buffer = cur_rollout_buffers[car_idx]
                if dones[car_idx] or cur_rollout_buffer is None:
                    continue
                action_log_prob = float(action_dists[car_idx].log_prob(torch.tensor(actions[car_idx])))
                cur_rollout_buffer.add(obs[car_idx], actions[car_idx], action_log_prob, rewards[car_idx])
            
            if steps >= rollout_max_buffer_size:
                next_dones_ = []
                for next_done, state in zip(next_dones, gameinfo):
                    if not next_done:
                        state['TimeLimit.truncated'] = True
                    next_dones_.append(True)
                next_dones = np.array(next_dones_, dtype=bool)
            
            for car_idx in range(num_of_env_instances * num_cars):
                cur_rollout_buffer = cur_rollout_buffers[car_idx]
                if not next_dones[car_idx] or cur_rollout_buffer is None:
                    continue
                state = gameinfo[car_idx]
                cur_rollout_buffer.finish(next_obs[car_idx], truncated=state['TimeLimit.truncated'])
            
            for idx, reward in enumerate(rewards):
                if reward is None:
                    continue
                ep_rewards[idx] += reward
            obs = next_obs
            nexto_obs = next_nexto_obs
            dones = np.logical_or(dones, next_dones)
            steps += 1
        
        ep_rewards = [reward for reward in ep_rewards if reward is not None]
        for reward in ep_rewards:
            last_rewards.append(reward)
        last_mean_reward = sum(last_rewards) / len(last_rewards)
        
        hist_ext_values = {}
        for metric_name, metric_values in orange_mem_reward_values.items():
            mean_value = sum(metric_values) / max(len(metric_values), 1)
            hist_ext_values[f'orange_{metric_name}'] = mean_value
        for metric_name, metric_values in blue_mem_reward_values.items():
            mean_value = sum(metric_values) / max(len(metric_values), 1)
            hist_ext_values[f'blue_{metric_name}'] = mean_value
    
        trainer.set_ext_values(mean_reward=last_mean_reward, **hist_ext_values)
        
        ep_counter += 1
        
        cur_rollout_buffers = [cur_rollout_buffer for cur_rollout_buffer in cur_rollout_buffers
                               if cur_rollout_buffer is not None]
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
            save_histograms(trainer.log_writer.output_plot_folder, blue_mem_reward_values, 'hist_blue')
            save_histograms(trainer.log_writer.output_plot_folder, orange_mem_reward_values, 'hist_orange')
    
    
if __name__ == '__main__':
    from argparse import ArgumentParser
    
    args_parser = ArgumentParser()
    args_parser.add_argument('-n', '--num_instances', type=int, default=1)
    args_parser.add_argument('--tag', type=str, default='%dt')
    args = args_parser.parse_args()
    
    ppo_training(args.num_instances, args.tag)
