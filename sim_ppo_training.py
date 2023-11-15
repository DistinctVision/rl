import  typing as tp
from pathlib import Path
import sys
import threading
import math
import yaml
from collections import deque

import numpy as np
import torch

import plotly.express as px

from rlgym.utils.terminal_conditions.common_conditions import TimeoutCondition, GoalScoredCondition

from rl_sim_tools import rlgym_sim_vec_env, SubprocVecEnv

from ppocket_rocket.game_data import ModelDataProvider
from ppocket_rocket.training.ppo_trainer import PpoTrainer
from ppocket_rocket.training.rollout import RolloutBuffer, RolloutDataset
from ppocket_rocket.training import GymActionParser, GymObsBuilder, GeneralReward
from ppocket_rocket.training.state_setter import GeneralStateSetter
import time

from PyQt5 import QtGui
from PyQt5.QtWidgets import QWidget, QPushButton, QHBoxLayout, QSlider, QApplication
from PyQt5.QtCore import Qt, QRect


class OnOffSwitch(QPushButton):
    def __init__(self, swith_callback: tp.Callable[[int], None], parent=None):
        super().__init__(parent)
        self.setCheckable(True)
        self.setChecked(False)
        self.setMinimumWidth(66)
        self.setMinimumHeight(22)
        self.swith_callback = swith_callback
        self.clicked.connect(self.checkSlot)
        
    def checkSlot(self):
        self.swith_callback(self.isChecked())

    def paintEvent(self, event):
        label = "ON" if self.isChecked() else "OFF"
        bg_color = Qt.green if self.isChecked() else Qt.red

        radius = 10
        width = 32
        center = self.rect().center()

        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.translate(center)
        painter.setBrush(QtGui.QColor(0,0,0))

        pen = QtGui.QPen(Qt.black)
        pen.setWidth(2)
        painter.setPen(pen)

        painter.drawRoundedRect(QRect(-width, -radius, 2*width, 2*radius), radius, radius)
        painter.setBrush(QtGui.QBrush(bg_color))
        sw_rect = QRect(-radius, -radius, width + radius, 2*radius)
        if not self.isChecked():
            sw_rect.moveLeft(-width)
        painter.drawRoundedRect(sw_rect, radius, radius)
        painter.drawText(sw_rect, Qt.AlignCenter, label)


def make_render_widget() -> tp.Dict[str, tp.Union[QWidget, threading.Condition, threading.Thread]]:
    
    gui_data = {'widget': None,
                'notifier': threading.Condition(threading.RLock()),
                'gui_thread': None,
                'rendering_is_enabled': False,
                'speed': 1.0}
    
    def swith_callback(flag: bool):
        with gui_data['notifier']:
            gui_data['rendering_is_enabled'] = flag
            
    def speed_callback(speed: float):
        with gui_data['notifier']:
            gui_data['speed'] = speed
    
    def run_gui():
        app = QApplication(sys.argv)
    
        widget = QWidget()
        widget.setWindowTitle('Rendering')
        h_layout = QHBoxLayout()
        on_off_button = OnOffSwitch(swith_callback=swith_callback)
        slider = QSlider()
        slider.setOrientation(Qt.Horizontal)
        slider.setMinimum(1.0)
        slider.setMaximum(100.0)
        slider.setTickPosition(QSlider.TicksBelow)
        slider.setTickInterval(1)
        slider.valueChanged.connect(speed_callback)
        
        widget.setFixedHeight(50)
        
        h_layout.addWidget(on_off_button)
        h_layout.addWidget(slider)
        widget.setLayout(h_layout)
        widget.show()
        
        with gui_data['notifier']:
            gui_data['widget'] = widget
            gui_data['notifier'].notify()
        
        sys.exit(app.exec_())
    
    gui_thread = threading.Thread(target=run_gui)
    gui_thread.start()
    
    while True:
        with gui_data['notifier']:
            if gui_data['widget'] is None:
                gui_data['notifier'].wait(timeout=1)
            break
    
    return gui_data


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
    


def sim_ppo_training(num_envs: int, tag: str, enable_rendering: bool, overwrite: bool = False):
    
    device = 'cuda'
    
    cfg = yaml.safe_load(open(Path('ppocket_rocket') / 'ppo_cfg.yaml', 'r'))
    rollout_cfg = dict(cfg['rollout'])
    training_cfg = cfg['training']
    batch_size = int(training_cfg['batch_size'])
    sequence_size = int(cfg['model']['sequence_size'])
    discount_factor = float(cfg['rollout']['discount_factor'])
    
    rollout_max_buffer_size = int(rollout_cfg['max_buffer_size'])
    target_data_size = rollout_max_buffer_size
    rollout_max_buffer_size = math.ceil(rollout_max_buffer_size / num_envs)
    rollout_cfg['max_buffer_size'] = rollout_max_buffer_size + 1
        
    num_cars = 2
    rollout_buffers: tp.List[RolloutBuffer] = []
    
    model_data_provider = ModelDataProvider()
    action_parser = GymActionParser(model_data_provider)
    general_reward = GeneralReward(discount_factor=0.99)
    trainer = PpoTrainer(cfg, device, tag, overwrite)
    actor_critic_policy = trainer.models
    obs_builder = GymObsBuilder(model_data_provider, orange_mirror=True)
    
    blue_mem_reward_values = {reward_name: deque(maxlen=10_000) for reward_name in general_reward.rewards}
    orange_mem_reward_values = {reward_name: deque(maxlen=10_000) for reward_name in general_reward.rewards}
    
    if enable_rendering:
        gui_data = make_render_widget()
    else:
        gui_data = None
    
    tick_skip = 8
    max_num_of_frames = min(90 * 12, rollout_max_buffer_size)
    env: SubprocVecEnv = rlgym_sim_vec_env(num_envs,
                                           reward_fn=general_reward,
                                           terminal_conditions=[TimeoutCondition(max_num_of_frames),
                                                                GoalScoredCondition()],
                                           obs_builder=obs_builder,
                                           state_setter=GeneralStateSetter(dict(cfg['replays'])),
                                           action_parser=action_parser,
                                           tick_skip=tick_skip, spawn_opponents=True, team_size=1,
                                           render_env_idx=0 if enable_rendering else None)

    last_rewards = deque(maxlen=200)
    ep_counter = 0
    
    while True:
        cur_rollout_buffers: tp.List[tp.List[RolloutBuffer]] = []
        for _ in range(num_envs):
            env_buffers = []
            for _ in range(num_cars):
                env_buffers.append(RolloutBuffer(rollout_cfg, actor_critic_policy.value_net, sequence_size))   
            cur_rollout_buffers.append(env_buffers)
        
        episode_t0 = time.time()
        obs = env.reset()
        
        dones = np.array([False for _ in  range(num_envs)], dtype=bool)
        
        n_steps = 0
        ep_rewards = [[0] * num_cars for _ in range(num_envs)]
        
        frame_t0 = time.time()
        while not dones.all():
            action_dists: tp.List[tp.List[torch.distributions.Categorical]] = []
            actions: tp.List[tp.List[int]] = []
            for env_buffers, env_obs in zip(cur_rollout_buffers, obs):
                env_action_dists = []
                env_actions = []
                for cur_rollout_buffer, obs_tensor in zip(env_buffers, env_obs):
                    action_dist = actor_critic_policy.get_action_dist(cur_rollout_buffer.new_state(obs_tensor))
                    action = int(action_dist.sample())
                    env_action_dists.append(action_dist)
                    env_actions.append(action)
                action_dists.append(env_action_dists)
                actions.append(env_actions)
            env.step_async(actions)
            next_obs, splitted_rewards, next_dones, gameinfo = env.step_wait()
            
            rewards = []
            for env_splitted_rewards in splitted_rewards:
                blue_reward = 0.0
                for reward_name, reward_value in env_splitted_rewards[0].items():
                    blue_reward += reward_value
                    blue_mem_reward_values[reward_name].append(reward_value)
                orange_reward = 0.0
                for reward_name, reward_value in env_splitted_rewards[1].items():
                    orange_reward += reward_value
                    orange_mem_reward_values[reward_name].append(reward_value)
                rewards.append([blue_reward, orange_reward])

            for env_buffers, env_obs, env_action_dists, env_actions, env_rewards, env_done in \
                        zip(cur_rollout_buffers, obs, action_dists, actions, rewards, dones):
                if env_done:
                    continue
                for buffer, car_obs, car_action_dist, car_action, car_reward in zip(env_buffers, env_obs, 
                                                                                    env_action_dists, env_actions,
                                                                                    env_rewards):
                    action_log_prob = float(car_action_dist.log_prob(torch.tensor(car_action)))
                    buffer.add(car_obs, car_action, action_log_prob, car_reward)
            
            if n_steps >= max_num_of_frames:
                for env_buffers, env_next_obs, done, next_done in zip(cur_rollout_buffers, next_obs, dones, next_dones):
                    if done or not next_done:
                        continue
                    for car_buffer, car_next_obs in zip(env_buffers, env_next_obs):
                        car_buffer.finish(car_next_obs, truncated=True)
                dones[:] = True
            else:
                for env_buffers, env_next_obs, done, next_done in zip(cur_rollout_buffers, next_obs, dones, next_dones):
                    if done or not next_done:
                        continue
                    for car_buffer, car_next_obs in zip(env_buffers, env_next_obs):
                        car_buffer.finish(car_next_obs, truncated=False)
                dones = np.logical_or(dones, next_dones)
            
            for env_idx, env_rewards in enumerate(rewards):
                for car_idx in range(num_cars):
                    ep_rewards[env_idx][car_idx] += env_rewards[car_idx]
            obs = next_obs
            n_steps += 1
            
            frame_t1 = time.time()
            if gui_data is not None:
                with gui_data['notifier']:
                    if gui_data['rendering_is_enabled']:
                        env.render()
                        dt = frame_t1 - frame_t0
                        target_frame_dt = (tick_skip / 120) / gui_data['speed']
                        if dt < target_frame_dt:
                            time.sleep(target_frame_dt - dt)
            
            frame_t0 = frame_t1
            
        ep_duration = time.time() - episode_t0
        
        ep_rewards = [reward for reward in ep_rewards if reward is not None]
        for env_rewards in ep_rewards:
            for reward in env_rewards:
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
        
        for env_buffers in cur_rollout_buffers:
            rollout_buffers += env_buffers
        
        assert all([rollout_buffer.is_finished for rollout_buffer in rollout_buffers])
        
        data_size = sum([len(rollout_buffer) for rollout_buffer in rollout_buffers])
        ep_rewards_str = ', '.join([f'[{env_ep_reward[0]:.2f}, {env_ep_reward[1]:.2f}]'
                                       for env_ep_reward in ep_rewards])
        print(f'Episode: {ep_counter} | Duration: {ep_duration:.2f} | Rollout buffer size: {data_size} | '\
              f'Mean rewards: {last_mean_reward:.2f} | Episode Rewards: {ep_rewards_str}')
        
        if data_size > target_data_size:
            while data_size > 0:
                dataset, rollout_buffers = RolloutDataset.collect_data(target_data_size, batch_size, sequence_size,
                                                                       rollout_buffers)
                trainer.train(dataset)
                data_size = sum([len(rollout_buffer) for rollout_buffer in rollout_buffers])
            rollout_buffers.clear()
            
        if ep_counter % int(training_cfg['save']['save_hist_every_n_step']) == 0:
            save_histograms(trainer.log_writer.output_plot_folder, blue_mem_reward_values, 'hist_blue')
            save_histograms(trainer.log_writer.output_plot_folder, orange_mem_reward_values, 'hist_orange')

    if gui_data is not None:
        gui_data['widget'].close()
        gui_data['gui_thread'].join()
        
        
if __name__ == '__main__':
    from argparse import ArgumentParser
    
    args_parser = ArgumentParser()
    args_parser.add_argument('-n', '--num_instances', type=int, default=1)
    args_parser.add_argument('--tag', type=str, default='%dt')
    args_parser.add_argument('--render', action='store_true')
    args_parser.add_argument('--overwrite', action='store_true')
    args = args_parser.parse_args()
    
    sim_ppo_training(args.num_instances, args.tag, args.render, args.overwrite)
