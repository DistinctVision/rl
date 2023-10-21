from pathlib import Path
import yaml
import time
from collections import deque
import shutil
from functools import partial
from threading import Thread

import rlgym
from rlgym.gamelaunch import LaunchPreference

from daft_quick_nick.game_data import ModelDataProvider


cfg = yaml.safe_load(open(Path(__file__).parent / 'daft_quick_nick' / 'cfg.yaml', 'r'))
data_provider = ModelDataProvider()


def one_process(process_index: int = 0):
    env = rlgym.make(game_speed=100, tick_skip=24, spawn_opponents=True, team_size=1,
                    launch_preference=LaunchPreference.EPIC)
    
    stat = deque()

    while True:
        world_state_tensor, state = env.reset()
        done = False
        steps = 0
        ep_reward = 0
        
        data_is_updated = False
        
        t0 = time.time()
        
        while not done:
            action = env.action_space.sample()
            next_obs, reward, done, gameinfo = env.step(action)
            
            ep_reward += reward
            obs = next_obs
            steps += 1
            
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            stat.append(dt)
        
            avg_time = sum(stat) / len(stat)
            print(f'Time[{process_index}]: {avg_time:.4f}')
            
            
def one_process_many_envs(n_envs: int):
    envs = [rlgym.make(game_speed=100, tick_skip=24, spawn_opponents=True, team_size=1,
                       launch_preference=LaunchPreference.EPIC) for _ in range(n_envs)]
    
    stat = deque()
    
    for env in envs:
        env.reset()
    
    t0 = time.time()
    while True:
        done_flags = []
        for env in envs:
            action = env.action_space.sample()
            next_obs, reward, done, gameinfo = env.step(action)
            done_flags.append(done)
        
        ep_reward += reward
        obs = next_obs
        steps += 1
        
        t1 = time.time()
        dt = t1 - t0
        stat.append(dt)
        
        avg_time = sum(stat) / len(stat)
        avg_time /= n_envs
        print(f'Time: {avg_time:.4f}')
        
        for env, done in zip(envs, done_flags):
            if done:
                env.reset()
        
        t0 = time.time()


def test1():
    threads = [Thread(target=partial(one_process, idx)) for idx in range(4)]
    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()
    

def test2():
    one_process_many_envs(4)
    
    
test1()