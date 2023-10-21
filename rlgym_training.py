import  typing as tp
from pathlib import Path
import yaml
import time
from collections import deque

import numpy as np

import rlgym
from rlgym.envs import Match
from rlgym.utils.reward_functions import DefaultReward
from rlgym.utils.obs_builders import DefaultObs
from rlgym.utils.state_setters import DefaultState
from rlgym.utils.action_parsers import DefaultAction
from rlgym.utils.terminal_conditions.common_conditions import TimeoutCondition
from rlgym.gamelaunch import LaunchPreference
from rlgym_tools.sb3_utils import SB3MultipleInstanceEnv

from daft_quick_nick.game_data import ModelDataProvider


cfg = yaml.safe_load(open(Path(__file__).parent / 'daft_quick_nick' / 'cfg.yaml', 'r'))
data_provider = ModelDataProvider()

action_parser = DefaultAction()

def get_match():
    
    # Here we configure our Match. If you want to use custom configuration objects, make sure to replace the default arguments here with instances of the objects you want.
    return Match(
        reward_function=DefaultReward(),
        terminal_conditions=[TimeoutCondition(225)],
        obs_builder=DefaultObs(),
        state_setter=DefaultState(),
        action_parser=action_parser,
        game_speed=100, tick_skip=24, spawn_opponents=True, team_size=1
    )


if __name__ == '__main__':
    stat = deque()
    env = SB3MultipleInstanceEnv(match_func_or_matches=get_match, num_instances=2, wait_time=20)

    action_space = action_parser.get_action_space()
    while True:
        world_state_tensor = env.reset()
        done = np.array([False])
        steps = 0
        ep_reward = 0
        
        data_is_updated = False
        
        t0 = time.time()
        
        while not done.all():
            action = [action_space.sample(), action_space.sample()]
            env.step_async(action)
            next_obs, reward, done, gameinfo = env.step_wait()
            
            ep_reward += reward
            obs = next_obs
            steps += 1
            
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            stat.append(dt)
        
            avg_time = sum(stat) / len(stat)
            print(f'Time: {avg_time:.4f}')
    