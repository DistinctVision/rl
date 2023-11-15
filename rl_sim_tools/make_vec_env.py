import typing as tp

import multiprocessing as mp

from rlgym_sim.gym import Gym
from rlgym_sim.envs import Match
from rlgym_sim.utils.terminal_conditions import common_conditions
from rlgym_sim.utils.reward_functions import DefaultReward
from rlgym_sim.utils.obs_builders import DefaultObs
from rlgym_sim.utils.action_parsers import DefaultAction
from rlgym_sim.utils.state_setters import DefaultState
from rlgym_sim.utils import common_values

from .sub_proc_vec_env import SubprocVecEnv


def rlgym_sim_vec_env(num_envs: int,
                      tick_skip: int = 8,
                      spawn_opponents: bool = False,
                      team_size: int = 1,
                      gravity: float = 1,
                      boost_consumption: float = 1,
                      copy_gamestate_every_step = True,
                      dodge_deadzone = 0.8,
                      terminal_conditions: tp.List[object] = [common_conditions.TimeoutCondition(225),
                                                              common_conditions.GoalScoredCondition()],
                      reward_fn: object = DefaultReward(),
                      obs_builder: object = DefaultObs(),
                      action_parser: object = DefaultAction(),
                      state_setter: object = DefaultState(),
                      render_env_idx: tp.Optional[int] = None) -> SubprocVecEnv:
    env_args_list = [dict(
        match=Match(reward_function=reward_fn,
                    terminal_conditions=terminal_conditions,
                    obs_builder=obs_builder,
                    action_parser=action_parser,
                    state_setter=state_setter,
                    team_size=team_size,
                    spawn_opponents=spawn_opponents),
        tick_skip=tick_skip, gravity=gravity, boost_consumption=boost_consumption,
        copy_gamestate_every_step=copy_gamestate_every_step, dodge_deadzone=dodge_deadzone,
    ) for _ in range(num_envs)]
    forkserver_available = "forkserver" in mp.get_all_start_methods()
    start_method = "forkserver" if forkserver_available else "spawn"
    return SubprocVecEnv(env_args_list, start_method=start_method, render_env_idx=render_env_idx)
    