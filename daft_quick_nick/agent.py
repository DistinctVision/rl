import typing as tp
import itertools
from pathlib import Path
from dataclasses import dataclass

import random
import logging

import yaml
import numpy as np
import torch
import einops

from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.utils.structures.game_data_struct import GameTickPacket

from daft_quick_nick.utils import Vec3, EulerAngles
from daft_quick_nick.game_data import WorldState, ModelDataProvider
from daft_quick_nick.actor_critic_policy import ActorCriticPolicy

from rlgym_compat import GameState
from rlgym.utils.gamestates import PlayerData, PhysicsObject
from daft_quick_nick.game_data import BallInfo, EulerAngles, Vec3, PlayerInfo



class DaftQuickNick(BaseAgent):
    
    def  __init__(self, name: str, team: int, index: int):
        super().__init__(name, team, index)
        
        cfg = yaml.safe_load(open(Path(__file__).parent / 'ppo_cfg.yaml', 'r'))
        model_cfg = cfg['model']
        game_cfg = cfg['game']
        
        actor_critic_policy = ActorCriticPolicy.build(model_cfg)
        ckpt = torch.load(str(model_cfg['models_path']))
        actor_critic_policy.load_state_dict(ckpt)
        policy_net = actor_critic_policy.policy_net
        del actor_critic_policy
        
        self.fps = float(game_cfg['fps'])
        
        self.device = torch.device('cuda')
        self.policy_net = policy_net.to(self.device)
        self.policy_net.eval()
    
        self.data_provider = ModelDataProvider()
        
        self.last_time_tick: tp.Optional[float] = None
        self.last_controls = SimpleControllerState()
        
    def initialize_agent(self):
        ...
        self.field_info = self.get_field_info()
        self.game_state = GameState(self.field_info)
    
    def _build_world_state(self, agent: PlayerData, agent_obj: PhysicsObject,
                           enemy: PlayerData, enemy_obj: PhysicsObject,
                           ball_obj: PhysicsObject, boost_pads: np.ndarray) -> WorldState:
        
        ball_info = BallInfo(location=Vec3.from_array(ball_obj.position),
                             rotation=EulerAngles.from_array(ball_obj.euler_angles()),
                             velocity=Vec3.from_array(ball_obj.linear_velocity),
                             angular_velocity=Vec3.from_array(ball_obj.angular_velocity))
        
        agent_info = PlayerInfo(location=Vec3.from_array(agent_obj.position),
                                rotation=EulerAngles.from_array(agent_obj.euler_angles()),
                                velocity=Vec3.from_array(agent_obj.linear_velocity),
                                angular_velocity=Vec3.from_array(agent_obj.angular_velocity),
                                boost=agent.boost_amount,
                                is_demolished=agent.is_demoed,
                                has_wheel_contact=agent.on_ground,
                                is_super_sonic=True,
                                jumped=agent.has_jump,
                                double_jumped=agent.has_flip)
        
        enemy_info = PlayerInfo(location=Vec3.from_array(enemy_obj.position),
                                rotation=EulerAngles.from_array(enemy_obj.euler_angles()),
                                velocity=Vec3.from_array(enemy_obj.linear_velocity),
                                angular_velocity=Vec3.from_array(enemy_obj.angular_velocity),
                                boost=enemy.boost_amount,
                                is_demolished=enemy.is_demoed,
                                has_wheel_contact=enemy.on_ground,
                                is_super_sonic=True,
                                jumped=enemy.has_jump,
                                double_jumped=enemy.has_flip)
        
        world_state = WorldState(ball=ball_info, players=[[agent_info], [enemy_info]], boosts=boost_pads)
        
        return world_state
                
    def get_output(self, packet: GameTickPacket) -> SimpleControllerState:
        if not packet.game_info.is_round_active:
            self.last_time_tick = None
            self.memory = []
            return self.last_controls
        
        cur_time_tick = packet.game_info.seconds_elapsed
        if self.last_time_tick is not None:
            frame_time = 1.0 / self.fps
            delta_time = cur_time_tick - self.last_time_tick
            if delta_time > frame_time * 1.25:
                self.logger.warning(f'Too slow processing: {delta_time:.2f}s')
            elif delta_time < frame_time * 0.75:
                return self.last_controls
            
        self.game_state.decode(packet)
        agent: PlayerData = self.game_state.players[self.team]
        player: PlayerData = self.game_state.players[1 - self.team]
        world_state_1 = self._build_world_state(agent, agent.car_data, player, player.car_data, self.game_state.ball, self.game_state.boost_pads)
        
        world_state = WorldState.from_game_packet(packet)
        world_states_tensor = self.data_provider.world_state_to_tensor(world_state_1, self.team, self.device)
        
        with torch.no_grad():
            logits: torch.Tensor = self.policy_net(world_states_tensor.unsqueeze(0))
            probs =  torch.nn.functional.softmax(logits.squeeze(0).cpu())
            action_dist  = torch.distributions.Categorical(probs)
            action_idx = int(action_dist.sample())
        
        action = self.data_provider.action_lookup_table[action_idx, :].tolist()
        [throttle, steer, pitch, yaw, roll, jump, boost, handbrake] = action
        jump = jump > 0.5
        boost = boost > 0.5
        handbrake = handbrake > 0.5
        
        self.last_controls.throttle = throttle
        self.last_controls.steer = steer
        self.last_controls.pitch = pitch
        self.last_controls.yaw = yaw
        self.last_controls.roll = roll
        self.last_controls.jump = jump
        self.last_controls.boost = boost
        self.last_controls.handbrake = handbrake
        
        self.last_time_tick = cur_time_tick
        return self.last_controls
    