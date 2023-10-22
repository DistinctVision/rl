import typing as tp
import itertools
from pathlib import Path
from dataclasses import dataclass
import random
import logging

import yaml
import torch
import einops

from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.utils.structures.game_data_struct import GameTickPacket

from daft_quick_nick.utils import Vec3, EulerAngles
from daft_quick_nick.game_data import WorldState
from daft_quick_nick.model import CriticModel, get_model_num_params, ModelDataProvider



class DaftQuickNick(BaseAgent):
    
    def  __init__(self, name: str, team: int, index: int):
        super().__init__(name, team, index)
        
        cfg = yaml.safe_load(open(Path(__file__).parent / 'cfg.yaml', 'r'))
        model_cfg = cfg['model']
        game_cfg = cfg['game']
        self.action_set = [0, 2, 4, 6, 8, 10, 12, 16, 20]
        
        self.fps = float(game_cfg['fps'])
        
        self.device = torch.device('cuda')
    
        self.data_provider = ModelDataProvider()
        self.cirtic_model: tp.Optional[CriticModel] = None
        
        self.cirtic_model = CriticModel.build_model(self.data_provider, model_cfg)
        self.cirtic_model.eval()
        ckpt_path = str(model_cfg['critic_model_path'])
        ckpt = torch.load(ckpt_path, map_location='cpu')
        self.cirtic_model.load_state_dict(ckpt, strict=False)
        self.logger.info(f'Model is loaded from "{ckpt_path}"')
        
        self.last_time_tick: tp.Optional[float] = None
        self.last_controls = SimpleControllerState()
        
    def initialize_agent(self):
        self.cirtic_model = self.cirtic_model.to(self.device)
                
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
        
        world_state = WorldState.from_game_packet(packet)
        
        world_states_tensor = self.data_provider.world_state_to_tensor(world_state, self.team, self.device)
        
        with torch.no_grad():
            predicted_rewards: torch.Tensor = self.cirtic_model(
                    batch_world_states_tensor=world_states_tensor.unsqueeze(0).to(self.cirtic_model.device))
            predicted_rewards = predicted_rewards.detach().cpu().squeeze(0)
            
            for action_idx in range(predicted_rewards.shape[0]):
                if action_idx not in self.action_set:
                    predicted_rewards[action_idx] = -1e6
            action_idx = int(predicted_rewards.argmax(0))
        
        action = self.data_provider.action_lookup_table[action_idx, :].tolist()
        [throttle, pitch, steer_or_yaw, roll, jump, boost, handbrake] = action
        jump = jump > 0.5
        boost = boost > 0.5
        handbrake = handbrake > 0.5
        
        self.last_controls.throttle = throttle
        self.last_controls.steer = steer_or_yaw
        self.last_controls.yaw = steer_or_yaw
        self.last_controls.pitch = pitch
        self.last_controls.roll = roll
        self.last_controls.jump = jump
        self.last_controls.boost = boost
        self.last_controls.handbrake = handbrake
        
        self.last_time_tick = cur_time_tick
        return self.last_controls
    