import typing as tp

import gym.spaces
import numpy as np
import torch

from rlgym.utils.obs_builders import ObsBuilder
from rlgym.utils.gamestates import PlayerData, GameState, PhysicsObject

from ppocket_rocket.game_data import WorldState, BallInfo, EulerAngles, Vec3, PlayerInfo, ModelDataProvider


class GymObsBuilder(ObsBuilder):

    def __init__(self, model_data_provider: ModelDataProvider, use_mirror: bool = True):
        self.model_data_provider = model_data_provider
        self.use_mirror = use_mirror

    def get_obs_space(self) -> gym.spaces.Space:
        return gym.spaces.Box(low=-np.inf, high=np.inf,
                              shape=(self.model_data_provider.WORLD_STATE_SIZE,),
                              dtype=np.float32)

    def reset(self, initial_state: GameState):
        ...

    def pre_step(self, state: GameState):
        ...

    def build_obs(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> torch.Tensor:
        enemy_data: tp.Optional[PlayerData] = None
        for p_player in state.players:
            if p_player.car_id == player.car_id:
                continue
            enemy_data = p_player
        assert enemy_data is not None
        
        world_state = self._build_world_state(player,  player.car_data,
                                              enemy_data, enemy_data.car_data,
                                              state.ball,  state.boost_pads)
        world_state_tensor = self.model_data_provider.world_state_to_tensor(world_state=world_state,
                                                                            agent_team_idx=0,
                                                                            copy=False)
        if not self.use_mirror:
            return world_state_tensor
        
        world_state = self._build_world_state(player, player.inverted_car_data,
                                              enemy_data, enemy_data.inverted_car_data,
                                              state.inverted_ball, state.inverted_boost_pads)
        inverted_world_state_tensor = self.model_data_provider.world_state_to_tensor(world_state=world_state,
                                                                                     agent_team_idx=0,
                                                                                     copy=False)
        return torch.stack([world_state_tensor, inverted_world_state_tensor])
    
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
                                boost=int(agent.boost_amount * 100),
                                is_demolished=agent.is_demoed,
                                has_wheel_contact=agent.on_ground,
                                is_super_sonic=False,
                                jumped=not agent.has_jump,
                                double_jumped=not agent.has_flip)
        
        enemy_info = PlayerInfo(location=Vec3.from_array(enemy_obj.position),
                                rotation=EulerAngles.from_array(enemy_obj.euler_angles()),
                                velocity=Vec3.from_array(enemy_obj.linear_velocity),
                                angular_velocity=Vec3.from_array(enemy_obj.angular_velocity),
                                boost=int(enemy.boost_amount * 100),
                                is_demolished=enemy.is_demoed,
                                has_wheel_contact=enemy.on_ground,
                                is_super_sonic=False,
                                jumped=not enemy.has_jump,
                                double_jumped=not enemy.has_flip)
        
        world_state = WorldState(ball=ball_info, players=[[agent_info], [enemy_info]], boosts=boost_pads)
        
        return world_state
        
    