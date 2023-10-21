import typing as tp

import gym.spaces
import numpy as np
import torch

from rlgym.utils.obs_builders import ObsBuilder
from rlgym.utils.gamestates import PlayerData, GameState

from game_data import WorldState, BallInfo, EulerAngles, Vec3, PlayerInfo, ModelDataProvider


class GymObsBuilder(ObsBuilder):

    def __init__(self):
        self.data_provider = ModelDataProvider()
        self.inverted = False

    def get_obs_space(self) -> gym.spaces.Space:
        return gym.spaces.Space(shape=(self.data_provider.WORLD_STATE_SIZE,), dtype=np.float32)

    def reset(self, initial_state: GameState):
        self.inverted = np.random.choice([True, False])

    def pre_step(self, state: GameState):
        ...

    def build_obs(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> torch.Tensor:
        ball_data = state.inverted_ball if self.inverted else state.ball
        
        enemy_data: tp.Optional[PlayerData] = None
        for p_player in state.players:
            if p_player.car_id == player.car_id:
                continue
            enemy_data = p_player
        assert enemy_data is not None
        
        enemy_car_data = enemy_data.car_data if self.inverted else enemy_data.inverted_car_data
        
        player_car_data = player.inverted_car_data if self.inverted else player.car_data
        
        ball_info = BallInfo(location=Vec3.from_array(ball_data.position),
                             rotation=EulerAngles.from_array(ball_data.euler_angles()),
                             velocity=Vec3.from_array(ball_data.linear_velocity),
                             angular_velocity=Vec3.from_array(ball_data.angular_velocity))
        
        enemy_info = PlayerInfo(location=Vec3.from_array(enemy_car_data.position),
                                rotation=EulerAngles.from_array(enemy_car_data.euler_angles()),
                                velocity=Vec3.from_array(enemy_car_data.linear_velocity),
                                angular_velocity=Vec3.from_array(enemy_car_data.angular_velocity),
                                boost=enemy_data.boost_amount,
                                is_demolished=enemy_data.is_demoed,
                                has_wheel_contact=enemy_data.on_ground,
                                is_super_sonic=True,
                                jumped=enemy_data.has_jump,
                                double_jumped=enemy_data.has_flip)
        
        agent_info = PlayerInfo(location=Vec3.from_array(player_car_data.position),
                                rotation=EulerAngles.from_array(player_car_data.euler_angles()),
                                velocity=Vec3.from_array(player_car_data.linear_velocity),
                                angular_velocity=Vec3.from_array(player_car_data.angular_velocity),
                                boost=player.boost_amount,
                                is_demolished=player.is_demoed,
                                has_wheel_contact=player.on_ground,
                                is_super_sonic=True,
                                jumped=player.has_jump,
                                double_jumped=player.has_flip)
        boosts = state.inverted_boost_pads if self.inverted else state.boost_pads
        world_state = WorldState(ball=ball_info, players=[[agent_info], [enemy_info]], boosts=boosts)
        
        world_state_tensor = self.data_provider.world_state_to_tensor(world_state=world_state, agent_team_idx=0,
                                                                      copy=False)
        return world_state_tensor
    