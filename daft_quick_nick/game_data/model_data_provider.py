import typing as tp
import math
from copy import deepcopy

import torch

from daft_quick_nick.game_data.data_types import Vec3, EulerAngles, BallInfo, PlayerInfo, WorldState, ActionState


class DataScaler:
    def __init__(self, pos_std: float = 2300, angle_std: float = math.pi):
        self.pos_std = pos_std
        self.angle_std = angle_std
        
    def scale(self, world_state: WorldState, copy: bool = True) -> WorldState:
        # if copy:
        #     world_state = deepcopy(world_state)
        # world_state.ball.location = world_state.ball.location / self.pos_std
        # world_state.ball.velocity = world_state.ball.velocity / self.pos_std
        # world_state.ball.angular_velocity /= self.angle_std
        # for team_players in world_state.players:
        #     for player_info in team_players:
        #         player_info.location = player_info.location / self.pos_std
        #         player_info.velocity = player_info.velocity / self.pos_std
        #         player_info.boost /= 100
        #         player_info.angular_velocity /= self.angle_std
        return world_state
        
    def unscale(self, world_state: WorldState, copy: bool = True) -> WorldState:
        # if copy:
        #     world_state = deepcopy(world_state)
        # world_state.ball.location = world_state.ball.location * self.pos_std
        # world_state.ball.velocity = world_state.ball.velocity * self.pos_std
        # world_state.ball.angular_velocity *= self.angle_std
        # for team_players in world_state.players:
        #     for player_info in team_players:
        #         player_info.location = player_info.location * self.pos_std
        #         player_info.velocity = player_info.velocity * self.pos_std
        #         player_info.boost *= 100
        #         player_info.angular_velocity *= self.angle_std
        return world_state


class ModelDataProvider:
    
    WORLD_STATE_SIZE = 6 + (14 + 4) * 2 + 1
        
    @staticmethod
    def world_state_categorial_flags() -> tp.List[bool]:
        ball_flags = [False] * 6
        bot_flags = [False] * 14 + [True] * 4
        player_flags = [False] * 14 + [True] * 4
        flags = ball_flags + bot_flags + player_flags + [False]
        return flags
            
    def __init__(self):
        self.action_lookup_table, self.action_states = self._make_action_lookup_table()
        self.data_scaler = DataScaler()
        
    @property
    def default_action_index(self) -> int:
        return 8
    
    def world_state_to_tensor(self, world_state: WorldState, agent_team_idx: int,
                              device: tp.Union[torch.device, str] = 'cpu',
                              copy: bool = True) -> \
                                  tp.Union[torch.Tensor, tp.Tuple[torch.Tensor, tp.List[bool]]]:
        assert agent_team_idx in {0, 1}
        assert len(world_state.players) == 2 and len(world_state.players[0]) == 1 and len(world_state.players[1]) == 1
        
        world_state = self.data_scaler.scale(world_state, copy=copy)
        
        ball = world_state.ball
        ball_features = [ball.location.x, ball.location.y, ball.location.z,
                         ball.velocity.x, ball.velocity.y, ball.velocity.z]
        
        bot = world_state.players[agent_team_idx][0]
        bot_q = bot.rotation.to_quaternion()
        bot_features = [bot.location.x, bot.location.y, bot.location.z,
                        bot_q[0], bot_q[1], bot_q[2], bot_q[3],
                        bot.velocity.x, bot.velocity.y, bot.velocity.z,
                        bot.angular_velocity.x, bot.angular_velocity.y, bot.angular_velocity.z,
                        float(bot.boost),
                        1.0 if bot.has_wheel_contact else 0.0,
                        1.0 if bot.is_super_sonic else 0.0,
                        1.0 if bot.jumped else 0.0,
                        1.0 if bot.double_jumped else 0.0]
        
        player = world_state.players[1 - agent_team_idx][0]
        player_q = player.rotation.to_quaternion()
        player_features = [player.location.x, player.location.y, player.location.z,
                           player_q[0], player_q[1], player_q[2], player_q[3],
                           player.velocity.x, player.velocity.y, player.velocity.z,
                           player.angular_velocity.x, player.angular_velocity.y, player.angular_velocity.z,
                           float(player.boost),
                           1.0 if player.has_wheel_contact else 0.0,
                           1.0 if player.is_super_sonic else 0.0,
                           1.0 if player.jumped else 0.0,
                           1.0 if player.double_jumped else 0.0]
        
        features = ball_features + bot_features + player_features + [(bot.location - ball.location).length()]
        return torch.tensor(features, dtype=torch.float32, device=device)
    
    def tensor_to_world_state(self, tensor: torch.Tensor,
                              agent_team_idx: tp.Union[int, tp.List[int]]) -> tp.Union[WorldState,
                                                                                       tp.List[WorldState]]:
        assert agent_team_idx in {0, 1}
        
        if len(tensor.shape) == 1:
            assert isinstance(agent_team_idx, int)
            agent_team_idx = [agent_team_idx]
            tensor = tensor.unsqueeze(0)
            is_batch = False
        else:
            assert isinstance(agent_team_idx, list) and len(agent_team_idx) == tensor.shape[0]
            is_batch = True
            
        out = []
        for data, team_idx in zip(tensor.unbind(0), agent_team_idx):
            ball_data = data[:6]
            ball = BallInfo(location=Vec3(ball_data[0], ball_data[1], ball_data[2]),
                            rotation=EulerAngles(0, 0, 0),
                            velocity=Vec3(ball_data[3], ball_data[4], ball_data[5]),
                            angular_velocity=Vec3(0, 0, 0))
            
            agent_num_data = data[6:20]
            agent_cat_data = data[20:24]
            agent_info = PlayerInfo(location=Vec3(agent_num_data[0], agent_num_data[1], agent_num_data[2]),
                                    rotation=EulerAngles.from_quaternion(agent_num_data[3:7]),
                                    velocity=Vec3(agent_num_data[7], agent_num_data[8], agent_num_data[9]),
                                    angular_velocity=Vec3(agent_num_data[10], agent_num_data[11], agent_num_data[12]),
                                    boost=float(agent_num_data[13]),
                                    is_demolished=False,
                                    has_wheel_contact=agent_cat_data[0] > 0.5,
                                    is_super_sonic=agent_cat_data[1] > 0.5,
                                    jumped=agent_cat_data[2] > 0.5,
                                    double_jumped=agent_cat_data[3] > 0.5)
            
            enemy_num_data = data[24:38]
            enemy_cat_data = data[38:42]
            enemy_info = PlayerInfo(location=Vec3(enemy_num_data[0], enemy_num_data[1], enemy_num_data[2]),
                                    rotation=EulerAngles.from_quaternion(enemy_num_data[3:7]),
                                    velocity=Vec3(enemy_num_data[7], enemy_num_data[8], enemy_num_data[9]),
                                    angular_velocity=Vec3(enemy_num_data[10], enemy_num_data[11],
                                                          enemy_num_data[12]),
                                    boost=float(enemy_num_data[13]),
                                    is_demolished=False,
                                    has_wheel_contact=enemy_cat_data[0] > 0.5,
                                    is_super_sonic=enemy_cat_data[1] > 0.5,
                                    jumped=enemy_cat_data[2] > 0.5,
                                    double_jumped=enemy_cat_data[3] > 0.5)
            
            teams = [[agent_info], [enemy_info]] if team_idx == 0 else [[enemy_info], [agent_info]]
            world_state = WorldState(ball=ball, players=teams, boosts=None)
            world_state = self.data_scaler.unscale(world_state, copy=True)
            out.append(world_state)
        return out if is_batch else out[0]
    
    def batch_world_states_to_tensor(self, batch_world_states: tp.List[tp.List[WorldState]],
                                     agent_team_idx: tp.List[int],
                                     device: tp.Union[torch.device, str] = 'cpu',
                                     copy: bool = True) -> torch.Tensor:
        world_states_tensor = [torch.stack([self.world_state_to_tensor(world_state, team_idx, device, copy=copy)
                                            for world_state in frame_elements])
                               for frame_elements, team_idx in zip(batch_world_states, agent_team_idx)]
        world_states_tensor = torch.stack(world_states_tensor).to(device)
        return world_states_tensor
    
    def _make_action_lookup_table(self) -> tp.Tuple[torch.Tensor, tp.List[ActionState]]:
        action_table = []
        action_states = []
        # Ground
        for throttle in (-1, 0, 1):
            for steer in (-1, 0, 1):
                for boost in (0, 1):
                    for handbrake in (0, 1):
                        if boost == 1 and throttle != 1:
                            continue
                        action_table.append([throttle or boost, steer, 0, steer, 0, 0, boost, handbrake])
                        action_states.append(ActionState(steer=steer,
                                                         throttle=throttle or boost,
                                                         pitch=0, yaw=steer, roll=0,
                                                         jump=False, boost=boost > 0, handbrake=handbrake > 0,
                                                         use_item=False))
                        
        # print(f'Ground: {len(action_states)}')
        
        # Aerial
        for pitch in (-1, 0, 1):
            for yaw in (-1, 0, 1):
                for roll in (-1, 0, 1):
                    for jump in (0, 1):
                        for boost in (0, 1):
                            if jump == 1 and yaw != 0:  # Only need roll for sideflip
                                continue
                            if pitch == roll == jump == 0:  # Duplicate with ground
                                continue
                            # Enable handbrake for potential wavedashes
                            handbrake = jump == 1 and (pitch != 0 or yaw != 0 or roll != 0)
                            action_table.append([boost, yaw, pitch, yaw, roll, jump, boost, handbrake])
                            action_states.append(ActionState(steer=yaw,
                                                             throttle=boost,
                                                             pitch=pitch, yaw=yaw, roll=roll,
                                                             jump=jump > 0, boost=boost > 0, handbrake=handbrake > 0,
                                                             use_item=False))
        action_lookup_table = torch.tensor(action_table, dtype=torch.long)
        return action_lookup_table, action_states
    
    def action_state_to_action_idx(self, action_state: ActionState) -> int:
        steer_or_yaw = max(action_state.steer, action_state.yaw)
        features = [action_state.throttle, steer_or_yaw,
                    action_state.pitch, steer_or_yaw, action_state.roll,
                    1.0 if action_state.jump else 0.0,
                    1.0 if action_state.boost else 0.0,
                    1.0 if action_state.handbrake else 0.0]
        features = torch.tensor(features, dtype=torch.float32)
        distance_sq = torch.sum(torch.square(torch.sub(self.action_lookup_table, features)), dim=1)
        return int(distance_sq.argmin(dim=0))
    
    def batch_action_states_to_indices(self,
                                       batch_action_states: tp.List[tp.List[ActionState]],
                                       device: tp.Union[torch.device, str] = 'cpu') -> torch.Tensor:
        batch_action_indices = torch.tensor([[self.action_state_to_action_idx(action_state)
                                              for action_state in frame_elements]
                                             for frame_elements in batch_action_states],
                                            dtype=torch.long, device=device)
        return batch_action_indices
    
    @property
    def num_actions(self) -> int:
        return self.action_lookup_table.shape[0]
    
    