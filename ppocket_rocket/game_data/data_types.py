import typing as tp
from dataclasses import dataclass

from rlbot.utils.structures.game_data_struct import GameTickPacket
from rlbot.agents.base_agent import SimpleControllerState
from ppocket_rocket.utils import Vec3, EulerAngles


@dataclass
class BallInfo:
    location: Vec3
    rotation: EulerAngles
    
    velocity: Vec3
    angular_velocity: Vec3
    
    @staticmethod
    def from_dict(data: tp.Dict[str, tp.List[float]]) -> 'BallInfo':
        d_location = data['location']
        d_rotation = data['rotation']
        d_velocity = data['velocity']
        d_angular_velocity = data['angular_velocity']
        return BallInfo(location=Vec3(float(d_location[0]), float(d_location[1]), float(d_location[2])),
                        rotation=EulerAngles(float(d_rotation[0]), float(d_rotation[1]), float(d_rotation[2])),
                        velocity=Vec3(float(d_velocity[0]), float(d_velocity[1]), float(d_velocity[2])),
                        angular_velocity=Vec3(float(d_angular_velocity[0]),
                                              float(d_angular_velocity[1]),
                                              float(d_angular_velocity[2])))
    
    def to_dict(self) -> tp.Dict[str, tp.List[float]]:
        return {
            'location': [self.location.x, self.location.y, self.location.z],
            'rotation': [self.rotation.pitch, self.rotation.yaw, self.rotation.roll],
            'velocity': [self.velocity.x, self.velocity.y, self.velocity.z],
            'angular_velocity': [self.angular_velocity.x, self.angular_velocity.y, self.angular_velocity.z]
        }
    
    
@dataclass
class PlayerInfo:
    location: Vec3
    rotation: EulerAngles
    
    velocity: Vec3
    angular_velocity: Vec3
    
    boost: int
    
    is_demolished: bool
    has_wheel_contact: bool
    is_super_sonic: bool
    jumped: bool
    double_jumped: bool
    
    
    @staticmethod
    def from_dict(data: tp.Dict[str, tp.Union[tp.List[float], float, int]]) -> 'PlayerInfo':
        d_location = data['location']
        d_rotation = data['rotation']
        d_velocity = data['velocity']
        d_angular_velocity = data['angular_velocity']
        boost = int(data['boost'])
        is_demolished = int(data['is_demolished']) > 0
        has_wheel_contact = int(data['has_wheel_contact']) > 0
        is_super_sonic = int(data['is_super_sonic']) > 0
        jumped = int(data['jumped']) > 0
        double_jumped = int(data['double_jumped']) > 0
        return PlayerInfo(location=Vec3(float(d_location[0]), float(d_location[1]), float(d_location[2])),
                          rotation=EulerAngles(float(d_rotation[0]), float(d_rotation[1]), float(d_rotation[2])),
                          velocity=Vec3(float(d_velocity[0]), float(d_velocity[1]), float(d_velocity[2])),
                          angular_velocity=Vec3(float(d_angular_velocity[0]),
                                                float(d_angular_velocity[1]),
                                                float(d_angular_velocity[2])),
                          boost=boost,
                          is_demolished=is_demolished, has_wheel_contact=has_wheel_contact,
                          is_super_sonic=is_super_sonic, jumped=jumped, double_jumped=double_jumped)         
    
    def to_dict(self) -> tp.Dict[str, tp.Union[tp.List[float], float, int]]:
        return {
            'location': [self.location.x, self.location.y, self.location.z],
            'rotation': [self.rotation.pitch, self.rotation.yaw, self.rotation.roll],
            'velocity': [self.velocity.x, self.velocity.y, self.velocity.z],
            'angular_velocity': [self.angular_velocity.x, self.angular_velocity.y, self.angular_velocity.z],
            'boost': self.boost,
            'is_demolished': 1 if self.is_demolished else 0,
            'has_wheel_contact': 1 if self.has_wheel_contact else 0,
            'is_super_sonic':  1 if self.is_super_sonic else 0,
            'jumped': 1 if self.jumped else 0,
            'double_jumped': 1 if self.double_jumped else 0
        }


@dataclass
class WorldState:
    
    ball: BallInfo
    players: tp.List[tp.List[PlayerInfo]]
    boosts: tp.List[bool]
    
    @staticmethod
    def from_game_packet(packet: GameTickPacket) -> 'WorldState':
        ball_body = packet.game_ball.physics
        ball = BallInfo(location=Vec3(ball_body.location), rotation=EulerAngles(ball_body.rotation),
                        velocity=Vec3(ball_body.velocity), angular_velocity=Vec3(ball_body.angular_velocity))
        
        teams = [[] for _ in range(packet.num_teams)]
        for p_player in packet.game_cars[:packet.num_cars]:
            p_player_body = p_player.physics
            player = PlayerInfo(location=Vec3(p_player_body.location), rotation=EulerAngles(p_player_body.rotation),
                                velocity=Vec3(p_player_body.velocity), angular_velocity=Vec3(p_player_body.angular_velocity),
                                boost=int(p_player.boost), is_demolished=bool(p_player.is_demolished),
                                has_wheel_contact=bool(p_player.has_wheel_contact), is_super_sonic=bool(p_player.is_super_sonic),
                                jumped=bool(p_player.jumped), double_jumped=bool(p_player.double_jumped))
            teams[p_player.team].append(player)
        
        boosts = [1 if packet.game_boosts[boost_idx].is_active else 0 for boost_idx in range(packet.num_boost)]
        return WorldState(ball=ball, players=teams, boosts=boosts)
    
    @staticmethod
    def from_dict(data: tp.Dict[str, tp.Union[tp.List[float], float, int]]) -> 'WorldState':
        ball = BallInfo.from_dict(data['ball'])
        players = [[PlayerInfo.from_dict(d_player) for d_player in list(d_team)] for d_team in list(data['players'])]
        boosts = [(d_boost > 0) for d_boost in data['boosts']]
        return WorldState(ball=ball, players=players, boosts=boosts)
    
    def to_dict(self) -> tp.Dict[str, tp.Union[tp.List[float], float, int]]:
        return {
            'ball': self.ball.to_dict(),
            'players': [[player.to_dict() for player in team] for team in self.players],
            'boosts': [1 if b else 0 for b in self.boosts]
        }
        
        
@dataclass
class ActionState:
    steer: float
    throttle: float
    pitch: float
    yaw: float
    roll: float
    jump: bool
    boost: bool
    handbrake: bool
    use_item: bool
    
    @staticmethod
    def from_controller_state(controls: SimpleControllerState) -> 'ActionState':
        return ActionState(steer=float(controls.steer), throttle=float(controls.throttle),
                           pitch=float(controls.pitch), yaw=float(controls.yaw), roll=float(controls.roll),
                           jump=bool(controls.jump), boost=bool(controls.boost), handbrake=bool(controls.handbrake),
                           use_item=bool(controls.use_item))
    
    @staticmethod
    def from_dict(data: tp.Dict[str, tp.Union[tp.List[float], float, int]]) -> 'ActionState':
        steer = float(data['steer'])
        throttle = float(data['throttle'])
        pitch = float(data['pitch'])
        yaw = float(data['yaw'])
        roll = float(data['roll'])
        jump = int(data['jump']) > 0
        boost = int(data['boost']) > 0
        handbrake = int(data['handbrake']) > 0
        use_item = int(data['use_item']) > 0
        return ActionState(steer=steer, throttle=throttle,
                           pitch=pitch, yaw=yaw, roll=roll,
                           jump=jump, boost=boost, handbrake=handbrake, use_item=use_item)
        
    def to_dict(self) -> tp.Dict[str, tp.Union[float, int]]:
        return {
            'steer': self.steer,
            'throttle': self.throttle,
            'pitch': self.pitch,
            'yaw': self.yaw,
            'roll': self.roll,
            'jump': 1 if self.jump else 0,
            'boost': 1 if self.boost else 0,
            'handbrake': 1 if self.handbrake else 0,
            'use_item': 1 if self.use_item else 0
        }
    
    
@dataclass
class FrameRecord:
    world_state: WorldState
    action_state: tp.Optional[ActionState] = None
    reward: tp.Optional[float] = None
    
    @staticmethod
    def from_dict(data: tp.Dict[str, tp.Union[tp.List[float], float, int]]) -> 'FrameRecord':
        world_state = WorldState.from_dict(data['world_state'])
        action_state = data.get('action_state', None)
        if action_state is not None:
            action_state = ActionState.from_dict(action_state)
        reward = data.get('reward', None)
        if reward is not None:
            reward = float(reward)
        return FrameRecord(world_state=world_state, action_state=action_state, reward=reward)
    
    def to_dict(self) -> tp.Dict[str, tp.Union[tp.List[float], float, int]]:
        return {
            'world_state': self.world_state.to_dict(),
            'action_state': self.action_state.to_dict() if self.action_state is not None else None,
            'reward': self.reward
        }
        