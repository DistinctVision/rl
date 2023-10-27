import typing as tp
import yaml
from pathlib import Path

from dataclasses import dataclass, field
import time

import torch

from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.utils.structures.game_data_struct import GameTickPacket

from daft_quick_nick.game_data import WorldState, ActionState, ModelDataProvider
from daft_quick_nick.state_predictor import StatePredictorModel


AgentType = tp.TypeVar('AgentType', bound=BaseAgent, covariant=True)
    
    
@dataclass
class StatePredictorDebugData:
    fps: float
    model: StatePredictorModel
    team_idx: int
    last_time_tick: tp.Optional[float] = None
    last_world_states: tp.List[WorldState] = field(default_factory=lambda: [])
    rnn_hidden: tp.Optional[tp.Tuple[torch.Tensor,  torch.Tensor]] = None


class StatePredictorDebugWrapper(tp.Generic[AgentType], BaseAgent):
    
    # This is a dirty hack but it is a simplest way that I have found to make a inherited template class
    def __new__(cls, *args, **kwargs):
        AgentClass = cls.__orig_bases__[0].__args__[0]
        name_agent_type = AgentClass.__name__
        
        # Replace base class in order to use the template class instead of the abstract base class
        DerivedType = type(f'StatePredictorDebugWrapper[{name_agent_type}]', (cls,), {})
        DerivedType.__bases__ = (DerivedType.__bases__[0], AgentClass,)
        instance =  object.__new__(DerivedType)
        instance.__init__(*args, **kwargs)
        return instance
    
    def __init__(self, name, team, index):
        cfg = yaml.safe_load(open(Path(__file__).parent / '..' / 'cfg.yaml', 'r'))
        game_cfg = cfg['game']
        rnn_model_cfg = cfg['model']['rnn']
        data_provider = ModelDataProvider()
        state_model_predictor = StatePredictorModel.build_model(rnn_model_cfg, data_provider)
        state_model_predictor_cpkt = torch.load(rnn_model_cfg['state_predictor_path'])
        state_model_predictor.load_state_dict(state_model_predictor_cpkt)
        state_model_predictor.eval()
        state_model_predictor = state_model_predictor.cuda()
        self.state_predictor_data = StatePredictorDebugData(fps=float(game_cfg['fps']),
                                                            model=state_model_predictor,
                                                            team_idx=team)
        super().__init__(name, team, index)
        
    def predict(self, packet: GameTickPacket, controls: SimpleControllerState, n_steps: int) -> tp.List[WorldState]:
        data_provider = self.state_predictor_data.model.data_provider
        model = self.state_predictor_data.model
        
        world_state = WorldState.from_game_packet(packet)
        out_world_states = [world_state]
        action_state = ActionState.from_controller_state(controls)
        world_states_tensor = data_provider.world_state_to_tensor(world_state, agent_team_idx=self.team)
        action_index = data_provider.action_state_to_action_idx(action_state)
        
        world_states_tensor = world_states_tensor.view(1, 1, -1).to(model.device)
        action_indices_tensor = torch.tensor([action_index]).view(1, 1).to(model.device)
        
        t0 = time.time()
        # with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=torch.float16):
        with torch.no_grad():
            rnn_hidden = self.state_predictor_data.rnn_hidden
            next_world_state_tensor, rnn_hidden = model(batch_world_states_tensor=world_states_tensor,
                                                        batch_action_indices=action_indices_tensor,
                                                        prev_rnn_hidden=rnn_hidden,
                                                        return_rnn_hidden=True)
            self.state_predictor_data.rnn_hidden = rnn_hidden
            world_state = data_provider.tensor_to_world_state(next_world_state_tensor.flatten().cpu().detach(),
                                                              agent_team_idx=0)
            out_world_states.append(world_state)
            world_states_tensor =  next_world_state_tensor.unsqueeze(0)
            for _ in range(n_steps):
                next_world_state_tensor, rnn_hidden = model(batch_world_states_tensor=world_states_tensor,
                                                            batch_action_indices=action_indices_tensor,
                                                            prev_rnn_hidden=rnn_hidden,
                                                            return_rnn_hidden=True)
                world_state = data_provider.tensor_to_world_state(next_world_state_tensor.flatten().cpu().detach(),
                                                                  agent_team_idx=0)
                out_world_states.append(world_state)
                world_states_tensor =  next_world_state_tensor.unsqueeze(0)
        dt = time.time() - t0
        # print(f'dt={dt:.3f}')
                
        return out_world_states
        
    def draw_state_debug(self, world_states: tp.List[WorldState]):
        self.renderer.begin_rendering('Agent path')
        for cur_world_state, next_world_state in zip(world_states[:-1], world_states[1:]):
            self.renderer.draw_line_3d(cur_world_state.players[0][0].location,
                                       next_world_state.players[0][0].location,
                                       self.renderer.green())
            self.renderer.draw_line_3d(cur_world_state.players[1][0].location,
                                       next_world_state.players[1][0].location,
                                       self.renderer.red())
            self.renderer.draw_line_3d(cur_world_state.ball.location, next_world_state.ball.location,
                                       self.renderer.blue())
        self.renderer.end_rendering()
        
    def get_output(self, packet: GameTickPacket) -> SimpleControllerState:
        controls = super().get_output(packet)
        
        if not packet.game_info.is_round_active:
            self.state_predictor_data.last_time_tick = None
            self.state_predictor_data.last_world_states = []
            self.state_predictor_data.rnn_hidden = None
            return controls
        
        time_tick = packet.game_info.seconds_elapsed
        
        if self.state_predictor_data.last_time_tick is not None:
            if (time_tick - self.state_predictor_data.last_time_tick) < (1.0 / self.state_predictor_data.fps) * 0.9:
                if self.state_predictor_data.last_world_states:
                    self.draw_state_debug(self.state_predictor_data.last_world_states)
                return controls
            dt = time_tick - self.state_predictor_data.last_time_tick
            print(f'dt={dt:.3f}')
        
        self.state_predictor_data.last_world_states = self.predict(packet, controls, 10)
        self.draw_state_debug(self.state_predictor_data.last_world_states)
        self.state_predictor_data.last_time_tick = time_tick
        return controls
        