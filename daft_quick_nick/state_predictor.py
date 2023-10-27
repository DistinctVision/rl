import typing as tp
import functools
import math

import torch
import einops

from daft_quick_nick.game_data import ModelDataProvider, WorldState, ActionState
from daft_quick_nick.utils.constans import BALL_RADIUS


class RnnCoreModel(torch.nn.Module):
    
    def __init__(self,
                 in_size: int,
                 num_actions: int,
                 action_dim: int,
                 inner_dim: int,
                 hidden_dim: int,
                 n_lstm_layers: int):
        super().__init__()
        assert inner_dim >= (in_size + action_dim)
        self.in_proj = torch.nn.Sequential(torch.nn.Linear(in_size, inner_dim - action_dim),
                                           torch.nn.ReLU())
        self.action_embeddings = torch.nn.Embedding(num_actions, embedding_dim=action_dim)
        self.lstm = torch.nn.LSTM(inner_dim, hidden_dim,
                                  num_layers=n_lstm_layers, batch_first=True)
        
    @property
    def device(self) -> torch.device:
        return self.action_embeddings.weight.device
    
    def make_default_rnn_output(self, batch_size: int = 1) \
            -> tp.Tuple[torch.Tensor, tp.Tuple[torch.Tensor, torch.Tensor]]:
        device = self.device
        hidden_size = self.lstm.hidden_size
        n_layers = self.lstm.num_layers
        rnn_output = torch.zeros((batch_size, 1, hidden_size,), dtype=torch.float32, device=device)
        hidden_state = torch.zeros((n_layers, batch_size, hidden_size,), dtype=torch.float32, device=device)
        cell_state = torch.zeros((n_layers, batch_size, hidden_size,), dtype=torch.float32, device=device)
        return rnn_output, (hidden_state, cell_state)
    
    def flat_rnn_output(self, out: tp.Tuple[torch.Tensor, tp.Tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        rnn_output, (hidden_state, cell_state) = out
        batch_size = rnn_output.shape[0]
        assert rnn_output.shape == (batch_size, 1, self.lstm.hidden_size)
        assert hidden_state.shape == cell_state.shape == (self.lstm.num_layers, batch_size, self.lstm.hidden_size)
        out_batch = []
        for batch_item_idx in range(batch_size):
            batch_out_el = torch.cat([rnn_output[batch_item_idx, -1, :],
                                      hidden_state[:, batch_item_idx, :].flatten(),
                                      cell_state[:, batch_item_idx, :].flatten()])
            out_batch.append(batch_out_el)
        return torch.stack(out_batch)
    
    def unflat_rnn_output(self, flat_in: torch.Tensor) -> tp.Tuple[torch.Tensor, tp.Tuple[torch.Tensor, torch.Tensor]]:
        batch_size = flat_in.shape[0]
        hidden_size = self.lstm.hidden_size
        num_layers = self.lstm.num_layers
        rnn_outputs, hidden_states, cell_states = [], [], []
        for batch_item_idx in range(batch_size):
            in_vec = flat_in[batch_item_idx, :]
            vec_splits = (hidden_size, hidden_size+hidden_size*num_layers)
            rnn_output = in_vec[:vec_splits[0]]
            hidden_state = in_vec[vec_splits[0]:vec_splits[1]].view(num_layers, hidden_size)
            cell_state = in_vec[vec_splits[1]:].view(num_layers, hidden_size)
            rnn_outputs.append(rnn_output),
            hidden_states.append(hidden_state)
            cell_states.append(cell_state)
        return torch.stack(rnn_outputs, dim=0), (torch.stack(hidden_states, dim=1), torch.stack(cell_states, dim=1))
        
    def init_weights(self):
        torch.nn.init.normal_(self.action_embeddings.weight)
        torch.nn.init.xavier_uniform_(self.in_proj[0].weight)
        torch.nn.init.uniform_(self.in_proj[0].bias, a=-5e-3, b=5e-3)
        for layer_idx in range(self.lstm.num_layers):
            for weight in self.lstm._all_weights[layer_idx]:
                if "weight" in weight:
                    torch.nn.init.xavier_uniform_(getattr(self.lstm, weight))
                if "bias" in weight:
                    torch.nn.init.uniform_(getattr(self.lstm, weight), a=-5e-3, b=5e-3)
    
    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        
    def forward(self,
                batch_world_states_tensor: torch.Tensor,
                batch_action_indices: torch.Tensor,*,
                hidden: tp.Optional[tp.Tuple[torch.Tensor, torch.Tensor]] = None) -> tp.Tuple[torch.Tensor,
                                                                                              tp.Tuple[torch.Tensor,
                                                                                                       torch.Tensor]]:
        assert len(batch_world_states_tensor.shape) == 3 and len(batch_action_indices.shape) == 2
        assert batch_world_states_tensor.shape[0] == batch_action_indices.shape[0] and \
            batch_world_states_tensor.shape[1] == batch_action_indices.shape[1]
        
        action_states_tensors = self.action_embeddings(batch_action_indices)
        in_states = self.in_proj(batch_world_states_tensor)
        in_states = torch.cat([in_states, action_states_tensors], dim=-1)
        
        b, s, _ = in_states.shape
        
        device = in_states.device
        
        if hidden is None:
            hidden = (torch.zeros(self.lstm.num_layers, b, self.lstm.hidden_size, device=device),
                      torch.zeros(self.lstm.num_layers, b, self.lstm.hidden_size, device=device))
        
        rnn_out_seq = []
        for seq_idx in range(s):
            rnn_out, hidden = self.lstm(in_states[:, seq_idx:seq_idx+1, :], hidden)
            rnn_out_seq.append(rnn_out)
        
        rnn_out = torch.cat([r for r in rnn_out_seq], dim=-2)
        return rnn_out, hidden
        

class StatePredictorModel(torch.nn.Module):
    
    @staticmethod
    def build_model(model_state_cfg: tp.Dict[str, int],
                    data_provider: ModelDataProvider) -> 'StatePredictorModel':
        rnn_config = dict(model_state_cfg['rnn'])
        return StatePredictorModel(data_provider=data_provider,
                                   action_dim=int(rnn_config['action_dim']),
                                   inner_dim=int(rnn_config['inner_dim']),
                                   hidden_dim=int(rnn_config['hidden_dim']),
                                   n_lstm_layers=int(rnn_config['n_lstm_layers']))
    
    def __init__(self,
                 data_provider: ModelDataProvider,
                 action_dim: int,
                 inner_dim: int,
                 hidden_dim: int,
                 n_lstm_layers: int):
        super().__init__()
        self.data_provider = data_provider
        self.rnn_core = RnnCoreModel(data_provider.WORLD_STATE_SIZE, data_provider.num_actions,
                                     action_dim, inner_dim, hidden_dim, n_lstm_layers)
        v_state_size = hidden_dim
        self.out_projector = torch.nn.Sequential(torch.nn.Linear(v_state_size, v_state_size * 2),
                                                 torch.nn.ReLU(),
                                                 torch.nn.Linear(v_state_size * 2,
                                                                 ModelDataProvider.WORLD_STATE_SIZE))
        cat_flags = ModelDataProvider.world_state_categorial_flags()
        cat_feat_indices = torch.tensor([idx for idx, flag in enumerate(cat_flags) if flag], dtype=torch.long)
        cat_feat_mask = torch.tensor(cat_flags, dtype=torch.bool)
        num_feat_indices = torch.tensor([idx for idx, flag in enumerate(cat_flags) if not flag], dtype=torch.long)
        num_feat_mask = ~cat_feat_mask
        self.register_buffer('cat_feat_indices', cat_feat_indices)
        # self.register_buffer('cat_feat_mask', cat_feat_mask)
        self.register_buffer('num_feat_indices', num_feat_indices)
        # self.register_buffer('num_feat_mask', num_feat_mask)
        self.cat_feat_mask = cat_feat_mask
        self.num_feat_mask = num_feat_mask
        # for linter
        self.cat_feat_indices: torch.Tensor = self.cat_feat_indices
        self.num_feat_indices: torch.Tensor = self.num_feat_indices
    
    @property
    def device(self) -> torch.device:
        return self.rnn_core.device
    
    def init_weights(self):
        self.rnn_core.init_weights()
        
        torch.nn.init.xavier_uniform_(self.out_projector[0].weight)
        torch.nn.init.uniform_(self.out_projector[0].bias, a=-5e-3, b=5e-3)
        torch.nn.init.xavier_uniform_(self.out_projector[2].weight)
        torch.nn.init.uniform_(self.out_projector[2].bias, a=-5e-3, b=5e-3)
    
    def forward(self,
                batch_world_states_tensor: tp.Optional[torch.Tensor],
                batch_action_indices: tp.Optional[torch.Tensor],
                return_losses: bool = False,
                prev_rnn_hidden: tp.Optional[tp.Tuple[torch.Tensor, torch.Tensor]] = None,
                return_rnn_hidden: bool = False) -> tp.Union[torch.Tensor,
                                                             tp.Tuple[torch.Tensor,  torch.Tensor],
                                                             tp.Dict[str, tp.Union[torch.Tensor, float]]]:
        
        if return_losses:
            target_world_states_tensor = batch_world_states_tensor[:, 1:, :].detach()
            batch_world_states_tensor = batch_world_states_tensor[:, :-1, :]
        
        assert len(batch_world_states_tensor.shape) == 3 and len(batch_action_indices.shape) == 2
        assert batch_world_states_tensor.shape[0] == batch_action_indices.shape[0] and \
            batch_world_states_tensor.shape[1] == batch_action_indices.shape[1]
        
        rnn_output, (hidden_state, cell_state) = self.rnn_core(batch_world_states_tensor, batch_action_indices,
                                                               hidden=prev_rnn_hidden)
        output: torch.Tensor = self.out_projector(rnn_output)
        cat_feat_mask = einops.repeat(self.cat_feat_mask, 'd -> b s d', b=rnn_output.shape[0], s=rnn_output.shape[1])
        
        if not return_losses:
            output[cat_feat_mask] = torch.sigmoid(output[cat_feat_mask])
            output = output[:, -1, :]
            
            if return_rnn_hidden:
                return output, (hidden_state, cell_state)
            return output
        
        pr_num_features = output.index_select(dim=2, index=self.num_feat_indices)
        target_num_features = target_world_states_tensor.index_select(dim=2, index=self.num_feat_indices)
        num_losses = torch.nn.functional.mse_loss(pr_num_features, target_num_features, reduction='none')
        
        pr_cat_features = output.index_select(dim=2, index=self.cat_feat_indices)
        target_cat_features = target_world_states_tensor.index_select(dim=2, index=self.cat_feat_indices)
        cat_losses = torch.nn.functional.binary_cross_entropy_with_logits(pr_cat_features, target_cat_features,
                                                                          reduction='none')
        
        def get_vec_error(vec: torch.Tensor) -> float:
            errors = torch.sqrt(torch.sum(vec, dim=2).detach().cpu())
            return float(errors.mean())
        
        def get_cat_error(batch_v: torch.Tensor) -> float:
            return float(batch_v.mean().detach().cpu())
        
        pos_scale = self.data_provider.data_scaler.pos_std
        angle_scale = self.data_provider.data_scaler.angle_std
        
        with torch.no_grad():
            losses = {
                'ball/location': get_vec_error(num_losses[:, :, 0:3]),
                'ball/velocity': get_vec_error(num_losses[:, :, 3:6]),
                'agent/location': get_vec_error(num_losses[:, :, 6:9]),
                'agent/q': get_vec_error(num_losses[:, :, 9:13]),
                'agent/velocity': get_vec_error(num_losses[:, :, 13:16]),
                'agent/angular_velocity': get_vec_error(num_losses[:, :, 16:19]),
                'agent/boost': get_vec_error(num_losses[:, :, 19:20]),
                'player/location': get_vec_error(num_losses[:, :, 20:23]),
                'player/q': get_vec_error(num_losses[:, :, 23:27]),
                'player/velocity': get_vec_error(num_losses[:, :, 27:30]),
                'player/angular_velocity': get_vec_error(num_losses[:, :, 30:33]),
                'player/boost': get_vec_error(num_losses[:, :, 33:34]),
                'dist_to_ball': get_vec_error(num_losses[:, :, 34:35]),
                'agent/has_wheel_contact': get_cat_error(cat_losses[:, :, 0]),
                'agent/is_super_sonic': get_cat_error(cat_losses[:, :, 1]),
                'agent/jumped': get_cat_error(cat_losses[:, :, 2]),
                'agent/double_jumped': get_cat_error(cat_losses[:, :, 3]),
                'player/has_wheel_contact': get_cat_error(cat_losses[:, :, 4]),
                'player/is_super_sonic': get_cat_error(cat_losses[:, :, 5]),
                'player/jumped': get_cat_error(cat_losses[:, :, 6]),
                'player/double_jumped': get_cat_error(cat_losses[:, :, 7]),
            }
            
        ball_radius_sq = BALL_RADIUS * BALL_RADIUS
        
        num_losses[:, :, 0:9] /= ball_radius_sq
        num_losses[:, :, 9:13] *= 10
        num_losses[:, :, 13:16] /= ball_radius_sq
        num_losses[:, :, 16:19] /= ball_radius_sq
        num_losses[:, :, 19:20] /= 100.0
        num_losses[:, :, 20:23] /= ball_radius_sq
        num_losses[:, :, 23:27] *= 10
        num_losses[:, :, 27:30] /= ball_radius_sq
        num_losses[:, :, 30:33] /= ball_radius_sq
        num_losses[:, :, 33:34] /= 100.0
        num_losses[:, :, 34:35] /= ball_radius_sq
        
        # num_losses[:, :, 9:13] *= 0.1
        # num_losses[:, :, 23:27] *= 0.1
        
        num_loss = num_losses.mean()
        cat_loss = cat_losses.mean() * 0.05
            
        losses['num_loss'] = num_loss
        losses['cat_loss'] = cat_loss
        
        return losses
