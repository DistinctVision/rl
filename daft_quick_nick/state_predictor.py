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
        
        hidden_seq = []
        rnn_out_seq = []
        
        for seq_idx in range(s):
            rnn_out, hidden = self.lstm(in_states[:, seq_idx:seq_idx+1, :], hidden)
            hidden_seq.append(hidden)
            rnn_out_seq.append(rnn_out)
        
        hidden = (torch.stack([h[0] for h in hidden_seq], dim=-2), torch.stack([h[1] for h in hidden_seq], dim=-2))
        rnn_out = torch.cat([r for r in rnn_out_seq], dim=-2)
        return rnn_out, hidden
        

class StatePredictorModel(torch.nn.Module):
    
    @staticmethod
    def build_model(model_state_cfg: tp.Dict[str, int],
                    data_provider: ModelDataProvider) -> 'StatePredictorModel':
        return StatePredictorModel(data_provider=data_provider,
                                   action_dim=int(model_state_cfg['action_dim']),
                                   inner_dim=int(model_state_cfg['inner_dim']),
                                   hidden_dim=int(model_state_cfg['hidden_dim']),
                                   n_lstm_layers=int(model_state_cfg['n_lstm_layers']))
    
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
        v_state_size = inner_dim + hidden_dim * n_lstm_layers
        self.out_projector = torch.nn.Sequential(torch.nn.Linear(v_state_size, v_state_size * 4),
                                                 torch.nn.ReLU(),
                                                 torch.nn.Linear(v_state_size * 4,
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
            target_world_states_tensor = batch_world_states_tensor[:, 1:, :]
            batch_world_states_tensor = batch_world_states_tensor[:, :-1, :]
        
        assert len(batch_world_states_tensor.shape) == 3 and len(batch_action_indices.shape) == 2
        assert batch_world_states_tensor.shape[0] == batch_action_indices.shape[0] and \
            batch_world_states_tensor.shape[1] == batch_action_indices.shape[1]
        
        rnn_output, (hidden_state, cell_state) = self.rnn_core(batch_world_states_tensor, batch_action_indices,
                                                               hidden=prev_rnn_hidden)
        rnn_output = torch.cat([rnn_output] + [hidden_state[idx, :, :, :] for idx in range(hidden_state.shape[0])],
                               dim=-1)
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
                'ball/location': get_vec_error(num_losses[:, :, 0:3]) * pos_scale,
                'ball/velocity': get_vec_error(num_losses[:, :, 3:6]) * pos_scale,
                'agent/location': get_vec_error(num_losses[:, :, 6:9]) * pos_scale,
                'agent/q': get_vec_error(num_losses[:, :, 9:13]),
                'agent/velocity': get_vec_error(num_losses[:, :, 13:16]) * pos_scale,
                'agent/angular_velocity': get_vec_error(num_losses[:, :, 16:19]) * angle_scale,
                'agent/boost': get_vec_error(num_losses[:, :, 19:20]) * 100,
                'player/location': get_vec_error(num_losses[:, :, 20:23]) * pos_scale,
                'player/q': get_vec_error(num_losses[:, :, 23:27]),
                'player/velocity': get_vec_error(num_losses[:, :, 27:30]) * pos_scale,
                'player/angular_velocity': get_vec_error(num_losses[:, :, 30:33]) * angle_scale,
                'player/boost': get_vec_error(num_losses[:, :, 33:34]) * 100,
                'dist_to_ball': get_vec_error(num_losses[:, :, 34:35]) * pos_scale,
                'agent/has_wheel_contact': get_cat_error(cat_losses[:, :, 0]),
                'agent/is_super_sonic': get_cat_error(cat_losses[:, :, 1]),
                'agent/jumped': get_cat_error(cat_losses[:, :, 2]),
                'agent/double_jumped': get_cat_error(cat_losses[:, :, 3]),
                'player/has_wheel_contact': get_cat_error(cat_losses[:, :, 4]),
                'player/is_super_sonic': get_cat_error(cat_losses[:, :, 5]),
                'player/jumped': get_cat_error(cat_losses[:, :, 6]),
                'player/double_jumped': get_cat_error(cat_losses[:, :, 7]),
            }
        
        num_loss = num_losses.mean()
        cat_loss = cat_losses.mean()
            
        losses['num_loss'] = num_loss
        losses['cat_loss'] = cat_loss
        
        return losses
