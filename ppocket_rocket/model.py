import typing as tp
import functools
import torch

from ppocket_rocket.game_data import ModelDataProvider
from ppocket_rocket.state_predictor import RnnCoreModel, StatePredictorModel


def get_model_num_params(model: torch.nn.Module) -> str:
    """
    Return the number of model parameters in text format 
    """
    n_params = sum(p.numel() for p in model.parameters())
    
    stage_postfixes = ['', ' M', ' B']
    stage_numbers = [1, 10 ** 6, 10 ** 9]
    
    stage_idx = 0
    for idx, stage_n in enumerate(stage_numbers):
        if n_params > stage_n:
            stage_idx = idx
    
    if stage_idx > 0:
        n_params /= stage_numbers[stage_idx]
        n_params_str = f'{n_params:.2f}'
    else:
        n_params_str = str(n_params)
    return f'{n_params_str}{stage_postfixes[stage_idx]}'

    
class CriticModel(torch.nn.Module):
    
    @staticmethod
    def build_model(data_provider: ModelDataProvider,
                    model_cfg: tp.Dict[str, tp.Union[int, float]],
                    without_backbone: bool = False) -> 'CriticModel':
        in_size = data_provider.WORLD_STATE_SIZE
        has_backbone = False
        if 'rnn' in model_cfg:
            rnn_config = dict(model_cfg['rnn'])
            if without_backbone:
                backbone = None
            else:
                state_predictor = StatePredictorModel.build_model(model_cfg, data_provider)
                ckpt = torch.load(str(rnn_config['state_predictor_path']))
                state_predictor.load_state_dict(ckpt)
                backbone = state_predictor.rnn_core
                backbone.freeze()
            hidden_dim = int(rnn_config['hidden_dim'])
            n_lstm_layers = int(rnn_config['n_lstm_layers'])
            in_size += hidden_dim * (n_lstm_layers * 2 + 1)
            has_backbone = True
        else:
            backbone = None
        return CriticModel(has_backbone, backbone,
                           in_size=in_size,
                           out_size=data_provider.num_actions,
                           reward_decay=float(model_cfg['reward_decay']),
                           layers=[int(layer) for layer in list(model_cfg['layers'])],
                           dropout=float(model_cfg['dropout']))
    
    def __init__(self,
                 has_backbone: bool,
                 backbone: tp.Optional[RnnCoreModel],
                 in_size: int,
                 out_size: int,
                 reward_decay: float,
                 layers: tp.List[int],
                 dropout: float = 0.1):
        super().__init__()
        self.reward_decay = reward_decay
        
        self.has_backbone = has_backbone
        self.backbone = backbone
        self.in_proj = torch.nn.Sequential(torch.nn.Linear(in_size, layers[0]), torch.nn.ReLU())
        
        block_layers = []
        for layer_in, layar_out in zip(layers[:-1], layers[1:]):
            block_layers.append(torch.nn.Sequential(torch.nn.Dropout(dropout),
                                                    torch.nn.Linear(layer_in, layar_out),
                                                    torch.nn.ReLU()))
        self.blocks = torch.nn.Sequential(*block_layers)
        self.out_proj = torch.nn.Linear(layers[-1], out_size, bias=True)
    
    @property
    def device(self) -> torch.device:
        return self.out_proj.weight.device
        
    def init_weights(self):
        torch.nn.init.kaiming_uniform_(self.in_proj[0].weight)
        torch.nn.init.uniform_(self.in_proj[0].bias, a=-5e-3, b=5e-3)
        
        def _init_fn(depth: int, module: torch.nn.Module):
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.uniform_(module.bias, a=-5e-3, b=5e-3)
                
        for depth, block in enumerate(self.blocks):
            block.apply(functools.partial(_init_fn, depth))
        
        torch.nn.init.kaiming_uniform_(self.out_proj.weight)
        torch.nn.init.uniform_(self.out_proj.bias, a=-5e-3, b=5e-3)
    
    def forward(self, 
                batch_world_states_tensor: torch.Tensor, *,
                batch_action_indices: tp.Optional[torch.Tensor] = None,
                prev_rnn_output: tp.Optional[tp.Tuple[torch.Tensor,
                                                      torch.Tensor,
                                                      torch.Tensor]] = None) -> torch.Tensor:
        
        if self.has_backbone and prev_rnn_output is None:
            assert self.backbone is not None
            assert batch_action_indices is not None
            prev_rnn_output = self.backbone(batch_world_states_tensor[:, :-1, :],
                                            batch_action_indices[:, :])
        
        if self.has_backbone:
            input_vecs = [batch_world_states_tensor[:, -1, :], prev_rnn_output[0][:, -1, :]]
            for lstm_layer_idx in range(prev_rnn_output[1][0].shape[0]):
                input_vecs += [prev_rnn_output[1][0][lstm_layer_idx, :, :],
                               prev_rnn_output[1][1][lstm_layer_idx, :, :]]
            in_state = torch.cat(input_vecs, dim=-1)
        else:
            in_state = batch_world_states_tensor[:, -1, :]
        
        x: torch.Tensor = self.in_proj(in_state)
        x = self.blocks(x)
        x = self.out_proj(x)
        return x
