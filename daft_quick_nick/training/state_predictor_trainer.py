import typing as tp

from pathlib import Path
import logging
import yaml
from tqdm import tqdm
from contextlib import nullcontext

import numpy as np
import torch
from tqdm import tqdm

from daft_quick_nick.training.log_writer import LogWriter, get_run_name, make_output_folder, BatchValueList
from daft_quick_nick.training.replay_buffer import ReplayBuffer
from daft_quick_nick.game_data import ModelDataProvider
from daft_quick_nick.state_predictor import StatePredictorModel
from daft_quick_nick.model import get_model_num_params
from daft_quick_nick.training.replay_buffer_batch_sampler import ReplayBufferBatchSampler


class StatePredictorTrainer:

    def __init__(self,
                 cfg: tp.Dict[str, tp.Any],
                 train_replay_buffer: ReplayBuffer,
                 val_replay_buffer: ReplayBuffer):
        self.cfg = cfg
        self.logger = logging.getLogger(__name__)
        self.device = torch.device('cuda:0')

        self.data_provider = ModelDataProvider()

        self.train_replay_buffer = train_replay_buffer
        self.val_replay_buffer = val_replay_buffer
        
        self.model: tp.Optional[StatePredictorModel] = None
        self.optimizer: tp.Optional[torch.optim.Optimizer] = None

        self.log_writer: tp.Optional[LogWriter] = None

        self._init_model()
        self._init_log()
        

    def _init_model(self):
        model_cfg = dict(self.cfg['model'])
        rnn_config = dict(model_cfg['rnn'])
        training_cfg = dict(self.cfg['state_predictor_training'])
        
        self.model = StatePredictorModel.build_model(rnn_config, self.data_provider)
        if 'state_model_path' in rnn_config:
            ckpt = torch.load(str(rnn_config['state_model_path']), map_location='cpu')
            self.model.load_state_dict(ckpt)
        else:
            self.model.init_weights()
            
        self.model = self.model.to(self.device).train()
        
        self.logger.info(f'A size of the model: {get_model_num_params(self.model)}')
        print(f'A size of the model: {get_model_num_params(self.model)}')
        
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=float(training_cfg['lr']),
                                          betas=(0.9, 0.999), eps=1e-8)
        optimizer_path = model_cfg.get('critic_optimizer_path', None)
        if optimizer_path is not None:
            optimizer_path = Path(optimizer_path)
            optimizer_ckpt = torch.load(optimizer_path)
            self.optimizer.load_state_dict(optimizer_ckpt)
            self.logger.info(f'Optimizer is loaded from "{optimizer_path}"')


    def _init_log(self):
        training_cfg = dict(self.cfg['state_predictor_training'])
        save_cfg = dict(training_cfg['save'])

        run_name = get_run_name(f'{str(save_cfg["model_name"])}_%dt')
        run_output_folder = make_output_folder(training_cfg['output_folder'], run_name, False)
        
        logging.basicConfig(filename=run_output_folder / 'log.txt',
                            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                            datefmt='%d-%b-%y %H:%M:%S',
                            level=logging.INFO)
        self.cfg['meta'] = {'run_name': run_name}
        with open(run_output_folder / 'cfg.yaml', 'w') as cfg_file:
            yaml.safe_dump(self.cfg, cfg_file)
        output_weights_folder: Path = run_output_folder / 'weights'
        output_weights_folder.mkdir()
        self.log_writer = LogWriter(output_plot_folder=run_output_folder,
                                    project_name='RL', run_name=run_name,
                                    output_weights_folder=output_weights_folder,
                                    save_cfg=save_cfg,
                                    subsets=['train', 'val'])
    
    
    def train_epoch(self):
        
        training_cfg: dict = self.cfg['state_predictor_training']
        
        n_grad_accum_steps: int = training_cfg.get('n_grad_accum_steps', 1)
        batch_size: int = training_cfg['batch_size']
        grad_norm: tp.Optional[float] = training_cfg.get('grad_norm', None)
        
        if bool(training_cfg['fp16']):
            precision_ctx = torch.amp.autocast(device_type='cuda', dtype=torch.float16)
            grad_scaler = torch.cuda.amp.GradScaler(enabled=True)
        else:
            precision_ctx = nullcontext()
            grad_scaler = torch.cuda.amp.GradScaler(enabled=False)
        
        sequence_size = int(self.cfg['model']['memory_size'])
        
        train_epoch_data_size = int(training_cfg['train_size']['for_epoch'])
        val_epoch_data_size = int(training_cfg['val_size']['for_epoch'])
        
        train_indices = self.train_replay_buffer.make_seq_indices(sequence_size + 1)
        np.random.shuffle(train_indices)
        train_indices = train_indices[:train_epoch_data_size]
        val_indices = self.val_replay_buffer.make_seq_indices(sequence_size + 1)
        np.random.shuffle(val_indices)
        val_indices = val_indices[:val_epoch_data_size]
        
        train_batch_sampler = ReplayBufferBatchSampler(batch_size, train_indices)
        val_batch_sampler = ReplayBufferBatchSampler(batch_size, val_indices)
        
        batch_value_list = self.log_writer.make_batch_value_list()
        grad_accum_counter = 0
        
        self.model.train()
        for item_idx, batch_indices in enumerate(tqdm(train_batch_sampler, desc='Training')):
            sync_grad = (grad_accum_counter + 1) >= n_grad_accum_steps or (item_idx + 1) >= len(train_batch_sampler)
            
            batch = self.train_replay_buffer.get_seq_batch(batch_indices, sequence_size + 1)
            batch_world_states_tensor = batch.world_states.to(self.device)
            batch_action_indices = batch.action_indices[:, :sequence_size].long().to(self.device)
            
            with precision_ctx:
                losses = self.model(batch_world_states_tensor=batch_world_states_tensor,
                                    batch_action_indices=batch_action_indices,
                                    return_losses=True)
                loss: torch.Tensor = losses['num_loss'] + losses['cat_loss']
                
                loss /= n_grad_accum_steps
                grad_scaler.scale(loss).backward()
            
            if sync_grad:
                if grad_norm is not None:
                    grad_scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=grad_norm)
                grad_scaler.step(self.optimizer)
                grad_scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                grad_accum_counter = 0
            else:
                grad_accum_counter += 1
            
            for metric_name, metric_value in losses.items():
                if isinstance(metric_value, torch.Tensor):
                    metric_value = float(metric_value.detach().cpu())
                batch_value_list.add(metric_value, metric_name, 'train')
        
        self.model.eval()
        with torch.no_grad():
            for item_idx, batch_indices in enumerate(tqdm(val_batch_sampler, desc='Validation')):
                batch = self.val_replay_buffer.get_seq_batch(batch_indices, sequence_size + 1)
                batch_world_states_tensor = batch.world_states.to(self.device)
                batch_action_indices = batch.action_indices[:, :sequence_size].long().to(self.device)
                
                with precision_ctx:
                    losses = self.model(batch_world_states_tensor=batch_world_states_tensor,
                                        batch_action_indices=batch_action_indices,
                                        return_losses=True)
                
                for metric_name, metric_value in losses.items():
                    if isinstance(metric_value, torch.Tensor):
                        metric_value = float(metric_value.detach().cpu())
                    batch_value_list.add(metric_value, metric_name, 'val')
        
        self.log_writer.add_batch_values(batch_value_list)
        loss_values = batch_value_list.get_values()
        for subset_name, subset_values in loss_values.items():
            self.logger.info(f'Subset [{subset_name}]:')
            for metric_name, metric_value in subset_values.items():
                self.logger.info(f'  {metric_name}: {metric_value}')
        self.logger.info(' ')
        self.log_writer.save_plots()
        self.log_writer.save_weights(self.model, self.optimizer)
        self.log_writer.update_step()
        