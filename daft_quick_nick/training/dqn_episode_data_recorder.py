import typing as tp
import numpy as np
import torch

from daft_quick_nick.training.dqn_trainer import DqnTrainer
from daft_quick_nick.training.rp_data_types import RP_SeqOfRecords
from daft_quick_nick.state_predictor import RnnCoreModel
        

class DqnEpisodeDataRecorder:
    
    def __init__(self, trainer: DqnTrainer, rnn_backbone: tp.Optional[RnnCoreModel] = None):
        self.trainer = trainer
        self.current_episode = RP_SeqOfRecords()
        self._rnn_backbone = rnn_backbone
        if self._rnn_backbone is not None:
            self._def_rnn_output = self._rnn_backbone.make_default_rnn_output()
        else:
            self._def_rnn_output = None
        self._prev_rnn_output = self._def_rnn_output
    
    def __len__(self) -> int:
        return len(self.current_episode)
    
    def get_action(self, world_state_tensor: torch.Tensor) -> int:
        eps_greedy_coeff = self.trainer._get_eps_greedy_coeff()
        if np.random.uniform(0, 1) < eps_greedy_coeff:
            action_idx = np.random.choice(self.trainer.action_set)
        else:
            model = self.trainer.model
            with torch.no_grad():
                pr_rewards: torch.Tensor = model(
                    batch_world_states_tensor=world_state_tensor.view(1, 1, -1).to(model.device),
                    prev_rnn_output=self._prev_rnn_output)
                pr_rewards = pr_rewards.squeeze(0).cpu()
            for action_idx in range(pr_rewards.shape[0]):
                if action_idx not in self.trainer.action_set:
                    pr_rewards[action_idx] = -1000000
            action_idx = pr_rewards.argmax(0)
        return int(action_idx)
    
    def record(self,
               world_state_tensor: torch.Tensor,
               action: int,
               reward: float,
               episode_is_done: bool):
        if self._rnn_backbone:
            device = self._rnn_backbone.device
            prev_rnn_output_vec = self._rnn_backbone.flat_rnn_output(self._prev_rnn_output).cpu()
            prev_rnn_output_vec.squeeze_(0)
            self.current_episode.add(world_state_tensor.clone(), action, reward, prev_rnn_output_vec)
            world_state_tensor = world_state_tensor.view(1, 1, -1).to(device)
            action_indices = torch.tensor([action], dtype=torch.long, device=device).view(1, 1)
            with torch.no_grad():
                rnn_output = self._rnn_backbone(world_state_tensor, action_indices,
                                                hidden=self._prev_rnn_output[1])
            self._prev_rnn_output = (rnn_output[0].detach(),
                                     (rnn_output[1][0].detach(), rnn_output[1][1].detach()))
        else:
            self.current_episode.add(world_state_tensor.clone(), action, reward)
        
        if episode_is_done:
            self.trainer.replay_buffer.add_episode(self.current_episode)
            self.current_episode = RP_SeqOfRecords()
            self._prev_rnn_output = self._def_rnn_output
            max_buffer_size = int(self.trainer.cfg['replay_buffer']['max_buffer_size'])
            while len(self.trainer.replay_buffer) > max_buffer_size:
                self.trainer.replay_buffer.remove_first_episode()
