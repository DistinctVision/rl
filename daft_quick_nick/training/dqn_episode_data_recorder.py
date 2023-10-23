import numpy as np
import torch

from daft_quick_nick.training.dqn_trainer import DqnTrainer
from daft_quick_nick.training.rp_data_types import RP_SeqOfRecords
        

class DqnEpisodeDataRecorder:
    
    def __init__(self, trainer: DqnTrainer):
        self.trainer = trainer
        self.current_episode = RP_SeqOfRecords()
    
    def __len__(self) -> int:
        return len(self.current_episode)
    
    def get_action(self, world_state_tensor: torch.Tensor) -> int:
        return self._get_action_idx(world_state_tensor)
    
    def record(self,
               world_state_tensor: torch.Tensor,
               action: int, reward: float,
               episode_is_done: bool):
        self.current_episode.add(world_state_tensor.clone(), action, reward)
        
        if episode_is_done:
            self.trainer.replay_buffer.add_episode(self.current_episode)
            self.current_episode = RP_SeqOfRecords()
            max_buffer_size = int(self.trainer.cfg['replay_buffer']['max_buffer_size'])
            while len(self.trainer.replay_buffer) > max_buffer_size:
                self.trainer.replay_buffer.remove_first_episode()
    
    def _get_action_idx(self, world_state_tensor: torch.Tensor) -> int:
        eps_greedy_coeff = self.trainer._get_eps_greedy_coeff()
        if np.random.uniform(0, 1) < eps_greedy_coeff:
            action_idx = np.random.choice(self.trainer.action_set)
        else:
            model = self.trainer.model
            with torch.no_grad():
                pr_rewards: torch.Tensor = model(
                    batch_world_states_tensor=world_state_tensor.unsqueeze(0).to(model.device))
                pr_rewards = pr_rewards.squeeze(0).cpu()
            for action_idx in range(pr_rewards.shape[0]):
                if action_idx not in self.trainer.action_set:
                    pr_rewards[action_idx] = -1000000
            action_idx = pr_rewards.argmax(0)
        return int(action_idx)
