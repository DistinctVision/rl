import typing as tp

import numpy as np
import torch

from ppocket_rocket.training.replay_buffer import ReplayBuffer
from ppocket_rocket.training.rp_data_types import RP_SeqOfRecords
        

class StatePredictorEpisodeDataRecorder:
    
    def __init__(self,
                 train_replay_buffer: ReplayBuffer,
                 max_train_replay_buffer_size: int,
                 val_replay_buffer: ReplayBuffer,
                 max_val_replay_buffer_size: int):
        self.train_replay_buffer = train_replay_buffer
        self.val_replay_buffer = val_replay_buffer
        self.current_episode = RP_SeqOfRecords()
        self.max_train_replay_buffer_size = max_train_replay_buffer_size
        self.max_val_replay_buffer_size = max_val_replay_buffer_size
        all_data_size = max_val_replay_buffer_size + max_train_replay_buffer_size
        self.target_train_ratio = max_train_replay_buffer_size / all_data_size
    
    def __len__(self) -> int:
        return len(self.current_episode)
    
    def record(self,
               world_state_tensor: torch.Tensor,
               action: int, reward: tp.Optional[float],
               episode_is_done: bool):
        self.current_episode.add(world_state_tensor.clone(), action, reward)
        
        if episode_is_done:
            if np.random.uniform(0, 1) < self.target_train_ratio:
                self.train_replay_buffer.add_episode(self.current_episode)
                while len(self.train_replay_buffer) > self.max_train_replay_buffer_size:
                    self.train_replay_buffer.remove_first_episode()
            else:
                self.val_replay_buffer.add_episode(self.current_episode)
                while len(self.val_replay_buffer) > self.max_val_replay_buffer_size:
                    self.val_replay_buffer.remove_first_episode()
            
            self.current_episode = RP_SeqOfRecords()
