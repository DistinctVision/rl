import typing as tp
from pathlib import Path
from collections import deque
import random

from threading import RLock

from tqdm import tqdm

import torch

from ppocket_rocket.training.rp_data_types import RP_Record, RP_RecordArray, RP_SeqOfRecords


class ReplayBuffer:
    
    @staticmethod
    def load(folder_path: tp.Union[Path, str], progress_desc: str = 'Loading') -> 'ReplayBuffer':
        rp = ReplayBuffer()
        
        episode_paths: tp.List[Path] = []
        for ep_folder_path in folder_path.iterdir():
            if not ep_folder_path.is_dir():
                continue
            episode_paths.append(ep_folder_path)
        
        for ep_folder_path in tqdm(episode_paths,  desc=progress_desc):
            world_states = torch.load(ep_folder_path / 'world_states.pth')
            action_indices = torch.load(ep_folder_path / 'action_indices.pth')
            rewards_filepath = ep_folder_path / 'rewards.pth'
            rewards = torch.load(rewards_filepath) if rewards_filepath.exists() else None
            rnn_outputs_filepath = ep_folder_path / 'rnn_outputs.pth'
            rnn_outputs = torch.load(rnn_outputs_filepath) if rnn_outputs_filepath.exists() else None
            episode = RP_RecordArray(world_states, action_indices, rewards, rnn_outputs)
            rp.buffer.append(episode)
        return rp
    
    def __init__(self):
        self.buffer: tp.Deque[RP_RecordArray] = deque()
    
    def __len__(self) -> int:
        return sum([len(session) for session in self.buffer])
    
    def make_indices(self) -> tp.List[tp.Tuple[int, int]]:
        indices = []
        for session_idx, session in enumerate(self.buffer):
            for frame_idx in range(len(session)):
                indices.append((session_idx, frame_idx))
        return indices
    
    @property
    def num_episodes(self) -> int:
        return len(self.buffer)
    
    def num_records(self, episode_index: int) -> int:
        return len(self.buffer[episode_index])
    
    def get_record(self, episode_index: int, frame_index: int) -> RP_Record:
        episode = self.buffer[episode_index]
        return RP_Record(episode.world_states[frame_index,  :],
                         int(episode.action_indices[frame_index]),
                         float(episode.rewards[frame_index]) if episode.rewards is not None else None,
                         episode.prev_rnn_outputs[frame_index, :] if episode.prev_rnn_outputs is not None else None)

    def add_episode(self, episode: RP_SeqOfRecords):
        self.buffer.append(RP_RecordArray.from_seq(episode))
        
    def clear(self):
        self.buffer = deque()
        
    def remove_first_episode(self):
        del self.buffer[0]
        
    def save(self, folder_path: tp.Union[Path, str], progress_desc: str = 'Saving'):
        folder_path = Path(folder_path)
        if not folder_path.exists():
            folder_path.mkdir(parents=True)
        for episode_idx, episode in enumerate(tqdm(self.buffer, desc=progress_desc)):
            episode_folder = folder_path / f'ep_{episode_idx}'
            episode_folder.mkdir()
            episode = tp.cast(RP_RecordArray, episode)
            torch.save(episode.world_states, episode_folder / 'world_states.pth')
            torch.save(episode.action_indices, episode_folder / 'action_indices.pth')
            if episode.rewards is not None:
                torch.save(episode.rewards, episode_folder / 'rewards.pth')
            
    def sample_batch_indices(self, batch_size: int) -> tp.List[int]:
        batch_indices = []
        for _ in range(batch_size):
            rand_ep_idx = random.randint(0, self.num_episodes - 1)
            rand_record_idx = random.randint(0, self.num_records(rand_ep_idx) - 1)
            batch_indices.append((rand_ep_idx, rand_record_idx))
        return batch_indices
    
    def get_batch(self, batch_indices: tp.List[tp.Tuple[int,  int]]) \
                -> tp.Tuple[RP_RecordArray, RP_RecordArray,  torch.Tensor]:
        cur_records: tp.List[RP_Record] = [self.get_record(idx[0], idx[1]) for idx in batch_indices]
        has_rewards = cur_records[0].reward is not None
        has_prev_rnn_outputs = cur_records[0].prev_rnn_output is not None
        cur_batch = RP_RecordArray(world_states=torch.stack([record.world_state_tensor
                                                            for record in cur_records]),
                                   action_indices=torch.tensor([record.action_index for record in cur_records],
                                                               dtype=torch.int16),
                                   rewards=torch.tensor([record.reward for record in cur_records],
                                                        dtype=torch.float32) if has_rewards else None,
                                   prev_rnn_outputs=torch.stack([record.prev_rnn_output
                                                                 for record in cur_records]) \
                                                    if has_prev_rnn_outputs else None)
        
        next_records: tp.List[RP_Record] = []
        mask_done: tp.List[bool] = []
        for r_idx,  idx in enumerate(batch_indices):
            if (idx[1] + 1) >= self.num_records(idx[0]):
                next_records.append(cur_records[r_idx])
                mask_done.append(True)
                continue
            next_records.append(self.get_record(idx[0], idx[1] + 1))
            mask_done.append(False)
        
        next_batch = RP_RecordArray(world_states=torch.stack([record.world_state_tensor
                                                              for record in next_records]),
                                    action_indices=torch.tensor([record.action_index
                                                                 for record in next_records],
                                                                dtype=torch.int16),
                                    rewards=torch.tensor([record.reward for record in next_records],
                                                         dtype=torch.float32) if has_rewards else None,
                                    prev_rnn_outputs=torch.stack([record.prev_rnn_output
                                                                  for record in next_records]) \
                                                     if has_prev_rnn_outputs else None)
        mask_done = torch.tensor(mask_done, dtype=torch.bool)
        return cur_batch, next_batch, mask_done
    
    def sample_seq_batch_indices(self, batch_size: int, sequence_size: int) -> tp.List[int]:
        batch_indices = []
        for _ in range(batch_size):
            rand_ep_idx = random.randint(0, self.num_episodes - 1)
            rand_record_idx = random.randint(0, self.num_records(rand_ep_idx) - sequence_size)
            batch_indices.append((rand_ep_idx, rand_record_idx))
        return batch_indices
    
    def get_seq_batch(self, batch_indices: tp.List[tp.Tuple[int,  int]], sequence_size: int) -> RP_RecordArray:
        list_batch: tp.List[RP_RecordArray] = []
        for idx in batch_indices:
            episode: RP_RecordArray = self.buffer[idx[0]]
            has_rewards = episode.rewards is not None
            seq = RP_RecordArray(world_states=episode.world_states[idx[1]:idx[1] + sequence_size],
                                 action_indices=episode.action_indices[idx[1]:idx[1] + sequence_size],
                                 rewards=episode.rewards[idx[1]:idx[1] + sequence_size] if has_rewards else None,
                                 rnn_outputs=None)
            list_batch.append(seq)
        batch = RP_RecordArray(world_states=torch.stack([record.world_states for record in list_batch]),
                               action_indices=torch.stack([record.action_indices for record in list_batch]),
                               rewards=torch.stack([record.rewards for record in list_batch]) if has_rewards else None,
                               rnn_outputs=None)
        return batch
    
    def make_seq_indices(self, sequence_size: int) -> tp.List[tp.Tuple[int, int]]:
        indices = []
        for ep_idx, episode in enumerate(self.buffer):
            for rec_idx in range(len(episode) - sequence_size):
                indices.append((ep_idx, rec_idx))
        return indices
        