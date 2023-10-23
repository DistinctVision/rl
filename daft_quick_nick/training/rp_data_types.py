import typing as tp
from dataclasses import dataclass, field

import torch


@dataclass
class RP_Record:
    world_state_tensor: torch.Tensor
    action_index: int
    reward: tp.Optional[float]


@dataclass
class RP_SeqOfRecords:
    world_state_tensors:  tp.List[torch.Tensor] = field(default_factory=lambda: [])
    acton_indices: tp.List[int] = field(default_factory=lambda: [])
    rewards: tp.Optional[tp.List[float]] = None
    
    def __len__(self) -> int:
        return len(self.world_state_tensors)

    def add(self,
            world_state_tensor: torch.Tensor,
            action_index: int,
            reward: tp.Optional[float] = None):
        self.world_state_tensors.append(world_state_tensor)
        self.acton_indices.append(action_index)
        if reward:
            self.rewards.append(reward)
    

@dataclass
class RP_RecordArray:
    world_states: torch.Tensor
    action_indices: torch.Tensor
    rewards: tp.Optional[torch.Tensor] = None

    @staticmethod
    def from_seq(seq_data: RP_SeqOfRecords) -> 'RP_RecordArray':
        world_states = torch.stack(seq_data.world_state_tensors)
        action_indices = torch.tensor(seq_data.acton_indices, dtype=torch.int16)
        rewards = torch.tensor(seq_data.rewards, dtype=torch.float32) if seq_data.rewards else None

        cached_array = RP_RecordArray(world_states=world_states,
                                      action_indices=action_indices,
                                      rewards=rewards)
        return cached_array

    def __len__(self) -> int:
        return self.world_states.shape[0]
