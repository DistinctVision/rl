import typing as tp

import math
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class RolloutData:
    observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    log_probs: torch.Tensor
    values: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    
    def __init__(self, observation_size: int, data_size: int):
        self.observations = torch.zeros((data_size, observation_size), dtype=torch.float32)
        self.actions = torch.zeros(data_size, dtype=torch.int16)
        self.rewards = torch.zeros(data_size, dtype=torch.float32)
        self.log_probs = torch.zeros(data_size, dtype=torch.float32)
        self.values = torch.zeros(data_size, dtype=torch.float32)
        self.advantages = torch.zeros(data_size, dtype=torch.float32)
        self.returns = torch.zeros(data_size, dtype=torch.float32)
        self.size = 0
        
    def __len__(self) -> int:
        return self.observations.shape[0]
    
    def copy_from(self, start_index: int, src_buffer: 'RolloutData', src_start_index: int, size: int):
        src_end_index = src_start_index + size
        dst_end_index = start_index + size
        self.observations[start_index:dst_end_index] = src_buffer.observations[src_start_index:src_end_index]
        self.actions[start_index:dst_end_index] = src_buffer.actions[src_start_index:src_end_index]
        self.rewards[start_index:dst_end_index] = src_buffer.rewards[src_start_index:src_end_index]
        self.log_probs[start_index:dst_end_index] = src_buffer.log_probs[src_start_index:src_end_index]
        self.values[start_index:dst_end_index] = src_buffer.values[src_start_index:src_end_index]
        self.advantages[start_index:dst_end_index] = src_buffer.advantages[src_start_index:src_end_index]
        self.returns[start_index:dst_end_index] = src_buffer.returns[src_start_index:src_end_index]
    
    def get_state(self, index: int, sequence_size: int) -> torch.Tensor:
        start_idx = index - sequence_size + 1
        if start_idx < 0:
            prepend_tensor = torch.stack([self.observations[0]] * (- start_idx))
            state_tensor =  torch.cat([prepend_tensor, self.observations[0:index+1]], dim=0)
        else:
            state_tensor = self.observations[start_idx:index+1]
        d = state_tensor.flatten()
        assert d.shape[0] == 43 * sequence_size
        return d


class RolloutBuffer:
    
    def __init__(self,
                 rollout_cfg: tp.Dict[str, tp.Union[float, int, bool]],
                 value_net: torch.nn.Module,
                 sequence_size: int):
        self.value_net = value_net
        self.rollout_cfg = rollout_cfg
        observation_size = int(self.rollout_cfg['observation_size'])
        max_buffer_size = int(self.rollout_cfg['max_buffer_size'])
        self.buffer = RolloutData(observation_size, max_buffer_size)
        self.is_finished = False
        self.sequence_size = sequence_size
        
    @property
    def observation_size(self) -> int:
        return self.buffer.observations.shape[1]
    
    @property
    def  size(self) -> int:
        return self.buffer.size
    
    def get_state(self, index: int) -> torch.Tensor:
        self.buffer.get_state(index, self.sequence_size)
    
    def new_state(self, observation: torch.Tensor) -> torch.Tensor:
        if self.sequence_size == 1:
            return observation
        pre_state_size = self.sequence_size - 1
        if self.buffer.size == 0:
            return torch.cat([observation] * self.sequence_size, dim=-1)
        pre_state = self.buffer.get_state(self.buffer.size - 1, pre_state_size)
        return torch.cat([pre_state, observation], dim=-1)
        
    def add(self, observation: torch.Tensor, action: int, action_log_prob: float, reward: float):
        self.buffer.observations[self.buffer.size] = observation
        self.buffer.actions[self.buffer.size] = action
        self.buffer.rewards[self.buffer.size] = reward
        self.buffer.log_probs[self.buffer.size] = action_log_prob
        self.buffer.size += 1
        assert self.buffer.size <= len(self.buffer)
    
    def __len__(self) -> int:
        return self.buffer.size
    
    def cut_data(self, size: int):
        assert self.buffer.size > size
        rest_size = self.buffer.size - size
        self.buffer.observations[:rest_size] = self.buffer.observations[size:self.size].clone()
        self.buffer.actions[:rest_size] = self.buffer.actions[size:self.size].clone()
        self.buffer.rewards[:rest_size] = self.buffer.rewards[size:self.size].clone()
        self.buffer.log_probs[:rest_size] = self.buffer.log_probs[size:self.size].clone()
        self.buffer.values[:rest_size] = self.buffer.values[size:self.size].clone()
        self.buffer.advantages[:rest_size] = self.buffer.advantages[size:self.size].clone()
        self.buffer.returns[:rest_size] = self.buffer.returns[size:self.size].clone()
        self.buffer.size = rest_size
        
    def start(self):
        self.buffer.size = 0
        self.is_finished = False
        
    def finish(self, observation: torch.Tensor, truncated: bool = False):
        
        first_param = next(iter(self.value_net.parameters()))
        device = first_param.device
        
        calc_batch_size = int(self.rollout_cfg['calc_batch_size'])
        discount_factor = float(self.rollout_cfg['discount_factor'])
        gae_lambda = float(self.rollout_cfg['gae_lambda'])
        
        self.value_net.eval()
        with torch.no_grad():
            for batch_idx in range(math.ceil(self.buffer.size / calc_batch_size)):
                r = (batch_idx * calc_batch_size, (batch_idx + 1) * calc_batch_size)
                batch_states = torch.stack([self.buffer.get_state(idx, self.sequence_size) for idx in range(r[0], r[1])])
                values: torch.Tensor = self.value_net(batch_states.to(device))
                values = values.cpu().squeeze(1).detach()
                self.buffer.values[r[0]:r[1]] = values
            if truncated:
                state_tensor = torch.cat([self.buffer.observations[self.buffer.size-self.sequence_size:
                                                                   self.buffer.size],
                                          observation], dim=0)
                state_tensor = state_tensor.flatten()
                
                next_value: torch.Tensor = self.value_net(state_tensor.unsqueeze(0).to(device))
                next_value = float(next_value.cpu().squeeze(0))
            else:
                next_value = 0
        
        advantage = 0
        for pos in reversed(range(self.buffer.size)):
            td_target = self.buffer.rewards[pos] + discount_factor * next_value
            td_error = td_target - self.buffer.values[pos]
            advantage = td_error + discount_factor * gae_lambda * advantage
            self.buffer.advantages[pos] = advantage
            next_value = self.buffer.values[pos]
        self.buffer.returns[:self.buffer.size] = self.buffer.advantages[:self.buffer.size] + \
            self.buffer.values[:self.buffer.size]
            
        # normalize_advantage = bool(self.rollout_cfg['normalize_advantage'])
        # if normalize_advantage:
        #     mean_advantages = self.buffer.advantages[:self.buffer.size].mean()
        #     std_advantages = self.buffer.advantages[:self.buffer.size].std()
        #     self.buffer.advantages[:self.buffer.size] = (self.buffer.advantages[:self.buffer.size] - mean_advantages) / std_advantages
            
        # normalize_returns  = bool(self.rollout_cfg['normalize_returns '])
        # if normalize_returns:
        #     mean_returns = self.buffer.returns[:self.buffer.size].mean()
        #     std_returns = self.buffer.returns[:self.buffer.size].std()
        #     self.buffer.returns[:self.buffer.size] = (self.buffer.returns[:self.buffer.size] - mean_returns) / std_returns
        
        self.is_finished = True


@dataclass
class RolloutBatch:
    states: torch.Tensor
    actions: torch.Tensor
    advantages: torch.Tensor
    log_probs: torch.Tensor
    returns: torch.Tensor
    values: torch.Tensor
    
    def to_device(self, device: tp.Union[torch.device, str]):
        self.states = self.states.to(device)
        self.actions = self.actions.to(device)
        self.advantages = self.advantages.to(device)
        self.log_probs = self.log_probs.to(device)
        self.returns = self.returns.to(device)
        self.values = self.values.to(device)

    
class RolloutDataset:
    
    @staticmethod
    def collect_data(data_size: int, batch_size: int, sequence_size: int,
                     buffers: tp.List[RolloutBuffer],
                     prepend_pad: bool = True) -> tp.Tuple['RolloutDataset', tp.List[RolloutBuffer]]:
        in_data_size = sum([buffer.size for buffer in buffers])
        if in_data_size < data_size:
            data_size = in_data_size
        
        data = RolloutData(buffers[0].observation_size, data_size)
        collected_data_size = 0
        out_buffers = []
        for buffer in buffers:
            assert buffer.is_finished
            rest_data_size = data_size - collected_data_size
            if rest_data_size >= buffer.size:
                data.copy_from(collected_data_size, buffer.buffer, 0, buffer.size)
                collected_data_size += buffer.size
            elif rest_data_size > 0:
                data.copy_from(collected_data_size, buffer.buffer, 0, rest_data_size)
                collected_data_size += rest_data_size
                buffer.cut_data(rest_data_size)
                out_buffers.append(buffer)
            else:
                out_buffers.append(buffer)
        dataset = RolloutDataset(data, batch_size, sequence_size, prepend_pad)
        return dataset, out_buffers
    
    def __init__(self, data: RolloutData, batch_size: int, sequence_size: int = 1, prepend_pad: bool = True):
        self.data = data
        self.sequence_size = sequence_size
        self.batch_size = batch_size
        self.indices = self._make_indices(prepend_pad)
        
    def _make_indices(self, prepend_pad: bool) -> tp.List[int]:
        start_idx = 0 if prepend_pad else self.sequence_size - 1
        indices = [idx for idx in range(start_idx, len(self.data))]
        return indices
    
    def shuffle(self):
        np.random.shuffle(self.indices)
    
    def __len__(self) -> int:
        return len(self.indices) // self.batch_size
    
    def __iter__(self) -> tp.Iterator[RolloutBatch]:
        return (self.get_seq_batch(self.indices[batch_idx*self.batch_size:(batch_idx+1)*self.batch_size])
                for batch_idx in range(len(self.indices) // self.batch_size))
    
    def get_seq_batch(self, batch_indices: tp.List[int]) -> RolloutBatch:
        batch = RolloutBatch(actions=torch.stack([self.data.actions[item_idx] for item_idx in batch_indices]).long(),
                             states=torch.stack([self.data.get_state(item_idx, self.sequence_size) for item_idx in batch_indices]),
                             advantages=torch.stack([self.data.advantages[item_idx] for item_idx in batch_indices]),
                             log_probs=torch.stack([self.data.log_probs[item_idx] for item_idx in batch_indices]),
                             returns=torch.stack([self.data.returns[item_idx] for item_idx in batch_indices]),
                             values=torch.stack([self.data.values[item_idx] for item_idx in batch_indices]))
        return batch
