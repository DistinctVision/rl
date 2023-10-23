import typing as tp

import torch
import torch.utils.data
import torch.distributed as dist
import torch.utils.data


class ReplayBufferBatchSampler(torch.utils.data.BatchSampler):

    def __init__(self,
                 batch_size: int,
                 indices: tp.List[tp.Tuple[int, int]],
                 ddp_is_enabled: bool = False,
                 device: tp.Optional[tp.Union[torch.device, str]] = None):
        super().__init__(sampler=None, batch_size=batch_size, drop_last=True)
        
        if ddp_is_enabled:
            assert device is not None
            ddp_rank, ddp_world_size = dist.get_rank(), dist.get_world_size()
            indices_tensor = torch.tensor(indices, device=device)
            dist.broadcast(indices_tensor, src=0)
            indices = indices_tensor.cpu().tolist()
            part_size = len(indices) // ddp_world_size
            indices = indices[ddp_rank * part_size:(ddp_rank + 1) * part_size]
        
        self.indices = indices

    def __iter__(self) -> tp.Iterator[tp.List[tp.Tuple[int, int]]]:
        return (self.get_batch_indices(batch_idx) for batch_idx in range(len(self)))

    def __len__(self) -> int:
        return len(self.indices) // self.batch_size

    def get_batch_indices(self, batch_index: int) -> tp.List[tp.Tuple[int, int]]:
        offset = batch_index * self.batch_size
        return [self.indices[item_index] for item_index in range(offset, offset + self.batch_size)]
