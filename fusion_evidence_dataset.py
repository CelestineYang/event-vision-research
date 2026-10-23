"""Spawn-safe deterministic export dataset; no model construction or CUDA use."""
import numpy as np
import torch

from ucf101_fusion_window_data import FusionWindowDataset
from routing_data import protocol_config


def event_statistics(events, row, point_count):
    t, x, y, p = events
    n = len(t)
    width, height = int(row['sensor_width']), int(row['sensor_height'])
    duration = (int(row['end_us'])-int(row['start_us']))/1e6
    if n == 0 or duration<=0:
        raise ValueError('Invalid event statistics')
    coverage = len(np.unique(np.asarray(y, dtype=np.int64)*width+np.asarray(x, dtype=np.int64)))/(width*height)
    return np.array([np.log1p(n/(duration*width*height)), np.log1p(n), coverage, point_count/n], dtype=np.float32)


class ExportDataset(FusionWindowDataset):
    def __init__(self, experiment, split, store, roles, dataset_id, *, seed=0,
                 sampling_epoch=0, selection_protocol='strict'):
        super().__init__(experiment, split, store, point_count=2048,
                         point_normalization='official', seed=seed)
        protocol_config(selection_protocol)
        self.roles, self.dataset_id, self.selection_protocol = roles, dataset_id, selection_protocol
        self.augmentation = 'none'
        self.set_epoch(sampling_epoch)
        available = {r['video_id'] for r in self.rows}
        requested = {video for source, video in roles if source == split}
        if requested-available:
            raise ValueError('Role mapping names videos absent from audited source split')
        if split == 'test' and any(roles[(split, v)] == 'gate_train' for v in requested):
            raise ValueError('Original Test cannot be gate_train')
        if split == 'test' and selection_protocol != 'original_test_as_val' and any(roles[(split, v)] == 'gate_val' for v in requested):
            raise ValueError('Test-as-Val requires explicit protocol')
        self.rows = [r for r in self.rows if (split, r['video_id']) in roles]
        if not self.rows:
            raise ValueError('Role mapping selects no videos from this source split')

    def __getitem__(self, index):
        batch = super().__getitem__(index)
        row = self.rows[index]
        batch['statistics'] = torch.from_numpy(event_statistics(self.events.get(row), row, self.point_count))
        batch['metadata'] = dict(dataset_id=self.dataset_id, source_split=self.split,
            role=self.roles[(self.split, row['video_id'])], video_id=row['video_id'],
            window_id=str(row['window_id']), start_us=int(row['start_us']), end_us=int(row['end_us']),
            label_id=int(row['label_id']), selection_protocol=self.selection_protocol,
            expert_fit_exposure='seen' if self.split == 'train' else 'unknown',
            expert_selection_exposure='seen' if self.split == 'test' else 'unknown')
        return batch
