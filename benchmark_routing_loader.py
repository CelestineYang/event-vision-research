"""Read-only loader comparison on identical, distributed Train windows."""
import hashlib
import json
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from export_fusion_evidence import export_collate, make_export_dataset, role_mapping


def main():
    torch.set_num_threads(2)
    roles = role_mapping('configs/fusion_routing_roles.original_test_as_val.json',
                         'UCF101-DVS', 'original_test_as_val')
    data = make_export_dataset('experiments/ucf101dvs_gcstr_sliding_txtsplit_v2', 'train',
        '/remote-home/datasets/event_project_cache/ucf101_txtsplit_v2_events_v1', roles,
        'UCF101-DVS', seed=0, sampling_epoch=0, selection_protocol='original_test_as_val')
    # Four separated contiguous batches, not just the first easy video.
    starts = np.linspace(0, len(data.rows)-32, 4, dtype=int)
    data.rows = [r for start in starts for r in data.rows[start:start+32]]
    reference = None
    for workers in (0, 4, 0, 4):
        loader = DataLoader(data, batch_size=32, shuffle=False, num_workers=workers,
                            collate_fn=export_collate,
                            **({'multiprocessing_context': 'spawn', 'prefetch_factor': 2} if workers else {}))
        begin = time.perf_counter()
        digest = hashlib.sha256()
        ready = []
        for batch, rows in loader:
            ready.append(time.perf_counter()-begin)
            for key in sorted(batch):
                value = batch[key]
                digest.update(key.encode())
                digest.update(value.numpy().tobytes() if torch.is_tensor(value)
                              else json.dumps(value, sort_keys=True).encode())
            digest.update(json.dumps(rows, sort_keys=True).encode())
        seconds = time.perf_counter()-begin
        identity = digest.hexdigest()
        if reference is None:
            reference = identity
        if identity != reference:
            raise RuntimeError('Worker count changed inputs or metadata')
        print(json.dumps(dict(workers=workers, windows=len(data), seconds=seconds,
                              windows_per_second=len(data)/seconds, batch_ready_seconds=ready,
                              inputs_identical=True)), flush=True)


if __name__ == '__main__':
    main()
