# DailyDVS-200 prepared_v1 adapter

`train_dailydvs200_window_gcstr.py` now detects prepared_v1 (`config.json` + `train_windows.jsonl`) and uses `PreparedDailyViews`. The old experiment/event_store path remains supported. Archived assets and DVSAction sources are unchanged. Old training checkpoints whose source/config identity predates this change cannot be resumed by the changed entry; use the corresponding archived source for those runs.

`dailydvs200_prepared_data.py` reads full NPZ events, verifies source SHA256 and timestamp origin, uses integer origin subtraction, validates window counts, and shares event augmentation before all three window projections. A per-worker LRU holds at most `--npz-cache-mib` (default256) MiB of retained decompressed event arrays. A recording larger than this is processed transiently but not retained, so this is a cache budget, not a hard RSS ceiling. Random-window sampling can still miss the cache; this reduces repeat decompression but does not replace an mmap event store or constitute a throughput benchmark.

`render_dailydvs200_prepared.py` generates unaugmented float32 [3,224,224] cstr3/xt_p/ty_p arrays directly from prepared_v1. Each filename is its HDF5 point_row. It verifies saved images and stores per-window receipts with SHA256. `--resume` checks data/code identity and receipts; corrupt frames fail rather than being silently reused. The completion marker is invalidated before resumed work. Only a full complete cache can be passed to training `--frames`; augmentation still renders events online.

Full cache command (NOT launched):

```bash
.venv-eventmamba/bin/python render_dailydvs200_prepared.py --data '/path/to/prepared_v1' --output '/path/to/dailydvs200_frames'
```

Add `--resume` after an interruption. Add `--smoke` with a separate output path to generate only three windows from each split. A full cache contains 425883 NPY arrays, roughly256 GB of image payload; confirm storage before intentionally launching it.

Training smoke command (completed on GPU4, 17 Train windows, batch8, workers2; resume to epoch2 also checked):

```bash
CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=2 .venv-eventmamba/bin/python train_dailydvs200_window_gcstr.py --data '/path/to/prepared_v1' --output experiments/daily_smoke --smoke --test-as-val --batch-size 8 --epochs 2 --workers 2 --memory-fraction .15
```

Repeat with `--resume` to verify the next epoch. Formal training is intentionally not launched. Original Test-as-validation selection is unchanged.

Evidence: reports/dailydvs200_adapter_20261031/, experiments/dailydvs200_adapter_smoke_v2_20261031/status.json, logs/dailydvs200_adapter_smoke_v2_20261031.log. The smoke image cache is not a full-dataset training cache.
