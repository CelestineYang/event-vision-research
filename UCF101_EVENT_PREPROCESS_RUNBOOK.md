# Full-event preprocessing for the approved TXT split

Started 2026-10-15, in a dedicated derived-cache directory:
`/remote-home/datasets/event_project_cache/ucf101_txtsplit_v2_events_v1/`.
Original dataset files and the active training entry are not modified.

`preprocess_ucf101_event_store.py` handles the 12,851 videos that have valid
windows in `experiments/ucf101dvs_gcstr_sliding_txtsplit_v2/window_manifest.csv`.
It preserves each eligible video's **entire** event stream once, including all
original t/x/y/p values, dtypes and ordering. Overlapping windows share storage,
using precomputed left/right event indices, not duplicate event arrays. There
are 135,450 windows: 108,066 Train and 27,384 Test. No event downsampling,
time-axis changes, split changes or fixed augmentation are introduced.

Storage layout:

```text
signature.json
videos.json
videos/<video-id-hash>/events.npy
videos/<video-id-hash>/metadata.json
summary.json
```

Each `events.npy` is an uncompressed structured array, accessible with
`np.load(..., mmap_mode='r')`. Fields retain original types; common records use
13 bytes/event. Estimated payload is about 145 GiB, external to the instance
project directory. Startup checks a conservative disk-space upper bound. The
original dataset's capacity policy for derived caches remains an administrative
matter; files are kept strictly under this new project-specific directory.

Each source NPZ is SHA-256 checked once when building its video cache, then
loaded once. Every output field is compared exactly against its source before
the video is committed. All window boundaries and event counts are verified.
Per-video metadata is atomically published; interrupted builds support
`--resume`, checking already built content hashes. Nothing is deleted.

After building, 32 real windows per split are sampled deterministically. Their
raw event slices are compared to original NPZ, and the three normalized images
are compared bit-for-bit with the original dataset implementation. Train uses
identical epoch/seed-dependent spatial augmentation; Test compares to the
existing image cache. Full data verification covers every event at build time;
rendering verification is sampled. A ready summary is only published after both
stages succeed. `EventStore` rejects incomplete or mismatched stores.

`PreparedThreeViewDataset` in `ucf101_event_store.py` uses bounded per-worker
memory-map handles and the same augmentation seed and renderer. The joint entry
now selects it for Train with `--event-store`. Its optional Test reader renders
full events online for verification only; production Test evaluation continues
to use the faster pre-rendered Test images.

Progress (project root):

```bash
watch -n 5 cat experiments/ucf101dvs_gcstr_sliding_txtsplit_v2/event_preprocess_status.json
tail -F logs/ucf101_event_preprocess_20261015.log
```

Final verification report:
`experiments/ucf101dvs_gcstr_sliding_txtsplit_v2/event_preprocess_report.json`.
The external `summary.json` also includes exactness results and small CPU timing
comparisons. The Test comparison measures image-cache reads versus online
rendering and is not an event-loading benchmark. No GPU epoch speedup is claimed
by these timings. The subprocess is detached; it does not stop, restart or switch
training automatically.

Resume preprocessing only after the prior process has exited:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv-eventmamba/bin/python -u preprocess_ucf101_event_store.py --resume
```

CPU validation tests: `python -m unittest test_ucf101_event_store test_ucf101_window_gcstr`.
Eight tests passed before launch, including exact original/preprocessed three-view
equivalence, overlapping windows, original dtypes, spawn-safe serialization,
pending-store rejection and manifest identity checks.

## Approved training-backend transition after epoch 6

Training is stopped at the verified complete epoch6 joint checkpoint. The current
joint entry accepts `--event-store` to select `PreparedThreeViewDataset` **only
for Train**. Test continues using the pre-rendered image loader. The optional
`--allow-data-backend-transition` is required for the first resume from legacy
NPZ checkpoints; it permits only the added mmap backend and changed trainer/
reader code fingerprints. All other protocol settings and upstream code hashes
must match. It cannot change the split, model, batch size, epochs or optimizer.

`config.json` remains epoch1 provenance. After a successfully saved resumed epoch,
`active_config.json` records the current data backend. Future checkpoints contain
the current backend config. No formal checkpoint was changed by the validation.

A resume Smoke was run against the actual epoch6 weights/Adam on two GPUs, using
129 Train windows. Its outputs are `smoke_joint_mmap/` and
`joint_smoke_mmap_passed.json`. It checks original epoch6 loading, optimizer
restoration, updates and full sample coverage, same-state model hashes across
both ranks, and matching distributed video metrics. The single-rank reference
uses the same batch shapes as distributed evaluation; differing batch shapes
can select different convolution kernels and invalidate overly strict numerical
comparisons with trained logits. No precision or training hyperparameter was
changed to address this reference-test issue. Nine CPU tests also passed.

The new Test cache location is
`/remote-home/datasets/event_project_cache/ucf101_txtsplit_v2_test_images_v1/`.
`relocate_ucf101_test_cache.py` copies/verifies every image with SHA-256, preserves
the original `cache/` path via a symlink, then removes only the verified redundant
project copy. It holds `.training.lock` throughout; no training process can
restart while the copy/switch is in progress. Progress is recorded in
`image_cache_migration_status.json` in the experiment. All 82,152 images must
be verified before switching. Payload is ~46.1 GiB; old GPFS allocated usage
was ~93 GiB due to filesystem allocation overhead.

Relocation completed successfully on 2026-10-15: all 82,152 hashes matched,
`cache/` now points to the external directory, and the redundant project copy
was removed. Real Train samples were checked to open only NPY memory maps;
Test samples at the start/middle/end of its index remain readable via the old
symlink path. Project allocated usage dropped to approximately 48 GiB.
The formal epoch6 joint checkpoint still matches its saved SHA-256. No formal
epoch7 training was launched by this change; both GPU test processes exited.

Once data relocation and validation are confirmed, the following is the explicit
**future** resume command (not launched by the backend modification):

```bash
CUDA_VISIBLE_DEVICES=0,1 .venv-eventmamba/bin/python -m torch.distributed.run --standalone --nproc_per_node=2 train_ucf101_window_joint.py --train --resume --allow-data-backend-transition --event-store /remote-home/datasets/event_project_cache/ucf101_txtsplit_v2_events_v1 --output experiments/ucf101dvs_gcstr_sliding_txtsplit_v2
```

This restores epoch6 and begins epoch7. The corresponding Smoke command replaces
`--train` with `--smoke`. Use the exact same hyperparameters as the saved run.
