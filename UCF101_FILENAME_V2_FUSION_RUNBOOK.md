# UCF101 filename V2 pretrained A fusion

Entry: `train_ucf101_pretrained_fusion.py`. Detached supervisor:
`run_ucf101_pretrained_fusion.py`. This is a new experiment, not a resume of
the completed frame pretraining or the historical whole-recording A experiment.

## Protocol

- Reuse TXT video split (10819 Train / 2704 Test before filtering). Valid
  windows: 108066 Train / 27384 Test, from 10296 / 2555 retained videos.
- 1s windows / 0.5s stride; raw count >10000; all valid windows used once per
  epoch, no padding/dropping. Split by video before window extraction.
- Shared spatial augmentation on Train, full-event gCSTR projections and
  2048 sorted point samples per window. Official point normalization t,x,y:
  sampled-time min/max, x/320, y/240. Test sampling deterministic by window.
- Train reads existing verified mmap; Test reads the same mmap for points and
  existing frozen frame image cache. No new large dataset cache is built.
- Images: three pretrained `best_loss.pt`, frozen parameters and BN.
- Points: new uploaded filename `best_checkpoint.pth`, full V2 backbone/head,
  32 embedding channels and stage widths 64/128/256, groups1024/512/256, K24,
  feature-space KNN and BiMamba-v2. Backbone/head fine-tuned from epoch147
  pretraining checkpoint. Fusion adapters/positions/prototypes/head initialized.
- 150 epochs, global batch16 (8/rank), 4 workers/rank, two GPUs0/1, DDP with
  SyncBatchNorm, seed0, FP32 (no AMP). AdamW LR1e-4 fixed, weight_decay1e-4,
  no scheduler, no early stopping, no label smoothing, clip global grad norm1.
- Objective: fusion cross entropy +0.5 * pretrained point-head cross entropy.
  Unequal last shards weighted by actual sample counts; SyncBN permits local
  singleton tails with at least 2 global samples. Frozen image BN stays eval.
- Every epoch evaluates Test: window CE/Top1/Top5 and per-video soft-vote
  NLL/Top1/Top5, for fusion and point outputs; hard-vote Top1 diagnostic too.
  Fusion video NLL selects best (tie: higher Top1, then earlier epoch).
  Test participates in selection and is not an independent held-out estimate.
- Eval FPS uses fixed seeds each epoch; training RNG is restored afterwards.
  Checkpoints store per-rank Python/NumPy/torch/CUDA RNG and AdamW states.

## Locations

Logs and protocol: `experiments/ucf101_filename_v2_fusion_A_seed0_v1/`.
Console/supervisor: `logs/ucf101_filename_v2_fusion_A_seed0_v1/`.
All formal checkpoints (outside the instance quota):
`/remote-home/datasets/event_project_cache/ucf101_filename_v2_fusion_A_seed0_v1_checkpoints/`.
Save atomic `last.pt` every epoch, `best_loss.pt` on improvement and
`epoch_005.pt`, `epoch_010.pt`, ... every5 epochs and final. Atomic replacement
means hard-linked epoch snapshots remain immutable. No old checkpoints deleted.
Full source/data/input-checkpoint fingerprints are checked on resume.
The experiment's `checkpoints` symlink points to that external directory.
Smoke weights were likewise moved to an external directory with original
paths retained through symlinks, leaving project allocation around48GiB.

Metrics: `metrics.jsonl` in experiment directory. `status.json` gives last
completed epoch and best model; prediction CSVs retain last and selected-best
video results. Console prints progress every20 steps and at phase boundaries.

## Start / resume / monitor

Supervisor requires completed two-epoch DDP Smoke with matching source hashes.

```bash
.venv-eventmamba/bin/python -u run_ucf101_pretrained_fusion.py
# After a stopped run (do not run while another supervisor holds the lock):
.venv-eventmamba/bin/python -u run_ucf101_pretrained_fusion.py --resume
```

The actual start is detached using a new session, with supervisor stdout in
`logs/ucf101_filename_v2_fusion_A_seed0_v1/supervisor.log`.

```bash
tail -F logs/ucf101_filename_v2_fusion_A_seed0_v1/train.log | grep --line-buffered -E '"event": "(progress|epoch|finished)"'
watch -n 2 nvidia-smi
```

For a deliberate epoch-boundary pause, use trainer `--stop-after-epoch N` on
a future invocation; current process does not hot-read source/CLI changes.

## Smoke

Isolated output `experiments/ucf101_filename_v2_fusion_A_smoke_v3`, 33 Train
windows only. Step1 trains epoch1 and stops after saving; a second process
loads that checkpoint/optimizer/RNG and completes epoch2. Includes uneven
global batches and local singleton tail, real workers, full forward/backward,
coverage, checkpoint read-back and model-state equality on both ranks. Smoke
uses Train payload for evaluation and never seeds the formal run. Nine reused
window/event-store/metric CPU checks pass.

## Formal launch (2026-10-16 13:23 UTC)

Detached supervisor PID1482593, torchrun PID1482704. Formal run initialized
from original four pretrained checkpoints (not Smoke). Two ranks started with
108066 Train windows /27384 Test windows. First40/6755 training steps of epoch1
completed with finite fusion and auxiliary losses. Only existing nonfatal DDP
gradient-stride performance warnings were observed. The historical paused job
on GPU0 remains untouched. Checkpoints write to the external directory above.
