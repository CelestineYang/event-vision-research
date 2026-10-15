# Current UCF101-DVS split: supplied TXT lists

The user approved `train_ucf.txt` and `test_ucf(1).txt` on 2026-10-15.
These lists define the video split, superseding the previous RGB-split-derived
lists. They are approximately 80/20, not group-disjoint. They are not labeled
as RGB UCF101 official split 1. No video or window is shared across Train/Test.

New experiment: `experiments/ucf101dvs_gcstr_sliding_txtsplit_v2/`.
Both original input lists are snapshotted under its `split_inputs/` directory
with hashes recorded in `dataset_audit.json`. Stable original video names and
class labels determine logical split; the physical directory of an existing
NPZ does not determine its new split.

`prepare_ucf101_txt_split.py` reuses the prior complete raw audit and window
boundaries for 13,209 existing NPZ videos, then reads original MAT for omitted
videos. The 212 additional nonempty videos are converted only under the new
experiment's `processed/` directory. The 102 empty raw videos remain recorded
in the video split and exclusion report; no synthetic events/polarity are added.
It saves compatible `video_splits.csv`, `window_manifest.csv`, class mapping,
and an audit report. No large image cache is created by split preparation.

Expected counts after the unchanged window filters:

| | Train | Test |
|---|---:|---:|
| Listed videos | 10,819 | 2,704 |
| Empty videos | 75 | 27 |
| Videos with valid windows | 10,296 | 2,555 |
| Valid windows | 108,066 | 27,384 |

Protocol remains 1s windows/0.5s stride, full windows only, raw event count >10000,
candidate windows <40 per video. Train spatial augmentation; all window events
rendered, no point sampling. Three independent ResNet-18 branches share each
DDP batch; global batch64 per branch, Adam1e-4, 50 epochs, no early stopping.
Test participates in model selection and is not an independent held-out score.

The old paused job was terminated when fresh training was authorized; saved
epoch19 checkpoints and logs remain. Do not resume its weights as a new-split
baseline: 1,877 new Test videos were part of its old Train. A new-split run
starts each branch from ImageNet weights and a fresh Adam optimizer.

Fresh training has now been explicitly authorized. Use the new output argument
throughout, never `--resume` from
the old experiment:

```bash
.venv-eventmamba/bin/python ucf101_window_data.py cache --output experiments/ucf101dvs_gcstr_sliding_txtsplit_v2
CUDA_VISIBLE_DEVICES=0,1 .venv-eventmamba/bin/python -m torch.distributed.run --standalone --nproc_per_node=2 train_ucf101_window_joint.py --smoke --output experiments/ucf101dvs_gcstr_sliding_txtsplit_v2
CUDA_VISIBLE_DEVICES=0,1 .venv-eventmamba/bin/python -m torch.distributed.run --standalone --nproc_per_node=2 train_ucf101_window_joint.py --train --output experiments/ucf101dvs_gcstr_sliding_txtsplit_v2
```

The authorized pipeline uses existing prepared data and automatically runs
Test cache → matching joint Smoke → fresh formal training. It has no `--resume`
flag and refuses existing formal checkpoints/config:

```bash
.venv-eventmamba/bin/python -u run_ucf101_window_joint.py --prepared --output experiments/ucf101dvs_gcstr_sliding_txtsplit_v2 --log-dir logs/ucf101_window_joint_txtsplit_v2
```

It is already started in a detached session; do not launch a second copy.
Monitor status: `watch -n 5 cat logs/ucf101_window_joint_txtsplit_v2/status.json`.
Training log: `tail -F logs/ucf101_window_joint_txtsplit_v2/train.log`.
On any failing stage it stops without launching the next stage.
