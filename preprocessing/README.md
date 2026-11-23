# Four main-experiment datasets: preprocessing

This directory publishes the audited preprocessing sources for THU-EACT-50-CHL,
UCF101-DVS, DVS128 Gesture, and DailyAction-DVS. Existing training entry points
and historical source snapshots are preserved. No raw events, generated caches,
checkpoints, or training logs are included.

## Entry points and protocols

| Dataset | Entry point | Window / stride | Point cache |
|---|---|---|---|
| THU-CHL | `thu_family/prepare_chl.py` | 1.0 / 0.5 s | 2,048 |
| DVS128 Gesture | `thu_family/prepare_dvsgesture.py` | 0.5 / 0.25 s | 1,024; 10 classes |
| DailyAction-DVS | `thu_family/prepare_daily.py` | 1.5 / 0.5 s | 2,048 |
| UCF101-DVS | root-level `prepare_ucf101_txt_split.py` | 1.0 / 0.5 s | full-event window index; GM samples 2,048 online |

All commands below are run from the repository root. Use fresh output directories
on a large data volume; preprocessing can require substantial disk space. These
commands do not launch training or use GPUs.

## THU-CHL, Gesture, and DailyAction

Install `preprocessing/thu_family/requirements.txt` with Python 3.10 or 3.11.
Open3D is needed for THU radius filtering; Gesture and DailyAction do not apply it.

```bash
python -m pip install -r preprocessing/thu_family/requirements.txt
cd preprocessing/thu_family
python -m unittest test_preprocessing test_dvsgesture test_daily -v
cd ../..

python preprocessing/thu_family/prepare_chl.py \
  --root /data/THU-CHL --train-list /data/THU-CHL/train.txt \
  --test-list /data/THU-CHL/test.txt --output /data/prepared/chl --seed 0

python preprocessing/thu_family/prepare_dvsgesture.py \
  --root /data/DvsGesture --output /data/prepared/gesture \
  --classes 10 --points 1024 --seed 0

python preprocessing/thu_family/prepare_daily.py \
  --root /data/DailyAction-DVS --output /data/prepared/daily --seed 0

python preprocessing/build_daily_complete_test_manifest.py \
  --store /data/prepared/daily \
  --output /data/prepared/daily_test_complete/test_windows.jsonl
```

Use `--smoke-per-split 2` and a separate output directory for an initial check.
THU uses the released labels, not the action number parsed from filenames.
Gesture uses the official recording lists and annotation segments, excluding
class 11 in this experiment. DailyAction sorts class/file names and applies the
per-class 80/20 rule with Python seed 0. If preserving an existing DailyAction
split, pass `--split-manifest /data/split_records.json` to verify and reuse it.

The base scripts save complete events, point HDF5 caches, and window manifests.
They preserve the audited EventMamba boundary behavior: the last event index
at or before the end boundary is itself excluded, and regular window ends must
precede the final timestamp. They do not replace that convention with another
interval rule. The base manifests record ineligible windows as `valid=false`.

The complete DailyAction test helper reproduces the existing evaluation's short
recording fallback without loading weights. It keeps eligible base windows and
adds all events from missing nonempty short recordings. The audited full data
produce 288 test recordings / 737 windows, including eight added recordings.
Training retains 1,123 recordings from the original 1,152-recording training
partition after window eligibility checks. Empty or unexpectedly long missing
recordings cause an error instead of silent removal.

THU and Gesture GM evaluation should read every nonempty test-manifest window,
including sparse windows, rather than apply the training eligibility flag again.
Point-only HDF5 rows and full-event manifests therefore have different coverage.

## UCF101-DVS: raw conversion, filename split, full-event cache

The UCF pipeline additionally needs PyTorch, SciPy, and the repository's existing
representation/dataset modules. Use the project's compatible environment, such
as PyTorch 2.0.1 and torchvision 0.15.2. Execute the following stages in order:

```bash
python preprocessing/prepare_ucf101dvs.py --root /data/ucf_archives --split 1 --workers 4

python ucf101_window_data.py prepare \
  --root /data/ucf_archives/UCF101-DVS --output /data/prepared/ucf_initial_audit

python prepare_ucf101_txt_split.py \
  --root /data/ucf_archives/UCF101-DVS \
  --train-list train_ucf.txt --test-list 'test_ucf(1).txt' \
  --previous /data/prepared/ucf_initial_audit --output /data/prepared/ucf_filename

python ucf101_window_data.py cache --output /data/prepared/ucf_filename

python preprocess_ucf101_event_store.py \
  --experiment /data/prepared/ucf_filename --store /data/prepared/ucf_event_store
```

The raw converter expects five `*UCF101-DVS-Part*.zip` archives and
`UCF101TrainTestSplits-RecognitionTask.zip` under its root. The initial audit uses
the archive's standard split to verify raw MAT against decoded NPZ. The next
stage applies this project's versioned filename lists; do not substitute the
initial standard split for the experiment split. The filename lists are not
strictly group-disjoint. Final manifests retain 10,296 training and 2,555 test
videos, from input lists containing 10,819 and 2,704 videos, respectively.

UCF keeps full windows with at least 10,001 events and applies the inherited
exclusive 40-window limit. It uses its own half-open-window implementation,
rather than the THU-family boundary convention. `ucf101_window_data.py cache`
creates the test projection caches required by the event-store rendering audit.
The event store checks field values, window indices, hashes, and rendered-input
equivalence. `--resume` is supported for an identical event-store configuration.

## Frame rendering and augmentation

`thu_family/render_frames.py` generates XY (`cstr3`), XT (`xt_p`), and TY (`ty_p`)
representations from all window events. It is optional: GM normally constructs
training representations online after shared event-space augmentation. Point
sampling uses the same transformed window. Fixed preprocessing caches do not
encode future random training augmentation. UCF rendering and cache construction
remain in the root-level modules to retain their existing import dependencies.

## Source identity

`four_dataset_sources.json` records SHA256 identities for the published entry
points, common projection code, and UCF filename lists. Full-data reproduction
also requires the original data and the appropriate local split files. The
checks validate code and protocol behavior; they do not assert byte-identical
random samples to upstream EventMamba's unseeded caches.
