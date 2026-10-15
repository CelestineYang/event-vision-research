# UCF101-DVS gCSTR / A: official Train/Test, fixed epochs, dual GPU

## Current decision (2026-10-12)
After isolated Smoke Tests passed, the user authorized formal training.
Image-only dual-GPU training started at 2026-10-12 16:22 UTC; see launch below.
The old UCF single-GPU processes remain paused in memory; existing DailyDVS V2
continues running. The new protocol does not apply to those already-loaded old
processes. Existing CSVs, logs and weights were preserved.

## Data
- Dataset: UCF101-DVS, official split1 matched to available valid event samples,
  101 classes,240x180. Source:
  https://www.crcv.ucf.edu/data/UCF101.php
- Train9464,Test3745. NO validation split. Preserve every official manifest row,
  sample ID, class and NPZ path; check Train/Test paths/IDs/class-groups disjoint.
- Future manifest: ucf101_a_official_split1.csv and corresponding .report.json.
  Old ucf101_a_splits_seed0.csv is a historical three-way manifest and is not used.
- Whole-recording8192-event input; gCSTR cstr3/xt_p/ty_p uses ALL augmented events
  from the same recording. Images224x224, temporal bins224, x/240,y/180.
- Train-only spatial augmentation, five fixed variants sampled by epoch/sample
  seed. Test has no augmentation and deterministic event sampling.
- No full-event/image disk cache generated. Raw dataset files remain read-only.

## Training protocol
Both stages run ALL configured epochs. Default50; configurable with --epochs.
There is no patience, bad-epoch counter, validation loop, best-checkpoint
selection, metric-dependent stopping, or Test-feedback scheduling.

| Stage | Model | Global batch | Batch/card on two GPUs | Optimizer |
|---|---|---|---|---|
| images | Three independent ImageNet-initialized ResNet18 branches,101-class fc |64|32|Adam lr1e-4,wd0|
| fusion | Original A: frozen UCF ResNets + EventMamba V1 + prototype + point aux |16|8|AdamW lr1e-4,wd1e-4|

Ordinary CE; A objective=fusion CE+0.5*point CE, gradient clipping1. No scheduler.
Three image branches share a data pass, have separate parameters/optimizers,
and are optimized sequentially within each rank. Update all image backbone/BN
parameters during image pretraining. Frozen image parameters and BN in A.
A uses V1 groups1024/512/256,K24,channels64/128/256, real BiMamba-v2; not V2 encoder.

## DDP and checkpoint correctness
- Launcher defaults to --devices0,1; uses torch.distributed.run, two processes,
  NCCL/DDP and SyncBatchNorm. --device is a backward-compatible alias.
- GLOBAL batch stays64/16. DistributedSampler assigns disjoint Train samples.
  Actual9464 divides two ranks evenly. Keep the final smaller batch: ALL9464
  examples are processed per epoch, with no padding or discarded samples.
  A requested rank count that cannot divide the dataset is rejected.
- Data loader workers use spawn; per-epoch/per-rank seeds allow reproducible
  epoch-boundary resume at the SAME world size (CUDA bitwise identity not claimed).
- Both A outputs are INSIDE the DDP forward; both losses backpropagate through it.
- Rank0 alone writes metrics/checkpoints. Scalar train totals are all-reduced,
  weighted by example counts. Errors during rank0 checkpoint I/O are relayed.
- Save raw model keys, without DDP module prefixes. Resume checks source/protocol,
  data hashes, epoch count, optimizer configuration and world size.
- Old validation/early-stop or single-GPU checkpoints are NOT automatically
  converted to this protocol. New output default:
  ucf101_checkpoints/A_official_train_test_ddp_v3
- images/last.pt: three models+optimizers,epoch and completion state.
  After the final epoch: images/cstr3_final.pt,xt_p_final.pt,ty_p_final.pt.
- A loads these FINAL UCF image weights and freezes them. Fresh point/fusion/aux
  initialization; no DailyDVS fusion weights.
  fusion/last.pt contains model/optimizer resume states;
  fusion/final.pt contains final-epoch A weights.
- No best.pt/best_top1.pt generated under this protocol.

## Evaluation
Training loops read only Train. Each stage's Test evaluation runs in ONE process
after its entire fixed schedule has finished, retaining every3745 Test recording
once. Test does not select weights, stop training or alter settings.
- image_test/test_final.json: final individual branches and probability-product
  ensemble, CE loss,Top1,Top5,all at original-recording level.
- test/test_final.json: final A, fusion CE/Top1/Top5, point auxiliary metrics and
  weighted objective. Fusion training never consumes image_test result scores.
- Smoke (when subsequently authorized) uses only small TRAIN subsets for both
  optimization and evaluation; no official Test events.
- Existing evaluation files must match the current protocol before being reused.

## Future commands (DOCUMENTATION ONLY; not run during this code-change turn)
From /remote-home/chenhaoquan/event_project:

First run image pretraining and its final Test, then stop:
~~~bash
.venv-eventmamba/bin/python -u run_ucf101_a.py --devices 0,1 --epochs 50 --stop-after-images
~~~

For supervised detached execution, prepend setsid/nohup and redirect to a new log
when startup is explicitly requested. The wrapper emits PID/GPU heartbeat each
minute. The stage command and settings are saved in process.json. It does not
automatically kill, restart or reconfigure other experiments.

To later continue A using completed final image weights, add --resume and omit
--stop-after-images. Completed image pretraining/evaluation will be skipped.
Do not run concurrent supervisors or unpause the old code's process as a way to
activate this new protocol; a fresh process is needed.

Progress:
~~~bash
tail -F /remote-home/chenhaoquan/event_project/ucf101_checkpoints/A_official_train_test_ddp_v3/images/console.log
~~~

## Implementation / verification
- ucf101_a_data.py: official Train/Test adapter.
- train_ucf101_a.py: fixed-epoch single/DDP training and final-epoch evaluation.
- run_ucf101_a.py: dual-card stage orchestration.
- test_ucf101_a.py: separate CPU-only protocol/config/sampling tests.
- Changed only these independent UCF files; existing V1/V2/SECNet/fusion sources
  and ongoing DailyDVS V2 training were not modified.
- CPU-only tests verify unchanged Train/Test membership, exact two-rank sampling,
  no padding/dropped tails, single-process Test, final-epoch gate and CLI defaults.
- No training, optimization/backward steps, GPU Smoke Test or DDP runtime
  validation was performed for this new protocol during the code-only turn.
  Previous UCF smoke results describe the OLD validation/early-stop protocol.

## Authorized v3 runtime smoke (2026-10-12 UTC)
- Eight CPU protocol tests passed. Generated official Train/Test-only manifest:
  ucf101_a_official_split1.csv (9464 Train / 3745 Test metadata rows).
- Independent artifacts: ucf101_checkpoints/A_ddp_smoke_20261012_v3.
- Real two-rank NCCL/DDP/SyncBN image training: global64,32/card; checkpoint
  after smoke epoch1, new-process resume to epoch2, three final weights saved.
- Real two-rank A training: global16,8/card,8192 events; fusion CE plus point
  auxiliary0.5, finite objectives/gradients, epoch1 checkpoint and epoch2 resume.
- Single-process image/ensemble and A final-weight evaluation completed on ALL
  128 TRAIN smoke recordings; results in image_test/smoke_eval.json and
  test/smoke_eval.json. Official Test event files were not used.
- Additional same-process multi-step/cross-epoch DDP checks use an isolated
  /tmp/ucf101_ddp_multistep_smoke.py harness forcing two steps per smoke epoch.
  Fusion artifacts: ucf101_checkpoints/A_ddp_multistep_smoke_20261012_v3;
  image artifacts: ucf101_checkpoints/images_ddp_multistep_smoke_20261012_v3.
  Both completed two consecutive steps per epoch for two epochs, exit code0;
  image sampling processed all128 smoke rows per epoch without padding/dropping.
  The initial temporary harness lacked a spawn main guard; that harness-only
  error was corrected and retried, without changing project training sources.
- A reports a nonfatal DDP gradient-stride performance warning. An early-break
  smoke worker also printed a shutdown diagnostic; the parent run exited0,
  saved its checkpoint and passed resume/evaluation. These are not a claim of
  full-run performance or convergence validation.
- Smoke weights/metrics are NOT baselines and must not seed formal experiments.
  V2 stayed running; old UCF processes stayed paused. No formal UCF launch.

## Formal image-only launch (2026-10-12 16:22 UTC)
- User authorized starting training after smoke verification.
- Detached supervisor PID289272, torchrun PID289277; two ranks on physical0,1.
- Command: .venv-eventmamba/bin/python -u run_ucf101_a.py --devices 0,1
  --epochs50 --stop-after-images (actual CLI uses separate --epochs 50 tokens).
- Output: ucf101_checkpoints/A_official_train_test_ddp_v3; supervisor.log,
  process.json, pipeline.jsonl; images/console.log and images/metrics.jsonl.
- Train9464, global batch64 (32/rank), 50 fixed epochs, no Validation/ES,
  ImageNet initialization (not Smoke weights). Final official Test after epoch50.
  A fusion is NOT automatically launched in this image-only run.
- Attempt to terminate old paused UCF processes was rejected by safety review
  because unsaved in-memory progress would be lost. No termination occurred;
  old PIDs210878/210880 and workers remain paused and consume GPU0 memory.
  Their files were preserved. DailyDVS V2 PID5209 remains running on GPU1.

## Formal A launch (2026-10-13 03:13 UTC)
- Image training completed50 epochs and its final Test result was saved.
- User fixed A at150 epochs, global batch16 (8/card), all other A settings
  unchanged: AdamW1e-4,weight_decay1e-4,fixed LR,no scheduler,no early stopping.
- Added --fusion-epochs to run_ucf101_a.py so the completed image protocol stays
  at50 epochs while fusion/final Test use150. Training/model/data source hashes
  remain unchanged and image checkpoints remain compatible.
- Executed detached: .venv-eventmamba/bin/python -u run_ucf101_a.py
  --devices 0,1 --epochs 50 --fusion-epochs 150 --fusion-batch 16 --resume.
- Supervisor PID569400, torchrun569718. Image stages skipped existing completed
  training/evaluation; fusion started fresh, not from historical/Smoke weights.
- First20/592 steps of epoch1 completed with finite losses. Logs:
  supervisor_fusion150.log and fusion/console.log under the existing v3 root.
- Final fusion weights: fusion/final.pt at epoch150; epoch-boundary resume:
  fusion/last.pt. Official final Test automatically follows all150 epochs and
  saves test/test_final.json. No Test-based selection or tuning.
- V2 remains running; old paused UCF remains untouched.

## A intermediate checkpoint retention (2026-10-13)
- User requested full intermediate models every5 epochs while A was in epoch20.
- Independent archive_ucf101_a_checkpoints.py monitors last.pt without modifying
  the already-running training code or any source hashes in its signature.
- It pins a completed checkpoint inode, verifies its epoch/models/optimizers,
  and retains fusion/epoch_020.pt,epoch_025.pt,...,epoch_150.pt. Hard links are
  safe because atomic_save replaces last.pt rather than mutating its inode.
- Epochs5/10/15 were already overwritten and cannot be reconstructed. Each
  retained checkpoint is about158MiB;20..150 requires about4.2GiB additional
  storage after subsequent replacements. Three isolated archive tests passed.
- Detached monitor watches supervisor569400 and exits on final epoch or its exit.
  Logs: fusion/archive_console.log and fusion/archive.jsonl. Monitor must be
  started again if training is later restarted under a new supervisor PID.
- No training interruption, official Test access or model selection introduced.
