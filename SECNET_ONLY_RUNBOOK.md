# SECNet-only / DailyDVS-200

This experiment evaluates the project's unchanged SECNetEncoder, not a claim
of exact reproduction of the entire SECNet paper protocol.

## Model and fixed protocol

- Reuse `secnet_encoder.py` unchanged: embed 64, groups 1024/512/256,
  neighbors 24, final channels 540; input `[B,4,8192]` in x,y,t,p order.
- Concatenate final mean and max pooling (1080 channels). Author-style head:
  Dropout(0.1), Linear(1080,256), BatchNorm, GELU, Dropout(0.5), Linear(256,200).
  Reference: https://github.com/rhwxmx/SECNet_ICML/blob/main/models/FECNet.py
- Fresh initialization, AdamW lr 0.001, weight decay 0.0001, default betas,
  cosine to zero over 150 epochs, batch 16, seed 0, float32, ordinary CE.
  No early stopping, no validation, no test-based model/hyperparameter selection.
- Read dataset-root `splits/train.csv` (14,773) and `splits/test.csv` (4,093)
  directly. The old internal validation samples are part of official Train.
  Check metadata overlap, labels, duplicates and existence of processed files.
- Reuse `sample_event_cloud` and spatial `augment_events`, five fixed augmentation
  variants per sample; choose variants and sample events with per-sample/epoch
  seeds. No gCSTR images/caches. Test has deterministic sampling, no augmentation.
- Final Test is evaluated only after epoch 150, from `final.pt`. No best checkpoint.
  Metrics: sample-weighted CE loss, Top-1, Top-5 (fractions, not percentages).
  FPS remains randomly initialized as in the unchanged encoder; evaluation RNG
  is fixed to 100000. Bitwise cross-device determinism is not promised.

## Files

`secnet_only.py`: classifier; `secnet_only_dataset.py`: event-only data;
`train_secnet_only.py`: train/resume/final test; `test_secnet_only.py`: tests.
Existing fusion code/checkpoints are not modified.

## Launch (GPU 1 sharing explicitly approved)

From `/remote-home/chenhaoquan/event_project`:

```bash
mkdir -p logs/secnet_only
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 setsid nohup .venv-eventmamba/bin/python -u train_secnet_only.py --output-dir checkpoints/secnet_only_official_seed0_v1 --batch-size 16 --num-workers 4 --device cuda --log-every 20 > logs/secnet_only/official_seed0_v1.log 2>&1 < /dev/null &
```

Do not run a second launch while this run is active. Output-directory flock
prevents simultaneous writers. A nonempty run directory requires `--resume`.

```bash
tail -f logs/secnet_only/official_seed0_v1.log
watch -n 2 nvidia-smi
pgrep -af '[t]rain_secnet_only.py'
```

## Resume after interruption

Use the SAME settings, adding:

```bash
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 setsid nohup .venv-eventmamba/bin/python -u train_secnet_only.py --output-dir checkpoints/secnet_only_official_seed0_v1 --resume checkpoints/secnet_only_official_seed0_v1/last.pt --batch-size 16 --num-workers 4 --device cuda --log-every 20 >> logs/secnet_only/official_seed0_v1.log 2>&1 < /dev/null &
```

Resume starts after the last fully saved epoch (an interrupted partial epoch
is replayed). Checkpoints contain model, optimizer, scheduler, RNG, data-loader
generator, configuration, source/manifest hashes. Incompatible resumes fail.
Only load trusted checkpoints. `last.pt` is atomically replaced each epoch;
`final.pt` is retained at completion. This bounds checkpoint disk usage.
`metrics.jsonl` records progress in the checkpoint directory.
`test_final.json` holds the final Test result. An existing result is not rerun.
If Test evaluation itself was interrupted, resume repeats evaluation of the
same final model and fixed seed, without selecting or updating weights.

## Smoke testing

`--smoke` uses only 32 official Train samples at batch 16 for two small epochs;
its evaluation pass also uses those Train samples, NEVER official Test events.
Its artifacts are isolated and must not be used to initialize the formal run.
`--stop-after-epoch 1` (smoke only) allows testing resume into epoch 2.
Reduced-size synthetic CPU tests do not replace the full 8192-point CUDA smoke.
This environment's PyTorch 2.0.1 CPU oneDNN backward produced nonfinite gradients;
the independent CPU entry disables oneDNN. CUDA uses its normal kernels. The
shared Encoder source is unchanged.

```bash
OMP_NUM_THREADS=2 .venv-eventmamba/bin/python -m unittest -v test_secnet_only test_gcstr_fusion
```

## Interpretation

This uses more training samples (14,773) and a different schedule than earlier
fusion train/validation runs (11,818 training samples). Compare on the SAME
official Test split, and explicitly report the training-protocol difference;
do not compare its Test number directly to earlier validation peaks.

## Verification record (2026-10-09 UTC)

- 7 tests passed: 4 standalone tests plus 3 existing fusion regressions.
- Full CPU smoke passed with native CPU kernels, 8192 points, batch 2.
- Full CUDA smoke passed on GPU 1, 8192 points, batch 16, 4 workers;
  resumed from epoch 1 into epoch 2 and completed Train-only evaluation.
  CUDA allocated-memory peak: 4745.63 MiB (~4.64 GiB).
- Final smoke artifacts: `checkpoints/secnet_only_gpu_smoke_v2/`.
  Earlier failed CPU/earlier-version smoke artifacts are retained separately.
- Encoder SHA256 remained
  `a3dcf0b2ff28bf909f2bd5e6f7923d3f6e38c3b506f566d6f7bcf1adcf4f5682`.
- User explicitly approved sharing GPU 1 with the ongoing B experiment.
