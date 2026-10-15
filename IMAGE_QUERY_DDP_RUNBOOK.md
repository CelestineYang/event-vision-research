# Pure-image learned-query control (DDP)

Purpose: remove the point encoder and Attention1, retain three independent query
banks, each separately attending to cstr3/xt_p/ty_p. Concatenate the three view
outputs per bank, MLP, query residual, mean pool, concatenate banks, classify.
This is a structural ablation, not a parameter-matched control or EV-CGNet reproduction.

## Fixed protocol

- Existing official-train-derived train/val manifest; never reads test samples.
- Existing spatial best checkpoints initialize independent ResNet18 branches.
  All image parameters and BN statistics remain frozen. 49 spatial tokens at 224px.
- Three query banks, 16 queries each, d=128, four heads. All attention independent.
- Same sample/epoch augmentation variant and event transforms as fusion run.
  No point sampling or SECNet forward; no new full-event caches.
- AdamW lr=1e-4, decay=1e-4, gradient clip=1, global batch=16 (8 per GPU).
- No AMP or scheduler, maximum 50 epochs. Early stopping remains validation loss,
  patience 10, to match the previous run. Additionally retain best Top-1 checkpoint.
- Training global shuffled batches are sharded without duplicated/dropped samples;
  the last batch on 11818 samples contains 5 samples/rank. All-reduced loss weighting
  also handles unequal nonempty local batches.
- Validation indices are rank-strided, no padding: full val samples=2955 exactly.
  Validation runs through the underlying model without DDP forward collectives;
  final metrics are all-reduced. Progress rows marked rank0_only are NOT full-val
  metrics. Epoch summary metrics are global on both phases.
- Only rank 0 writes best_loss.pt, best_top1.pt, last.pt and metrics.jsonl.
- Resume is epoch-boundary, deterministic epoch seeds; signature locks model,
  manifest content, batch, world size, augmentation and optimizer settings.
  GPU floating-point execution is not promised bit-identical across hardware.
- Constructor loads no point pretraining and never uses prior fusion smoke weights.

## Start (only after both GPUs are available for your use)

```bash
mkdir -p logs/gcstr_image_query
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 nohup /root/anaconda3/envs/torch201/bin/python -m torch.distributed.run --standalone --nproc_per_node=2 train_gcstr_image_query_ddp.py --manifest gcstr_splits_seed0.csv --output-dir gcstr_checkpoints/image_query_ddp_full_seed0_v1 --epochs 50 --batch-size 16 --num-workers 1 --augmentation spatial --augmentation-count 5 --learning-rate 1e-4 --weight-decay 1e-4 --patience 10 --seed 0 --log-every 20 > logs/gcstr_image_query/full_seed0_v1.log 2>&1 &
echo $!
tail -f logs/gcstr_image_query/full_seed0_v1.log
```

One worker per GPU, two total. Default no cache, matching previous fusion run.
Optional --cache-dir gcstr_cache reuses existing image-only caches (not necessary
for this run); cache misses create image arrays, not full event streams.

New output directory required. Do not rerun the start command over an existing log.
For resume use identical arguments plus
`--resume gcstr_checkpoints/image_query_ddp_full_seed0_v1/last.pt`, with a NEW log
filename (e.g. resume1.log) to preserve the initial log. Do not resume a diagnostic
checkpoint for the official experiment. --epochs denotes total target epochs.

## Checks

`python test_gcstr_image_query.py` covers CPU forward/backward, frozen BN,
non-duplicated training shards and exact odd-size validation partition.
Short DDP runs use --max-train-batches 2 --max-val-batches 2 in a separate output
directory. Diagnostic metrics are not experimental results.

Model code: gcstr_image_query.py. Trainer: train_gcstr_image_query_ddp.py.
Previous fusion/baseline files remain unchanged.
