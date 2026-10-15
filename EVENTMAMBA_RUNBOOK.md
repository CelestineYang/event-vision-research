# EventMamba V1 encoder integration

This is an encoder adaptation, not a reproduction of the original windowed
classification experiment. Do not infer DailyDVS-200 performance from its
12-class Daily DVS results.

## Architecture and protocol

- EventMamba source commit: `7f61fd397cf69f10f0f92a182109d36cd7b6fdbb`.
- V1, coordinate KNN, anchor normalization, K=24, BiMamba `bimamba_type=v2`.
- Our 8192-point whole-recording input is retained, NOT official sliding windows.
- Input `[B,4,N]` x,y,t,p becomes chronological t,x,y; polarity is omitted.
- Groups 1024/512/256 (upstream commented larger configuration, not its active
  512/256/128 default); stage channels 64/128/256.
- Actual FPS center coordinates t,x,y feed 3D position embeddings. Random FPS
  starts follow the existing training/validation seed policy.
- Frozen task-trained image encoders and prototype fusion are unchanged.
- Encoder randomly initialized. Old SECNet fusion checkpoints are incompatible.
- Existing train/val split stays fixed; official test is never used here.
- The old trainer still defaults to SECNet; select `--point-encoder eventmamba`.
- Single-GPU integration first. This trainer is NOT a DDP entry point.

## Isolated environment

`.venv-eventmamba` uses torch201's Python and system site packages; new packages
are installed only in this venv. The original torch201 environment is unchanged.

Compatible candidate dependency source (not confirmed as the author's exact fork):
`hustvl/Vim` commit `dd0358ad1e42701f22afbefa0717cc8825cf9f45`, `mamba-1p1p1`.
Important: its setup may download a vanilla upstream wheel without BiMamba.
Force a source build and check the signature; successful installation alone
does NOT prove compatibility.

```bash
/root/anaconda3/envs/torch201/bin/python -m venv --system-site-packages .venv-eventmamba
.venv-eventmamba/bin/python -m pip install --index-url https://pypi.org/simple ninja einops==0.7.0 transformers==4.35.2
.venv-eventmamba/bin/python -m pip install --no-deps --no-build-isolation --index-url https://pypi.org/simple causal-conv1d==1.4.0
MAMBA_FORCE_BUILD=TRUE MAX_JOBS=2 .venv-eventmamba/bin/python -m pip install --no-cache-dir --force-reinstall --no-deps --no-build-isolation 'https://github.com/hustvl/Vim/archive/dd0358ad1e42701f22afbefa0717cc8825cf9f45.zip#subdirectory=mamba-1p1p1'
```

The pinned Vim revision uses the newer causal-conv1d API, hence 1.4.0 rather
than 1.1.1. For this server's RTX 4090 (sm_89), the local source build in
`/tmp/eventmamba-vim-dd0358ad-build` narrows setup.py's hardcoded seven GPU
architectures to `arch=compute_89,code=sm_89`. No Python model or CUDA kernel
algorithm is changed. The generic command above builds all upstream targets.
The locally built binary is specific to this GPU architecture/environment.

## Tests before training

CPU tests deliberately inject a linear test mixer to verify interfaces, ordering,
coordinate selection, polarity omission, finite gradients and frozen image BN.
They do not certify Mamba kernels. Separate real CUDA tests use actual BiMamba.

```bash
OMP_NUM_THREADS=2 /root/anaconda3/envs/torch201/bin/python -m unittest test_eventmamba_encoder test_gcstr_fusion -v
CUDA_VISIBLE_DEVICES=0 .venv-eventmamba/bin/python smoke_eventmamba.py
CUDA_VISIBLE_DEVICES=0 .venv-eventmamba/bin/python smoke_eventmamba.py --full-size
```

Only after real CUDA smoke passes, test a real train and validation batch:

```bash
CUDA_VISIBLE_DEVICES=0 .venv-eventmamba/bin/python -u train_gcstr_fusion.py \
  --point-encoder eventmamba \
  --manifest gcstr_splits_seed0.csv \
  --output-dir gcstr_checkpoints/eventmamba_smoke_v1 \
  --epochs 1 --batch-size 2 --num-workers 0 --point-count 8192 \
  --augmentation spatial --max-train-batches 1 --max-val-batches 1 \
  --log-every 1 --device cuda
```

Use a fresh output directory for each new run. Do not resume SECNet checkpoints.
This is not yet an instruction to start full training or choose hyperparameters.
