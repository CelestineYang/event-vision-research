# Fusion framework v1

This is our custom classification architecture, not official EV-CGNet code or
a strict reproduction of its keypoint detection model. No training has been run.

## Confirmed configuration

- DailyDVS-200, existing manifest/splits, no test-driven tuning.
- Independent cstr3 / xt_p / ty_p ResNet18 encoders, load existing spatial best
  checkpoints explicitly with `model.load_image_checkpoints(paths)` before training.
  Constructor intentionally does NOT download or load weights automatically.
- Frozen image backbone including BatchNorm; trainable token adapters/positions.
- SECNet input `[B,4,8192]`, channel order x,y,t,p; spatial division by sensor
  width/height, time normalized over full augmented event range, polarity -1/+1.
- Random point sampling, replacement for short samples, time order preserved.
  Evaluation dataset sampling deterministic per sample. SECNet FPS retains the
  official random start: evaluation must also reset torch seed for reproducibility.
- SECNet width 64, stages 132/268/540, centers 1024/512/256, 24 neighbors.
- d=128, 16 prototypes per stage, 4 heads, zero attention dropout, pre-LayerNorm.
  Three independent Attention1 and nine independent Attention2 modules.
- Per-view learned 2D positional MLP, per-stage learned 4D coordinate MLP;
  positional information added before attention normalization. This is not an
  explicit geometric correspondence mechanism.
- No attention-internal residual or FFN. Stage fusion MLP 384->256->128,
  GELU, external A residual; mean over prototypes then concatenate stages;
  linear 384->200 classifier, cross-entropy on logits.
- Point backbone randomly initialized; gradient flows from all three interactions;
  no feedback into later point stages.

## Files

- `secnet_encoder.py`: feature-only adaptation of official SECNet commit
  0638bb18776f5cdb44912770f8f117c96e59b684. Keeps 3-coordinate FPS,
  feature KNN, coordinate mean update, channel FFT and sequence FFT. Removes
  unused parameters/head, parameterizes sizes, specifies inverse FFT length.
  It is not state-dict-compatible with upstream classification checkpoints.
- `gcstr_secnet_fusion.py`: spatial tokens, prototype interactions, classifier.
- `gcstr_fusion_dataset.py`: jointly constructs all views and points after the
  same spatial augmentation variant. Reuses existing variant seed convention.
- `test_gcstr_fusion.py`: small CPU forward/backward and sampling checks.

## Current limits / next steps

The dataset intentionally has no joint caching yet and rejects cache_dir rather
than silently accepting mismatched cached views. Do not use it for full training
without measuring preprocessing throughput. Existing baseline code/cache is untouched.
Optimizer, training CLI, resume/evaluation entry points and joint cache are not
implemented in this step. Tiny tests are engineering checks, not point-only
experiments and not evidence of accuracy. Default-size GPU memory is unmeasured.

Run tests without GPU or pretrained downloads:

```bash
/root/anaconda3/envs/torch201/bin/python test_gcstr_fusion.py
```
