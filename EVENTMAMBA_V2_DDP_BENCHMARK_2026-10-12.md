# EventMamba V2 two-GPU feasibility and throughput check

Date: 2026-10-12. User requested trying two cards after GPU 0 became idle.

## Result

Two-GPU NCCL/DDP with SyncBatchNorm successfully completed real training,
finite-gradient checks, two smoke epochs, checkpoint saving and final
Train-only window/sample voting evaluation. No official Test events used.
Artifacts: `checkpoints/eventmamba_v2_ddp_smoke_v1/`.

Throughput probes used the epoch 23 checkpoint, identical official Train cache,
float32 and global batch 32. Two GPUs use 16 examples per rank; single GPU uses
32. Both had 10 warm-up steps, two loader workers per rank, identical optimizer
state, finite-gradient checks, and timed forward/backward/optimizer steps.
Benchmark weights were discarded and never saved into the formal experiment.
The formal process (PID 5209) was temporarily SIGSTOP-paused to remove compute
competition, retaining its live epoch 24 state, then resumed with SIGCONT.

| Measured steps | GPUs | Seconds/step | Windows/second |
|---|---|---|---|
| 120 | 1 | 0.238705 | 134.057 |
| 120 | 2 | 0.251412 | 127.281 |
| 300 | 1 | 0.232265 | 137.774 |
| 300 | 2 | 0.249765 | 128.120 |

The longer probe shows two-GPU steps about 7.5% slower (throughput about 7.0%
lower). GPU topology is NODE across PCIe host bridges, not NVLink. DDP logs a
nonfatal attention-gradient stride warning. Frequent SyncBN/gradient collectives
and the unchanged iterative FPS launch overhead are plausible contributors;
their individual costs were not separately profiled. These short probes do not
prove every possible batch/backend would be slower, but provide no basis for
migrating the current global-batch32 experiment for speed.

Formal training therefore remains single-GPU on GPU 1, with the original model,
optimizer, scheduler, seed, global batch, 350-epoch protocol and final-only Test
policy. No source file included in the checkpoint signature was modified, no
checkpoint converted, and no partial epoch discarded. GPU 0 was released after
the probes. A larger GLOBAL batch would change the experimental protocol and
was not attempted.

Reproducible probe entry: `benchmark_eventmamba_v2_ddp.py`.
