"""Supervised UCF A pipeline: image pretraining -> A fusion -> final Test."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-root', type=Path,
                        default=Path('ucf101_checkpoints/A_official_train_test_ddp_v3'))
    parser.add_argument('--devices', '--device', dest='devices', default='0,1',
                        help='Comma-separated physical GPUs; defaults to two cards')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--fusion-epochs', type=int,
                        help='Independent A epoch budget; defaults to --epochs')
    parser.add_argument('--image-batch', type=int, default=64, help='GLOBAL image batch size')
    parser.add_argument('--fusion-batch', type=int, default=16, help='GLOBAL A batch size')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--stop-after-images', action='store_true',
                        help='Train UCF gCSTR branches only; defer formal A training')
    args = parser.parse_args()
    devices = args.devices.split(',')
    if not all(d.isdigit() for d in devices) or len(set(devices)) != len(devices):
        parser.error('Provide distinct comma-separated GPU indices')
    if args.epochs < 1 or (args.fusion_epochs is not None and args.fusion_epochs < 1):
        parser.error('Epoch count must be positive')
    if any(b % len(devices) or b // len(devices) < 2 for b in (args.image_batch, args.fusion_batch)):
        parser.error('Global batch must divide GPU count, with >=2 examples per rank')
    root = Path(__file__).resolve().parent
    args.output_root.mkdir(parents=True, exist_ok=True)
    lock = (args.output_root / 'pipeline.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=args.devices, OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1')
    def emit(record):
        record = {'time': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **record}
        line = json.dumps(record)
        print(line, flush=True)
        with (args.output_root / 'pipeline.jsonl').open('a') as handle:
            handle.write(line + '\n')
    stages = ('images', 'image_test') if args.stop_after_images else ('images', 'image_test', 'fusion', 'test')
    for stage in stages:
        directory = args.output_root / stage
        directory.mkdir(parents=True, exist_ok=True)
        command = stage_command(args, stage, root, sys.executable)
        if args.resume:
            command.append('--resume')
        if args.smoke:
            command += ['--smoke', '--workers', '2']
        with (directory / 'console.log').open('a') as log:
            child = subprocess.Popen(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
            (args.output_root / 'process.json').write_text(json.dumps(
                {'supervisor_pid': os.getpid(), 'stage': stage, 'training_pid': child.pid, 'command': command}, indent=2) + '\n')
            emit({'event': 'stage_start', 'stage': stage, 'pid': child.pid})
            while True:
                try:
                    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu',
                                                   '--format=csv,noheader,nounits'], text=True, timeout=10).strip()
                except (OSError, subprocess.SubprocessError) as exc:
                    gpu = str(exc)
                metrics = directory / 'metrics.jsonl'
                age = time.time() - metrics.stat().st_mtime if metrics.exists() else None
                emit({'event': 'heartbeat', 'stage': stage, 'pid': child.pid, 'exit_code': child.poll(),
                      'gpu': gpu, 'metrics_age_seconds': age})
                if child.poll() is not None:
                    if child.returncode:
                        emit({'event': 'pipeline_failed', 'stage': stage, 'exit_code': child.returncode})
                        raise SystemExit(child.returncode)
                    break
                try:
                    child.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    pass
            emit({'event': 'stage_finished', 'stage': stage})
    if args.stop_after_images:
        emit({'event': 'image_pretraining_finished', 'next_stage': 'fusion',
              'message': 'Final image checkpoints and Test results saved; fusion not launched'})
        return
    result = args.output_root / 'test' / ('smoke_eval.json' if args.smoke else 'test_final.json')
    if not result.exists():
        raise RuntimeError('Pipeline exited without evaluation result')
    emit({'event': 'pipeline_finished', 'result': str(result)})


def stage_command(args, stage, root, executable):
    count = len(args.devices.split(','))
    fusion_epochs = getattr(args, 'fusion_epochs', None)
    epochs = fusion_epochs if stage in ('fusion', 'test') and fusion_epochs is not None else args.epochs
    # Training uses all specified cards; final Test deliberately stays single-process.
    if count > 1 and stage in ('images', 'fusion'):
        command = [executable, '-m', 'torch.distributed.run', '--standalone', f'--nproc_per_node={count}']
    else:
        command = [executable, '-u']
    return command + [str(root / 'train_ucf101_a.py'), '--stage', stage,
                      '--output-root', str(args.output_root), '--epochs', str(epochs),
                      '--image-batch', str(args.image_batch), '--fusion-batch', str(args.fusion_batch)]


if __name__ == '__main__':
    main()
