"""Wait for sustained idle GPU, run GM smoke + resume once, then exit.

No CUDA allocations while waiting. Advisory locks coordinate this launcher only;
this is NOT a cluster scheduler and cannot reserve a GPU against other users.
"""
import argparse
import csv
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent


def emit(event, **fields):
    print(json.dumps(dict(time=datetime.now(timezone.utc).isoformat(), event=event, **fields)), flush=True)


def query_gpus():
    def query(fields, kind):
        result = subprocess.run(['nvidia-smi', f'--query-{kind}={fields}', '--format=csv,noheader,nounits'],
                                capture_output=True, text=True, check=True, timeout=15)
        return list(csv.reader(result.stdout.splitlines(), skipinitialspace=True))
    rows = query('index,uuid,memory.used,utilization.gpu', 'gpu')
    processes = query('gpu_uuid,pid', 'compute-apps')
    busy = set()
    for row in processes:
        if len(row) != 2 or not row[0].startswith('GPU-'):
            raise ValueError('Unknown process-query output; refusing idle inference')
        int(row[1])
        busy.add(row[0])
    result = {}
    for row in rows:
        if len(row) != 4 or not row[1].startswith('GPU-'):
            raise ValueError('Unknown GPU query output')
        index, uuid, memory, utilization = row
        result[int(index)] = dict(uuid=uuid, memory_mib=int(memory), utilization=int(utilization),
                                  has_compute_process=uuid in busy)
    if not result:
        raise ValueError('No GPUs returned')
    return result


def is_idle(gpu, max_memory, max_utilization):
    return (not gpu['has_compute_process'] and gpu['memory_mib'] <= max_memory
            and gpu['utilization'] <= max_utilization)


class IdleTracker:
    def __init__(self, duration):
        self.duration, self.since = duration, {}

    def update(self, snapshot, indices, now, memory, utilization):
        ready = []
        for index in indices:
            gpu = snapshot.get(index)
            if gpu is None or not is_idle(gpu, memory, utilization):
                self.since.pop(index, None)
                continue
            old = self.since.get(index)
            if old is None or old[0] != gpu['uuid']:
                self.since[index] = (gpu['uuid'], now)
            if now-self.since[index][1] >= self.duration:
                ready.append(index)
        return ready


def lock_file(path):
    stream = Path(path).open('a')
    try:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        stream.close()
        return None
    return stream


def smoke_commands(python, output):
    base = [str(python), '-u', str(ROOT/'train_gcstr_feature_memory.py'),
            '--config', str(ROOT/'configs/feature_memory/GM.json'),
            '--output', str(output), '--device', 'cuda:0', '--smoke']
    return [base, base + ['--resume']]


def run_smoke(python, output, uuid):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=uuid, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
    for phase, command in zip(('fresh', 'resume'), smoke_commands(python, output)):
        logfile = output.parent/f'launcher_{phase}.log'
        emit('smoke_start', phase=phase, gpu_uuid=uuid, log=str(logfile), command=command)
        with logfile.open('x') as log:
            # Parent remains responsible for terminating only its own child on cancel.
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True)
            try:
                code = child.wait()
            except BaseException:
                child.terminate()
                try:
                    child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGTERM)
                    child.wait(timeout=20)
                raise
        if code:
            raise RuntimeError(f'{phase} smoke exited {code}; see {logfile}; no automatic retry')
    status = json.loads((output/'status.json').read_text())
    if status['completed_epoch'] != 2 or not (output/'initial_regression.json').is_file():
        raise RuntimeError('Smoke did not complete initial regression and two epoch-boundary runs')
    emit('smoke_complete', output=str(output), completed_epoch=2, formal_training_started=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpus', type=int, nargs='+', default=[0, 1])
    parser.add_argument('--idle-seconds', type=float, default=120)
    parser.add_argument('--interval', type=float, default=10)
    parser.add_argument('--max-memory-mib', type=int, default=512)
    parser.add_argument('--max-utilization', type=int, default=5)
    parser.add_argument('--timeout-hours', type=float, default=24, help='Maximum waiting time; exits without training')
    parser.add_argument('--python', type=Path, default=ROOT/'.venv-eventmamba/bin/python')
    parser.add_argument('--output', type=Path, default=ROOT/'experiments/feature_memory_GM_auto_smoke_v1')
    parser.add_argument('--check-only', action='store_true', help='Read GPU status once; no waiting, files or children')
    args = parser.parse_args()
    if (args.idle_seconds < 1 or not 1 <= args.interval <= 60 or args.timeout_hours <= 0
            or args.max_memory_mib < 0 or not 0 <= args.max_utilization <= 100 or min(args.gpus) < 0):
        parser.error('Invalid wait/threshold/GPU arguments')
    if args.check_only:
        emit('snapshot', gpus=query_gpus())
        return
    # Keep virtualenv executable path (do NOT resolve its symlink to another environment).
    python = args.python.absolute()
    output = args.output.absolute()
    if not python.is_file() or not os.access(python, os.X_OK):
        raise FileNotFoundError(python)
    if 'smoke' not in output.name:
        raise ValueError('Output name must contain smoke')
    output.mkdir(parents=True, exist_ok=True)
    own_lock = lock_file(output/'waiter.lock')
    if own_lock is None:
        raise RuntimeError('Another waiter owns this output')
    if any(p.name != 'waiter.lock' for p in output.iterdir()):
        raise FileExistsError('Use a fresh smoke output; existing results are never overwritten')
    # Trainer refuses nonempty output. Launcher metadata/logs therefore live in a
    # sibling directory; the dedicated child directory remains empty until start.
    child_output = output/'smoke'
    tracker = IdleTracker(args.idle_seconds)
    started = time.monotonic()
    emit('waiting', pid=os.getpid(), gpus=args.gpus, idle_seconds=args.idle_seconds,
         timeout_hours=args.timeout_hours, output=str(child_output), exclusive_reservation=False)
    selected_lock = None
    try:
        while time.monotonic()-started < args.timeout_hours*3600:
            try:
                snapshot = query_gpus()
                if not set(args.gpus).issubset(snapshot):
                    raise ValueError('A requested GPU is missing')
                ready = tracker.update(snapshot, args.gpus, time.monotonic(), args.max_memory_mib, args.max_utilization)
                emit('poll', gpus={i: snapshot[i] for i in args.gpus}, ready=ready)
                for index in ready:
                    uuid = snapshot[index]['uuid']
                    selected_lock = lock_file(Path('/tmp')/f'event-feature-smoke-{os.getuid()}-{uuid}.lock')
                    if selected_lock is None:
                        continue
                    final = query_gpus().get(index)
                    if final is None or final['uuid'] != uuid or not is_idle(final, args.max_memory_mib, args.max_utilization):
                        selected_lock.close()
                        selected_lock = None
                        tracker.since.pop(index, None)
                        continue
                    emit('selected', gpu=index, gpu_uuid=uuid)
                    # Launch outside the polling exception handler: failure must exit,
                    # never loop back and start another experiment.
                    break
                if selected_lock is not None:
                    break
            except (subprocess.SubprocessError, ValueError, OSError) as error:
                if selected_lock is not None:
                    selected_lock.close()
                    selected_lock = None
                tracker.since.clear()  # unknown status invalidates the entire idle interval
                emit('query_failed', error=str(error))
            time.sleep(args.interval)
        if selected_lock is None:
            emit('wait_timeout', training_started=False)
            return
        run_smoke(python, child_output, uuid)
    finally:
        if selected_lock is not None:
            selected_lock.close()
        own_lock.close()


def cancel(signum, frame):
    raise KeyboardInterrupt


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, cancel)
    try:
        main()
    except KeyboardInterrupt:
        emit('cancelled')
        sys.exit(130)
    except Exception as error:
        emit('failed', error=str(error))
        sys.exit(1)
