"""Archive immutable, atomically replaced A checkpoints without interrupting DDP.

Only reads last.pt. Hard links retain its original inode when the next epoch
atomically replaces last.pt. Training sources/signatures are left unchanged.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import tempfile
import time

import torch


def archive_latest(directory, every=5):
    directory = Path(directory)
    source = directory / 'last.pt'
    if not source.exists():
        return None
    # Pin the exact completed checkpoint before loading; concurrent atomic
    # replacement of last.pt cannot alter this snapshot.
    with tempfile.TemporaryDirectory(prefix='.archive_snapshot_', dir=directory) as temporary:
        snapshot = Path(temporary) / 'snapshot.pt'
        os.link(source, snapshot)
        state = torch.load(snapshot, map_location='cpu', weights_only=False)
        epoch = state['epoch']
        if not isinstance(epoch, int) or epoch < 1:
            raise ValueError('Invalid checkpoint epoch')
        if not all(key in state for key in ('signature', 'models', 'optimizers')):
            raise ValueError('Incomplete A resume checkpoint')
        if state['signature'].get('stage') != 'fusion':
            raise ValueError('Refusing to archive a non-fusion checkpoint')
        record = {'epoch': epoch, 'archived': False, 'complete': bool(state.get('complete'))}
        if epoch % every:
            return record
        target = directory / f'epoch_{epoch:03d}.pt'
        try:
            os.link(snapshot, target)  # Atomic creation; never overwrite a saved archive.
            record.update(archived=True, path=str(target), bytes=target.stat().st_size)
        except FileExistsError:
            previous = torch.load(target, map_location='cpu', weights_only=False)
            if previous['epoch'] != epoch or previous['signature'] != state['signature']:
                raise ValueError(f'Existing archive has a different protocol: {target}')
            record['already_saved'] = True
        return record


def supervisor_running(pid):
    try:
        command = Path(f'/proc/{pid}/cmdline').read_bytes()
    except FileNotFoundError:
        return False
    return b'run_ucf101_a.py' in command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--every', type=int, default=5)
    parser.add_argument('--supervisor-pid', type=int, required=True)
    parser.add_argument('--interval', type=float, default=3.)
    args = parser.parse_args()
    if args.every < 1 or args.interval <= 0:
        parser.error('every and interval must be positive')
    if not args.directory.is_dir():
        parser.error('Training directory must already exist')
    with (args.directory / 'archive.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        def emit(event, **fields):
            record = {'time': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                      'event': event, **fields}
            line = json.dumps(record)
            print(line, flush=True)
            with (args.directory / 'archive.jsonl').open('a') as handle:
                handle.write(line + '\n')
        emit('archiver_started', pid=os.getpid(), every=args.every,
             supervisor_pid=args.supervisor_pid)
        previous_inode = None
        while True:
            running = supervisor_running(args.supervisor_pid)
            source = args.directory / 'last.pt'
            inode = source.stat().st_ino if source.exists() else None
            if inode is not None and inode != previous_inode:
                result = archive_latest(args.directory, args.every)
                # If last.pt changed during the snapshot, leave the observed
                # old inode here so the next poll also processes the new one.
                previous_inode = inode
                if result:
                    emit('checkpoint_archived' if result['archived'] else 'checkpoint_observed', **result)
                    if result['complete']:
                        emit('archiver_finished', reason='final_epoch_saved')
                        return
            if not running:
                emit('archiver_finished', reason='training_supervisor_exited')
                return
            time.sleep(args.interval)


if __name__ == '__main__':
    main()
