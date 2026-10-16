"""Wait for a complete validated joint epoch, then stop only pinned processes.

Designed for a checkpoint-based pause before changing the event-reading backend.
Does not edit training code, restart a job, or modify checkpoint contents.
"""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import signal
import time

import torch

from ucf101_window_data import sha,write_json


def process_identity(pid):
    path=Path('/proc')/str(pid)
    stat=(path/'stat').read_text().rsplit(')',1)[1].split()
    return dict(pid=pid,start_ticks=stat[19],pgid=os.getpgid(pid),
                command=(path/'cmdline').read_bytes().replace(b'\0',b' ').decode())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--epoch',type=int,required=True)
    p.add_argument('--supervisor',type=int,required=True)
    p.add_argument('--training',type=int,required=True)
    p.add_argument('--output',type=Path,default=Path('experiments/ucf101dvs_gcstr_sliding_txtsplit_v2'))
    p.add_argument('--logs',type=Path,default=Path('logs/ucf101_window_joint_txtsplit_v2'))
    args=p.parse_args()
    torch.set_num_threads(1)
    supervisor=process_identity(args.supervisor)
    training=process_identity(args.training)
    if supervisor['pgid']!=args.supervisor or training['pgid']!=args.training:
        raise ValueError('Expected dedicated process groups')
    if 'run_ucf101_window_joint.py' not in supervisor['command'] or 'train_ucf101_window_joint.py' not in training['command']:
        raise ValueError('Wrong process targets')
    if str(args.output) not in supervisor['command'] or str(args.output) not in training['command']:
        raise ValueError('Wrong experiment process targets')
    stop_status=args.logs/'pause_after_epoch.json'
    if stop_status.exists(): raise FileExistsError(stop_status)
    state=dict(status='watching',target_epoch=args.epoch,watcher_pid=os.getpid(),
               supervisor=supervisor,training=training,checkpoint=str(args.output/'checkpoints/joint_last.pt'))
    def emit(**updates):
        state.update(updates,updated_at=datetime.now(timezone.utc).isoformat())
        write_json(stop_status,state)
        print(json.dumps(state),flush=True)
    emit()
    def check_identities():
        for original in (supervisor,training):
            if process_identity(original['pid'])!=original:
                raise RuntimeError('Pinned process exited or identity changed; refusing to signal')
    try:
        while True:
            check_identities()
            report_path=args.output/'reports/summary.json'
            summary=json.loads(report_path.read_text())
            if summary['completed_epoch']<args.epoch:
                time.sleep(1)
                continue
            epoch=summary['completed_epoch']
            emit(status='validating',saved_epoch=epoch)
            checkpoint=args.output/'checkpoints/joint_last.pt'
            bank=torch.load(checkpoint,map_location='cpu',weights_only=False)
            if bank['epoch']!=epoch or set(bank['branches'])!={'cstr3','xt_p','ty_p'}:
                raise RuntimeError('Joint checkpoint is incomplete')
            if sha(checkpoint)!=summary['joint_checkpoint_sha256']:
                raise RuntimeError('Joint checkpoint checksum mismatch')
            if bank['config']['manifest_sha256']!=sha(args.output/'window_manifest.csv'):
                raise RuntimeError('Joint checkpoint uses a different split')
            for view,branch in bank['branches'].items():
                if branch['epoch']!=epoch or not branch['model_state'] or not branch['optimizer_state']['state']:
                    raise RuntimeError(f'{view}: missing model/Adam state')
                path=args.output/'checkpoints'/view/'last.pt'
                if sha(path)!=summary['checkpoint_hashes'][view]['last.pt']:
                    raise RuntimeError(f'{view}: checkpoint checksum mismatch')
            del bank
            check_identities()
            # Freeze before terminating so no new checkpoint can be interrupted.
            os.killpg(supervisor['pgid'],signal.SIGSTOP)
            os.killpg(training['pgid'],signal.SIGSTOP)
            emit(status='stopping',saved_epoch=epoch,checkpoint_verified=True)
            os.killpg(training['pgid'],signal.SIGKILL)
            os.killpg(supervisor['pgid'],signal.SIGKILL)
            pipeline=json.loads((args.logs/'status.json').read_text())
            pipeline.update(status='paused_after_saved_epoch',saved_epoch=epoch,child_pid=None,
                            pause_reason='User requested safe epoch-boundary pause for data backend/layout changes',
                            resume_method='Restore joint_last.pt after explicitly validating backend/config transition',
                            updated_at=datetime.now(timezone.utc).isoformat())
            write_json(args.logs/'status.json',pipeline)
            emit(status='paused',saved_epoch=epoch,checkpoint_verified=True,
                 resume_method='checkpoint; old process memory is released')
            return
    except Exception as exc:
        emit(status='failed',error=f'{type(exc).__name__}: {exc}')
        raise


if __name__=='__main__': main()
