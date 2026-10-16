"""Detached, fail-closed preparation -> smoke -> formal training pipeline."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

from ucf101_window_data import OUTPUT, checked_audit, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=OUTPUT)
    parser.add_argument('--log-dir',type=Path,default=Path('logs/ucf101_window_joint'))
    parser.add_argument('--prepared',action='store_true',
                        help='Use existing passed full audit/splits; still run cache and Smoke before fresh training')
    parser.add_argument('--resume-training',action='store_true',help='Resume the saved joint checkpoint; do not rebuild data')
    parser.add_argument('--event-store',type=Path,help='Verified event mmap directory for resumed training')
    args = parser.parse_args()
    project = Path(__file__).resolve().parent
    os.chdir(project)
    output = args.output.resolve()
    logs = args.log_dir.resolve()
    logs.mkdir(parents=True,exist_ok=True)
    with (logs/'pipeline.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        resume_epoch=None
        previous_state=None
        if args.resume_training:
            audit=checked_audit(output)
            if audit['config'].get('audit_scope')!='full' or args.event_store is None:
                raise ValueError('Resume requires full audit and --event-store')
            if not (output/'checkpoints/joint_last.pt').is_file(): raise FileNotFoundError('Missing joint checkpoint')
            from ucf101_event_store import EventStore
            EventStore(args.event_store,output)
            resume_epoch=json.loads((output/'reports/summary.json').read_text())['completed_epoch']
            status_path=logs/'status.json'
            if status_path.exists(): previous_state=json.loads(status_path.read_text())
        elif args.prepared:
            audit=checked_audit(output)
            if audit['config'].get('audit_scope')!='full':
                raise ValueError('Prepared pipeline requires a full audit')
            if (output/'checkpoints').exists() or (output/'config.json').exists():
                raise FileExistsError('Fresh training refuses existing formal checkpoints/config')
        elif output.exists():
            raise FileExistsError(f'Fresh pipeline requires a new output directory: {output}')
        env = dict(os.environ,CUDA_VISIBLE_DEVICES='0,1',PYTHONUNBUFFERED='1')
        python = sys.executable
        common = ['--output',str(output)]
        distributed = [python,'-u','-m','torch.distributed.run','--standalone','--nproc_per_node=2',
                       'train_ucf101_window_joint.py']
        stages = [
            ('audit',[python,'-u','ucf101_window_data.py','prepare',*common]),
            ('cache',[python,'-u','ucf101_window_data.py','cache',*common]),
            ('smoke',[*distributed,'--smoke',*common]),
            ('train',[*distributed,'--train',*common]),
        ]
        if args.prepared: stages=stages[1:]
        if args.resume_training:
            stages=[('train',[*distributed,'--train','--resume','--allow-data-backend-transition',
                             '--event-store',str(args.event_store.resolve()),*common])]
        state = dict(supervisor_pid=os.getpid(),output=str(output),log_dir=str(logs),
                     devices=[0,1],epochs=50,global_batch_per_branch=64,
                     prepared=args.prepared or args.resume_training,fresh_imagenet_initialization=not args.resume_training,
                     resumed_from_epoch=resume_epoch,event_store=str(args.event_store.resolve()) if args.event_store else None,
                     history=previous_state.get('history',[]) if previous_state else [])
        def save(**updates):
            state.update(updates,updated_at=datetime.now(timezone.utc).isoformat())
            write_json(logs/'status.json',state)
        save(status='starting')
        for name,command in stages:
            if name!='audit': checked_audit(output)
            path = logs/f'{name}.log'
            with path.open('a' if args.resume_training else 'x') as stream:
                if args.resume_training:
                    stream.write(json.dumps(dict(event='pipeline_resume',saved_epoch=resume_epoch,
                                                 next_epoch=resume_epoch+1,command=command,
                                                 time=datetime.now(timezone.utc).isoformat()))+'\n')
                    stream.flush()
                process = subprocess.Popen(command,cwd=project,env=env,stdout=stream,
                                           stderr=subprocess.STDOUT,start_new_session=True)
                save(status='running',stage=name,child_pid=process.pid,command=command,log=str(path))
                code = process.wait()
            state['history'].append(dict(stage=name,exit_code=code,log=str(path)))
            if code:
                save(status='failed',stage=name,exit_code=code,child_pid=None)
                return code
        save(status='complete',child_pid=None)
        return 0


if __name__=='__main__':
    raise SystemExit(main())
