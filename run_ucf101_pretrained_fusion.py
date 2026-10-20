"""Detached supervisor for the authorized pretrained V2 fusion experiment."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

from ucf101_window_data import sha, write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    project=Path(__file__).resolve().parent
    os.chdir(project)
    output=project/'experiments/ucf101_filename_v2_fusion_A_seed0_v1'
    logs=project/'logs/ucf101_filename_v2_fusion_A_seed0_v1'
    checkpoints=Path('/remote-home/datasets/event_project_cache/ucf101_filename_v2_fusion_A_seed0_v1_checkpoints')
    smoke=project/'experiments/ucf101_filename_v2_fusion_A_smoke_v3'
    status=json.loads((smoke/'status.json').read_text())
    config=json.loads((smoke/'config.json').read_text())
    if status['status']!='complete' or status['completed_epoch']!=2:
        raise RuntimeError('Two-epoch DDP/resume Smoke has not completed')
    for file,digest in config['code'].items():
        if sha(project/file)!=digest: raise RuntimeError(f'Source changed since Smoke: {file}')
    if not args.resume and ((output/'config.json').exists() or (checkpoints/'last.pt').exists()):
        raise FileExistsError('Use --resume for an existing run')
    logs.mkdir(parents=True,exist_ok=True)
    checkpoints.mkdir(parents=True,exist_ok=True)
    output.mkdir(parents=True,exist_ok=True)
    link=output/'checkpoints'
    if not link.exists(): link.symlink_to(checkpoints,target_is_directory=True)
    with (logs/'supervisor.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        command=[sys.executable,'-u','-m','torch.distributed.run','--standalone',
                 '--nproc_per_node=2','train_ucf101_pretrained_fusion.py',
                 '--output',str(output),'--checkpoint-dir',str(checkpoints),
                 '--epochs','150','--batch-size','16','--workers','4']
        if args.resume: command.append('--resume')
        state=dict(supervisor_pid=os.getpid(),command=command,output=str(output),
                   checkpoints=str(checkpoints),log=str(logs/'train.log'),devices=[0,1])
        def update(**fields):
            state.update(fields,time=datetime.now(timezone.utc).isoformat())
            write_json(logs/'status.json',state)
        with (logs/'train.log').open('a' if args.resume else 'x') as stream:
            child=subprocess.Popen(command,cwd=project,
                env=dict(os.environ,CUDA_VISIBLE_DEVICES='0,1',OMP_NUM_THREADS='2',PYTHONUNBUFFERED='1'),
                stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            update(status='running',child_pid=child.pid)
            code=child.wait()
        update(status='complete' if code==0 else 'failed',exit_code=code,child_pid=None)
        return code


if __name__=='__main__': raise SystemExit(main())
