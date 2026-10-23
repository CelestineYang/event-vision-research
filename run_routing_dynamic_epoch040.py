"""Launch the three authorized dynamic fits on the same frozen evidence."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
BASE = Path('/remote-home/datasets/event_project_cache/routing_epoch040_static_20261022_v3')
MODES = ('current', 'gru', 'conditional')


def main():
    for mode in MODES:
        if (BASE/mode).exists() or (BASE/(mode+'.log')).exists():
            raise FileExistsError('Refusing to overwrite '+mode)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='2', PYTHONUNBUFFERED='1')
    jobs = {}
    for mode in MODES:
        command = [sys.executable, '-u', 'train_fusion_routing.py', 'fit',
                   '--config', str(BASE/'run_config.json'), '--mode', mode,
                   '--cache', str(BASE/'train'), '--val-cache', str(BASE/'test'),
                   '--static-init', str(BASE/'static/best_val_nll.pt'),
                   '--train-role', 'gate_train', '--val-role', 'gate_val',
                   '--selection-protocol', 'original_test_as_val', '--device', 'cuda:0',
                   '--output', str(BASE/mode)]
        with (BASE/(mode+'.log')).open('x') as log:
            proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        jobs[mode] = proc
    while True:
        state = {mode:dict(pid=p.pid, exit_code=p.poll(),
                         state='running' if p.poll() is None else 'complete' if p.returncode==0 else 'failed')
                 for mode,p in jobs.items()}
        temporary = BASE/'dynamic_status.tmp'
        temporary.write_text(json.dumps(dict(time=time.time(), jobs=state), indent=2))
        temporary.replace(BASE/'dynamic_status.json')
        if all(p.poll() is not None for p in jobs.values()):
            if any(p.returncode for p in jobs.values()):
                raise RuntimeError('One or more dynamic runs failed; see per-mode logs')
            return
        time.sleep(5)


if __name__ == '__main__':
    main()
