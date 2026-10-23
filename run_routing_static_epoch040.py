"""Explicit single-GPU, fail-closed evidence export and Static experiment."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
OUT = Path('/remote-home/datasets/event_project_cache/routing_epoch040_static_20261022_v3')
CHECKPOINT = '/remote-home/datasets/event_project_cache/ucf101_filename_v2_fusion_A_seed0_v1_checkpoints/epoch_040.pt'


def main():
    OUT.mkdir(exist_ok=False)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='4', PYTHONUNBUFFERED='1')
    def run(stage, command):
        print(stage, flush=True)
        (OUT/'status.json').write_text(json.dumps({'stage': stage, 'state': 'running', 'time': time.time()}))
        with (OUT/(stage+'.log')).open('w') as log:
            subprocess.run([sys.executable, '-u', *command], cwd=ROOT, env=env,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
    common = ['export_fusion_evidence.py', '--checkpoint', CHECKPOINT,
              '--roles', 'configs/fusion_routing_roles.original_test_as_val.json',
              '--selection-protocol', 'original_test_as_val', '--workers', '4', '--device', 'cuda:0']
    try:
        run('unit_tests', ['-m', 'unittest', 'test_fusion_routing', 'test_ucf101_window_gcstr', 'test_ucf101_event_store'])
        for source in ('train', 'test'):
            run('regression_'+source, common+['--source-split', source, '--max-videos', '30', '--batch-size', '64', '--integration-regression', '--regression-batches', '3'])
        # Complete-video export exercises the batch-64 loader before the full dataset.
        for source in ('train', 'test'):
            run('smoke_'+source, common+['--source-split', source, '--max-videos', '10', '--batch-size', '64',
                                        '--output', str(OUT/('smoke_'+source))])
        config = json.loads((ROOT/'configs/fusion_routing.json').read_text())
        config.update(epochs=80, batch_size=32)
        (OUT/'run_config.json').write_text(json.dumps(config, indent=2))
        smoke = dict(config, epochs=1)
        (OUT/'smoke_config.json').write_text(json.dumps(smoke, indent=2))
        def fit(train, val, config_path, output):
            return ['train_fusion_routing.py', 'fit', '--mode', 'static', '--config', str(config_path),
                    '--cache', str(train), '--val-cache', str(val), '--train-role', 'gate_train',
                    '--val-role', 'gate_val', '--selection-protocol', 'original_test_as_val',
                    '--device', 'cuda:0', '--output', str(output)]
        run('smoke_static', fit(OUT/'smoke_train', OUT/'smoke_test', OUT/'smoke_config.json', OUT/'smoke_static'))
        for source in ('train', 'test'):
            run('export_'+source, common+['--source-split', source, '--batch-size', '64', '--output', str(OUT/source)])
        from routing_data import EvidenceCache
        for source, count in [('train', 108066), ('test', 27384)]:
            if len(EvidenceCache(OUT/source).rows) != count:
                raise RuntimeError('Full export count mismatch: '+source)
        run('static', fit(OUT/'train', OUT/'test', OUT/'run_config.json', OUT/'static'))
        (OUT/'status.json').write_text(json.dumps({'state': 'complete', 'time': time.time()}))
    except BaseException as exc:
        (OUT/'status.json').write_text(json.dumps({'state': 'failed', 'error': repr(exc), 'time': time.time()}))
        raise


if __name__ == '__main__':
    main()
