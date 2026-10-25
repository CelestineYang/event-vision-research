"""Copy an explicit self-contained feature experiment bundle; never move originals."""
import ast
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = Path(__file__).resolve().parent


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def main():
    target = Path(sys.argv[1])
    resume = '--resume' in sys.argv[2:]
    if resume and (target/'COMPLETE.json').exists():
        raise RuntimeError('Already completed; refusing overwrite')
    target.mkdir(parents=True, exist_ok=resume)
    cfg = json.loads((ROOT/'configs/feature_memory/GM.json').read_text())
    old = json.loads((ROOT/'experiments/ucf101_filename_v2_fusion_A_seed0_v1/config.json').read_text())
    seeds = set(old['code']) | set(['gcstr_feature_memory_fusion.py', 'gcstr_video_sequence_dataset.py',
        'train_gcstr_feature_memory.py', 'evaluate_gcstr_feature_memory.py', 'test_gcstr_feature_memory.py',
        'prepare_feature_migration_config.py'])
    visited = set()
    while seeds:
        name = seeds.pop()
        if name in visited:
            continue
        path = ROOT/name
        visited.add(name)
        for node in ast.walk(ast.parse(path.read_text())):
            modules = ([node.module] if isinstance(node, ast.ImportFrom) and node.module else
                       [a.name for a in node.names] if isinstance(node, ast.Import) else [])
            for module in modules:
                file = module.split('.')[0]+'.py'
                if (ROOT/file).is_file() and file not in visited:
                    seeds.add(file)
    files = [(ROOT/name, Path('code')/name) for name in sorted(visited)]
    for name in ['FEATURE_GATE_MEMORY_RUNBOOK.md', 'CODEX_FEATURE_GATE_MEMORY_JOINT_TASK.md',
                 'train_ucf.txt', 'test_ucf(1).txt', 'eventmamba_v2_upstream/LICENSE']:
        files.append((ROOT/name, Path('code')/name))
    for mode in ('B', 'G', 'GM'):
        name = f'configs/feature_memory/{mode}.json'
        files.append((ROOT/name, Path('code')/name))
    files.append((ROOT/'FEATURE_MIGRATION_HANDOFF.md', Path('HANDOFF.md')))
    files.append((Path(cfg['source_checkpoint']), Path('weights/epoch_040.pt')))
    experiment = ROOT/cfg['experiment']
    for name in ['dataset_audit.json', 'window_manifest.csv', 'video_splits.csv', 'class_to_idx.json']:
        files.append((experiment/name, Path('data/experiment')/name))
    for source, destination in [(Path(cfg['store']), Path('data/event_store')),
                                ((experiment/'cache').resolve(), Path('data/experiment/cache'))]:
        for path in sorted(source.rglob('*')):
            if path.is_file():
                files.append((path, destination/path.relative_to(source)))
    build = Path('/tmp/eventmamba-vim-dd0358ad-build')
    for name in ['setup.py', 'README.md', 'LICENSE', 'AUTHORS']:
        files.append((build/name, Path('environment/mamba_build')/name))
    for path in sorted((build/'csrc').rglob('*')):
        if path.is_file():
            files.append((path, Path('environment/mamba_build/csrc')/path.relative_to(build/'csrc')))
    runtime = ROOT/'.venv-eventmamba/lib/python3.10/site-packages/mamba_ssm'
    for path in sorted(runtime.rglob('*.py')):
        files.append((path, Path('environment/mamba_build/mamba_ssm')/path.relative_to(runtime)))
    assert len({str(dest) for _, dest in files}) == len(files)
    total = sum(src.stat().st_size for src, _ in files)
    print(json.dumps(dict(event='plan', files=len(files), bytes=total, target=str(target))), flush=True)
    records, done, last = [], 0, time.monotonic()
    # Independent files, no hard links/symlinks back to the old instance.
    def copy_one(pair):
        source, relative = pair
        dest = target/relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        before = source.stat()
        if dest.exists():
            if not resume:
                raise FileExistsError(dest)
            expected = digest(source)
            if dest.stat().st_size == before.st_size and digest(dest) == expected:
                return dict(path=str(relative), source=str(source), bytes=before.st_size, sha256=expected)
        temporary = dest.with_name(dest.name+'.copying')
        h = hashlib.sha256()
        with source.open('rb') as src, temporary.open('wb') as dst:
            for block in iter(lambda: src.read(8*1024*1024), b''):
                h.update(block)
                dst.write(block)
        after = source.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError(f'Source changed during copy: {source}')
        expected = h.hexdigest()
        if digest(temporary) != expected:
            raise RuntimeError(f'Copy checksum mismatch: {relative}')
        temporary.replace(dest)
        return dict(path=str(relative), source=str(source), bytes=after.st_size, sha256=expected)
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(copy_one, pair) for pair in files]
        for index, future in enumerate(as_completed(futures), 1):
            record = future.result()
            records.append(record)
            done += record['bytes']
            if time.monotonic()-last >= 20 or index == len(files):
                progress = dict(event='verified_progress', files=index, total_files=len(files), bytes=done, total_bytes=total)
                (target/'PROGRESS.json').write_text(json.dumps(progress)+'\n')
                print(json.dumps(progress), flush=True)
                last = time.monotonic()
    records.sort(key=lambda r: r['path'])
    for name, expected in old['code'].items():
        if digest(target/'code'/name) != expected:
            raise RuntimeError(f'Protected code mismatch: {name}')
    if digest(target/'weights/epoch_040.pt') != cfg['source_sha256']:
        raise RuntimeError('Source checkpoint mismatch')
    versions = {name: importlib.metadata.version(name) for name in
                ['torch', 'torchvision', 'mamba-ssm', 'causal-conv1d', 'numpy', 'scipy',
                 'transformers', 'einops', 'triton', 'ninja', 'packaging']}
    (target/'environment/versions.json').write_text(json.dumps(dict(python=sys.version, packages=versions), indent=2)+'\n')
    (target/'MANIFEST.json').write_text(json.dumps(records, indent=2)+'\n')
    (target/'SHA256SUMS').write_text(''.join(f'{r["sha256"]}  {r["path"]}\n' for r in records))
    (target/'COMPLETE.json').write_text(json.dumps(dict(status='verified', files=len(files), bytes=done,
        manifest_sha256=digest(target/'MANIFEST.json'), source_checkpoint_sha256=cfg['source_sha256'],
        original_files_unchanged=True, real_gpu_smoke='not_run'), indent=2)+'\n')
    print('VERIFIED COMPLETE', target, flush=True)


if __name__ == '__main__':
    main()
