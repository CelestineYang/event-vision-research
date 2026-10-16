"""Verified Test-image relocation while training is stopped; preserve old path.

Copy each expected image, verify SHA-256, then atomically install a symlink.
Only after verification and link installation is the redundant source removed.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import time

from ucf101_window_data import checked_audit,read_csv,sha,write_json,VIEWS


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment',type=Path,default=Path('experiments/ucf101dvs_gcstr_sliding_txtsplit_v2'))
    parser.add_argument('--destination',type=Path,default=Path('/remote-home/datasets/event_project_cache/ucf101_txtsplit_v2_test_images_v1'))
    args=parser.parse_args()
    checked_audit(args.experiment)
    source=args.experiment.resolve()/'cache';destination=args.destination.resolve()
    destination.relative_to(Path('/remote-home/datasets/event_project_cache'))
    if not source.is_dir() or source.is_symlink(): raise ValueError('Expected original physical cache directory')
    rows=[r for r in read_csv(args.experiment/'window_manifest.csv') if r['split']=='test']
    expected={f"{v}/{r['cache_key']}.npy" for v in VIEWS for r in rows}
    actual={str(p.relative_to(source)) for p in source.rglob('*') if p.is_file()}
    if actual!=expected: raise ValueError('Unexpected/missing source cache files')
    destination.mkdir(parents=True,exist_ok=True)
    size=sum((source/n).stat().st_size for n in expected)
    remaining=sum((source/n).stat().st_size for n in expected if not (destination/n).exists())
    if shutil.disk_usage(destination).free<remaining+1024**3: raise RuntimeError('Insufficient destination capacity')
    with (args.experiment/'.training.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with (destination/'.migration.lock').open('a') as migration_lock:
            fcntl.flock(migration_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            started=time.perf_counter();hashes={}
            status_path=args.experiment/'image_cache_migration_status.json'
            def status(**updates):
                write_json(status_path,dict(source=str(source),destination=str(destination),
                           files=len(expected),bytes=size,seconds=time.perf_counter()-started,**updates))
            status(status='copying',verified_files=0)
            for i,name in enumerate(sorted(expected),1):
                old,new=source/name,destination/name
                digest=sha(old)
                if not new.exists() or sha(new)!=digest:
                    new.parent.mkdir(parents=True,exist_ok=True)
                    tmp=new.with_suffix('.npy.copying')
                    shutil.copyfile(old,tmp)
                    if sha(tmp)!=digest: raise RuntimeError(f'Copied image differs: {name}')
                    tmp.replace(new)
                hashes[name]=digest
                if i%1000==0 or i==len(expected):
                    status(status='copying',verified_files=i)
                    print('Verified images',i,'/',len(expected),flush=True)
            write_json(destination/'image_hashes.json',hashes)
            write_json(destination/'migration_report.json',dict(source=str(source),files=len(expected),bytes=size,
                       manifest_sha256=sha(args.experiment/'window_manifest.csv'),verified=True))
            backup=source.with_name('cache.migration_backup')
            link=source.with_name('cache.next_link')
            if backup.exists() or link.exists(): raise FileExistsError('Unexpected migration staging paths')
            os.symlink(destination,link,target_is_directory=True)
            source.rename(backup)
            try: link.replace(source)
            except Exception:
                backup.rename(source);link.unlink();raise
            if not source.is_symlink() or source.resolve()!=destination: raise RuntimeError('Link verification failed')
            for name in expected:
                if not (source/name).is_file(): raise RuntimeError('Migrated image missing')
            status(status='removing_verified_source_copy',verified_files=len(expected))
            # Exact redundant source, validated above; never delete the destination.
            if backup.name!='cache.migration_backup' or backup.is_symlink(): raise RuntimeError('Invalid cleanup target')
            shutil.rmtree(backup)
            status(status='complete',verified_files=len(expected),source_redundant_copy_removed=True)
            print('Test image cache relocation complete',destination,flush=True)


if __name__=='__main__': main()
