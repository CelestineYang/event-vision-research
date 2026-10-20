"""Export the explicitly approved small research materials without weights/data.

Read-only toward running experiments. Capture a complete metric prefix and
publish a new snapshot directory only after population/provenance checks.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


PROJECT = Path(__file__).resolve().parent
VIEWS = ('cstr3','xt_p','ty_p')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def complete_metrics(path):
    raw=Path(path).read_bytes()
    end=raw.rfind(b'\n')+1
    if end==0: raise ValueError('No complete metric records')
    payload=raw[:end]
    rows=[json.loads(line) for line in payload.splitlines() if line.strip()]
    if [r['epoch'] for r in rows]!=list(range(1,len(rows)+1)):
        raise ValueError('Metric epochs must be contiguous and unique')
    return payload,rows


def check_predictions(payload, expected):
    rows=list(csv.DictReader(io.StringIO(payload.decode())))
    if len(rows)!=expected['videos'] or len({r['video_id'] for r in rows})!=len(rows):
        raise ValueError('Prediction video population mismatch')
    if sum(int(r['windows']) for r in rows)!=expected['windows']:
        raise ValueError('Prediction window population mismatch')
    top1=sum(int(r['label_id'])==int(r['prediction']) for r in rows)/len(rows)
    nll=sum(float(r['nll']) for r in rows)/len(rows)
    if not math.isclose(top1,expected['video_top1'],abs_tol=1e-12):
        raise ValueError('Predictions do not match selected-epoch Top-1')
    if not math.isclose(nll,expected['video_nll'],rel_tol=1e-10,abs_tol=1e-12):
        raise ValueError('Predictions do not match selected-epoch NLL')


def export(args):
    fusion=PROJECT/args.fusion
    images=PROJECT/args.images
    output=(PROJECT/args.output).absolute()
    if output.exists() or output.is_symlink(): raise FileExistsError(output)
    payload,rows=complete_metrics(fusion/'metrics.jsonl')
    epoch=rows[-1]['epoch']
    selected=min(rows,key=lambda r:(r['test']['fusion']['video_nll'],-r['test']['fusion']['video_top1'],r['epoch']))
    peak=min(rows,key=lambda r:(-r['test']['fusion']['video_top1'],r['test']['fusion']['video_nll'],r['epoch']))
    config=json.loads((fusion/'config.json').read_text())
    mismatches=[p for p,h in config['code'].items() if digest((PROJECT/p).read_bytes())!=h]
    if mismatches: raise ValueError(f'Current sources differ from executed experiment: {mismatches}')
    audit=json.loads((images/'dataset_audit.json').read_text())
    if audit['manifest_sha256']!=config['manifest_sha256']:
        raise ValueError('Image and fusion manifests differ')
    if audit['video_splits_sha256']!=config['video_splits_sha256']:
        raise ValueError('Image and fusion video lists differ')
    population=(rows[-1]['test']['fusion']['videos'],rows[-1]['test']['fusion']['windows'])
    for row in rows:
        if (row['test']['fusion']['videos'],row['test']['fusion']['windows'])!=population:
            raise ValueError('Test population changed during fusion')
    image_rows={v:complete_metrics(images/'logs'/f'{v}_metrics.jsonl') for v in VIEWS}
    image_best={v:min(rs,key=lambda r:(r['test']['video_nll'],-r['test']['video_top1'],r['epoch']))
                for v,(_,rs) in image_rows.items()}
    for _,rs in image_rows.values():
        if any((r['test']['videos'],r['test']['windows'])!=population for r in rs):
            raise ValueError('Image and fusion Test populations differ')
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.snapshot-',dir=output.parent) as temporary:
        stage=Path(temporary)/'payload'
        stage.mkdir()
        entries={}

        def put(name,data,source):
            if Path(name).name!=name or len(data)>5*1024*1024:
                raise ValueError(f'Unexpected snapshot member: {name}')
            (stage/name).write_bytes(data)
            entries[name]=dict(bytes=len(data),sha256=digest(data),source=str(source))

        def copy(name,source):
            put(name,source.read_bytes(),source.relative_to(PROJECT))

        def json_file(name,data,source):
            put(name,(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode(),source)

        put('fusion_metrics.jsonl',payload,(fusion/'metrics.jsonl').relative_to(PROJECT))
        copy('fusion_config.json',fusion/'config.json')
        copy('image_config.json',images/'config.json')
        copy('image_summary.json',images/'reports/summary.json')
        copy('class_to_idx.json',images/'class_to_idx.json')
        copy('eventmamba_official_training.txt',PROJECT/'eventmamba.txt')
        for name in ('UCF101_FILENAME_V2_FUSION_RUNBOOK.md','UCF101_TXTSPLIT_V2_RUNBOOK.md'):
            copy(name,PROJECT/name)
        for v,(data,_) in image_rows.items():
            put(v+'_metrics.jsonl',data,(images/'logs'/f'{v}_metrics.jsonl').relative_to(PROJECT))
            source=images/'reports'/f'{v}_test_video_predictions_best_loss.csv'
            data=source.read_bytes()
            check_predictions(data,image_best[v]['test'])
            put(source.name,data,source.relative_to(PROJECT))
        for branch in ('fusion','point'):
            source=fusion/f'{branch}_predictions_best_loss.csv'
            data=source.read_bytes()
            # Point predictions here correspond to the fusion-NLL selected epoch.
            check_predictions(data,selected['test'][branch])
            put(source.name,data,source.relative_to(PROJECT))
        videos=list(csv.DictReader((images/'video_splits.csv').open()))
        windows=list(csv.DictReader((images/'window_manifest.csv').open()))
        raw={s:[r for r in videos if r['split']==s] for s in ('train','test')}
        ids={s:{r['video_id'] for r in rs} for s,rs in raw.items()}
        groups={s:{(r['class_name'],r['group']) for r in rs} for s,rs in raw.items()}
        if ids['train']&ids['test']: raise ValueError('Duplicate videos across splits')
        data_protocol=dict(source_audit_status=audit['status'],audit_config=audit['config'],
            raw_video_counts={s:len(rs) for s,rs in raw.items()},
            valid_video_counts={s:len({r['video_id'] for r in windows if r['split']==s}) for s in raw},
            valid_window_counts={s:sum(r['split']==s for r in windows) for s in raw},
            class_group_counts={s:len(g) for s,g in groups.items()},
            overlapping_class_groups=len(groups['train']&groups['test']),exact_video_overlap=0,
            tracked_split_lists=['train_ucf.txt','test_ucf(1).txt'],
            scope='aggregate metadata only; no raw events, images or full window manifest exported')
        json_file('data_protocol_summary.json',data_protocol,'audited source manifests; aggregate only')
        with tempfile.TemporaryDirectory(prefix='ucf101-curves-') as plot_temporary:
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MPLBACKEND='Agg')
            subprocess.run([sys.executable,str(PROJECT/'analyze_ucf101_fusion_curves.py'),
                '--fusion',str(fusion),'--images',str(images),'--through-epoch',str(epoch),
                '--output',plot_temporary],env=env,check=True,stdout=subprocess.DEVNULL)
            plot=Path(plot_temporary)
            summary=json.loads((plot/'summary.json').read_text())
            if summary['completed_epochs']!=epoch: raise ValueError('Plot epoch mismatch')
            for name in ('curves.png','curves.pdf','matched_epochs.csv','key_epochs.json'):
                put(name,(plot/name).read_bytes(),f'analyze_ucf101_fusion_curves.py --through-epoch {epoch}')
            json_file('curve_summary.json',summary,f'analysis of metric prefix through epoch {epoch}')
        index=dict(weights_uploaded=False,selection_rule='fusion video Test NLL',
            official_point_pretraining=dict(log='eventmamba_official_training.txt',
                artifact_sha256=config['point_checkpoint_sha256'],epoch_zero_based=146,
                note='Author pretraining log, not an independently reproduced point-only baseline'),
            image_initialization={v:dict(epoch=image_best[v]['epoch'],
                path=str((images/'checkpoints'/v/'best_loss.pt').relative_to(PROJECT)),
                sha256=config['image_checkpoint_sha256'][v]) for v in VIEWS},
            fusion_nll_selected=dict(epoch=selected['epoch'],filename='best_loss.pt',metrics=selected['test']['fusion']),
            fusion_peak_top1=dict(epoch=peak['epoch'],filename=f"epoch_{peak['epoch']:03d}.pt" if peak['epoch']%5==0 else None,
                                 metrics=peak['test']['fusion']),
            checkpoint_directory=config['checkpoint_dir'])
        json_file('checkpoint_index.json',index,'metric records and executed configuration; no weight payload')
        readme=f'''# UCF101 filename V2 + gCSTR A: discussion snapshot

Snapshot exported at {datetime.now(timezone.utc).isoformat()}.
Contains complete fusion epochs **1–{epoch} of 150**, not a completed150-epoch
experiment. Image pretraining contains all50 epochs. The source training can
continue independently; this directory is a static reviewed snapshot.

## Main findings

- Preselected by video Test NLL: epoch{selected['epoch']}, Top-1
  {selected['test']['fusion']['video_top1']*100:.4f}%, Top-5
  {selected['test']['fusion']['video_top5']*100:.4f}%, NLL
  {selected['test']['fusion']['video_nll']:.6f}.
- Peak video Test Top-1 (supplementary statistic): epoch{peak['epoch']},
  {peak['test']['fusion']['video_top1']*100:.4f}%. This is not the NLL-selected model.
- Read `curve_summary.json` for latest metrics and matched-epoch comparisons.
  Accuracy plateaus while Test NLL rises later in training; the figures are
  diagnostic, not a new model-selection rule.

## Read these files first

1. `UCF101_FILENAME_V2_FUSION_RUNBOOK.md`: executed architecture/training recipe.
2. `fusion_config.json` and `fusion_metrics.jsonl`: actual config and full prefix.
3. `image_summary.json`, `image_config.json` and three `*_metrics.jsonl`: baselines.
4. `curves.png` / `curves.pdf`, `curve_summary.json`, `matched_epochs.csv`.
5. `data_protocol_summary.json`, `UCF101_TXTSPLIT_V2_RUNBOOK.md`, and the root
   repository's tracked `train_ucf.txt`, `test_ucf(1).txt`: data protocol.
6. `eventmamba_official_training.txt` and `checkpoint_index.json`: pretraining
   provenance. Official pretraining is not the current fine-tuned point head.
7. Five `*_predictions_best_loss.csv` files: video-level error analysis. Fusion
   and point CSVs both come from the fusion-NLL selected epoch; they are not
   independently selected point-only results. CSVs contain top5 predictions
   and true-class probabilities, not full101-class probability vectors.

## Architecture and questions for the next design

Frozen three gCSTR ResNet-18 backbones produce49 tokens/view. Trainable official
V2 stages produce1024×64,512×128,256×256 point features. Each stage uses16
prototype queries, then attends to each image view; three128-dimensional stage
vectors form the101-class fusion head. A pretrained attention-pooled point head
has an auxiliary CE loss with weight0.5. Image adapters and position embeddings
are trainable. Points and full-event images share a1s/0.5s window/augmentation.

Useful questions: Does prototype compression remove needed information? Does
the point branch add value over an equally trained three-image control? How
can later confidence overfitting be reduced? Propose small, controlled ablations
and measurable hypotheses before expanding the model.

## Interpretation limits

- Test participates in checkpoint selection; it is not independent held-out data.
- Original TXT lists have {len(groups['train']&groups['test'])} shared class/group
  identities, although exact video overlap is zero. This is not group-disjoint.
- Only retained valid-window videos are evaluated ({population[0]} videos /
  {population[1]} windows); filtering changes the evaluation population.
- Fusion warm-starts from task-trained images and pretrained points. Global
  batch16 versus image batch64 means equal epochs are not equal update budgets.
- No matched three-image fusion/ensemble control has yet been established.
- Official normalization uses x/320,y/240 despite actual240×180 dimensions;
  window indexing and training augmentation are recorded adaptations.

No checkpoints, raw dataset payloads, event/frame caches, environments or
authentication material are included. `MANIFEST.json` lists SHA256 provenance
for every exported member. Original experiment files remain unchanged.
'''
        put('README.md',readme.encode(),'reviewed snapshot description generated from source metadata')
        manifest=dict(schema=1,created_utc=datetime.now(timezone.utc).isoformat(),
            completed_fusion_epoch=epoch,planned_fusion_epochs=config['epochs'],
            experiment_complete=epoch==config['epochs'],code_matches_executed_config=True,
            source_config_code_hashes=config['code'],members=entries)
        (stage/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
        for name,entry in entries.items():
            if digest((stage/name).read_bytes())!=entry['sha256']: raise ValueError('Export integrity failure')
        os.replace(stage,output)
    return dict(output=str(output),completed_epoch=epoch,files=len(entries)+1,
                bytes=sum(p.stat().st_size for p in output.iterdir()))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fusion',type=Path,default=Path('experiments/ucf101_filename_v2_fusion_A_seed0_v1'))
    parser.add_argument('--images',type=Path,default=Path('experiments/ucf101dvs_gcstr_sliding_txtsplit_v2'))
    parser.add_argument('--output',type=Path,default=Path('research_exports/ucf101_filename_v2_fusion_A_2026-10-20'))
    args=parser.parse_args()
    print(json.dumps(export(args),indent=2))


if __name__=='__main__': main()
