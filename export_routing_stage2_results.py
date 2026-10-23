"""Build a small, immutable stage-2 sharing snapshot; exclude arrays and weights."""
import json
import shutil
import statistics
from pathlib import Path

from routing_data import digest, json_write


def main():
    root=Path(__file__).resolve().parent
    output=root/'research_exports/ucf101_routing_stage2_2026-10-23'
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True)
    sources={}
    def copy(source, relative):
        source=Path(source);target=output/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,target)
        sources[relative]=dict(source=str(source.resolve()),sha256=digest(source),included=True)
    base=Path('/remote-home/datasets/event_project_cache/routing_epoch040_static_20261022_v3')
    selected={};summary={}
    for mode in ('current','gru','conditional','static_frame'):
        values=[]
        for seed in ((0,) if mode=='static_frame' else (0,1,2)):
            path=(root/'checkpoints/routing_stage2_static_frame_seed0' if mode=='static_frame'
                  else base/mode if seed==0 else root/f'checkpoints/routing_stage2_{mode}_seed{seed}')
            state=json.loads((path/'status.json').read_text())
            rows=[json.loads(x) for x in (path/'metrics.jsonl').read_text().splitlines()]
            assert state['status']=='complete' and len(rows)==80
            best=min(rows,key=lambda x:x['validation']['video_nll'])
            name=f'{mode}_seed{seed}'
            selected[name]=dict(epoch=best['epoch'],metrics=best['validation'])
            values.append(best['validation']['video_nll'])
            for filename in ('config.json','metrics.jsonl','status.json','validation_predictions_best.json'):
                copy(path/filename,f'{name}/{filename}')
            for filename in ('best_val_nll.pt','last.pt'):
                sources[f'{name}/{filename}']=dict(source=str((path/filename).resolve()),sha256=digest(path/filename),included=False)
        summary[mode]=dict(video_nll_mean=statistics.mean(values),video_nll_sample_std=statistics.stdev(values) if len(values)>1 else None,seeds=len(values))
    suite=root/'reports/routing_stage2_full_v1'
    assert json.loads((suite/'status.json').read_text())['state']=='complete'
    for filename in ('config.json','audit.json','suite.json','status.json'):
        copy(suite/filename,f'suite/{filename}')
    for path in sorted(suite.glob('*/metrics.json')):
        copy(path,f'suite/{path.parent.name}/metrics.json')
    copy(root/'reports/routing_stage2_full_v1_analysis/analysis.md','suite/analysis.md')
    copy(root/'reports/routing_stage2_static_frame_comparison/analysis.md','static_frame_comparison/analysis.md')
    pairs=json.loads((root/'reports/routing_stage2_static_frame_comparison/paired.json').read_text())
    compact={mode:{k:v for k,v in item.items() if k!='videos'} for mode,item in pairs.items()}
    json_write(output/'static_frame_comparison/paired_summary.json',compact)
    json_write(output/'summary.json',dict(selection_protocol='original_test_as_val',independent_test=False,
        variance_scope='Fixed experts and seed0 unrestricted Static; only dynamic initialization/order vary',
        selected=selected,aggregates=summary))
    json_write(output/'sources.json',sources)
    print(output)


if __name__=='__main__':main()
