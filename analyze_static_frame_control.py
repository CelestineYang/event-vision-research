"""Evaluate the selected restricted Static and pair against stage-2 dynamics."""
import argparse
import json
from pathlib import Path
import numpy as np

from routing_data import EvidenceCache, RoutingDataset, digest, json_write, video_key
from routing_stage2 import restored
from train_fusion_routing import load_checkpoint
from evaluate_fusion_routing_suite import evaluate_one, verify_reproduction
from analyze_fusion_routing import load_result, compare


def run(checkpoint, cache_path, suite, output):
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    cache=EvidenceCache(cache_path)
    saved=load_checkpoint(checkpoint,cache.identity)
    if saved['config']['mode']!='static' or not saved['config']['fusion_frame_only']:
        raise ValueError('Expected restricted Static checkpoint')
    if saved['val_cache']!=cache.fingerprint:raise ValueError('Cache mismatch')
    model,norm=restored(saved)
    data=RoutingDataset(cache,saved['val_role'],sequence=True)
    result=evaluate_one(model,norm,data,saved['config'])
    verify_reproduction(result['metrics'],saved['metrics']['validation'])
    output.mkdir(parents=True)
    np.save(output/'video_log_probs.npy',result['log_probs'],allow_pickle=False)
    json_write(output/'video_index.json',result['index'])
    metadata=dict(method='static_frame',dataset_id=cache.identity['dataset_id'],
        class_to_idx=cache.identity['class_to_idx'],cache_fingerprint=cache.fingerprint,
        selection_protocol=cache.identity['selection_protocol'],independent_test=False,
        role=saved['val_role'],prefix=1,checkpoint=str(checkpoint),checkpoint_sha256=digest(checkpoint),
        epoch=saved['epoch'],metrics=result['metrics'],diagnostics=result['diagnostics'],
        files={name:digest(output/name) for name in ('video_log_probs.npy','video_index.json')})
    json_write(output/'metrics.json',metadata)
    left=load_result(output)
    names={v:k for k,v in cache.identity['class_to_idx'].items()}
    leftrows={video_key(row):(row,lp) for row,lp in zip(left['index'],left['log_probs'])}
    pairs={}
    for mode in ('current','gru','conditional'):
        right=load_result(Path(suite)/f'{mode}__normal__prefix_1')
        paired=compare(left,right)
        rightrows={video_key(row):(row,lp) for row,lp in zip(right['index'],right['log_probs'])}
        groups={name:[] for name in ('both_correct','both_wrong','static_only_correct','dynamic_only_correct')}
        for row in paired['videos']:
            key=tuple(row['identity']);a,ap=leftrows[key];b,bp=rightrows[key];label=row['label_id']
            ac=a['prediction']==label;bc=b['prediction']==label
            group='both_correct' if ac and bc else 'static_only_correct' if ac else 'dynamic_only_correct' if bc else 'both_wrong'
            row.update(class_name=names[label],static_prediction_name=names[a['prediction']],
                dynamic_prediction_name=names[b['prediction']],static_nll=float(-ap[label]),
                dynamic_nll=float(-bp[label]),static_true_probability=float(np.exp(ap[label])),
                dynamic_true_probability=float(np.exp(bp[label])),group=group)
            groups[group].append(row['nll_right_minus_left'])
        n=len(paired['videos'])
        paired['nll_attribution']={g:dict(videos=len(v),sum_dynamic_minus_static=float(sum(v)),
            contribution_to_mean=float(sum(v)/n)) for g,v in groups.items()}
        delta=np.array([v['nll_right_minus_left'] for v in paired['videos']])
        paired['nll_counts']=dict(static_lower=int((delta>1e-8).sum()),dynamic_lower=int((delta < -1e-8).sum()),
            tied=int((np.abs(delta)<=1e-8).sum()))
        np.testing.assert_allclose(delta.mean(),paired['metrics_right_minus_left']['video_nll'],atol=1e-8)
        paired['top_static_nll_advantages']=sorted(paired['videos'],key=lambda x:x['nll_right_minus_left'],reverse=True)[:10]
        paired['top_dynamic_nll_advantages']=sorted(paired['videos'],key=lambda x:x['nll_right_minus_left'])[:10]
        pairs[mode]=paired
    json_write(output/'paired.json',pairs)
    lines=['# Static Fusion+Frame versus dynamic routers','',
        'Original Test used for selection; not independent-test results. Positive ΔNLL = dynamic worse than static.',
        '', '| Dynamic | Static-only correct | Dynamic-only correct | Changed predictions | ΔNLL |',
        '|---|---:|---:|---:|---:|']
    for mode,p in pairs.items():
        c=p['counts']
        lines.append(f"| {mode} | {c['left_only_correct']} | {c['right_only_correct']} | {c['prediction_changed']} | {p['metrics_right_minus_left']['video_nll']:.8f} |")
    for mode,p in pairs.items():
        lines.extend(['',f'## {mode}: changed predictions',''])
        for row in p['videos']:
            if row['left_prediction']!=row['right_prediction']:
                lines.append(f"- {row['identity'][2]}: truth={row['class_name']}; static={row['static_prediction_name']}; dynamic={row['dynamic_prediction_name']}; {row['group']}")
        lines.extend(['','NLL attribution (sum and contribution to dataset mean):','',
                      '```json',json.dumps(p['nll_attribution'],indent=2),'```'])
    (output/'analysis.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({mode:{k:v[k] for k in ('counts','nll_counts','nll_attribution')} for mode,v in pairs.items()},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--cache',required=True)
    p.add_argument('--suite',required=True)
    p.add_argument('--output',required=True)
    a=p.parse_args();run(a.checkpoint,a.cache,a.suite,a.output)
