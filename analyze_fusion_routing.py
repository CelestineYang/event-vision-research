"""Offline paired-video diagnostics for completed stage-2 suites."""
import argparse
import json
from pathlib import Path

import numpy as np

from routing_data import digest, json_write, video_key

PAIRS = [('static','current'), ('current','gru'), ('current','conditional'), ('gru','conditional')]


def load_result(folder):
    folder = Path(folder)
    metadata = json.loads((folder/'metrics.json').read_text())
    for filename, sha in metadata['files'].items():
        if digest(folder/filename) != sha:
            raise ValueError('Changed result file: '+filename)
    index = json.loads((folder/'video_index.json').read_text())
    lp = np.load(folder/'video_log_probs.npy', allow_pickle=False)
    if lp.shape != (len(index), len(metadata['class_to_idx'])) or not np.isfinite(lp).all():
        raise ValueError('Invalid video probability shape/values')
    if not np.allclose(np.exp(lp).sum(-1), 1, atol=2e-6):
        raise ValueError('Video probabilities not normalized')
    keys = [video_key(row) for row in index]
    if len(set(keys)) != len(keys):
        raise ValueError('Duplicate video identity')
    for row, probabilities in zip(index, lp):
        windows = row['window_index']
        if len(windows) != row['windows'] or len({w['window_id'] for w in windows}) != len(windows):
            raise ValueError('Invalid window identities/count')
        if row['prediction'] != int(probabilities.argmax()) or not np.isclose(row['nll'], -probabilities[row['label_id']], atol=2e-6):
            raise ValueError('Prediction/NLL metadata disagrees with probabilities')
    return dict(metadata=metadata, index=index, log_probs=lp)


def compare(left, right):
    """Exact identity pairing; equal accuracies never imply equal error sets."""
    for key in ('dataset_id','class_to_idx','cache_fingerprint','selection_protocol','role','prefix'):
        if left['metadata'][key] != right['metadata'][key]:
            raise ValueError('Pair identity mismatch: '+key)
    li = {video_key(row):(row,lp) for row,lp in zip(left['index'],left['log_probs'])}
    ri = {video_key(row):(row,lp) for row,lp in zip(right['index'],right['log_probs'])}
    if len(li)!=len(left['index']) or len(ri)!=len(right['index']) or li.keys()!=ri.keys():
        raise ValueError('Video sets differ or contain duplicates')
    sets = {k:[] for k in ('prediction_changed','left_only_correct','right_only_correct',
                          'both_correct','both_wrong','left_rescued','right_rescued','left_damaged','right_damaged')}
    details=[]
    for key in sorted(li):
        a,ap=li[key]; b,bp=ri[key]
        for field in ('label_id','windows','window_index','fusion_prediction'):
            if a[field]!=b[field]:raise ValueError('Paired video mismatch: '+field)
        label=a['label_id']; ac=a['prediction']==label; bc=b['prediction']==label
        identity=list(key)
        sets['both_correct' if ac and bc else 'left_only_correct' if ac else 'right_only_correct' if bc else 'both_wrong'].append(identity)
        if a['prediction']!=b['prediction']:sets['prediction_changed'].append(identity)
        for side,row,correct in [('left',a,ac),('right',b,bc)]:
            base=row['fusion_prediction']==label
            if correct and not base:sets[side+'_rescued'].append(identity)
            if base and not correct:sets[side+'_damaged'].append(identity)
        diff=np.abs(np.exp(ap)-np.exp(bp))
        details.append(dict(identity=identity,label_id=label,left_prediction=a['prediction'],
            right_prediction=b['prediction'],nll_right_minus_left=float(-bp[label]+ap[label]),
            probability_l1=float(diff.sum()),probability_max_abs=float(diff.max())))
    metric_delta={k:right['metadata']['metrics'][k]-left['metadata']['metrics'][k]
                  for k in ('video_top1','video_nll','video_brier','video_ece')}
    return dict(counts={k:len(v) for k,v in sets.items()},sets=sets,videos=details,
                metrics_right_minus_left=metric_delta,
                right_window_diagnostics=right['metadata'].get('diagnostics',{}))


def analyze(suite, output):
    suite,output=Path(suite),Path(output)
    if output.exists():raise FileExistsError(output)
    if json.loads((suite/'status.json').read_text())['state']!='complete':
        raise ValueError('Suite incomplete')
    entries=json.loads((suite/'suite.json').read_text())
    results={k:load_result(suite/v['directory']) for k,v in entries.items()}
    pairs={}
    for key,result in results.items():
        meta=result['metadata']; ratio=meta['prefix']; mode=meta['method']
        if meta['intervention']!='normal':
            normal=f'{mode}__normal__prefix_{ratio:g}'
            pairs[key]=compare(results[normal],result)
    for ratio in sorted({r['metadata']['prefix'] for r in results.values()}):
        for a,b in PAIRS:
            left=f'{a}__normal__prefix_{ratio:g}';right=f'{b}__normal__prefix_{ratio:g}'
            pairs[f'{a}_vs_{b}__prefix_{ratio:g}']=compare(results[left],results[right])
    output.mkdir(parents=True)
    json_write(output/'paired.json',pairs)
    json_write(output/'summary.json',{key:dict(metrics=value['metadata']['metrics'],
        diagnostics=value['metadata'].get('diagnostics',{}),
        checkpoint=value['metadata'].get('checkpoint')) for key,value in results.items()})
    lines=['# Frozen-checkpoint routing diagnostics','',
           'Selection protocol: '+str(next(iter(results.values()))['metadata']['selection_protocol'])+'. Selection-set results are not independent-test results.',
           'Interventions are inference-time diagnostics, not retrained architectural ablations.',
           'Path weights are explicit mixture weights, not causal modality contributions.','',
           '| Evaluation | Video Top1 | Video Top5 | Video NLL | Brier | ECE |',
           '|---|---:|---:|---:|---:|---:|']
    for key,value in results.items():
        m=value['metadata']['metrics']
        lines.append(f"| {key} | {m['video_top1']:.8f} | {m['video_top5']:.8f} | {m['video_nll']:.8f} | {m['video_brier']:.8f} | {m['video_ece']:.8f} |")
    lines.extend(['',
           '| Comparison | Prediction changes | Left only correct | Right only correct | Δ NLL (right−left) |',
           '|---|---:|---:|---:|---:|'])
    for key,value in pairs.items():
        c=value['counts']
        lines.append(f"| {key} | {c['prediction_changed']} | {c['left_only_correct']} | {c['right_only_correct']} | {value['metrics_right_minus_left']['video_nll']:.8f} |")
    (output/'analysis.md').write_text('\n'.join(lines)+'\n')
    return pairs


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    analyze(args.suite,args.output)
