"""Stage-3 read-only audit, baseline diagnostics, evaluation and paired tables."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from fusion_router import Router, FIXED
from fusion_router_v3 import RouterV3
from routing_data import HEADS, ObservationNormalizer, EvidenceCache, RoutingDataset, json_write, video_key
from routing_stage2 import restored
from train_fusion_routing import batches, move
from train_fusion_routing_v3 import inputs, fresh_output, evaluate_data, write_result, restore_checkpoint
from evaluate_fusion_routing_suite import evaluate_one, verify_reproduction
from analyze_fusion_routing import load_result, compare


def audit(config,output):
    if Path(output).exists():raise FileExistsError(output)
    train,val,static,norm,report=inputs(config)
    old,_=restored(static)
    batch=next(iter(batches(val,dict(batch_size=4))))
    obs=norm(batch['observations']);args=(batch['logits'],obs,batch['mask'],batch['dt'])
    report['v3_initialization']={}
    with torch.no_grad():
        expected=old(*args)
        for mode in ('fixed','global','current'):
            model=RouterV3(dict(config,eta_mode=mode)).initialize_static(static['router_state'])
            started=time.perf_counter();actual=model(*args);seconds=time.perf_counter()-started
            torch.testing.assert_close(actual['log_probs'],expected['log_probs'],rtol=0,atol=1e-6)
            report['v3_initialization'][mode]=dict(parameter_count=sum(p.numel() for p in model.parameters()),
                max_abs_log_probability_difference=float((actual['log_probs']-expected['log_probs']).abs().max()),
                batch_videos=len(batch['rows']),batch_seconds=seconds)
        # Explicit old Current regression only, never a fit initialization.
        from train_fusion_routing import load_checkpoint
        source=report['verified_sources']['current/best_val_nll.pt']['path']
        saved=load_checkpoint(source,val.cache.identity);legacy,legacy_norm=restored(saved)
        args=(batch['logits'],legacy_norm(batch['observations']),batch['mask'],batch['dt'])
        model=RouterV3(dict(config,eta_mode='fixed')).copy_legacy_current(legacy)
        a,b=model(*args),legacy(*args)
        torch.testing.assert_close(a['log_probs'],b['log_probs'],rtol=0,atol=1e-6)
        report['legacy_current_regression']=dict(epoch=saved['epoch'],max_abs_log_probability_difference=float((a['log_probs']-b['log_probs']).abs().max()))
    folder=fresh_output(output);json_write(folder/'audit.json',report)
    print(json.dumps(report['v3_initialization'],indent=2))


def baseline_train(config, suite, output):
    """One explicit diagnostic pass per head/fixed mix; no expert forward/fit."""
    if Path(output).exists():raise FileExistsError(output)
    train,val,_,_,audit_report=inputs(config)
    folder=fresh_output(output);results={}
    for name in list(HEADS)+[n for n in FIXED if n!='fusion']:
        old=load_result(Path(suite)/f'{name}__normal__prefix_1')
        if old['metadata']['cache_fingerprint']!=val.cache.fingerprint or old['metadata']['class_to_idx']!=val.cache.identity['class_to_idx']:
            raise ValueError('Selection baseline identity mismatch')
        model=Router(dict(mode='fixed',fixed=name if name in FIXED else 'fusion'))
        result=evaluate_one(model,ObservationNormalizer(),train,config,head=name if name in HEADS else None)
        results[name]=dict(train=result['metrics'],selection=old['metadata']['metrics'])
        json_write(folder/'summary.json',results)
        print(name,result['metrics']['video_top1'],result['metrics']['video_nll'],flush=True)
    json_write(folder/'audit.json',audit_report)
    lines=['# Frozen expert Train versus selection diagnostics','',
           'Train is in-sample for the experts. Selection uses original Test; not independent testing.','',
           '| Method | Train Top1 | Train NLL | Selection Top1 | Selection NLL |','|---|---:|---:|---:|---:|']
    for k,v in results.items():lines.append(f"| {k} | {v['train']['video_top1']:.8f} | {v['train']['video_nll']:.8f} | {v['selection']['video_top1']:.8f} | {v['selection']['video_nll']:.8f} |")
    (folder/'analysis.md').write_text('\n'.join(lines)+'\n')
    json_write(folder/'status.json',dict(status='complete',methods=len(results)))


def evaluate(checkpoint, cache_path, output, device='cpu', traces=False):
    if Path(output).exists():raise FileExistsError(output)
    cache=EvidenceCache(cache_path);model,norm,saved=restore_checkpoint(checkpoint,cache,device)
    data=RoutingDataset(cache,saved['config']['val_role'],sequence=True)
    folder=fresh_output(output)
    result=evaluate_data(model,norm,data,saved['config'],device,folder/'window_traces.jsonl' if traces else None)
    verify_reproduction(result['metrics'],saved['metrics']['validation'])
    write_result(folder,result,data,saved['config'],checkpoint)
    return result


def paired_tables(result_paths,suite,static_frame,output):
    if Path(output).exists():raise FileExistsError(output)
    baselines={name:load_result(Path(suite)/f'{name}__normal__prefix_1') for name in ('fusion','frame','fusion_frame','current')}
    baselines['static_frame']=load_result(static_frame)
    results={Path(p).resolve().parent.name+':'+Path(p).name:load_result(p) for p in result_paths}
    pairs={}
    for name,result in results.items():
        pairs[name]={}
        for reference,base in baselines.items():
            comparison=compare(base,result)
            groups={k:[] for k in ('both_correct','both_wrong','baseline_only_correct','new_only_correct')}
            for row in comparison['videos']:
                a=row['left_prediction']==row['label_id'];b=row['right_prediction']==row['label_id']
                g='both_correct' if a and b else 'baseline_only_correct' if a else 'new_only_correct' if b else 'both_wrong'
                groups[g].append(row['nll_right_minus_left'])
            comparison['nll_attribution']={g:dict(videos=len(v),sum_new_minus_baseline=float(sum(v)),
                contribution_to_dataset_mean=float(sum(v)/len(result['index']))) for g,v in groups.items()}
            pairs[name][reference]=comparison
    folder=fresh_output(output);json_write(folder/'paired.json',pairs)
    combined={**baselines,**results}
    json_write(folder/'summary.json',{k:v['metadata'] for k,v in combined.items()})
    lines=['# Stage-3 comparison','',
        'Same-cache selection results, not independent testing. Path weights are not causal modality contributions.','',
        '| Method | Video Top1 | Top5 | NLL | Brier | ECE |','|---|---:|---:|---:|---:|---:|']
    for name,r in combined.items():
        m=r['metadata']['metrics']
        lines.append(f"| {name} | {m['video_top1']:.8f} | {m['video_top5']:.8f} | {m['video_nll']:.8f} | {m['video_brier']:.8f} | {m['video_ece']:.8f} |")
    (folder/'analysis.md').write_text('\n'.join(lines)+'\n')
    return pairs


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    for command in ('audit','baseline-train'):
        s=sub.add_parser(command);s.add_argument('--config',default='configs/routing_stage3/G1.json');s.add_argument('--output',required=True)
        if command=='baseline-train':s.add_argument('--suite',required=True)
    s=sub.add_parser('eval');s.add_argument('--checkpoint',required=True);s.add_argument('--cache',required=True)
    s.add_argument('--output',required=True);s.add_argument('--device',default='cpu');s.add_argument('--traces',action='store_true')
    s=sub.add_parser('compare');s.add_argument('--results',nargs='+',required=True);s.add_argument('--suite',required=True)
    s.add_argument('--static-frame',required=True);s.add_argument('--output',required=True)
    a=p.parse_args()
    if a.command=='audit':audit(json.loads(Path(a.config).read_text()),a.output)
    elif a.command=='baseline-train':baseline_train(json.loads(Path(a.config).read_text()),a.suite,a.output)
    elif a.command=='eval':evaluate(a.checkpoint,a.cache,a.output,a.device,a.traces)
    else:paired_tables(a.results,a.suite,a.static_frame,a.output)


if __name__=='__main__':main()
