"""Explicit cache-only stage-2 evaluation; default command audits, never trains."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from fusion_router import FIXED, Router
from routing_data import (HEADS, ObservationNormalizer, RoutingDataset, digest,
                          json_write, video_key)
from train_fusion_routing import RoutingMetrics, batches, move, new_output
from routing_stage2 import (MODES, INTERVENTIONS, PrefixDataset, forward_diagnostic,
                            restored, verified_artifacts)


def distribution(values):
    x=np.asarray(values,dtype=np.float64)
    return dict(mean=float(x.mean()),std=float(x.std()),min=float(x.min()),max=float(x.max()),
                p05=float(np.quantile(x,.05)),median=float(np.median(x)),p95=float(np.quantile(x,.95)))


def restore_head(name, device):
    if name not in HEADS: raise ValueError('Unknown head')
    # Only used as a placeholder for compatible metric signatures, never for mixing.
    return Router(dict(mode='fixed',fixed='fusion')).to(device).eval(), ObservationNormalizer().to(device)


@torch.no_grad()
def evaluate_one(router, normalizer, data, config, *, device='cpu', head=None,
                 intervention='normal', ema_decay=None, trajectory_path=None):
    if head is not None and head not in HEADS: raise ValueError('Unknown head')
    metric=RoutingMetrics(len(data.cache.identity['class_to_idx']),int(config.get('ece_bins',15)))
    statistics={}; per_video_windows={}; drift={'alpha':[],'probability_l1':[],'probability_max_abs':[]}
    stream=Path(trajectory_path).open('x') if trajectory_path else None
    router.eval()
    try:
        for batch in batches(data,config):
            batch=move(batch,device)
            obs=normalizer(batch['observations'])
            if head is not None:
                lp=batch['logits'][...,HEADS.index(head),:].log_softmax(-1)
                output=dict(log_probs=lp,alpha=torch.zeros_like(batch['dt']),eta=torch.zeros_like(batch['dt']))
                trace={}
            else:
                output,trace=forward_diagnostic(router,batch['logits'],obs,batch['mask'],batch['dt'],
                    intervention=intervention,ema_decay=ema_decay,trace=True)
                if intervention!='normal':
                    normal,_=forward_diagnostic(router,batch['logits'],obs,batch['mask'],batch['dt'],
                                               intervention='normal',ema_decay=ema_decay)
                    mask=batch['mask']
                    diff=(output['log_probs'].exp()-normal['log_probs'].exp()).abs()
                    drift['alpha'].extend((output['alpha']-normal['alpha']).abs()[mask].cpu().tolist())
                    drift['probability_l1'].extend(diff.sum(-1)[mask].cpu().tolist())
                    drift['probability_max_abs'].extend(diff.max(-1).values[mask].cpu().tolist())
                for key,value in trace.items():
                    statistics.setdefault(key,[]).extend(value[batch['mask']].cpu().tolist())
            metric.update_log_probs(output['log_probs'],batch['logits'][...,0,:].log_softmax(-1),
                                    batch['labels'],batch['mask'],batch['rows'],output['alpha'],output['eta'])
            trace_cpu={k:v.cpu() for k,v in trace.items()}
            lp_cpu=output['log_probs'].cpu()
            for b,rows in enumerate(batch['rows']):
                for t,row in enumerate(rows):
                    window=dict(window_id=str(row['window_id']),start_us=int(row['start_us']),end_us=int(row['end_us']))
                    per_video_windows.setdefault(video_key(row),[]).append(window)
                    if stream:
                        # Labels appear only after model forward; never in observations/state.
                        record=dict(**row,dt=float(batch['dt'][b,t]),
                                    **{k:float(v[b,t]) for k,v in trace_cpu.items()},
                                    prediction=int(lp_cpu[b,t].argmax()),
                                    true_class_probability=float(lp_cpu[b,t,int(row['label_id'])].exp()))
                        stream.write(json.dumps(record,allow_nan=False)+'\n')
    finally:
        if stream:stream.close()
    metrics,predictions=metric.finish()
    if head is not None:
        # Independent heads do not have meaningful alpha/eta mixture weights.
        metrics.pop('alpha');metrics.pop('eta')
    index=[]; log_probs=[]
    for prediction in predictions:
        key=video_key(prediction);value=metric.videos[key]
        log_probs.append((value['total']-math.log(value['count'])).numpy())
        index.append(dict(**prediction,window_index=per_video_windows[key]))
    diagnostic={k:distribution(v) for k,v in statistics.items()}
    if 'alpha' in statistics:
        a=np.asarray(statistics['alpha'])
        diagnostic['alpha_near_endpoints']=dict(threshold=.01,below=float((a<=.01).mean()),above=float((a>=.99).mean()))
    if intervention!='normal' and head is None:
        diagnostic['change_vs_normal']={k:distribution(v) for k,v in drift.items()}
    return dict(metrics=metrics,diagnostics=diagnostic,index=index,log_probs=np.stack(log_probs))


def verify_reproduction(actual, expected, atol=2e-6):
    for key in ('windows','videos','rescued','damaged','net_change'):
        if actual[key]!=expected[key]:raise ValueError('Snapshot integer metric differs: '+key)
    for key in ('video_top1','video_top5','video_nll','window_top1','window_top5','window_nll','video_brier','video_ece'):
        if not math.isclose(actual[key],expected[key],rel_tol=1e-6,abs_tol=atol):
            raise ValueError('Snapshot metric differs: '+key)


def run_suite(config, output, *, device='cpu', traces=False):
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    _,cache,saved,audit=verified_artifacts(config['sources'])
    if config['selection_protocol']!=cache.identity['selection_protocol']:
        raise ValueError('Suite/cache protocol mismatch')
    if any(x not in INTERVENTIONS for x in config['interventions']):raise ValueError('Unknown intervention')
    if any(not 0<r<=1 for r in config['prefixes']):raise ValueError('Invalid prefix')
    if config['ema_decay']!=saved['current']['config'].get('ema_decay',.9):
        raise ValueError('EMA differs from predeclared coefficient')
    if any(config['role']!=checkpoint['val_role'] for checkpoint in saved.values()):
        raise ValueError('Suite must use saved selection role for reproduction')
    if not config['prefixes'] or not config['interventions'] or 'normal' not in config['interventions']:
        raise ValueError('Require nonempty prefixes and normal intervention')
    output.mkdir(parents=True)
    json_write(output/'audit.json',audit)
    json_write(output/'config.json',config)
    dataset=RoutingDataset(cache,config['role'],sequence=True)
    # fusion/frame duplicates are explicitly removed by canonical names.
    specs=[(name,'head',name) for name in HEADS]
    specs += [(name,'fixed',name) for name in FIXED if name!='fusion']
    specs += [(name,'trained',name) for name in MODES]
    specs += [('current_ema','trained','current')]
    result_index={}
    try:
        for ratio in config['prefixes']:
            data=PrefixDataset(dataset,ratio)
            for name,kind,value in specs:
                if kind=='trained':router,norm=restored(saved[value],device)
                elif kind=='fixed':
                    router=Router(dict(mode='fixed',fixed=value)).to(device).eval();norm=ObservationNormalizer().to(device)
                else:router,norm=restore_head(value,device)
                decay=config['ema_decay'] if name=='current_ema' else None
                interventions=config['interventions'] if name in MODES else ['normal']
                for intervention in interventions:
                    key=f'{name}__{intervention}__prefix_{ratio:g}'
                    folder=new_output(output/key)
                    evaluation=dict(batch_size=config['batch_size'],ece_bins=config['ece_bins'])
                    result=evaluate_one(router,norm,data,evaluation,device=device,
                         head=value if kind=='head' else None,intervention=intervention,ema_decay=decay,
                         trajectory_path=folder/'window_traces.jsonl' if traces else None)
                    reproduced=False
                    if kind=='trained' and name!='current_ema' and intervention=='normal' and ratio==1:
                        if config['role']!=saved[value]['val_role']:raise ValueError('Snapshot comparison role differs')
                        verify_reproduction(result['metrics'],saved[value]['metrics']['validation'])
                        reproduced=True
                    np.save(folder/'video_log_probs.npy',result['log_probs'],allow_pickle=False)
                    json_write(folder/'video_index.json',result['index'])
                    metadata=dict(method=name,intervention=intervention,prefix=ratio,
                        prefix_definition='max(1,ceil(ratio*valid_window_count)), chronological',
                        role=config['role'],cache_fingerprint=cache.fingerprint,
                        dataset_id=cache.identity['dataset_id'],class_to_idx=cache.identity['class_to_idx'],
                        selection_protocol=cache.identity['selection_protocol'],independent_test=False,
                        inference_only=True,snapshot_reproduced=reproduced,ema_decay=decay,
                        checkpoint=audit['models'][value] if kind=='trained' else None,
                        metrics=result['metrics'],diagnostics=result['diagnostics'],
                        files={f:digest(folder/f) for f in ('video_log_probs.npy','video_index.json')})
                    if traces:metadata['files']['window_traces.jsonl']=digest(folder/'window_traces.jsonl')
                    json_write(folder/'metrics.json',metadata)
                    result_index[key]=dict(directory=key,metrics=result['metrics'])
                    json_write(output/'status.json',dict(state='running',completed=list(result_index)))
                    print(json.dumps(dict(run=key,metrics=result['metrics'])),flush=True)
        json_write(output/'suite.json',result_index)
        json_write(output/'status.json',dict(state='complete',runs=len(result_index)))
    except BaseException as exc:
        json_write(output/'status.json',dict(state='failed',error=repr(exc),completed=list(result_index)))
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('audit','evaluate'))
    p.add_argument('--config',type=Path,default=Path('configs/fusion_routing_stage2.json'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cpu')
    p.add_argument('--traces',action='store_true')
    p.add_argument('--prefixes',type=float,nargs='+',help='Optional valid-window-count prefixes, e.g. .25 .5 .75 1')
    args=p.parse_args();config=json.loads(args.config.read_text())
    if args.prefixes:config['prefixes']=args.prefixes
    if args.command=='audit':
        if args.output.exists():raise FileExistsError(args.output)
        *_,report=verified_artifacts(config['sources'])
        args.output.parent.mkdir(parents=True,exist_ok=True);json_write(args.output,report)
        print(json.dumps(report,indent=2))
    else:run_suite(config,args.output,device=args.device,traces=args.traces)


if __name__=='__main__':main()
