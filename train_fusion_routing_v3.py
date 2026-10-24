"""Explicit stage-3 fits on frozen schema-2 evidence; never resumes implicitly."""
import argparse
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch

from fusion_router_v3 import ARCHITECTURE, MODEL_SCHEMA, LOSSES, RouterV3, loss_terms, training_objective
from routing_data import OBSERVATIONS, OBSERVATION_SCHEMA, ObservationNormalizer, RoutingDataset, digest, json_write, video_key
from routing_stage2 import verified_artifacts
from train_fusion_routing import batches, move, RoutingMetrics, save_checkpoint, code_identity as old_code_identity
from evaluate_fusion_routing_suite import distribution


def code_identity():
    names=('fusion_router_v3.py','train_fusion_routing_v3.py','evaluate_fusion_routing_v3.py',
           'routing_stage2.py','evaluate_fusion_routing_suite.py','analyze_fusion_routing.py')
    return dict(old_code_identity(),**{name:digest(Path(__file__).parent/name) for name in names})


def inputs(config):
    train,val,saved,audit=verified_artifacts(config['sources'])
    if config['selection_protocol']!=train.identity['selection_protocol']:
        raise ValueError('Requested protocol differs from cache')
    static=saved['static']
    if static['config'].get('fusion_frame_only',False):raise ValueError('Require unrestricted Static')
    if static['config'].get('normalization_fit_role')!='gate_train':raise ValueError('Normalizer was not Train-fitted')
    if config.get('train_role','gate_train')!=static['train_role'] or config.get('val_role','gate_val')!=static['val_role']:
        raise ValueError('Requested roles differ from verified Static')
    norm=ObservationNormalizer().to(dtype=static['normalizer_state']['mean'].dtype)
    norm.load_state_dict(static['normalizer_state'],strict=True)
    norm.requires_grad_(False).eval()
    return RoutingDataset(train,static['train_role'],sequence=True),RoutingDataset(val,static['val_role'],sequence=True),static,norm,audit


def fresh_output(path):
    path=Path(path)
    if path.exists():raise FileExistsError(path)
    path.mkdir(parents=True)
    return path


@torch.no_grad()
def evaluate_data(model,norm,data,config,device='cpu',trace_path=None):
    model.eval();norm.eval()
    metric=RoutingMetrics(len(data.cache.identity['class_to_idx']),int(config.get('ece_bins',15)))
    stats={}; windows={};total=0.;units=0
    stream=Path(trace_path).open('x') if trace_path else None
    try:
        for batch in batches(data,config):
            batch=move(batch,device)
            out=model(batch['logits'],norm(batch['observations']),batch['mask'],batch['dt'])
            terms=loss_terms(out['log_probs'],batch['labels'],batch['mask'],config['loss_mode'])
            total+=float(terms.double().sum());units+=terms.numel()
            metric.update_log_probs(out['log_probs'],batch['logits'][...,0,:].log_softmax(-1),
                batch['labels'],batch['mask'],batch['rows'],out['alpha'],out['eta'])
            for name in ('alpha','eta','w_A','w_F','w_P','delta_alpha','delta_eta'):
                stats.setdefault(name,[]).extend(out[name][batch['mask']].cpu().tolist())
            for b,rows in enumerate(batch['rows']):
                for t,row in enumerate(rows):
                    windows.setdefault(video_key(row),[]).append({k:row[k] for k in ('window_id','start_us','end_us')})
                    windows[video_key(row)][-1]['window_id']=str(row['window_id'])
                    if stream:
                        record=dict(row,dt=float(batch['dt'][b,t]),**{k:float(out[k][b,t]) for k in stats})
                        stream.write(json.dumps(record,allow_nan=False)+'\n')
    finally:
        if stream:stream.close()
    metrics,predictions=metric.finish()
    diagnostics={key:distribution(value) for key,value in stats.items()}
    for key in ('alpha','eta','w_A','w_F','w_P'):
        a=np.asarray(stats[key]);diagnostics[key].update(near_zero=float((a<=.01).mean()),near_one=float((a>=.99).mean()))
    metrics['objective_nll']=total/units
    index=[];lp=[]
    for row in predictions:
        key=video_key(row);v=metric.videos[key]
        index.append(dict(row,window_index=windows[key]))
        lp.append((v['total']-math.log(v['count'])).numpy())
    return dict(metrics=metrics,diagnostics=diagnostics,index=index,log_probs=np.stack(lp))


def write_result(folder,result,data,config,checkpoint=None):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    np.save(folder/'video_log_probs.npy',result['log_probs'],allow_pickle=False)
    json_write(folder/'video_index.json',result['index'])
    identity=data.cache.identity
    json_write(folder/'metrics.json',dict(method=config.get('id','v3'),architecture=ARCHITECTURE,
        model_schema=MODEL_SCHEMA,role=config.get('val_role','gate_val'),prefix=1,
        dataset_id=identity['dataset_id'],class_to_idx=identity['class_to_idx'],cache_fingerprint=data.cache.fingerprint,
        selection_protocol=identity['selection_protocol'],independent_test=False,
        checkpoint_sha256=digest(checkpoint) if checkpoint else None,
        metrics=result['metrics'],diagnostics=result['diagnostics'],
        files={name:digest(folder/name) for name in ('video_log_probs.npy','video_index.json')}))


def restore_checkpoint(path,cache,device='cpu'):
    saved=torch.load(path,map_location='cpu',weights_only=False)
    if saved.get('architecture')!=ARCHITECTURE or saved.get('model_schema')!=MODEL_SCHEMA:
        raise ValueError('Not a stage-3 checkpoint')
    c=saved['config']
    if saved['identity']!=cache.identity or saved['val_cache']!=cache.fingerprint:
        raise ValueError('Checkpoint/cache identity mismatch')
    if c['code']!=code_identity() or c['observations']!=list(OBSERVATIONS) or c['observation_schema']!=OBSERVATION_SCHEMA:
        raise ValueError('Checkpoint code/observations mismatch')
    if c['selection_protocol']!=cache.identity['selection_protocol'] or c['architecture']!=ARCHITECTURE or c['model_schema']!=MODEL_SCHEMA:
        raise ValueError('Checkpoint protocol/schema mismatch')
    model=RouterV3(c).to(dtype=saved['router_state']['b_alpha'].dtype)
    model.load_state_dict(saved['router_state'],strict=True)
    norm=ObservationNormalizer().to(dtype=saved['normalizer_state']['mean'].dtype)
    norm.load_state_dict(saved['normalizer_state'],strict=True)
    return model.to(device),norm.to(device).requires_grad_(False).eval(),saved


def gradient_norm(parameters):
    gradients=[p.grad.detach().double().square().sum() for p in parameters if p.grad is not None]
    return float(torch.stack(gradients).sum().sqrt()) if gradients else 0.


def fit(config,output,device='cpu'):
    if Path(output).exists():raise FileExistsError(output)
    config=dict(config)
    if config['eta_mode']=='fixed':raise ValueError('Fixed mode is regression-only; reuse completed Current')
    if config['loss_mode'] not in LOSSES or config.get('batch_unit')!='video':raise ValueError('Invalid loss/batch unit')
    if int(config['epochs'])<1 or int(config['batch_size'])<1 or float(config['lr'])<=0:raise ValueError('Invalid budget')
    train,val,static,norm,audit=inputs(config)
    if static['router_state']['alpha_logit'].dtype!=torch.float32:raise ValueError('Real fits require FP32 Static')
    seed=int(config['seed']);random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    model=RouterV3(config).initialize_static(static['router_state']).to(device);norm.to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=config['lr'],weight_decay=config['weight_decay'])
    steps_per_epoch=len(batches(train,config))
    config.update(architecture=ARCHITECTURE,model_schema=MODEL_SCHEMA,cache_schema=2,
        observations=list(OBSERVATIONS),observation_schema=OBSERVATION_SCHEMA,code=code_identity(),
        parameter_count=sum(p.numel() for p in model.parameters() if p.requires_grad),optimizer='Adam',precision='FP32',
        static_init_sha256=audit['models']['static']['sha256'],static_init_epoch=static['epoch'],
        normalizer_fit='copied unchanged from unrestricted Static Train state; never refit',
        train_role=static['train_role'],val_role=static['val_role'],independent_test=False,
        epoch0_selection=False,selection_metric='video_nll',effective_steps_per_epoch=steps_per_epoch,
        epoch_loss_scaling='sum(loss_terms) * steps_per_epoch / '+('windows_per_epoch' if config['loss_mode']=='window_nll' else 'videos_per_epoch'),
        expert_provenance=train.cache.identity['expert_provenance'])
    output=fresh_output(output);json_write(output/'config.json',config);json_write(output/'audit.json',audit)
    steps=0;best=float('inf')
    def checkpoint(epoch,item):
        return dict(architecture=ARCHITECTURE,model_schema=MODEL_SCHEMA,config=config,
            identity=train.cache.identity,train_cache=train.cache.fingerprint,val_cache=val.cache.fingerprint,
            router_state=model.state_dict(),normalizer_state=norm.state_dict(),optimizer_state=optimizer.state_dict(),
            epoch=epoch,optimizer_steps=steps,metrics=item,rng=torch.get_rng_state())
    try:
        initial_train=evaluate_data(model,norm,train,config,device)
        initial_val=evaluate_data(model,norm,val,config,device)
        initial=dict(epoch=0,train=initial_train['metrics'],validation=initial_val['metrics'],
                     train_diagnostics=initial_train['diagnostics'],validation_diagnostics=initial_val['diagnostics'],selected=False)
        json_write(output/'epoch0.json',initial);save_checkpoint(output/'epoch0.pt',checkpoint(0,initial))
        for epoch in range(1,config['epochs']+1):
            started=time.perf_counter();model.train();online=0.;units=0;grad={'alpha':[],'eta':[]};seen=0
            for batch in batches(train,config,training=True,seed=seed+epoch):
                # Metadata is checked outside the model; only 17 normalized observations enter it.
                if any(r['role']!=static['train_role'] for rows in batch['rows'] for r in rows):raise ValueError('Non-Train row in optimizer batch')
                batch=move(batch,device);optimizer.zero_grad(set_to_none=True)
                out=model(batch['logits'],norm(batch['observations']),batch['mask'],batch['dt'])
                terms=loss_terms(out['log_probs'],batch['labels'],batch['mask'],config['loss_mode'])
                objective=training_objective(out['log_probs'],batch['labels'],batch['mask'],config['loss_mode'],len(train.indices),len(train),steps_per_epoch)
                if not torch.isfinite(objective):raise FloatingPointError('Nonfinite objective')
                objective.backward()
                grad['alpha'].append(gradient_norm(model.alpha_net.parameters()))
                grad['eta'].append(gradient_norm([p for name,p in model.named_parameters() if name=='b_eta' or name.startswith('eta_net.')]))
                torch.nn.utils.clip_grad_norm_(model.parameters(),config['gradient_clip'],error_if_nonfinite=True)
                optimizer.step();steps+=1;online+=float(terms.detach().double().sum());units+=terms.numel();seen+=int(batch['mask'].sum())
            if seen!=len(train.indices):raise ValueError('Epoch did not cover every Train window')
            tr=evaluate_data(model,norm,train,config,device);va=evaluate_data(model,norm,val,config,device)
            item=dict(epoch=epoch,train=tr['metrics'],validation=va['metrics'],
                train_diagnostics=tr['diagnostics'],validation_diagnostics=va['diagnostics'],
                train_online_objective_nll=online/units,gradient_norms={k:distribution(v) for k,v in grad.items()},
                optimizer_steps=steps,seconds=time.perf_counter()-started)
            save_checkpoint(output/'last.pt',checkpoint(epoch,item))
            write_result(output/'last',va,val,config,output/'last.pt')
            if va['metrics']['video_nll']<best:
                best=va['metrics']['video_nll'];save_checkpoint(output/'best_val_nll.pt',checkpoint(epoch,item))
                write_result(output/'best',va,val,config,output/'best_val_nll.pt')
            with (output/'metrics.jsonl').open('a') as stream:stream.write(json.dumps(item,allow_nan=False)+'\n')
            json_write(output/'status.json',dict(status='running',epoch=epoch,optimizer_steps=steps,best_video_nll=best))
            print(json.dumps(dict(epoch=epoch,validation=va['metrics'])),flush=True)
        json_write(output/'status.json',dict(status='complete',epochs=config['epochs'],optimizer_steps=steps,best_video_nll=best))
    except BaseException as exc:
        json_write(output/'status.json',dict(status='failed',optimizer_steps=steps,error=repr(exc)));raise
    return output


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['fit']);p.add_argument('--config',required=True)
    p.add_argument('--output',required=True);p.add_argument('--device',default='cpu')
    args=p.parse_args();fit(json.loads(Path(args.config).read_text()),args.output,args.device)
