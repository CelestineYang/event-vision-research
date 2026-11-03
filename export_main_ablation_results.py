"""Export auditable main/ablation results without weights or machine paths."""
import argparse
import csv
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path


def main_metric(row):
    metric = row.get('evaluation', row.get('test', {}))
    return metric.get('main', metric.get('fusion', metric))


def select(rows, rule):
    if rule not in ('video_nll', 'video_top1_then_nll'):
        raise ValueError('Unknown model selection rule: ' + rule)
    if not rows or any('video_top1' not in main_metric(r) for r in rows):
        raise ValueError('Expected complete fixed-parameter video evaluation rows')
    if rule == 'video_nll':
        return min(rows, key=lambda r: (main_metric(r)['video_nll'], r['epoch']))
    return max(rows, key=lambda r: (main_metric(r)['video_top1'],
                                  -main_metric(r)['video_nll'], -r['epoch']))


def check_predictions(predictions, metrics):
    ids = [p['video_id'] for p in predictions]
    if len(ids) != len(set(ids)) or len(ids) != metrics['videos']:
        raise ValueError('Prediction coverage or uniqueness mismatch')
    correct = sum(p['label'] == p['prediction'] for p in predictions)
    if abs(correct / len(predictions) - metrics['video_top1']) > 1e-10:
        raise ValueError('Predictions do not match the selected epoch score')
    if sum(p['windows'] for p in predictions) != metrics['windows']:
        raise ValueError('Prediction window coverage mismatch')
    return correct


def portable(value, replacements):
    if isinstance(value, dict):
        return {k: portable(v, replacements) for k, v in value.items()
                if k not in ('environment', 'gpu_status', 'pid', 'hostname')}
    if isinstance(value, list):
        return [portable(v, replacements) for v in value]
    if isinstance(value, str):
        for old, new in replacements:
            value = value.replace(old, new)
        if value.startswith('/'):
            value = 'LOCAL_PATH/' + Path(value).name
    return value


def export(project, assets, output, fusion_a_checkpoints=None):
    project, assets, output = map(lambda p: Path(p).resolve(), (project, assets, output))
    sources = {}
    replacements = [(str(project), 'PROJECT_ROOT'), (str(assets), 'ASSET_ROOT')]
    def read(path, alias, lines=False):
        raw = path.read_bytes()
        sources[alias] = {'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}
        if lines:
            return [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
        return json.loads(raw)
    def save(name, value):
        p = output / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    def csv_save(name, rows):
        p = output / name
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open('w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
            writer.writeheader()
            writer.writerows(rows)

    specs = [
        ('THU-CHL', 'G', project/'experiments/thu_chl_g_gm_aligned_batch48_seed0_20261027', 50, 'fresh_fusion', 'single_gpu_global48'),
        ('THU-CHL', 'GM', assets/'records/THU_CHL_GM/training', 50, 'fresh_fusion', 'DDP3_local16_global48'),
        ('DVSAction', 'GM', project/'experiments/dvsaction_gm_batch48_seed0_20261031', 50, 'fresh_fusion', 'single_gpu_global48'),
        ('UCF101-DVS', 'Fusion_A', project/'experiments/ucf101_filename_v2_fusion_A_seed0_v1', 150, 'independent_pretrained_branches', 'DDP2_global_window16'),
        ('UCF101-DVS', 'historical_GM', assets/'records/UCF101_GM/stage3_ddp3_batch48/training', 50, 'trained_Fusion_A_epoch40', 'epoch1_single16_then_DDP3_global48'),
    ]
    results, protocols, checkpoints, selected_predictions = [], {}, {}, {}
    for dataset, method, directory, expected, initialization, execution in specs:
        key = dataset.lower().replace('-', '_') + '_' + method.lower()
        path = directory/'metrics.jsonl'
        if method == 'historical_GM':
            path = assets/'records/UCF101_GM/epochs_001_050.jsonl'
        rows = read(path, key+'/original_metrics.jsonl', True)
        rows = [r for r in rows if 'epoch' in r and ('evaluation' in r or 'test' in r)]
        if [r['epoch'] for r in rows] != list(range(1, expected+1)):
            raise ValueError('Missing/duplicate epochs: ' + key)
        config = read(directory/'config.json', key+'/original_config.json')
        identity = config.get('identity', {})
        protocols[key] = portable({
            'dataset': dataset, 'initialization': initialization, 'execution': execution,
            'config': identity.get('config', config),
            'data_identity': identity.get('data', {}),
            'code_identity': identity.get('code', {}),
            'coverage': identity.get('coverage', {}),
            'pretrained_sources': identity.get('source', {}),
        }, replacements)
        curve = []
        for r in rows:
            train = r.get('train', {})
            train = train.get('main', train.get('fusion', train))
            curve.append({'epoch': r['epoch'], 'train': train, 'evaluation': main_metric(r)})
        p = output/'curves'/f'{key}.jsonl'
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(''.join(json.dumps(r, allow_nan=False)+'\n' for r in curve))
        rules = ['video_top1_then_nll']
        if method in ('Fusion_A', 'historical_GM'):
            rules.append('video_nll')
        for rule in rules:
            winner = select(rows, rule)
            m = main_metric(winner)
            checkpoint_name = 'best_video_top1.pt'
            checkpoint_root = directory
            if dataset == 'THU-CHL' and method == 'GM':
                checkpoint_name = 'best_top1.pt'
            if method == 'historical_GM':
                checkpoint_name = 'best_video_nll.pt' if rule == 'video_nll' else 'epoch_047.pt'
            if method == 'Fusion_A':
                checkpoint_root = Path(fusion_a_checkpoints) if fusion_a_checkpoints else directory/'checkpoints'
                checkpoint_name = 'best_loss.pt' if rule == 'video_nll' else 'epoch_040.pt'
            checkpoint_path = checkpoint_root/checkpoint_name
            retained = checkpoint_path.is_file()
            weight = {'file':key+'/'+checkpoint_name, 'exists_at_export':retained}
            if retained:
                digest = hashlib.sha256()
                with checkpoint_path.open('rb') as handle:
                    for block in iter(lambda: handle.read(8*1024*1024), b''):
                        digest.update(block)
                weight.update(bytes=checkpoint_path.stat().st_size, sha256=digest.hexdigest())
            checkpoints[key+'/'+rule] = weight
            record = dict(dataset=dataset, method=method, selection=rule,
                completed_epochs=expected, selected_epoch=winner['epoch'],
                video_top1_pct=m['video_top1']*100, video_top5_pct=m['video_top5']*100,
                video_nll=m['video_nll'], videos=m['videos'], windows=m['windows'],
                correct=round(m['video_top1']*m['videos']),
                winning_weights_retained=retained, initialization=initialization,
                execution=execution, independent_test=False)
            results.append(record)
            pred_path = None
            if method in ('G','GM'):
                pred_path = directory/'predictions_best.json'
            elif method == 'historical_GM' and rule == 'video_nll':
                pred_path = directory/'predictions_best.json'
            if pred_path:
                preds = read(pred_path, key+'/selected_predictions_original.json')['main']
                check_predictions(preds, m)
                selected_predictions[key] = {p['video_id']:p for p in preds}
                csv_save('predictions/'+key+'.csv', [
                    {k: p[k] for k in ('video_id','label','prediction','windows','confidence','nll')}
                    for p in preds])

    g, gm = selected_predictions['thu_chl_g'], selected_predictions['thu_chl_gm']
    if g.keys() != gm.keys():
        raise ValueError('THU paired evaluation IDs differ')
    paired = dict(both_correct=0, g_only_correct=0, gm_only_correct=0, both_wrong=0)
    for video_id, before in g.items():
        after = gm[video_id]
        if (before['label'],before['windows']) != (after['label'],after['windows']):
            raise ValueError('THU paired evaluation labels/windows differ')
        a, b = before['label']==before['prediction'], after['label']==after['prediction']
        paired['both_correct' if a and b else 'g_only_correct' if a else 'gm_only_correct' if b else 'both_wrong'] += 1
    save('paired_thu_g_vs_gm.json', {'counts':paired,'videos':len(g),
        'scope':'Selected checkpoints on original Test; single seed, execution differs, descriptive comparison only.'})

    status = [
        {'dataset':'THU-CHL','B':'no_completed_result','G':'complete','M':'implemented_not_trained','GM':'complete','G-FixedGate':'no_completed_result','G-GlobalGate':'deferred'},
        {'dataset':'DVSAction','B':'not_trained','G':'not_trained','M':'not_trained','GM':'complete','G-FixedGate':'not_trained','G-GlobalGate':'not_trained'},
        {'dataset':'UCF101-DVS fresh initialization','B':'no_formal_result','G':'no_formal_result','M':'not_trained','GM':'prepared_not_trained','G-FixedGate':'smoke_only','G-GlobalGate':'smoke_only'},
        {'dataset':'DailyDVS-200 current window protocol','B':'not_trained','G':'not_trained','M':'not_trained','GM':'not_trained','G-FixedGate':'not_trained','G-GlobalGate':'not_trained'},
    ]
    detailed = [{'dataset':'UCF101-DVS','variant':v,'status':'smoke_only_no_formal_result'} for v in
                ['G-PointQuery','G-PointComplement','G-FrameComplement','GM-FeatureOnly','GM-GateOnly','GM-FixedUpdate','GM-NoWriteGate']]
    legacy = []
    for stage in ['ucf101_routing_epoch040_2026-10-23','ucf101_routing_stage2_2026-10-23','ucf101_routing_stage3_2026-10-24']:
        d = read(project/'research_exports'/stage/'summary.json', stage+'/summary.json')
        for name,r in d.get('results', d.get('selected', d.get('runs', {}))).items():
            m = r.get('validation', r.get('metrics', {}))
            legacy.append(dict(stage=stage, variant=name,
                selected_epoch=r.get('selected_epoch',r.get('epoch')),
                video_top1_pct=m['video_top1']*100, video_top5_pct=m['video_top5']*100,
                video_nll=m['video_nll'], selection='video_nll', independent_test=False))
    aggregates = read(project/'research_exports/ucf101_routing_stage3_2026-10-24/summary.json',
                      'ucf101_routing_stage3_2026-10-24/summary.json')['aggregate']
    suite = read(project/'research_exports/ucf101_routing_stage2_2026-10-23/suite/suite.json',
                 'ucf101_routing_stage2_2026-10-23/suite.json')
    interventions = {k:v['metrics'] for k,v in suite.items() if any(t in k for t in
                    ('reset_each_window','zero_memory_readout','gru__normal','conditional__normal'))}
    save('legacy_probability_ablations.json', {'results':legacy,'stage3_multiseed':aggregates,
        'inference_interventions':interventions,
        'scope':'Frozen-expert probability routing; not feature-level B/G/M/GM. Inference interventions are not retrained ablations.'})
    csv_save('legacy_probability_ablations.csv', legacy)
    save('data_protocols.json', protocols)
    save('checkpoints.json', checkpoints)
    save('summary.json', {'snapshot_date':'2026-11-03',
        'generated_at':datetime.now(timezone(timedelta(hours=8))).isoformat(),
        'scope':'Main fusion experiments and ablations; standalone branch pretraining and early prototypes excluded.',
        'results':results,'status':status,'detailed_ablation_status':detailed,
        'limitations':['Original Test used for checkpoint selection.',
            'UCF historical GM initializes from Fusion A and changes batch size; not a matched fresh-initialization ablation.',
            'UCF historical GM epoch47 weights were not retained; its Top1 maximum is log-only.',
            'THU G and GM match reviewed hyperparameters but single-GPU/DDP random streams differ; single seed only.',
            'B-to-G adds the entire PF supplement, not just a gate. M retains the memory writer gate.']})
    csv_save('main_results.csv', results)
    csv_save('ablation_status.csv', status)
    save('sources.json', sources)
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=Path.cwd())
    parser.add_argument('--asset-root', type=Path, required=True, help='Asset package containing records/THU_CHL_GM and records/UCF101_GM')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fusion-a-checkpoints', type=Path,
                        help='Relocated Fusion A checkpoint directory, if different from the experiment directory')
    args = parser.parse_args()
    print(json.dumps(export(args.project_root,args.asset_root,args.output,args.fusion_a_checkpoints),indent=2))
