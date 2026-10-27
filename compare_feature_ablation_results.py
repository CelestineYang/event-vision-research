"""Explicit exploratory video pairing across historical/new training protocols."""
import argparse
import csv
import json
from pathlib import Path


def read_predictions(path):
    path = Path(path)
    if path.suffix == '.csv':
        with path.open(newline='') as stream:
            rows = list(csv.DictReader(stream))
    else:
        payload = json.loads(path.read_text())
        rows = payload if isinstance(payload, list) else payload['main']
    result = {}
    for row in rows:
        video = row['video_id']
        if video in result:
            raise ValueError('Duplicate video')
        result[video] = dict(label=int(row.get('label', row.get('label_id'))),
                            prediction=int(row['prediction']), windows=int(row['windows']),
                            nll=float(row['nll']) if 'nll' in row else None)
    if not result:
        raise ValueError('Empty predictions')
    return result


def compare(reference, candidate):
    a, b = read_predictions(reference), read_predictions(candidate)
    if a.keys() != b.keys():
        raise ValueError('Video coverage differs')
    rescued, damaged, rows = [], [], []
    for video in sorted(a):
        old, new = a[video], b[video]
        if (old['label'], old['windows']) != (new['label'], new['windows']):
            raise ValueError('Video labels/window counts differ')
        old_hit, new_hit = old['prediction'] == old['label'], new['prediction'] == new['label']
        if new_hit and not old_hit:
            rescued.append(video)
        if old_hit and not new_hit:
            damaged.append(video)
        rows.append(dict(video_id=video, label=old['label'], windows=old['windows'],
            reference_prediction=old['prediction'], candidate_prediction=new['prediction'],
            delta_nll=new['nll']-old['nll'] if old['nll'] is not None and new['nll'] is not None else None))
    return dict(videos=len(rows), rescued=rescued, damaged=damaged,
                net_correct=len(rescued)-len(damaged), video_pairs=rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference-predictions', type=Path, required=True)
    p.add_argument('--candidate-predictions', type=Path, required=True)
    p.add_argument('--reference-config', type=Path, required=True)
    p.add_argument('--candidate-config', type=Path, required=True)
    p.add_argument('--reference-label', required=True, help='Include exact checkpoint epoch/selection rule')
    p.add_argument('--candidate-label', required=True)
    p.add_argument('--scope', choices=['exploratory'], required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    import hashlib
    configs = {}
    sources = {}
    for side in ('reference', 'candidate'):
        path = getattr(args, side+'_config')
        c = json.loads(path.read_text())
        if c.get('smoke') is True:
            raise ValueError('Do not compare smoke results as formal performance')
        configs[side] = c.get('identity', {}).get('config', c.get('config', c))
        sources[side] = dict(label=getattr(args, side+'_label'), config=str(path),
            config_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            predictions=str(getattr(args, side+'_predictions')),
            predictions_sha256=hashlib.sha256(getattr(args, side+'_predictions').read_bytes()).hexdigest())
    keys = set(configs['reference']) | set(configs['candidate'])
    differences = {k: dict(reference=configs['reference'].get(k), candidate=configs['candidate'].get(k))
                   for k in sorted(keys) if configs['reference'].get(k) != configs['candidate'].get(k)}
    result = compare(args.reference_predictions, args.candidate_predictions)
    result.update(scope='exploratory_unmatched_training_protocol', sources=sources,
                  protocol_differences=differences, independent_test=False,
                  note='Paired descriptive outcomes only; not a controlled causal effect of one module. '
                       'Equal window counts do not establish identical window identities/FPS paths.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
