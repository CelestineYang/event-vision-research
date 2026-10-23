"""Export the explicitly approved small completed routing experiment snapshot."""
import json
from pathlib import Path
import shutil
import torch
from routing_data import digest, json_write

BASE = Path('/remote-home/datasets/event_project_cache/routing_epoch040_static_20261022_v3')
DEST = Path('research_exports/ucf101_routing_epoch040_2026-10-23')


def main():
    if json.loads((BASE/'dynamic_status.json').read_text())['jobs'].keys() != {'current','gru','conditional'}:
        raise ValueError('Unexpected run set')
    if DEST.exists(): raise FileExistsError(DEST)
    DEST.mkdir(parents=True)
    sources, summary = {}, {}
    for mode in ('static','current','gru','conditional'):
        source = BASE/mode
        status = json.loads((source/'status.json').read_text())
        rows = [json.loads(line) for line in (source/'metrics.jsonl').read_text().splitlines()]
        if status['status'] != 'complete' or [r['epoch'] for r in rows] != list(range(1,81)):
            raise ValueError('Incomplete training: '+mode)
        best = min(rows,key=lambda r:r['validation']['video_nll'])
        saved = torch.load(source/'best_val_nll.pt',map_location='cpu',weights_only=False)
        if saved['epoch'] != best['epoch']: raise ValueError('Checkpoint selection mismatch')
        predictions = json.loads((source/'validation_predictions_best.json').read_text())
        if len(predictions) != 2555 or sum(p['windows'] for p in predictions) != 27384:
            raise ValueError('Prediction coverage mismatch')
        target = DEST/mode; target.mkdir()
        for filename in ('config.json','metrics.jsonl','validation_predictions_best.json','status.json'):
            src=source/filename
            shutil.copyfile(src,target/filename)
            sources[f'{mode}/{filename}']=dict(source=str(src),sha256=digest(src))
        sources[f'{mode}/best_val_nll.pt']=dict(source=str(source/'best_val_nll.pt'),
                                               sha256=digest(source/'best_val_nll.pt'),included=False)
        summary[mode]=dict(selected_epoch=best['epoch'],validation=best['validation'],
                           final_validation=rows[-1]['validation'],epochs=80,optimizer_steps=status['optimizer_steps'])
    json_write(DEST/'summary.json',dict(selection_protocol='original_test_as_val',independent_test=False,
               expert_checkpoint_epoch=40,selection_metric='video_nll',ema_evaluated=False,results=summary))
    for split in ('train','test'):
        # Identity metadata only, no logits or event data.
        path=BASE/split/'manifest.json'
        if not path.exists(): raise FileNotFoundError(path)
        sources[f'evidence_{split}']=dict(source=str(path),sha256=digest(path))
    json_write(DEST/'sources.json',sources)
    print(DEST)


if __name__=='__main__':main()
