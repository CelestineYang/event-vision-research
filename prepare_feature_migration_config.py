"""Generate relocated configs without modifying audited data or protected source."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    root = args.bundle.resolve()
    if json.loads((root/'COMPLETE.json').read_text())['status'] != 'verified':
        raise RuntimeError('Migration package not complete')
    args.output.mkdir(parents=True, exist_ok=False)
    for mode in ('B', 'G', 'GM'):
        cfg = json.loads((root/'code/configs/feature_memory'/f'{mode}.json').read_text())
        cfg.update(experiment=str(root/'data/experiment'), store=str(root/'data/event_store'),
                   source_checkpoint=str(root/'weights/epoch_040.pt'))
        (args.output/f'{mode}.json').write_text(json.dumps(cfg, indent=2)+'\n')
    print('Generated B/G/GM configs:', args.output)


if __name__ == '__main__':
    main()
