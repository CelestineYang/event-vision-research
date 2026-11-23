"""Build the complete DailyAction test manifest without inference or training."""
import argparse
import json
from pathlib import Path

import numpy as np


def build(store, output):
    store, output = Path(store), Path(output)
    records = [r for r in json.loads((store / 'split_records.json').read_text())
               if r['split'] == 'test']
    expected = {r['sample_id']: r['label'] for r in records}
    if len(expected) != len(records):
        raise ValueError('Duplicate test recording')
    rows = [json.loads(line) for line in (store / 'test_windows.jsonl').read_text().splitlines()
            if line.strip() and json.loads(line)['valid']]
    represented = {r['sample_id'] for r in rows}
    if not represented <= expected.keys():
        raise ValueError('Manifest contains an unknown test recording')
    added = []
    for record in records:
        if record['sample_id'] in represented:
            continue
        events = np.load(store / record['event_file'], mmap_mode='r', allow_pickle=False)
        if not len(events) or events[-1, 2] <= events[0, 2]:
            raise ValueError('Empty or zero-duration test recording: ' + record['sample_id'])
        if events[-1, 2] - events[0, 2] > 1_500_000:
            raise ValueError('Missing recording is not short: ' + record['sample_id'])
        added.append(dict(record, window_id=0, start_us=int(events[0, 2]),
                          end_us=int(events[-1, 2]), left=0, right=len(events),
                          raw_events=len(events), valid=True, point_row=None,
                          fallback='entire short recording; actual time span; no event-count exclusion'))
    rows.extend(added)
    if {r['sample_id'] for r in rows} != expected.keys():
        raise ValueError('Incomplete test coverage')
    if len({(r['sample_id'], r['window_id']) for r in rows}) != len(rows):
        raise ValueError('Duplicate test window')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        stream.writelines(json.dumps(row) + '\n' for row in rows)
    return dict(test_recordings=len(expected), windows=len(rows),
                added_short_recordings=len(added))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--store', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.store, args.output), indent=2))
