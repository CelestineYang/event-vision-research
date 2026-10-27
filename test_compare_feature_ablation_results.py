import csv
import json
from pathlib import Path
import tempfile
import unittest

from compare_feature_ablation_results import compare


class PairTests(unittest.TestCase):
    def test_csv_json_historical_pairing(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp)/'a.csv', Path(tmp)/'b.json'
            rows = [dict(video_id='a', label_id=0, prediction=1, windows=2, nll=2.),
                    dict(video_id='b', label_id=1, prediction=1, windows=3, nll=.1)]
            with a.open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader(); writer.writerows(rows)
            b.write_text(json.dumps(dict(main=[dict(video_id='a', label=0, prediction=0, windows=2, nll=.2),
                                               dict(video_id='b', label=1, prediction=0, windows=3, nll=2.)])))
            result = compare(a, b)
            self.assertEqual(result['rescued'], ['a'])
            self.assertEqual(result['damaged'], ['b'])
            self.assertEqual(result['net_correct'], 0)
            self.assertAlmostEqual(result['video_pairs'][0]['delta_nll'], -1.8)

    def test_coverage_labels_and_duplicates_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp)/'a.json', Path(tmp)/'b.json'
            row = dict(video_id='a', label=0, prediction=0, windows=2, nll=.2)
            a.write_text(json.dumps([row]))
            for bad in ([dict(row, label=1)], [dict(row, windows=3)], [dict(row, video_id='b')], [row, row]):
                b.write_text(json.dumps(bad))
                with self.assertRaises(ValueError):
                    compare(a, b)


if __name__ == '__main__':
    unittest.main()
