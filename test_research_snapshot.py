"""CPU-only provenance/export edge cases; never load models or call CUDA."""
import csv
import io
from pathlib import Path
import tempfile
import unittest

from export_ucf101_research_snapshot import complete_metrics,check_predictions


class SnapshotTests(unittest.TestCase):
    def test_active_metric_partial_tail_and_epoch_gap(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'metrics.jsonl'
            path.write_bytes(b'{"epoch":1}\n{"epoch":2}\n{"epoch":')
            payload,rows=complete_metrics(path)
            self.assertEqual([r['epoch'] for r in rows],[1,2])
            self.assertEqual(payload,b'{"epoch":1}\n{"epoch":2}\n')
            path.write_bytes(b'{"epoch":1}\n{"epoch":3}\n')
            with self.assertRaises(ValueError): complete_metrics(path)

    def test_prediction_epoch_mismatch_and_population(self):
        stream=io.StringIO()
        writer=csv.DictWriter(stream,fieldnames=['video_id','label_id','prediction','windows','nll'])
        writer.writeheader();writer.writerow(dict(video_id='v1',label_id=2,prediction=2,windows=3,nll=.25))
        payload=stream.getvalue().encode()
        expected=dict(videos=1,windows=3,video_top1=1.,video_nll=.25)
        check_predictions(payload,expected)
        with self.assertRaises(ValueError): check_predictions(payload,dict(expected,video_nll=.5))
        with self.assertRaises(ValueError): check_predictions(payload,dict(expected,windows=4))


if __name__=='__main__': unittest.main()
