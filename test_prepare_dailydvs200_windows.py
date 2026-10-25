import unittest
import numpy as np
from prepare_dailydvs200_windows import bounds, summarize


class WindowTests(unittest.TestCase):
    def test_short_preserves_last_without_fabrication(self):
        w = bounds(np.array([100, 200, 300]))
        self.assertEqual(len(w), 1)
        self.assertEqual(w[0]['event_count'], 3)
        self.assertEqual(w[0]['end_us'], 1000100)
        self.assertTrue(w[0]['short_sample'])

    def test_half_open_overlap_and_tail(self):
        w = bounds(np.array([0, 500000, 1000000, 1500000, 1600000]))
        self.assertEqual([x['start_us'] for x in w], [0, 500000])
        self.assertEqual([x['event_count'] for x in w], [2, 2])
        self.assertEqual(w[-1]['right'], 3)

    def test_sparse_window_is_not_removed(self):
        w = bounds(np.array([0, 3000000]))
        self.assertEqual(len(w), 5)
        self.assertTrue(any(x['event_count'] == 0 for x in w))

    def test_invalid_order(self):
        for t in (np.array([]), np.array([5, 1])):
            with self.assertRaises(ValueError):
                bounds(t)

    def test_threshold_strict_and_sample_coverage(self):
        rows = [dict(split='train', status='ok', sample_id='a', label_id=0,
                     event_count=10000, trailing_uncovered_events=0,
                     windows=[dict(event_count=10000, short_sample=False)])]
        result = summarize(rows)['train']['threshold_candidates']
        self.assertEqual(result['10000']['retained_windows'], 0)
        self.assertEqual(result['10000']['samples_without_windows'], ['a'])
        self.assertEqual(result['2048']['retained_windows'], 1)


if __name__ == '__main__':
    unittest.main()
