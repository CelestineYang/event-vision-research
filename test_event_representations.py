import unittest
from pathlib import Path

import numpy as np

from daily_dvs200_loader import EVENT_DTYPE, load_events
from event_representations import build_representations


DATASET_ROOT = Path(
    "/remote-home/datasets/Joind-dataset for Action Recognition/DailyDvs-200"
)
SAMPLE = DATASET_ROOT / "action_001/C0P10M0S2_20231118_09_44_16.aedat4"


def make_events(rows):
    return np.array(rows, dtype=EVENT_DTYPE)


class RepresentationTests(unittest.TestCase):
    def test_shapes_counts_polarity_and_last_bin(self):
        events = make_events(
            [
                (100, 0, 0, False),
                (125, 1, 0, True),
                (150, 1, 1, True),
                (200, 3, 1, False),
            ]
        )
        result = build_representations(
            events, height=2, width=4, num_bins=4, points_per_bin=2,
            deterministic=True,
        )
        self.assertEqual(result["frame"].shape, (2, 2, 4))
        self.assertEqual(result["voxel"].shape, (2, 4, 2, 4))
        self.assertTrue(np.array_equal(result["frame"], result["voxel"].sum(axis=1)))
        self.assertEqual(int(result["frame"].sum()), len(events))
        self.assertEqual(int(result["voxel"].sum()), len(events))
        self.assertEqual(int(result["frame"][0].sum()), 2)
        self.assertEqual(int(result["frame"][1].sum()), 2)
        self.assertEqual(result["voxel"][0, 3, 1, 3], 1)
        self.assertEqual(result["duration_us"], 100)
        self.assertEqual(result["point_mask"].sum(), 4)
        self.assertEqual(set(result["points"][:4, 3]), {-1.0, 1.0})

    def test_empty_and_zero_duration_streams(self):
        empty = np.empty(0, dtype=EVENT_DTYPE)
        empty_result = build_representations(
            empty, height=2, width=3, num_bins=2, points_per_bin=2
        )
        self.assertEqual(empty_result["duration_us"], 0)
        self.assertEqual(int(empty_result["voxel"].sum()), 0)
        self.assertFalse(empty_result["point_mask"].any())
        self.assertTrue(np.all(empty_result["point_bin"] == -1))

        simultaneous = make_events([(50, 0, 0, False), (50, 1, 1, True)])
        result = build_representations(
            simultaneous, height=2, width=2, num_bins=3, points_per_bin=2,
            deterministic=True,
        )
        self.assertEqual(result["duration_us"], 0)
        self.assertEqual(int(result["voxel"][:, 0].sum()), 2)
        self.assertEqual(int(result["voxel"][:, 1:].sum()), 0)
        self.assertTrue(np.all(result["points"][:2, 2] == 0.0))

    def test_deterministic_and_seeded_sampling(self):
        events = make_events(
            [(index, index % 8, index % 4, bool(index % 2)) for index in range(100)]
        )
        kwargs = dict(height=4, width=8, num_bins=2, points_per_bin=5)
        first = build_representations(events, deterministic=True, **kwargs)
        second = build_representations(events, deterministic=True, seed=999, **kwargs)
        self.assertTrue(np.array_equal(first["points"], second["points"]))

        seeded_a = build_representations(events, deterministic=False, seed=7, **kwargs)
        seeded_b = build_representations(events, deterministic=False, seed=7, **kwargs)
        seeded_c = build_representations(events, deterministic=False, seed=8, **kwargs)
        self.assertTrue(np.array_equal(seeded_a["points"], seeded_b["points"]))
        self.assertFalse(np.array_equal(seeded_a["points"], seeded_c["points"]))
        valid_bins = seeded_a["point_bin"][seeded_a["point_mask"]]
        self.assertEqual(np.count_nonzero(valid_bins == 0), 5)
        self.assertEqual(np.count_nonzero(valid_bins == 1), 5)

    def test_rejects_invalid_coordinates_and_timestamp_order(self):
        invalid_xy = make_events([(0, 3, 0, True)])
        with self.assertRaisesRegex(ValueError, "outside"):
            build_representations(invalid_xy, height=2, width=3)

        reversed_t = make_events([(2, 0, 0, True), (1, 0, 0, False)])
        with self.assertRaisesRegex(ValueError, "monotonic"):
            build_representations(reversed_t, height=2, width=3)


@unittest.skipUnless(SAMPLE.exists(), "DailyDVS-200 sample is unavailable")
class RealRepresentationTests(unittest.TestCase):
    def test_real_sample_conserves_all_events(self):
        events = load_events(SAMPLE)
        result = build_representations(events, deterministic=True)
        self.assertEqual(result["frame"].shape, (2, 240, 320))
        self.assertEqual(result["voxel"].shape, (2, 8, 240, 320))
        self.assertTrue(np.array_equal(result["frame"], result["voxel"].sum(axis=1)))
        self.assertEqual(int(result["frame"].sum()), len(events))
        self.assertLessEqual(int(result["point_mask"].sum()), 2048)
        valid_points = result["points"][result["point_mask"]]
        self.assertTrue(np.all(valid_points[:, :3] >= -1.0))
        self.assertTrue(np.all(valid_points[:, :3] <= 1.0))


if __name__ == "__main__":
    unittest.main()
