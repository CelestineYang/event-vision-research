import tempfile
import unittest
from pathlib import Path

from audit_daily_dvs200 import parse_sample_path, select_files
from daily_dvs200_loader import inspect_event_file, load_events


DATASET_ROOT = Path(
    "/remote-home/datasets/Joind-dataset for Action Recognition/DailyDvs-200"
)
SAMPLE = DATASET_ROOT / "action_001/C0P10M0S2_20231118_09_44_16.aedat4"


class PathParsingTests(unittest.TestCase):
    def test_daily_dvs_filename(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "action_001/C0P10M0S2_20231118_09_44_16.aedat4"
            path.parent.mkdir()
            path.write_bytes(b"test")
            row = parse_sample_path(path, root)
        self.assertEqual(row["action_id"], 1)
        self.assertEqual(row["class_index"], 0)
        self.assertTrue(row["class_consistent"])
        self.assertEqual(row["person_id"], 10)
        self.assertEqual(row["motion_id"], 0)
        self.assertEqual(row["session_id"], 2)

    def test_selection_is_deterministic(self):
        paths = [Path(f"action_001/sample_{index}.aedat4") for index in range(10)]
        self.assertEqual(
            select_files(paths, per_class=3, max_files=None, seed=7),
            select_files(paths, per_class=3, max_files=None, seed=7),
        )


@unittest.skipUnless(SAMPLE.exists(), "DailyDVS-200 sample is unavailable")
class RealSampleTests(unittest.TestCase):
    def test_load_and_streaming_audit_agree(self):
        events = load_events(SAMPLE)
        stats = inspect_event_file(SAMPLE)
        self.assertEqual(stats["event_count"], len(events))
        self.assertEqual((stats["width"], stats["height"]), (320, 240))
        self.assertTrue(stats["timestamps_monotonic"])
        self.assertEqual(stats["invalid_coordinate_count"], 0)
        self.assertEqual(stats["on_count"], int(events["on"].sum()))


if __name__ == "__main__":
    unittest.main()
