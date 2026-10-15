import unittest

from create_gcstr_splits import stratified_train_validation_split


class GCSTRSplitTests(unittest.TestCase):
    def test_split_is_exact_stratified_and_deterministic(self):
        rows = [
            {"sample_id": f"{label}_{index}", "label_id": str(label)}
            for label in range(4)
            for index in range(10)
        ]
        first = stratified_train_validation_split(rows, 0.2, seed=0)
        second = stratified_train_validation_split(rows, 0.2, seed=0)
        self.assertEqual(first, second)
        training, validation = first
        self.assertEqual((len(training), len(validation)), (32, 8))
        self.assertEqual(
            {label: sum(int(row["label_id"]) == label for row in validation) for label in range(4)},
            {0: 2, 1: 2, 2: 2, 3: 2},
        )
        self.assertFalse(
            {row["sample_id"] for row in training}
            & {row["sample_id"] for row in validation}
        )


if __name__ == "__main__":
    unittest.main()
