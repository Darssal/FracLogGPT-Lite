import unittest
from collections import Counter

from fraclog.data import stratified_split


class SplitTests(unittest.TestCase):
    def test_stratified_split_is_deterministic_and_preserves_all_rows(self):
        rows = []
        for label in ["0", "1", "2", ">=3"]:
            for idx in range(10):
                rows.append({"id": f"{label}-{idx}", "count_label": label})

        first = stratified_split(rows, label_key="count_label", seed=42, train_ratio=0.8, val_ratio=0.1)
        second = stratified_split(rows, label_key="count_label", seed=42, train_ratio=0.8, val_ratio=0.1)

        self.assertEqual(first, second)
        all_ids = [row["id"] for split_rows in first.values() for row in split_rows]
        self.assertEqual(len(all_ids), len(set(all_ids)))
        self.assertEqual(len(all_ids), 40)
        self.assertEqual({name: len(values) for name, values in first.items()}, {"train": 32, "val": 4, "test": 4})

    def test_stratified_split_keeps_each_class_in_each_split_when_possible(self):
        rows = []
        for label in ["0", "1", "2", ">=3"]:
            for idx in range(20):
                rows.append({"id": f"{label}-{idx}", "count_label": label})

        splits = stratified_split(rows, label_key="count_label", seed=42, train_ratio=0.8, val_ratio=0.1)

        for split_name, expected_per_class in [("train", 16), ("val", 2), ("test", 2)]:
            counts = Counter(row["count_label"] for row in splits[split_name])
            self.assertEqual(counts, Counter({"0": expected_per_class, "1": expected_per_class, "2": expected_per_class, ">=3": expected_per_class}))


if __name__ == "__main__":
    unittest.main()
