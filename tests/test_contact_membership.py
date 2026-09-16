"""Bulk reads must preserve unordered membership and the list-contact contract."""

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Hooke"))
from contact_state import touching


class ContactMembership(unittest.TestCase):
    def test_bulk_pairs_agree_with_reference_for_sparse_groups_and_negative_ids(self):
        rng = np.random.default_rng(7)
        for _ in range(100):
            pairs = rng.integers(-1, 300, (int(rng.integers(0, 1000)), 2))
            first = set(map(int, rng.integers(0, 300, int(rng.integers(0, 10)))))
            second = set(map(int, rng.integers(0, 300, int(rng.integers(0, 10)))))
            expected = any(
                (int(a) in first and int(b) in second)
                or (int(b) in first and int(a) in second)
                for a, b in pairs
            )
            self.assertEqual(
                touching(
                    SimpleNamespace(contact=SimpleNamespace(geom=pairs)), first, second
                ),
                expected,
            )
            contacts = [SimpleNamespace(geom=pair) for pair in pairs]
            self.assertEqual(
                touching(SimpleNamespace(contact=contacts), first, second), expected
            )


if __name__ == "__main__":
    unittest.main()
