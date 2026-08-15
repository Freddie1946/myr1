#!/usr/bin/env python3

import csv
import json
import tempfile
import unittest
from pathlib import Path

from serve_expert_review_app import EVENTS, RatingStore, process_score


class RatingStoreTest(unittest.TestCase):
    def test_exact_six_events_are_saved_and_exported(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = RatingStore(Path(temporary))
            events = {field: False for field in EVENTS}
            events[EVENTS[0]] = True
            events[EVENTS[1]] = True
            events[EVENTS[2]] = True
            record, count = store.save({
                "reviewer_id": "pathologist_A",
                "case_id": "HRA-001",
                "events": events,
                "reviewer_notes": "checked",
            })
            self.assertEqual(count, 1)
            self.assertEqual(record["process_score"], 1.0)
            self.assertEqual(store.load("pathologist_A", "HRA-001")["events"], events)
            with (Path(temporary) / "pathologist_A/reward_six_event_ratings.csv").open(encoding="utf-8-sig") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["process_score"], "1.000000")

    def test_rejects_extra_or_nonboolean_fields(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = RatingStore(Path(temporary))
            events = {field: False for field in EVENTS} | {"extra": False}
            with self.assertRaisesRegex(ValueError, "exactly"):
                store.save({"reviewer_id": "A", "case_id": "HRA-001", "events": events, "reviewer_notes": ""})
            events = {field: False for field in EVENTS}
            events[EVENTS[0]] = "false"
            with self.assertRaisesRegex(ValueError, "booleans"):
                store.save({"reviewer_id": "A", "case_id": "HRA-001", "events": events, "reviewer_notes": ""})

    def test_score_geometry(self):
        all_good = {field: True for field in EVENTS[:3]} | {field: False for field in EVENTS[3:]}
        self.assertEqual(process_score(all_good), 1.0)
        one_missing = dict(all_good)
        one_missing[EVENTS[0]] = False
        self.assertAlmostEqual(process_score(one_missing), 0.8)
        all_bad = {field: False for field in EVENTS[:3]} | {field: True for field in EVENTS[3:]}
        self.assertEqual(process_score(all_bad), 0.0)


if __name__ == "__main__":
    unittest.main()
