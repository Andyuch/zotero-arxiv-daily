"""Offline tests for overlap-safe Crossref ingestion state."""
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from crossref_ingestion import (
    crossref_keys,
    load_ingestion_history,
    partition_unseen_crossref,
    save_ingestion_history,
)


NOW = datetime(2026, 9, 30, 22, 0, tzinfo=timezone.utc)


def paper(title, doi=None, source="Crossref"):
    return SimpleNamespace(title=title, doi=doi, source=source)


class CrossrefIdentityTests(unittest.TestCase):
    def test_doi_and_title_are_both_recorded(self):
        keys = crossref_keys(paper("A Journal Paper", "https://doi.org/10.1234/ABC"))
        self.assertIn("doi:10.1234/abc", keys)
        self.assertIn("title:ajournalpaper", keys)

    def test_arxiv_identity_is_not_part_of_crossref_ledger(self):
        p = SimpleNamespace(
            title="Shared title",
            doi=None,
            arxiv_id="2609.12345v2",
            source="Crossref",
        )
        keys = crossref_keys(p)
        self.assertFalse(any(key.startswith("arxiv:") for key in keys))
        self.assertIn("title:sharedtitle", keys)


class CrossrefLedgerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "data" / "daily").mkdir(parents=True)

    def test_bootstrap_from_existing_crossref_archive(self):
        payload = {
            "date": "2026-09-29",
            "papers": [
                {
                    "title": "Already shown",
                    "doi": "10.1234/already",
                    "source": "Crossref",
                },
                {
                    "title": "arXiv item",
                    "arxiv_id": "2609.00001",
                    "source": "arXiv",
                },
            ],
        }
        (self.root / "data" / "daily" / "2026-09-29.json").write_text(
            json.dumps(payload),
            encoding="utf-8",
        )
        history = load_ingestion_history(str(self.root))
        self.assertIn("doi:10.1234/already", history)
        self.assertNotIn("title:arxivitem", history)

    def test_overlap_record_is_suppressed_before_ranking(self):
        old = paper("Boundary paper", "10.1234/boundary")
        fresh = paper("New paper", "10.1234/new")
        history = {key: "2026-09-29T22:00:00Z" for key in crossref_keys(old)}
        unseen, seen = partition_unseen_crossref([old, fresh], history)
        self.assertEqual(unseen, [fresh])
        self.assertEqual(seen, [old])

    def test_title_fallback_suppresses_record_if_doi_spelling_changes(self):
        old = paper("Same Useful Title", None)
        new = paper("Same useful title!", "10.1234/new-doi")
        history = {key: "2026-09-29T22:00:00Z" for key in crossref_keys(old)}
        unseen, seen = partition_unseen_crossref([new], history)
        self.assertEqual(unseen, [])
        self.assertEqual(seen, [new])

    def test_successful_save_records_all_observed_not_only_recommended(self):
        first = paper("Selected", "10.1234/selected")
        second = paper("Not selected", "10.1234/unselected")
        path = save_ingestion_history([first, second], str(self.root), now=NOW)
        payload = json.loads(path.read_text())
        self.assertEqual(payload["version"], 1)
        for p in (first, second):
            self.assertTrue(crossref_keys(p) <= payload["entries"].keys())

    def test_first_seen_timestamp_is_preserved_across_overlap_reruns(self):
        p = paper("Persistent", "10.1234/persistent")
        first = datetime(2026, 9, 30, 22, 0, tzinfo=timezone.utc)
        later = datetime(2026, 10, 1, 22, 0, tzinfo=timezone.utc)
        save_ingestion_history([p], str(self.root), now=first)
        save_ingestion_history([p], str(self.root), now=later)
        history = load_ingestion_history(str(self.root))
        self.assertEqual(
            history["doi:10.1234/persistent"],
            "2026-09-30T22:00:00Z",
        )

    def test_failed_atomic_replace_preserves_previous_ledger(self):
        old = paper("Old", "10.1234/old")
        fresh = paper("Fresh", "10.1234/fresh")
        ledger = save_ingestion_history([old], str(self.root), now=NOW)
        before = ledger.read_bytes()
        with patch("crossref_ingestion.os.replace", side_effect=OSError("disk failure")):
            with self.assertRaises(OSError):
                save_ingestion_history([fresh], str(self.root), now=NOW)
        self.assertEqual(ledger.read_bytes(), before)
        self.assertFalse(crossref_keys(fresh) & load_ingestion_history(str(self.root)).keys())

    def test_invalid_ledger_fails_closed(self):
        ledger = self.root / "data" / "crossref-ingestion-history.json"
        for payload in (
            {"version": 2, "entries": {}},
            {"version": 1, "entries": []},
            {"version": 1, "entries": {"doi:10.1234/x": None}},
        ):
            with self.subTest(payload=payload):
                ledger.write_text(json.dumps(payload), encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_ingestion_history(str(self.root))


if __name__ == "__main__":
    unittest.main()
