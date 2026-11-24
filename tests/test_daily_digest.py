import json
import tempfile
import unittest
from datetime import date, datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import daily_digest as d
import pages_guard as p


class DigestTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / ".daily-digest"
        self.site = Path(self.temp.name) / "site"
        self.day = date(2026, 11, 25)
        self.folder = self.root / self.day.isoformat()
        self.paper = SimpleNamespace(title="Battery", authors=[], source="Crossref",
                                     doi="10.1234/test", tldr="Summary")

    def prepare(self):
        d.prepare_digest(self.folder, self.day, [self.paper], [self.paper], "<p>digest</p>")

    def test_clock_uses_taipei_and_never_publishes_early(self):
        self.assertIsNone(d.seconds_until_publish(datetime(2026, 11, 24, 20, 17, tzinfo=timezone.utc)))
        self.assertEqual(d.seconds_until_publish(datetime(2026, 11, 24, 23, 17, tzinfo=timezone.utc)), 2580)
        self.assertEqual(d.seconds_until_publish(datetime(2026, 11, 25, 0, 0, tzinfo=timezone.utc)), 0)
        self.assertEqual(d.seconds_until_publish(datetime(2026, 11, 25, 4, 49, tzinfo=timezone.utc)), 0)

    def test_prepare_does_not_write_live_site_and_pins_local_day(self):
        self.prepare()
        self.assertEqual(d.phase(self.root, self.site, self.day), "prepared")
        self.assertFalse(self.site.exists())
        payload = json.loads((self.folder / "archive.json").read_text())
        self.assertEqual(payload["date"], self.day.isoformat())
        self.assertEqual(payload["papers"][0]["seen_date"], self.day.isoformat())

    def test_send_then_finalize_exactly_once_in_normal_execution(self):
        self.prepare()
        send = Mock()
        d.claim(self.root, self.site, self.day, "run", "1")
        d.deliver(self.root, self.site, self.day, "run", "1", send)
        self.assertEqual(d.phase(self.root, self.site, self.day), "accepted")
        self.assertFalse(self.site.exists())
        with self.assertRaises(RuntimeError):
            d.deliver(self.root, self.site, self.day, "run", "2", send)
        d.recover_accepted(self.root, self.site)
        self.assertEqual(d.phase(self.root, self.site, self.day), "sent")
        self.assertTrue(d.archive_path(self.site, self.day).is_file())
        self.assertTrue((self.site / "data" / "recommendation-history.json").is_file())
        self.assertTrue((self.site / "data" / "crossref-ingestion-history.json").is_file())
        self.assertFalse((self.folder / "email.html").exists())
        d.recover_accepted(self.root, self.site)
        send.assert_called_once_with("<p>digest</p>", self.day)

    def test_ambiguous_smtp_failure_cannot_be_reclaimed_by_backup(self):
        self.prepare()
        d.claim(self.root, self.site, self.day, "run", "1")
        send = Mock(side_effect=OSError("connection lost"))
        with self.assertRaises(OSError):
            d.deliver(self.root, self.site, self.day, "run", "1", send)
        self.assertEqual(d.phase(self.root, self.site, self.day), "sending")
        with self.assertRaises(RuntimeError):
            d.claim(self.root, self.site, self.day, "backup", "1")
        with self.assertRaises(RuntimeError):
            d.deliver(self.root, self.site, self.day, "backup", "1", send)
        send.assert_called_once()

    def test_archive_failure_recovers_without_resending(self):
        self.prepare()
        send = Mock()
        d.claim(self.root, self.site, self.day, "run", "1")
        d.deliver(self.root, self.site, self.day, "run", "1", send)
        with patch.object(d, "publish_daily_payload", side_effect=OSError("disk")):
            with self.assertRaises(OSError):
                d.finalize(self.root, self.site, self.day)
        self.assertEqual(d.phase(self.root, self.site, self.day), "accepted")
        d.recover_accepted(self.root, self.site)
        self.assertEqual(d.phase(self.root, self.site, self.day), "sent")
        send.assert_called_once()

    def test_legacy_archive_and_empty_digest_do_not_send(self):
        path = d.archive_path(self.site, self.day)
        path.parent.mkdir(parents=True)
        path.write_text("{}")
        self.assertEqual(d.phase(self.root, self.site, self.day), "legacy")
        path.unlink()
        d.prepare_empty(self.folder, self.day)
        self.assertEqual(d.phase(self.root, self.site, self.day), "empty")


class PagesTests(unittest.TestCase):
    def test_skipped_and_failed_deployments_do_not_count(self):
        for result in ("skipped", "failure"):
            jobs = [{"steps": [{"name": "Deploy to GitHub Pages", "conclusion": result}]}]
            self.assertIsNone(p.actual_deployed_tree(jobs, "sha", Mock()))

    def test_exact_tree_overrides_event_sha(self):
        tree = "a" * 40
        jobs = [{"steps": [{"name": "Deploy to GitHub Pages", "conclusion": "success"},
                          {"name": "Record deployed site tree " + tree, "conclusion": "success"}]}]
        resolve = Mock()
        self.assertEqual(p.actual_deployed_tree(jobs, "old", resolve), tree)
        resolve.assert_not_called()

    def test_look_past_successful_skipped_run(self):
        query = Mock(side_effect=[
            {"workflow_runs": [{"id": 3, "head_sha": "new"}, {"id": 2, "head_sha": "old"}]},
            {"jobs": [{"steps": [{"name": "Deploy to GitHub Pages", "conclusion": "skipped"}]}]},
            {"jobs": [{"steps": [{"name": "Deploy to GitHub Pages", "conclusion": "success"}]}]},
        ])
        resolve = Mock(return_value="deployed-tree")
        self.assertEqual(p.last_deployed_tree("owner/repo", 4, query, resolve), "deployed-tree")
        resolve.assert_called_once_with("old")


if __name__ == "__main__":
    unittest.main()
