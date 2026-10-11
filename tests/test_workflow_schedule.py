"""Regression checks for the production daily GitHub Actions schedule."""
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "main.yml"


class ProductionScheduleTests(unittest.TestCase):
    def test_daily_schedule_exists_and_avoids_top_of_hour(self):
        text = WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("schedule:", text)

        match = re.search(
            r"-\s*cron:\s*['\"](\d{1,2})\s+(\d{1,2})\s+\*\s+\*\s+\*['\"]",
            text,
        )
        self.assertIsNotNone(match, "Daily cron schedule is missing from main.yml")

        minute, hour = map(int, match.groups())
        self.assertNotEqual(
            minute,
            0,
            "Avoid minute 0 because GitHub scheduled workflows are most congested at the top of the hour.",
        )
        self.assertEqual(
            (minute, hour),
            (17, 20),
            "Production schedule should target 20:17 UTC (about 07:00 UTC+8 after observed GitHub delay).",
        )

    def test_manual_dispatch_remains_available(self):
        text = WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("workflow_dispatch:", text)


if __name__ == "__main__":
    unittest.main()
