"""Production schedule is a best-effort wake-up, not the publication clock."""
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class ProductionScheduleTests(unittest.TestCase):
    def test_prepare_wake_and_recovery_schedule(self):
        text = (ROOT / ".github/workflows/main.yml").read_text()
        times = [(int(h), int(m)) for m, h in
                 re.findall(r"-\s*cron:\s*'(\d+) (\d+) \* \* \*'", text)]
        self.assertEqual(times, [(20, 17), (22, 43), (23, 17), (0, 21)])
        self.assertTrue(all(minute != 0 for _, minute in times))
        self.assertIn("workflow_dispatch:", text)
        self.assertIn("if: github.ref == 'refs/heads/main'", text)
        self.assertIn("cancel-in-progress: false", text)
        self.assertLess(text.index('daily_digest.py claim'), text.index('daily_digest.py send'))
        self.assertLess(text.index('Claim research email'), text.index('daily_digest.py send'))
        self.assertLess(text.index('SMTP accepted research email'), text.index('daily_digest.py finalize'))

    def test_pages_waits_for_artifact_and_skips_unchanged_tree(self):
        text = (ROOT / ".github/workflows/pages.yml").read_text()
        self.assertIn("needs.plan.outputs.deploy == 'true'", text)
        self.assertIn("python pages_guard.py", text)
        self.assertIn("actions: read", text)
        self.assertLess(text.index("Upload site artifact"), text.index("Wait for artifact metadata"))
        self.assertLess(text.index("Wait for artifact metadata"), text.index("Deploy to GitHub Pages"))


if __name__ == "__main__":
    unittest.main()
