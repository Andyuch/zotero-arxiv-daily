"""Exercise the real commit helper against a disposable local bare remote."""
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from recommendation_history import save_history

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(shutil.which("git") and shutil.which("bash"), "git and bash required")
class WorkflowCommitTests(unittest.TestCase):
    def test_untracked_ledger_is_committed_and_pushed(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            remote = base / "remote.git"
            repo = base / "work"
            repo.mkdir()
            def git(*args, check=True):
                return subprocess.run(["git", *args], cwd=repo, check=check,
                                      text=True, capture_output=True)
            git("init", "-q", "-b", "main")
            git("init", "--bare", str(remote))
            git("remote", "add", "origin", str(remote))
            git("config", "user.name", "Offline Test")
            git("config", "user.email", "offline-test@example.invalid")
            (repo / "README.md").write_text("Test repository\n")
            git("add", "README.md")
            git("commit", "-qm", "Initial fixture")
            git("push", "origin", "main")
            initial = git("rev-parse", "HEAD").stdout.strip()
            save_history([SimpleNamespace(title="Successfully delivered")], str(repo / "site"))
            self.assertEqual(git("diff", "--quiet", "--", "site/data", check=False).returncode, 0)
            subprocess.run(["bash", str(ROOT / "scripts/commit-digest.sh"),
                            "Update archive", "site/data"], cwd=repo, check=True,
                           text=True, capture_output=True)
            self.assertNotEqual(git("rev-parse", "HEAD").stdout.strip(), initial)
            committed = git("show", "origin/main:site/data/recommendation-history.json").stdout
            self.assertIn("title:successfullydelivered", committed)
            self.assertEqual(git("status", "--porcelain").stdout.strip(), "")


if __name__ == "__main__":
    unittest.main()
