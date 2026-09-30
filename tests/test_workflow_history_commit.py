"""Exercise the archive commit shell in a disposable repository, without push."""
import shutil
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path

from recommendation_history import save_history
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(shutil.which('git') and shutil.which('bash'), 'git and bash required')
class WorkflowCommitTests(unittest.TestCase):
    def test_first_untracked_ledger_is_committed_even_without_archive_changes(self):
        workflow = (ROOT / '.github' / 'workflows' / 'main.yml').read_text()
        block = workflow.split('- name: Commit updated research archive', 1)[1]
        script = textwrap.dedent(block.split('run: |', 1)[1]).strip()
        # Keep the actual commit/detection code; explicitly remove its remote action.
        lines = [line for line in script.splitlines() if not line.strip().startswith('git push ')]
        script = '\n'.join(lines)
        self.assertNotIn('git push', script)
        self.assertLess(script.index('git add site/data'), script.index('git diff --cached --quiet'))
        with tempfile.TemporaryDirectory() as folder:
            def git(*args, check=True):
                return subprocess.run(['git', *args], cwd=folder, check=check,
                                      text=True, capture_output=True)
            git('init', '-q')
            git('config', 'user.name', 'Offline Test')
            git('config', 'user.email', 'offline-test@example.invalid')
            Path(folder, 'README.md').write_text('Test repository\n')
            git('add', 'README.md')
            git('commit', '-qm', 'Initial fixture')
            initial = git('rev-parse', 'HEAD').stdout.strip()
            save_history([SimpleNamespace(title='Successfully delivered')], str(Path(folder) / 'site'))
            self.assertEqual(git('diff', '--quiet', '--', 'site/data', check=False).returncode, 0)
            subprocess.run(['bash', '-e'], input=script, cwd=folder, check=True,
                           text=True, capture_output=True)
            self.assertNotEqual(git('rev-parse', 'HEAD').stdout.strip(), initial)
            committed = git('show', 'HEAD:site/data/recommendation-history.json').stdout
            self.assertIn('title:successfullydelivered', committed)
            self.assertEqual(git('status', '--porcelain').stdout.strip(), '')


if __name__ == '__main__':
    unittest.main()
