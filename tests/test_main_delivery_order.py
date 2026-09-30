"""Execute the actual main orchestration with every external boundary replaced.

Compile only its __main__ guard to avoid importing model/native dependencies.
No rewritten copy of the pipeline and no live external services are involved.
"""
import ast
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from recommendation_history import select_recommendations

ROOT = Path(__file__).resolve().parents[1]


class MainDeliveryTests(unittest.TestCase):
    def run_main(self, send_error=None, publish_site=True, render_error=None, archive_error=None):
        parsed = ast.parse((ROOT / 'main.py').read_text())
        guard = next(node for node in parsed.body if isinstance(node, ast.If)
                     and isinstance(node.test, ast.Compare)
                     and isinstance(node.test.left, ast.Name)
                     and node.test.left.id == '__name__')
        executable = compile(ast.Module(body=guard.body, type_ignores=[]), str(ROOT / 'main.py'), 'exec')
        args = SimpleNamespace(recommendation_cooldown_days=7, max_paper_num=1,
            site_output_dir='not-a-real-output-directory', use_llm_api=False,
            openai_api_key=None, debug=False, zotero_id=None, zotero_key=None,
            zotero_ignore=None, arxiv_query='', enable_crossref=False,
            send_empty=False, language='English', sender=None, receiver=None,
            sender_password=None, smtp_server=None, smtp_port=None,
            publish_site=publish_site)
        first = SimpleNamespace(title='Chosen')
        second = SimpleNamespace(title='Not chosen')
        calls = []
        parser = MagicMock()
        parser.parse_args.return_value = args
        def record(name, result=None):
            def method(*values):
                calls.append((name, values))
                if name == 'send' and send_error:
                    raise send_error
                if name == 'render' and render_error:
                    raise render_error
                if name == 'archive' and archive_error:
                    raise archive_error
                return result
            return method
        namespace = dict(add_argument=MagicMock(), parser=parser, sys=sys,
            logger=MagicMock(), load_history=MagicMock(return_value={}),
            get_zotero_corpus=MagicMock(return_value=[]),
            get_arxiv_paper=MagicMock(return_value=[first, second]),
            deduplicate_papers=lambda values: values,
            rerank_paper=record('rank', [first, second]),
            select_recommendations=select_recommendations,
            set_global_llm=MagicMock(), render_email=record('render', '<html/>'),
            send_email=record('send'), save_history=record('save'),
            update_site_archive=record('archive'))
        try:
            exec(executable, namespace)
        except RuntimeError:
            if send_error is None and render_error is None:
                raise
        return calls, first, second

    def test_only_selected_papers_render_and_persist_after_successful_send(self):
        calls, selected, unselected = self.run_main()
        self.assertEqual([name for name, _ in calls], ['rank', 'render', 'send', 'save', 'archive'])
        for name, values in calls:
            if name in {'render', 'save', 'archive'}:
                self.assertEqual(values[0], [selected])
                self.assertNotIn(unselected, values[0])

    def test_failed_send_does_not_write_history_or_site(self):
        calls, _, _ = self.run_main(send_error=RuntimeError('SMTP unavailable'))
        self.assertEqual([name for name, _ in calls], ['rank', 'render', 'send'])

    def test_failed_render_does_not_send_or_write_history_or_site(self):
        calls, _, _ = self.run_main(render_error=RuntimeError('TLDR unavailable'))
        self.assertEqual([name for name, _ in calls], ['rank', 'render'])

    def test_archive_failure_occurs_after_successful_history_save(self):
        calls, _, _ = self.run_main(archive_error=RuntimeError('archive write failed'))
        self.assertEqual([name for name, _ in calls], ['rank', 'render', 'send', 'save', 'archive'])

    def test_disabled_site_still_persists_delivery_history(self):
        calls, _, _ = self.run_main(publish_site=False)
        self.assertEqual([name for name, _ in calls], ['rank', 'render', 'send', 'save'])


if __name__ == '__main__':
    unittest.main()
