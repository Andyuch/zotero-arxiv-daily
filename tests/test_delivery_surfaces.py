"""Presentation and source-window regressions; never contact external services."""
import importlib.util
import json
import sys
import tempfile
import types
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import site_builder

ROOT = Path(__file__).resolve().parents[1]


def load_isolated(name, stubs):
    spec = importlib.util.spec_from_file_location('_test_' + name, ROOT / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, stubs):
        spec.loader.exec_module(module)
    return module


def simple_module(name, **values):
    module = types.ModuleType(name)
    module.__dict__.update(values)
    return module


def example_paper(**values):
    defaults = dict(title='Battery finding', summary='A battery abstract', authors=[],
                    journal='Nature', source='Crossref', doi='10.1234/test',
                    paper_url='https://doi.org/10.1234/test', pdf_url=None,
                    code_url=None, affiliations=[], score=0.8, tldr='A useful result')
    defaults.update(values)
    return SimpleNamespace(**defaults)


class SiteTests(unittest.TestCase):
    def test_new_fields_default_for_legacy_paper_objects(self):
        record = site_builder._record_from_paper(example_paper(), '2026-09-30')
        self.assertEqual(record['recommendation_status'], 'new')
        self.assertIsNone(record['previous_recommended_at'])
        self.assertEqual(record['seen_date'], '2026-09-30')
        self.assertEqual(record['doi'], '10.1234/test')

    def test_repeat_status_and_previous_date_survive_archive_and_index(self):
        p = example_paper(recommendation_status='repeat_highlight', previous_recommended_at='2026-09-29')
        with tempfile.TemporaryDirectory() as folder:
            index_path = site_builder.update_site_archive([p], folder)
            index = json.loads(index_path.read_text())
            record = index['papers'][0]
            self.assertEqual(record['recommendation_status'], 'repeat_highlight')
            self.assertEqual(record['previous_recommended_at'], '2026-09-29')
            daily = list((Path(folder) / 'data' / 'daily').glob('*.json'))
            self.assertEqual(len(daily), 1)
            daily_record = json.loads(daily[0].read_text())['papers'][0]
            self.assertEqual(daily_record['recommendation_status'], 'repeat_highlight')

    def test_rebuild_accepts_legacy_records_without_new_fields(self):
        with tempfile.TemporaryDirectory() as folder:
            data = Path(folder)
            daily = data / 'daily'
            daily.mkdir()
            (daily / '2026-09-29.json').write_text(json.dumps({
                'date': '2026-09-29', 'papers': [{'title': 'Old paper', 'doi': '10.1234/old'}]
            }))
            index = site_builder._rebuild_index(data)
            self.assertEqual(index['paper_count'], 1)
            self.assertEqual(index['papers'][0]['title'], 'Old paper')

    def test_date_filter_metadata_preserves_original_labels_and_legacy_defaults(self):
        with tempfile.TemporaryDirectory() as folder:
            data = Path(folder)
            daily = data / 'daily'
            daily.mkdir()
            for day, metadata in (
                ('2026-09-28', {}),
                ('2026-09-29', {'recommendation_status': 'repeat_highlight',
                                'previous_recommended_at': '2026-09-28'}),
                ('2026-10-07', {'recommendation_status': 'revisit',
                                'previous_recommended_at': '2026-09-29'}),
            ):
                record = {'title': 'Recurring paper', 'doi': '10.1234/recurring', **metadata}
                (daily / (day + '.json')).write_text(json.dumps({'date': day, 'papers': [record]}))
            result = site_builder._rebuild_index(data)
            self.assertEqual(result['paper_count'], 1)
            paper = result['papers'][0]
            self.assertEqual(paper['recommendation_status'], 'revisit')
            dates = paper['recommendations_by_date']
            self.assertEqual(dates['2026-09-28'], {'recommendation_status': 'new',
                                                   'previous_recommended_at': None})
            self.assertEqual(dates['2026-09-29'], {'recommendation_status': 'repeat_highlight',
                                                   'previous_recommended_at': '2026-09-28'})
            self.assertEqual(dates['2026-10-07'], {'recommendation_status': 'revisit',
                                                   'previous_recommended_at': '2026-09-29'})


class EmailTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.email = load_isolated('construct_email', {
            'loguru': simple_module('loguru', logger=MagicMock()),
            'tqdm': simple_module('tqdm', tqdm=lambda values, **kwargs: values),
        })

    def block(self, **kwargs):
        return self.email.get_block_html(title='Title', authors='A. Author',
            rate='0.8', journal='Nature', source='Crossref', abstract='Abstract',
            paper_url='https://example.com/paper', **kwargs)

    def test_legacy_block_call_still_works(self):
        rendered = self.block()
        self.assertIn('Title', rendered)
        self.assertNotIn('Repeat highlight', rendered)

    def test_repeat_highlight_has_explicit_label_and_previous_date(self):
        rendered = self.block(recommendation_status='repeat_highlight', previous_recommended_at='2026-09-29')
        self.assertIn('Repeat highlight', rendered)
        self.assertIn('2026-09-29', rendered)
        self.assertIn('shortfall', rendered)

    def test_old_recommendation_is_labeled_revisit(self):
        rendered = self.block(recommendation_status='revisit', previous_recommended_at='2026-09-01')
        self.assertIn('Revisit', rendered)
        self.assertIn('2026-09-01', rendered)

    def test_previous_date_is_html_escaped(self):
        rendered = self.block(recommendation_status='repeat_highlight', previous_recommended_at='<script>alert(1)</script>')
        self.assertNotIn('<script>', rendered)

    def test_smtp_acceptance_survives_quit_disconnect(self):
        smtp = MagicMock()
        smtp.quit.side_effect = self.email.smtplib.SMTPServerDisconnected('QUIT disconnected')
        with patch.object(self.email.smtplib, 'SMTP', return_value=smtp), \
             patch.object(self.email.smtplib, 'SMTP_SSL') as fallback:
            self.email.send_email('sender@example.invalid', 'receiver@example.invalid',
                                  'fake-test-secret', 'smtp.example.invalid', 587, '<p>test</p>')
        smtp.sendmail.assert_called_once()
        smtp.close.assert_called_once()
        fallback.assert_not_called()

    def test_smtp_send_failure_propagates(self):
        smtp = MagicMock()
        smtp.sendmail.side_effect = self.email.smtplib.SMTPDataError(550, b'rejected')
        with patch.object(self.email.smtplib, 'SMTP', return_value=smtp), \
             patch.object(self.email.smtplib, 'SMTP_SSL') as fallback:
            with self.assertRaises(self.email.smtplib.SMTPDataError):
                self.email.send_email('sender@example.invalid', 'receiver@example.invalid',
                                      'fake-test-secret', 'smtp.example.invalid', 587, '<p>test</p>')
        smtp.quit.assert_not_called()
        fallback.assert_not_called()

    def test_render_email_passes_repeat_metadata(self):
        p = example_paper(recommendation_status='repeat_highlight', previous_recommended_at='2026-09-29')
        with patch.object(self.email.time, 'sleep'):
            rendered = self.email.render_email([p])
        self.assertIn('Repeat highlight', rendered)
        self.assertIn('2026-09-29', rendered)
        self.assertIn('shortfall', rendered)


class SourceWindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sources = load_isolated('sources', {
            'requests': simple_module('requests', Session=MagicMock()),
            'requests.adapters': simple_module('requests.adapters', HTTPAdapter=MagicMock()),
            'urllib3': simple_module('urllib3'),
            'urllib3.util': simple_module('urllib3.util'),
            'urllib3.util.retry': simple_module('urllib3.util.retry', Retry=MagicMock()),
            'loguru': simple_module('loguru', logger=MagicMock()),
            'paper': simple_module('paper', JournalPaper=SimpleNamespace),
        })

    def fetch(self, now, **kwargs):
        session = MagicMock()
        session.get.return_value.json.return_value = {'message': {'items': [], 'total-results': 0}}
        with patch.object(self.sources, 'load_journal_catalog',
                          return_value=[{'issn': '1234-5678', 'journal': 'Nature'}]), \
             patch.object(self.sources, '_crossref_session', return_value=session):
            self.sources.fetch_crossref_papers(now=now, **kwargs)
        return session.get.call_args.kwargs['params']

    def test_morning_run_widens_completed_arxiv_day_by_two_hours(self):
        # 22:00 UTC on Sep 30 is 18:00 EDT. The logical day is
        # Sep 28 20:00 -> Sep 29 20:00 ET; retrieval widens it to
        # Sep 28 18:00 -> Sep 29 22:00 ET.
        params = self.fetch(datetime(2026, 9, 30, 22, 0, tzinfo=timezone.utc))
        self.assertIn('from-created-date:2026-09-28T22:00:00', params['filter'])
        self.assertIn('until-created-date:2026-09-30T02:00:00', params['filter'])
        self.assertEqual(params['sort'], 'created')
        self.assertEqual(params['order'], 'desc')

    def test_same_arxiv_day_is_stable_across_reruns(self):
        morning = self.fetch(datetime(2026, 9, 30, 22, 0, tzinfo=timezone.utc))
        later = self.fetch(datetime(2026, 9, 30, 23, 30, tzinfo=timezone.utc))
        self.assertEqual(morning['filter'], later['filter'])

    def test_after_20_et_rolls_to_next_widened_day(self):
        params = self.fetch(datetime(2026, 10, 1, 1, 0, tzinfo=timezone.utc))
        self.assertIn('from-created-date:2026-09-29T22:00:00', params['filter'])
        self.assertIn('until-created-date:2026-10-01T02:00:00', params['filter'])

    def test_winter_boundary_tracks_eastern_time_not_fixed_utc(self):
        # 23:00 UTC is 18:00 EST. The completed logical day is widened
        # from 20:00->20:00 ET to 18:00->22:00 ET.
        params = self.fetch(datetime(2026, 1, 15, 23, 0, tzinfo=timezone.utc))
        self.assertIn('from-created-date:2026-01-13T23:00:00', params['filter'])
        self.assertIn('until-created-date:2026-01-15T03:00:00', params['filter'])

    def test_old_lookback_override_is_ignored_but_rows_remain_configurable(self):
        params = self.fetch(
            datetime(2026, 9, 30, 22, 0, tzinfo=timezone.utc),
            lookback_days=7,
            rows_per_journal=17,
        )
        self.assertIn('from-created-date:2026-09-28T22:00:00', params['filter'])
        self.assertLessEqual(params['rows'], 17)


    def test_consecutive_daily_windows_overlap_by_four_hours(self):
        first_start, first_end = self.sources.widened_crossref_window(
            datetime(2026, 9, 30, 22, 0, tzinfo=timezone.utc)
        )
        second_start, second_end = self.sources.widened_crossref_window(
            datetime(2026, 10, 1, 22, 0, tzinfo=timezone.utc)
        )
        self.assertEqual((first_end - second_start).total_seconds(), 4 * 3600)
        self.assertLess(first_start, second_start)
        self.assertLess(first_end, second_end)

    def test_negative_overlap_is_rejected(self):
        with self.assertRaises(ValueError):
            self.sources.widened_crossref_window(
                datetime(2026, 9, 30, 22, 0, tzinfo=timezone.utc),
                overlap_hours=-1,
            )

    def test_naive_clock_input_is_rejected(self):
        with self.assertRaises(ValueError):
            self.sources.latest_completed_arxiv_clock_window(datetime(2026, 9, 30, 22, 0))


if __name__ == '__main__':
    unittest.main()
