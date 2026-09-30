"""Offline regression tests for delivered-paper identity and repeat cooldowns.

Run with: python -m unittest discover -s tests -v
"""
import json
import tempfile
import unittest
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from recommendation_history import (
    load_history, paper_keys, save_history, select_recommendations,
)


TODAY = date(2026, 9, 30)


def paper(title, **kwargs):
    return SimpleNamespace(title=title, **kwargs)


def previous(p, day):
    return {key: day for key in paper_keys(p)}


class IdentityTests(unittest.TestCase):
    def test_doi_urls_prefixes_case_and_whitespace_are_same_identity(self):
        variants = ['10.1234/ABC.def', ' 10.1234/abc.DEF ',
                    'https://doi.org/10.1234/ABC.def',
                    'http://dx.doi.org/10.1234/abc.def',
                    'doi:10.1234/abc.def']
        keys = [paper_keys({'doi': value}) for value in variants]
        self.assertTrue(keys[0])
        for candidate in keys[1:]:
            self.assertEqual(keys[0], candidate)

    def test_arxiv_versions_and_urls_share_identity(self):
        variants = ['2609.12345', '2609.12345v1', '2609.12345v12',
                    'https://arxiv.org/abs/2609.12345v2',
                    'https://arxiv.org/pdf/2609.12345v3.pdf']
        keys = [paper_keys({'arxiv_id': value}) for value in variants]
        self.assertTrue(keys[0])
        for candidate in keys[1:]:
            self.assertEqual(keys[0], candidate)

    def test_old_style_arxiv_versions_share_identity(self):
        self.assertEqual(paper_keys({'arxiv_id': 'cond-mat/9901234v2'}),
                         paper_keys({'arxiv_id': 'cond-mat/9901234'}))

    def test_title_is_unicode_aware_and_ignores_punctuation(self):
        self.assertEqual(paper_keys({'title': 'Battery: α-phase, 纳米材料!'}),
                         paper_keys({'title': 'battery α phase 纳米材料'}))
        self.assertNotEqual(paper_keys({'title': '纳米材料'}),
                            paper_keys({'title': '半导体'}))
        self.assertEqual(paper_keys({'title': ' \t!!! '}), set())

    def test_object_and_archive_dict_have_same_keys(self):
        values = dict(title='A useful paper', doi='10.1234/example', arxiv_id='2609.12345v2')
        self.assertEqual(paper_keys(values), paper_keys(SimpleNamespace(**values)))
        self.assertGreaterEqual(len(paper_keys(values)), 3)

    def test_title_matches_between_preprint_and_published_article(self):
        preprint = paper('Same title!', arxiv_id='2609.12345v1')
        journal = paper('Same title', doi='10.1234/journal')
        self.assertTrue(paper_keys(preprint) & paper_keys(journal))


class SelectionTests(unittest.TestCase):
    def test_new_papers_take_priority_then_recent_fallback_in_original_rank(self):
        recent_high, new_low, recent_low = map(paper, ['High repeat', 'New', 'Low repeat'])
        history = {**previous(recent_high, '2026-09-29'),
                   **previous(recent_low, '2026-09-28')}
        selected = select_recommendations([recent_high, new_low, recent_low], history, 2, today=TODAY)
        self.assertEqual(selected, [new_low, recent_high])
        self.assertEqual(new_low.recommendation_status, 'new')
        self.assertIsNone(new_low.previous_recommended_at)
        self.assertEqual(recent_high.recommendation_status, 'repeat_highlight')
        self.assertEqual(recent_high.previous_recommended_at, '2026-09-29')

    def test_exact_seven_day_boundary_is_eligible_but_six_days_is_cooldown(self):
        recent, boundary, unseen = map(paper, ['Six days', 'Seven days', 'Never shown'])
        history = {**previous(recent, '2026-09-24'), **previous(boundary, '2026-09-23')}
        selected = select_recommendations([recent, boundary, unseen], history, 2, cooldown_days=7, today=TODAY)
        self.assertEqual(selected, [boundary, unseen])
        self.assertEqual(boundary.recommendation_status, 'revisit')
        self.assertEqual(boundary.previous_recommended_at, '2026-09-23')

    def test_same_day_rerun_is_recent_even_after_doi_or_version_change(self):
        old = paper('An important finding', arxiv_id='2609.12345v1')
        rerun = paper('An important finding', doi='https://doi.org/10.1234/finding', arxiv_id='2609.12345v2')
        fresh = paper('Unseen finding')
        selected = select_recommendations([rerun, fresh], previous(old, TODAY.isoformat()), 2, today=TODAY)
        self.assertEqual(selected, [fresh, rerun])
        self.assertEqual(rerun.recommendation_status, 'repeat_highlight')
        self.assertEqual(rerun.previous_recommended_at, TODAY.isoformat())

    def test_latest_matching_alias_controls_cooldown(self):
        candidate = paper('Two aliases', doi='10.1234/aliases')
        history = {key: '2026-09-01' for key in paper_keys(candidate)}
        history.update(previous(paper('', doi='10.1234/aliases'), '2026-09-29'))
        selected = select_recommendations([candidate], history, 1, today=TODAY)
        self.assertEqual(selected[0].previous_recommended_at, '2026-09-29')
        self.assertEqual(selected[0].recommendation_status, 'repeat_highlight')

    def test_empty_input_and_zero_limit(self):
        self.assertEqual(select_recommendations([], {}, 10, today=TODAY), [])
        self.assertEqual(select_recommendations([paper('Not selected')], {}, 0, today=TODAY), [])

    def test_zero_cooldown_preserves_rank_and_marks_known_papers_revisit(self):
        known, fresh = paper('Known'), paper('Fresh')
        selected = select_recommendations([known, fresh], previous(known, TODAY.isoformat()),
                                          1, cooldown_days=0, today=TODAY)
        self.assertEqual(selected, [known])
        self.assertEqual(known.recommendation_status, 'revisit')

    def test_unlimited_selection_includes_all_candidates_in_tier_order(self):
        known, fresh = paper('Known'), paper('Fresh')
        selected = select_recommendations([known, fresh], previous(known, TODAY.isoformat()),
                                          -1, today=TODAY)
        self.assertEqual(selected, [fresh, known])

    def test_invalid_limits_fail_explicitly(self):
        with self.assertRaises(ValueError):
            select_recommendations([], {}, -2, today=TODAY)
        with self.assertRaises(ValueError):
            select_recommendations([], {}, 1, cooldown_days=-1, today=TODAY)

    def test_selection_does_not_mutate_history_or_ranking(self):
        ranked = [paper('Repeat'), paper('Fresh')]
        original_ranked = list(ranked)
        history = previous(ranked[0], '2026-09-29')
        original_history = dict(history)
        select_recommendations(ranked, history, 1, today=TODAY)
        self.assertEqual(history, original_history)
        self.assertEqual(ranked, original_ranked)


class PersistenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.daily = self.root / 'data' / 'daily'
        self.daily.mkdir(parents=True)

    def archive(self, filename, payload):
        (self.daily / filename).write_text(json.dumps(payload), encoding='utf-8')

    def test_bootstraps_legacy_archives_and_uses_latest_day(self):
        old = {'title': 'Already recommended', 'doi': '10.1234/already'}
        self.archive('2026-09-26.json', {'date': '2026-09-26', 'papers': [old]})
        self.archive('2026-09-29.json', {'date': '2026-09-29', 'papers': [old]})
        history = load_history(str(self.root))
        for key in paper_keys(old):
            self.assertEqual(history[key], '2026-09-29')

    def test_legacy_archive_date_falls_back_to_filename(self):
        old = {'title': 'Legacy no date'}
        self.archive('2026-09-28.json', {'papers': [old]})
        history = load_history(str(self.root))
        self.assertEqual(history[next(iter(paper_keys(old)))], '2026-09-28')

    def test_invalid_archive_dates_and_broken_json_do_not_poison_history(self):
        self.archive('2026-09-28.json', {'date': '2026-09-28', 'papers': [{'title': 'Valid'}]})
        self.archive('bad.json', {'date': '2026-02-30', 'papers': [{'title': 'Invalid'}]})
        (self.daily / 'broken.json').write_text('{', encoding='utf-8')
        history = load_history(str(self.root))
        self.assertTrue(paper_keys({'title': 'Valid'}) <= history.keys())
        self.assertFalse(paper_keys({'title': 'Invalid'}) & history.keys())

    def test_only_selected_papers_are_saved_and_same_day_saves_merge(self):
        first, second, unselected = map(paper, ['First', 'Second', 'Candidate only'])
        selected = select_recommendations([first, unselected], {}, 1, today=TODAY)
        save_history(selected, str(self.root), today=TODAY)
        save_history([second], str(self.root), today=TODAY)
        history = load_history(str(self.root))
        self.assertTrue((paper_keys(first) | paper_keys(second)) <= history.keys())
        self.assertFalse(paper_keys(unselected) & history.keys())
        self.assertTrue(all(value == TODAY.isoformat() for value in history.values()))

    def test_saved_history_survives_archive_replacement(self):
        legacy, fresh = paper('From old archive'), paper('Fresh recommendation')
        self.archive('2026-09-30.json', {'date': '2026-09-30', 'papers': [vars(legacy)]})
        save_history([fresh], str(self.root), today=TODAY)
        self.archive('2026-09-30.json', {'date': '2026-09-30', 'papers': [vars(fresh)]})
        history = load_history(str(self.root))
        self.assertTrue(paper_keys(legacy) <= history.keys())
        self.assertTrue(paper_keys(fresh) <= history.keys())

    def test_ledger_schema_and_invalid_values_fail_closed(self):
        ledger = self.root / 'data' / 'recommendation-history.json'
        for payload in ({'version': 2, 'entries': {}},
                        {'version': 1, 'entries': []},
                        {'version': 1, 'entries': {'title:invalid': '2026-02-30'}},
                        {'version': 1, 'entries': {'title:invalid': '20260930'}}):
            with self.subTest(payload=payload):
                ledger.write_text(json.dumps(payload), encoding='utf-8')
                with self.assertRaises(ValueError):
                    load_history(str(self.root))
        ledger.write_text('{', encoding='utf-8')
        with self.assertRaises(ValueError):
            load_history(str(self.root))

    def test_atomic_save_failure_preserves_previous_ledger_and_cleans_temp(self):
        old, fresh = paper('Previously saved'), paper('New save fails')
        ledger = save_history([old], str(self.root), today=TODAY)
        before = ledger.read_bytes()
        with patch('recommendation_history.os.replace', side_effect=OSError('simulated disk failure')):
            with self.assertRaises(OSError):
                save_history([fresh], str(self.root), today=TODAY)
        self.assertEqual(ledger.read_bytes(), before)
        self.assertEqual(list(ledger.parent.glob('.history-*.tmp')), [])
        self.assertFalse(paper_keys(fresh) & load_history(str(self.root)).keys())

    def test_missing_history_is_empty(self):
        self.assertEqual(load_history(str(self.root)), {})


if __name__ == '__main__':
    unittest.main()
