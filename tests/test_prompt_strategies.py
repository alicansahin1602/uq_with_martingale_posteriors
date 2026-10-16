"""Exhaustive built-in rendering checks; no network or models required."""

import hashlib
import itertools
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from mmengine import Config

from martingale_consistency_and_posteriors.utils.prompt_types import HistoryState, PromptContext
from martingale_consistency_and_posteriors.utils.prompt_strategies import (
    build_prompt_strategy, insert_legacy_history, register_prompt_strategy,
    PromptStrategy, STRATEGIES,
)
from martingale_consistency_and_posteriors.utils.system_prompt import mcqa_system_prompt
from tools.preview_prompt_strategies import (
    BASE_PROMPT, preview_configs, five_step_preview, export_previews,
)


class PromptStrategyTests(unittest.TestCase):
    def test_every_builtin_combination_at_all_five_steps(self):
        configs = preview_configs()
        self.assertEqual(len(configs), 19)
        for name, config in configs.items():
            with self.subTest(name=name):
                preview = five_step_preview(config)
                self.assertEqual(len(preview['steps']), 5)
                for t, step in enumerate(preview['steps']):
                    system, user = [x['content'] for x in step['messages']]
                    self.assertEqual(len(step['previous_probabilities']), t)
                    self.assertTrue(user.endswith('\nAnswer:'))
                    self.assertEqual(user.count('Question: What is 2 + 2?'), 1)
                    self.assertNotIn('answer_key', user)
                    if 'paragraphs' in name:
                        self.assertLess(user.index('The following block'), user.index('Consider all available'))
                        self.assertIn('Previous sampled answers are not correctness feedback.', system)
                    if config.get('treatment') == 'sampling_aware':
                        self.assertIn('repetition alone does not resolve', system)
                    if t == 0 and config.get('treatment') not in (None, 'current'):
                        self.assertIn('No previous sampling steps.', user)
                final = preview['steps'][-1]['messages'][1]['content']
                fmt = config.get('history_format', 'labels')
                if 'mean_probs' in fmt:
                    for row in ('A | 1 | 0.162500', 'B | 2 | 0.612500',
                                'C | 1 | 0.150000', 'D | 0 | 0.075000'):
                        self.assertIn(row, final)
                elif fmt == 'counts':
                    self.assertIn('A | 1\nB | 2\nC | 1\nD | 0', final)
                    self.assertNotIn('Mean previous probability', final)
                else:
                    self.assertIn('B, A, B, C', final)
                if fmt == 'labels_counts_mean_probs':
                    self.assertIn('Ordered sampled labels: B, A, B, C', final)

    def test_legacy_is_byte_identical(self):
        self.assertEqual(hashlib.sha256(mcqa_system_prompt(4).encode()).hexdigest(),
                         '7340def6324ee540118c8cc1d7fc8d12b54c5ae09094819301ee9178dfd00521')
        for step in five_step_preview({})['steps']:
            expected = BASE_PROMPT
            if step['step']:
                expected = (BASE_PROMPT[:-len('\nAnswer:')]
                            + '\nYour prior answers in previous steps to this specific question were: '
                            + ', '.join(step['sampled_labels']) + '\nAnswer:')
            self.assertEqual(step['messages'][1]['content'], expected)
            self.assertEqual(step['messages'][0]['content'], mcqa_system_prompt(4))
        current = five_step_preview({'strategy': 'configurable', 'treatment': 'current'})
        legacy = five_step_preview({})
        self.assertEqual(current['steps'], legacy['steps'])

    def test_immutable_history_full_precision_and_zero_counts(self):
        vector = np.array([0.123456789, 0.7, 0.1, 0.076543211])
        parent = HistoryState()
        a = parent.advance('A', vector)
        b = parent.advance('B', vector)
        vector[0] = 0
        self.assertEqual(parent.n_steps, 0)
        self.assertEqual(a.sampled_labels, ('A',))
        self.assertEqual(b.sampled_labels, ('B',))
        self.assertEqual(a.previous_probabilities[0][0], 0.123456789)
        strategy = build_prompt_strategy({'history_format': 'counts_mean_probs', 'probability_decimals': 3})
        prompt = strategy.render(PromptContext(BASE_PROMPT, tuple('ABCD'), a))
        self.assertIn('A | 1 | 0.123', prompt.user_text)
        self.assertIn('C | 0 | 0.100', prompt.user_text)
        self.assertEqual(a.previous_probabilities[0][0], 0.123456789)

    def test_paragraphs_are_literal_and_ordered(self):
        strategy = build_prompt_strategy({
            'strategy': 'configurable', 'history_format': 'counts',
            'extra_system_paragraphs': ['FIRST {x}', 'SECOND 50%'],
            'extra_user_paragraphs': {'before_history': ['BEFORE'], 'after_history': ['E) AFTER']},
        })
        prompt = strategy.render(PromptContext(BASE_PROMPT, tuple('ABCD')))
        self.assertLess(prompt.system_text.index('FIRST {x}'), prompt.system_text.index('SECOND 50%'))
        self.assertLess(prompt.system_text.index('SECOND 50%'), prompt.system_text.index('RESPONSE FORMAT'))
        self.assertLess(prompt.user_text.index('BEFORE'), prompt.user_text.index('HISTORY'))
        self.assertLess(prompt.user_text.index('HISTORY'), prompt.user_text.index('E) AFTER'))

    def test_each_paragraph_position_can_be_enabled_independently(self):
        for system, before, after in itertools.product((False, True), repeat=3):
            with self.subTest(system=system, before=before, after=after):
                strategy = build_prompt_strategy({
                    'history_format': 'counts_mean_probs',
                    'extra_system_paragraphs': ['SYSTEM EXTRA'] if system else [],
                    'extra_user_paragraphs': {
                        'before_history': ['BEFORE EXTRA'] if before else [],
                        'after_history': ['AFTER EXTRA'] if after else [],
                    },
                })
                rendered = strategy.render(PromptContext(BASE_PROMPT, tuple('ABCD')))
                self.assertEqual('SYSTEM EXTRA' in rendered.system_text, system)
                self.assertEqual('BEFORE EXTRA' in rendered.user_text, before)
                self.assertEqual('AFTER EXTRA' in rendered.user_text, after)
                self.assertTrue(rendered.user_text.endswith('\nAnswer:'))

    def test_invalid_configuration_and_history(self):
        invalid = [[], {'typo': True}, {'strategy': 'unknown'}, {'history_format': 'unknown'},
                   {'treatment': 'unknown'}, {'probability_decimals': -1}, {'probability_decimals': True},
                   {'extra_system_paragraphs': 'not a list'}, {'extra_system_paragraphs': ['']},
                   {'extra_user_paragraphs': {'wrong_position': []}},
                   {'strategy': 'legacy', 'extra_system_paragraphs': ['Change']},
                   {'history_format': 'counts', 'treatment': 'current'}, {'params': {'unused': 1}}]
        for config in invalid:
            with self.subTest(config=config), self.assertRaises(ValueError):
                build_prompt_strategy(config)
        for vector in ([0.1, 0.2], [1.1, -0.1], [float('nan'), 0.5]):
            with self.assertRaises(ValueError):
                HistoryState().advance('A', vector)
        with self.assertRaises(ValueError):
            insert_legacy_history('Missing suffix', ['A'])

    def test_fingerprints_and_config_files(self):
        a = build_prompt_strategy({'history_format': 'counts'})
        b = build_prompt_strategy({'extra_system_paragraphs': ['Changed'], 'history_format': 'counts'})
        self.assertNotEqual(a.fingerprint, b.fingerprint)
        self.assertEqual(a.fingerprint, build_prompt_strategy(a.config).fingerprint)
        root = Path(__file__).resolve().parents[1]
        for path in (root / 'configs/prompts').glob('*.yaml'):
            with self.subTest(path=path):
                preset = build_prompt_strategy(Config.fromfile(str(path)).prompt_cfg)
                # Every runnable preset has an exact counterpart in the exported examples.
                fingerprints = {build_prompt_strategy(c).fingerprint for c in preview_configs().values()}
                self.assertIn(preset.fingerprint, fingerprints)

    def test_custom_registry_extension(self):
        @register_prompt_strategy('test_recent')
        class Recent(PromptStrategy):
            """Example extension delegates rendering after selecting a history window."""
            def render(self, context):
                size = self.config['params']['window']
                recent = HistoryState(context.history.sampled_labels[-size:],
                                      context.history.previous_probabilities[-size:])
                return build_prompt_strategy({'history_format': 'counts'}).render(
                    PromptContext(context.base_user_prompt, context.valid_labels, recent))
        try:
            history = HistoryState().advance('A', [0.25] * 4).advance('B', [0.25] * 4)
            strategy = build_prompt_strategy({'strategy': 'test_recent', 'params': {'window': 1}})
            prompt = strategy.render(PromptContext(BASE_PROMPT, tuple('ABCD'), history))
            self.assertIn('Completed sampling steps: 1', prompt.user_text)
            self.assertIn('A | 0', prompt.user_text)
        finally:
            STRATEGIES.pop('test_recent')

    def test_preview_export_roundtrip_and_overwrite_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            count = export_previews(directory, preview_configs())
            self.assertEqual(count, 19)
            for name, config in preview_configs().items():
                saved = json.loads((Path(directory) / f'{name}.json').read_text())
                self.assertEqual(saved, json.loads(json.dumps(five_step_preview(config))))
            with self.assertRaises(FileExistsError):
                export_previews(directory, preview_configs())


if __name__ == '__main__':
    unittest.main()
