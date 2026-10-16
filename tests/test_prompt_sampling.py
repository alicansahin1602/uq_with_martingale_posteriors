"""Offline regression and branching checks for every built-in prompt option."""

import hashlib
import json
import logging
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from martingale_consistency_and_posteriors.utils.martingale_helpers import (
    run_martingale_sampling_check, run_martingale_check, save_martingale_sampling_results,
)
from martingale_consistency_and_posteriors.utils.prompt_types import HistoryState, PromptContext, RenderedPrompt
from martingale_consistency_and_posteriors.utils.prompt_strategies import build_prompt_strategy
from tools.preview_prompt_strategies import preview_configs
from tools import martingale_property as runner


class OneQuestion:
    """Small dataset with no external files or correctness information in prompts."""
    prompt = 'Question: What is 2 + 2?\nChoices:\nA) 3\nB) 4\nC) 5\nD) 6\nAnswer:'

    def __len__(self):
        return 1

    def __getitem__(self, index):
        return {'prompt': self.prompt, 'label': 1}


def deterministic_provider(prompts):
    """Prompt-dependent vectors make changes to history observable in regression tests."""
    texts = [p.user_text if isinstance(p, RenderedPrompt) else p for p in prompts]
    probs = np.array([[1 + v for v in hashlib.sha256(s.encode()).digest()[:4]] for s in texts], dtype=float)
    probs /= probs.sum(axis=1, keepdims=True)
    return np.log(probs), probs


def run(strategy=None, **kwargs):
    settings = dict(get_probs=deterministic_provider, n_classes=4, dataset=OneQuestion(),
                    rng=np.random.default_rng(42), K=5, J=2, seed=42, prompt_strategy=strategy)
    settings.update(kwargs)
    return run_martingale_sampling_check(**settings)


class PromptSamplingTests(unittest.TestCase):
    def test_legacy_matches_prechange_recorded_outputs(self):
        # Captured from the untouched branch before implementing the prompt layer.
        hashes = {
            'probs': '5aa62b285fa7e3ab65051ee913e4a705e142ea3cde3a3d66d44258dc4200ece3',
            'branch_probs': '1d1f593f5aa288a027f16a13828db590e80318b6946e9034cd3a876add812019',
            'residuals': '343699389c3f44523a9de7c073cd6d53a45ee9c68894f50b85d498957995adda',
            'labels': 'e049854b3de43bb2f7051b7da0427d7c583bc017c4d44d0cb4ca30a38e0666c1',
            'prompt_history': '32e795edd4e6e049f701d2c6f2abe4b606aa441f1cbaab059e4fb924e8bdde1a',
            'branch_prompt_history': '4c41669b8277a25898eea81116f5a200a22815e9993930db72b636eb5e3182b5',
        }
        result = run()
        for name, expected in hashes.items():
            values = result[name]
            data = json.dumps(values.tolist(), ensure_ascii=False).encode() if values.dtype == object else values.tobytes()
            self.assertEqual(hashlib.sha256(data).hexdigest(), expected, name)

    def test_all_strategies_branch_history_and_cached_continuation(self):
        for name, config in preview_configs().items():
            with self.subTest(name=name):
                strategy = build_prompt_strategy(config)
                requests = []

                def provider(prompts):
                    requests.extend(prompts)
                    return deterministic_provider(prompts)

                result = run(strategy, get_probs=provider, batch_size=3)
                self.assertEqual(len(requests), 1 + 2 * 4 * 4)
                self.assertEqual(result['probs'].shape, (1, 2, 5, 4))
                expected = np.einsum('...y,...yc->...c', result['probs'][..., :-1, :], result['branch_probs'])
                np.testing.assert_allclose(result['residuals'], expected - result['probs'][..., :-1, :])
                for j in range(2):
                    history = HistoryState()
                    for t in range(5):
                        context = PromptContext(OneQuestion.prompt, tuple('ABCD'), history, tuple('ABCD'))
                        rendered = strategy.render(context)
                        self.assertEqual(result['prompt_history'][0, j, t], rendered.user_text)
                        self.assertEqual(result['system_prompt_history'][0, j, t], rendered.system_text)
                        if t == 4:
                            break
                        parent = result['probs'][0, j, t]
                        for y, label in enumerate('ABCD'):
                            child = history.advance(label, parent)
                            expected_prompt = strategy.render(PromptContext(
                                OneQuestion.prompt, tuple('ABCD'), child, tuple('ABCD')))
                            self.assertEqual(result['branch_prompt_history'][0, j, t, y], expected_prompt.user_text)
                            self.assertEqual(result['branch_system_prompt_history'][0, j, t, y], expected_prompt.system_text)
                            self.assertEqual(child.n_steps, t + 1)
                        selected = result['labels'][0, j, t]
                        np.testing.assert_array_equal(result['probs'][0, j, t + 1], result['branch_probs'][0, j, t, selected])
                        history = history.advance('ABCD'[selected], parent)
                self.assertEqual(result['metadata']['prompt_fingerprint'], strategy.fingerprint)

    def test_absent_classes_are_not_displayed_or_sampled(self):
        calls = []
        def provider(prompts):
            calls.extend(prompts)
            probs = np.tile([0.1, 0.6, 0.2, 0.1, 0.0], (len(prompts), 1))
            return np.log(np.clip(probs, 1e-30, None)), probs
        result = run(build_prompt_strategy({'history_format': 'counts_mean_probs'}),
                     get_probs=provider, n_classes=5, J=1)
        self.assertEqual(len(calls), 17)
        self.assertFalse(result['valid_class_mask'][0, 4])
        self.assertTrue((result['branch_prompt_history'][..., 4] == '').all())
        self.assertTrue((result['branch_probs'][..., 4, :] == 0).all())
        self.assertTrue((result['labels'] < 4).all())
        for request in calls:
            self.assertNotIn('\nE |', request.user_text)

    def test_one_step_and_save_roundtrip(self):
        result = run(build_prompt_strategy({'history_format': 'counts'}), K=1)
        self.assertEqual(result['branch_probs'].shape, (1, 2, 0, 4, 4))
        with tempfile.TemporaryDirectory() as directory:
            path = save_martingale_sampling_results(directory, result)
            with np.load(path, allow_pickle=True) as saved:
                self.assertEqual(saved['metadata'].item(), result['metadata'])
                np.testing.assert_array_equal(saved['system_prompt_history'], result['system_prompt_history'])
                np.testing.assert_array_equal(saved['prompt_history'], result['prompt_history'])

    def test_iterative_legacy_still_accepts_strings(self):
        calls = []
        def provider(prompts):
            self.assertTrue(all(isinstance(p, str) for p in prompts))
            calls.extend(prompts)
            return deterministic_provider(prompts)[1]
        result = run_martingale_check(provider, 4, OneQuestion(), 4, 1, 1,
                                      np.random.default_rng(42), logging.getLogger('test'), list('ABCD'), J=2)
        self.assertEqual(result['distributions'].shape, (1, 2, 5, 4))
        self.assertIn('Your prior answers in previous steps', calls[-1])

    def test_output_paths_separate_prompt_text_and_repeat_runs(self):
        legacy = runner._build_work_dir('results', 'configs/model/run.yaml', 'sampling')
        self.assertEqual(legacy, 'results/model/martingale_check/sampling')
        a = build_prompt_strategy({'history_format': 'counts'})
        b = build_prompt_strategy({'history_format': 'counts', 'extra_system_paragraphs': ['Changed']})
        path_a = runner._build_work_dir('results', 'configs/model/run.yaml', 'sampling', a, 'run1')
        path_b = runner._build_work_dir('results', 'configs/model/run.yaml', 'sampling', b, 'run1')
        self.assertNotEqual(path_a, path_b)
        self.assertNotEqual(path_a, runner._build_work_dir('results', 'configs/model/run.yaml', 'sampling', a, 'run2'))

    def test_cli_rejects_new_iterative_before_loading(self):
        root = Path(__file__).resolve().parents[1]
        args = ['runner', '--config', str(root / 'configs/simple_mcqa_gpt4o_mini/simple_mcqa_gpt4o_mini.yaml'),
                '--work-dir', '/tmp/unused-prompt-test', '--mode', 'iterative',
                '--prompt-config', str(root / 'configs/prompts/counts.yaml')]
        with patch('sys.argv', args), patch.object(runner.DATASETS, 'build') as load:
            with self.assertRaisesRegex(ValueError, 'require --mode sampling'):
                runner.main()
            load.assert_not_called()

    def test_cli_protects_existing_legacy_sampling_before_loading(self):
        root = Path(__file__).resolve().parents[1]
        config = str(root / 'configs/simple_mcqa_gpt4o_mini/simple_mcqa_gpt4o_mini.yaml')
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(runner._build_work_dir(directory, config, 'sampling'))
            destination.mkdir(parents=True)
            np.savez(destination / 'martingale_sampling_results.npz', sentinel=np.array([42]))
            args = ['runner', '--config', config, '--work-dir', directory, '--mode', 'sampling']
            with patch('sys.argv', args), patch.object(runner.DATASETS, 'build') as load:
                with self.assertRaises(FileExistsError):
                    runner.main()
                load.assert_not_called()
            with np.load(destination / 'martingale_sampling_results.npz') as saved:
                self.assertEqual(saved['sentinel'][0], 42)

    def test_cli_end_to_end_with_mock_inference(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            args = ['runner', '--config', str(root / 'configs/simple_mcqa_gpt4o_mini/simple_mcqa_gpt4o_mini.yaml'),
                    '--work-dir', directory, '--mode', 'sampling', '--K', '5', '--J', '1', '--n-samples', '1',
                    '--prompt-config', str(root / 'configs/prompts/custom_paragraphs.yaml')]
            with patch('sys.argv', args), patch.object(runner, '_build_openai_martingale_sampling_provider',
                                                       return_value=deterministic_provider):
                runner.main()
            paths = list(Path(directory).rglob('martingale_sampling_results.npz'))
            self.assertEqual(len(paths), 1)
            with np.load(paths[0], allow_pickle=True) as result:
                self.assertEqual(result['probs'].shape, (1, 1, 5, 4))
                self.assertIn('Consider all available answer choices', result['prompt_history'][0, 0, 0])


if __name__ == '__main__':
    unittest.main()
