"""Offline checks for the easy control set and unchanged sampling pipeline."""

import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from mmengine import Config

from martingale_consistency_and_posteriors.datasets import DATASETS, LocalMCQADataset
from martingale_consistency_and_posteriors.datasets.gpqa import GPQADataset
from martingale_consistency_and_posteriors.utils.martingale_helpers import run_martingale_sampling_check
from martingale_consistency_and_posteriors.utils.model_calls import extract_choice_labels


ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = 'data/datasets/simple_mcqa.json'


class LocalMCQADatasetTests(unittest.TestCase):
    def test_questions_labels_and_existing_prompt_format(self):
        dataset = LocalMCQADataset(DATA_FILE)
        self.assertEqual(len(dataset), 5)
        self.assertEqual(dataset.get_true_labels(), [1, 3, 0, 1, 1])
        self.assertEqual(list(dataset.get_data_indices()), list(range(5)))
        self.assertEqual(dataset.preamble, GPQADataset.preamble)
        for i, correct in enumerate(['4', '7', '3', '60', 'Cat']):
            item = dataset[i]
            self.assertEqual(extract_choice_labels(item['prompt']), list('ABCD'))
            self.assertTrue(item['prompt'].endswith('\nAnswer:'))
            self.assertNotIn('answer_key', item['prompt'])
            self.assertNotIn('Your prior answers', item['prompt'])
            self.assertEqual(dataset.dataset[i]['choices'][item['label']], correct)
        self.assertEqual(dataset[0]['prompt'],
            'Return the label of the correct answer for the question below.\n\n'
            'Question: What is 2 + 2?\nChoices:\nA) 3\nB) 4\nC) 5\nD) 6\nAnswer:')

    def test_subset_paths_and_collation(self):
        dataset = LocalMCQADataset(str(ROOT / DATA_FILE), subset_size=100)
        self.assertEqual(len(dataset), 5)
        self.assertEqual(len(LocalMCQADataset(DATA_FILE, subset_size=2)), 2)
        batch = dataset.get_collate_fn()([dataset[0], dataset[1]])
        self.assertEqual(batch['labels'].tolist(), [1, 3])
        self.assertEqual(batch['prompts'], [dataset[0]['prompt'], dataset[1]['prompt']])

    def test_invalid_rows_fail_before_inference(self):
        good = {'question': 'Question?', 'choices': ['one', 'two', 'three', 'four'], 'answer_key': 'A'}
        bad_values = [[], [dict(good, question='')], [dict(good, answer_key='E')],
                      [dict(good, choices=['one'])], [dict(good, choices=['one'] * 4)]]
        for records in bad_values:
            with self.subTest(records=records):
                with patch('martingale_consistency_and_posteriors.datasets.local_mcqa.json.load', return_value=records):
                    with self.assertRaises(ValueError):
                        LocalMCQADataset(DATA_FILE)
        with self.assertRaises(ValueError):
            LocalMCQADataset(DATA_FILE, subset_size=-1)

    def test_all_three_configs_build_same_questions(self):
        reference = LocalMCQADataset(DATA_FILE)
        for model in ('gpt4o_mini', 'deepseek_v4_pro', 'qwen2_7b'):
            cfg = Config.fromfile(str(ROOT / 'configs' / f'simple_mcqa_{model}' / f'simple_mcqa_{model}.yaml'))
            self.assertEqual(set(cfg.data), {'test'})
            dataset = DATASETS.build(cfg.data.test, default_args={'tokenizer': None})
            self.assertEqual([dataset[i] for i in range(5)], [reference[i] for i in range(5)])

    def test_existing_sampling_runner_with_mock_provider(self):
        dataset = LocalMCQADataset(DATA_FILE)
        calls = []

        def get_probs(prompts):
            calls.extend(prompts)
            for prompt in prompts:
                self.assertEqual(extract_choice_labels(prompt), list('ABCD'))
            probs = np.tile([0.01, 0.97, 0.01, 0.01], (len(prompts), 1))
            return np.log(probs), probs

        result = run_martingale_sampling_check(
            get_probs=get_probs, n_classes=4, dataset=dataset,
            rng=np.random.default_rng(42), K=3, n_samples=5,
            batch_size=2, seed=42, label_chars=dataset.label_chars, J=2,
        )
        self.assertEqual(result['probs'].shape, (5, 2, 3, 4))
        self.assertEqual(result['branch_probs'].shape, (5, 2, 2, 4, 4))
        np.testing.assert_allclose(result['residuals'], 0, atol=1e-15)
        self.assertTrue(any('Your prior answers in previous steps to this specific question were:' in p for p in calls))
        for q, index in enumerate(result['data_indices']):
            self.assertEqual(result['true_labels'][q], dataset[int(index)]['label'])


if __name__ == '__main__':
    unittest.main()
