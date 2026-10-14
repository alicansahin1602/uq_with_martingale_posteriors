"""Offline GPQA adapter/config checks using synthetic rows, never benchmark data."""

import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from datasets import Dataset
from mmengine import Config

from martingale_consistency_and_posteriors.datasets import DATASETS, GPQADataset
from martingale_consistency_and_posteriors.utils.martingale_helpers import (
    run_martingale_sampling_check,
)
from martingale_consistency_and_posteriors.utils.model_calls import extract_choice_labels


ROOT = Path(__file__).resolve().parents[1]
LOADER = 'martingale_consistency_and_posteriors.datasets.gpqa.load_dataset'


def synthetic_rows(count=32):
    return Dataset.from_list([
        {
            'Question': f'Synthetic question {i}?',
            'Correct Answer': f'Correct option {i}',
            'Incorrect Answer 1': f'First distractor {i}',
            'Incorrect Answer 2': f'Second distractor {i}',
            'Incorrect Answer 3': f'Third distractor {i}',
            'Explanation': 'PRIVATE ANNOTATION NOT FOR THE MODEL',
        }
        for i in range(count)
    ])


class GPQADatasetTests(unittest.TestCase):
    def build(self, **kwargs):
        with patch(LOADER, return_value=synthetic_rows()):
            return GPQADataset(dataset_cfg={}, **kwargs)

    def test_choices_and_gold_labels_are_aligned(self):
        dataset = self.build()
        self.assertEqual(dataset.n_labels, 4)
        self.assertEqual(dataset.label_chars, list('ABCD'))
        self.assertEqual(list(dataset.get_data_indices()), list(range(32)))
        self.assertEqual(len(dataset.get_input_text()), 32)
        self.assertEqual(dataset.get_true_labels(), [dataset[i]['label'] for i in range(32)])
        for i in range(len(dataset)):
            item = dataset[i]
            lines = item['prompt'].split('Choices:\n')[1].split('\nAnswer:')[0].splitlines()
            self.assertEqual(len(lines), 4)
            self.assertEqual(extract_choice_labels(item['prompt']), list('ABCD'))
            self.assertEqual(lines[item['label']][3:], f'Correct option {i}')
            self.assertEqual({line[3:] for line in lines}, {
                f'Correct option {i}', f'First distractor {i}',
                f'Second distractor {i}', f'Third distractor {i}',
            })
            self.assertTrue(item['prompt'].endswith('\nAnswer:'))
            self.assertNotIn('PRIVATE ANNOTATION', item['prompt'])
        self.assertEqual(set(dataset.get_true_labels()), {0, 1, 2, 3})

    def test_order_is_stable_across_access_rebuilds_and_subsets(self):
        first = self.build(shuffle_seed=73)
        rebuilt = self.build(shuffle_seed=73)
        subset = self.build(shuffle_seed=73, subset_size=8)
        different_seed = self.build(shuffle_seed=74)
        for i in reversed(range(8)):
            self.assertEqual(first[i], rebuilt[i])
            self.assertEqual(first[i], subset[i])
            self.assertEqual(first[i], first[i])
        self.assertTrue(any(first[i] != different_seed[i] for i in range(32)))

    def test_defaults_subset_and_collation(self):
        cfg = {}
        with patch(LOADER, return_value=synthetic_rows(3)) as loader:
            dataset = GPQADataset(dataset_cfg=cfg, subset_size=100)
        loader.assert_called_once_with('Idavidrein/gpqa', name='gpqa_main', split='train')
        self.assertEqual(cfg, {})
        self.assertEqual(len(dataset), 3)
        batch = dataset.get_collate_fn()([dataset[0], dataset[1]])
        self.assertEqual(batch['prompts'], [dataset[0]['prompt'], dataset[1]['prompt']])
        self.assertEqual(batch['labels'].tolist(), [dataset[0]['label'], dataset[1]['label']])

    def test_invalid_input_fails_before_provider_calls(self):
        for kwargs in ({'shuffle_seed': -1}, {'subset_size': -1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.build(**kwargs)
        missing = synthetic_rows(1).remove_columns('Incorrect Answer 3')
        with patch(LOADER, return_value=missing), self.assertRaisesRegex(ValueError, 'missing required'):
            GPQADataset(dataset_cfg={})
        blank_row = dict(synthetic_rows(1)[0], Question=' ')
        with patch(LOADER, return_value=Dataset.from_list([blank_row])):
            with self.assertRaisesRegex(ValueError, 'non-empty text'):
                GPQADataset(dataset_cfg={})

    def test_all_model_configs_build_registered_dataset(self):
        for model in ('gpt4o_mini', 'deepseek_v4_pro', 'qwen2_7b'):
            with self.subTest(model=model):
                cfg = Config.fromfile(str(ROOT / 'configs' / f'gpqa_{model}' / f'gpqa_{model}.yaml'))
                self.assertEqual(set(cfg.data), {'test'})
                self.assertEqual(cfg.process_preds.npz_file, 'gpqa.npz')
                with patch(LOADER, return_value=synthetic_rows(2)) as loader:
                    dataset = DATASETS.build(cfg.data.test, default_args={'tokenizer': None})
                self.assertIsInstance(dataset, GPQADataset)
                loader.assert_called_once_with('Idavidrein/gpqa', name='gpqa_main', split='train')
                cfg.merge_from_dict({'data.test.dataset_cfg.name': 'gpqa_diamond'})
                with patch(LOADER, return_value=synthetic_rows(2)) as loader:
                    DATASETS.build(cfg.data.test, default_args={'tokenizer': None})
                self.assertEqual(loader.call_args.kwargs['name'], 'gpqa_diamond')

    def test_adapter_feeds_existing_branching_runner(self):
        dataset = self.build(subset_size=3)
        calls = []

        def get_probs(prompts):
            calls.extend(prompts)
            for prompt in prompts:
                self.assertEqual(extract_choice_labels(prompt), list('ABCD'))
                self.assertTrue(prompt.endswith('\nAnswer:'))
            probs = np.tile([0.1, 0.2, 0.3, 0.4], (len(prompts), 1))
            return np.log(probs), probs

        result = run_martingale_sampling_check(
            get_probs=get_probs, n_classes=4, dataset=dataset,
            rng=np.random.default_rng(60), K=3, n_samples=3,
            batch_size=2, seed=60, label_chars=dataset.label_chars, J=2,
        )
        self.assertEqual(result['probs'].shape, (3, 2, 3, 4))
        self.assertEqual(result['branch_probs'].shape, (3, 2, 2, 4, 4))
        np.testing.assert_allclose(result['residuals'], 0, atol=1e-15)
        self.assertTrue(any('Your prior answers' in prompt for prompt in calls))
        for q, index in enumerate(result['data_indices']):
            self.assertEqual(result['true_labels'][q], dataset[int(index)]['label'])
            self.assertEqual(result['input_texts'][q], dataset[int(index)]['prompt'])


if __name__ == '__main__':
    unittest.main()
