from typing import Any, Callable, Dict, Optional

import numpy as np
from datasets import load_dataset
from transformers import PreTrainedTokenizer

from .builder import DATASETS
from .classification import ClassificationDataset
from .dataset_utils import qa_dataset_collate_fn


_ANSWER_COLUMNS = (
    'Correct Answer',
    'Incorrect Answer 1',
    'Incorrect Answer 2',
    'Incorrect Answer 3',
)


def _prepare_gpqa_sample(sample: Dict[str, Any], index: int, shuffle_seed: int) -> dict:
    """Combine answer columns once, preserving the correct answer's identity."""
    for column in ('Question', *_ANSWER_COLUMNS):
        value = sample[column]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f'GPQA row {index}: {column!r} must be non-empty text.')

    # Per-row seeds make ordering independent of traversal, batching and how
    # many other rows are prepared. Source column 0 is always the correct one.
    rng = np.random.default_rng(np.random.SeedSequence([shuffle_seed, index]))
    order = rng.permutation(4)
    return {
        'question': sample['Question'].strip(),
        'choices': {
            'label': ['A', 'B', 'C', 'D'],
            'text': [sample[_ANSWER_COLUMNS[int(i)]].strip() for i in order],
        },
        'label': int(np.flatnonzero(order == 0)[0]),
        'row_id': index,
    }


@DATASETS.register_module()
class GPQADataset(ClassificationDataset):
    """GPQA MCQA with fixed, reproducibly shuffled A-D choices.

    ``dataset_cfg`` is passed to Hugging Face's ``load_dataset``; ``name``
    defaults to ``gpqa_main`` and ``split`` to ``train`` (the upstream benchmark
    split, even when used for evaluation). Authentication uses the normal
    Hugging Face login/token mechanism. ``shuffle_seed`` controls option order
    separately from the experiment seed used to select questions/trajectories.
    """

    preamble = (
        'Return the label of the correct answer for the question below.\n'
        '\n'
        'Question: {question}\n'
        'Choices:\n'
        '{choices}\n'
        'Answer:'
    )

    def __init__(
        self,
        dataset_cfg: Dict[str, Any],
        tokenizer: Optional[PreTrainedTokenizer] = None,
        add_space: bool = True,
        subset_size: Optional[int] = None,
        noisy_level: Optional[float] = None,
        shuffle_seed: int = 42,
    ) -> None:
        if shuffle_seed < 0:
            raise ValueError('shuffle_seed must be non-negative.')
        if subset_size is not None and subset_size < 0:
            raise ValueError('subset_size must be non-negative or None.')

        load_cfg = dict(dataset_cfg)
        load_cfg.setdefault('name', 'gpqa_main')
        load_cfg.setdefault('split', 'train')
        data_set = load_dataset('Idavidrein/gpqa', **load_cfg)
        missing = set(('Question', *_ANSWER_COLUMNS)) - set(data_set.column_names)
        if missing:
            raise ValueError(f'GPQA is missing required columns: {sorted(missing)}')
        if subset_size is not None:
            data_set = data_set.select(range(min(subset_size, len(data_set))))

        self.shuffle_seed = shuffle_seed
        # Normalize once, before any prompt or provider call. Only the question
        # and shuffled option texts enter the prompt; gold labels stay metadata.
        data_set = data_set.map(
            _prepare_gpqa_sample,
            with_indices=True,
            fn_kwargs={'shuffle_seed': shuffle_seed},
            remove_columns=data_set.column_names,
            desc='Preparing GPQA choices',
        )
        super().__init__(
            data_set,
            tokenizer,
            n_labels=4,
            preamble=self.preamble,
            add_space=add_space,
            numerical=False,
            noisy_level=noisy_level,
        )

    def _format_prompt(self, sample: Dict[str, Any]) -> str:
        choices = '\n'.join(
            f'{label}) {text}'
            for label, text in zip(sample['choices']['label'], sample['choices']['text'])
        )
        return self.preamble.format(question=sample['question'], choices=choices)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self.dataset[index]
        label = int(sample['label'])
        if self.label_ptb is not None:
            label = self.label_ptb.process_label(index, label)
        return {'prompt': self._format_prompt(sample), 'label': label}

    def get_collate_fn(self) -> Callable:
        return qa_dataset_collate_fn

    def get_data_indices(self):
        return self.dataset['row_id']

    def get_input_text(self):
        return self.dataset['question']

    def get_true_labels(self):
        return list(self.dataset['label'])
