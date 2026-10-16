"""Local JSON MCQA data with the existing question/prompt contract."""

import json
from pathlib import Path
from typing import Optional

from datasets import Dataset

from .builder import DATASETS
from .classification import ClassificationDataset
from .dataset_utils import qa_dataset_collate_fn


@DATASETS.register_module()
class LocalMCQADataset(ClassificationDataset):
    """Read question/choices/answer_key records without downloads or HF access.

    Relative paths are resolved against the repository root, not the caller's
    working directory. Option order is preserved across all calls and models.
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
        data_file: str,
        tokenizer=None,
        n_labels: int = 4,
        add_space: bool = True,
        subset_size: Optional[int] = None,
        noisy_level: Optional[float] = None,
    ):
        if not 2 <= n_labels <= 26:
            raise ValueError('n_labels must be between 2 and 26.')
        if subset_size is not None and subset_size < 0:
            raise ValueError('subset_size must be non-negative or None.')
        path = Path(data_file).expanduser()
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[2] / path
        with path.open(encoding='utf-8') as source:
            records = json.load(source)
        if not isinstance(records, list) or not records:
            raise ValueError('Local MCQA data must be a non-empty JSON list.')

        labels = [chr(ord('A') + i) for i in range(n_labels)]
        rows = []
        for index, record in enumerate(records):
            if not isinstance(record, dict):
                raise ValueError(f'Row {index}: expected a JSON object.')
            question = record.get('question')
            choices = record.get('choices')
            answer_key = record.get('answer_key')
            if not isinstance(question, str) or not question.strip():
                raise ValueError(f'Row {index}: question must be non-empty text.')
            if (
                not isinstance(choices, list) or len(choices) != n_labels
                or any(not isinstance(c, str) or not c.strip() for c in choices)
            ):
                raise ValueError(f'Row {index}: expected {n_labels} non-empty choices.')
            if len({c.strip() for c in choices}) != n_labels:
                raise ValueError(f'Row {index}: choices must be distinct.')
            if answer_key not in labels:
                raise ValueError(f'Row {index}: answer_key must be one of {labels}.')
            rows.append({
                'question': question.strip(),
                'choices': [c.strip() for c in choices],
                'label': labels.index(answer_key),
                'row_id': index,
            })

        data_set = Dataset.from_list(rows)
        if subset_size is not None:
            data_set = data_set.select(range(min(subset_size, len(data_set))))
        super().__init__(
            data_set, tokenizer, n_labels=n_labels, preamble=self.preamble,
            add_space=add_space, numerical=False, noisy_level=noisy_level,
        )

    def __getitem__(self, index):
        sample = self.dataset[index]
        choices = '\n'.join(
            f'{label}) {text}'
            for label, text in zip(self.label_chars, sample['choices'])
        )
        label = int(sample['label'])
        if self.label_ptb is not None:
            label = self.label_ptb.process_label(index, label)
        return {
            'prompt': self.preamble.format(question=sample['question'], choices=choices),
            'label': label,
        }

    def get_collate_fn(self):
        return qa_dataset_collate_fn

    def get_data_indices(self):
        return self.dataset['row_id']

    def get_input_text(self):
        return self.dataset['question']

    def get_true_labels(self):
        return list(self.dataset['label'])
