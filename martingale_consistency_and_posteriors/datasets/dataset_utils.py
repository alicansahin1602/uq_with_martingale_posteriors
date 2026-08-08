from typing import Any, Dict, List, Union
from collections import Counter
import torch

def add_index_to_dataset(data_set):
    """Adds a 'row_id' field to each split in the dataset, containing the index of each sample."""
    data_set = data_set.map(
        lambda _, idx: {"row_id": idx},
        with_indices=True
    )
    return data_set['row_id']

def qa_dataset_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Union[List[str], torch.Tensor]]:
    prompts: List[str] = []
    labels: List[int] = []
    for sample in batch:
        prompts.append(sample['prompt'])
        labels.append(sample['label'])

    return {'prompts': prompts, 'labels': torch.LongTensor(labels)}


def summarize_dataset(ds, ds_type, dataset_name, text_col="text", label_col="label"):
    """Print basic dataset statistics for each split.

    Args:
        ds (DatasetDict): Hugging Face dataset dictionary.
        text_col (str): Text column name.
        label_col (str): Label column name.

    Returns:
        None: Prints summary statistics to stdout.
    """


    print("\n")
    ds_summary = ds[ds_type]
    print(f"{ds_type.upper()} DATASET SUMMARY for {dataset_name.upper()}")
    print("=" * 40)
    
    # Number of samples
    print(f"Total samples : {ds_summary.num_rows}")
    
    # Number of unique labels (classification)
    if label_col in ds_summary.column_names:
        labels = set(ds_summary[label_col])
        print(f"Number of labels : {len(labels)}")
        print(f"Labels : {labels}")
    else:
        print(f"Label column '{label_col}' not found.")
    print(f"Label Distribution : {Counter(ds_summary[label_col])}")

    # Text length statistics
    if text_col in ds_summary.column_names:
        lengths = [len(x) for x in ds_summary[text_col]]
        print(f"Min text length : {min(lengths)}")
        print(f"Max text length : {max(lengths)}")
        print(f"Avg text length : {sum(lengths) / len(lengths):.2f}")
    else:
        print(f"Text column '{text_col}' not found.")
