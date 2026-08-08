# MCQA prompt format and the next-token answer-classification setup are inherited
# from the IB-EDL reference implementation (https://github.com/sandylaker/ib-edl),

from .arc import ARCDataset
from .builder import DATASETS
from .classification import ClassificationDataset
from .csqa import CSQADataset
from .dataset_utils import qa_dataset_collate_fn
from .obqa import OBQADataset
from .race import RaceDataset
from .sciq import SciQDataset

__all__ = [
    'DATASETS',
    'ClassificationDataset',
    'qa_dataset_collate_fn',
    'ARCDataset',
    'OBQADataset',
    'CSQADataset',
    'SciQDataset',
    'RaceDataset',
]
