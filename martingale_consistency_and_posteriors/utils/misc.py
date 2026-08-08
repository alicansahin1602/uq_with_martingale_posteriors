import logging
import os.path as osp
from os import PathLike
from typing import Optional, Union
import mmengine
import numpy as np
from transformers.trainer_utils import PredictionOutput
from .logging import setup_logger


def save_predictions(preds: PredictionOutput, file_path: Union[str, PathLike[str]], seed: int, data_idx, input_text, logger: Optional[logging.Logger] = None) -> None:
    if osp.dirname(file_path) != '':
        mmengine.mkdir_or_exist(osp.dirname(file_path))

    if osp.exists(file_path):
        existing_dict = dict(np.load(file_path, allow_pickle = True))
    else:
        existing_dict = {}

    if isinstance(preds.predictions, tuple):
        # PredictionOutput.predictions can be a tuple of (logits, uncertainties)
        logits, uncertainties = preds.predictions
    else:
        logits, uncertainties = preds.predictions, None

    save_dict = {str(seed) : {'idx': np.array(data_idx).astype(np.int32), 'input': np.array(input_text), \
                 'logits': logits.astype(np.float32), 'true_labels': preds.label_ids}}
    if uncertainties is not None:
        save_dict[str(seed)]['uncertainties'] = uncertainties.astype(np.float32)

    print(f'Seeds in the output file before update with seed {seed}: {existing_dict.keys()} \n')
    existing_dict.update(save_dict)
    print(f'Seeds in the output file after update with seed {seed}: {existing_dict.keys()} \n')
    np.savez_compressed(file_path, **existing_dict)
    logger = logger if logger is not None else setup_logger('ib-edl')
    logger.info(f'Predictions with seed {seed} saved to {file_path}')
