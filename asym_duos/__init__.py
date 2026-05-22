from .datasets import DATASETS
from .models import get_model_and_tokenizer
from .train_eval import (
    ClassificationMetric,
    EvidentialTrainer,
    FTTrainer,
    UpdateRegWeightCallback,
    plot_predictions,
)
from .utils import (
    _softmax_from_log,
    compute_uncertainty_metrics,
    optimize_laplace,
    optimize_temperature_scaling,
    optimize_weights,
    optimize_weights_blob,
    optimize_weights_laplace,
    optimize_weights_ts,
    save_predictions,
    setup_logger,
)

__all__ = [
    'DATASETS',
    'get_model_and_tokenizer',
    'ClassificationMetric',
    'EvidentialTrainer',
    'FTTrainer',
    'UpdateRegWeightCallback',
    'plot_predictions',
    '_softmax_from_log',
    'compute_uncertainty_metrics',
    'optimize_laplace',
    'optimize_temperature_scaling',
    'optimize_weights',
    # optimize_weights_blob is used by tools/postprocess_blob_asymduos.py
    # to combine BLoB predictions produced by the bayesian-peft fork.
    'optimize_weights_blob',
    'optimize_weights_laplace',
    'optimize_weights_ts',
    'save_predictions',
    'setup_logger',
]
