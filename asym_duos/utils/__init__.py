from .logging import setup_logger
from .misc import save_predictions
from .typing import Device
from .optimizer_helpers import _softmax_from_log
from .duo_optimizer import optimize_weights, optimize_weights_ts, optimize_weights_laplace, optimize_weights_blob
from .temperature_scaling import optimize_temperature_scaling
from .uncertainty_metrics import compute_uncertainty_metrics
from .laplace_lora import optimize_laplace

__all__ = [
    'setup_logger',
    'Device',
    'save_predictions',
    '_softmax_from_log',
    'optimize_weights',
    'optimize_weights_ts',
    # optimize_weights_blob is the duo-fitting entry point used by
    # tools/postprocess_blob_asymduos.py (BLoB training itself lives in the
    # bayesian-peft fork; this repo only post-processes the dumped npzs).
    'optimize_weights_blob',
    'optimize_temperature_scaling',
    'compute_uncertainty_metrics',
    'optimize_laplace',
    'optimize_weights_laplace',
]
