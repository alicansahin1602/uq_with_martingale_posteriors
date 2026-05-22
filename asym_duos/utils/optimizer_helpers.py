"""Shared low-level helpers used by all post-hoc optimizers."""
import os
import os.path as osp

import numpy as np


# ---------------------------------------------------------------------------
# Numerical helpers
# ---------------------------------------------------------------------------

def _logsumexp(arr: np.ndarray, axis: int = 1, keepdims: bool = True) -> np.ndarray:
    """Numerically stable log-sum-exp."""
    m = np.max(arr, axis=axis, keepdims=True)
    return m + np.log(np.sum(np.exp(arr - m), axis=axis, keepdims=keepdims) + 1e-12)


def _softmax_from_log(log_p: np.ndarray, ax: int = 1) -> np.ndarray:
    """Convert raw logits to normalised probabilities."""
    log_p_norm = log_p - _logsumexp(log_p, axis=ax, keepdims=True)
    return np.exp(log_p_norm)


def _sort_by_idx(arr: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Reorder *arr* using argsort index *idx* (supports 1-D and N-D arrays)."""
    arr = np.asarray(arr)
    idx = np.asarray(idx)
    if arr.ndim == 1:
        return arr[idx]
    elif arr.ndim >= 2:
        return arr[idx]
    else:
        raise ValueError(f'arr must be at least 1-D (got shape {arr.shape})')


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _read_or_create_path(path: str) -> dict:
    """Load an existing NPZ file as a dict, or create the parent directory."""
    if osp.exists(path):
        return dict(np.load(path, allow_pickle=True))
    os.makedirs(osp.dirname(path), exist_ok=True)
    return {}


def _merge_dictionaries(base: dict, update: dict) -> dict:
    """Recursively merge *update* into *base* (update wins on conflicts)."""
    result = base.copy()
    for k, v in update.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _merge_dictionaries(result[k], v)
        else:
            result[k] = v
    return result


def _save_val_test_results(
    seed: int,
    data_name: str,
    idx: np.ndarray,
    input: np.ndarray,
    logits: np.ndarray,
    softmax_probs: np.ndarray,
    preds: np.ndarray,
    true_labels: np.ndarray,
    existing_dict: dict,
    path: str,
) -> None:
    """Merge this seed's predictions into *existing_dict* and write to *path*."""
    save_dict = {
        str(seed): {
            'data': data_name,
            'idx': idx,
            'input': input,
            'logits': logits,
            'softmax_probs': softmax_probs,
            'preds': preds,
            'true_labels': true_labels,
        }
    }
    merged = _merge_dictionaries(existing_dict, save_dict)
    np.savez_compressed(path, **merged)


# ---------------------------------------------------------------------------
# Debug helpers
# ---------------------------------------------------------------------------

def _sanity_check_on_probs(probs: np.ndarray, label: str) -> None:
    print(f'[{label}] n={probs.shape[0]}, sum-of-probs={probs.sum(axis=1).sum():.4f}')
