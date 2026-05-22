import numpy as np


def _expected_calibration_error(probs: np.ndarray, labels: np.ndarray, num_bins: int = 15) -> float:
    """ECE with uniform confidence bins."""
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == labels).astype(float)

    bin_boundaries = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0
    n = len(labels)
    for i in range(num_bins):
        mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        if not np.any(mask):
            continue
        bin_acc = accuracies[mask].mean()
        bin_conf = confidences[mask].mean()
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return float(ece)


def compute_uncertainty_metrics(probs: np.ndarray, labels: np.ndarray, num_bins: int = 15) -> dict:
    """Compute NLL, Brier, ECE, LPPD, and mean uncertainty from class probabilities."""
    labels = labels.astype(int)
    if probs.ndim == 3:
        probs = probs.squeeze(axis=0)

    true_probs = probs[np.arange(len(labels)), labels]
    nll = float(-np.mean(np.log(true_probs + 1e-12)))
    lppd = float(np.sum(np.log(true_probs + 1e-12)))

    one_hot = np.zeros_like(probs)
    one_hot[np.arange(len(labels)), labels] = 1.0
    brier = float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))

    ece = _expected_calibration_error(probs, labels, num_bins=num_bins)
    mean_uncertainty = float(1.0 - probs.max(axis=1).mean())

    return {
        'nll': nll,
        'lppd': lppd,
        'brier': brier,
        'ece': ece,
        'mean_uncertainty': mean_uncertainty,
    }
