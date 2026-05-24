from copy import deepcopy
from typing import List, Callable, Optional

import transformers
import numpy as np
import mmengine
import os.path as osp

_ANSWER_SUFFIX = "\nAnswer:"

def _insert_prev_answers(initial_prompt: str, prev_labels: List[str]) -> str:
    """Splice the answer history into a prompt just before 'Answer:'.

    Turns:
        ...
        Choices:
        A) ...
        Answer:
    Into:
        ...
        Choices:
        A) ...
        Your previous answers were: A, C, B
        Answer:
    """
    if not initial_prompt.endswith(_ANSWER_SUFFIX):
        raise ValueError(
            f"Prompt does not end with '{_ANSWER_SUFFIX}'. "
            "Check that the dataset preamble is unchanged."
        )
    body = initial_prompt[: -len(_ANSWER_SUFFIX)]
    history = ", ".join(prev_labels)
    return f"{body}\nYour previous answers were: {history}{_ANSWER_SUFFIX}"


def _load_tokenizer_only(cfg) -> transformers.PreTrainedTokenizer:
    """Load just the tokenizer from the model config section, no weights."""
    tokenizer_cfg = deepcopy(dict(cfg.model.tokenizer_cfg))
    tokenizer_cls = getattr(transformers, tokenizer_cfg.pop("type"))
    tokenizer = tokenizer_cls.from_pretrained(
        cfg.model.model_name_or_path, **tokenizer_cfg
    )
    special_tokens = {
        k: getattr(tokenizer, v.split(".")[-1])
        if isinstance(v, str) and v.startswith("tokenizer")
        else v
        for k, v in cfg.model.special_tokens.items()
    }
    if special_tokens:
        tokenizer.add_special_tokens(special_tokens)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def run_martingale_check(
    get_probs: Callable[[List[str]], np.ndarray],
    n_classes: int,
    dataset,
    K: int,
    n_samples: Optional[int],
    batch_size: int,
    rng: np.random.Generator,
    logger,
    label_chars: list
) -> dict:
    """Iterate K feedback steps and record the full distribution trajectory.

    At each step k:
      1. Call get_probs on the current prompts → p_k.
      2. Sample a label from p_k for each question.
      3. Rebuild prompts with the sampled label appended to the history.

    get_probs is a provider callable built by one of the _build_*_provider
    factories; it abstracts over HuggingFace, OpenAI, and Anthropic backends.

    Returns
    -------
    dict with keys:
        distributions  (N, K+1, C) -- p_0 through p_K for every question
        true_labels    (N,)
        input_texts    list[str]   -- raw initial prompt per question
        data_indices   (N,)        -- dataset row ids
        prompt_history (N, K+1)   -- exact prompt fed to the model at each step
    """
    N = min(n_samples, len(dataset)) if n_samples is not None else len(dataset)

    distributions = np.zeros((N, K + 1, n_classes), dtype=np.float32)
    true_labels = np.zeros(N, dtype=np.int32)
    input_texts: List[str] = []

    # Build initial prompts and collect ground-truth labels
    initial_prompts: List[str] = []
    for i in range(N):
        sample = dataset[i]
        initial_prompts.append(sample["prompt"])
        true_labels[i] = int(sample["label"])
        input_texts.append(sample["prompt"])

    # Row ids (best-effort; fall back to positional indices)
    try:
        all_row_ids = dataset.get_data_indices()
        data_indices = np.array([all_row_ids[i] for i in range(N)], dtype=np.int32)
    except Exception:
        data_indices = np.arange(N, dtype=np.int32)

    # Per-question answer history accumulated across iterations
    history: List[List[str]] = [[] for _ in range(N)]
    current_prompts = list(initial_prompts)
    # all_prompts[k] holds the N prompts fed to the model at step k
    all_prompts: List[List[str]] = []

    for k in range(K + 1):
        all_prompts.append(list(current_prompts))
        logger.info(f"  Iteration k={k}/{K} ...")
        probs_k = np.zeros((N, n_classes), dtype=np.float32)

        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            probs_k[start:end] = get_probs(current_prompts[start:end])

        distributions[:, k, :] = probs_k

        if k < K:
            next_prompts: List[str] = []
            for i in range(N):
                # Renormalise to guard against float32 rounding before sampling
                p = probs_k[i].astype(np.float64)
                p /= p.sum()
                sampled_idx = int(rng.choice(n_classes, p=p))
                history[i].append(label_chars[sampled_idx])
                next_prompts.append(
                    _insert_prev_answers(initial_prompts[i], history[i])
                )
            current_prompts = next_prompts

    # Reshape to (N, K+1): all_prompts[k][i] -> prompt_history[i][k]
    prompt_history = np.array(all_prompts, dtype=object).T  # (N, K+1)

    return {
        "distributions": distributions,
        "true_labels": true_labels,
        "input_texts": input_texts,
        "data_indices": data_indices,
        "prompt_history": prompt_history,
    }

def compute_martingale_metrics(distributions: np.ndarray, true_labels: np.ndarray) -> dict:
    """Compute drift statistics from the full distribution trajectory.

    Parameters
    ----------
    distributions : (N, K+1, C)
    true_labels   : (N,)

    Returns
    -------
    dict of scalar summary metrics and per-question / per-step arrays.
    """
    p0 = distributions[:, 0, :]       # (N, C)  initial distribution
    p_iter = distributions[:, 1:, :]  # (N, K, C) iterated distributions

    # Mean iterated distribution per question -- martingale says this ≈ p0
    p_mean = p_iter.mean(axis=1)      # (N, C)

    # TV distance between mean iterated and initial: 0.5 * ||p_mean - p0||_1
    tv_per_q = 0.5 * np.abs(p_mean - p0).sum(axis=-1)           # (N,)

    # L1 drift at each step: ||p_k - p0||_1
    l1_drift = np.abs(p_iter - p0[:, None, :]).sum(axis=-1)      # (N, K)

    # Drift profile: mean L1 drift at each step (averaged over questions)
    drift_profile = l1_drift.mean(axis=0)                        # (K,)

    # Variance of each class probability across iterations (per question)
    class_variance = p_iter.var(axis=1)                          # (N, C)

    # Greedy accuracy on the initial prediction
    pred_labels = p0.argmax(axis=-1)
    accuracy = float((pred_labels == true_labels).mean())

    return {
        # Scalar summaries
        "mean_tv": float(tv_per_q.mean()),
        "pass_rate_tv_005": float((tv_per_q < 0.05).mean()),
        "pass_rate_tv_010": float((tv_per_q < 0.10).mean()),
        "mean_l1": float(l1_drift.mean()),
        "accuracy_at_p0": accuracy,
        # Arrays
        "tv_per_question": tv_per_q,
        "mean_l1_per_question": l1_drift.mean(axis=1),
        "drift_profile": drift_profile,
        "class_variance": class_variance,
    }

def save_martingale_results(
    work_dir: str,
    seed: int,
    K: int,
    distributions: np.ndarray,
    true_labels: np.ndarray,
    data_indices: np.ndarray,
    input_texts: List[str],
    prompt_history: np.ndarray,
    metrics: dict,
    logger,
) -> str:
    """Append results for this seed to <work_dir>/martingale_results.npz.

    The npz mirrors the structure used by save_predictions: top-level keys are
    seed strings, values are nested dicts (saved as numpy object arrays).
    """
    mmengine.mkdir_or_exist(work_dir)
    out_path = osp.join(work_dir, "martingale_results.npz")

    existing = (
        dict(np.load(out_path, allow_pickle=True)) if osp.exists(out_path) else {}
    )
    existing[str(seed)] = {
        "distributions": distributions,          # (N, K+1, C)
        "true_labels": true_labels,              # (N,)
        "data_indices": data_indices,            # (N,)
        "input_texts": np.array(input_texts, dtype=object),
        "prompt_history": prompt_history,        # (N, K+1) prompts fed at each step
        "K": np.int32(K),
        # Scalar metrics
        "mean_tv": np.float32(metrics["mean_tv"]),
        "pass_rate_tv_005": np.float32(metrics["pass_rate_tv_005"]),
        "pass_rate_tv_010": np.float32(metrics["pass_rate_tv_010"]),
        "mean_l1": np.float32(metrics["mean_l1"]),
        "accuracy_at_p0": np.float32(metrics["accuracy_at_p0"]),
        # Per-question / per-step arrays
        "tv_per_question": metrics["tv_per_question"],
        "mean_l1_per_question": metrics["mean_l1_per_question"],
        "drift_profile": metrics["drift_profile"],
        "class_variance": metrics["class_variance"],
    }

    np.savez_compressed(out_path, **existing)
    logger.info(f"Results saved to {out_path}")
    return out_path
