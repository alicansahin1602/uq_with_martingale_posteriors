from copy import deepcopy
from typing import List, Callable, Optional

import transformers
import numpy as np
import mmengine
import os.path as osp

_ANSWER_SUFFIX = "\nAnswer:"


def _strip_answer_suffix(prompt: str) -> str:
    """Remove trailing '\\nAnswer:' from a prompt so it can be fed to a PPR provider."""
    if prompt.endswith(_ANSWER_SUFFIX):
        return prompt[: -len(_ANSWER_SUFFIX)]
    return prompt


#def _insert_prev_answers_ppr(prompt_body: str, prev_labels: List[str]) -> str:
#    """Prepend seed answers to a PPR prompt body (Answer: suffix already stripped).

#    Adds a line of the form 'Previously sampled answers: A, B, A, ...' before
#    the final newline so the model can use them as a warm-start signal.
#    """
#    history = ", ".join(prev_labels)
#    return f"{prompt_body}\nPreviously sampled answers: {history}"

def _insert_prev_answers_ppr(prompt_body: str, prev_labels: List[str], label_chars: List[str] = None, prompt_dist: List[str] = None) -> str:
    if not prev_labels:
        return prompt_body

    if prompt_dist is not None:
        dist_str = ", ".join(f"{label_chars[c]}={prompt_dist[c]:.0%}" for c in range(len(prompt_dist)))
        seed_line = f"\nPreviously sampled answers: {', '.join(prev_labels)}" if prev_labels else ""
        return (
            f"{prompt_body}"
            f"{seed_line}"
            f"\nCurrent belief distribution: {dist_str}."
            f"\nAnchor to this distribution and only update proportionally to new evidence."
        )   
    # fallback
    return f"{prompt_body}\nPreviously sampled answers: {', '.join(prev_labels)}"


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

    # Draw N random indices without replacement for reproducibility via rng
    selected_indices = rng.choice(len(dataset), size=N, replace=False)

    distributions = np.zeros((N, K + 1, n_classes), dtype=np.float32)
    true_labels = np.zeros(N, dtype=np.int32)
    input_texts: List[str] = []

    # Build initial prompts and collect ground-truth labels
    initial_prompts: List[str] = []
    for i, dataset_idx in enumerate(selected_indices):
        sample = dataset[int(dataset_idx)]
        initial_prompts.append(sample["prompt"])
        true_labels[i] = int(sample["label"])
        input_texts.append(sample["prompt"])

    # Row ids (best-effort; fall back to selected dataset indices)
    try:
        all_row_ids = dataset.get_data_indices()
        data_indices = np.array([all_row_ids[int(idx)] for idx in selected_indices], dtype=np.int32)
    except Exception:
        data_indices = selected_indices.astype(np.int32)

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
    logger
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


def run_ppr_check(
    get_probs: Callable[[List[str]], np.ndarray],
    n_classes: int,
    dataset,
    K: int,
    n_samples: Optional[int],
    rng: np.random.Generator,
    logger,
    label_chars: list,
    n_ppr_samples: int = 100,
    get_probs_seed: Optional[Callable[[List[str]], np.ndarray]] = None,
    n_seed_answers: int = 0,
    J: int = 1,
) -> dict:
    """PPR variant of the martingale check with optional direct-query seeding and J trajectories.

    J independent trajectories are run per question to estimate the martingale
    posterior Law(θ_∞ | x_Q).  Each trajectory is a K-step PPR chain; the J
    converged distributions (step K of each run) are the posterior samples.

    Seeding protocol (when get_probs_seed and n_seed_answers are set):
      - get_probs_seed is called ONCE per question to get the direct-query
        distribution; n_seed_answers seeds are re-sampled independently for
        each of the J trajectories so they remain i.i.d.
      - k=0: question body + direct-query seeds for this trajectory.
      - k>0: question body + direct-query seeds + PPR samples from step k-1.

    prompt_history records only trajectory j=0 to keep memory bounded.

    Returns
    -------
    dict with keys:
        distributions  (N, J, K+1, C)
        true_labels    (N,)
        input_texts    list[str]
        data_indices   (N,)
        prompt_history (N, J, K+1) -- prompts for every trajectory and step
    """
    N = min(n_samples, len(dataset)) if n_samples is not None else len(dataset)
    selected_indices = rng.choice(len(dataset), size=N, replace=False)

    distributions = np.zeros((N, J, K + 1, n_classes), dtype=np.float32)
    true_labels = np.zeros(N, dtype=np.int32)
    input_texts: List[str] = []
    initial_prompts: List[str] = []

    for i, dataset_idx in enumerate(selected_indices):
        sample = dataset[int(dataset_idx)]
        input_texts.append(sample["prompt"])
        true_labels[i] = int(sample["label"])
        initial_prompts.append(_strip_answer_suffix(sample["prompt"]))

    try:
        all_row_ids = dataset.get_data_indices()
        data_indices = np.array(
            [all_row_ids[int(idx)] for idx in selected_indices], dtype=np.int32
        )
    except Exception:
        data_indices = selected_indices.astype(np.int32)

    # Direct-query seed distribution: one call per question, shared across all J runs.
    # Each run re-samples its own n_seed_answers labels so the J trajectories are i.i.d.
    seed_dist: Optional[np.ndarray] = None
    if get_probs_seed is not None and n_seed_answers > 0:
        logger.info(
            f"  Direct-query seeding: {n_seed_answers} seeds × {J} trajectories ..."
        )
        seed_dist = get_probs_seed(input_texts)  # (N, n_classes)

    def _make_ppr_prompt(body: str, seeds: List[str]) -> str:
        return _insert_prev_answers_ppr(body, seeds) if seeds else body

    all_run_prompts: List[List[List[str]]] = []  # [j][k] = list of N prompts

    for j in range(J):
        logger.info(f"  Trajectory j={j + 1}/{J} ...")

        # Re-sample seeds independently for each trajectory
        direct_seeds_j: List[List[str]] = [[] for _ in range(N)]
        if seed_dist is not None:
            for i in range(N):
                p = seed_dist[i].astype(np.float64)
                p /= p.sum()
                idxs = rng.choice(n_classes, size=n_seed_answers, p=p)
                direct_seeds_j[i] = [label_chars[idx] for idx in idxs]

        current_prompts = [
            _make_ppr_prompt(initial_prompts[i], direct_seeds_j[i]) for i in range(N)
        ]

        traj_prompts: List[List[str]] = []
        accumulated_ppr_seeds: List[List[str]] = [[] for _ in range(N)]
        for k in range(K + 1):
            traj_prompts.append(list(current_prompts))
            logger.info(f"    k={k}/{K} ...")
            probs_k = get_probs(current_prompts)
            distributions[:, j, k, :] = probs_k

            if k < K:
                next_prompts: List[str] = []
                for i in range(N):
                    p = probs_k[i].astype(np.float64)
                    p /= p.sum()
                    accumulated_ppr_seeds[i].append(label_chars[int(np.argmax(p))])
                    combined = direct_seeds_j[i] + accumulated_ppr_seeds[i]
                    next_prompts.append(
                        _insert_prev_answers_ppr(initial_prompts[i], combined, label_chars=label_chars, prompt_dist=p)
                    )
                current_prompts = next_prompts

        all_run_prompts.append(traj_prompts)

    # all_run_prompts[j][k] is a list of N prompts -> transpose to (N, J, K+1)
    prompt_history = np.array(all_run_prompts, dtype=object).transpose(2, 0, 1)

    return {
        "distributions": distributions,   # (N, J, K+1, C)
        "true_labels": true_labels,
        "input_texts": input_texts,
        "data_indices": data_indices,
        "prompt_history": prompt_history,
    }


def compute_emd_metrics(distributions: np.ndarray) -> dict:
    """Compute Expected Martingale Drift (EMD) from a PPR distribution trajectory.

    EMD = (1/K) * mean_{j,n} sum_{k=1}^{K} TV(p^(k,j), p^(0,j))

    Accepts both (N, J, K+1, C) from run_ppr_check and the legacy (N, K+1, C)
    shape (J=1 case); the latter is promoted to (N, 1, K+1, C) internally.

    Returns
    -------
    dict with scalar 'emd' and per-step array 'emd_profile' (length K),
    averaged over both questions N and trajectories J.
    """
    if distributions.ndim == 3:
        distributions = distributions[:, None, :, :]  # (N, 1, K+1, C)

    p0 = distributions[:, :, 0:1, :]       # (N, J, 1,  C) — anchor per trajectory
    p_rest = distributions[:, :, 1:, :]    # (N, J, K,  C)
    tv = 0.5 * np.abs(p_rest - p0).sum(axis=-1)  # (N, J, K)
    emd_profile = tv.mean(axis=(0, 1))             # (K,)  averaged over N and J
    emd = float(emd_profile.mean())
    return {
        "emd": emd,
        "emd_profile": emd_profile,
    }


def compute_martingale_posterior_metrics(distributions: np.ndarray) -> dict:
    """Estimate the martingale posterior from J converged distributions.

    Uses the K-th step (last step) of each of the J independent trajectories
    as i.i.d. samples from Law(θ_∞ | x_Q).

    Parameters
    ----------
    distributions : (N, J, K+1, C)  — output of run_ppr_check with J > 1

    Returns
    -------
    dict with:
        posterior_samples       (N, J, C)  -- converged dist from each trajectory
        posterior_mean          (N, C)     -- mean converged distribution
        posterior_var           (N, C)     -- variance across J trajectories
        posterior_entropy       (N,)       -- entropy of the mean distribution
        mean_posterior_var      float      -- scalar: mean inter-trajectory variance
        mean_posterior_entropy  float      -- scalar: mean entropy
    """
    posterior_samples = distributions[:, :, -1, :]   # (N, J, C)
    posterior_mean = posterior_samples.mean(axis=1)   # (N, C)
    posterior_var = posterior_samples.var(axis=1)     # (N, C)

    eps = 1e-10
    entropy = -(posterior_mean * np.log(posterior_mean + eps)).sum(axis=-1)  # (N,)

    return {
        "posterior_samples": posterior_samples,
        "posterior_mean": posterior_mean,
        "posterior_var": posterior_var,
        "posterior_entropy": entropy,
        "mean_posterior_var": float(posterior_var.mean()),
        "mean_posterior_entropy": float(entropy.mean()),
    }
