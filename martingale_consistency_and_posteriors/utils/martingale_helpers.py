from copy import deepcopy
from typing import List, Callable, Optional
import re
import transformers
import numpy as np
import mmengine
import os.path as osp


_ANSWER_SUFFIX = "\nAnswer:"


# ---------------------------------------------------------------------------
# Prompt manipulation helpers
# ---------------------------------------------------------------------------
def _strip_answer_suffix(prompt: str) -> str:
    """Remove trailing '\\nAnswer:' from a prompt so it can be fed to a PPR provider."""
    if prompt.endswith(_ANSWER_SUFFIX):
        return prompt[: -len(_ANSWER_SUFFIX)]
    return prompt

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
    return f"{body}\nYour prior answers in previous steps to this specific question were: {history}{_ANSWER_SUFFIX}"

# ---------------------------------------------------------------------------
# Running martingale checks for different methods
# ---------------------------------------------------------------------------

def run_martingale_check(
    get_probs: Callable[[List[str]], np.ndarray],
    n_classes: int,
    dataset,
    K: int,
    n_samples: Optional[int],
    batch_size: int,
    rng: np.random.Generator,
    logger,
    label_chars: list,
    J: int = 1,
) -> dict:
    """Iterate K feedback steps and record the full distribution trajectory.

    At each step k:
      1. Call get_probs on the current prompts → p_k.
      2. Sample a label from p_k for each question.
      3. Rebuild prompts with the sampled label appended to the history.

    get_probs is a provider callable built by one of the _build_*_provider
    factories; it abstracts over HuggingFace, OpenAI, and Anthropic backends.

    J independent trajectories are run per question, each starting fresh
    from the same initial prompt and independently resampling its own answer
    history at every step (same idea as run_ppr_check's J, applied to the
    one-call-per-step iterative provider). The J converged (k=K) distributions
    are i.i.d. samples from the martingale posterior Law(theta_K | x_Q).

    Returns
    -------
    dict with keys:
        distributions  (N, J, K+1, C) -- p_0 through p_K for every question and trajectory
        true_labels    (N,)
        input_texts    list[str]   -- raw initial prompt per question
        data_indices   (N,)        -- dataset row ids
        prompt_history (N, J, K+1) -- exact prompt fed to the model at each step
    """
    N = min(n_samples, len(dataset)) if n_samples is not None else len(dataset)

    # Draw N random indices without replacement for reproducibility via rng
    selected_indices = rng.choice(len(dataset), size=N, replace=False)

    distributions = np.zeros((N, J, K + 1, n_classes), dtype=np.float64)
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

    all_run_prompts: List[List[List[str]]] = []  # [j][k] = list of N prompts

    for j in range(J):
        logger.info(f"  Trajectory j={j + 1}/{J} ...")

        # Per-question answer history accumulated across iterations, fresh per trajectory
        history: List[List[str]] = [[] for _ in range(N)]
        current_prompts = list(initial_prompts)
        # traj_prompts[k] holds the N prompts fed to the model at step k
        traj_prompts: List[List[str]] = []

        for k in range(K + 1):
            traj_prompts.append(list(current_prompts))
            logger.info(f"    k={k}/{K} ...")
            probs_k = np.zeros((N, n_classes), dtype=np.float64)

            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)
                probs_k[start:end] = get_probs(current_prompts[start:end])

            distributions[:, j, k, :] = probs_k

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

        all_run_prompts.append(traj_prompts)

    # all_run_prompts[j][k] is a list of N prompts -> transpose to (N, J, K+1)
    prompt_history = np.array(all_run_prompts, dtype=object).transpose(2, 0, 1)

    return {
        "distributions": distributions,
        "true_labels": true_labels,
        "input_texts": input_texts,
        "data_indices": data_indices,
        "prompt_history": prompt_history,
    }


def run_ppr_check(
    get_theta_trajectory: Callable[[List[str]], np.ndarray],
    n_classes: int,
    dataset,
    n_ppr_samples: int,
    n_samples: Optional[int],
    rng: np.random.Generator,
    logger,
    label_chars: list,
    get_probs_seed: Optional[Callable[[List[str]], np.ndarray]] = None,
    n_seed_answers: int = 0,
    J: int = 1,
) -> dict:
    """PPR check following Kim et al.'s Section 4.1 protocol exactly.

    For each of J independent trajectories per question:
      1. (optional) answer seeding: draw n_seed_answers i.i.d. samples from
         the direct-query distribution (get_probs_seed) and prepend them
         ONCE to the PPR prompt -- x_tilde_Q = [I; Q; S_1:m]. This is the
         paper's Section 3/4.1 "answer seeding," not a per-step re-injected
         belief summary.
      2. Run ONE continuous PPR generation of n_ppr_samples answers via
         get_theta_trajectory, which returns theta_n for n=1..n_ppr_samples
         directly from that single generation (exact per-position logits
         where available, else the cumulative empirical-frequency MLE --
         see the individual provider docstrings in model_calls.py).

    Unlike earlier versions of this function, there is no round-by-round loop
    that re-prompts the model with a textual "current belief distribution"
    summary between steps: the paper's PPR is one uninterrupted generation,
    and theta_n is read off at every prefix length within it.

    k=0 in the returned `distributions` array is a plain Direct-Query call
    (no PPR instruction, via get_probs_seed) -- this is the paper's own
    Direct-Query baseline (used in their Tables 1-2) and keeps this
    function's output shape compatible with the rest of the pipeline
    (compute_martingale_metrics etc. expect a "prior" at index 0). It is
    computed once per question and shared across all J trajectories, same as
    the seed distribution it doubles as.

    Returns
    -------
    dict with keys:
        distributions  (N, J, n_ppr_samples+1, C) -- k=0 is Direct-Query,
                        k=1..n_ppr_samples is theta_n from the single PPR
                        rollout of that trajectory
        true_labels    (N,)
        input_texts    list[str]
        data_indices   (N,)
        prompt_history (N, J, n_ppr_samples+1) -- k=0 is the direct-query
                        prompt; k=1..n_ppr_samples all record the SAME single
                        PPR prompt for that trajectory, since every theta_n
                        in it comes from one continuous generation, not a
                        distinct prompt per step
    """
    N = min(n_samples, len(dataset)) if n_samples is not None else len(dataset)
    selected_indices = rng.choice(len(dataset), size=N, replace=False)

    K = n_ppr_samples
    distributions = np.zeros((N, J, K + 1, n_classes), dtype=np.float64)
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

    # Direct-Query baseline (k=0): one call per question, shared across all J
    # trajectories. Doubles as the distribution answer seeds are drawn from,
    # matching the paper's S_1:m ~ p_phi(.|Q) (Section 4.1).
    logger.info("  Direct-Query baseline (k=0) ...")
    if get_probs_seed is not None:
        direct_probs = get_probs_seed(input_texts)  # (N, n_classes)
    else:
        direct_probs = np.full((N, n_classes), 1.0 / n_classes, dtype=np.float64)
        logger.warning(
            "  No get_probs_seed provided; k=0 Direct-Query baseline left "
            "uniform. Pass a direct-query provider to populate it."
        )
    distributions[:, :, 0, :] = direct_probs[:, None, :]

    def _make_ppr_prompt(body: str, seeds: List[str]) -> str:
        return _insert_prev_answers_ppr(body, seeds) if seeds else body

    direct_query_prompts = list(input_texts)
    ppr_prompts_by_traj: List[List[str]] = []  # [j] -> list of N PPR prompts

    for j in range(J):
        logger.info(f"  Trajectory j={j + 1}/{J} ...")

        # Answer seeding (Section 3/4.1): m i.i.d. direct-query samples,
        # re-sampled independently per trajectory so the J rollouts stay
        # i.i.d., prepended ONCE before the single PPR generation.
        seeds_j: List[List[str]] = [[] for _ in range(N)]
        if n_seed_answers > 0:
            for i in range(N):
                p = direct_probs[i].astype(np.float64)
                p /= p.sum()
                idxs = rng.choice(n_classes, size=n_seed_answers, p=p)
                seeds_j[i] = [label_chars[idx] for idx in idxs]

        ppr_prompts = [_make_ppr_prompt(initial_prompts[i], seeds_j[i]) for i in range(N)]
        ppr_prompts_by_traj.append(ppr_prompts)

        logger.info(f"    Single PPR generation (N={n_ppr_samples} answers) ...")
        theta_traj = get_theta_trajectory(ppr_prompts)  # (N, n_ppr_samples, C)
        distributions[:, j, 1:, :] = theta_traj

    # Build (N, J, K+1) prompt history: k=0 is the direct-query prompt, and
    # k=1..K all record the single PPR prompt for that (question, trajectory)
    # -- there is genuinely only one prompt per trajectory, since the whole
    # theta_1..theta_K trajectory comes from one continuous generation.
    prompt_history = np.empty((N, J, K + 1), dtype=object)
    for j in range(J):
        for i in range(N):
            prompt_history[i, j, 0] = direct_query_prompts[i]
            for k in range(1, K + 1):
                prompt_history[i, j, k] = ppr_prompts_by_traj[j][i]

    return {
        "distributions": distributions,   # (N, J, K+1, C)
        "true_labels": true_labels,
        "input_texts": input_texts,
        "data_indices": data_indices,
        "prompt_history": prompt_history,
    }


# ---------------------------------------------------------------------------
# Computing metrics
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Saving results
# ---------------------------------------------------------------------------

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