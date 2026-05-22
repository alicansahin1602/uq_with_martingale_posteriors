"""WP1: Martingale property check for LLMs on Q&A benchmarks.

Tests whether a model is self-consistent in the martingale sense (Eq. 1 of the
proposal) by iteratively feeding its own sampled answers back as context and
measuring how much the output distribution drifts:

    E[p(y | y_{1:n}, Y_{n+1:n+k}) | y_{1:n}] = p(y | y_{1:n})

If this holds, the mean of the iterated distributions p_1, ..., p_K should
stay close to the initial distribution p_0 (measured by TV distance).

The script reuses the same config format, model builder, and dataset registry
as train.py -- it only adds the iterative-prompting loop on top.

Usage
-----
# Zero-shot base model (no PEFT)
python tools/martingale_property.py \\
    --config configs/arc_c_qwen2_7b/lora_arc_c_qwen2_7b.yaml \\
    -w data/results_mp \\
    --K 20 --n-samples 200 --seed 42 \\
    --cfg-options model.use_peft=False

# Fine-tuned model (peft_path must point to a trained checkpoint)
python tools/martingale_property.py \\
    --config configs/arc_c_qwen2_7b/lora_arc_c_qwen2_7b.yaml \\
    -w data/results_mp \\
    --K 20 --n-samples 200 --seed 42 \\
    --cfg-options model.peft_path=/path/to/checkpoint

# Quick smoke test
python tools/martingale_property.py \\
    --config configs/arc_c_qwen2_7b/lora_arc_c_qwen2_7b.yaml \\
    -w data/results_mp --test-run \\
    --cfg-options model.use_peft=False
"""

import argparse
import os.path as osp
from datetime import datetime
from typing import List, Optional

import mmengine
import numpy as np
import torch
import torch.nn.functional as F
from mmengine.runner.utils import set_random_seed

from asym_duos import DATASETS, get_model_and_tokenizer, setup_logger

# All dataset preambles end with this exact suffix.
_ANSWER_SUFFIX = "\nAnswer:"


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="WP1: Martingale property check for LLMs on Q&A benchmarks."
    )
    p.add_argument("--config", required=True,
                   help="Config file for the model to evaluate.")
    p.add_argument("--work-dir", "-w", required=True,
                   help="Root directory for outputs.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpu-id", type=int, default=0)

    # Martingale check hyper-parameters
    p.add_argument("--K", type=int, default=20,
                   help="Number of iterative feedback steps (depth of the chain).")
    p.add_argument("--n-samples", type=int, default=None,
                   help="Number of questions to evaluate (None = full split).")
    p.add_argument("--batch-size", type=int, default=16,
                   help="Forward-pass batch size.")
    p.add_argument("--split", default="test",
                   choices=["train", "val", "test"],
                   help="Dataset split to evaluate on.")
    p.add_argument("--max-length", type=int, default=512,
                   help="Tokenizer max_length (increased vs training to fit history).")

    # Config overrides
    p.add_argument("--cfg-options", "-o", nargs="+", action=mmengine.DictAction,
                   help="Override config values, e.g. model.use_peft=False.")

    # Quick test
    p.add_argument("--test-run", action="store_true",
                   help="Smoke test: 8 samples, K=3, batch_size=4.")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Output path
# ---------------------------------------------------------------------------

def _build_work_dir(root: str, cfg_path: str) -> str:
    # 'configs/arc_c_qwen2_7b/lora_arc_c_qwen2_7b.yaml'
    # -> '<root>/arc_c_qwen2_7b/martingale_check'
    folder = osp.basename(osp.dirname(cfg_path))
    return osp.join(root, folder, "martingale_check")


# ---------------------------------------------------------------------------
# Prompt utilities
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Model inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def _get_class_probs_batch(
    model,
    tokenizer,
    prompts: List[str],
    target_ids: torch.Tensor,
    tokenizer_run_cfg: dict,
    device: torch.device,
) -> np.ndarray:
    """Single batched forward pass → softmax class probabilities.

    Returns
    -------
    probs : np.ndarray, shape (len(prompts), n_classes)
    """
    inputs = tokenizer(prompts, **tokenizer_run_cfg).to(device)
    logits = model(**inputs).logits[:, -1, target_ids]  # (B, n_classes)
    probs = F.softmax(logits.float(), dim=-1).cpu().numpy()
    return probs


# ---------------------------------------------------------------------------
# Main martingale check loop
# ---------------------------------------------------------------------------

def run_martingale_check(
    model,
    tokenizer,
    dataset,
    target_ids: torch.Tensor,
    tokenizer_run_cfg: dict,
    device: torch.device,
    K: int,
    n_samples: Optional[int],
    batch_size: int,
    rng: np.random.Generator,
    logger,
) -> dict:
    """Iterate K feedback steps and record the full distribution trajectory.

    At each step k:
      1. Run a batched forward pass on the current prompts → p_k.
      2. Sample a label from p_k for each question.
      3. Rebuild prompts with the sampled label appended to the history.

    Returns
    -------
    dict with keys:
        distributions  (N, K+1, C) -- p_0 through p_K for every question
        true_labels    (N,)
        input_texts    list[str]   -- raw initial prompt per question
        data_indices   (N,)        -- dataset row ids
        prompt_history (N, K+1)   -- exact prompt fed to the model at each step
    """
    n_classes = target_ids.shape[0]
    label_chars = [chr(ord("A") + i) for i in range(n_classes)]

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
            batch_probs = _get_class_probs_batch(
                model, tokenizer,
                current_prompts[start:end],
                target_ids, tokenizer_run_cfg, device,
            )
            probs_k[start:end] = batch_probs

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


# ---------------------------------------------------------------------------
# Metrics
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


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_results(
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    set_random_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(
        f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
    )
    timestamp = datetime.now().strftime("%m%d_%H%M_%S")

    # Load and optionally patch config
    cfg = mmengine.Config.fromfile(args.config)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)

    if args.test_run:
        args.K = 3
        args.n_samples = 8
        args.batch_size = 4
        cfg.data[args.split]["subset_size"] = 8

    # Warn if the config would attach an untrained PEFT adapter
    if cfg.model.get("use_peft") and cfg.model.get("peft_path") is None:
        print(
            "[martingale_property] WARNING: use_peft=True but peft_path is None. "
            "An untrained LoRA adapter will be attached. "
            "For zero-shot evaluation pass --cfg-options model.use_peft=False."
        )

    work_dir = _build_work_dir(args.work_dir, args.config)
    mmengine.mkdir_or_exist(work_dir)

    logger = setup_logger(
        name="martingale-check",
        filepath=osp.join(work_dir, f"{timestamp}.log"),
    )
    logger.info(f"Config:\n{'='*60}\n{cfg.pretty_text}\n{'='*60}")
    logger.info(
        f"K={args.K}  n_samples={args.n_samples}  "
        f"batch_size={args.batch_size}  seed={args.seed}  split={args.split}"
    )

    # Patch tokenizer max_length to accommodate the history suffix
    tokenizer_run_cfg = dict(cfg.tokenizer_run_cfg)
    tokenizer_run_cfg["max_length"] = args.max_length

    # Load model (inference only — no training)
    model, tokenizer = get_model_and_tokenizer(**cfg.model, device=device)
    model.eval()

    # Build dataset
    dataset = DATASETS.build(
        cfg.data[args.split], default_args=dict(tokenizer=tokenizer)
    )
    target_ids = dataset.target_ids.to(device)
    n_classes = target_ids.shape[0]

    logger.info(
        f"Dataset: {type(dataset).__name__}  "
        f"size={len(dataset)}  n_classes={n_classes}"
    )
    logger.info(f"Running martingale check: K={args.K} iterations ...")

    result = run_martingale_check(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        target_ids=target_ids,
        tokenizer_run_cfg=tokenizer_run_cfg,
        device=device,
        K=args.K,
        n_samples=args.n_samples,
        batch_size=args.batch_size,
        rng=rng,
        logger=logger,
    )

    metrics = compute_martingale_metrics(result["distributions"], result["true_labels"])

    # Log summary
    N = result["distributions"].shape[0]
    logger.info("=" * 60)
    logger.info("Martingale Property Check — Summary")
    logger.info(f"  Questions evaluated : {N}")
    logger.info(f"  Iterations (K)      : {args.K}")
    logger.info(f"  Accuracy at p0      : {metrics['accuracy_at_p0']:.4f}")
    logger.info(f"  Mean TV drift       : {metrics['mean_tv']:.4f}  "
                f"(0 = perfect martingale, 1 = maximum drift)")
    logger.info(f"  Pass rate TV < 0.05 : {metrics['pass_rate_tv_005']:.4f}")
    logger.info(f"  Pass rate TV < 0.10 : {metrics['pass_rate_tv_010']:.4f}")
    logger.info(f"  Mean L1 drift       : {metrics['mean_l1']:.4f}")
    logger.info(
        f"  Drift profile (k=1..{args.K}): "
        + np.array2string(metrics["drift_profile"], precision=4, separator=", ")
    )
    logger.info("=" * 60)

    out_path = save_results(
        work_dir=work_dir,
        seed=args.seed,
        K=args.K,
        distributions=result["distributions"],
        true_labels=result["true_labels"],
        data_indices=result["data_indices"],
        input_texts=result["input_texts"],
        prompt_history=result["prompt_history"],
        metrics=metrics,
        logger=logger,
    )

    print(f"\n[martingale_property] Completed.")
    print(f"  Output              : {out_path}")
    print(f"  Accuracy at p0      : {metrics['accuracy_at_p0']:.4f}")
    print(f"  Mean TV drift       : {metrics['mean_tv']:.4f}")
    print(f"  Pass rate (TV<0.05) : {metrics['pass_rate_tv_005']:.4f}")
    print(f"  Pass rate (TV<0.10) : {metrics['pass_rate_tv_010']:.4f}")


if __name__ == "__main__":
    main()
