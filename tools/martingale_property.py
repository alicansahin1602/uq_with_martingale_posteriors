"""WP1: Martingale property check for LLMs on Q&A benchmarks.

Tests whether a model is self-consistent in the martingale sense (Eq. 1 of the
proposal) by iteratively feeding its own sampled answers back as context and
measuring how much the output distribution drifts:

    E[p(y | y_{1:n}, Y_{n+1:n+k}) | y_{1:n}] = p(y | y_{1:n})

Supports both open-source (HuggingFace) and closed-source (OpenAI, Anthropic)
models through a unified provider interface.

Open-source (HuggingFace) — config unchanged from train.py
-----------------------------------------------------------
python tools/martingale_property.py \\
    --config configs/arc_c_qwen2_7b/lora_arc_c_qwen2_7b.yaml \\
    -w data/results_mp --K 20 --n-samples 200 --seed 42 \\
    --cfg-options model.use_peft=False

Closed-source (API) — add api_model section to config
------------------------------------------------------
The `model` section is still required to load the tokenizer for prompt
formatting; model weights are not loaded when `api_model` is present.

    # configs/arc_c_gpt4o/martingale_arc_c_gpt4o.yaml
    _base_: ['../_base_/arc_c.yaml', '../_base_/qwen2_7b.yaml',
             '../_base_/misc.yaml', '../_base_/non_edl_schedule.yaml']
    api_model:
        provider: openai          # openai | anthropic
        model_name: gpt-4o
        use_logprobs: true        # OpenAI only; false falls back to sampling
        n_samples: 30             # samples per prompt (sampling mode / Anthropic)

Set OPENAI_API_KEY or ANTHROPIC_API_KEY in the environment before running.
"""

import argparse
import os.path as osp
from copy import deepcopy
from datetime import datetime
from typing import Callable, List, Optional

import mmengine
import numpy as np
import torch
import torch.nn.functional as F
import transformers
from torch.utils.data import ConcatDataset
from mmengine.runner.utils import set_random_seed
import openai
import anthropic
from asym_duos import DATASETS, get_model_and_tokenizer, setup_logger

# All dataset preambles end with this exact suffix.
_ANSWER_SUFFIX = "\nAnswer:"


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Martingale property check for LLMs on Q&A benchmarks."
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
    p.add_argument("--split", default="test",
                   choices=["train", "val", "test", "all"],
                   help="Dataset split to evaluate on. 'all' concatenates train+val+test.")
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
# Tokenizer-only loader (used by API providers that skip model weight loading)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Provider factories — each returns Callable[[List[str]], np.ndarray]
# ---------------------------------------------------------------------------

def _build_hf_provider(
    model,
    tokenizer,
    target_ids: torch.Tensor,
    tokenizer_run_cfg: dict,
    device: torch.device,
) -> Callable[[List[str]], np.ndarray]:
    """HuggingFace open-source provider: batched forward pass over logits."""
    @torch.no_grad()
    def get_probs(prompts: List[str]) -> np.ndarray:
        inputs = tokenizer(prompts, **tokenizer_run_cfg).to(device)
        logits = model(**inputs).logits[:, -1, target_ids]  # (B, n_classes)
        return F.softmax(logits.float(), dim=-1).cpu().numpy()
    return get_probs


def _build_openai_provider(
    model_name: str,
    label_chars: List[str],
    use_logprobs: bool,
    n_samples: int,
) -> Callable[[List[str]], np.ndarray]:
    """OpenAI provider.

    use_logprobs=True  — one API call per prompt using top_logprobs (fast, exact).
    use_logprobs=False — n_samples calls per prompt using temperature sampling.
    """

    client = openai.OpenAI()
    n_classes = len(label_chars)

    def get_probs(prompts: List[str]) -> np.ndarray:
        probs = np.zeros((len(prompts), n_classes), dtype=np.float32)
        for i, prompt in enumerate(prompts):
            if use_logprobs:
                resp = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1,
                    logprobs=True,
                    top_logprobs=20,
                )
                top = resp.choices[0].logprobs.content[0].top_logprobs
                log_p = {t.token.strip(): t.logprob for t in top}
                raw = np.array(
                    [np.exp(log_p[c]) if c in log_p else 1e-10 for c in label_chars],
                    dtype=np.float64,
                )
                probs[i] = (raw / raw.sum()).astype(np.float32)
            else:
                counts = np.zeros(n_classes, dtype=np.float64)
                for _ in range(n_samples):
                    resp = client.chat.completions.create(
                        model=model_name,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=1,
                        temperature=1.0,
                    )
                    ans = resp.choices[0].message.content.strip().upper()
                    if ans in label_chars:
                        counts[label_chars.index(ans)] += 1
                    else:
                        counts += 1.0 / n_classes  # uniform fallback for off-label replies
                probs[i] = (counts / counts.sum()).astype(np.float32)
        return probs

    return get_probs


def _build_anthropic_provider(
    model_name: str,
    label_chars: List[str],
    n_samples: int,
) -> Callable[[List[str]], np.ndarray]:
    """Anthropic provider: sampling-based (no logprob API available)."""
    client = anthropic.Anthropic()
    n_classes = len(label_chars)

    def get_probs(prompts: List[str]) -> np.ndarray:
        probs = np.zeros((len(prompts), n_classes), dtype=np.float32)
        for i, prompt in enumerate(prompts):
            counts = np.zeros(n_classes, dtype=np.float64)
            for _ in range(n_samples):
                resp = client.messages.create(
                    model=model_name,
                    max_tokens=1,
                    messages=[{"role": "user", "content": prompt}],
                )
                ans = resp.content[0].text.strip().upper()
                if ans in label_chars:
                    counts[label_chars.index(ans)] += 1
                else:
                    counts += 1.0 / n_classes
            probs[i] = (counts / counts.sum()).astype(np.float32)
        return probs

    return get_probs


# ---------------------------------------------------------------------------
# Main martingale check loop
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
        cfg.train_cfg["per_device_eval_batch_size"] = 4
        for s in (["train", "val", "test"] if args.split == "all" else [args.split]):
            cfg.data[s]["subset_size"] = 8

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
    batch_size = cfg.train_cfg.per_device_eval_batch_size
    tokenizer_run_cfg = dict(cfg.tokenizer_run_cfg)

    logger.info(
        f"K={args.K}  n_samples={args.n_samples}  "
        f"batch_size={batch_size}  max_length={tokenizer_run_cfg['max_length']}  "
        f"seed={args.seed}  split={args.split}"
    )

    # Build provider and dataset — dispatch on whether api_model is present
    api_cfg = cfg.get("api_model", None)

    if api_cfg is not None:
        # Closed-source path: load only the tokenizer for prompt formatting
        tokenizer = _load_tokenizer_only(cfg)
        _split_names = ["train", "val", "test"] if args.split == "all" else [args.split]
        splits = [
            DATASETS.build(cfg.data[s], default_args=dict(tokenizer=tokenizer))
            for s in _split_names
        ]
        dataset = ConcatDataset(splits) if len(splits) > 1 else splits[0]
        n_classes = splits[0].n_labels
        label_chars = [chr(ord("A") + i) for i in range(n_classes)]
        provider = api_cfg.provider.lower()
        if provider == "openai":
            get_probs = _build_openai_provider(
                model_name=api_cfg.model_name,
                label_chars=label_chars,
                use_logprobs=api_cfg.get("use_logprobs", True),
                n_samples=api_cfg.get("n_samples", 30),
            )
        elif provider == "anthropic":
            get_probs = _build_anthropic_provider(
                model_name=api_cfg.model_name,
                label_chars=label_chars,
                n_samples=api_cfg.get("n_samples", 30),
            )
        else:
            raise ValueError(f"Unknown api_model.provider '{provider}'. Use 'openai' or 'anthropic'.")
        logger.info(f"API provider: {provider} / {api_cfg.model_name}")
    else:
        # Open-source path: load HuggingFace model + tokenizer
        model, tokenizer = get_model_and_tokenizer(**cfg.model, device=device)
        model.eval()
        _split_names = ["train", "val", "test"] if args.split == "all" else [args.split]
        splits = [
            DATASETS.build(cfg.data[s], default_args=dict(tokenizer=tokenizer))
            for s in _split_names
        ]
        target_ids = splits[0].target_ids.to(device)
        dataset = ConcatDataset(splits) if len(splits) > 1 else splits[0]
        n_classes = target_ids.shape[0]
        get_probs = _build_hf_provider(model, tokenizer, target_ids, tokenizer_run_cfg, device)
        logger.info(f"HuggingFace provider: {cfg.model.model_name_or_path}")

    logger.info(
        f"Dataset: {'+'.join(_split_names)}  "
        f"size={len(dataset)}  n_classes={n_classes}"
    )
    logger.info(f"Running martingale check: K={args.K} iterations ...")

    result = run_martingale_check(
        get_probs=get_probs,
        n_classes=n_classes,
        dataset=dataset,
        K=args.K,
        n_samples=args.n_samples,
        batch_size=batch_size,
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
