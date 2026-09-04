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
        n_api_samples: 30         # API calls per prompt for probability estimation (sampling mode / Anthropic)

Set OPENAI_API_KEY or ANTHROPIC_API_KEY in the environment before running.
"""

import argparse
import os.path as osp
from datetime import datetime

import mmengine
import numpy as np
import torch
from torch.utils.data import ConcatDataset
from mmengine.runner.utils import set_random_seed
from martingale_consistency_and_posteriors import (
     DATASETS,
     get_model_and_tokenizer,
     setup_logger,
     _build_hf_provider,
     _build_openai_provider,
     _build_anthropic_provider,
     _build_deepseek_provider,
     _build_ppr_hf_provider,
     _build_ppr_openai_provider,
     _build_ppr_anthropic_provider,
     _build_ppr_deepseek_provider,
     run_martingale_check,
     compute_martingale_metrics,
     save_martingale_results,
     run_ppr_check,
     compute_emd_metrics,
     compute_martingale_posterior_metrics,
     _build_openai_martingale_sampling_provider,
     run_martingale_sampling_check
)
import os
from dotenv import load_dotenv, find_dotenv


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
                   help="Number of iterative feedback steps (depth of the chain). "
                        "Only used with --mode iterative; ignored for --mode ppr, "
                        "which uses --n-ppr-samples as its single trajectory-length "
                        "parameter (matching the paper's one-continuous-generation PPR).")
    p.add_argument("--n-samples", type=int, default=None,
                   help="Number of questions to evaluate (None = full split).")
    p.add_argument("--n-api-samples", type=int, default=None,
                   help="API calls per prompt for probability estimation when use_logprobs=False "
                        "(overrides api_model.n_api_samples in config; default: 30).")
    p.add_argument("--split", default="test",
                   choices=["train", "val", "test", "all"],
                   help="Dataset split to evaluate on. 'all' concatenates train+val+test.")
    
    # Config overrides
    p.add_argument("--cfg-options", "-o", nargs="+", action=mmengine.DictAction,
                   help="Override config values, e.g. model.use_peft=False.")

    # PPR / retrieval mode
    p.add_argument("--mode", default="iterative", choices=["iterative", "ppr", "sampling"],
                   help="'iterative': one call per step (existing); "
                        "'ppr': single call generates N i.i.d. samples (PPR); "
                        "'sampling': use sampling for probability estimation.")

    p.add_argument("--n-ppr-samples", type=int, default=100,
                   help="Length N of the single continuous PPR generation (the paper's "
                        "Section 4.1 protocol: one completion produces A_1..A_N, and "
                        "theta_n is recorded for every n=1..N). Only used with --mode ppr.")
    
    p.add_argument("--n-seed-answers", type=int, default=0,
                   help="Direct-query warm-start seeds per question (paper's seeding "
                        "protocol). 0 = no seeding. Only used with --mode ppr.")
    
    p.add_argument("--J", type=int, default=1,
                   help="Number of independent trajectories (full K-step runs) per question, "
                        "each restarted fresh from the initial prompt. J>1 estimates the "
                        "martingale posterior Law(θ_∞ | x_Q). Used with --mode iterative/ppr.")

    p.add_argument("--n-shots", type=int, default=4,
                   help="Number of similar training examples to retrieve as few-shot context D. "
                        "Only used with --mode retrieval.")

    # Quick test
    p.add_argument("--test-run", action="store_true",
                   help="Smoke test: 8 samples, K=3, batch_size=4.")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Output path
# ---------------------------------------------------------------------------

def _build_work_dir(root: str, cfg_path: str, method: str) -> str:
    # 'configs/arc_c_qwen2_7b/lora_arc_c_qwen2_7b.yaml'
    # -> '<root>/arc_c_qwen2_7b/martingale_check'
    folder = osp.basename(osp.dirname(cfg_path))
    return osp.join(root, folder, "martingale_check", method)

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
        args.n_ppr_samples = min(args.n_ppr_samples, 5)
        args.n_samples = 8
        for s in (["train", "val", "test"] if args.split == "all" else [args.split]):
            cfg.data[s]["subset_size"] = 8
        ## Reduce the batch_size for open source models
        if not cfg.get("api_model", None):
            cfg.train_cfg["per_device_eval_batch_size"] = 4

    method = args.mode  # 'iterative' | 'ppr'
    work_dir = _build_work_dir(args.work_dir, args.config, method)
    mmengine.mkdir_or_exist(work_dir)

    logger = setup_logger(
        name="martingale-check",
        filepath=osp.join(work_dir, f"{timestamp}.log"),
    )
    logger.info(f"Config:\n{'='*60}\n{cfg.pretty_text}\n{'='*60}")


    logger.info(
        f"K={args.K}  n_samples={args.n_samples}  "
        f"seed={args.seed}  split={args.split}"
    )

    # Build provider and dataset — dispatch on whether api_model is present
    api_cfg = cfg.get("api_model", None)
    batch_size = None

    #if api_cfg is not None:
    ## Load the api keys
    env_file = find_dotenv(usecwd=True)
    if env_file:
        load_dotenv(env_file)
    else:
        logger.warning("No .env file found; API keys must be set in the environment.")

    ## Load the dataset(s)
    _split_names = ["train", "val", "test"] if args.split == "all" else [args.split]
    splits = [
        DATASETS.build(cfg.data[s], default_args=dict(tokenizer=None))
        for s in _split_names
    ]
    dataset = ConcatDataset(splits) if len(splits) > 1 else splits[0]
    n_classes = splits[0].n_labels
    label_chars = splits[0].label_chars

    ## Get closed source provider and HPs
    provider = api_cfg.provider.lower()
    n_api_samples = args.n_api_samples or api_cfg.get("n_api_samples", 30)
    n_ppr_samples = args.n_ppr_samples
    get_probs_seed = None

    if args.mode == "ppr":
        # Paper's Section 4.1 protocol: theta_n is read exactly from
        # logprobs when the provider exposes them (OpenAI, DeepSeek);
        # Anthropic has no logprobs API so it always falls back to the
        # cumulative empirical-frequency MLE (Eq. 6). get_probs_seed is
        # now ALWAYS built (not gated on n_seed_answers>0): it doubles as
        # the k=0 Direct-Query baseline that run_ppr_check needs
        # regardless of whether seeding is enabled.
        ppr_use_logprobs = api_cfg.get("use_logprobs", True)
        if provider == "openai":
            get_probs = _build_ppr_openai_provider(
                model_name=api_cfg.model_name,
                label_chars=label_chars,
                n_ppr_samples=n_ppr_samples,
                use_logprobs=ppr_use_logprobs,
                api=os.getenv("OPENAI_API_KEY"),
            )
            get_probs_seed = _build_openai_provider(
                model_name=api_cfg.model_name,
                label_chars=label_chars,
                use_logprobs=ppr_use_logprobs,
                n_api_samples=n_api_samples,
                api=os.getenv("OPENAI_API_KEY"),
            )
        elif provider == "anthropic":
            get_probs = _build_ppr_anthropic_provider(
                model_name=api_cfg.model_name,
                label_chars=label_chars,
                n_ppr_samples=n_ppr_samples,
                api=os.getenv("ANTHROPIC_API_KEY"),
            )
            get_probs_seed = _build_anthropic_provider(
                model_name=api_cfg.model_name,
                label_chars=label_chars,
                n_api_samples=n_api_samples,
                api=os.getenv("ANTHROPIC_API_KEY"),
            )
        elif provider == "deepseek":
            raw_log_path = osp.join(work_dir, f"raw_ppr_responses_{timestamp}.jsonl")
            get_probs = _build_ppr_deepseek_provider(
                model_name=api_cfg.model_name,
                label_chars=label_chars,
                n_ppr_samples=n_ppr_samples,
                use_logprobs=ppr_use_logprobs,
                api=os.getenv("DEEPSEEK_API_KEY"),
                raw_log_path=raw_log_path,
            )
            logger.info(f"Raw PPR responses logged to: {raw_log_path}")
            direct_query_log_path = osp.join(work_dir, f"raw_direct_query_responses_{timestamp}.jsonl")
            get_probs_seed = _build_deepseek_provider(
                model_name=api_cfg.model_name,
                label_chars=label_chars,
                use_logprobs=ppr_use_logprobs,
                n_api_samples=n_api_samples,
                api=os.getenv("DEEPSEEK_API_KEY"),
                raw_log_path=direct_query_log_path,
            )
            logger.info(f"Raw Direct-Query (k=0) responses logged to: {direct_query_log_path}")
        else:
            raise ValueError(f"Unknown api_model.provider '{provider}'.")
        logger.info(
            f"PPR provider: {provider} / {api_cfg.model_name}  "
            f"n_ppr_samples={n_ppr_samples}  n_seed_answers={args.n_seed_answers}  "
            f"use_logprobs={ppr_use_logprobs if provider != 'anthropic' else 'N/A (empirical frequency)'}"
        )

    elif args.mode == "sampling":
        if provider == "openai":
            get_probs = _build_openai_martingale_sampling_provider(
                model_name=api_cfg.model_name,
                label_chars=label_chars,
                use_logprobs=api_cfg.get("use_logprobs", False),
                n_api_samples=n_api_samples,
                api=os.getenv("OPENAI_API_KEY"),
                raw_log_path=osp.join(work_dir, f"raw_direct_query_responses_{timestamp}.jsonl"),
            )
        else:
            raise ValueError(f"Sampling mode is only supported for OpenAI provider, not '{provider}'.")

    else:
        if provider == "openai":
            get_probs = _build_openai_provider(
                model_name=api_cfg.model_name,
                label_chars=label_chars,
                use_logprobs=api_cfg.get("use_logprobs", False),
                n_api_samples=n_api_samples,
                api=os.getenv("OPENAI_API_KEY"),
                raw_log_path=osp.join(work_dir, f"raw_direct_query_responses_{timestamp}.jsonl"),
            )
        elif provider == "anthropic":
            get_probs = _build_anthropic_provider(
                model_name=api_cfg.model_name,
                label_chars=label_chars,
                n_api_samples=n_api_samples,
                api=os.getenv("ANTHROPIC_API_KEY"),
            )
        elif provider == "deepseek":
            get_probs = _build_deepseek_provider(
                model_name=api_cfg.model_name,
                label_chars=label_chars,
                use_logprobs=api_cfg.get("use_logprobs", False),
                n_api_samples=n_api_samples,
                api=os.getenv("DEEPSEEK_API_KEY"),
                raw_log_path=osp.join(work_dir, f"raw_direct_query_responses_{timestamp}.jsonl"),
            )
        elif provider == "huggingface":
            get_probs = _build_hf_provider(
                model_name=api_cfg.model_name,
                label_chars=label_chars,
                use_logprobs=api_cfg.get("use_logprobs", False),
                n_api_samples=n_api_samples,
                api=os.getenv("HUGGINGFACE_API_KEY"),
                raw_log_path=osp.join(work_dir, f"raw_direct_query_responses_{timestamp}.jsonl")
            )
        else:
            raise ValueError(f"Unknown api_model.provider '{provider}'.")
        logger.info(f"API provider: {provider} / {api_cfg.model_name}")
    # else:
    #     # Open-source path: load HuggingFace model + tokenizer
    #     batch_size = cfg.train_cfg.per_device_eval_batch_size
    #     tokenizer_run_cfg = dict(cfg.tokenizer_run_cfg)
    #     logger.info(f"Batch size: {batch_size}")
    #     logger.info(f"Tokenizer run config: {tokenizer_run_cfg}")

    #     # Warn if the config would attach an untrained PEFT adapter
    #     if cfg.model.get("use_peft") and cfg.model.get("peft_path") is None:
    #         print(
    #             "[martingale_property] WARNING: use_peft=True but peft_path is None. "
    #             "An untrained LoRA adapter will be attached. "
    #             "For zero-shot evaluation pass --cfg-options model.use_peft=False."
    #         )

    #     ## Loading the open source model
    #     model, tokenizer = get_model_and_tokenizer(**cfg.model, device=device)
    #     model.eval()

    #     ## Make the dataset(s) and concatenate if needed
    #     _split_names = ["train", "val", "test"] if args.split == "all" else [args.split]
    #     splits = [
    #         DATASETS.build(cfg.data[s], default_args=dict(tokenizer=tokenizer))
    #         for s in _split_names
    #     ]
    #     ## Get some HPs to use
    #     target_ids = splits[0].target_ids.to(device)
    #     dataset = ConcatDataset(splits) if len(splits) > 1 else splits[0]
    #     n_classes = target_ids.shape[0]
    #     label_chars = splits[0].label_chars
    #     n_ppr_samples = args.n_ppr_samples
    #     get_probs_seed = None

    #     ## Build the provider function for HuggingFace models
    #     if args.mode == "ppr":
    #         get_probs = _build_ppr_hf_provider(
    #             model, tokenizer, label_chars, n_ppr_samples, device
    #         )
    #         # Always built (not gated on n_seed_answers>0): doubles as the
    #         # k=0 Direct-Query baseline run_ppr_check needs regardless of
    #         # whether seeding is enabled.
    #         get_probs_seed = _build_hf_provider(
    #             model, tokenizer, target_ids, tokenizer_run_cfg, device
    #         )
    #         logger.info(
    #             f"PPR HuggingFace provider: {cfg.model.model_name_or_path}  "
    #             f"n_ppr_samples={n_ppr_samples}  n_seed_answers={args.n_seed_answers}  "
    #             f"theta_n from exact per-step logits"
    #         )

    #     else:
    #         get_probs = _build_hf_provider(model, tokenizer, target_ids, tokenizer_run_cfg, device)
    #         logger.info(f"HuggingFace provider: {cfg.model.model_name_or_path}")

    ## Information logging
    # For PPR mode, the paper's protocol has a single trajectory-length
    # parameter (n_ppr_samples = N in Section 4.1); --K is only meaningful
    # for iterative mode.
    effective_K = args.n_ppr_samples if args.mode == "ppr" else args.K
    logger.info(
        f"Dataset: {'+'.join(_split_names)}  "
        f"size={len(dataset)}  n_classes={n_classes}"
    )
    logger.info(f"Running {args.mode} check: K={effective_K} iterations ...")

    ## Run the related method check
    if args.mode == "ppr":
        result = run_ppr_check(
            get_theta_trajectory=get_probs,
            n_classes=n_classes,
            dataset=dataset,
            n_ppr_samples=args.n_ppr_samples,
            n_samples=args.n_samples,
            rng=rng,
            logger=logger,
            label_chars=label_chars,
            get_probs_seed=get_probs_seed,
            n_seed_answers=args.n_seed_answers,
            J=args.J,
        )

    elif args.mode == "sampling":
        result = run_martingale_sampling_check(
            get_probs=get_probs,
            n_classes=n_classes,
            dataset=dataset,
            K=args.K,
            n_samples=args.n_samples,
            batch_size=batch_size if batch_size else 1,
            rng=rng,
            logger=logger,
            label_chars=label_chars,
            J=args.J,
            seed =args.seed
        )
    else:
        result = run_martingale_check(
            get_probs=get_probs,
            n_classes=n_classes,
            dataset=dataset,
            K=args.K,
            n_samples=args.n_samples,
            batch_size=batch_size if batch_size else 1,
            rng=rng,
            logger=logger,
            label_chars=label_chars,
            J=args.J,
        )


    # compute_martingale_metrics expects (N, K+1, C); PPR/retrieval return (N, J, K+1, C).
    # For retrieval mode K=0, so drift metrics are not meaningful — only accuracy_at_p0 is used.
    dists_for_metrics = (
        result["distributions"][:, 0, :, :]   # (N, K+1, C)
        if result["distributions"].ndim == 4
        else result["distributions"]
    )
    metrics = compute_martingale_metrics(dists_for_metrics, result["true_labels"])

    # Log summary
    N = result["distributions"].shape[0]
    J_actual = result["distributions"].shape[1] if result["distributions"].ndim == 4 else 1
    logger.info("=" * 60)
    logger.info(f"Martingale Property Check ({args.mode.upper()}) — Summary")
    logger.info(f"  Questions evaluated : {N}")
    logger.info(f"  Trajectories (J)    : {J_actual}")
    logger.info(f"  Accuracy at p0      : {metrics['accuracy_at_p0']:.4f}")
    if args.mode != "retrieval":
        logger.info(f"  Iterations (K)      : {effective_K}")
        logger.info(f"  Mean TV drift       : {metrics['mean_tv']:.4f}  "
                    f"(0 = perfect martingale, 1 = maximum drift)")
        logger.info(f"  Pass rate TV < 0.05 : {metrics['pass_rate_tv_005']:.4f}")
        logger.info(f"  Pass rate TV < 0.10 : {metrics['pass_rate_tv_010']:.4f}")
        logger.info(f"  Mean L1 drift       : {metrics['mean_l1']:.4f}")
        logger.info(
            f"  Drift profile (k=1..{effective_K}): "
            + np.array2string(metrics["drift_profile"], precision=4, separator=", ")
        )

    ## Compute EMD and posterior metrics for iterative/PPR mode; posterior metrics for retrieval mode
    if args.mode in ("iterative", "ppr"):
        emd_metrics = compute_emd_metrics(result["distributions"])
        metrics.update(emd_metrics)
        logger.info(f"  EMD                 : {emd_metrics['emd']:.4f}")
        logger.info(
            f"  EMD profile (k=1..{effective_K}): "
            + np.array2string(emd_metrics["emd_profile"], precision=4, separator=", ")
        )
        if J_actual > 1:
            post_metrics = compute_martingale_posterior_metrics(result["distributions"])
            metrics.update({
                k: v for k, v in post_metrics.items()
                if np.isscalar(v) or isinstance(v, float)
            })
            logger.info(
                f"  Posterior entropy   : {post_metrics['mean_posterior_entropy']:.4f}"
            )
            logger.info(
                f"  Inter-traj variance : {post_metrics['mean_posterior_var']:.4f}"
            )

    logger.info("=" * 60)

    ## Saving the results
    out_path = save_martingale_results(
        work_dir=work_dir,
        seed=args.seed,
        K=args.K,
        distributions=result["distributions"],
        true_labels=result["true_labels"],
        data_indices=result["data_indices"],
        input_texts=result["input_texts"],
        prompt_history=result["prompt_history"],
        metrics=metrics,
        logger=logger
    )

    print(f"\n[martingale_property] Completed.")
    print(f"  Output              : {out_path}")
    print(f"  Accuracy at p0      : {metrics['accuracy_at_p0']:.4f}")
    print(f"  Mean TV drift       : {metrics['mean_tv']:.4f}")
    print(f"  Pass rate (TV<0.05) : {metrics['pass_rate_tv_005']:.4f}")
    print(f"  Pass rate (TV<0.10) : {metrics['pass_rate_tv_010']:.4f}")


if __name__ == "__main__":
    main()
