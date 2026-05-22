"""Empirical inference MACs/FLOPs analysis using ptflops.

Loads actual model weights and measures MACs via a real forward pass,
unlike compute_flops.py which uses analytical formulas.

Note on Laplace LoRA: ptflops measures one forward pass. The Jacobian
overhead (C backward passes, each ≈ 2× forward) cannot be captured by
ptflops and is instead reported as a theoretical multiplier on top of the
measured forward MACs.

Usage
-----
# Single model
python tools/compute_flops_empirical.py \\
    --config_base configs/arc_c_qwen2_7b/lora_arc_c_qwen2_7b.yaml

# Duo (loads both models sequentially, measures each)
python tools/compute_flops_empirical.py \\
    --config_base    configs/arc_c_qwen2_7b/lora_arc_c_qwen2_7b.yaml \\
    --config_sidekick configs/arc_c_qwen2_15b/lora_arc_c_qwen2_15b.yaml

Requirements: pip install ptflops
"""

import argparse
import inspect
import re
from typing import Optional

import mmengine
import torch
from ptflops import get_model_complexity_info

from asym_duos import DATASETS, get_model_and_tokenizer


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Empirical MACs/FLOPs analysis for Asymmetric LLM Duo methods."
    )
    p.add_argument("--config-base", required=True,
                   help="Any config YAML for the base model (lora_ and ib_ variants are "
                        "equivalent — only model_name_or_path, peft_cfg, and dataset type "
                        "are used). E.g. configs/arc_c_qwen2_7b/lora_arc_c_qwen2_7b.yaml")
    p.add_argument("--config-sidekick", default=None,
                   help="Any config YAML for the sidekick model (same rules as "
                        "--config-base). Must share the same dataset as the base config. "
                        "E.g. configs/arc_c_qwen2_15b/lora_arc_c_qwen2_15b.yaml")
    p.add_argument("--gpu-id", type=int, default=0)
    p.add_argument("--seq-len", type=int, default=None,
                   help="Sequence length override (default: max_length from config).")
    p.add_argument("--laplace-samples", type=int, default=100_000,
                   help="MC samples used in Laplace posterior predictive. Default: 100000.")
    p.add_argument("--blob-samples", type=int, default=10,
                   help="MC weight samples used in BLoB posterior predictive "
                        "(--bayes-eval-n-samples-final in bayesian-peft). Default: 10.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# ptflops wrapper
# ---------------------------------------------------------------------------

class _MacsWrapper(torch.nn.Module):
    """Wrap a CausalLM model to expose a simple forward for ptflops."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids):
        return self.model(input_ids=input_ids, attention_mask=None)["logits"]


def estimate_macs_per_sample(model, vocab_size: int, seq_len: int, device: str) -> float:
    """Measure MACs for one forward pass using ptflops.

    Args:
        model: Loaded HuggingFace CausalLM (with or without LoRA adapters).
        vocab_size: Vocabulary size for synthetic input_ids generation.
        seq_len: Sequence length to profile.
        device: CUDA device string, e.g. 'cuda:0'.

    Returns:
        MACs as a float (multiply-accumulate operations; 1 MAC ≈ 2 FLOPs).
    """
    wrapper = _MacsWrapper(model).to(device)
    wrapper.eval()

    def input_constructor(_):
        return {"input_ids": torch.randint(0, vocab_size, (1, seq_len), device=device)}

    macs, _ = get_model_complexity_info(
        wrapper,
        (seq_len,),
        as_strings=False,
        print_per_layer_stat=False,
        verbose=False,
        input_constructor=input_constructor,
    )
    return float(macs)


# ---------------------------------------------------------------------------
# Parameter breakdown
# ---------------------------------------------------------------------------

def summarize_param_breakdown(model):
    """Summarize total and trainable parameters split into LoRA vs backbone.

    Returns:
        dict with keys: lora_total, lora_trainable, backbone_total,
        backbone_trainable, grand_total, grand_trainable.
    """
    buckets = [
        ("lora",     lambda n: "lora" in n),
        ("backbone", lambda n: True),
    ]

    total     = {name: 0 for name, _ in buckets}
    trainable = {name: 0 for name, _ in buckets}

    for name, p in model.named_parameters():
        bucket = next(b for b, fn in buckets if fn(name))
        total[bucket]     += p.numel()
        if p.requires_grad:
            trainable[bucket] += p.numel()

    grand_total     = sum(total.values())
    grand_trainable = sum(trainable.values())
    return {
        "lora_total":        total["lora"],
        "lora_trainable":    trainable["lora"],
        "backbone_total":    total["backbone"],
        "backbone_trainable":trainable["backbone"],
        "grand_total":       grand_total,
        "grand_trainable":   grand_trainable,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_num_classes(cfg) -> int:
    dataset_type = cfg.data.train.type
    cls = DATASETS.get(dataset_type)
    if cls is None:
        raise ValueError(f"Dataset type '{dataset_type}' not found in DATASETS registry.")
    src = inspect.getsource(cls.__init__)
    m = re.search(r'n_labels\s*=\s*(\d+)', src)
    if m is None:
        raise ValueError(f"Could not find 'n_labels=<int>' in {dataset_type}.__init__ source.")
    return int(m.group(1))


def _get_seq_len(cfg, override: Optional[int]) -> int:
    if override is not None:
        return override
    tok_run_cfg = cfg.get("tokenizer_run_cfg", {})
    return tok_run_cfg.get("max_length", 512)


def _human_macs(n: float) -> str:
    if n >= 1e12:
        return f"{n/1e12:.3f} TMACs"
    if n >= 1e9:
        return f"{n/1e9:.3f} GMACs"
    if n >= 1e6:
        return f"{n/1e6:.3f} MMACs"
    return f"{n:,.0f} MACs"


def _human_flops(n: float) -> str:
    if n >= 1e12:
        return f"{n/1e12:.3f} TFLOPs"
    if n >= 1e9:
        return f"{n/1e9:.3f} GFLOPs"
    if n >= 1e6:
        return f"{n/1e6:.3f} MFLOPs"
    return f"{n:,.0f} FLOPs"


def _human_params(n: int) -> str:
    if n >= 1e9:
        return f"{n/1e9:.3f}B"
    if n >= 1e6:
        return f"{n/1e6:.3f}M"
    return f"{n:,}"


# ---------------------------------------------------------------------------
# Per-model profiling
# ---------------------------------------------------------------------------

def profile_model(label: str, cfg_path: str, device: torch.device,
                  seq_len_override: Optional[int]) -> dict:
    """Load model, measure MACs and parameter breakdown.

    Returns a dict with keys: label, macs, params, seq_len, num_classes, vocab_size.
    """
    cfg = mmengine.Config.fromfile(cfg_path)
    seq_len     = _get_seq_len(cfg, seq_len_override)
    num_classes = _get_num_classes(cfg)

    print(f"\n[{label}] Loading model '{cfg.model.model_name_or_path}' ...")
    model, tokenizer = get_model_and_tokenizer(**cfg.model, device=device)
    model.eval()

    vocab_size = tokenizer.vocab_size or model.config.vocab_size
    param_info = summarize_param_breakdown(model)

    print(f"[{label}] Measuring MACs (seq_len={seq_len}) ...")
    macs = estimate_macs_per_sample(model, vocab_size, seq_len, str(device))

    del model
    torch.cuda.empty_cache()

    return {
        "label":       label,
        "macs":        macs,
        "params":      param_info,
        "seq_len":     seq_len,
        "num_classes": num_classes,
        "vocab_size":  vocab_size,
    }


# ---------------------------------------------------------------------------
# Method table
# ---------------------------------------------------------------------------

def build_method_table(base: dict, sidekick: Optional[dict],
                       n_laplace_samples: int,
                       n_blob_samples: int = 10) -> list[dict]:
    """Compute MACs per inference sample for each method.

    Laplace Jacobian cost is theoretical (C backward passes ≈ 2C × forward MACs)
    since ptflops cannot capture dynamic autograd loops.
    """
    C    = base["num_classes"]
    b_fw = base["macs"]

    rows = []

    def _laplace_macs(forward_macs):
        jacobian = (1 + 2 * C) * forward_macs   # forward + C backward passes
        sampling = n_laplace_samples * 3 * C * C  # L @ eps + softmax, per sample
        return jacobian + sampling

    # --- Base single-model ---
    rows.append(dict(method="lora_base",          macs=b_fw,
                     note="Single forward pass (measured)"))
    rows.append(dict(method="lora_ts_base",        macs=b_fw,
                     note="Same; TS scalar division is O(C), negligible"))
    rows.append(dict(method="ib_edl_base",         macs=b_fw,
                     note="Same; Dirichlet normalisation is O(C), negligible"))
    rows.append(dict(method="laplace_lora_base",   macs=_laplace_macs(b_fw),
                     note=f"Measured forward + theoretical {C} backward passes + {n_laplace_samples:,} MC samples"))
    rows.append(dict(method="blob_base",           macs=n_blob_samples * b_fw,
                     note=f"{n_blob_samples} stochastic weight-sample forward passes (measured each)"))

    if sidekick is None:
        return rows

    s_fw = sidekick["macs"]

    # --- Sidekick single-model ---
    rows.append(dict(method="lora_sidekick",          macs=s_fw,
                     note="Single forward pass (measured)"))
    rows.append(dict(method="lora_ts_sidekick",        macs=s_fw,
                     note="Same; TS scalar division is O(C), negligible"))
    rows.append(dict(method="ib_edl_sidekick",         macs=s_fw,
                     note="Same; Dirichlet normalisation is O(C), negligible"))
    rows.append(dict(method="laplace_lora_sidekick",   macs=_laplace_macs(s_fw),
                     note=f"Measured forward + theoretical {C} backward passes + {n_laplace_samples:,} MC samples"))
    rows.append(dict(method="blob_sidekick",           macs=n_blob_samples * s_fw,
                     note=f"{n_blob_samples} stochastic weight-sample forward passes (measured each)"))

    # --- Duo ---
    duo = b_fw + s_fw
    rows.append(dict(method="lora_duo",          macs=duo,
                     note="Base + sidekick forward passes (both measured)"))
    rows.append(dict(method="lora_duo_ts",        macs=duo,
                     note="Same; TS applied before combination"))
    rows.append(dict(method="ib_edl_duo",         macs=duo,
                     note="Same; Dirichlet on both sides"))
    rows.append(dict(method="laplace_lora_duo",   macs=_laplace_macs(b_fw) + _laplace_macs(s_fw),
                     note="Laplace(base) + Laplace(sidekick)"))
    rows.append(dict(method="blob_duo",           macs=n_blob_samples * (b_fw + s_fw),
                     note=f"{n_blob_samples} weight samples × (base + sidekick) forward passes"))

    return rows


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_report(rows: list[dict], base: dict, sidekick: Optional[dict]) -> None:
    print("\n" + "=" * 95)
    print("EMPIRICAL INFERENCE MACs ANALYSIS — Asymmetric LLM Duo Methods")
    print("=" * 95)

    for info, label in [(base, "BASE"), (sidekick, "SIDEKICK")]:
        if info is None:
            continue
        p = info["params"]
        print(f"\n  [{label}]")
        print(f"    Total params:     {_human_params(p['grand_total'])}  "
              f"(trainable: {_human_params(p['grand_trainable'])})")
        print(f"    LoRA params:      {_human_params(p['lora_total'])}  "
              f"(trainable: {_human_params(p['lora_trainable'])})")
        print(f"    Backbone params:  {_human_params(p['backbone_total'])}")
        print(f"    Seq len: {info['seq_len']}  |  Num classes: {info['num_classes']}  "
              f"|  Vocab size: {info['vocab_size']:,}")
        print(f"    Forward pass MACs: {_human_macs(info['macs'])}"
              f"  ({_human_flops(info['macs'] * 2)})")

    base_macs    = rows[0]["macs"]   # lora_base
    side_macs    = next((r["macs"] for r in rows if r["method"] == "lora_sidekick"), None)

    print(f"\n{'Method':<26} {'MACs/sample':>16} {'FLOPs/sample':>16} {'Relative':>10}  Note")
    print("-" * 95)
    for row in rows:
        m      = row["macs"]
        is_side = "sidekick" in row["method"] and "duo" not in row["method"]
        ref    = side_macs if (is_side and side_macs) else base_macs
        rel    = m / ref
        ref_label = "(vs sidekick)" if (is_side and side_macs) else "(vs base)   "
        print(f"  {row['method']:<24} {_human_macs(m):>16} {_human_flops(m*2):>16}"
              f"  {rel:>6.2f}x {ref_label}  {row['note']}")

    print("\n" + "=" * 95)
    print("Notes:")
    print("  - MACs measured via ptflops on a real forward pass with synthetic input_ids.")
    print("  - 1 MAC = 1 multiply + 1 accumulate ≈ 2 FLOPs.")
    print("  - Laplace Jacobian cost is theoretical: (1+2C)×forward_MACs "
          "(C backward passes, each ≈ 2× forward).")
    print("  - ptflops may undercount ops inside flash-attention or custom CUDA kernels.")
    print("=" * 95)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args  = parse_args()
    device = torch.device(f"cuda:{args.gpu_id}")

    base = profile_model("base", args.config_base, device, args.seq_len)

    sidekick = None
    if args.config_sidekick:
        sidekick = profile_model("sidekick", args.config_sidekick, device, args.seq_len)

    rows = build_method_table(base, sidekick, n_laplace_samples=args.laplace_samples,
                              n_blob_samples=args.blob_samples)
    print_report(rows, base, sidekick)


if __name__ == "__main__":
    main()
