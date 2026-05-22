"""Post-process BLoB asymmetric-duos predictions produced by the
``bayesian-peft`` fork (see ``scripts/asymmetric_duos/*.sbatch`` over there).

The fork dumps per-seed predictions in a *flat* layout under ``--results-root``:

    $RESULTS_ROOT/blob/{base,sidekick}/{val_preds,test_preds}/<dataset>.npz

This script:
  1. For each seed in ``--seeds``, calls ``optimize_weights_blob`` in
     unconstrained mode. That writes:
        $RESULTS_ROOT/duo_blob/{val_preds,test_preds}/<dataset>.npz
        $RESULTS_ROOT/metrics/<dataset>.npz
  2. Reuses the helpers from ``tools.average_metrics`` (directly, bypassing
     its nested ``_build_work_dir``) to produce:
        $RESULTS_ROOT/overall_metrics_summary_blob/<dataset>/metrics_summary_sd.csv
     plus a printed mean ± 1 SD test-set table, with deep ensemble (size 2)
     over same-model seed pairs for base_blob and sidekick_blob.

Usage
-----
    python tools/postprocess_blob_asymduos.py \
        --results-root $HOME/results \
        --datasets arc_c arc_e obqa csqa race sciq \
        --seeds 1 2 3 4 5 \
        --deep-ensemble-size 2
"""
from __future__ import annotations

import argparse
import os
import os.path as osp
import sys

import numpy as np
import pandas as pd

# Make sibling modules in tools/ importable when this file is invoked as
# ``python tools/postprocess_blob_asymduos.py`` from the repo root (no
# ``tools/__init__.py``).
sys.path.insert(0, osp.dirname(osp.abspath(__file__)))

from asym_duos.utils.duo_optimizer import optimize_weights_blob

# Reuse helpers from average_metrics so behaviour matches the standard pipeline.
from average_metrics import (  # type: ignore
    _create_metrics_df,
    _create_predictions_df,
    _deep_ensemble_metrics,
    _format_sd,
    _sample_combinations,
    _summarize,
    _build_metrics_from_preds,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Post-process BLoB asymmetric-duos predictions (flat layout).')
    p.add_argument('--results-root', required=True,
                   help='Root directory containing blob/{base,sidekick}/... .')
    p.add_argument('--datasets', nargs='+', required=True,
                   help='Dataset stems, e.g. arc_c arc_e obqa csqa race sciq.')
    p.add_argument('--seeds', nargs='+', type=int, default=[1, 2, 3, 4, 5])
    p.add_argument('--deep-ensemble-size', type=int, default=2)
    p.add_argument('--skip-duo', action='store_true',
                   help='Skip optimize_weights_blob (useful to re-aggregate only).')
    return p.parse_args()


def _collapse_mc_dim_inplace(npz_path: str) -> bool:
    """Collapse per-MC-sample logits ``(N, S, C)`` → Bayesian-averaged
    log-probs ``(N, C)``, in place. Idempotent: 2D inputs are left alone.

    The bayesian-peft fork dumps shape ``(N, S, C)`` because eval uses
    ``--bayes-eval-n-samples-final 10``, but ``optimize_weights_blob`` and
    the downstream sklearn metrics expect 2D log-probs. The Bayesian
    predictive is ``E_q[p(y|x, A)] ≈ mean_s softmax(logits_s)``; we then
    take ``log`` to keep the npz schema in log-prob space.
    """
    raw = dict(np.load(npz_path, allow_pickle=True))
    changed = False
    for seed_key in list(raw.keys()):
        sd = raw[seed_key].item()
        lo = sd['logits']
        if lo.ndim == 2:
            continue
        if lo.ndim != 3:
            raise ValueError(
                f'{npz_path}[seed={seed_key}]: unexpected logits shape {lo.shape}')
        # softmax over classes (numerically stable)
        x = lo - lo.max(axis=-1, keepdims=True)
        probs = np.exp(x)
        probs /= probs.sum(axis=-1, keepdims=True)
        # MC-average across the sample dim, then back to log-probs
        mean_probs = probs.mean(axis=1)
        mean_probs = np.clip(mean_probs, 1e-12, 1.0)
        sd['logits'] = np.log(mean_probs).astype(np.float32)
        raw[seed_key] = np.array(sd)
        changed = True
    if changed:
        np.savez_compressed(npz_path, **raw)
    return changed


def _collapse_all_blob_npzs(results_root: str, dataset: str) -> None:
    npz_file = f'{dataset}.npz'
    for role in ('base', 'sidekick'):
        for split in ('val_preds', 'test_preds'):
            p = osp.join(results_root, 'blob', role, split, npz_file)
            if not osp.exists(p):
                print(f'[collapse] missing: {p}')
                continue
            if _collapse_mc_dim_inplace(p):
                print(f'[collapse] {p}: collapsed (N,S,C) → (N,C)')
            else:
                print(f'[collapse] {p}: already 2D, skipping')


def _process_dataset(results_root: str, dataset: str, seeds: list[int],
                     de_size: int, skip_duo: bool) -> None:
    npz_file = f'{dataset}.npz'
    print('\n' + '=' * 70)
    print(f'[postprocess] dataset={dataset}')
    print('=' * 70)

    # --- Stage 0: collapse (N, S, C) per-MC-sample logits → (N, C) ----------
    # The bayesian-peft fork dumps per-sample logits; the duo optimizer and
    # sklearn metrics expect 2D log-probs. Idempotent — safe to re-run.
    _collapse_all_blob_npzs(results_root, dataset)

    # --- Stage 1: per-seed unconstrained duo optimisation -------------------
    if not skip_duo:
        for seed in seeds:
            print(f'\n--- optimize_weights_blob(seed={seed}, unconstrained) ---')
            optimize_weights_blob(
                result_dir=results_root,
                saved_file_name=npz_file,
                seed=seed,
                constrained=False,
                verbose=False,
            )

    # --- Stage 2: load metrics/<dataset>.npz and aggregate ------------------
    metrics_path = osp.join(results_root, 'metrics', npz_file)
    summary_dir = osp.join(results_root, 'overall_metrics_summary_blob', dataset)
    os.makedirs(summary_dir, exist_ok=True)

    # Ensure base/sidekick per-seed metrics also exist (duo step only writes
    # duo_blob_* keys). Merge in metrics computed from the raw preds.
    raw_base_side = _build_metrics_from_preds(
        results_root, npz_file, ['base', 'sidekick'], blob=True)
    metrics_data: dict = {}
    if osp.exists(metrics_path):
        metrics_data = dict(np.load(metrics_path, allow_pickle=True))
    # Merge: duo-written metrics have duo_blob_* in val/test; raw pass has
    # base_* / sidekick_* keys. Combine them per seed.
    merged: dict = {}
    all_seeds = set(metrics_data.keys()) | set(raw_base_side.keys())
    for seed_key in all_seeds:
        duo_d = metrics_data.get(seed_key)
        raw_d = raw_base_side.get(seed_key)
        duo_d = duo_d.item() if duo_d is not None else {'val': {}, 'test': {}, 'data': npz_file}
        raw_d = raw_d.item() if raw_d is not None else {'val': {}, 'test': {}, 'data': npz_file}
        out = {
            'data': npz_file,
            'val': {**raw_d.get('val', {}), **duo_d.get('val', {})},
            'test': {**raw_d.get('test', {}), **duo_d.get('test', {})},
        }
        merged[seed_key] = np.array(out)

    # Discover model types present from *_nll keys in val
    first_val = merged[next(iter(merged))].item()['val']
    model_types = sorted({k.rsplit('_', 1)[0] for k in first_val if k.endswith('_nll')})
    print(f'[postprocess] Model types present: {model_types}')

    metrics_df = _create_metrics_df(
        merged, ['val', 'test'], model_types,
        output_path=osp.join(summary_dir, 'metrics.npz'))

    summaries = []
    for mt in model_types:
        for split in ('val', 'test'):
            summaries.append(_summarize(metrics_df, mt, split))

    # --- Stage 3: deep ensemble (size m) over same-model seed pairs ---------
    m = de_size
    rng = np.random.default_rng(42)
    preds_root = osp.join(results_root, 'blob')

    base_tst_path = osp.join(preds_root, 'base', 'test_preds', npz_file)
    base_preds_df = _create_predictions_df(
        dict(np.load(base_tst_path, allow_pickle=True)),
        model_type='base_blob', output_path=base_tst_path)

    seeds_str = list(base_preds_df['seed'].unique())
    combos = _sample_combinations(seeds_str, m, n_samples=len(seeds_str), rng=rng)
    print(f'[postprocess] Deep ensemble (size {m}) combos: {combos}')

    base_de = _deep_ensemble_metrics(base_preds_df, combos, 'base_blob')
    base_de['model'] = f'base_blob_de({m})'
    base_de['split'] = 'test'

    side_tst_path = osp.join(preds_root, 'sidekick', 'test_preds', npz_file)
    side_preds_df = _create_predictions_df(
        dict(np.load(side_tst_path, allow_pickle=True)),
        model_type='sidekick_blob', output_path=side_tst_path)
    side_de = _deep_ensemble_metrics(side_preds_df, combos, 'sidekick_blob')
    side_de['model'] = f'sidekick_blob_de({m})'
    side_de['split'] = 'test'

    de_data = pd.concat([base_de, side_de], ignore_index=True)
    de_model_types = [f'base_blob_de({m})', f'sidekick_blob_de({m})']
    for mt in de_model_types:
        if mt in de_data['model'].values:
            summaries.append(_summarize(de_data, mt, 'test'))

    # --- Stage 4: write summary CSVs + pretty test table --------------------
    summary_df = pd.concat(summaries, ignore_index=True)
    summary_df.to_csv(osp.join(summary_dir, 'metrics_summary_sd_raw.csv'), index=False)

    metric_order = ['accuracy', 'f1_macro', 'nll', 'brier', 'ece', 'lppd', 'mean_uncertainty']
    all_model_types = model_types + de_model_types

    pretty = summary_df.copy()
    pretty['value'] = pretty.apply(lambda r: _format_sd(r['mean'], r['sd']), axis=1)
    pretty['metric'] = pd.Categorical(pretty['metric'], categories=metric_order, ordered=True)
    pretty['model'] = pd.Categorical(pretty['model'], categories=all_model_types, ordered=True)
    pretty['split'] = pd.Categorical(pretty['split'], categories=['val', 'test'], ordered=True)
    pretty = pretty.sort_values(['model', 'split', 'metric'])

    pretty_wide = pretty.pivot_table(
        index=['model', 'split'], columns='metric',
        values='value', aggfunc='first',
    ).reindex(columns=metric_order).reset_index()

    out_csv = osp.join(summary_dir, 'metrics_summary_sd.csv')
    pretty_wide.to_csv(out_csv, index=False)

    print(f'\n=== [{dataset}] Metrics Summary (mean ± 1 SD) — test set ===')
    test_rows = pretty_wide[pretty_wide['split'] == 'test']
    cols = list(test_rows.columns)
    rows = [list(map(str, r)) for r in test_rows.itertuples(index=False, name=None)]
    widths = [max(len(c), max((len(r[i]) for r in rows), default=0))
              for i, c in enumerate(cols)]
    print(' | '.join(c.ljust(widths[i]) for i, c in enumerate(cols)))
    print('-+-'.join('-' * w for w in widths))
    for row in rows:
        print(' | '.join(row[i].ljust(widths[i]) for i in range(len(cols))))
    print(f'\n[{dataset}] Saved summary to {out_csv}')


def main() -> None:
    args = parse_args()
    for dataset in args.datasets:
        _process_dataset(
            results_root=args.results_root,
            dataset=dataset,
            seeds=args.seeds,
            de_size=args.deep_ensemble_size,
            skip_duo=args.skip_duo,
        )
    print('\n[postprocess] All datasets complete.')


if __name__ == '__main__':
    main()
