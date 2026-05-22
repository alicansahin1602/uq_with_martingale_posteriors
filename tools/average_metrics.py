"""Aggregate per-seed metrics and compute deep-ensemble results.

Usage
-----
python tools/average_metrics.py \\
    --config_base configs/arc_c_qwen2_7b/lora_arc_c_qwen2_7b.yaml \\
    --config_sidekick configs/arc_c_qwen2_15b/lora_arc_c_qwen2_15b.yaml \\
    --work-dir data/results \\
    --dataset_name arc_c \\
    --duo --deep-ensemble --deep-ensemble-size 2

Flags mirror those used during training so the script can reconstruct the
correct output path automatically.
"""

import os
import os.path as osp
from argparse import ArgumentParser
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from asym_duos import _softmax_from_log, compute_uncertainty_metrics


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = ArgumentParser('Aggregate metrics across seeds.')
    p.add_argument('--config-base', required=True)
    p.add_argument('--config-sidekick', default=None)
    p.add_argument('--work-dir', '-w', required=True,
                   help='Root result directory (same as used in train.py).')
    p.add_argument('--dataset-name', '-d', required=True,
                   help='Dataset name matching the npz file stem, e.g. arc_c.')

    # method flags — must match what was used during training
    p.add_argument('--ib-edl', action='store_true')
    p.add_argument('--blob', action='store_true',
                   help='BLoB run: read predictions from blob/{model_type}/ subdirs.')
    p.add_argument('--duo', action='store_true')
    p.add_argument('--duo-constrained', action='store_true')
    p.add_argument('--temperature-scaling', action='store_true')
    p.add_argument('--laplace-lora', action='store_true',
                   help='Run Laplace LoRA after training.')
    p.add_argument('--deep-ensemble', action='store_true',
                   help='Compute deep ensemble metrics from same-model seed pairs.')
    p.add_argument('--deep-ensemble-size', type=int, default=2,
                   help='Number of models in each deep ensemble (default: 2).')
    p.add_argument('--beta-base', type=float, default=None,
                   help='IB-EDL beta-base override used during training. If set '
                        'here, it must match the value used in train.py so the '
                        'correct work_dir can be reconstructed.')
    p.add_argument('--beta-sidekick', type=float, default=None,
                   help='IB-EDL beta-sidekick override used during training.')
    return p.parse_args()


# ---------------------------------------------------------------------------
# Path reconstruction (mirrors train.py logic)
# ---------------------------------------------------------------------------

def _model_tag(cfg_path):
    """Parse dataset and model names from a config folder path.

    Works for both two-word dataset names (arc_c, arc_e) and single-word
    names (csqa, obqa). Model names always start with a known family prefix.
    """
    folder = cfg_path.split('/')[1]
    parts = folder.split('_')
    known_model_starts = ('qwen2', 'qwen3', 'llama', 'mistral', 'gemma')
    split_idx = next(
        (i for i, p in enumerate(parts) if p in known_model_starts),
        2,  # fallback: old behaviour
    )
    dataset = '_'.join(parts[:split_idx])
    model = '_'.join(parts[split_idx:])
    return dataset, model


def _build_work_dir(root, cfg_base, cfg_sidekick, args):
    dataset, base_model = _model_tag(cfg_base)
    has_sidekick = cfg_sidekick is not None
    model_tag = base_model
    if has_sidekick:
        _, side_model = _model_tag(cfg_sidekick)
        model_tag = f'{base_model}_{side_model}'

    method_parts = ['ib_edl' if args.ib_edl else 'lora']
    if getattr(args, 'blob', False):
        method_parts.append('blob')
    if args.beta_base is not None:
        method_parts.append(f'beta_base_{args.beta_base}')
    if args.beta_sidekick is not None:
        method_parts.append(f'beta_side_{args.beta_sidekick}')
    if args.temperature_scaling:
        method_parts.append('tempscaling')
    if args.duo:
        method_parts.append('duo_constrained' if args.duo_constrained else 'duo_unconstrained')
    if args.laplace_lora:
        method_parts.append('laplace_lora')
    method = '_'.join(method_parts)
    return osp.join(root, dataset, model_tag, method)


# ---------------------------------------------------------------------------
# Statistics: mean ± 1 SD
# ---------------------------------------------------------------------------

def _mean_sd(values: np.ndarray):
    values = np.asarray(values, dtype=float)
    n = values.size
    mean = float(values.mean()) if n else float('nan')
    sd = float(values.std(ddof=1)) if n >= 2 else float('nan')
    return mean, sd, n


def _format_sd(mean: float, sd: float) -> str:
    if np.isnan(mean):
        return 'nan'
    if np.isnan(sd):
        return f'{mean:.4f}'
    return f'{mean:.4f} ± {sd:.4f}'


# ---------------------------------------------------------------------------
# Summary builders
# ---------------------------------------------------------------------------

def _summarize(data: pd.DataFrame, model_type: str, split: str) -> pd.DataFrame:
    subset = data[(data['model'] == model_type) & (data['split'] == split)]
    metric_cols = [c for c in subset.columns if c not in ('model', 'seed', 'split', 'data')]
    rows = []
    for col in metric_cols:
        mean, sd, n = _mean_sd(subset[col].values)
        rows.append({'model': model_type, 'split': split, 'metric': col,
                     'mean': mean, 'sd': sd, 'n': n})
    return pd.DataFrame(rows)


def _create_metrics_df(data: dict, splits: list, model_types: list,
                       output_path: str) -> pd.DataFrame:
    """Flatten the per-seed NPZ metrics dict into a tidy DataFrame."""
    first = data[next(iter(data))].item()

    # Discover metric names from the stored dict.
    # Keys are stored as '{model_type}_{metric}', e.g. 'base_nll', 'duo_brier'.
    # Use the LONGEST matching model_type prefix to avoid 'base_' accidentally
    # matching 'base_ts_nll' (which belongs to model_type 'base_ts').
    # Sort model_types by length descending so longer prefixes are tried first.
    metric_names: set = set()
    sorted_mts = sorted(model_types or ['duo'], key=len, reverse=True)
    for k in first['val']:
        for mt in sorted_mts:
            prefix = f'{mt}_'
            if k.startswith(prefix) and not any(
                k.startswith(f'{other}_') and len(other) > len(mt)
                for other in sorted_mts
            ):
                metric_names.add(k[len(prefix):])
                break
    metrics = sorted(metric_names)

    rows = []
    for seed in data.keys():
        for m_type in model_types:
            for split in splits:
                row = {'seed': seed, 'data': first['data'].split('.')[0],
                       'model': m_type, 'split': split}
                for metric in metrics:
                    col_name = f'{m_type}_{metric}'
                    row[metric] = data[seed].item()[split].get(col_name, float('nan'))
                rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_path.replace('.npz', '.csv'), index=False)
    return df


def _create_predictions_df(data: dict, model_type: str, output_path: str) -> pd.DataFrame:
    """Flatten per-seed prediction NPZ into a tidy DataFrame."""
    seeds = list(data.keys())
    n_labels = data[seeds[0]].item()['logits'].shape[1]
    rows = []
    for seed in seeds:
        sd = data[seed].item()
        probs = _softmax_from_log(sd['logits'])
        n = len(sd['idx'])
        for i in range(n):
            row = {
                'seed': seed, 'model_type': model_type,
                'idx': int(sd['idx'][i]),
                'input': sd['input'][i],
                'true_label': int(sd['true_labels'][i]),
                'predicted_label': int(probs[i].argmax()),
            }
            for c in range(n_labels):
                row[f'prob_label_{c}'] = float(probs[i, c])
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(output_path.replace('.npz', '.csv'), index=False)
    return df


# ---------------------------------------------------------------------------
# Deep ensemble
# ---------------------------------------------------------------------------

def _sample_combinations(seeds, m: int, n_samples: int, rng) -> list:
    """Sample *n_samples* distinct size-*m* seed combinations."""
    all_combos = list(combinations(seeds, m))
    n_samples = min(n_samples, len(all_combos))
    chosen = rng.choice(len(all_combos), size=n_samples, replace=False)
    return [all_combos[i] for i in chosen]


def _deep_ensemble_metrics(pred_df: pd.DataFrame, combos: list,
                            model_type: str) -> pd.DataFrame:
    """Average probabilities over *m* same-model seeds and compute metrics."""
    prob_cols = [c for c in pred_df.columns if c.startswith('prob_label_')]
    rows = []
    for combo in combos:
        parts = [pred_df[pred_df['seed'] == s].sort_values('idx').reset_index(drop=True)
                 for s in combo]
        avg_probs = np.mean([p[prob_cols].values for p in parts], axis=0).astype(np.float32)
        preds = avg_probs.argmax(axis=1)
        y_true = parts[0]['true_label'].values

        combo_key = '_'.join(str(s) for s in combo)
        row = {'seed': combo_key,
               'accuracy': accuracy_score(y_true, preds),
               'f1_macro': f1_score(y_true, preds, average='macro')}
        row.update(compute_uncertainty_metrics(avg_probs, y_true))
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Fallback: build metrics/csqa.npz from raw prediction files
# ---------------------------------------------------------------------------

def _build_metrics_from_preds(work_dir: str, npz_file: str,
                               model_types_present: list,
                               blob: bool = False) -> dict:
    """Compute per-seed metrics from raw val/test prediction NPZs.

    Used when no duo optimisation or temperature scaling was run, so
    metrics/<npz_file> was never written by those steps.  Produces the
    same nested dict structure that duo_optimizer / temperature_scaling
    would have written, so the rest of main() can proceed unchanged.

    When *blob* is True, predictions are read from
    ``{work_dir}/blob/{model_type}/`` instead of ``{work_dir}/{model_type}/``.
    """
    from sklearn.metrics import accuracy_score, f1_score
    from asym_duos import compute_uncertainty_metrics

    def _preds_root(mt: str) -> str:
        return osp.join(work_dir, 'blob', mt) if blob else osp.join(work_dir, mt)

    # Discover seeds from the base model's val predictions
    base_val_path = osp.join(_preds_root('base'), 'val_preds', npz_file)
    base_val_raw = dict(np.load(base_val_path, allow_pickle=True))
    seeds = list(base_val_raw.keys())

    metrics_data = {}
    for seed in seeds:
        val_metrics: dict = {}
        tst_metrics: dict = {}

        for mt in model_types_present:
            for split_label, preds_dir in [('val', 'val_preds'), ('tst', 'test_preds')]:
                path = osp.join(_preds_root(mt), preds_dir, npz_file)
                raw = dict(np.load(path, allow_pickle=True))
                sd = raw[seed].item()
                probs = _softmax_from_log(sd['logits'])
                y_true = sd['true_labels']
                preds = probs.argmax(axis=1)
                m: dict = {
                    f'{mt}_accuracy': float(accuracy_score(y_true, preds)),
                    f'{mt}_f1_macro': float(f1_score(y_true, preds, average='macro')),
                }
                m.update({
                    f'{mt}_{k}': v
                    for k, v in compute_uncertainty_metrics(probs, y_true).items()
                })
                if split_label == 'val':
                    val_metrics.update(m)
                else:
                    tst_metrics.update(m)

        metrics_data[seed] = np.array({
            'data': npz_file,
            'val': val_metrics,
            'test': tst_metrics,
        })

    return metrics_data


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    has_sidekick = args.config_sidekick is not None

    work_dir = _build_work_dir(args.work_dir, args.config_base,
                               args.config_sidekick, args)
    npz_file = f'{args.dataset_name}.npz'
    metrics_path = osp.join(work_dir, 'metrics', npz_file)
    summary_dir  = osp.join(work_dir, 'overall_metrics_summary')
    os.makedirs(summary_dir, exist_ok=True)

    # --- load per-seed metrics -----------------------------------------
    # metrics/<npz> is written by duo_optimizer / temperature_scaling / laplace lora.
    # For a plain LoRA run without those steps, fall back to computing
    # metrics directly from the raw prediction files.
    if not osp.exists(metrics_path):
        print(f'[average_metrics] {metrics_path} not found — '
              'computing metrics from raw prediction files.')
        present = ['base'] + (['sidekick'] if has_sidekick else [])
        metrics_data = _build_metrics_from_preds(work_dir, npz_file, present,
                                                  blob=args.blob)
        os.makedirs(osp.dirname(metrics_path), exist_ok=True)
        np.savez_compressed(metrics_path, **metrics_data)
        print(f'[average_metrics] Saved computed metrics to {metrics_path}')
    else:
        metrics_data = dict(np.load(metrics_path, allow_pickle=True))

    # Discover which model types are present from metric keys
    first_val = metrics_data[next(iter(metrics_data))].item()['val']
    model_types = sorted({k.rsplit('_', 1)[0]
                          for k in first_val if k.endswith('_nll')})

    metrics_df = _create_metrics_df(metrics_data, ['val', 'test'], model_types,
                                    metrics_path)

    summaries = []
    for mt in model_types:
        for split in ('val', 'test'):
            summaries.append(_summarize(metrics_df, mt, split))

    # --- deep ensemble --------------------------------------------------
    if args.deep_ensemble:
        m = args.deep_ensemble_size
        rng = np.random.default_rng(42)

        # For BLoB runs, predictions live under blob/{model_type}/; the DE
        # model-type labels get a '_blob' suffix to distinguish them from the
        # plain-LoRA DE entries.
        preds_root = osp.join(work_dir, 'blob') if args.blob else work_dir
        base_de_label = 'base_blob' if args.blob else 'base'
        side_de_label = 'sidekick_blob' if args.blob else 'sidekick'

        base_tst_path = osp.join(preds_root, 'base', 'test_preds', npz_file)
        base_preds_df = _create_predictions_df(
            dict(np.load(base_tst_path, allow_pickle=True)),
            model_type=base_de_label, output_path=base_tst_path)

        seeds = list(base_preds_df['seed'].unique())
        combos = _sample_combinations(seeds, m, n_samples=len(seeds), rng=rng)
        print(f'[average_metrics] Deep ensemble ({m} models), combos: {combos}')

        de_frames = []
        base_de = _deep_ensemble_metrics(base_preds_df, combos, base_de_label)
        base_de['model'] = f'{base_de_label}_de({m})'
        base_de['split'] = 'test'
        de_frames.append(base_de)

        if has_sidekick:
            side_tst_path = osp.join(preds_root, 'sidekick', 'test_preds', npz_file)
            side_preds_df = _create_predictions_df(
                dict(np.load(side_tst_path, allow_pickle=True)),
                model_type=side_de_label, output_path=side_tst_path)
            side_de = _deep_ensemble_metrics(side_preds_df, combos, side_de_label)
            side_de['model'] = f'{side_de_label}_de({m})'
            side_de['split'] = 'test'
            de_frames.append(side_de)

        # DE on temperature-scaled outputs (only applies to non-blob runs)
        if args.temperature_scaling and not args.blob:
            base_ts_tst_path = osp.join(work_dir, 'temperature_scaling', 'base', 'test_preds', npz_file)
            if osp.exists(base_ts_tst_path):
                base_ts_preds_df = _create_predictions_df(
                    dict(np.load(base_ts_tst_path, allow_pickle=True)),
                    model_type='base_ts', output_path=base_ts_tst_path)
                base_ts_de = _deep_ensemble_metrics(base_ts_preds_df, combos, 'base_ts')
                base_ts_de['model'] = f'base_ts_de({m})'
                base_ts_de['split'] = 'test'
                de_frames.append(base_ts_de)

            if has_sidekick:
                side_ts_tst_path = osp.join(
                    work_dir, 'temperature_scaling', 'sidekick', 'test_preds', npz_file)
                if osp.exists(side_ts_tst_path):
                    side_ts_preds_df = _create_predictions_df(
                        dict(np.load(side_ts_tst_path, allow_pickle=True)),
                        model_type='sidekick_ts', output_path=side_ts_tst_path)
                    side_ts_de = _deep_ensemble_metrics(side_ts_preds_df, combos, 'sidekick_ts')
                    side_ts_de['model'] = f'sidekick_ts_de({m})'
                    side_ts_de['split'] = 'test'
                    de_frames.append(side_ts_de)

        # DE on laplace lora outputs
        if args.laplace_lora:
            base_laplace_tst_path = osp.join(work_dir, 'laplace_lora', 'base', 'test_preds', npz_file)
            if osp.exists(base_laplace_tst_path):
                base_laplace_preds_df = _create_predictions_df(
                    dict(np.load(base_laplace_tst_path, allow_pickle=True)),
                    model_type='base_laplace', output_path=base_laplace_tst_path)
                base_laplace_de = _deep_ensemble_metrics(base_laplace_preds_df, combos, 'base_laplace')
                base_laplace_de['model'] = f'base_laplace_de({m})'
                base_laplace_de['split'] = 'test'
                de_frames.append(base_laplace_de)

            if has_sidekick:
                side_laplace_tst_path = osp.join(
                    work_dir, 'laplace_lora', 'sidekick', 'test_preds', npz_file)
                if osp.exists(side_laplace_tst_path):
                    side_laplace_preds_df = _create_predictions_df(
                        dict(np.load(side_laplace_tst_path, allow_pickle=True)),
                        model_type='sidekick_laplace', output_path=side_laplace_tst_path)
                    side_laplace_de = _deep_ensemble_metrics(side_laplace_preds_df, combos, 'sidekick_laplace')
                    side_laplace_de['model'] = f'sidekick_laplace_de({m})'
                    side_laplace_de['split'] = 'test'
                    de_frames.append(side_laplace_de)

        de_data = pd.concat(de_frames, ignore_index=True)
        de_model_types = [f'{base_de_label}_de({m})'] + (
            [f'{side_de_label}_de({m})'] if has_sidekick else [])
        if args.temperature_scaling and not args.blob:
            de_model_types.append(f'base_ts_de({m})')
            if has_sidekick:
                de_model_types.append(f'sidekick_ts_de({m})')
        ## Add laplace lora to model_types
        if args.laplace_lora and not args.blob:
            de_model_types.append(f'base_laplace_de({m})')
            if has_sidekick:
                de_model_types.append(f'sidekick_laplace_de({m})')
        for mt in de_model_types:
            if mt in de_data['model'].values:
                summaries.append(_summarize(de_data, mt, 'test'))

    summary_df = pd.concat(summaries, ignore_index=True)
    summary_df.to_csv(osp.join(summary_dir, 'metrics_summary_sd_raw.csv'), index=False)

    # --- pretty pivot table (mean ± 1 SD) -------------------------------
    metric_order = ['accuracy', 'f1_macro', 'nll', 'brier', 'ece', 'lppd', 'mean_uncertainty']
    all_model_types = model_types + (de_model_types if args.deep_ensemble else [])

    pretty = summary_df.copy()
    pretty['value'] = pretty.apply(lambda r: _format_sd(r['mean'], r['sd']), axis=1)
    pretty['metric'] = pd.Categorical(
        pretty['metric'], categories=metric_order, ordered=True)
    pretty['model'] = pd.Categorical(
        pretty['model'], categories=all_model_types, ordered=True)
    pretty['split'] = pd.Categorical(
        pretty['split'], categories=['val', 'test'], ordered=True)
    pretty = pretty.sort_values(['model', 'split', 'metric'])

    pretty_wide = pretty.pivot_table(
        index=['model', 'split'], columns='metric',
        values='value', aggfunc='first',
    ).reindex(columns=metric_order).reset_index()

    out_csv = osp.join(summary_dir, 'metrics_summary_sd.csv')
    pretty_wide.to_csv(out_csv, index=False)

    # --- print test results to stdout -----------------------------------
    print('\n=== Metrics Summary (mean ± 1 SD) — test set ===')
    test_rows = pretty_wide[pretty_wide['split'] == 'test']
    cols = list(test_rows.columns)
    rows = [list(map(str, r)) for r in test_rows.itertuples(index=False, name=None)]
    widths = [max(len(c), max((len(r[i]) for r in rows), default=0))
              for i, c in enumerate(cols)]
    print(' | '.join(c.ljust(widths[i]) for i, c in enumerate(cols)))
    print('-+-'.join('-' * w for w in widths))
    for row in rows:
        print(' | '.join(row[i].ljust(widths[i]) for i in range(len(cols))))

    print(f'\nSaved to {out_csv}')


if __name__ == '__main__':
    main()
