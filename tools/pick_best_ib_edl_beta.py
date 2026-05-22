"""Pick the best IB-EDL beta for base or sidekick from a set of tuning runs."""

import argparse
import os.path as osp
import sys

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from asym_duos import _softmax_from_log, compute_uncertainty_metrics


def _model_tag(cfg_path):
    folder = cfg_path.split('/')[1]
    parts = folder.split('_')
    known_model_starts = ('qwen2', 'qwen3', 'llama', 'mistral', 'gemma')
    split_idx = next(
        (i for i, p in enumerate(parts) if p in known_model_starts),
        2,
    )
    dataset = '_'.join(parts[:split_idx])
    model = '_'.join(parts[split_idx:])
    return dataset, model


def _build_work_dir(root, cfg_base, cfg_sidekick, beta, args):
    """Reconstruct the tuning-run work-dir for a candidate beta.

    The tuning protocol sets both --beta-base and --beta-sidekick to the
    same candidate, so both tags are present with the same value.
    """
    dataset, base_model = _model_tag(cfg_base)
    model_tag = base_model
    if cfg_sidekick is not None:
        _, side_model = _model_tag(cfg_sidekick)
        model_tag = f'{base_model}_{side_model}'

    method_parts = ['ib_edl' if args.ib_edl else 'lora']
    method_parts.append(f'beta_base_{beta}')
    method_parts.append(f'beta_side_{beta}')
    if args.temperature_scaling:
        method_parts.append('tempscaling')
    if args.duo:
        method_parts.append('duo_constrained' if args.duo_constrained else 'duo_unconstrained')
    method = '_'.join(method_parts)
    return osp.join(root, dataset, model_tag, method)


_LOWER_IS_BETTER = ('ece', 'nll', 'brier')
_HIGHER_IS_BETTER = ('accuracy', 'f1_macro', 'lppd')


def _default_lower_is_better(target: str) -> bool:
    for suf in _LOWER_IS_BETTER:
        if target == suf or target.endswith('_' + suf):
            return True
    for suf in _HIGHER_IS_BETTER:
        if target == suf or target.endswith('_' + suf):
            return False
    return True  # conservative default (most calibration metrics are lower-is-better)


def parse_args():
    p = argparse.ArgumentParser(
        description='Pick the best IB-EDL beta for base or sidekick '
                    'from tuning runs.')
    p.add_argument('--config-base', required=True)
    p.add_argument('--config-sidekick', default=None)
    p.add_argument('--work-dir', '-w', required=True)
    p.add_argument('--dataset-name', '-d', required=True)

    # method flags — must match what was used during tuning runs (NOT the final runs)
    p.add_argument('--ib-edl', action='store_true',
                   help='Must match the --ib-edl flag used during tuning.')
    p.add_argument('--duo', action='store_true',
                   help='Should be OFF for the standard tuning protocol.')
    p.add_argument('--duo-constrained', action='store_true')
    p.add_argument('--temperature-scaling', action='store_true')

    # tuning spec
    p.add_argument('--tune', choices=['base', 'sidekick'], required=True,
                   help='Which side to pick the winner for.')
    p.add_argument('--betas', nargs='+', type=float, required=True,
                   help='The beta candidates that were tuned over.')
    p.add_argument('--seed', type=int, required=True,
                   help='Seed used across all tuning runs.')

    # target metric / direction
    p.add_argument('--target', default=None,
                   help='Metric key to compare. Defaults to base_ece when '
                        '--tune=base, sidekick_ece when --tune=sidekick.')
    p.add_argument('--split', choices=['val', 'test'], default='test',
                   help='Which split to read the target metric from '
                        '(default: test).')

    direction = p.add_mutually_exclusive_group()
    direction.add_argument('--minimize', action='store_true')
    direction.add_argument('--maximize', action='store_true')
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Default target picks the side being tuned.
    target = args.target or (f'{args.tune}_ece')

    if args.maximize:
        lower_better = False
    elif args.minimize:
        lower_better = True
    else:
        lower_better = _default_lower_is_better(target)

    bare_target = target
    for prefix in (f'{args.tune}_', 'base_', 'sidekick_'):
        if bare_target.startswith(prefix):
            bare_target = bare_target[len(prefix):]
            break

    npz_name = f'{args.dataset_name}.npz'
    scores = {}
    for beta in args.betas:
        work_dir = _build_work_dir(
            args.work_dir, args.config_base, args.config_sidekick, beta, args)
        preds_path = osp.join(work_dir, args.tune, f'{args.split}_preds', npz_name)

        if not osp.exists(preds_path):
            print(f'[pick_best_beta] Missing predictions for beta={beta}: '
                  f'{preds_path}', file=sys.stderr)
            continue

        data = dict(np.load(preds_path, allow_pickle=True))
        seed_key = str(args.seed)
        if seed_key not in data:
            print(f'[pick_best_beta] Seed {args.seed} not found for '
                  f'beta={beta} in {preds_path}', file=sys.stderr)
            continue

        sd = data[seed_key].item()
        order = np.argsort(sd['idx'])
        logits = np.asarray(sd['logits'])[order]
        y_true = np.asarray(sd['true_labels'])[order]
        probs = _softmax_from_log(logits)
        preds = probs.argmax(axis=1)

        metric_values = {
            'accuracy': float(accuracy_score(y_true, preds)),
            'f1_macro': float(f1_score(y_true, preds, average='macro')),
        }
        metric_values.update({
            k: float(v) for k, v in compute_uncertainty_metrics(probs, y_true).items()
        })

        if bare_target not in metric_values:
            avail = sorted(metric_values.keys())
            print(f'[pick_best_beta] Target "{target}" (unprefixed: '
                  f'"{bare_target}") not computable for beta={beta}. '
                  f'Available: {avail}', file=sys.stderr)
            continue

        scores[beta] = metric_values[bare_target]

    if not scores:
        print('[pick_best_beta] No valid tuning runs found — cannot pick a '
              'winner.', file=sys.stderr)
        sys.exit(1)

    best_beta = (min(scores, key=scores.get) if lower_better
                 else max(scores, key=scores.get))

    direction = 'lower is better' if lower_better else 'higher is better'
    print(f'[pick_best_beta] Picking {args.tune} beta by {target} on '
          f'{args.split} (seed {args.seed}, {direction}):', file=sys.stderr)
    for beta, score in sorted(scores.items()):
        marker = '   <-- winner' if beta == best_beta else ''
        print(f'  beta={beta}: {score:.6f}{marker}', file=sys.stderr)

    print(best_beta)


if __name__ == '__main__':
    main()
