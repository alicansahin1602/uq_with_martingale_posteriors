"""Configurable history and paragraph rendering, independent of model scoring."""

from copy import deepcopy
import hashlib
import json
from collections.abc import Mapping

import numpy as np

from .prompt_types import PromptContext, RenderedPrompt
from .system_prompt import mcqa_system_prompt, build_configurable_system_prompt


HISTORY_FORMATS = ('labels', 'counts', 'counts_mean_probs', 'labels_counts_mean_probs')
TREATMENTS = ('summary_neutral', 'sampling_aware')
STRATEGIES = {}
DEFAULT_CONFIG = {
    'strategy': 'legacy',
    'history_format': 'labels',
    'treatment': 'current',
    'probability_decimals': 6,
    'extra_system_paragraphs': [],
    'extra_user_paragraphs': {'before_history': [], 'after_history': []},
    'params': {},
}


def insert_legacy_history(initial_prompt, labels):
    """Preserve the original sentence and whitespace, including the empty case."""
    if not labels:
        return initial_prompt
    if not initial_prompt.endswith('\nAnswer:'):
        raise ValueError("Prompt does not end with '\\nAnswer:'. Check the dataset preamble.")
    return (initial_prompt[:-len('\nAnswer:')]
            + '\nYour prior answers in previous steps to this specific question were: '
            + ', '.join(labels) + '\nAnswer:')


def register_prompt_strategy(name):
    """Register a PromptStrategy subclass; custom behavior needs no loop changes."""
    def register(cls):
        if name in STRATEGIES:
            raise ValueError(f'Prompt strategy already registered: {name}')
        STRATEGIES[name] = cls
        return cls
    return register


def _paragraphs(value, field):
    """Validate literal text lists; do not interpret braces or execute templates."""
    if not isinstance(value, (list, tuple)) or any(
        not isinstance(x, str) or not x.strip() for x in value
    ):
        raise ValueError(f'{field} must be a list of non-empty strings.')
    return list(value)


class PromptStrategy:
    """Base for renderers. Increment version when changing their wording/logic."""

    version = 1
    is_legacy = False

    def __init__(self, config):
        self.config = deepcopy(config)

    @property
    def fingerprint(self):
        """Include paragraph text and implementation identity in experiment IDs."""
        payload = {'config': self.config, 'version': self.version,
                   'implementation': f'{type(self).__module__}.{type(self).__qualname__}'}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]

    def render(self, context: PromptContext) -> RenderedPrompt:
        """Return system/user text without calling a model or changing history."""
        raise NotImplementedError


@register_prompt_strategy('legacy')
class LegacyPromptStrategy(PromptStrategy):
    """Exact old messages. No added headers, empty-history text, or paragraphs."""

    is_legacy = True

    def render(self, context):
        return RenderedPrompt(
            mcqa_system_prompt(len(context.valid_labels)),
            insert_legacy_history(context.base_user_prompt, context.history.sampled_labels),
            context.valid_labels,
        )


def render_history(context, history_format, decimals):
    """Format only past draws; averages include every step, not selected labels only."""
    history = context.history
    if not history.n_steps:
        return 'HISTORY\nCompleted sampling steps: 0\nNo previous sampling steps.'
    if not set(history.sampled_labels).issubset(context.valid_labels):
        raise ValueError('History contains a label absent from this question.')
    lines = ['HISTORY', f'Completed sampling steps: {history.n_steps}']
    if history_format in ('labels', 'labels_counts_mean_probs'):
        lines.append('Ordered sampled labels: ' + ', '.join(history.sampled_labels))
    if history_format != 'labels':
        show_means = 'mean_probs' in history_format
        header = 'Label | Times sampled'
        if show_means:
            header += ' | Mean previous probability'
            probability_labels = context.probability_labels or context.valid_labels
            means = np.mean(history.previous_probabilities, axis=0)
            if len(means) != len(probability_labels):
                raise ValueError('Probability columns do not match probability_labels.')
        lines.append(header)
        for label in context.valid_labels:
            row = f'{label} | {history.sampled_labels.count(label)}'
            if show_means:
                row += f' | {means[probability_labels.index(label)]:.{decimals}f}'
            lines.append(row)
    return '\n'.join(lines)


@register_prompt_strategy('configurable')
class ConfigurablePromptStrategy(PromptStrategy):
    """Compose a chosen history representation and independent literal paragraphs."""

    def render(self, context):
        cfg = self.config
        if not context.base_user_prompt.endswith('\nAnswer:'):
            raise ValueError("Base question must end with '\\nAnswer:'.")
        user_extras = cfg['extra_user_paragraphs']
        if cfg['treatment'] == 'current':
            # This option allows adding paragraphs around the exact old wording.
            system = mcqa_system_prompt(len(context.valid_labels))
            extras = '\n\n'.join(cfg['extra_system_paragraphs'])
            if extras:
                system = system.replace('\nSUMMARY\n', '\n' + extras + '\n\nSUMMARY\n', 1)
            if not extras and not any(user_extras.values()):
                return RenderedPrompt(system, insert_legacy_history(
                    context.base_user_prompt, context.history.sampled_labels), context.valid_labels)
            history_text = '' if not context.history.n_steps else (
                'Your prior answers in previous steps to this specific question were: '
                + ', '.join(context.history.sampled_labels))
        else:
            system = build_configurable_system_prompt(
                context.valid_labels, cfg['history_format'], cfg['treatment'],
                cfg['extra_system_paragraphs'],
            )
            history_text = render_history(context, cfg['history_format'], cfg['probability_decimals'])
        sections = [context.base_user_prompt[:-len('\nAnswer:')]]
        sections.extend(user_extras['before_history'])
        if history_text:
            sections.append(history_text)
        sections.extend(user_extras['after_history'])
        return RenderedPrompt(system, '\n\n'.join(sections) + '\nAnswer:', context.valid_labels)


def build_prompt_strategy(prompt_cfg=None):
    """Resolve defaults and reject typos/incompatible settings before inference.

    No config selects exact legacy behavior. For custom classes, use the same
    fields and put renderer-specific settings in ``params``.
    """
    if prompt_cfg is not None and not isinstance(prompt_cfg, Mapping):
        raise ValueError('prompt_cfg must be a mapping.')
    supplied = dict(prompt_cfg or {})
    unknown = set(supplied) - set(DEFAULT_CONFIG)
    if unknown:
        raise ValueError(f'Unknown prompt config fields: {sorted(unknown)}')
    cfg = deepcopy(DEFAULT_CONFIG)
    # Selecting a new format/treatment is enough; an explicit strategy is optional.
    if supplied and 'strategy' not in supplied:
        supplied['strategy'] = 'configurable'
    if supplied.get('strategy') == 'configurable':
        cfg['treatment'] = 'summary_neutral'
    user_extras = supplied.pop('extra_user_paragraphs', {})
    if not isinstance(user_extras, Mapping) or set(user_extras) - {'before_history', 'after_history'}:
        raise ValueError('extra_user_paragraphs allows before_history and after_history only.')
    cfg.update(supplied)
    cfg['extra_user_paragraphs'].update(user_extras)
    for name in ('strategy', 'history_format', 'treatment'):
        if not isinstance(cfg[name], str):
            raise ValueError(f'{name} must be a string.')
    if cfg['strategy'] not in STRATEGIES:
        raise ValueError(f"Unknown prompt strategy: {cfg['strategy']}")
    if cfg['history_format'] not in HISTORY_FORMATS:
        raise ValueError(f"Unknown history format: {cfg['history_format']}")
    if cfg['treatment'] not in ('current', *TREATMENTS):
        raise ValueError(f"Unknown treatment: {cfg['treatment']}")
    if type(cfg['probability_decimals']) is not int or not 0 <= cfg['probability_decimals'] <= 15:
        raise ValueError('probability_decimals must be an integer between 0 and 15.')
    cfg['extra_system_paragraphs'] = _paragraphs(cfg['extra_system_paragraphs'], 'extra_system_paragraphs')
    for name, value in cfg['extra_user_paragraphs'].items():
        cfg['extra_user_paragraphs'][name] = _paragraphs(value, name)
    if not isinstance(cfg['params'], Mapping):
        raise ValueError('params must be a mapping.')
    cfg['params'] = dict(cfg['params'])
    if cfg['strategy'] in ('legacy', 'configurable') and cfg['params']:
        raise ValueError('Built-in strategies do not accept extra params.')
    if cfg['treatment'] == 'current' and cfg['history_format'] != 'labels':
        raise ValueError('The current history explanation supports labels only.')
    if cfg['strategy'] == 'legacy' and cfg != DEFAULT_CONFIG:
        raise ValueError('Legacy is unchanged; use strategy=configurable to customize prompts.')
    return STRATEGIES[cfg['strategy']](cfg)
