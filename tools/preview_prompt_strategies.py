"""Export five exact message pairs per strategy, using illustrative history only.

No provider, API key, dataset download, or model is used. These are formatting
examples, NOT model results. Run from the repo root in the project environment.
"""

import argparse
from copy import deepcopy
import json
from pathlib import Path

from mmengine import Config

from martingale_consistency_and_posteriors.utils.prompt_types import HistoryState, PromptContext
from martingale_consistency_and_posteriors.utils.prompt_strategies import (
    HISTORY_FORMATS, TREATMENTS, build_prompt_strategy,
)


BASE_PROMPT = (
    'Return the label of the correct answer for the question below.\n\n'
    'Question: What is 2 + 2?\nChoices:\nA) 3\nB) 4\nC) 5\nD) 6\nAnswer:'
)
ILLUSTRATIVE_DRAWS = ('B', 'A', 'B', 'C')
ILLUSTRATIVE_PROBS = (
    (0.10, 0.70, 0.15, 0.05),
    (0.20, 0.60, 0.15, 0.05),
    (0.15, 0.65, 0.10, 0.10),
    (0.20, 0.50, 0.20, 0.10),
)
EXTRA_PARAGRAPHS = {
    'extra_system_paragraphs': ['Previous sampled answers are not correctness feedback.'],
    'extra_user_paragraphs': {
        'before_history': ['The following block summarizes earlier sampling steps.'],
        'after_history': ['Consider all available answer choices before selecting your answer.'],
    },
}


def preview_configs():
    """Every built-in combination, plus all paragraph-enabled combinations."""
    configs = {'legacy': {'strategy': 'legacy'}}
    variants = [('labels', 'current')]
    variants += [(history, treatment) for history in HISTORY_FORMATS for treatment in TREATMENTS]
    for history, treatment in variants:
        name = f'{history}__{treatment}'
        config = {'strategy': 'configurable', 'history_format': history, 'treatment': treatment}
        configs[name] = config
        configs[name + '__paragraphs'] = {**config, **deepcopy(EXTRA_PARAGRAPHS)}
    return configs


def five_step_preview(config):
    """Render t=0..4; p_t enters history only after the corresponding draw."""
    strategy = build_prompt_strategy(config)
    history = HistoryState()
    steps = []
    for step in range(5):
        rendered = strategy.render(PromptContext(BASE_PROMPT, tuple('ABCD'), history))
        steps.append({
            'step': step,
            'sampled_labels': list(history.sampled_labels),
            'previous_probabilities': history.previous_probabilities,
            'messages': [
                {'role': 'system', 'content': rendered.system_text},
                {'role': 'user', 'content': rendered.user_text},
            ],
        })
        if step < 4:
            history = history.advance(ILLUSTRATIVE_DRAWS[step], ILLUSTRATIVE_PROBS[step])
    return {
        'source': 'Illustrative synthetic history; no model calls or measured probabilities.',
        'prompt_cfg': strategy.config,
        'prompt_fingerprint': strategy.fingerprint,
        'steps': steps,
    }


def export_previews(output_dir, configs, overwrite=False):
    """Write readable Markdown and machine-readable exact message JSON pairs."""
    output_dir = Path(output_dir)
    paths = [output_dir / 'README.md']
    paths += [output_dir / f'{name}.{ext}' for name in configs for ext in ('md', 'json')]
    if not overwrite and any(path.exists() for path in paths):
        raise FileExistsError('Preview files exist; choose a new directory or pass --overwrite.')
    output_dir.mkdir(parents=True, exist_ok=True)
    index = [
        '# Five-step prompt previews', '',
        'These are **illustrative formatting examples, not model results**. No model calls were made.', '',
        'Each file contains complete system/user messages for steps t=0,1,2,3,4 '
        '(zero through four completed draws). Use each pair as a separate request, '
        'not as five turns in one conversation. Local Hugging Face scoring additionally '
        'uses its existing assistant prefill `Answer: `.', '',
        'All variants use the same question and synthetic history: B, A, B, C. '
        'The final counts are A=1, B=2, C=1, D=0; means are '
        'A=0.162500, B=0.612500, C=0.150000, D=0.075000. '
        'These numbers are deliberately illustrative, not confidence estimates for 2+2.', '',
        '| Configuration | Readable messages | Exact JSON |',
        '|---|---|---|',
    ]
    for name, config in configs.items():
        preview = five_step_preview(config)
        (output_dir / f'{name}.json').write_text(json.dumps(preview, indent=2) + '\n', encoding='utf-8')
        lines = [f'# {name}', '', preview['source'], '', '## Resolved configuration', '',
                 '```json', json.dumps(preview['prompt_cfg'], indent=2), '```']
        for step in preview['steps']:
            lines += ['', f"## Step {step['step']} — {len(step['sampled_labels'])} completed draws", '']
            for message in step['messages']:
                lines += [f"### {message['role'].capitalize()}", '', '````text', message['content'], '````', '']
        (output_dir / f'{name}.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
        index.append(f'| {name} | [Markdown]({name}.md) | [JSON]({name}.json) |')
    (output_dir / 'README.md').write_text('\n'.join(index) + '\n', encoding='utf-8')
    return len(configs)


def main():
    """Preview every built-in option, or one edited prompt configuration file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', default='examples/prompt_previews')
    parser.add_argument('--prompt-config', help='YAML containing a prompt_cfg block.')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    if args.prompt_config:
        config = Config.fromfile(args.prompt_config)
        if set(config) != {'prompt_cfg'}:
            raise ValueError('Prompt config file must contain only prompt_cfg.')
        configs = {'custom': config.prompt_cfg.to_dict()}
    else:
        configs = preview_configs()
    count = export_previews(args.output_dir, configs, args.overwrite)
    print(f'Exported {count} configurations x 5 steps to {args.output_dir}; no model calls.')


if __name__ == '__main__':
    main()
