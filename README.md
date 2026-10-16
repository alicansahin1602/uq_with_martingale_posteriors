# uq_with_martingale_posteriors

## Configurable prompts (sampling experiments)

The default is **legacy**: omitting `prompt_cfg` and `--prompt-config` preserves
the existing system message, history sentence, sampling seeds, probability
extraction, and martingale calculation. New prompt options require
`--mode sampling`; iterative mode still supports the original prompts only and
rejects new strategies before loading data or a model.

### Choose a prompt without editing a model config

From the repository root, in the existing project environment:

```bash
.venv/bin/python tools/martingale_property.py \
    --config configs/simple_mcqa_gpt4o_mini/simple_mcqa_gpt4o_mini.yaml \
    --prompt-config configs/prompts/counts_mean_probs.yaml \
    --work-dir data/results_prompt_strategies \
    --mode sampling --split test --n-samples 5 --K 5 --J 1 --seed 42
```

This runs inference (85 prompt distributions for these settings), and requires
your normal provider credentials/model setup. Replace the model config to use
DeepSeek or local Qwen. It does not require changing the dataset.

Available prompt presets in `configs/prompts/`:

| Preset | History and wording |
|---|---|
| `legacy.yaml` | Exact original prompts |
| `labels.yaml` | Ordered labels; neutral definitions |
| `counts.yaml` | Label counts; neutral definitions |
| `counts_mean_probs.yaml` | Counts and all-step mean probabilities |
| `labels_counts_mean_probs.yaml` | Ordered labels plus counts and means |
| `sampling_aware.yaml` | Counts/means plus the anti-self-reinforcement treatment |
| `custom_paragraphs.yaml` | Counts/means with editable paragraphs in all three positions |
| `current_with_paragraphs.yaml` | Existing wording with editable extra paragraphs |

The same `prompt_cfg` block can live directly in a model config. An optional
`--prompt-config` replaces that block; `--cfg-options` overrides are applied last.
For example, use a counts preset and append:

```bash
--cfg-options prompt_cfg.treatment=sampling_aware
```

### Add paragraphs independently of history

```yaml
prompt_cfg:
    strategy: configurable
    history_format: counts_mean_probs
    treatment: summary_neutral
    probability_decimals: 6
    extra_system_paragraphs:
        - |-
            Previous sampled answers are not correctness feedback.
    extra_user_paragraphs:
        before_history:
            - |-
                The following block summarizes earlier sampling steps.
        after_history:
            - |-
                Consider all available answer choices before selecting your answer.
```

All paragraph lists are optional and preserve their order. Text is literal:
braces are not template variables. The renderer inserts user paragraphs before
or after the history and always keeps `Answer:` last. System paragraphs precede
the output requirement. Paragraphs apply to the empty initial history too.
Use `treatment: current` with `history_format: labels` to retain the old wording
while adding paragraphs. Empty extras with this pair produce exactly the old
messages. `strategy: legacy` rejects customization to prevent silent changes.

Supported history formats are `labels`, `counts`, `counts_mean_probs`, and
`labels_counts_mean_probs`. All support `summary_neutral` and `sampling_aware`;
only `labels` also supports `current`. Unknown fields/combinations fail early.

### Probability-history meaning

At step t the history contains t draws and their t generating distributions:
`p_0, ..., p_(t-1)`. A mean averages a label's normalized probability over
**every previous step**, including steps when that label was not selected.
It is not a mean of log probabilities or a selected-label-only mean.

Every hypothetical child gets its own count increment and the same parent
`p_t` added to probability history. A child's future probabilities are never
used to construct its input. Full precision is retained; only displayed values
are rounded. At t=0, new formats show no previous sampling steps, not invented
uniform averages. Counts summarize chronological history; they are not new
correctness evidence and the prompt does not guarantee martingale consistency.

### Preview five steps without model calls

The generated [preview index](examples/prompt_previews/README.md) links to
19 configurations, each with five complete system/user message pairs (t=0..4),
in Markdown and JSON. These cover every built-in format/treatment combination
and every customizable combination with extra paragraphs. All use the same
**illustrative** draws/probabilities; they are not measured model outputs.

Regenerate all examples, or preview your own edited prompt YAML:

```bash
.venv/bin/python tools/preview_prompt_strategies.py \
    --output-dir examples/prompt_previews --overwrite

.venv/bin/python tools/preview_prompt_strategies.py \
    --prompt-config configs/prompts/custom_paragraphs.yaml \
    --output-dir tmp/my_prompt_preview
```

The preview does not add prior prompts as chat turns. Submit each system/user
pair separately, matching the experiment. Local Hugging Face additionally
retains its existing assistant prefill `Answer: `.

### Results and extending the code

New strategies write to
`<work-dir>/<model-config-folder>/martingale_check/sampling/<format>_<fingerprint>/<run-id>/`.
The fingerprint includes resolved paragraphs, settings, and strategy version;
each run gets a unique ID. Legacy paths stay unchanged, but existing sampling
result files require `--overwrite` or a fresh work directory. Iterative result
files retain their original seed-append behavior.

Existing numeric result keys and user `prompt_history` arrays stay intact.
Additional result fields store system-message histories and valid-class masks;
metadata includes `prompt_cfg`, strategy version, and fingerprint. Raw provider
logs also retain system text. Reports with fixed run paths need the new file
path supplied explicitly; they do not discover new experiments automatically.

For questions with fewer options than the global class dimension, new strategies
only query valid branches. Absent branches have zero arrays and empty strings,
identified by `valid_class_mask`. Their probability must be zero. Legacy retains
its original absent-branch calls for reproducibility.

The implementation is intentionally split into small pieces:

- `utils/prompt_types.py`: immutable history, rendering context, message record.
- `utils/prompt_strategies.py`: configuration validation, history formatting,
  paragraph assembly, and strategy registry.
- `utils/system_prompt.py`: original prompt plus the new system builder.
- `utils/martingale_helpers.py`: advances history and renders branches.
- `utils/model_calls.py`: accepts strings or rendered messages; scoring is unchanged.

For a genuinely different history algorithm, add a `PromptStrategy` subclass
decorated with `@register_prompt_strategy("your_name")` in `prompt_strategies.py`.
Implement `render(context)` returning `RenderedPrompt` and use `params` for
custom settings. External modules must be imported to register their classes;
YAML does not execute/import arbitrary Python. Increment the strategy's `version`
when changing wording or logic. Renderers see no gold answer or future output.

Offline regression tests (mock APIs and a tiny CPU model, no downloads):

```bash
MPLCONFIGDIR=/tmp/codex-prompt-mpl .venv/bin/python -m unittest discover -s tests -v
```

Tests cover pre-change legacy outputs, all 19 configurations over five steps,
branch bookkeeping, all three active providers, result serialization, paragraph
placement, CLI validation, and a complete runner invocation with mock inference.

## Simple MCQA confidence control

`data/datasets/simple_mcqa.json` contains five deliberately easy questions with
four fixed A-D options each. Correct labels are B, D, A, B, B; gold answers stay
in metadata and are never appended to the prompt. `LocalMCQADataset` uses the
same question preamble as ARC/GPQA. System prompts, history wording, probability
extraction, temperature defaults, and the branching/sampling logic are unchanged.
No Hugging Face dataset access or download is needed.

Run from the repository root in the existing project environment:

```bash
python tools/martingale_property.py \
    --config configs/simple_mcqa_gpt4o_mini/simple_mcqa_gpt4o_mini.yaml \
    --work-dir data/results_simple_mcqa \
    --mode sampling --split test --n-samples 5 --K 50 --J 5 --seed 42
```

For the other models, substitute the config path with:

- `configs/simple_mcqa_deepseek_v4_pro/simple_mcqa_deepseek_v4_pro.yaml`
- `configs/simple_mcqa_qwen2_7b/simple_mcqa_qwen2_7b.yaml`

The existing provider keys/model setup are still required. This command runs
inference; it is not an offline test. There is only a `test` split, so do not
use `--split all`. With four labels, Q=5, K=50 and J=5, the runner evaluates
4,905 prompt distributions per model (five initial queries plus all branches).
Choose a fresh work directory for repeat experiments to avoid overwriting
results. Each model gets its own subdirectory under that root.

To add or edit local questions later, use the same JSON schema: `question`
(text), `choices` (four distinct strings in A-D order), and `answer_key`
(one of A-D). Relative `data.test.data_file` paths resolve against the repository
root. The loader validates records before they reach `get_probs`.

Offline validation (no model calls):

```bash
python -m unittest discover -s tests -p 'test_local_mcqa_dataset.py' -v
```

Treat this as a small easy-question control, not proof that high confidence
causes consistency. Check the measured initial probabilities, residuals, and
branch movement: an almost constant distribution can have tiny residuals.
The existing fixed-dataset report generator does not automatically include
this new set; its run list would need a separate extension.

## GPQA evaluation

`GPQADataset` combines `Correct Answer` and the three `Incorrect Answer` columns
into four A-D choices before prompts reach the model. All four answers are
shuffled once during dataset preparation, using a per-row seed derived from
`shuffle_seed` (default: 42). Repeated accesses, trajectories and model runs with
the same dataset ordering and shuffle seed retain the same answer mapping. The
gold label is updated after shuffling and remains evaluation metadata.

The shared config `configs/_base_/gpqa.yaml` defaults to `gpqa_main`. GPQA's
Hugging Face split is named `train`; the config exposes it as `data.test` for
evaluation. Use `--split test` (the runner default), not `--split all`: this config
does not create artificial training/validation aliases for the same questions.

Model configs:

- `configs/gpqa_gpt4o_mini/gpqa_gpt4o_mini.yaml`
- `configs/gpqa_deepseek_v4_pro/gpqa_deepseek_v4_pro.yaml`
- `configs/gpqa_qwen2_7b/gpqa_qwen2_7b.yaml`

From the repository root, for example:

```bash
python tools/martingale_property.py \
    --config configs/gpqa_gpt4o_mini/gpqa_gpt4o_mini.yaml \
    --work-dir data/results_gpqa_main \
    --mode sampling --split test --n-samples 30 --K 20 --J 1 --seed 60
```

This uses the existing runner's history-continuation rule without changing it.
`--seed` controls the experiment's question selection; option ordering is
controlled separately by `data.test.shuffle_seed`.

To select Diamond, add
`--cfg-options data.test.dataset_cfg.name=gpqa_diamond` and choose a separate
work directory (for example `data/results_gpqa_diamond`) to preserve both runs.
Keep `data.test.shuffle_seed` identical across models for matched comparisons.

Access to the [official Hugging Face dataset](https://huggingface.co/datasets/Idavidrein/gpqa)
requires accepting its access conditions and authenticating with an authorized
Hugging Face account (for example via `hf auth login`). The adapter uses the
standard Hugging Face authentication mechanism; do not put tokens in configs.
The example command makes model calls and requires that provider's normal setup.

Offline checks use synthetic rows and a mock probability provider; they do not
download GPQA, load model weights, or make API calls:

```bash
python -m unittest discover -s tests -p 'test_gpqa_dataset.py' -v
```
