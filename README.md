# uq_with_martingale_posteriors

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
