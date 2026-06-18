import re
import torch
import numpy as np
import torch.nn.functional as F
import openai
import anthropic

from typing import Callable, List
from openai import PermissionDeniedError
from .system_prompt import mcqa_system_prompt, ppr_system_prompt


def _parse_ppr_response(text: str, label_chars: List[str], N: int = 100) -> np.ndarray:
    """Parse a newline-separated PPR response into an empirical distribution.

    Reads up to N lines; each valid label increments its count.
    Invalid lines receive a uniform spread instead of being discarded.
    """
    n_classes = len(label_chars)
    label_set = set(label_chars)
    counts = np.zeros(n_classes, dtype=np.float64)
    lines = [l.strip().upper() for l in text.strip().split("\n") if l.strip()]
    for line in lines[:N]:
        if line in label_set:
            counts[label_chars.index(line)] += 1
        else:
            # Verbose models (e.g. Qwen2-7B) emit lines like "The correct answer is B) Mexico".
            # Extract the single standalone label letter, if unambiguous.
            candidates = [c for c in re.findall(r'\b[A-Z]\b', line) if c in label_set]
            if len(candidates) == 1:
                counts[label_chars.index(candidates[0])] += 1
            else:
                counts += 1.0 / n_classes
    if counts.sum() == 0:
        counts = np.ones(n_classes, dtype=np.float64)
    return (counts / counts.sum()).astype(np.float32)

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
    n_api_samples: int,
    api: str = None,
) -> Callable[[List[str]], np.ndarray]:
    """OpenAI provider.

    use_logprobs=True  — one API call per prompt using top_logprobs (fast, exact).
    use_logprobs=False — n_api_samples calls per prompt using temperature sampling.
    """

    client = openai.OpenAI(api_key=api)
    n_classes = len(label_chars)

    def get_probs(prompts: List[str]) -> np.ndarray:
        probs = np.zeros((len(prompts), n_classes), dtype=np.float32)
        for i, prompt in enumerate(prompts):
            if use_logprobs:
                try:
                    resp = client.chat.completions.create(
                        model=model_name,
                        messages=[{
                            "role": "system",
                            "content": mcqa_system_prompt(n_classes)
                        }, {
                            "role": "user",
                            "content": prompt
                        }],
                        extra_body={"thinking": {"type": "disabled"}},
                        max_tokens=1024,
                        logprobs=True,
                        top_logprobs=20,
                    )
                except PermissionDeniedError as e:
                    print("status_code:", getattr(e, "status_code", None))
                    print("message:", str(e))
                    print("body:", getattr(e, "body", None))
                    raise

                top = resp.choices[0].logprobs.content[0].top_logprobs
                log_p = {t.token.strip(): t.logprob for t in top}
                raw = np.array(
                    [np.exp(log_p[c]) if c in log_p else 1e-10 for c in label_chars],
                    dtype=np.float64,
                )
                probs[i] = (raw / raw.sum()).astype(np.float32)
            else:
                counts = np.zeros(n_classes, dtype=np.float64)
                for _ in range(n_api_samples):
                    try:
                        resp = client.chat.completions.create(
                            model=model_name,
                            messages=[{
                                "role": "system",
                                "content": mcqa_system_prompt(n_classes)
                            }, {
                                "role": "user",
                                "content": prompt
                            }],
                            extra_body={"thinking": {"type": "disabled"}},
                            max_tokens=1024,
                            logprobs=True,
                            top_logprobs=20,
                        )
                    except PermissionDeniedError as e:
                        print("status_code:", getattr(e, "status_code", None))
                        print("message:", str(e))
                        print("body:", getattr(e, "body", None))
                        raise

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
    n_api_samples: int,
    api: str = None,
) -> Callable[[List[str]], np.ndarray]:
    """Anthropic provider: sampling-based (no logprob API available)."""
    client = anthropic.Anthropic(api_key=api)
    n_classes = len(label_chars)

    def get_probs(prompts: List[str]) -> np.ndarray:
        probs = np.zeros((len(prompts), n_classes), dtype=np.float32)
        for i, prompt in enumerate(prompts):
            counts = np.zeros(n_classes, dtype=np.float64)
            for _ in range(n_api_samples):
                resp = client.messages.create(
                    model=model_name,
                    max_tokens=1024,
                    thinking={"type": "disabled"},
                    system=mcqa_system_prompt(n_classes),
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


def _build_deepseek_provider(
    model_name: str,
    label_chars: List[str],
    use_logprobs: bool,
    n_api_samples: int,
    api: str = None,
) -> Callable[[List[str]], np.ndarray]:

    """DeepSeek provider.

    use_logprobs=True  — one API call per prompt using top_logprobs (fast, exact).
    use_logprobs=False — n_api_samples calls per prompt using temperature sampling.
    """

    client = openai.OpenAI(api_key=api, base_url="https://api.deepseek.com")
    n_classes = len(label_chars)

    def get_probs(prompts: List[str]) -> np.ndarray:
        probs = np.zeros((len(prompts), n_classes), dtype=np.float32)
        for i, prompt in enumerate(prompts):
            if use_logprobs:
                try:
                    resp = client.chat.completions.create(
                        model=model_name,
                        messages=[{
                            "role": "system",
                            "content": mcqa_system_prompt(n_classes)
                        }, {
                            "role": "user",
                            "content": prompt
                        }],
                        extra_body={"thinking": {"type": "disabled"}},
                        max_tokens=1024,
                        logprobs=True,
                        top_logprobs=20,
                    )
                except PermissionDeniedError as e:
                    print("status_code:", getattr(e, "status_code", None))
                    print("message:", str(e))
                    print("body:", getattr(e, "body", None))
                    raise

                top = resp.choices[0].logprobs.content[0].top_logprobs
                log_p = {t.token.strip(): t.logprob for t in top}
                raw = np.array(
                    [np.exp(log_p[c]) if c in log_p else 1e-10 for c in label_chars],
                    dtype=np.float64,
                )
                probs[i] = (raw / raw.sum()).astype(np.float32)
            else:
                counts = np.zeros(n_classes, dtype=np.float64)
                for _ in range(n_api_samples):
                    try:
                        resp = client.chat.completions.create(
                            model=model_name,
                            messages=[{
                                "role": "system",
                                "content": mcqa_system_prompt(n_classes)
                            }, {
                                "role": "user",
                                "content": prompt
                            }],
                            extra_body={"thinking": {"type": "disabled"}},
                            max_tokens=1024,
                            logprobs=True,
                            top_logprobs=20,
                        )
                    except PermissionDeniedError as e:
                        print("status_code:", getattr(e, "status_code", None))
                        print("message:", str(e))
                        print("body:", getattr(e, "body", None))
                        raise

                    ans = resp.choices[0].message.content.strip().upper()
                    if ans in label_chars:
                        counts[label_chars.index(ans)] += 1
                    else:
                        counts += 1.0 / n_classes  # uniform fallback for off-label replies
                probs[i] = (counts / counts.sum()).astype(np.float32)
        return probs

    return get_probs


# ---------------------------------------------------------------------------
# PPR providers — generate N i.i.d. samples in a single call
# ---------------------------------------------------------------------------

def _build_ppr_hf_provider(
    model,
    tokenizer,
    label_chars: List[str],
    n_ppr_samples: int,
    device: torch.device,
) -> Callable[[List[str]], np.ndarray]:
    """HuggingFace PPR provider: generate n_ppr_samples answers in one forward pass.

    Uses prefix_allowed_tokens_fn to hard-enforce the LETTER\\n pattern at the
    token level, so instruction-tuned models that ignore format instructions
    (e.g. Qwen2-7B) still produce single-character outputs.
    """
    n_classes = len(label_chars)
    sys_prompt = ppr_system_prompt(n_classes, N=n_ppr_samples)

    # Collect every token ID that decodes to a label char, across context variants
    # (bare, space-prefixed, newline-prefixed) to be robust across tokenizers.
    _label_ids: set = set()
    for c in label_chars:
        for prefix in ("", " ", "\n"):
            ids = tokenizer.encode(prefix + c, add_special_tokens=False)
            if ids:
                _label_ids.add(ids[-1])
    label_token_ids = sorted(_label_ids)

    newline_token_ids = tokenizer.encode("\n", add_special_tokens=False)
    if not newline_token_ids:
        newline_token_ids = [tokenizer.encode("\n")[0]]

    def get_probs(prompts: List[str]) -> np.ndarray:
        probs = np.zeros((len(prompts), n_classes), dtype=np.float32)
        for i, prompt in enumerate(prompts):
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": prompt},
            ]
            tokenized = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
            )
            # apply_chat_template returns a BatchEncoding (dict-like), not a raw
            # tensor — extract input_ids explicitly before passing to generate().
            input_ids = tokenized["input_ids"].to(device)
            input_length = input_ids.shape[1]

            def prefix_allowed_tokens_fn(_batch_id, prefix_ids):
                gen_len = len(prefix_ids) - input_length
                return label_token_ids if gen_len % 2 == 0 else newline_token_ids

            with torch.no_grad():
                output_ids = model.generate(
                    input_ids,
                    max_new_tokens=n_ppr_samples * 2,
                    do_sample=True,
                    temperature=1.0,
                    pad_token_id=tokenizer.eos_token_id,
                    prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                )
            generated = tokenizer.decode(
                output_ids[0][input_length:], skip_special_tokens=True
            )
            probs[i] = _parse_ppr_response(generated, label_chars, n_ppr_samples)
        return probs

    return get_probs


def _build_ppr_openai_provider(
    model_name: str,
    label_chars: List[str],
    n_ppr_samples: int,
    api: str = None,
) -> Callable[[List[str]], np.ndarray]:
    """OpenAI PPR provider: one call per prompt, parses n_ppr_samples answers."""
    client = openai.OpenAI(api_key=api)
    n_classes = len(label_chars)
    sys_prompt = ppr_system_prompt(n_classes, N=n_ppr_samples)

    def get_probs(prompts: List[str]) -> np.ndarray:
        probs = np.zeros((len(prompts), n_classes), dtype=np.float32)
        for i, prompt in enumerate(prompts):
            try:
                resp = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=n_ppr_samples * 4,
                    temperature=1.0,
                )
            except PermissionDeniedError as e:
                print("status_code:", getattr(e, "status_code", None))
                print("message:", str(e))
                print("body:", getattr(e, "body", None))
                raise
            text = resp.choices[0].message.content or ""
            probs[i] = _parse_ppr_response(text, label_chars, n_ppr_samples)
        return probs

    return get_probs


def _build_ppr_anthropic_provider(
    model_name: str,
    label_chars: List[str],
    n_ppr_samples: int,
    api: str = None,
) -> Callable[[List[str]], np.ndarray]:
    """Anthropic PPR provider: one call per prompt, parses n_ppr_samples answers."""
    client = anthropic.Anthropic(api_key=api)
    n_classes = len(label_chars)
    sys_prompt = ppr_system_prompt(n_classes, N=n_ppr_samples)

    def get_probs(prompts: List[str]) -> np.ndarray:
        probs = np.zeros((len(prompts), n_classes), dtype=np.float32)
        for i, prompt in enumerate(prompts):
            resp = client.messages.create(
                model=model_name,
                max_tokens=n_ppr_samples * 4,
                system=sys_prompt,
                messages=[{"role": "user", "content": prompt}],
            )
            text = resp.content[0].text if resp.content else ""
            probs[i] = _parse_ppr_response(text, label_chars, n_ppr_samples)
        return probs

    return get_probs


def _build_ppr_deepseek_provider(
    model_name: str,
    label_chars: List[str],
    n_ppr_samples: int,
    api: str = None,
) -> Callable[[List[str]], np.ndarray]:
    """DeepSeek PPR provider: one call per prompt, parses n_ppr_samples answers."""
    client = openai.OpenAI(api_key=api, base_url="https://api.deepseek.com")
    n_classes = len(label_chars)
    sys_prompt = ppr_system_prompt(n_classes, N=n_ppr_samples)

    def get_probs(prompts: List[str]) -> np.ndarray:
        probs = np.zeros((len(prompts), n_classes), dtype=np.float32)
        for i, prompt in enumerate(prompts):
            try:
                resp = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=n_ppr_samples * 4,
                    temperature=1.0,
                )
            except PermissionDeniedError as e:
                print("status_code:", getattr(e, "status_code", None))
                print("message:", str(e))
                print("body:", getattr(e, "body", None))
                raise
            text = resp.choices[0].message.content or ""
            probs[i] = _parse_ppr_response(text, label_chars, n_ppr_samples)
        return probs

    return get_probs
