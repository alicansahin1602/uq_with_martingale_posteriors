import json
import os
import re
import time
import torch
import numpy as np
import torch.nn.functional as F
import openai
from typing import Callable, List, Optional, Tuple
from openai import PermissionDeniedError
from .system_prompt import mcqa_system_prompt

from google import genai
from google.genai import types
from google.genai.errors import APIError as GoogleAPIError

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer



# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def _retry_with_backoff(fn: Callable, max_retries: int = 10, base_delay: float = 1.0):
    """Call fn(), retrying on rate-limit errors with exponential backoff.

    OpenAI/DeepSeek TPM limits are often hit in tight per-prompt loops; a
    transient 429 shouldn't abort a whole batch run. Google's genai SDK
    doesn't raise openai.RateLimitError -- its 429s surface as a
    GoogleAPIError (ClientError) with .code == 429 -- so that's checked
    separately; any other GoogleAPIError (e.g. a 400/500) is re-raised
    immediately rather than burning retries on a non-transient failure.
    """
    for attempt in range(max_retries):
        try:
            return fn()
        except (openai.RateLimitError, GoogleAPIError) as e:
            if isinstance(e, GoogleAPIError) and getattr(e, "code", None) != 429:
                raise
            if attempt == max_retries - 1:
                raise
            delay = base_delay * (2 ** attempt)
            print(f"Rate limit hit, retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries})...")
            time.sleep(delay)


def _log_raw_response(log_path: Optional[str], record: dict) -> None:
    """Append one JSON record (one raw API call + our interpretation of it)
    to a JSONL debug log, so you can inspect exactly what the model returned
    for later calls -- e.g. grep for "fallback_used": true to see how often
    the empirical-frequency fallback fires, or read "message_content" to check
    whether the model is following the expected answer format.

    Best-effort: a logging failure must never break the actual run, so any
    exception here is caught and printed as a warning instead of raised.
    Does nothing if log_path is falsy.
    """
    if not log_path:
        return
    try:
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        with open(log_path, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except Exception as e:
        print(f"[raw_response_log] WARNING: failed to write log entry: {e}")


def _label_probs_from_top_logprobs(top_logprobs, label_chars: List[str]) -> np.ndarray:
    """Turn one token position's top_logprobs list into a normalized
    per-class probability vector.

    Multiple raw token spellings can normalize to the same label (e.g. "A",
    " A", "a" all mean class A). These are combined via logsumexp -- i.e.
    their actual probability mass is summed -- rather than naively keying a
    dict by the normalized token, which silently lets whichever variant
    happens to appear LAST in the list win. That bug is real and severe:
    top_logprobs is sorted by descending confidence, so a near-zero-probability
    lowercase/whitespace variant appearing later would clobber the correct,
    dominant entry (e.g. "A" at ~99.9999% getting overwritten by "a" at
    ~0.00006%, making some unrelated class look dominant after renormalizing).
    """
    label_set = set(label_chars)
    logprobs_by_label = {c: [] for c in label_chars}
    for t in top_logprobs:
        tok = t.token.strip().upper()
        if tok in label_set:
            logprobs_by_label[tok].append(t.logprob)
    raw = np.zeros(len(label_chars), dtype=np.float64)
    for i, c in enumerate(label_chars):
        lps = logprobs_by_label[c]
        if lps:
            # Getting the sum logprob of token. You can have different tokens as output like 'A', ' A', 'a' etc. which all map to the same label 'A'. So we need to sum the probabilities of all these tokens to get the probability of label 'A'.
            m = max(lps)
            raw[i] = np.exp(m) * np.sum(np.exp(np.array(lps) - m))  # logsumexp, in probability space
        else:
            raw[i] = 1e-10
    return raw / raw.sum()


def _first_label_probs(content, label_chars: List[str]) -> np.ndarray:
    """Scan a Chat Completions logprobs.content list for the first token that
    resolves to a valid label, and return the normalized per-class
    probability vector at that position (see _label_probs_from_top_logprobs).

    Blindly trusting content[0] breaks whenever anything precedes the answer
    token (e.g. reasoning-mode responses, a leading space/punctuation token) --
    this scans instead, and only falls back to content[0] if nothing in the
    whole response matches (preserving the old behavior as a last resort).
    Returns a uniform distribution if content is empty/None.
    """
    n_classes = len(label_chars)
    if not content:
        return np.full(n_classes, 1.0 / n_classes, dtype=np.float64)
    label_set = set(label_chars)
    for entry in content:
        tok = entry.token.strip().strip(".,:;)").upper()
        if tok in label_set:
            return _label_probs_from_top_logprobs(entry.top_logprobs, label_chars)
    return _label_probs_from_top_logprobs(content[0].top_logprobs, label_chars)


def _martingale_sampling_label_logits_and_probs(
    top_logprobs,
    label_chars: List[str],
    missing_probability: float = 1e-10,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract class scores and probabilities for martingale sampling.

    OpenAI does not expose raw pre-softmax logits. It exposes token
    log-probabilities, which equal logits up to a shared additive constant.
    We therefore retain one log-probability score per class as the experiment's
    ``logits`` tensor and softmax those same scores to obtain ``p_t``.

    Several token spellings can represent one class (``"A"``, ``" A"``,
    ``"a"``). Their masses are combined with logsumexp. A class omitted from
    the API's truncated ``top_logprobs`` list receives a documented finite
    floor so centered-score diagnostics remain defined.
    """
    if not 0.0 < missing_probability < 1.0:
        raise ValueError("missing_probability must lie strictly between 0 and 1.")

    label_set = set(label_chars)
    logprobs_by_label = {label: [] for label in label_chars}
    for token_info in top_logprobs or []:
        token = token_info.token.strip().strip(".,:;)").upper()
        if token in label_set:
            logprobs_by_label[token].append(float(token_info.logprob))

    class_scores = np.full(
        len(label_chars), np.log(missing_probability), dtype=np.float64
    )
    for class_idx, label in enumerate(label_chars):
        values = logprobs_by_label[label]
        if values:
            # Stable logsumexp sums the probability mass of all spellings.
            maximum = max(values)
            class_scores[class_idx] = maximum + np.log(
                np.exp(np.asarray(values, dtype=np.float64) - maximum).sum()
            )

    shifted = class_scores - class_scores.max()
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum()
    return class_scores, probabilities


def _first_martingale_sampling_logits_and_probs(
    content,
    label_chars: List[str],
    missing_probability: float = 1e-10,
) -> Tuple[np.ndarray, np.ndarray]:
    """Use the first answer-token position carrying MCQA class scores.

    Prefer a position whose emitted token is itself a class label. If a model
    emits punctuation or reasoning first, fall back to the first position
    whose top-logprob alternatives contain at least one class. Unlike the old
    iterative path, this formal experiment fails loudly when no class scores
    are available instead of silently inserting a uniform distribution.
    """
    if not content:
        raise ValueError("Provider returned no logprob content for martingale_sampling.")

    label_set = set(label_chars)
    for entry in content:
        emitted = entry.token.strip().strip(".,:;)").upper()
        if emitted in label_set:
            return _martingale_sampling_label_logits_and_probs(
                entry.top_logprobs, label_chars, missing_probability
            )

    for entry in content:
        alternatives = {
            token_info.token.strip().strip(".,:;)").upper()
            for token_info in (entry.top_logprobs or [])
        }
        if alternatives & label_set:
            return _martingale_sampling_label_logits_and_probs(
                entry.top_logprobs, label_chars, missing_probability
            )

    raise ValueError(
        "Could not find any MCQA class token in provider top_logprobs for "
        "martingale_sampling. Inspect the raw response log and prompt."
    )


def extract_choice_labels(prompt: str) -> list[str]:
    """Extract labels such as A, B, C, D from the Choices section."""

    if "Choices:\n" not in prompt:
        raise ValueError("Prompt does not contain a Choices section.")

    choice_block = prompt.split("Choices:\n", 1)[1]

    # Remove history and answer suffixes.
    choice_block = choice_block.split(
        "\nYour prior answers in previous steps", 1
    )[0]
    choice_block = choice_block.rsplit("\nAnswer:", 1)[0]

    labels = re.findall(
        r"(?m)^([A-Z])\)\s+",
        choice_block,
    )

    labels = list(dict.fromkeys(labels))

    if not labels:
        raise ValueError(
            f"Could not find answer labels in prompt:\n{prompt}"
        )

    expected_labels = [
        chr(ord("A") + index)
        for index in range(len(labels))
    ]

    if labels != expected_labels:
        raise ValueError(
            f"Expected consecutive labels {expected_labels}, "
            f"but found {labels}."
        )

    return labels


def _prompt_label_info(
    prompt: str,
    label_chars: List[str],
) -> Tuple[List[str], np.ndarray, List[int]]:
    """Return the prompt's valid labels, global mask, and global indices."""

    valid_labels = extract_choice_labels(prompt)
    unknown_labels = [label for label in valid_labels if label not in label_chars]
    if unknown_labels:
        raise ValueError(
            f"Prompt contains labels {unknown_labels} that are not in the "
            f"configured label set {label_chars}."
        )

    valid_indices = [label_chars.index(label) for label in valid_labels]
    valid_mask = np.zeros(len(label_chars), dtype=bool)
    valid_mask[valid_indices] = True
    return valid_labels, valid_mask, valid_indices


def _embed_prompt_probabilities(
    local_probs: np.ndarray,
    valid_indices: List[int],
    n_classes: int,
) -> np.ndarray:
    """Embed prompt-local probabilities into the configured class space."""

    local_probs = np.asarray(local_probs, dtype=np.float64)
    if local_probs.shape[-1] != len(valid_indices):
        raise ValueError(
            f"Expected {len(valid_indices)} local classes, got shape "
            f"{local_probs.shape}."
        )

    global_shape = local_probs.shape[:-1] + (n_classes,)
    global_probs = np.zeros(global_shape, dtype=np.float64)
    global_probs[..., valid_indices] = local_probs
    return global_probs


def _embed_prompt_scores(
    local_scores: np.ndarray,
    valid_indices: List[int],
    n_classes: int,
    missing_probability: float,
) -> np.ndarray:
    """Embed scores, using a finite floor for classes absent from the prompt."""

    local_scores = np.asarray(local_scores, dtype=np.float64)
    if local_scores.shape[-1] != len(valid_indices):
        raise ValueError(
            f"Expected {len(valid_indices)} local classes, got shape "
            f"{local_scores.shape}."
        )

    global_shape = local_scores.shape[:-1] + (n_classes,)
    global_scores = np.full(
        global_shape,
        np.log(missing_probability),
        dtype=np.float64,
    )
    global_scores[..., valid_indices] = local_scores
    return global_scores


# ---------------------------------------------------------------------------
# Iterative Method Builders
# ---------------------------------------------------------------------------

#def _build_hf_provider(
#    model,
#    tokenizer,
#    target_ids: torch.Tensor,
#    tokenizer_run_cfg: dict,
#    device: torch.device,
#) -> Callable[[List[str]], np.ndarray]:
#    """HuggingFace open-source provider: batched forward pass over logits."""
#    @torch.no_grad()
#    def get_probs(prompts: List[str]) -> np.ndarray:
#        inputs = tokenizer(prompts, **tokenizer_run_cfg).to(device)
#        logits = model(**inputs).logits[:, -1, target_ids]  # (B, n_classes)
#        return F.softmax(logits.float(), dim=-1).cpu().numpy()
#    return get_probs

def _build_hf_provider(
    model_name: str,
    label_chars: List[str],
    use_logprobs: bool,
    n_api_samples: int,
    api: str = None,
    raw_log_path: Optional[str] = None,
) -> Callable[[List[str]], np.ndarray]:
    """HuggingFace provider.

    use_logprobs=True  — one API call per prompt using top_logprobs (fast, exact).
    use_logprobs=False — n_api_samples calls per prompt using temperature sampling.
    """

    client = openai.OpenAI(api_key=api, base_url="https://router.huggingface.co/v1")
    n_classes = len(label_chars)

    def get_probs(prompts: List[str]) -> np.ndarray:
        probs = np.zeros((len(prompts), n_classes), dtype=np.float64)
        for i, prompt in enumerate(prompts):
            valid_labels, valid_mask, valid_indices = _prompt_label_info(
                prompt, label_chars
            )
            if use_logprobs:
                try:
                    resp = _retry_with_backoff(lambda: client.chat.completions.create(
                        model=f'{model_name}:cheapest',
                        messages=[{
                            "role": "system",
                            "content": mcqa_system_prompt(len(valid_labels))
                        }, {
                            "role": "user",
                            "content": prompt
                        }],
                        #max_tokens=1024 * 4,
                        #extra_body={"thinking": {"type": "enabled"}},
                        logprobs=True,
                        top_logprobs=20,
                        temperature=0.5
                    ))
                except PermissionDeniedError as e:
                    print("status_code:", getattr(e, "status_code", None))
                    print("message:", str(e))
                    print("body:", getattr(e, "body", None))
                    raise

                content = resp.choices[0].logprobs.content if resp.choices[0].logprobs else None
                local_probs = _first_label_probs(content, valid_labels)
                probs[i] = _embed_prompt_probabilities(
                    local_probs, valid_indices, n_classes
                )

                _log_raw_response(raw_log_path, {
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "provider": "openai_iterative",
                    "model_name": model_name,
                    "prompt_index": i,
                    "use_logprobs": True,
                    "prompt": prompt,
                    "valid_labels": valid_labels,
                    "valid_class_mask": valid_mask.tolist(),
                    "finish_reason": resp.choices[0].finish_reason,
                    "reasoning_content_len": len(getattr(resp.choices[0].message, "reasoning_content", None) or ""),
                    "message_content": resp.choices[0].message.content or "",
                    "has_logprobs_content": content is not None,
                    "n_logprob_entries": len(content) if content else 0,
                    "resulting_probs": probs[i].tolist(),
                    "raw_response": resp.model_dump() if hasattr(resp, "model_dump") else str(resp),
                })
            else:
                counts = np.zeros(n_classes, dtype=np.float64)
                for _ in range(n_api_samples):
                    try:
                        resp = _retry_with_backoff(lambda: client.chat.completions.create(
                            model=model_name,
                            messages=[{
                                "role": "system",
                                "content": mcqa_system_prompt(len(valid_labels))
                            }, {
                                "role": "user",
                                "content": prompt
                            }],
                            max_tokens=1024,
                            logprobs=True,
                            top_logprobs=20,
                        ))
                    except PermissionDeniedError as e:
                        print("status_code:", getattr(e, "status_code", None))
                        print("message:", str(e))
                        print("body:", getattr(e, "body", None))
                        raise

                    ans = resp.choices[0].message.content.strip().upper()
                    if ans in valid_labels:
                        counts[label_chars.index(ans)] += 1
                    else:
                        counts[valid_indices] += 1.0 / len(valid_labels)
                probs[i] = counts / counts.sum()
        return probs

    return get_probs


def _build_openai_provider(
    model_name: str,
    label_chars: List[str],
    use_logprobs: bool,
    n_api_samples: int,
    api: str = None,
    raw_log_path: Optional[str] = None,
) -> Callable[[List[str]], np.ndarray]:
    """OpenAI provider.

    use_logprobs=True  — one API call per prompt using top_logprobs (fast, exact).
    use_logprobs=False — n_api_samples calls per prompt using temperature sampling.
    """

    client = openai.OpenAI(api_key=api)
    n_classes = len(label_chars)

    def get_probs(prompts: List[str]) -> np.ndarray:
        probs = np.zeros((len(prompts), n_classes), dtype=np.float64)
        for i, prompt in enumerate(prompts):
            valid_labels, valid_mask, valid_indices = _prompt_label_info(
                prompt, label_chars
            )
            if use_logprobs:
                try:
                    resp = _retry_with_backoff(lambda: client.chat.completions.create(
                        model=model_name,
                        messages=[{
                            "role": "system",
                            "content": mcqa_system_prompt(len(valid_labels))
                        }, {
                            "role": "user",
                            "content": prompt
                        }],
                        #max_tokens=1024 * 4,
                        #extra_body={"thinking": {"type": "enabled"}},
                        logprobs=True,
                        top_logprobs=20,
                        temperature=0.5
                    ))
                except PermissionDeniedError as e:
                    print("status_code:", getattr(e, "status_code", None))
                    print("message:", str(e))
                    print("body:", getattr(e, "body", None))
                    raise

                content = resp.choices[0].logprobs.content if resp.choices[0].logprobs else None
                local_probs = _first_label_probs(content, valid_labels)
                probs[i] = _embed_prompt_probabilities(
                    local_probs, valid_indices, n_classes
                )

                _log_raw_response(raw_log_path, {
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "provider": "openai_iterative",
                    "model_name": model_name,
                    "prompt_index": i,
                    "use_logprobs": True,
                    "prompt": prompt,
                    "valid_labels": valid_labels,
                    "valid_class_mask": valid_mask.tolist(),
                    "finish_reason": resp.choices[0].finish_reason,
                    "reasoning_content_len": len(getattr(resp.choices[0].message, "reasoning_content", None) or ""),
                    "message_content": resp.choices[0].message.content or "",
                    "has_logprobs_content": content is not None,
                    "n_logprob_entries": len(content) if content else 0,
                    "resulting_probs": probs[i].tolist(),
                    "raw_response": resp.model_dump() if hasattr(resp, "model_dump") else str(resp),
                })
            else:
                counts = np.zeros(n_classes, dtype=np.float64)
                for _ in range(n_api_samples):
                    try:
                        resp = _retry_with_backoff(lambda: client.chat.completions.create(
                            model=model_name,
                            messages=[{
                                "role": "system",
                                "content": mcqa_system_prompt(len(valid_labels))
                            }, {
                                "role": "user",
                                "content": prompt
                            }],
                            max_tokens=1024,
                            logprobs=True,
                            top_logprobs=20,
                        ))
                    except PermissionDeniedError as e:
                        print("status_code:", getattr(e, "status_code", None))
                        print("message:", str(e))
                        print("body:", getattr(e, "body", None))
                        raise

                    ans = resp.choices[0].message.content.strip().upper()
                    if ans in valid_labels:
                        counts[label_chars.index(ans)] += 1
                    else:
                        counts[valid_indices] += 1.0 / len(valid_labels)
                probs[i] = counts / counts.sum()
        return probs

    return get_probs


def _build_deepseek_provider(
    model_name: str,
    label_chars: List[str],
    use_logprobs: bool,
    n_api_samples: int,
    api: str = None,
    raw_log_path: Optional[str] = None,
) -> Callable[[List[str]], np.ndarray]:

    """DeepSeek provider.

    use_logprobs=True  — one API call per prompt using top_logprobs (fast, exact).
    use_logprobs=False — n_api_samples calls per prompt using temperature sampling.

    raw_log_path: if set, every raw API response (plus our interpretation of
        it) is appended as one JSON line to this file -- see
        _log_raw_response for the exact schema. In the use_logprobs=False
        branch, every one of the n_api_samples calls per prompt gets its own
        line (so the file grows with n_api_samples * n_prompts).
    """

    client = openai.OpenAI(api_key=api, base_url="https://api.deepseek.com")
    n_classes = len(label_chars)

    def get_probs(prompts: List[str]) -> np.ndarray:
        probs = np.zeros((len(prompts), n_classes), dtype=np.float64)
        for i, prompt in enumerate(prompts):
            valid_labels, valid_mask, valid_indices = _prompt_label_info(
                prompt, label_chars
            )
            if use_logprobs:
                try:
                    resp = _retry_with_backoff(lambda: client.chat.completions.create(
                        model=model_name,
                        messages=[
                            {
                                "role": "system",
                                "content": mcqa_system_prompt(len(valid_labels))
                            },
                            {
                                "role": "user",
                                "content": prompt
                            }],
                        # Reasoning mode deliberately OFF here: with thinking
                        # enabled, the model's first generated token is part of
                        # a reasoning preamble, not the answer letter, so
                        # content[0] (and even a scan for the first matching
                        # label) becomes unreliable. Getting a clean one-letter
                        # answer doesn't need reasoning anyway -- matches the
                        # sampling branch below, which already disables it.
                        extra_body={"thinking": {"type": "enabled"}},
                        #max_tokens=1024 * 4,
                        logprobs=True,
                        top_logprobs=20,
                        top_p=1.0
                    ))
                except PermissionDeniedError as e:
                    print("status_code:", getattr(e, "status_code", None))
                    print("message:", str(e))
                    print("body:", getattr(e, "body", None))
                    raise

                content = resp.choices[0].logprobs.content if resp.choices[0].logprobs else None
                local_probs = _first_label_probs(content, valid_labels)
                probs[i] = _embed_prompt_probabilities(
                    local_probs, valid_indices, n_classes
                )

                _log_raw_response(raw_log_path, {
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "provider": "deepseek_iterative",
                    "model_name": model_name,
                    "prompt_index": i,
                    "use_logprobs": True,
                    "prompt": prompt,
                    "valid_labels": valid_labels,
                    "valid_class_mask": valid_mask.tolist(),
                    "finish_reason": resp.choices[0].finish_reason,
                    "reasoning_content_len": len(getattr(resp.choices[0].message, "reasoning_content", None) or ""),
                    "message_content": resp.choices[0].message.content or "",
                    "has_logprobs_content": content is not None,
                    "n_logprob_entries": len(content) if content else 0,
                    "resulting_probs": probs[i].tolist(),
                    "raw_response": resp.model_dump() if hasattr(resp, "model_dump") else str(resp),
                })
            else:
                counts = np.zeros(n_classes, dtype=np.float64)
                for sample_idx in range(n_api_samples):
                    try:
                        resp = _retry_with_backoff(lambda: client.chat.completions.create(
                            model=model_name,
                            messages=[{
                                "role": "system",
                                "content": mcqa_system_prompt(len(valid_labels))
                            }, {
                                "role": "user",
                                "content": prompt
                            }],
                            extra_body={"thinking": {"type": "enabled"}},
                            #max_tokens=1024 * 4,
                            logprobs=True,
                            top_logprobs=20,
                            #top_p = 1.0
                            temperature=0.0 # As the temperature increases, tests are violated.
                        ))
                    except PermissionDeniedError as e:
                        print("status_code:", getattr(e, "status_code", None))
                        print("message:", str(e))
                        print("body:", getattr(e, "body", None))
                        raise
                    ans = resp.choices[0].message.content.strip().upper()

                    if ans in valid_labels:
                        counts[label_chars.index(ans)] += 1
                    else:
                        counts[valid_indices] += 1.0 / len(valid_labels)

                    _log_raw_response(raw_log_path, {
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                        "provider": "deepseek_direct_sampling",
                        "model_name": model_name,
                        "prompt_index": i,
                        "sample_index": sample_idx,
                        "use_logprobs": False,
                        "prompt": prompt,
                        "valid_labels": valid_labels,
                        "valid_class_mask": valid_mask.tolist(),
                        "finish_reason": resp.choices[0].finish_reason,
                        "message_content": resp.choices[0].message.content or "",
                        "parsed_answer": ans,
                        "matched_label": ans in valid_labels,
                        "raw_response": resp.model_dump() if hasattr(resp, "model_dump") else str(resp),
                    })
                probs[i] = counts / counts.sum()
        return probs

    return get_probs


# ---------------------------------------------------------------------------
# Sampling Method Builders
# ---------------------------------------------------------------------------

def _build_hf_martingale_sampling_provider(
    model_name: str,
    label_chars: List[str],
    api: str = None,
    raw_log_path: Optional[str] = None,
    temperature: float = 0.5,
    missing_probability: float = 1e-10,
    inference_provider = 'cheapest'
) -> Callable[[List[str]], Tuple[np.ndarray, np.ndarray]]:
    """Build a Hugging Face router scorer for the branching experiment.

    Hugging Face exposes an OpenAI-compatible Chat Completions endpoint. The
    returned callable requests token log probabilities and returns both their
    per-class log scores and the corresponding normalized probabilities, with
    shapes ``(batch, C)``. The runner samples continuations locally from those
    probabilities, so the trajectory transition distribution is exactly the
    returned ``p_t``.
    """
    if len(label_chars) > 20:
        raise ValueError(
            "The Hugging Face OpenAI-compatible endpoint is queried with at "
            "most 20 top_logprobs, so no more than 20 classes are supported."
        )

    client = openai.OpenAI(
        api_key=api,
        base_url="https://router.huggingface.co/v1",
    )
    n_classes = len(label_chars)
    routed_model_name = f"{model_name}:{inference_provider}"

    def get_probs(prompts: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        class_scores = np.zeros((len(prompts), n_classes), dtype=np.float64)
        probabilities = np.zeros_like(class_scores)

        for prompt_index, prompt in enumerate(prompts):
            valid_labels, valid_mask, valid_indices = _prompt_label_info(
                prompt, label_chars
            )
            try:
                response = _retry_with_backoff(
                    lambda: client.chat.completions.create(
                        model=routed_model_name,
                        messages=[
                            {
                                "role": "system",
                                "content": mcqa_system_prompt(len(valid_labels)),
                            },
                            {"role": "user", "content": prompt},
                        ],
                        logprobs=True,
                        top_logprobs=20,
                        temperature=temperature,
                    )
                )
            except PermissionDeniedError as error:
                print("status_code:", getattr(error, "status_code", None))
                print("message:", str(error))
                print("body:", getattr(error, "body", None))
                raise

            choice = response.choices[0]
            content = choice.logprobs.content if choice.logprobs else None
            local_scores, local_probs = _first_martingale_sampling_logits_and_probs(
                content,
                valid_labels,
                missing_probability=missing_probability,
            )
            scores_i = _embed_prompt_scores(
                local_scores, valid_indices, n_classes, missing_probability
            )
            probs_i = _embed_prompt_probabilities(
                local_probs, valid_indices, n_classes
            )
            class_scores[prompt_index] = scores_i
            probabilities[prompt_index] = probs_i

            _log_raw_response(
                raw_log_path,
                {
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "provider": "huggingface_martingale_sampling",
                    "model_name": model_name,
                    "routed_model_name": routed_model_name,
                    "prompt_index": prompt_index,
                    "prompt": prompt,
                    "valid_labels": valid_labels,
                    "valid_class_mask": valid_mask.tolist(),
                    "temperature": temperature,
                    "top_logprobs": 20,
                    "missing_probability": missing_probability,
                    "finish_reason": choice.finish_reason,
                    "message_content": choice.message.content or "",
                    "class_logprob_scores": scores_i.tolist(),
                    "resulting_probs": probs_i.tolist(),
                    "raw_response": response.model_dump()
                    if hasattr(response, "model_dump")
                    else str(response),
                },
            )

        return class_scores, probabilities

    get_probs.martingale_sampling_metadata = {
        "provider": "huggingface",
        "model_identifier": model_name,
        "routed_model_identifier": routed_model_name,
        "tokenizer_identifier": "Hugging Face server-side tokenizer (not exposed)",
        "tokenization_verified": False,
        "class_token_mapping": {
            str(index): label for index, label in enumerate(label_chars)
        },
        "score_type": (
            "Hugging Face router top_logprobs-derived class log scores; "
            "raw model logits are not exposed"
        ),
        "missing_class_probability_floor": float(missing_probability),
        "decoding_parameters": {
            "temperature": float(temperature),
            "logprobs": True,
            "top_logprobs": 20,
            "routing_policy": "cheapest",
        },
        "numpy_version": np.__version__,
        "openai_version": getattr(openai, "__version__", "unknown"),
    }
    return get_probs

def _build_openai_martingale_sampling_provider(
    model_name: str,
    label_chars: List[str],
    api: str = None,
    raw_log_path: Optional[str] = None,
    temperature: float = 0.5,
    missing_probability: float = 1e-10,
) -> Callable[[List[str]], Tuple[np.ndarray, np.ndarray]]:
    """Build the OpenAI scorer for the exact branching experiment.

    The returned ``get_probs(prompts)`` callable returns a pair:

    - class_scores: ``(batch, C)`` OpenAI token log-probability scores
    - probabilities: ``(batch, C)`` softmax-normalized class probabilities

    OpenAI's API does not provide raw logits. Token log probabilities are
    logit-equivalent up to a shared additive constant, so they preserve all
    centered-logit comparisons requested by the experiment. Scores for labels
    omitted by the truncated top-20 response use ``missing_probability`` as a
    finite floor; that limitation is recorded in the callable's metadata.

    Sampling the trajectory is deliberately *not* delegated to OpenAI. The
    runner uses ``numpy.random.Generator.choice`` directly on the returned
    five-class vector, making the actual continuation distribution q_t=p_t.
    """
    if len(label_chars) > 20:
        raise ValueError(
            "OpenAI supports at most 20 top_logprobs; this provider cannot "
            "score more than 20 classes consistently."
        )

    client = openai.OpenAI(api_key=api)
    n_classes = len(label_chars)

    def get_probs(prompts: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        class_scores = np.zeros((len(prompts), n_classes), dtype=np.float64)
        probabilities = np.zeros_like(class_scores)

        for prompt_index, prompt in enumerate(prompts):
            valid_labels, valid_mask, valid_indices = _prompt_label_info(
                prompt, label_chars
            )
            try:
                response = _retry_with_backoff(
                    lambda: client.chat.completions.create(
                        model=model_name,
                        messages=[
                            {
                                "role": "system",
                                "content": mcqa_system_prompt(len(valid_labels)),
                            },
                            {"role": "user", "content": prompt},
                        ],
                        logprobs=True,
                        top_logprobs=20,
                        temperature=temperature,
                    )
                )
            except PermissionDeniedError as error:
                print("status_code:", getattr(error, "status_code", None))
                print("message:", str(error))
                print("body:", getattr(error, "body", None))
                raise

            choice = response.choices[0]
            content = choice.logprobs.content if choice.logprobs else None
            local_scores, local_probs = _first_martingale_sampling_logits_and_probs(
                content,
                valid_labels,
                missing_probability=missing_probability,
            )
            scores_i = _embed_prompt_scores(
                local_scores, valid_indices, n_classes, missing_probability
            )
            probs_i = _embed_prompt_probabilities(
                local_probs, valid_indices, n_classes
            )
            class_scores[prompt_index] = scores_i
            probabilities[prompt_index] = probs_i

            _log_raw_response(
                raw_log_path,
                {
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "provider": "openai_martingale_sampling",
                    "model_name": model_name,
                    "prompt_index": prompt_index,
                    "prompt": prompt,
                    "valid_labels": valid_labels,
                    "valid_class_mask": valid_mask.tolist(),
                    "temperature": temperature,
                    "top_logprobs": 20,
                    "missing_probability": missing_probability,
                    "finish_reason": choice.finish_reason,
                    "message_content": choice.message.content or "",
                    "class_logprob_scores": scores_i.tolist(),
                    "resulting_probs": probs_i.tolist(),
                    "raw_response": response.model_dump()
                    if hasattr(response, "model_dump")
                    else str(response),
                },
            )

        return class_scores, probabilities

    # The runner copies this into the saved result so the class-scoring and
    # decoding choices travel with every experiment artifact.
    get_probs.martingale_sampling_metadata = {
        "provider": "openai",
        "model_identifier": model_name,
        "tokenizer_identifier": "OpenAI server-side tokenizer (not exposed)",
        "tokenization_verified": False,
        "class_token_mapping": {
            str(index): label for index, label in enumerate(label_chars)
        },
        "score_type": (
            "OpenAI top_logprobs-derived class log scores; raw API logits "
            "are not exposed"
        ),
        "missing_class_probability_floor": float(missing_probability),
        "decoding_parameters": {
            "temperature": float(temperature),
            "logprobs": True,
            "top_logprobs": 20,
        },
        "numpy_version": np.__version__,
        "openai_version": getattr(openai, "__version__", "unknown"),
    }
    return get_probs

def _build_deepseek_martingale_sampling_provider(
    model_name: str,
    label_chars: List[str],
    api: str = None,
    raw_log_path: Optional[str] = None,
    temperature: float = 0.5,
    missing_probability: float = 1e-10,
) -> Callable[[List[str]], Tuple[np.ndarray, np.ndarray]]:
    """Build the OpenAI scorer for the exact branching experiment.

    The returned ``get_probs(prompts)`` callable returns a pair:

    - class_scores: ``(batch, C)`` OpenAI token log-probability scores
    - probabilities: ``(batch, C)`` softmax-normalized class probabilities

    OpenAI's API does not provide raw logits. Token log probabilities are
    logit-equivalent up to a shared additive constant, so they preserve all
    centered-logit comparisons requested by the experiment. Scores for labels
    omitted by the truncated top-20 response use ``missing_probability`` as a
    finite floor; that limitation is recorded in the callable's metadata.

    Sampling the trajectory is deliberately *not* delegated to OpenAI. The
    runner uses ``numpy.random.Generator.choice`` directly on the returned
    five-class vector, making the actual continuation distribution q_t=p_t.
    """
    if len(label_chars) > 20:
        raise ValueError(
            "OpenAI supports at most 20 top_logprobs; this provider cannot "
            "score more than 20 classes consistently."
        )

    client = openai.OpenAI(api_key=api, base_url="https://api.deepseek.com")
    n_classes = len(label_chars)

    def get_probs(prompts: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        class_scores = np.zeros((len(prompts), n_classes), dtype=np.float64)
        probabilities = np.zeros_like(class_scores)

        for prompt_index, prompt in enumerate(prompts):
            valid_labels, valid_mask, valid_indices = _prompt_label_info(
                prompt, label_chars
            )
            try:
                response = _retry_with_backoff(
                    lambda: client.chat.completions.create(
                        model=model_name,
                        messages=[
                            {
                                "role": "system",
                                "content": mcqa_system_prompt(len(valid_labels)),
                            },
                            {"role": "user", "content": prompt},
                        ],
                        logprobs=True,
                        top_logprobs=20,
                        temperature=temperature,
                    )
                )
            except PermissionDeniedError as error:
                print("status_code:", getattr(error, "status_code", None))
                print("message:", str(error))
                print("body:", getattr(error, "body", None))
                raise

            choice = response.choices[0]
            content = choice.logprobs.content if choice.logprobs else None
            local_scores, local_probs = _first_martingale_sampling_logits_and_probs(
                content,
                valid_labels,
                missing_probability=missing_probability,
            )
            scores_i = _embed_prompt_scores(
                local_scores, valid_indices, n_classes, missing_probability
            )
            probs_i = _embed_prompt_probabilities(
                local_probs, valid_indices, n_classes
            )
            class_scores[prompt_index] = scores_i
            probabilities[prompt_index] = probs_i

            _log_raw_response(
                raw_log_path,
                {
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "provider": "deepseek_martingale_sampling",
                    "model_name": model_name,
                    "prompt_index": prompt_index,
                    "prompt": prompt,
                    "valid_labels": valid_labels,
                    "valid_class_mask": valid_mask.tolist(),
                    "temperature": temperature,
                    "top_logprobs": 20,
                    "missing_probability": missing_probability,
                    "finish_reason": choice.finish_reason,
                    "message_content": choice.message.content or "",
                    "class_logprob_scores": scores_i.tolist(),
                    "resulting_probs": probs_i.tolist(),
                    "raw_response": response.model_dump()
                    if hasattr(response, "model_dump")
                    else str(response),
                },
            )

        return class_scores, probabilities

    # The runner copies this into the saved result so the class-scoring and
    # decoding choices travel with every experiment artifact.
    get_probs.martingale_sampling_metadata = {
        "provider": "deepseek",
        "model_identifier": model_name,
        "tokenizer_identifier": "DeepSeek server-side tokenizer (not exposed)",
        "tokenization_verified": False,
        "class_token_mapping": {
            str(index): label for index, label in enumerate(label_chars)
        },
        "score_type": (
            "DeepSeek top_logprobs-derived class log scores; raw API logits "
            "are not exposed"
        ),
        "missing_class_probability_floor": float(missing_probability),
        "decoding_parameters": {
            "temperature": float(temperature),
            "logprobs": True,
            "top_logprobs": 20,
        },
        "numpy_version": np.__version__,
        "openai_version": getattr(openai, "__version__", "unknown"),
    }
    return get_probs

def _build_local_hf_martingale_sampling_provider(
    model_name, 
    model,
    tokenizer,
    label_chars: List[str],
    temperature: float = 0.5,
    raw_log_path: Optional[str] = None,
    log_top_k: int = 20,
) -> Callable[[List[str]], Tuple[np.ndarray, np.ndarray]]:
    """
    Load a Hugging Face model locally and construct a provider compatible with
    run_martingale_sampling_check.

    Returns
    -------
    get_probs(prompts)
        A function returning:

        class_log_scores: shape (batch, C)
        class_probs:      shape (batch, C)
    """
    if temperature <= 0:
        raise ValueError("temperature must be greater than zero.")
    if log_top_k < 1:
        raise ValueError("log_top_k must be at least one.")

    n_classes = len(label_chars)

    ## Evaluation mode of the model
    model.eval()
    ## Adding padding size as left to the tokenizer
    tokenizer.padding_side = 'left'

    # Include common tokenizer spellings of each answer label.
    candidate_token_ids = {}

    for label in label_chars:
        variants = {
            label,
            f" {label}",
            label.lower(),
            f" {label.lower()}",
        }

        token_ids = set()

        for variant in variants:
            ids = tokenizer.encode(
                variant,
                add_special_tokens=False,
            )

            if len(ids) == 1:
                token_ids.add(ids[0])

        if not token_ids:
            raise ValueError(
                f"Class {label!r} has no single-token representation. "
                "Sequence-level scoring is required for this tokenizer."
            )

        candidate_token_ids[label] = sorted(token_ids)

    def get_probs(
        prompts: List[str],
    ) -> Tuple[np.ndarray, np.ndarray]:

        conversations = []
        valid_label_lists = []
        for prompt in prompts:
            ## Different questions have different final labels
            valid_labels = extract_choice_labels(prompt)
            unknown_labels = set(valid_labels) - set(label_chars)
            if unknown_labels:
                raise ValueError(
                    "Prompt contains labels not supported by the provider: "
                    f"{sorted(unknown_labels)}"
                )
            valid_label_lists.append(valid_labels)


            conversations.append([
                {
                    "role": "system",
                    "content": mcqa_system_prompt(len(valid_labels)),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
                {
                    "role": "assistant",
                    "content": "Answer: "
                }
            ])

        # Tokenize and pad the complete batch in one operation. In recent
        # Transformers versions apply_chat_template returns a BatchEncoding by
        # default; collecting those objects and passing them as `input_ids` to
        # tokenizer.pad creates an invalid nested structure.
        batch = tokenizer.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=False,
            continue_final_message=True,
            padding=True,
            return_tensors="pt",
            return_dict=True,
        )

        # For a model spread over multiple GPUs, model.device normally refers
        # to the device containing the input embeddings.
        batch = {
            name: tensor.to(model.device)
            for name, tensor in batch.items()
        }

        with torch.inference_mode():
            output = model(**batch)

            # Because inputs are left-padded, position -1 is the next-token
            # prediction position for every prompt.
            next_token_logits = output.logits[:, -1, :].float()

            # Temperature-adjusted full-vocabulary log probabilities.
            vocabulary_log_probs = torch.log_softmax(
                next_token_logits / temperature,
                dim=-1,
            )

        class_scores = []

        for label in label_chars:
            ids = candidate_token_ids[label]

            # Sum probability mass from variants such as "A", " A", and "a".
            label_score = torch.logsumexp(
                vocabulary_log_probs[:, ids],
                dim=-1,
            )

            class_scores.append(label_score)

        class_scores = torch.stack(class_scores, dim=-1)

        # Preserve the runner's fixed global class dimension while assigning
        # exactly zero probability to choices that do not exist in a prompt.
        # For example, an A-D question in an A-E run receives the mask
        # [True, True, True, True, False].
        valid_class_mask = torch.tensor(
            [
                [label in valid_labels for label in label_chars]
                for valid_labels in valid_label_lists
            ],
            dtype=torch.bool,
            device=class_scores.device,
        )
        if not torch.all(valid_class_mask.any(dim=-1)):
            raise ValueError("Every prompt must contain at least one valid class.")

        probability_scores = class_scores.masked_fill(
            ~valid_class_mask, -torch.inf
        )
        class_probs = torch.softmax(probability_scores, dim=-1)

        # The shared runner rejects non-finite scores. Retain a finite score for
        # invalid classes while their probabilities remain exactly zero.
        invalid_log_score = float(np.log(1e-30))
        returned_class_scores = class_scores.masked_fill(
            ~valid_class_mask, invalid_log_score
        )

        class_scores_np = (
            returned_class_scores.cpu().numpy().astype(np.float64)
        )
        class_probs_np = class_probs.cpu().numpy().astype(np.float64)

        # Do not serialize the full vocabulary distribution: for a local model
        # that would make the JSONL log enormous. Record the top alternatives,
        # the complete class distribution, and the exact formatted input.
        n_top = min(log_top_k, vocabulary_log_probs.shape[-1])
        top_logprobs, top_token_ids = torch.topk(
            vocabulary_log_probs, k=n_top, dim=-1
        )
        top_logprobs = top_logprobs.cpu()
        top_token_ids = top_token_ids.cpu()
        input_ids_cpu = batch["input_ids"].detach().cpu()
        attention_mask_cpu = batch["attention_mask"].detach().cpu()

        for prompt_index, prompt in enumerate(prompts):
            valid_ids = input_ids_cpu[prompt_index][
                attention_mask_cpu[prompt_index].bool()
            ].tolist()
            alternatives = []
            for token_id, logprob in zip(
                top_token_ids[prompt_index].tolist(),
                top_logprobs[prompt_index].tolist(),
            ):
                alternatives.append(
                    {
                        "token_id": int(token_id),
                        "token": tokenizer.convert_ids_to_tokens(int(token_id)),
                        "decoded": tokenizer.decode([int(token_id)]),
                        "logprob": float(logprob),
                        "probability": float(np.exp(logprob)),
                    }
                )

            predicted_index = int(np.argmax(class_probs_np[prompt_index]))
            _log_raw_response(
                raw_log_path,
                {
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "provider": "local_transformers_martingale_sampling",
                    "model_name": model_name,
                    "prompt_index": prompt_index,
                    "prompt": prompt,
                    "formatted_prompt": tokenizer.decode(
                        valid_ids, skip_special_tokens=False
                    ),
                    "input_token_count": len(valid_ids),
                    "temperature": float(temperature),
                    "padding_side": tokenizer.padding_side,
                    "pad_token": tokenizer.pad_token,
                    "pad_token_id": tokenizer.pad_token_id,
                    "candidate_token_ids": candidate_token_ids,
                    "valid_labels": valid_label_lists[prompt_index],
                    "valid_class_mask": (
                        valid_class_mask[prompt_index]
                        .detach()
                        .cpu()
                        .tolist()
                    ),
                    "top_next_token_alternatives": alternatives,
                    "class_logprob_scores": class_scores_np[prompt_index].tolist(),
                    "resulting_probs": class_probs_np[prompt_index].tolist(),
                    "predicted_class_index": predicted_index,
                    "predicted_class_label": label_chars[predicted_index],
                    "message_content": None,
                    "note": (
                        "Local provider performs next-token scoring only; "
                        "no text completion is generated."
                    ),
                },
            )

        return class_scores_np, class_probs_np

    get_probs.martingale_sampling_metadata = {
        "provider": "local_transformers",
        "model_identifier": model_name,
        "tokenizer_identifier": tokenizer.name_or_path,
        "class_token_mapping": candidate_token_ids,
        "temperature": temperature,
        "raw_log_path": raw_log_path,
        "logged_top_next_tokens": int(log_top_k),
        "score_type": "local next-token log probability",
        "tokenization_verified": True,
    }

    return get_probs
