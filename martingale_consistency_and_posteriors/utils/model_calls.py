import json
import os
import re
import time
import torch
import numpy as np
import torch.nn.functional as F
import openai
import anthropic

from typing import Callable, List, Optional, Tuple
from openai import PermissionDeniedError
from .system_prompt import mcqa_system_prompt, ppr_system_prompt

from google import genai
from google.genai import types
from google.genai.errors import APIError as GoogleAPIError



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
    whether the model is following the PPR one-letter-per-line format.

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
        raise ValueError("OpenAI returned no logprob content for martingale_sampling.")

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
        "Could not find any MCQA class token in OpenAI top_logprobs for "
        "martingale_sampling. Inspect the raw response log and prompt."
    )


def _parse_ppr_response_votes(text: str, label_chars: List[str], N: int = 100) -> np.ndarray:
    """Parse a newline-separated PPR response into per-line vote vectors.

    Reads up to N lines; each line becomes one row of the returned (M, C)
    array (M <= N, fewer if the response had fewer usable lines): a one-hot
    vote for a resolved label, or a uniform 1/C spread when the line can't be
    resolved to a single label. The row-wise cumulative mean of this array is
    theta_n -- the paper's MLE / empirical-frequency estimator (Eq. 6),
    evaluated at every prefix length n = 1..M within a single PPR generation.
    """
    n_classes = len(label_chars)
    label_set = set(label_chars)
    lines = [l.strip().upper() for l in text.strip().split("\n") if l.strip()][:N]
    votes = np.zeros((len(lines), n_classes), dtype=np.float64)
    for i, line in enumerate(lines):
        if line in label_set:
            votes[i, label_chars.index(line)] = 1.0
        else:
            # Verbose models (e.g. Qwen2-7B) emit lines like "The correct answer is B) Mexico".
            # Extract the single standalone label letter, if unambiguous.
            candidates = [c for c in re.findall(r'\b[A-Z]\b', line) if c in label_set]
            if len(candidates) == 1:
                votes[i, label_chars.index(candidates[0])] = 1.0
            else:
                votes[i, :] = 1.0 / n_classes
    return votes


def _cumulative_theta_trajectory(votes: np.ndarray, N: int) -> np.ndarray:
    """votes: (M, C) per-line vote vectors from one PPR generation (M <= N).

    Returns (N, C): theta_n = cumulative mean of the first n votes (the
    paper's empirical-frequency MLE, Eq. 6, evaluated at every prefix
    length). If the generation produced fewer than N usable lines, the
    trajectory is padded by repeating the last available theta_n.
    """
    M, C = votes.shape
    if M == 0:
        return np.full((N, C), 1.0 / C, dtype=np.float64)
    cum = np.cumsum(votes, axis=0) / np.arange(1, M + 1)[:, None]
    if M < N:
        pad = np.repeat(cum[-1:], N - M, axis=0)
        cum = np.concatenate([cum, pad], axis=0)
    return cum[:N]  # already float64: votes (from _parse_ppr_response_votes) is float64


def _parse_ppr_response(text: str, label_chars: List[str], N: int = 100) -> np.ndarray:
    """Parse a newline-separated PPR response into an aggregate empirical distribution.

    Backward-compatible wrapper around _parse_ppr_response_votes: returns the
    mean of the per-line vote vectors (i.e. theta_N, the final aggregate).
    """
    n_classes = len(label_chars)
    votes = _parse_ppr_response_votes(text, label_chars, N)
    if votes.shape[0] == 0:
        return np.ones(n_classes, dtype=np.float64) / n_classes
    return votes.mean(axis=0)


def _extract_ppr_trajectory_from_logprob_content(
    content, label_chars: List[str], N: int
) -> np.ndarray:
    """Read theta_n directly from a Chat Completions logprobs response.

    `content` is `resp.choices[0].logprobs.content` from a single completion
    where the model free-generated a whole PPR answer sequence. Every output
    token whose text resolves to a valid label is one answer position n; its
    top_logprobs (the model's distribution immediately before emitting that
    token) IS theta_n. This mirrors the paper's Figure 1a methodology (exact
    logit-based theta_n) without needing per-step access to raw model
    internals -- OpenAI/DeepSeek expose it through top_logprobs.

    Padded by repeating the last recorded theta_n if the completion produced
    fewer than N answer tokens.
    """
    label_set = set(label_chars)
    n_classes = len(label_chars)
    theta_list = []
    for entry in content:
        tok = entry.token.strip().upper()
        if tok not in label_set:
            continue
        theta_list.append(
            _label_probs_from_top_logprobs(entry.top_logprobs, label_chars)
        )
        if len(theta_list) >= N:
            break
    if not theta_list:
        return np.full((N, n_classes), 1.0 / n_classes, dtype=np.float64)
    traj = np.stack(theta_list)
    if len(theta_list) < N:
        pad = np.repeat(traj[-1:], N - len(theta_list), axis=0)
        traj = np.concatenate([traj, pad], axis=0)
    return traj


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
            if use_logprobs:
                try:
                    resp = _retry_with_backoff(lambda: client.chat.completions.create(
                        model=model_name,
                        messages=[{
                            "role": "system",
                            "content": mcqa_system_prompt(n_classes)
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
                probs[i] = _first_label_probs(content, label_chars)

                _log_raw_response(raw_log_path, {
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "provider": "openai_iterative",
                    "model_name": model_name,
                    "prompt_index": i,
                    "use_logprobs": True,
                    "prompt": prompt,
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
                                "content": mcqa_system_prompt(n_classes)
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
                    if ans in label_chars:
                        counts[label_chars.index(ans)] += 1
                    else:
                        counts += 1.0 / n_classes  # uniform fallback for off-label replies
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
            if use_logprobs:
                try:
                    resp = _retry_with_backoff(lambda: client.chat.completions.create(
                        model=model_name,
                        messages=[{
                            "role": "system",
                            "content": mcqa_system_prompt(n_classes)
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
                probs[i] = _first_label_probs(content, label_chars)

                _log_raw_response(raw_log_path, {
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "provider": "openai_iterative",
                    "model_name": model_name,
                    "prompt_index": i,
                    "use_logprobs": True,
                    "prompt": prompt,
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
                                "content": mcqa_system_prompt(n_classes)
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
                    if ans in label_chars:
                        counts[label_chars.index(ans)] += 1
                    else:
                        counts += 1.0 / n_classes  # uniform fallback for off-label replies
                probs[i] = counts / counts.sum()
        return probs

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
            try:
                response = _retry_with_backoff(
                    lambda: client.chat.completions.create(
                        model=model_name,
                        messages=[
                            {
                                "role": "system",
                                "content": mcqa_system_prompt(n_classes),
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
            scores_i, probs_i = _first_martingale_sampling_logits_and_probs(
                content,
                label_chars,
                missing_probability=missing_probability,
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
        probs = np.zeros((len(prompts), n_classes), dtype=np.float64)
        for i, prompt in enumerate(prompts):
            counts = np.zeros(n_classes, dtype=np.float64)
            for _ in range(n_api_samples):
                resp = client.messages.create(
                    model=model_name,
                    max_tokens=1024,
                    thinking={"type": "enabled"},
                    system=mcqa_system_prompt(n_classes),
                    messages=[{"role": "user", "content": prompt}],
                )
                ans = resp.content[0].text.strip().upper()
                if ans in label_chars:
                    counts[label_chars.index(ans)] += 1
                else:
                    counts += 1.0 / n_classes
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
            if use_logprobs:
                try:
                    resp = _retry_with_backoff(lambda: client.chat.completions.create(
                        model=model_name,
                        messages=[
                            {
                                "role": "system",
                                "content": mcqa_system_prompt(n_classes)
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
                probs[i] = _first_label_probs(content, label_chars)

                _log_raw_response(raw_log_path, {
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "provider": "deepseek_iterative",
                    "model_name": model_name,
                    "prompt_index": i,
                    "use_logprobs": True,
                    "prompt": prompt,
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
                                "content": mcqa_system_prompt(n_classes)
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

                    if ans in label_chars:
                        counts[label_chars.index(ans)] += 1
                    else:
                        counts += 1.0 / n_classes  # uniform fallback for off-label replies

                    _log_raw_response(raw_log_path, {
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                        "provider": "deepseek_direct_sampling",
                        "model_name": model_name,
                        "prompt_index": i,
                        "sample_index": sample_idx,
                        "use_logprobs": False,
                        "prompt": prompt,
                        "finish_reason": resp.choices[0].finish_reason,
                        "message_content": resp.choices[0].message.content or "",
                        "parsed_answer": ans,
                        "matched_label": ans in label_chars,
                        "raw_response": resp.model_dump() if hasattr(resp, "model_dump") else str(resp),
                    })
                probs[i] = counts / counts.sum()
        return probs

    return get_probs


# ---------------------------------------------------------------------------
# PPR Builders
# ---------------------------------------------------------------------------

def _build_ppr_hf_provider(
    model,
    tokenizer,
    label_chars: List[str],
    n_ppr_samples: int,
    device: torch.device,
) -> Callable[[List[str]], np.ndarray]:
    """HuggingFace PPR provider matching the paper's Section 4.1 protocol.

    ONE continuous generation produces the whole answer sequence A_1..A_N
    (prefix_allowed_tokens_fn hard-enforces the LETTER\\n pattern at the
    token level). theta_n = p(A_{n+1} | A_1:n, x_Q) is read EXACTLY from the
    model's own per-step logits (output_scores) at each answer-token
    position -- not estimated by counting -- mirroring the paper's Figure 1a
    methodology (Section 2.1's "direct access to internal beliefs").

    Returns get_theta_trajectory(prompts) -> (batch, n_ppr_samples, C): the
    full theta_n trajectory for n=1..n_ppr_samples, one forward pass per
    prompt.
    """
    n_classes = len(label_chars)
    sys_prompt = ppr_system_prompt(n_classes, N=n_ppr_samples)

    # Per-class token-id variants (bare, space-prefixed, newline-prefixed) so we
    # can both (a) build the flat allowed-token set for constrained decoding and
    # (b) map a generation step's logits back to a per-class probability vector.
    label_ids_by_class: List[List[int]] = []
    _all_label_ids: set = set()
    for c in label_chars:
        ids_for_c: set = set()
        for prefix in ("", " ", "\n"):
            ids = tokenizer.encode(prefix + c, add_special_tokens=False)
            if ids:
                ids_for_c.add(ids[-1])
        label_ids_by_class.append(sorted(ids_for_c))
        _all_label_ids.update(ids_for_c)
    label_token_ids = sorted(_all_label_ids)

    newline_token_ids = tokenizer.encode("\n", add_special_tokens=False)
    if not newline_token_ids:
        newline_token_ids = [tokenizer.encode("\n")[0]]

    def get_theta_trajectory(prompts: List[str]) -> np.ndarray:
        trajectories = np.zeros((len(prompts), n_ppr_samples, n_classes), dtype=np.float64)
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
            if isinstance(tokenized, torch.Tensor):
                input_ids = tokenized.to(device)
            else:
                input_ids = tokenized["input_ids"].to(device)
            input_length = input_ids.shape[1]

            def prefix_allowed_tokens_fn(_batch_id, prefix_ids):
                gen_len = len(prefix_ids) - input_length
                return label_token_ids if gen_len % 2 == 0 else newline_token_ids

            with torch.no_grad():
                outputs = model.generate(
                    input_ids,
                    max_new_tokens=n_ppr_samples * 2,
                    do_sample=True,
                    temperature=1.0,
                    pad_token_id=tokenizer.eos_token_id,
                    prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                    output_scores=True,
                    return_dict_in_generate=True,
                )

            # Even generation steps (0, 2, 4, ...) are answer-letter steps under
            # the alternating LETTER/newline constraint above; each step's
            # pre-sampling logits ARE theta_n for n = step_idx // 2.
            n_recorded = 0
            for step_idx in range(0, len(outputs.scores), 2):
                if n_recorded >= n_ppr_samples:
                    break
                # Computed in float64: the model's own logits are float32, but a
                # dominant class's softmax probability (e.g. 0.999999987) can sit
                # well within float32's ~1.2e-7 epsilon near 1.0 and collapse to
                # exactly 1.0 if softmax/storage stays in float32 -- silently
                # discarding real signal in exactly the highly-skewed distributions
                # this whole pipeline is built to examine.
                step_logits = outputs.scores[step_idx][0].double()  # (vocab_size,)
                class_logits = torch.stack([
                    torch.logsumexp(step_logits[ids], dim=0)
                    if ids else torch.tensor(-1e9, dtype=torch.float64, device=step_logits.device)
                    for ids in label_ids_by_class
                ])
                trajectories[i, n_recorded] = F.softmax(class_logits, dim=-1).cpu().numpy()
                n_recorded += 1

            # Defensive: pad with the last recorded theta if generation was cut short.
            if n_recorded == 0:
                trajectories[i] = 1.0 / n_classes
            elif n_recorded < n_ppr_samples:
                trajectories[i, n_recorded:] = trajectories[i, n_recorded - 1]

        return trajectories

    return get_theta_trajectory



def _build_ppr_openai_provider(
    model_name: str,
    label_chars: List[str],
    n_ppr_samples: int,
    use_logprobs: bool = True,
    api: str = None,
) -> Callable[[List[str]], np.ndarray]:
    """OpenAI PPR provider matching the paper's Section 4.1 protocol: ONE
    continuous completion generates the whole answer sequence A_1..A_N.

    use_logprobs=True  -- theta_n read exactly from top_logprobs at each
                           answer-token position within that single
                           completion (mirrors Figure 1a via the Chat
                           Completions logprobs API).
    use_logprobs=False -- theta_n estimated as the cumulative empirical
                           frequency of the first n parsed answers (the
                           paper's own Eq. 6 MLE), for deployments without
                           logprob access.

    Returns get_theta_trajectory(prompts) -> (batch, n_ppr_samples, C).
    """
    client = openai.OpenAI(api_key=api)
    n_classes = len(label_chars)
    sys_prompt = ppr_system_prompt(n_classes, N=n_ppr_samples)

    def get_theta_trajectory(prompts: List[str]) -> np.ndarray:
        trajectories = np.zeros((len(prompts), n_ppr_samples, n_classes), dtype=np.float64)
        for i, prompt in enumerate(prompts):
            try:
                resp = _retry_with_backoff(lambda: client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=n_ppr_samples * 4,
                    temperature=1.0,
                    logprobs=use_logprobs,
                    top_logprobs=20 if use_logprobs else None,
                ))
            except PermissionDeniedError as e:
                print("status_code:", getattr(e, "status_code", None))
                print("message:", str(e))
                print("body:", getattr(e, "body", None))
                raise
            content = resp.choices[0].logprobs.content if (use_logprobs and resp.choices[0].logprobs) else None
            if use_logprobs and content:
                trajectories[i] = _extract_ppr_trajectory_from_logprob_content(
                    content, label_chars, n_ppr_samples
                )
            else:
                if use_logprobs:
                    print(
                        "[ppr_openai] WARNING: no logprobs.content in response "
                        "(reasoning models typically don't expose per-token logprobs); "
                        "falling back to empirical-frequency theta_n for this prompt."
                    )
                text = resp.choices[0].message.content or ""
                votes = _parse_ppr_response_votes(text, label_chars, n_ppr_samples)
                trajectories[i] = _cumulative_theta_trajectory(votes, n_ppr_samples)
        return trajectories

    return get_theta_trajectory


def _build_ppr_anthropic_provider(
    model_name: str,
    label_chars: List[str],
    n_ppr_samples: int,
    api: str = None,
) -> Callable[[List[str]], np.ndarray]:
    """Anthropic PPR provider matching the paper's Section 4.1 protocol: ONE
    continuous generation produces the whole answer sequence A_1..A_N.

    Anthropic exposes no logprobs API, so theta_n is estimated as the
    cumulative empirical frequency of the first n parsed answers -- the
    paper's own Eq. 6 MLE estimator, evaluated at every prefix length within
    this single generation.

    Returns get_theta_trajectory(prompts) -> (batch, n_ppr_samples, C).
    """
    client = anthropic.Anthropic(api_key=api)
    n_classes = len(label_chars)
    sys_prompt = ppr_system_prompt(n_classes, N=n_ppr_samples)

    def get_theta_trajectory(prompts: List[str]) -> np.ndarray:
        trajectories = np.zeros((len(prompts), n_ppr_samples, n_classes), dtype=np.float64)
        for i, prompt in enumerate(prompts):
            resp = client.messages.create(
                model=model_name,
                max_tokens=n_ppr_samples * 4,
                system=sys_prompt,
                messages=[{"role": "user", "content": prompt}],
            )
            text = resp.content[0].text if resp.content else ""
            votes = _parse_ppr_response_votes(text, label_chars, n_ppr_samples)
            trajectories[i] = _cumulative_theta_trajectory(votes, n_ppr_samples)
        return trajectories

    return get_theta_trajectory


def _build_ppr_deepseek_provider(
    model_name: str,
    label_chars: List[str],
    n_ppr_samples: int,
    use_logprobs: bool = True,
    api: str = None,
    raw_log_path: Optional[str] = None,
) -> Callable[[List[str]], np.ndarray]:
    """DeepSeek PPR provider matching the paper's Section 4.1 protocol: ONE
    continuous completion generates the whole answer sequence A_1..A_N.

    use_logprobs=True  -- theta_n read exactly from top_logprobs at each
                           answer-token position (DeepSeek's API is
                           OpenAI-compatible and supports the same
                           logprobs/top_logprobs fields).
    use_logprobs=False -- theta_n estimated as the cumulative empirical
                           frequency of the first n parsed answers.

    raw_log_path: if set, every raw API response (plus our interpretation of
        it -- whether logprobs.content was present, which fallback fired,
        the parsed message content) is appended as one JSON line to this
        file, so later calls can be inspected for formatting/API issues.

    Returns get_theta_trajectory(prompts) -> (batch, n_ppr_samples, C).
    """
    client = openai.OpenAI(api_key=api, base_url="https://api.deepseek.com")
    n_classes = len(label_chars)
    sys_prompt = ppr_system_prompt(n_classes, N=n_ppr_samples)

    def get_theta_trajectory(prompts: List[str]) -> np.ndarray:
        trajectories = np.zeros((len(prompts), n_ppr_samples, n_classes), dtype=np.float64)
        for i, prompt in enumerate(prompts):
            try:
                resp = _retry_with_backoff(lambda: client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    # deepseek-v4-pro reasons by default unless told not to.
                    # PPR is a mechanical "emit N letters" task -- reasoning
                    # gains it nothing and, worse, can burn the whole
                    # max_tokens budget on reasoning_content before a single
                    # visible answer token is produced (finish_reason="length"
                    # with message.content="" and no logprobs to read at all).
                    extra_body={"thinking": {"type": "disabled"}},
                    max_tokens=n_ppr_samples * 4,
                    top_p=1.0,
                    logprobs=use_logprobs,
                    top_logprobs=20 if use_logprobs else None,
                ))
            except PermissionDeniedError as e:
                print("status_code:", getattr(e, "status_code", None))
                print("message:", str(e))
                print("body:", getattr(e, "body", None))
                raise
            content = resp.choices[0].logprobs.content if (use_logprobs and resp.choices[0].logprobs) else None
            message_content = resp.choices[0].message.content or ""
            fallback_used = not (use_logprobs and content)
            if use_logprobs and content:
                trajectories[i] = _extract_ppr_trajectory_from_logprob_content(
                    content, label_chars, n_ppr_samples
                )
            else:
                if use_logprobs:
                    print(
                        "[ppr_deepseek] WARNING: no logprobs.content in response "
                        "(reasoning-mode responses typically don't expose per-token "
                        "logprobs); falling back to empirical-frequency theta_n for "
                        "this prompt."
                    )
                votes = _parse_ppr_response_votes(message_content, label_chars, n_ppr_samples)
                trajectories[i] = _cumulative_theta_trajectory(votes, n_ppr_samples)

            _log_raw_response(raw_log_path, {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "provider": "deepseek_ppr",
                "model_name": model_name,
                "prompt_index": i,
                "use_logprobs": use_logprobs,
                "prompt": prompt,
                "finish_reason": resp.choices[0].finish_reason,
                "reasoning_content_len": len(getattr(resp.choices[0].message, "reasoning_content", None) or ""),
                "message_content": message_content,
                "has_logprobs_content": content is not None,
                "n_logprob_entries": len(content) if content else 0,
                "fallback_used": fallback_used,
                "theta_n=1": trajectories[i, 0].tolist(),
                "theta_n=N": trajectories[i, -1].tolist(),
                "raw_response": resp.model_dump() if hasattr(resp, "model_dump") else str(resp),
            })
        return trajectories

    return get_theta_trajectory
