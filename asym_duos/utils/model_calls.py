import torch
import numpy as np
import torch.nn.functional as F
import openai
import anthropic

from typing import Callable, List
from openai import PermissionDeniedError

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
    n_samples: int,
) -> Callable[[List[str]], np.ndarray]:
    """OpenAI provider.

    use_logprobs=True  — one API call per prompt using top_logprobs (fast, exact).
    use_logprobs=False — n_samples calls per prompt using temperature sampling.
    """

    client = openai.OpenAI()
    n_classes = len(label_chars)

    def get_probs(prompts: List[str]) -> np.ndarray:
        probs = np.zeros((len(prompts), n_classes), dtype=np.float32)
        for i, prompt in enumerate(prompts):
            if use_logprobs:
                try:
                    resp = client.chat.completions.create(
                        model=model_name,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=1,
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
                for _ in range(n_samples):
                    try:
                        resp = client.chat.completions.create(
                            model=model_name,
                            messages=[{"role": "user", "content": prompt}],
                            max_tokens=1,
                            temperature=1.0,
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
    n_samples: int,
) -> Callable[[List[str]], np.ndarray]:
    """Anthropic provider: sampling-based (no logprob API available)."""
    client = anthropic.Anthropic()
    n_classes = len(label_chars)

    def get_probs(prompts: List[str]) -> np.ndarray:
        probs = np.zeros((len(prompts), n_classes), dtype=np.float32)
        for i, prompt in enumerate(prompts):
            counts = np.zeros(n_classes, dtype=np.float64)
            for _ in range(n_samples):
                resp = client.messages.create(
                    model=model_name,
                    max_tokens=1,
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
