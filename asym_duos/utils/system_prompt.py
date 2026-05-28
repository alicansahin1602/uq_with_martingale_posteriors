import string
from typing import List


def _label_sequence(n_class: int) -> List[str]:
    """Return the first n_class uppercase letter labels: ['A', 'B', 'C', ...]."""
    if n_class < 1 or n_class > 26:
        raise ValueError(f"n_class must be between 1 and 26, got {n_class}.")
    return list(string.ascii_uppercase[:n_class])


def mcqa_system_prompt(n_class: int) -> str:
    """Build a system prompt for a multiple-choice question-answering assistant.

    Parameters
    ----------
    n_class : int
        Number of answer choices (1–26). Determines which labels are valid
        (A for n_class=1, A–B for n_class=2, A–C for n_class=3, etc.).

    Returns
    -------
    str
        A detailed system prompt ready to be passed as the ``system`` /
        ``"role": "system"`` message to any chat-completion API.

    Notes
    -----
    The prompt deliberately avoids instructing the model to either follow or
    ignore its previous answers.  That neutrality is essential when testing
    the martingale property: the experiment measures whether the model's
    distribution naturally satisfies E[p_{k+1} | history] = p_k.  Prescribing
    consistency *or* independence would confound the measurement.
    """
    labels = _label_sequence(n_class)
    label_list = ", ".join(labels)
    last_label = labels[-1]

    prompt = f"""\
You are a multiple-choice question answering assistant. Your sole task is to \
select the single best answer for each question from the provided set of \
labeled options.

VALID ANSWER LABELS
-------------------
The available answer labels for every question are: {label_list}.
Always choose exactly one label from this set ({label_list}). \
Do not invent new labels or use labels beyond {last_label}.

RESPONSE FORMAT
---------------
Respond with ONLY the single uppercase letter that corresponds to your answer \
(one of: {label_list}). Do not include any explanation, punctuation, \
reasoning, or additional text — just the single letter.

Example of a correct response: A

PREVIOUS ANSWERS
----------------
Some questions may include a line of the form:

    Your previous answers were: <label>, <label>, ...

This line records the answers that were given to the same question in \
earlier rounds of this session. It is provided as factual context about \
what was answered before. Answer each question on its own merits based on \
the question text and the answer choices — do not let the presence or \
absence of a previous-answer history change how carefully you reason about \
the correct option.

SUMMARY
-------
- Read the question and all answer choices carefully.
- Select the most accurate answer from {label_list}.
- Output exactly one letter and nothing else.\
"""
    return prompt
