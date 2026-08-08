import string
from typing import List


def _label_sequence(n_class: int) -> List[str]:
    """Return the first n_class uppercase letter labels: ['A', 'B', 'C', ...]."""
    if n_class < 1 or n_class > 26:
        raise ValueError(f"n_class must be between 1 and 26, got {n_class}.")
    return list(string.ascii_uppercase[:n_class])


def ppr_system_prompt(n_class: int, N: int = 100) -> str:
    """Build the PPR system prompt that instructs the model to generate N i.i.d. samples.

    The model outputs N answer letters separated by newlines in a single call.
    Empirical frequencies of those letters estimate the predictive distribution.

    Parameters
    ----------
    n_class : int
        Number of answer choices (1–26).
    N : int
        Number of i.i.d. samples to generate (default 100).
    """
    labels = _label_sequence(n_class)
    choice_str = ", ".join(labels)
    return f"""\
## Task
You are an expert in multiple-choice QA.
Return **a list of answer choices** among ({choice_str}) for the given question below.

### Output Format
- Your output must start immediately with a single letter among ({choice_str}).
- Your separator is a single newline(\\n).
- \\n must appear **only once** after each choice.
- Spaces, additional \\n, and punctuations (periods and commas) are STRICTLY NOT ALLOWED.
- You must output total {N} letters.
- You must generate **a list of answer choices** by following the generation rules below.

### Generation Rule
You are generating a **random sample** from your probability distribution for each choice being the answer for a given question.
Follow the steps.
(STEP 1) First, assign probability weight on each choice being an answer.
        - If a choice is likely to be an answer, it must have higher probability.
        - Conversely, if a choice is more likely to be a wrong answer, then it must have lower probability.
        - You are allowed to give a trivial probability distribution for the choices,
        if and only if you're certain of the answer choice.
        (i.e. multinomial distribution on {n_class}-dim answer choices.)

(STEP 2) Write down each line.
        Each line is a single alphabet sampled from your predictive distribution from STEP 1.
        (If you assigned zero probability weight for some choices, it MUST NOT BE SAMPLED!)
        - Line 1 = a single alphabet sampled from ({choice_str}) with **your probability weight**.
        - Line $i$ ($2 <= i <= {N}$) = a single alphabet independently sampled from ({choice_str}) with **your probability weight**.
        - You must not condition on your previous (Line 1 ~ $i-1$) answers.
        Your answer must be an i.i.d. sample from your distribution.

#### Generation examples
* If you think D is the answer for given question with 100% probability, then output might be :
D\\nD\\nD\\nD\\nD\\nD ...
* If you think B is the most plausible answer, but E can might also be an answer with small probability,
then output might be: B\\nB\\nB\\nB\\nE\\nB ...
* If you think either both A or C can be an answer with high probability, and the others(B, D, E, etc.) cannot
be the answer, then output might be : C\\nA\\nC\\nC\\nA\\nA\\n ... or A\\nC\\nC\\nA\\nA\\nC\\n ...

Now, generate the {N}-line answer list for the given question below.
You must follow the output format and the generating rules!\
"""


def imputation_system_prompt() -> str:
    """System prompt for the imputation step of Scenario 2.

    Instructs the model to generate exactly one new multiple-choice Q&A pair
    that continues the provided sequence in the same topic and style.
    """
    return """\
You are a question generation assistant. Your task is to extend a sequence of \
multiple-choice questions by writing exactly ONE new question-and-answer pair \
in the same topic, style, and difficulty as the examples provided.

OUTPUT FORMAT
-------------
Write the new question in exactly the same format as the examples shown. \
Include the question text, the labeled answer choices (A, B, C, …), and \
a final line "Answer: <letter>" giving the correct answer.

RULES
-----
- Generate exactly one new question-and-answer pair.
- Match the topic, style, and difficulty of the example questions.
- Do not repeat any question already in the sequence.
- Do not include any explanation or additional text beyond the question block.\
"""


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
