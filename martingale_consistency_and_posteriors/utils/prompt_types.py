"""Small, immutable inputs and outputs for prompt rendering."""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class HistoryState:
    """Past draws paired with the distributions that generated those draws.

    Tuples prevent a hypothetical branch from changing its parent or siblings.
    The full vectors are retained so future renderers can use recent windows.
    """

    sampled_labels: tuple[str, ...] = ()
    previous_probabilities: tuple[tuple[float, ...], ...] = ()

    def __post_init__(self):
        object.__setattr__(self, 'sampled_labels', tuple(self.sampled_labels))
        rows = tuple(tuple(float(x) for x in row) for row in self.previous_probabilities)
        object.__setattr__(self, 'previous_probabilities', rows)
        if len(self.sampled_labels) != len(rows):
            raise ValueError('Each historical label needs its generating distribution.')
        if rows:
            values = np.asarray(rows, dtype=float)
            if (values.ndim != 2 or values.shape[1] < 2
                    or not np.isfinite(values).all() or (values < 0).any()
                    or not np.allclose(values.sum(axis=1), 1, rtol=0, atol=1e-8)):
                raise ValueError('History probabilities must be finite normalized vectors.')

    @property
    def n_steps(self):
        """Number of completed draws, not the number of model calls."""
        return len(self.sampled_labels)

    def advance(self, sampled_label, generating_probs):
        """Return one child; include parent p_t, never the child's future p_t+1."""
        row = tuple(float(x) for x in generating_probs)
        return HistoryState(
            self.sampled_labels + (sampled_label,),
            self.previous_probabilities + (row,),
        )


@dataclass(frozen=True)
class PromptContext:
    """Information a renderer may see; deliberately excludes the gold answer."""

    base_user_prompt: str
    valid_labels: tuple[str, ...]
    history: HistoryState = HistoryState()
    # Probability vectors use the runner's global class order (e.g. A-E for ARC).
    probability_labels: tuple[str, ...] = ()


@dataclass(frozen=True)
class RenderedPrompt:
    """Messages plus explicit labels, so paragraph text cannot change parsing."""

    system_text: str
    user_text: str
    valid_labels: tuple[str, ...]

    def __post_init__(self):
        labels = tuple(self.valid_labels)
        object.__setattr__(self, 'valid_labels', labels)
        if (not labels or len(set(labels)) != len(labels)
                or any(len(x) != 1 or x not in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' for x in labels)):
            raise ValueError('valid_labels must be distinct uppercase answer letters.')
        if not isinstance(self.system_text, str) or not isinstance(self.user_text, str):
            raise TypeError('Prompt messages must be strings.')
