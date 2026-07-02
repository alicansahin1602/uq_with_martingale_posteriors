"""
Retrieval-based few-shot example selection using sentence embeddings.

Provides SimilarQuestionRetriever for use in martingale_helpers.run_retrieval_check.
FAISS is used when available; falls back to brute-force numpy otherwise.
"""

import numpy as np
from typing import List, Tuple, Optional
from sentence_transformers import SentenceTransformer
import faiss

DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def _embed_texts(
    model,
    texts: List[str],
    batch_size: int = 64,
    show_progress: bool = False,
) -> np.ndarray:
    return model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=show_progress,
    )


def _build_index(embeddings: np.ndarray):
    """FAISS (fast) if available, otherwise plain numpy brute-force."""
    try:
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings.astype("float32"))
        return "faiss", index
    except ImportError:
        return "numpy", embeddings


def _query_index(
    query_emb: np.ndarray, index_type: str, index, n: int
) -> Tuple[List[int], List[float]]:
    if index_type == "faiss":
        scores, idxs = index.search(query_emb.reshape(1, -1).astype("float32"), n)
        return idxs[0].tolist(), scores[0].tolist()
    sims = index @ query_emb
    top_idx = np.argsort(-sims)[:n]
    return top_idx.tolist(), sims[top_idx].tolist()


class SimilarQuestionRetriever:
    """Semantic similarity retriever backed by sentence embeddings.

    Build an index from a training dataset, then retrieve the top-k most
    similar examples for any test question.

    Parameters
    ----------
    embedding_model : str
        Sentence-Transformers model name (default: all-MiniLM-L6-v2).
    """

    def __init__(self, embedding_model: str = DEFAULT_EMBEDDING_MODEL):
        self._st = SentenceTransformer(embedding_model)
        self._index_type: Optional[str] = None
        self._index = None
        self._train_bodies: Optional[List[str]] = None
        self._train_answers: Optional[List[str]] = None

    def build_index(
        self,
        train_dataset,
        label_chars: List[str],
        answer_suffix: str = "\nAnswer:",
        batch_size: int = 64,
        logger=None,
    ) -> None:
        """Embed all training prompts and build a similarity index.

        Parameters
        ----------
        train_dataset : dataset with __len__ / __getitem__
            Each item must be a dict with "prompt" (str) and "label" (int).
        label_chars : answer letter strings, e.g. ["A", "B", "C", "D"]
        answer_suffix : suffix to strip from each prompt to get the question body
        """
        bodies, answers = [], []
        for i in range(len(train_dataset)):
            sample = train_dataset[i]
            body = sample["prompt"]
            if body.endswith(answer_suffix):
                body = body[: -len(answer_suffix)]
            bodies.append(body)
            answers.append(label_chars[int(sample["label"])])

        self._train_bodies = bodies
        self._train_answers = answers

        if logger:
            logger.info(f"  [Retriever] Embedding {len(bodies)} training examples ...")
        embs = _embed_texts(self._st, bodies, batch_size=batch_size)
        self._index_type, self._index = _build_index(embs)
        if logger:
            logger.info(
                f"  [Retriever] Index built ({self._index_type}), {len(bodies)} vectors"
            )

    def retrieve(self, query_body: str, n: int) -> List[Tuple[str, str]]:
        """Return [(prompt_body, answer_char)] for the n most similar training examples."""
        q_emb = _embed_texts(self._st, [query_body])[0]
        idxs, _ = _query_index(q_emb, self._index_type, self._index, n)
        return [(self._train_bodies[i], self._train_answers[i]) for i in idxs]

    def retrieve_batch(
        self, query_bodies: List[str], n: int
    ) -> List[List[Tuple[str, str]]]:
        """Retrieve few-shots for a batch of query bodies."""
        q_embs = _embed_texts(self._st, query_bodies)
        return [
            [
                (self._train_bodies[i], self._train_answers[i])
                for i in _query_index(q_emb, self._index_type, self._index, n)[0]
            ]
            for q_emb in q_embs
        ]
