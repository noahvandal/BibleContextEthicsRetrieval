"""
Bi-encoder reranker for Bible retrieval.

Harrier (microsoft/harrier-oss-v1-0.6b) is a decoder-only embedding model.
Reranking is done by encoding the query and each candidate passage separately,
then scoring by cosine similarity.  The model is loaded eagerly at construction
time so CUDA initialisation happens at startup, not on the first query.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.retrieval.retriever import RetrievalResult


class CrossEncoderReranker:
    """Bi-encoder reranker using harrier-style embedding models.

    Parameters
    ----------
    model_name:
        HuggingFace model ID or local path.
    device:
        ``"cpu"``, ``"cuda"``, ``"mps"``, or ``None`` for auto-detect.
    """

    def __init__(
        self,
        model_name: str = "microsoft/harrier-oss-v1-0.6b",
        device: str | None = None,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._model = None
        self._load()   # eager — fail fast, no cold start on first query

    def _load(self) -> None:
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer

        kwargs: dict = {}
        if self._device is not None:
            kwargs["device"] = self._device

        self._model = SentenceTransformer(self._model_name, **kwargs)

    def rerank(
        self,
        query: str,
        candidates: list["RetrievalResult"],
    ) -> list[tuple["RetrievalResult", float]]:
        """Return *(result, score)* pairs sorted best first.

        Scores are cosine similarities in [0, 1].
        """
        if not candidates:
            return []

        # Encode query — use the model's built-in "query" prompt if configured
        try:
            q_emb = self._model.encode(
                [query],
                prompt_name="query",
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
        except (KeyError, ValueError):
            q_emb = self._model.encode(
                [query],
                normalize_embeddings=True,
                convert_to_numpy=True,
            )

        p_embs = self._model.encode(
            [c.text for c in candidates],
            batch_size=len(candidates),   # single forward pass for all candidates
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        # Dot product of L2-normalised vectors = cosine similarity ∈ [-1, 1]
        sims = (q_emb @ p_embs.T)[0].tolist()
        scores = [max(0.0, float(s)) for s in sims]

        return sorted(
            zip(candidates, scores),
            key=lambda x: x[1],
            reverse=True,
        )
