"""
BibleRetriever — semantic search over the embedded Bible.

Retrieval modes
---------------
Dense-only
    Pure cosine similarity via ChromaDB HNSW.  Fast, no extra deps.

Hybrid (default)
    BM25 sparse + dense cosine → Reciprocal Rank Fusion (top ~50 candidates)
    → cross-encoder rerank.  Catches keyword matches that embeddings miss and
    vice-versa; reranker gives the final relevance order.

All parameters are exposed through RetrievalConfig so callers can tune or
disable any stage without touching retriever internals.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import pythonbible as bible
import pythonbible.book_groups as _bg

from src.embedding.embedder import EmbeddingService

if TYPE_CHECKING:
    from src.retrieval.bm25_index import BM25Index
    from src.retrieval.reranker import CrossEncoderReranker


# ---------------------------------------------------------------------------
# Book-group registry
# ---------------------------------------------------------------------------

_GROUP_BOOKS: dict[str, list[str]] = {
    key.lower(): [b.name for b in books]
    for key, books in _bg.BOOK_GROUPS.items()
}

_GROUP_ALIASES: dict[str, str] = {
    "ot":               "old testament",
    "nt":               "new testament",
    "law":              "law",
    "history":          "history",
    "poetry":           "poetry|wisdom",
    "wisdom":           "poetry|wisdom",
    "prophecy":         "prophecy",
    "prophets":         "prophecy",
    "major prophets":   "major prophets",
    "minor prophets":   "minor prophets",
    "gospels":          "gospels",
    "gospel":           "gospels",
    "epistles":         "epistles",
    "paul":             "pauline epistles|paul's epistles|epistles of paul",
    "pauline":          "pauline epistles|paul's epistles|epistles of paul",
    "general epistles": "general epistles",
    "apocalyptic":      "apocalyptic",
    "revelation":       "apocalyptic",
}

AVAILABLE_GROUPS: list[str] = sorted(_GROUP_ALIASES.keys())


def _resolve_group(name: str) -> list[str]:
    """Return a list of Book.name strings for a group alias or full name.

    Raises ValueError if the name is not recognised.
    """
    key = name.lower().strip()
    full = _GROUP_ALIASES.get(key, key)
    if full in _GROUP_BOOKS:
        return _GROUP_BOOKS[full]
    raise ValueError(
        f"Unknown group '{name}'. Available: {', '.join(AVAILABLE_GROUPS)}"
    )


def _books_from_where(where: dict | None) -> list[str] | None:
    """Extract the book allow-list from a ChromaDB where-clause, or None."""
    if where is None:
        return None
    val = where.get("book")
    if val is None:
        return None
    if isinstance(val, str):
        return [val]
    if isinstance(val, dict):
        return val.get("$in")
    return None


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """A single retrieved verse."""

    reference: str      # e.g. "John 3:16"
    text: str
    book: str           # Book enum name, e.g. "JOHN"
    chapter: int
    verse: int
    verse_id: int
    distance: float     # ChromaDB cosine distance (lower = more similar)
    rerank_score: float | None = None   # cross-encoder logit (higher = more relevant)

    @property
    def score(self) -> float:
        """Display score in [0, 1] — higher is more relevant.

        Uses the reranker cosine similarity when available (already in [0, 1]),
        otherwise falls back to 1 - cosine_distance.
        """
        if self.rerank_score is not None:
            return self.rerank_score
        return max(0.0, 1.0 - self.distance)


# ---------------------------------------------------------------------------
# Retrieval config
# ---------------------------------------------------------------------------

@dataclass
class RetrievalConfig:
    """Parameters for the hybrid retrieval pipeline.

    Set use_bm25=False and use_reranker=False to fall back to pure cosine.
    """

    # Candidate pool: how many results to pull from each stage before fusion
    candidates: int = 50

    # Reciprocal Rank Fusion constant (higher k = less aggressive fusion)
    rrf_k: int = 60

    # BM25 sparse retrieval
    use_bm25: bool = True
    bm25_index_path: str = "./bm25_index"

    # Cross-encoder reranking
    use_reranker: bool = True
    reranker_model: str = "microsoft/harrier-oss-v1-0.6b"
    reranker_device: str | None = None   # None = auto (CUDA if available, else CPU)


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------

class BibleRetriever:
    """Semantic retrieval over an embedded Bible stored in ChromaDB.

    Parameters
    ----------
    svc:
        A live EmbeddingService with a ChromaDB collection configured.
    config:
        RetrievalConfig controlling all pipeline parameters.  Defaults to
        hybrid mode (BM25 + dense → RRF → reranker).

    Quick examples
    --------------
        retriever = BibleRetriever(svc)

        # Hybrid search (BM25 + cosine → reranker)
        results = retriever.search("God so loved the world", n=5)

        # Scope to the Gospels
        results = retriever.search_group("shepherd", "Gospels", n=10)

        # Dense-only (no BM25, no reranker)
        cfg = RetrievalConfig(use_bm25=False, use_reranker=False)
        retriever = BibleRetriever(svc, config=cfg)
    """

    def __init__(
        self,
        svc: EmbeddingService,
        config: RetrievalConfig | None = None,
    ) -> None:
        if svc._collection is None:
            raise RuntimeError(
                "EmbeddingService must be initialised with a ChromaConfig."
            )
        self._svc = svc
        self._config = config or RetrievalConfig()
        self._bm25: BM25Index | None = None
        self._reranker: CrossEncoderReranker | None = None

        if self._config.use_bm25:
            self._bm25 = self._load_or_build_bm25()

        if self._config.use_reranker:
            from src.retrieval.reranker import CrossEncoderReranker as _RE
            self._reranker = _RE(
                model_name=self._config.reranker_model,
                device=self._config.reranker_device,
            )

    # ------------------------------------------------------------------
    # Public search API
    # ------------------------------------------------------------------

    def search(self, query: str, *, n: int = 5) -> list[RetrievalResult]:
        """Semantic search across all embedded verses."""
        return self._query(query, n=n, where=None)

    def search_books(
        self,
        query: str,
        books: list[str | bible.Book],
        *,
        n: int = 5,
    ) -> list[RetrievalResult]:
        """Semantic search limited to the given books."""
        names = [
            b.name if isinstance(b, bible.Book) else b.upper()
            for b in books
        ]
        where = {"book": {"$in": names}} if len(names) > 1 else {"book": names[0]}
        return self._query(query, n=n, where=where)

    def search_testament(
        self,
        query: str,
        testament: Literal["OT", "NT"],
        *,
        n: int = 5,
    ) -> list[RetrievalResult]:
        """Semantic search limited to the Old or New Testament."""
        return self.search_group(query, testament, n=n)

    def search_group(
        self,
        query: str,
        group: str,
        *,
        n: int = 5,
    ) -> list[RetrievalResult]:
        """Semantic search limited to a canonical book group.

        Recognised group names (case-insensitive):
            OT, NT, Law, History, Poetry, Wisdom, Prophecy, Prophets,
            Major Prophets, Minor Prophets, Gospels, Epistles, Paul,
            Pauline, General Epistles, Apocalyptic, Revelation
        """
        book_names = _resolve_group(group)
        where = (
            {"book": {"$in": book_names}}
            if len(book_names) > 1
            else {"book": book_names[0]}
        )
        return self._query(query, n=n, where=where)

    # ------------------------------------------------------------------
    # BM25 index management
    # ------------------------------------------------------------------

    def build_bm25_index(self, save_path: str | None = None) -> None:
        """(Re)build the BM25 index from ChromaDB and save it to disk.

        Useful when you want to pre-build the index before starting the
        interactive REPL.  Normally the retriever builds and saves it
        automatically on first startup.

        Parameters
        ----------
        save_path:
            Directory for the index files.  Defaults to
            ``config.bm25_index_path``.
        """
        from src.retrieval.bm25_index import BM25Index

        path = save_path or self._config.bm25_index_path
        n = self._svc.collection_count()
        print(f"Building BM25 index from {n:,} verses…")
        idx = BM25Index()
        idx.build(self._svc)
        idx.save(path)
        self._bm25 = idx
        print(f"BM25 index saved to {path}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_or_build_bm25(self) -> "BM25Index":
        from pathlib import Path
        from src.retrieval.bm25_index import BM25Index

        path = Path(self._config.bm25_index_path)
        if (path / "meta.pkl").exists():
            print(f"Loading BM25 index from {path}…", flush=True)
            return BM25Index.load(path)
        n = self._svc.collection_count()
        print(f"BM25 index not found — building from {n:,} verses…", flush=True)
        idx = BM25Index()
        idx.build(self._svc)
        idx.save(path)
        print(f"BM25 index built and saved to {path}", flush=True)
        return idx

    def _query(
        self,
        query: str,
        *,
        n: int,
        where: dict | None,
    ) -> list[RetrievalResult]:
        if self._bm25 is not None:
            return self._hybrid_query(query, n=n, where=where)
        return self._dense_query(query, n=n, where=where)

    def _dense_query(
        self,
        query: str,
        *,
        n: int,
        where: dict | None,
    ) -> list[RetrievalResult]:
        raw = self._svc.query(query, n_results=n, where=where)
        return [
            RetrievalResult(
                reference=meta["reference"],
                text=doc,
                book=meta["book"],
                chapter=meta["chapter"],
                verse=meta["verse"],
                verse_id=int(vid),
                distance=dist,
            )
            for doc, meta, dist, vid in zip(
                raw["documents"][0],
                raw["metadatas"][0],
                raw["distances"][0],
                raw["ids"][0],
            )
        ]

    def _hybrid_query(
        self,
        query: str,
        *,
        n: int,
        where: dict | None,
    ) -> list[RetrievalResult]:
        cfg = self._config

        # ---- Dense retrieval ----
        raw = self._svc.query(query, n_results=cfg.candidates, where=where)
        dense_results = [
            RetrievalResult(
                reference=meta["reference"],
                text=doc,
                book=meta["book"],
                chapter=meta["chapter"],
                verse=meta["verse"],
                verse_id=int(vid),
                distance=dist,
            )
            for doc, meta, dist, vid in zip(
                raw["documents"][0],
                raw["metadatas"][0],
                raw["distances"][0],
                raw["ids"][0],
            )
        ]
        dense_by_id = {r.verse_id: r for r in dense_results}
        dense_rank = {r.verse_id: i for i, r in enumerate(dense_results)}

        # ---- BM25 sparse retrieval ----
        books = _books_from_where(where)
        bm25_hits = self._bm25.search(query, n=cfg.candidates, books=books)
        bm25_rank = {vid: i for i, (vid, _) in enumerate(bm25_hits)}

        # ---- Reciprocal Rank Fusion ----
        k = cfg.rrf_k
        all_ids = set(dense_rank.keys()) | set(bm25_rank.keys())
        fused = sorted(
            all_ids,
            key=lambda vid: (
                (1.0 / (k + dense_rank[vid]) if vid in dense_rank else 0.0)
                + (1.0 / (k + bm25_rank[vid]) if vid in bm25_rank else 0.0)
            ),
            reverse=True,
        )
        pool_ids = fused[:cfg.candidates]

        # ---- Build candidate RetrievalResults ----
        candidates: list[RetrievalResult] = [
            dense_by_id[vid] if vid in dense_by_id else self._bm25.get_by_id(vid)
            for vid in pool_ids
        ]

        # ---- Cross-encoder reranking ----
        if self._reranker is not None:
            scored = self._reranker.rerank(query, candidates)
            candidates = [
                dataclasses.replace(r, rerank_score=s)
                for r, s in scored
            ]

        return candidates[:n]
