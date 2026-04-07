"""
BM25 sparse index over the embedded Bible.

Loads all verse text from a ChromaDB collection, tokenizes, and builds a
BM25 index (bm25s backend).  The index can be persisted to disk and reloaded
so startup is fast after the first build (< 1 s for 31 K verses).
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import TYPE_CHECKING

import bm25s

if TYPE_CHECKING:
    from src.embedding.embedder import EmbeddingService
    from src.retrieval.retriever import RetrievalResult


class BM25Index:
    """BM25 index built from a ChromaDB collection.

    Typical usage
    -------------
        idx = BM25Index()
        idx.build(svc)          # fetch from Chroma and tokenise
        idx.save("./bm25_index")

        idx2 = BM25Index.load("./bm25_index")
        hits = idx2.search("love your neighbour", n=50)
    """

    def __init__(self) -> None:
        self._retriever: bm25s.BM25 | None = None
        self._ids: list[int] = []
        self._id_to_idx: dict[int, int] = {}
        self._references: list[str] = []
        self._books: list[str] = []
        self._chapters: list[int] = []
        self._verses: list[int] = []
        self._texts: list[str] = []

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self, svc: "EmbeddingService", *, batch: int = 10_000) -> None:
        """Fetch all documents from ChromaDB and build the BM25 index.

        Parameters
        ----------
        svc:
            EmbeddingService that owns the ChromaDB collection.
        batch:
            Number of documents to pull from Chroma per request.
        """
        collection = svc._collection
        if collection is None:
            raise RuntimeError("EmbeddingService has no ChromaConfig.")
        total = collection.count()
        if total == 0:
            raise RuntimeError("ChromaDB collection is empty — run the embedding pipeline first.")

        ids_all: list[int] = []
        refs_all: list[str] = []
        books_all: list[str] = []
        ch_all: list[int] = []
        v_all: list[int] = []
        texts_all: list[str] = []

        offset = 0
        while offset < total:
            result = collection.get(
                limit=batch,
                offset=offset,
                include=["documents", "metadatas"],
            )
            for doc, meta, vid in zip(
                result["documents"],
                result["metadatas"],
                result["ids"],
            ):
                ids_all.append(int(vid))
                refs_all.append(meta["reference"])
                books_all.append(meta["book"])
                ch_all.append(meta["chapter"])
                v_all.append(meta["verse"])
                texts_all.append(doc)
            offset += batch

        corpus_tokens = bm25s.tokenize(texts_all, stopwords="en")
        retriever = bm25s.BM25()
        retriever.index(corpus_tokens)

        self._retriever = retriever
        self._ids = ids_all
        self._id_to_idx = {vid: i for i, vid in enumerate(ids_all)}
        self._references = refs_all
        self._books = books_all
        self._chapters = ch_all
        self._verses = v_all
        self._texts = texts_all

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Persist the index to *path* (a directory)."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        if self._retriever is None:
            raise RuntimeError("Index has not been built yet.")
        # bm25s uses the argument as a file-name prefix, not a directory
        self._retriever.save(str(path / "bm25_core"))
        with open(path / "meta.pkl", "wb") as f:
            pickle.dump(
                {
                    "ids": self._ids,
                    "references": self._references,
                    "books": self._books,
                    "chapters": self._chapters,
                    "verses": self._verses,
                    "texts": self._texts,
                },
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    @classmethod
    def load(cls, path: str | Path) -> "BM25Index":
        """Load a previously saved index from *path*."""
        path = Path(path)
        obj = cls()
        obj._retriever = bm25s.BM25.load(str(path / "bm25_core"), load_corpus=False)
        with open(path / "meta.pkl", "rb") as f:
            meta = pickle.load(f)
        obj._ids = meta["ids"]
        obj._id_to_idx = {vid: i for i, vid in enumerate(obj._ids)}
        obj._references = meta["references"]
        obj._books = meta["books"]
        obj._chapters = meta["chapters"]
        obj._verses = meta["verses"]
        obj._texts = meta["texts"]
        return obj

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        n: int = 50,
        *,
        books: list[str] | None = None,
    ) -> list[tuple[int, float]]:
        """Return (verse_id, bm25_score) pairs, best first.

        Parameters
        ----------
        query:
            The search string.
        n:
            Maximum number of results to return after filtering.
        books:
            Optional allow-list of Book.name strings (e.g. ``["JOHN"]``).
            When set, over-fetches internally and then filters.
        """
        if self._retriever is None:
            raise RuntimeError("Index not built. Call build() or load() first.")

        fetch = min(n * 4 if books else n, len(self._ids))
        query_tokens = bm25s.tokenize([query], stopwords="en")
        results, scores = self._retriever.retrieve(query_tokens, k=fetch)

        out: list[tuple[int, float]] = []
        for idx, score in zip(results[0].tolist(), scores[0].tolist()):
            if books is not None and self._books[idx] not in books:
                continue
            out.append((self._ids[idx], float(score)))
            if len(out) >= n:
                break
        return out

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get_by_id(self, verse_id: int) -> "RetrievalResult":
        """Build a RetrievalResult for a verse that did not appear in the
        dense results (BM25-only candidate).  distance=1.0 signals no
        cosine score; the reranker score will override it for display."""
        from src.retrieval.retriever import RetrievalResult

        idx = self._id_to_idx[verse_id]
        return RetrievalResult(
            reference=self._references[idx],
            text=self._texts[idx],
            book=self._books[idx],
            chapter=self._chapters[idx],
            verse=self._verses[idx],
            verse_id=verse_id,
            distance=1.0,
        )
