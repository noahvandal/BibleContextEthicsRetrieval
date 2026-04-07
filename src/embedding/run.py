"""
Bible embedding pipeline.

Iterates over every verse (or a filtered subset) using BibleHandler,
embeds each batch via the llama.cpp embedding server, and stores the
result in ChromaDB. Already-embedded verses are skipped by default;
pass --override to force recomputation.

Command-line usage
------------------
    uv run python -m src.embedding.run [options]

    --model-path PATH    GGUF model path  (or EMBED_MODEL_PATH env var)
    --chroma-path PATH   ChromaDB dir     [./chroma_db]
    --collection  NAME   Collection name  [bible]
    --books       NAMES  Comma-separated Book names (e.g. GENESIS,JOHN) [all]
    --batch-size  N      Verses per embedding request  [64]
    --override           Re-embed verses even if already stored
    --version     VER    Bible version  [KING_JAMES]
    --port        PORT   llama-server port  [8080]

Programmatic usage
------------------
    from src.embedding import ChromaConfig, EmbeddingConfig, EmbeddingService
    from src.embedding.run import BibleEmbeddingPipeline
    from src.datasets.bible_dataset.bible_handler import BibleHandler
    import pythonbible as bible

    with EmbeddingService(
        EmbeddingConfig(model_path="..."),
        chroma=ChromaConfig(path="./chroma_db"),
    ) as svc:
        pipeline = BibleEmbeddingPipeline(svc, BibleHandler())
        stats = pipeline.run()                          # all books
        stats = pipeline.run(books=[bible.Book.JOHN])   # one book
        stats = pipeline.run(                           # filtered range
            books=[bible.Book.GENESIS],
            filter_fn=lambda r: r.chapter == 1,
        )
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Callable

import pythonbible as bible
from tqdm import tqdm

from src.datasets.bible_dataset.bible_handler import BibleHandler, VerseRecord
from src.embedding.embedder import ChromaConfig, EmbeddingConfig, EmbeddingService


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class BibleEmbeddingPipeline:
    """Embed Bible verses and persist vectors to ChromaDB.

    Parameters
    ----------
    svc:
        A configured EmbeddingService (must have ChromaConfig).
    handler:
        BibleHandler for verse text. Defaults to AMERICAN_STANDARD.
    batch_size:
        Verses sent to the embedding model in a single request. Must be >= 2.
    """

    def __init__(
        self,
        svc: EmbeddingService,
        handler: BibleHandler | None = None,
        *,
        batch_size: int = 64,
    ) -> None:
        if batch_size < 2:
            raise ValueError("batch_size must be >= 2")
        self._svc = svc
        self._handler = handler or BibleHandler()
        self._batch_size = batch_size

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def run(
        self,
        books: list[bible.Book] | None = None,
        *,
        override: bool = False,
        filter_fn: Callable[[VerseRecord], bool] | None = None,
    ) -> dict[str, int]:
        """Embed and store verses, returning a stats dict.

        Parameters
        ----------
        books:
            Books to process. None = every book.
        override:
            False (default) — skip verses already in the collection.
            True            — re-embed and overwrite every entry.
        filter_fn:
            Optional callable applied to each VerseRecord before embedding.
            Return True to include, False to skip. Useful for chapter ranges::

                pipeline.run(
                    books=[bible.Book.GENESIS],
                    filter_fn=lambda r: r.chapter == 1,
                )

        Returns
        -------
        {"total": int, "embedded": int, "skipped": int}
        """
        # Verify the version's text data is actually installed before doing
        # anything — catches missing extras (e.g. pythonbible[kjv]) early.
        self._assert_version_available()

        # Pre-scan the verse structure (no text loaded) to get an accurate
        # total so tqdm can show a percentage bar from the very first batch.
        known_total = self._count_matching(books, filter_fn)

        total = embedded = skipped = 0

        with tqdm(
            total=known_total,
            unit="verse",
            desc="Starting…",
            bar_format=(
                "{desc:<22} {percentage:3.0f}%|{bar:30}|"
                " {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
                " {postfix}"
            ),
            dynamic_ncols=True,
        ) as bar:
            for batch in self._handler.iter_verses(books=books, batch_size=self._batch_size):
                if filter_fn is not None:
                    batch = [r for r in batch if filter_fn(r)]
                if not batch:
                    continue

                total += len(batch)

                to_embed = batch if override else self._new_only(batch)
                skipped += len(batch) - len(to_embed)

                if to_embed:
                    self._svc.add(
                        texts=[r.text for r in to_embed],
                        ids=[str(r.verse_id) for r in to_embed],
                        metadatas=[
                            {
                                "reference": r.reference,
                                "book": r.book.name,
                                "chapter": r.chapter,
                                "verse": r.verse,
                            }
                            for r in to_embed
                        ],
                    )
                    embedded += len(to_embed)

                last = batch[-1]
                bar.set_description(f"{last.reference:<22}")
                bar.set_postfix(embedded=embedded, skipped=skipped, refresh=False)
                bar.update(len(batch))

        print(f"Done.  total={total}  embedded={embedded}  skipped={skipped}")
        return {"total": total, "embedded": embedded, "skipped": skipped}

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _assert_version_available(self) -> None:
        """Raise a clear error if the Bible version's text package isn't installed."""
        import pythonbible as bible
        probe_id = bible.get_verse_id(bible.Book.GENESIS, 1, 1)
        try:
            bible.get_verse_text(probe_id, version=self._handler.version)
        except bible.MissingBiblePackageError as exc:
            version_name = self._handler.version.name
            raise RuntimeError(
                f"Bible version '{version_name}' is not installed.\n"
                f"Add the extra to your dependencies:\n"
                f"  pythonbible[{version_name.lower()}]\n"
                f"Or install it directly:\n"
                f"  uv add pythonbible[{version_name.lower()}]"
            ) from exc

    def _new_only(self, batch: list[VerseRecord]) -> list[VerseRecord]:
        """Return only the records whose IDs are not yet in the collection."""
        existing = self._svc.exists([str(r.verse_id) for r in batch])
        return [r for r in batch if str(r.verse_id) not in existing]

    def _count_matching(
        self,
        books: list[bible.Book] | None,
        filter_fn: Callable[[VerseRecord], bool] | None,
    ) -> int:
        """Fast pre-scan: count matching verses without loading any text.

        Constructs lightweight VerseRecord stubs (empty text/reference) and
        applies filter_fn so the tqdm total is accurate even for chapter ranges.
        """
        target = books if books is not None else list(bible.Book)
        total = 0
        for book in target:
            try:
                n_chapters = bible.get_number_of_chapters(book)
            except Exception:
                continue
            for ch in range(1, n_chapters + 1):
                try:
                    n_verses = bible.get_number_of_verses(book, ch)
                except Exception:
                    continue
                for v in range(1, n_verses + 1):
                    if filter_fn is not None:
                        stub = VerseRecord(
                            verse_id=bible.get_verse_id(book, ch, v),
                            text="",
                            reference="",
                            book=book,
                            chapter=ch,
                            verse=v,
                        )
                        if not filter_fn(stub):
                            continue
                    total += 1
        return total


# ---------------------------------------------------------------------------
# Server binary discovery (self-contained copy; mirrors conftest.py logic)
# ---------------------------------------------------------------------------

def _find_llama_server() -> str | None:
    env_bin = os.environ.get("LLAMA_SERVER_BIN")
    if env_bin and Path(env_bin).is_file():
        return env_bin

    on_path = shutil.which("llama-server")
    if on_path:
        return on_path

    patterns = [
        "*/llama.cpp/build/bin/llama-server",
        "*/llama.cpp/build/llama-server",
        "*/llama.cpp/llama-server",
        "**/llama-server",
    ]
    for pattern in patterns:
        for match in sorted(Path.home().glob(pattern)):
            if match.is_file() and os.access(match, os.X_OK):
                return str(match)

    return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Embed every Bible verse and store in ChromaDB.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--model-path",
        default=os.environ.get(
            "EMBED_MODEL_PATH",
            str(Path.home() / ".cache/llama.cpp/Qwen3-Embedding-4B-Q8_0.gguf"),
        ),
        metavar="PATH",
        help="Path to the GGUF embedding model.",
    )
    p.add_argument(
        "--chroma-path",
        default="./chroma_db",
        metavar="PATH",
        help="Directory for the ChromaDB persistent store.",
    )
    p.add_argument(
        "--collection",
        default="bible",
        metavar="NAME",
        help="ChromaDB collection name.",
    )
    p.add_argument(
        "--books",
        default=None,
        metavar="NAMES",
        help="Comma-separated Book enum names (e.g. GENESIS,JOHN). Default: all.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="Verses per embedding request.",
    )
    p.add_argument(
        "--override",
        action="store_true",
        help="Re-embed verses even if they are already in the collection.",
    )
    p.add_argument(
        "--version",
        default="KING_JAMES",
        metavar="VER",
        help="Bible version (e.g. KING_JAMES, AMERICAN_STANDARD).",
    )
    p.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("EMBED_SERVER_PORT", "8080")),
        metavar="PORT",
        help="Port the llama-server is (or will be) running on.",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    model_path = Path(args.model_path)
    if not model_path.is_file():
        print(f"ERROR: model not found at {model_path}", file=sys.stderr)
        print(
            "Set EMBED_MODEL_PATH or pass --model-path, or download with:\n"
            f"  huggingface-cli download Qwen/Qwen3-Embedding-4B-GGUF "
            f"Qwen3-Embedding-4B-Q8_0.gguf --local-dir {model_path.parent}",
            file=sys.stderr,
        )
        sys.exit(1)

    binary = _find_llama_server()
    if binary is None:
        print(
            "ERROR: llama-server not found. Add it to PATH or set LLAMA_SERVER_BIN.",
            file=sys.stderr,
        )
        sys.exit(1)

    books: list[bible.Book] | None = None
    if args.books:
        try:
            books = [bible.Book[name.strip().upper()] for name in args.books.split(",")]
        except KeyError as exc:
            print(f"ERROR: unknown book name {exc}.", file=sys.stderr)
            sys.exit(1)

    try:
        version = bible.Version[args.version.upper()]
    except KeyError:
        print(f"ERROR: unknown version '{args.version}'.", file=sys.stderr)
        sys.exit(1)

    config = EmbeddingConfig(model_path=str(model_path), port=args.port)
    chroma = ChromaConfig(path=args.chroma_path, collection_name=args.collection)

    print(f"Model   : {model_path}")
    print(f"Server  : {binary}  (port {args.port})")
    print(f"ChromaDB: {args.chroma_path} / {args.collection}")
    print(f"Books   : {[b.name for b in books] if books else 'ALL'}")
    print(f"Override: {args.override}")
    print()

    with EmbeddingService(config, chroma=chroma, binary=binary) as svc:
        pipeline = BibleEmbeddingPipeline(svc, BibleHandler(version=version), batch_size=args.batch_size)
        pipeline.run(books=books, override=args.override)


if __name__ == "__main__":
    main()
