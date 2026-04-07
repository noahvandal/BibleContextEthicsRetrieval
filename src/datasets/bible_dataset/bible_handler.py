"""
BibleHandler — thin, typed wrapper around pythonbible.

Provides:
  • Structured verse retrieval (VerseRecord dataclass)
  • Reference parsing / formatting helpers
  • iter_verses() generator for streaming all (or a subset of) verses
    into an embedding pipeline, with optional batching
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generator

import pythonbible as bible


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class VerseRecord:
    """A single Bible verse with all metadata needed for embedding."""

    verse_id: int           # e.g. 43003016 (John 3:16)
    text: str               # verse text in the configured version
    reference: str          # human-readable, e.g. "John 3:16"
    book: bible.Book
    chapter: int
    verse: int


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------

class BibleHandler:
    """Wraps pythonbible to provide verse retrieval and reference utilities.

    Parameters
    ----------
    version:
        Bible version to use for verse text. Defaults to AMERICAN_STANDARD.
        Other versions require their extra package to be installed
        (e.g. ``pip install pythonbible[kjv]``).
    """

    def __init__(self, version: bible.Version = bible.Version.AMERICAN_STANDARD) -> None:
        self.version = version

    # ------------------------------------------------------------------
    # Single-verse access
    # ------------------------------------------------------------------

    def get_verse(self, verse_id: int) -> VerseRecord:
        """Return a VerseRecord for a raw integer verse ID (e.g. 43003016)."""
        if not bible.is_valid_verse_id(verse_id):
            raise bible.InvalidVerseError(
                f"{verse_id} is not a valid verse id."
            )
        book, chapter, verse = bible.get_book_chapter_verse(verse_id)
        text = bible.get_verse_text(verse_id, version=self.version)
        ref = self._format_verse_ref(book, chapter, verse)
        return VerseRecord(
            verse_id=verse_id,
            text=text,
            reference=ref,
            book=book,
            chapter=chapter,
            verse=verse,
        )

    def get_verse_by_location(
        self, book: bible.Book, chapter: int, verse: int
    ) -> VerseRecord:
        """Return a VerseRecord for a (book, chapter, verse) triple."""
        verse_id = bible.get_verse_id(book, chapter, verse)
        return self.get_verse(verse_id)

    # ------------------------------------------------------------------
    # Reference parsing & formatting
    # ------------------------------------------------------------------

    def find_references(self, text: str) -> list[bible.NormalizedReference]:
        """Extract all scripture references from a free-form string."""
        return bible.get_references(text)

    def format_references(self, references: list[bible.NormalizedReference]) -> str:
        """Format a list of NormalizedReferences into a human-readable string."""
        return bible.format_scripture_references(references)

    def get_verses_for_text(self, text: str) -> list[VerseRecord]:
        """Parse references from *text* and return all matching VerseRecords."""
        references = bible.get_references(text)
        verse_ids = bible.convert_references_to_verse_ids(references)
        records: list[VerseRecord] = []
        for vid in verse_ids:
            try:
                records.append(self.get_verse(vid))
            except (
                bible.InvalidVerseError,
                bible.VersionMissingVerseError,
                bible.VersionMissingChapterError,
                bible.VersionMissingBookError,
            ):
                continue
        return records

    def reference_to_verse_ids(
        self, reference: bible.NormalizedReference
    ) -> tuple[int, ...]:
        """Convert a NormalizedReference to a tuple of integer verse IDs."""
        return bible.convert_reference_to_verse_ids(reference)

    def verse_ids_to_references(
        self, verse_ids: list[int]
    ) -> list[bible.NormalizedReference]:
        """Convert a list of verse IDs back into NormalizedReferences."""
        return bible.convert_verse_ids_to_references(verse_ids)

    # ------------------------------------------------------------------
    # Structure helpers
    # ------------------------------------------------------------------

    def get_books(self) -> list[bible.Book]:
        """All books in the Bible enum (canonical + deuterocanonical)."""
        return list(bible.Book)

    def get_chapter_count(self, book: bible.Book) -> int:
        return bible.get_number_of_chapters(book)

    def get_verse_count(self, book: bible.Book, chapter: int) -> int:
        return bible.get_number_of_verses(book, chapter)

    # ------------------------------------------------------------------
    # Embedding generator
    # ------------------------------------------------------------------

    def iter_verses(
        self,
        books: list[bible.Book] | None = None,
        *,
        batch_size: int = 1,
        skip_missing: bool = True,
    ) -> Generator[VerseRecord | list[VerseRecord], None, None]:
        """Iterate over every verse as a VerseRecord (or batch of VerseRecords).

        Designed for streaming into an embedding pipeline:

            for batch in handler.iter_verses(batch_size=64):
                vecs = embedding_svc.embed([r.text for r in batch])
                embedding_svc.add(
                    texts=[r.text for r in batch],
                    ids=[str(r.verse_id) for r in batch],
                    metadatas=[{"reference": r.reference, "book": r.book.name} for r in batch],
                )

        Parameters
        ----------
        books:
            Subset of books to iterate. Defaults to all books.
        batch_size:
            Yield individual VerseRecords when 1 (default), or lists of
            *batch_size* records when > 1. The final batch may be smaller.
        skip_missing:
            If True (default), silently skip verses not present in the
            configured version. If False, raise on missing verses.
        """
        target_books = books if books is not None else list(bible.Book)
        batch: list[VerseRecord] = []

        for book in target_books:
            try:
                n_chapters = bible.get_number_of_chapters(book)
            except Exception:
                continue

            for chapter in range(1, n_chapters + 1):
                try:
                    n_verses = bible.get_number_of_verses(book, chapter)
                except Exception:
                    continue

                for verse_num in range(1, n_verses + 1):
                    verse_id = bible.get_verse_id(book, chapter, verse_num)
                    try:
                        text = bible.get_verse_text(verse_id, version=self.version)
                    except (
                        bible.VersionMissingVerseError,
                        bible.VersionMissingChapterError,
                        bible.VersionMissingBookError,
                        bible.MissingBiblePackageError,
                    ):
                        if not skip_missing:
                            raise
                        continue

                    record = VerseRecord(
                        verse_id=verse_id,
                        text=text,
                        reference=self._format_verse_ref(book, chapter, verse_num),
                        book=book,
                        chapter=chapter,
                        verse=verse_num,
                    )

                    if batch_size == 1:
                        yield record
                    else:
                        batch.append(record)
                        if len(batch) == batch_size:
                            yield batch
                            batch = []

        if batch:
            yield batch

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _format_verse_ref(self, book: bible.Book, chapter: int, verse: int) -> str:
        ref = bible.NormalizedReference(book, chapter, verse, chapter, verse, book)
        return bible.format_single_reference(ref)
