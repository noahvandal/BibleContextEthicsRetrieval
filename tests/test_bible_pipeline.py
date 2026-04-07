"""
Integration tests for BibleEmbeddingPipeline (src/embedding/run.py).

Embeds Genesis chapter 1 (31 verses) against a real llama-server and
a throw-away ChromaDB, then verifies storage and semantic retrieval.

Run with:
    uv run pytest tests/test_bible_pipeline.py -m integration -v
"""

from __future__ import annotations

import pytest
import pythonbible as bible

from src.datasets.bible_dataset.bible_handler import BibleHandler
from src.embedding.embedder import ChromaConfig, EmbeddingService
from src.embedding.run import BibleEmbeddingPipeline

# Genesis 1 has exactly 31 verses.
GENESIS_1_VERSES = 31
GENESIS_1_FILTER = lambda r: r.book == bible.Book.GENESIS and r.chapter == 1  # noqa: E731


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def handler():
    return BibleHandler()


@pytest.fixture
def svc(tmp_path, live_embedding_server):
    """Fresh Chroma collection per test; shared server for the session."""
    chroma = ChromaConfig(
        path=str(tmp_path / "chroma_db"),
        collection_name="genesis_pipeline_test",
    )
    service = EmbeddingService(live_embedding_server, chroma=chroma)
    yield service
    service.stop()


@pytest.fixture
def pipeline(svc, handler):
    return BibleEmbeddingPipeline(svc, handler, batch_size=16)


# ---------------------------------------------------------------------------
# Embedding Genesis 1
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestGenesis1Embedding:
    """Embed Genesis chapter 1 and verify every verse lands in the store."""

    def test_run_returns_stats_dict(self, pipeline):
        stats = pipeline.run(
            books=[bible.Book.GENESIS],
            filter_fn=GENESIS_1_FILTER,
        )
        assert set(stats.keys()) == {"total", "embedded", "skipped"}

    def test_embeds_all_31_verses(self, pipeline, svc):
        pipeline.run(
            books=[bible.Book.GENESIS],
            filter_fn=GENESIS_1_FILTER,
        )
        assert svc.collection_count() == GENESIS_1_VERSES

    def test_embedded_count_equals_total_when_fresh(self, pipeline):
        stats = pipeline.run(
            books=[bible.Book.GENESIS],
            filter_fn=GENESIS_1_FILTER,
        )
        assert stats["total"] == GENESIS_1_VERSES
        assert stats["embedded"] == GENESIS_1_VERSES
        assert stats["skipped"] == 0

    def test_verse_ids_are_stored_as_strings(self, pipeline, svc, handler):
        pipeline.run(books=[bible.Book.GENESIS], filter_fn=GENESIS_1_FILTER)
        gen_1_1_id = str(bible.get_verse_id(bible.Book.GENESIS, 1, 1))
        existing = svc.exists([gen_1_1_id])
        assert gen_1_1_id in existing

    def test_metadata_is_stored(self, pipeline, svc, handler):
        """Each stored entry should carry reference, book, chapter, verse."""
        pipeline.run(books=[bible.Book.GENESIS], filter_fn=GENESIS_1_FILTER)
        gen_1_1_id = str(bible.get_verse_id(bible.Book.GENESIS, 1, 1))
        result = svc._collection.get(ids=[gen_1_1_id], include=["metadatas"])
        meta = result["metadatas"][0]
        assert meta["book"] == "GENESIS"
        assert meta["chapter"] == 1
        assert meta["verse"] == 1
        assert "Genesis 1:1" in meta["reference"]


# ---------------------------------------------------------------------------
# Skip / override behaviour
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestSkipAndOverride:
    """Verify the already-embedded check and override flag."""

    def test_second_run_skips_all_existing(self, pipeline):
        # First run: embed everything
        pipeline.run(books=[bible.Book.GENESIS], filter_fn=GENESIS_1_FILTER)
        # Second run: everything exists — should be skipped
        stats = pipeline.run(
            books=[bible.Book.GENESIS],
            filter_fn=GENESIS_1_FILTER,
        )
        assert stats["skipped"] == GENESIS_1_VERSES
        assert stats["embedded"] == 0

    def test_override_reembeds_all(self, pipeline):
        # First run: embed everything
        pipeline.run(books=[bible.Book.GENESIS], filter_fn=GENESIS_1_FILTER)
        # Second run with override=True: should re-embed everything
        stats = pipeline.run(
            books=[bible.Book.GENESIS],
            filter_fn=GENESIS_1_FILTER,
            override=True,
        )
        assert stats["embedded"] == GENESIS_1_VERSES
        assert stats["skipped"] == 0

    def test_count_unchanged_after_second_run(self, pipeline, svc):
        """Upserting the same IDs must not grow the collection."""
        pipeline.run(books=[bible.Book.GENESIS], filter_fn=GENESIS_1_FILTER)
        assert svc.collection_count() == GENESIS_1_VERSES
        pipeline.run(
            books=[bible.Book.GENESIS],
            filter_fn=GENESIS_1_FILTER,
            override=True,
        )
        assert svc.collection_count() == GENESIS_1_VERSES


# ---------------------------------------------------------------------------
# Semantic retrieval
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestGenesis1Retrieval:
    """Query the embedded collection and verify semantic relevance."""

    @pytest.fixture(autouse=True)
    def _embed_first(self, pipeline):
        pipeline.run(books=[bible.Book.GENESIS], filter_fn=GENESIS_1_FILTER)

    def test_query_returns_results(self, svc):
        results = svc.query("God created the world", n_results=3)
        assert len(results["documents"][0]) == 3

    def test_creation_query_finds_genesis_1_1(self, svc):
        """'In the beginning' should be the top hit for a creation query."""
        results = svc.query("In the beginning God created", n_results=1)
        top = results["documents"][0][0]
        assert "beginning" in top.lower(), f"unexpected top result: {top!r}"

    def test_light_query_finds_day_3_verse(self, svc):
        """'Let there be light' should surface verse 3."""
        results = svc.query("Let there be light", n_results=3)
        docs = results["documents"][0]
        assert any("light" in d.lower() for d in docs), f"no light verse in: {docs}"

    def test_results_include_metadata_with_chapter_1(self, svc):
        """All results from Genesis 1 should have chapter == 1 in metadata."""
        results = svc.query("the earth was without form", n_results=5)
        for meta in results["metadatas"][0]:
            assert meta["chapter"] == 1
            assert meta["book"] == "GENESIS"

    def test_query_with_where_filter(self, svc):
        """Metadata where-filter should still work on pipeline-stored data."""
        results = svc.query(
            "the spirit of God",
            n_results=1,
            where={"book": "GENESIS"},
        )
        assert results["metadatas"][0][0]["book"] == "GENESIS"

    def test_result_distances_are_floats(self, svc):
        results = svc.query("darkness over the deep", n_results=2)
        for d in results["distances"][0]:
            assert isinstance(d, float)
