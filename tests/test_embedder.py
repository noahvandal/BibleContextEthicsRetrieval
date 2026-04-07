"""
Tests for src/embedding/embedder.py

Unit tests (default):
    uv run pytest tests/test_embedder.py -v

Integration tests (requires model + llama-server):
    uv run pytest tests/test_embedder.py -m integration -v
"""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import httpx
import pytest

from src.embedding.embedder import (
    ChromaConfig,
    EmbeddingConfig,
    EmbeddingService,
    _LlamaCppServer,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def cfg() -> EmbeddingConfig:
    return EmbeddingConfig(model_path="/models/qwen3-embed-q8.gguf", port=18080)


@pytest.fixture
def chroma_cfg(tmp_path) -> ChromaConfig:
    return ChromaConfig(path=str(tmp_path / "chroma_db"), collection_name="test")


def _healthy_response() -> MagicMock:
    r = MagicMock()
    r.status_code = 200
    r.json.return_value = {"status": "ok"}
    return r


def _unhealthy_side_effect(*_args, **_kwargs):
    raise httpx.ConnectError("refused")


def _embed_response(vectors: list[list[float]], shuffled: bool = False) -> MagicMock:
    data = [{"index": i, "embedding": v} for i, v in enumerate(vectors)]
    if shuffled:
        data = list(reversed(data))
    r = MagicMock()
    r.status_code = 200
    r.json.return_value = {"data": data}
    r.raise_for_status = MagicMock()
    return r


@pytest.fixture
def healthy_server():
    with patch("httpx.get", return_value=_healthy_response()):
        yield


# ---------------------------------------------------------------------------
# EmbeddingConfig
# ---------------------------------------------------------------------------

class TestEmbeddingConfig:
    def test_base_url(self, cfg):
        assert cfg.base_url == "http://127.0.0.1:18080"

    def test_defaults(self, cfg):
        assert cfg.host == "127.0.0.1"
        assert cfg.context_size == 8192
        assert cfg.n_gpu_layers == -1
        assert cfg.extra_args == []


class TestChromaConfig:
    def test_defaults(self):
        cc = ChromaConfig()
        assert cc.distance == "cosine"
        assert cc.collection_name == "bible"


# ---------------------------------------------------------------------------
# _LlamaCppServer
# ---------------------------------------------------------------------------

class TestLlamaCppServer:
    def test_skips_spawn_when_already_healthy(self, cfg):
        server = _LlamaCppServer(cfg)
        with patch("httpx.get", return_value=_healthy_response()), \
             patch("subprocess.Popen") as mock_popen:
            server.ensure_running()
            mock_popen.assert_not_called()

    def test_spawns_when_not_healthy(self, cfg):
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None

        call_count = {"n": 0}
        def health(*_a, **_k):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise httpx.ConnectError("refused")
            return _healthy_response()

        with patch("httpx.get", side_effect=health), \
             patch("subprocess.Popen", return_value=mock_proc) as mock_popen, \
             patch("time.sleep"):
            _LlamaCppServer(cfg).ensure_running()

        mock_popen.assert_called_once()
        cmd = mock_popen.call_args[0][0]
        assert "--embeddings" in cmd
        assert cfg.model_path in cmd

    def test_spawn_includes_extra_args(self, cfg):
        cfg.extra_args = ["--threads", "4"]
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None

        call_count = {"n": 0}
        def health(*_a, **_k):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise httpx.ConnectError("refused")
            return _healthy_response()

        with patch("httpx.get", side_effect=health), \
             patch("subprocess.Popen", return_value=mock_proc) as mock_popen, \
             patch("time.sleep"):
            _LlamaCppServer(cfg).ensure_running()

        cmd = mock_popen.call_args[0][0]
        assert "--threads" in cmd and "4" in cmd

    def test_raises_if_process_exits_early(self, cfg):
        mock_proc = MagicMock()
        mock_proc.poll.return_value = 1
        mock_proc.returncode = 1

        with patch("httpx.get", side_effect=_unhealthy_side_effect), \
             patch("subprocess.Popen", return_value=mock_proc), \
             patch("time.sleep"):
            with pytest.raises(RuntimeError, match="exited with code 1"):
                _LlamaCppServer(cfg).ensure_running(startup_timeout=5.0)

    def test_raises_timeout_if_never_healthy(self, cfg):
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None

        with patch("httpx.get", side_effect=_unhealthy_side_effect), \
             patch("subprocess.Popen", return_value=mock_proc), \
             patch("time.sleep"), \
             patch("time.monotonic", side_effect=[0.0, 0.0, 99.0]):
            with pytest.raises(TimeoutError):
                _LlamaCppServer(cfg).ensure_running(startup_timeout=5.0)

    def test_stop_terminates_owned_process(self, cfg):
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None

        call_count = {"n": 0}
        def health(*_a, **_k):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise httpx.ConnectError("refused")
            return _healthy_response()

        with patch("httpx.get", side_effect=health), \
             patch("subprocess.Popen", return_value=mock_proc), \
             patch("time.sleep"):
            server = _LlamaCppServer(cfg)
            server.ensure_running()

        server.stop()
        mock_proc.terminate.assert_called_once()

    def test_stop_noop_when_not_owner(self, cfg):
        with patch("httpx.get", return_value=_healthy_response()):
            server = _LlamaCppServer(cfg)
            server.ensure_running()
        server.stop()   # should not raise

    def test_stop_force_kills_on_timeout(self, cfg):
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        # First wait() (after terminate) raises; second (after kill) succeeds
        mock_proc.wait.side_effect = [subprocess.TimeoutExpired(cmd=[], timeout=5), None]

        call_count = {"n": 0}
        def health(*_a, **_k):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise httpx.ConnectError("refused")
            return _healthy_response()

        with patch("httpx.get", side_effect=health), \
             patch("subprocess.Popen", return_value=mock_proc), \
             patch("time.sleep"):
            server = _LlamaCppServer(cfg)
            server.ensure_running()

        server.stop()
        mock_proc.kill.assert_called_once()


# ---------------------------------------------------------------------------
# EmbeddingService — raw embed
# ---------------------------------------------------------------------------

class TestEmbeddingServiceEmbed:
    def test_init_does_not_start_llama_server(self, cfg):
        with patch.object(_LlamaCppServer, "ensure_running") as mock_ensure:
            EmbeddingService(cfg)
        mock_ensure.assert_not_called()

    def test_single_string_returns_one_vector(self, cfg, healthy_server):
        vec = [0.1, 0.2, 0.3]
        with patch.object(httpx.Client, "post", return_value=_embed_response([vec])):
            assert EmbeddingService(cfg).embed("hello") == [vec]

    def test_list_returns_multiple_vectors(self, cfg, healthy_server):
        vecs = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        with patch.object(httpx.Client, "post", return_value=_embed_response(vecs)):
            assert EmbeddingService(cfg).embed(["a", "b", "c"]) == vecs

    def test_restores_order_when_response_shuffled(self, cfg, healthy_server):
        vecs = [[1.0], [2.0], [3.0]]
        with patch.object(
            httpx.Client, "post", return_value=_embed_response(vecs, shuffled=True)
        ):
            assert EmbeddingService(cfg).embed(["a", "b", "c"]) == vecs

    def test_posts_to_correct_endpoint(self, cfg, healthy_server):
        with patch.object(
            httpx.Client, "post", return_value=_embed_response([[0.0]])
        ) as mock_post:
            EmbeddingService(cfg).embed("test")
        assert mock_post.call_args[0][0].endswith("/v1/embeddings")

    def test_wraps_single_string_in_list(self, cfg, healthy_server):
        with patch.object(
            httpx.Client, "post", return_value=_embed_response([[0.0]])
        ) as mock_post:
            EmbeddingService(cfg).embed("single")
        assert mock_post.call_args.kwargs["json"]["input"] == ["single"]

    def test_http_error_propagates(self, cfg, healthy_server):
        bad = MagicMock()
        bad.raise_for_status.side_effect = httpx.HTTPStatusError(
            "500", request=MagicMock(), response=MagicMock()
        )
        with patch.object(httpx.Client, "post", return_value=bad):
            with pytest.raises(httpx.HTTPStatusError):
                EmbeddingService(cfg).embed("oops")

    def test_context_manager_calls_stop(self, cfg, healthy_server):
        with patch.object(httpx.Client, "post", return_value=_embed_response([[0.0]])):
            svc = EmbeddingService(cfg)
            with patch.object(svc, "stop") as mock_stop:
                with svc:
                    svc.embed("hi")
            mock_stop.assert_called_once()


# ---------------------------------------------------------------------------
# EmbeddingService — ChromaDB
# ---------------------------------------------------------------------------

class TestEmbeddingServiceChroma:
    """Uses a mocked Chroma client to avoid I/O in unit tests."""

    @pytest.fixture
    def svc(self, cfg, healthy_server):
        mock_collection = MagicMock()
        mock_collection.count.return_value = 0

        mock_chroma_client = MagicMock()
        mock_chroma_client.get_or_create_collection.return_value = mock_collection

        fake_chromadb = MagicMock()
        fake_chromadb.PersistentClient.return_value = mock_chroma_client

        with patch("src.embedding.embedder._import_chromadb", return_value=fake_chromadb):
            service = EmbeddingService(
                cfg,
                chroma=ChromaConfig(path="/tmp/fake", collection_name="test"),
            )
        service._mock_collection = mock_collection
        return service

    def test_add_upserts_to_collection(self, svc, healthy_server):
        vecs = [[0.1, 0.2], [0.3, 0.4]]
        texts = ["verse one", "verse two"]
        ids = ["v1", "v2"]
        metas = [{"book": "gen"}, {"book": "john"}]

        with patch.object(httpx.Client, "post", return_value=_embed_response(vecs)):
            svc.add(texts=texts, ids=ids, metadatas=metas)

        svc._mock_collection.upsert.assert_called_once_with(
            ids=ids,
            embeddings=vecs,
            documents=texts,
            metadatas=metas,
        )

    def test_query_passes_embedding_to_collection(self, svc, healthy_server):
        vec = [0.5, 0.6]
        with patch.object(httpx.Client, "post", return_value=_embed_response([vec])):
            svc.query("search text", n_results=3)

        svc._mock_collection.query.assert_called_once()
        kwargs = svc._mock_collection.query.call_args.kwargs
        assert kwargs["query_embeddings"] == [vec]
        assert kwargs["n_results"] == 3

    def test_query_passes_where_filter(self, svc, healthy_server):
        vec = [0.1]
        with patch.object(httpx.Client, "post", return_value=_embed_response([vec])):
            svc.query("text", where={"book": "genesis"})

        kwargs = svc._mock_collection.query.call_args.kwargs
        assert kwargs["where"] == {"book": "genesis"}

    def test_query_omits_where_when_none(self, svc, healthy_server):
        vec = [0.1]
        with patch.object(httpx.Client, "post", return_value=_embed_response([vec])):
            svc.query("text")

        kwargs = svc._mock_collection.query.call_args.kwargs
        assert "where" not in kwargs

    def test_collection_count(self, svc):
        svc._mock_collection.count.return_value = 42
        assert svc.collection_count() == 42

    def test_add_without_chroma_raises(self, cfg, healthy_server):
        svc = EmbeddingService(cfg)
        with pytest.raises(RuntimeError, match="ChromaConfig"):
            svc.add(["text"], ["id"])

    def test_query_without_chroma_raises(self, cfg, healthy_server):
        svc = EmbeddingService(cfg)
        with pytest.raises(RuntimeError, match="ChromaConfig"):
            svc.query("text")


# ===========================================================================
# Integration tests — hit a real llama-server and real ChromaDB
#
# Requires the Qwen3-Embedding model. If the file is missing, conftest.py
# attempts to download it from HuggingFace and FAILS HARD (not skip) if
# that also fails — these tests must always run.
#
# Run with:
#   uv run pytest tests/test_embedder.py -m integration -v
#
# Override paths via environment variables:
#   EMBED_MODEL_PATH      — path to .gguf (default: ~/.cache/llama.cpp/...)
#   EMBED_MODEL_REPO      — HuggingFace repo (default: Qwen/Qwen3-Embedding-4B-GGUF)
#   LLAMA_SERVER_BIN      — path to binary (default: auto-discovered via PATH / build dirs)
#   EMBED_SERVER_PORT     — port to use (default: 18081)
#
# The server starts ONCE per session (session-scoped live_embedding_server
# fixture). If a server is already running on the port, it is reused rather
# than spawning a second one.
# ===========================================================================

@pytest.mark.integration
class TestEmbeddingServiceIntegration:
    """End-to-end: one shared llama-server + per-test ChromaDB collection.

    `live_embedding_server` (session-scoped, defined in conftest.py) ensures
    exactly one llama-server is running for the entire session — if one is
    already up on the port it attaches rather than spawning a second.

    Each test gets a fresh throw-away ChromaDB in a temp directory so tests
    don't interfere with each other.
    """

    @pytest.fixture
    def svc(self, tmp_path, live_embedding_server):
        """Per-test EmbeddingService attached to the shared server."""
        chroma = ChromaConfig(
            path=str(tmp_path / "chroma_db"),
            collection_name="integ_test",
        )
        # live_embedding_server already started (or found) the server;
        # EmbeddingService will attach to it rather than spawn a new one.
        service = EmbeddingService(live_embedding_server, chroma=chroma)
        yield service
        # Closes the HTTP client but does NOT stop the server (_proc is None
        # because this instance never spawned it).
        service.stop()

    # ------------------------------------------------------------------
    # Vector format
    # ------------------------------------------------------------------

    def test_embed_returns_list_of_float_vectors(self, svc):
        vecs = svc.embed("In the beginning God created the heavens and the earth.")
        assert len(vecs) == 1
        vec = vecs[0]
        assert isinstance(vec, list)
        assert len(vec) > 0
        assert all(isinstance(x, float) for x in vec)

    def test_batch_embed_returns_one_vector_per_input(self, svc):
        texts = [
            "In the beginning God created the heavens and the earth.",
            "For God so loved the world that he gave his only begotten Son.",
            "The Lord is my shepherd; I shall not want.",
        ]
        vecs = svc.embed(texts)
        assert len(vecs) == len(texts)

    def test_all_vectors_have_same_dimension(self, svc):
        texts = [
            "Genesis 1:1",
            "John 3:16",
            "Psalm 23:1",
            "A much longer passage to ensure the model always outputs a fixed dimension.",
        ]
        vecs = svc.embed(texts)
        dims = {len(v) for v in vecs}
        assert len(dims) == 1, f"expected uniform dimension, got {dims}"

    def test_different_texts_produce_distinct_vectors(self, svc):
        vecs = svc.embed([
            "In the beginning God created the heavens and the earth.",
            "For God so loved the world that he gave his only begotten Son.",
        ])
        assert vecs[0] != vecs[1]

    # ------------------------------------------------------------------
    # ChromaDB round-trip
    # ------------------------------------------------------------------

    def test_add_stores_correct_document_count(self, svc):
        texts = [
            "In the beginning God created the heavens and the earth.",
            "For God so loved the world that he gave his only begotten Son.",
            "The Lord is my shepherd; I shall not want.",
        ]
        svc.add(
            texts=texts,
            ids=["gen1:1", "john3:16", "psalm23:1"],
            metadatas=[{"book": "genesis"}, {"book": "john"}, {"book": "psalms"}],
        )
        assert svc.collection_count() == len(texts)

    def test_query_returns_requested_result_count(self, svc):
        texts = [
            "In the beginning God created the heavens and the earth.",
            "For God so loved the world that he gave his only begotten Son.",
            "The Lord is my shepherd; I shall not want.",
            "I can do all things through Christ who strengthens me.",
            "Trust in the Lord with all your heart.",
        ]
        svc.add(texts=texts, ids=[f"v:{i}" for i in range(len(texts))])
        results = svc.query("creation of the world", n_results=2)
        assert len(results["documents"][0]) == 2

    def test_query_returns_semantically_relevant_result(self, svc):
        """The closest result to a creation query should be the Genesis verse."""
        svc.add(
            texts=[
                "In the beginning God created the heavens and the earth.",
                "For God so loved the world that he gave his only begotten Son.",
                "The Lord is my shepherd; I shall not want.",
            ],
            ids=["gen1:1", "john3:16", "psalm23:1"],
        )
        results = svc.query("creation and the beginning of the world", n_results=1)
        top_doc = results["documents"][0][0]
        assert "beginning" in top_doc, f"expected creation verse on top, got: {top_doc!r}"

    def test_result_includes_distances_and_metadata(self, svc):
        svc.add(
            texts=["The Lord is my shepherd."],
            ids=["psalm23:1"],
            metadatas=[{"book": "psalms", "chapter": 23}],
        )
        results = svc.query("shepherd", n_results=1)
        assert "distances" in results
        assert "metadatas" in results
        assert results["metadatas"][0][0]["book"] == "psalms"

    def test_metadata_where_filter_scopes_results(self, svc):
        svc.add(
            texts=[
                "In the beginning God created the heavens and the earth.",
                "For God so loved the world.",
            ],
            ids=["gen1:1", "john3:16"],
            metadatas=[{"book": "genesis"}, {"book": "john"}],
        )
        results = svc.query("God", n_results=1, where={"book": "genesis"})
        assert results["metadatas"][0][0]["book"] == "genesis"

    def test_upsert_does_not_duplicate(self, svc):
        svc.add(texts=["The Lord is my shepherd."], ids=["psalm23:1"])
        svc.add(texts=["The Lord is my shepherd."], ids=["psalm23:1"])  # same id
        assert svc.collection_count() == 1

    def test_stored_vectors_accepted_by_chroma(self, svc):
        """A successful add + query proves Chroma accepted the vector dimension."""
        svc.add(texts=["verse one", "verse two", "verse three"], ids=["v1", "v2", "v3"])
        results = svc.query("verse", n_results=3)
        assert len(results["documents"][0]) == 3
