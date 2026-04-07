"""
Embedding service backed by a llama.cpp server + ChromaDB.

Manages the llama-server lifecycle (start if not running, attach if already up),
exposes embed() for raw vectors, and optionally owns a ChromaDB collection for
add/query workflows. llama.cpp and Chroma are both implementation details —
callers only see this class.
"""

from __future__ import annotations

import socket
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Sequence

import httpx

if TYPE_CHECKING:
    import chromadb


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class EmbeddingConfig:
    """Tunables for the llama.cpp embedding server."""

    model_path: str
    host: str = "127.0.0.1"
    port: int = 8080
    context_size: int = 8192
    n_gpu_layers: int = -1          # -1 = offload everything to GPU
    extra_args: list[str] = field(default_factory=list)

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


@dataclass
class ChromaConfig:
    """Tunables for the ChromaDB persistent store."""

    path: str = "./chroma_db"
    collection_name: str = "bible"
    distance: str = "cosine"        # "cosine" or "l2"


# ---------------------------------------------------------------------------
# Server management (llama.cpp detail — not exposed to callers)
# ---------------------------------------------------------------------------

class _LlamaCppServer:
    """Starts and stops a llama-server subprocess; attaches to one that is
    already running without taking ownership of it."""

    _HEALTH = "/health"

    def __init__(self, config: EmbeddingConfig, binary: str = "llama-server") -> None:
        self._cfg = config
        self._binary = binary
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()

    def ensure_running(self, startup_timeout: float = 60.0) -> None:
        """Start the server if it is not reachable; no-op otherwise."""
        with self._lock:
            if self._is_healthy():
                return
            self._spawn(startup_timeout)

    def stop(self) -> None:
        """Terminate the server only if *we* started it."""
        with self._lock:
            if self._proc is None:
                return
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait()
            self._proc = None

    def _is_healthy(self) -> bool:
        # llama.cpp returns HTTP 200 while the model is still loading, but
        # the JSON body's "status" field is only "ok" once it is fully ready
        # and able to serve requests.  Checking the body prevents 503s from
        # the embedding endpoint when ensure_running() returns too early.
        try:
            r = httpx.get(self._cfg.base_url + self._HEALTH, timeout=2.0)
            if r.status_code != 200:
                return False
            return r.json().get("status") == "ok"
        except Exception:
            return False

    def _port_in_use(self) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex((self._cfg.host, self._cfg.port)) == 0

    def _spawn(self, timeout: float) -> None:
        if self._port_in_use():
            raise RuntimeError(
                f"Port {self._cfg.port} is already occupied by another process "
                f"that is not a healthy llama-server.\n"
                f"Either stop that process or choose a different port:\n"
                f"  run.py --port 8081\n"
                f"  EmbeddingConfig(model_path=..., port=8081)"
            )

        cmd: list[str] = [
            self._binary,
            "--model",        self._cfg.model_path,
            "--host",         self._cfg.host,
            "--port",         str(self._cfg.port),
            "--ctx-size",     str(self._cfg.context_size),
            "--n-gpu-layers", str(self._cfg.n_gpu_layers),
            "--embeddings",
            *self._cfg.extra_args,
        ]

        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._is_healthy():
                return
            if self._proc.poll() is not None:
                stderr = self._proc.stderr.read().decode(errors="replace").strip()
                raise RuntimeError(
                    f"llama-server exited with code {self._proc.returncode} "
                    f"before becoming healthy.\n"
                    f"stderr: {stderr or '(empty)'}"
                )
            time.sleep(0.5)

        self._proc.kill()
        self._proc = None
        raise TimeoutError(
            f"llama-server did not become healthy within {timeout}s."
        )


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def _import_chromadb() -> Any:
    try:
        import chromadb
    except ImportError as exc:
        raise ImportError(
            "EmbeddingService requires the `chromadb` package when a ChromaConfig "
            "is provided."
        ) from exc
    return chromadb

class EmbeddingService:
    """Embed text and optionally store/query vectors in ChromaDB.

    Construction is intentionally lazy with respect to llama.cpp: creating an
    EmbeddingService does not start or attach to a server. Startup only happens
    when ``ensure_running()``, ``embed()``, ``add()``, or ``query()`` is called.

    Usage — raw embeddings only:
        svc = EmbeddingService(EmbeddingConfig(model_path="..."))
        vec  = svc.embed("Hello world")            # list[list[float]]
        vecs = svc.embed(["Hello", "World"])

    Usage — with ChromaDB:
        svc = EmbeddingService(
            EmbeddingConfig(model_path="..."),
            chroma=ChromaConfig(path="./chroma_db", collection_name="bible"),
        )
        svc.add(
            texts=["In the beginning...", "For God so loved..."],
            ids=["gen1:1", "john3:16"],
            metadatas=[{"book": "genesis"}, {"book": "john"}],
        )
        results = svc.query("creation of the world", n_results=3)

    Context manager (auto-stops the server on exit):
        with EmbeddingService(config, chroma=ChromaConfig()) as svc:
            ...
    """

    _EMBED_PATH = "/v1/embeddings"

    def __init__(
        self,
        config: EmbeddingConfig,
        chroma: ChromaConfig | None = None,
        *,
        binary: str = "llama-server",
        http_timeout: float = 60.0,
    ) -> None:
        self._cfg = config
        self._server = _LlamaCppServer(config, binary)
        self._client = httpx.Client(timeout=http_timeout)

        self._collection: Any | None = None
        if chroma is not None:
            chromadb = _import_chromadb()
            db = chromadb.PersistentClient(path=chroma.path)
            self._collection = db.get_or_create_collection(
                name=chroma.collection_name,
                metadata={"hnsw:space": chroma.distance},
            )

    # ------------------------------------------------------------------
    # Core embedding
    # ------------------------------------------------------------------

    def ensure_running(self, startup_timeout: float = 60.0) -> None:
        """Guarantee the server is up. Called automatically by embed/add/query."""
        self._server.ensure_running(startup_timeout)

    def embed(
        self,
        text: str | Sequence[str],
        *,
        startup_timeout: float = 60.0,
    ) -> list[list[float]]:
        """Return embedding vector(s) for *text*.

        Accepts a single string or a list of strings.
        Always returns a list of vectors (one per input).
        """
        self.ensure_running(startup_timeout)

        inputs: list[str] = [text] if isinstance(text, str) else list(text)
        response = self._client.post(
            self._cfg.base_url + self._EMBED_PATH,
            json={"input": inputs},
        )
        response.raise_for_status()

        data: list[dict] = response.json()["data"]
        data.sort(key=lambda d: d["index"])
        return [d["embedding"] for d in data]

    # ------------------------------------------------------------------
    # ChromaDB helpers
    # ------------------------------------------------------------------

    def add(
        self,
        texts: list[str],
        ids: list[str],
        metadatas: list[dict] | None = None,
    ) -> None:
        """Embed *texts* and upsert them into the ChromaDB collection.

        Requires ChromaConfig to have been passed at construction time.
        """
        if self._collection is None:
            raise RuntimeError("ChromaConfig was not provided — no collection available.")

        embeddings = self.embed(texts)
        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

    def query(
        self,
        text: str,
        *,
        n_results: int = 5,
        where: dict | None = None,
    ) -> chromadb.QueryResult:
        """Embed *text* and return the closest *n_results* documents.

        Requires ChromaConfig to have been passed at construction time.
        Returns the raw ChromaDB QueryResult dict (keys: ids, documents,
        distances, metadatas).
        """
        if self._collection is None:
            raise RuntimeError("ChromaConfig was not provided — no collection available.")

        embedding = self.embed(text)[0]
        kwargs: dict = dict(
            query_embeddings=[embedding],
            n_results=n_results,
            include=["documents", "distances", "metadatas"],
        )
        if where is not None:
            kwargs["where"] = where
        return self._collection.query(**kwargs)

    def exists(self, ids: list[str]) -> set[str]:
        """Return the subset of *ids* already stored in the collection."""
        if self._collection is None:
            raise RuntimeError("ChromaConfig was not provided — no collection available.")
        result = self._collection.get(ids=ids, include=[])
        return set(result["ids"])

    def collection_count(self) -> int:
        """Number of vectors stored in the collection."""
        if self._collection is None:
            raise RuntimeError("ChromaConfig was not provided — no collection available.")
        return self._collection.count()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def stop(self) -> None:
        """Stop the server (only if this instance started it)."""
        self._server.stop()
        self._client.close()

    def __enter__(self) -> "EmbeddingService":
        return self

    def __exit__(self, *_) -> None:
        self.stop()
