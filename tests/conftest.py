"""
Pytest configuration and shared fixtures.

`embedding_model_path` (session) — guarantees the Qwen3-Embedding GGUF is on
disk, downloading from HuggingFace if absent. Hard-fails (pytest.fail) if it
cannot obtain the file.

`llama_server_bin` (session) — discovers llama-server via env var → PATH →
common build-dir globs under $HOME. Hard-fails with fix instructions if not
found.

`live_embedding_server` (session) — starts a SINGLE llama-server for the
entire test session and tears it down at the end. If a server is already
healthy on the target port, it attaches to it instead of spawning a second
one. All integration tests share this one server.

Override via environment variables:
    EMBED_MODEL_PATH      — full path to the .gguf file
    EMBED_MODEL_REPO      — HuggingFace repo id
    EMBED_MODEL_FILENAME  — filename within that repo
    LLAMA_SERVER_BIN      — explicit path to the llama-server binary
    EMBED_SERVER_PORT     — port for the integration-test server (default 18081)
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Config constants (all overridable via env)
# ---------------------------------------------------------------------------

_MODEL_PATH = Path(
    os.environ.get(
        "EMBED_MODEL_PATH",
        os.path.expanduser("~/.cache/llama.cpp/Qwen3-Embedding-4B-Q8_0.gguf"),
    )
)
_MODEL_REPO = os.environ.get(
    "EMBED_MODEL_REPO",
    "Qwen/Qwen3-Embedding-4B-GGUF",
)
_MODEL_FILENAME = os.environ.get(
    "EMBED_MODEL_FILENAME",
    "Qwen3-Embedding-4B-Q8_0.gguf",
)
INTEG_PORT: int = int(os.environ.get("EMBED_SERVER_PORT", "18081"))


# ---------------------------------------------------------------------------
# llama-server discovery
# ---------------------------------------------------------------------------

def _find_llama_server() -> str | None:
    """Return the path to llama-server, or None if it cannot be found.

    Search order:
    1. LLAMA_SERVER_BIN env var (explicit override)
    2. System PATH
    3. Common cmake build-dir patterns under $HOME
    """
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
# Session fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def embedding_model_path() -> str:
    """Path to the GGUF model; downloads from HuggingFace if missing.

    Hard-fails (not skip) so integration tests never silently pass without
    a real model.
    """
    if _MODEL_PATH.is_file():
        return str(_MODEL_PATH)

    print(f"\n[conftest] Model not found at {_MODEL_PATH}. Attempting download…")
    _MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import hf_hub_download

        local = hf_hub_download(
            repo_id=_MODEL_REPO,
            filename=_MODEL_FILENAME,
            local_dir=str(_MODEL_PATH.parent),
        )
        print(f"[conftest] Downloaded to {local}")
        return local
    except Exception as exc:
        pytest.fail(
            f"Embedding model could not be loaded.\n"
            f"  Expected path : {_MODEL_PATH}\n"
            f"  Tried repo    : {_MODEL_REPO}/{_MODEL_FILENAME}\n"
            f"  Download error: {exc}\n\n"
            f"Fix options:\n"
            f"  • Set EMBED_MODEL_PATH to an existing .gguf file\n"
            f"  • Set EMBED_MODEL_REPO / EMBED_MODEL_FILENAME to the correct HuggingFace location\n"
            f"  • Download manually:\n"
            f"      huggingface-cli download {_MODEL_REPO} {_MODEL_FILENAME} "
            f"--local-dir {_MODEL_PATH.parent}"
        )


@pytest.fixture(scope="session")
def llama_server_bin() -> str:
    """Path to the llama-server binary; hard-fails with instructions if absent."""
    found = _find_llama_server()
    if found:
        return found

    pytest.fail(
        "llama-server binary could not be found.\n\n"
        "Searched:\n"
        "  1. LLAMA_SERVER_BIN environment variable\n"
        "  2. System PATH\n"
        "  3. Common build directories under $HOME (*/llama.cpp/build/bin/llama-server)\n\n"
        "Fix options:\n"
        "  • Add llama-server to your PATH:\n"
        "      export PATH=$PATH:/path/to/llama.cpp/build/bin\n"
        "  • Or set LLAMA_SERVER_BIN=/absolute/path/to/llama-server"
    )


@pytest.fixture(scope="session")
def live_embedding_server(embedding_model_path, llama_server_bin):
    """Start ONE llama-server for the whole integration test session.

    If a server is already responding on INTEG_PORT, this fixture attaches
    to it rather than spawning a second process. The server is only stopped
    at the very end of the session, and only if this fixture started it.

    Yields the EmbeddingConfig so tests can build EmbeddingService instances
    that connect to the running server without re-spawning it.
    """
    from src.embedding.embedder import EmbeddingConfig, _LlamaCppServer

    config = EmbeddingConfig(model_path=embedding_model_path, port=INTEG_PORT)
    server = _LlamaCppServer(config, binary=llama_server_bin)

    # Attaches if already healthy; spawns otherwise. Either way, exactly one
    # server is running on INTEG_PORT after this call.
    server.ensure_running(startup_timeout=120.0)

    yield config

    # Only terminates the process if _this_ fixture spawned it.
    server.stop()
