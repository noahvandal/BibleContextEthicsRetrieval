"""
Interactive Bible Retrieval — chat-style REPL.

Type a query to find the most semantically similar verses. Results show
reference, similarity score, and verse text. Runs until you type /quit
or press Ctrl-C.

Usage:
    uv run python tests/interactive_retrieval.py [options]

    --chroma-path   PATH   ChromaDB directory        [./chroma_db]
    --collection    NAME   Collection name           [bible]
    --model-path    PATH   GGUF model                [EMBED_MODEL_PATH env / default cache]
    --port          PORT   llama-server port         [8081]
    --n             N      Default top-N             [5]
    --bm25-path     PATH   BM25 index directory      [./bm25_index]
    --candidates    N      Fusion candidate pool     [50]
    --reranker      NAME   HuggingFace reranker ID   [arcee-ai/harrier-oss-v1-0.6b]
    --no-bm25              Disable BM25 (dense-only)
    --no-reranker          Disable cross-encoder reranking
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_RESET  = "\033[0m"
_BOLD   = "\033[1m"
_DIM    = "\033[2m"
_CYAN   = "\033[96m"
_YELLOW = "\033[93m"
_GREEN  = "\033[92m"
_RED    = "\033[91m"
_BLUE   = "\033[94m"

def _c(color: str, text: str) -> str:
    return f"{color}{text}{_RESET}"

def _bar(score: float, width: int = 12) -> str:
    filled = round(score * width)
    return _c(_GREEN, "█" * filled) + _c(_DIM, "░" * (width - filled))

def _print_result(i: int, result, *, verbose: bool = True) -> None:
    score_bar = _bar(result.score)
    score_str = f"{result.score:.3f}"
    rerank_tag = ""
    if result.rerank_score is not None:
        rerank_tag = _c(_DIM, f"  [rerank: {result.rerank_score:+.2f}]")
    print(
        f"  {_c(_BOLD, str(i))}.  "
        f"{_c(_CYAN, f'{result.reference:<20}')}"
        f"  {score_bar}  {_c(_YELLOW, score_str)}"
        f"{rerank_tag}"
    )
    if verbose:
        words = result.text.split()
        line, lines = [], []
        for w in words:
            if sum(len(x) + 1 for x in line) + len(w) > 72:
                lines.append(" ".join(line))
                line = [w]
            else:
                line.append(w)
        if line:
            lines.append(" ".join(line))
        for l in lines:
            print(f"       {_c(_DIM, l)}")
    print()

def _print_header(chroma_path: str, collection: str, n: int, scope: str, mode: str) -> None:
    print()
    print(_c(_BOLD, "━" * 60))
    print(_c(_BOLD, "  Bible Retrieval — Interactive Search"))
    print(_c(_DIM,  f"  DB: {chroma_path} / {collection}"))
    print(_c(_DIM,  f"  Top-N: {n}   Scope: {scope}"))
    print(_c(_DIM,  f"  Mode: {mode}"))
    print(_c(_BOLD, "━" * 60))

def _print_help() -> None:
    print()
    print(_c(_BOLD, "  Commands:"))
    rows = [
        ("/n <number>",         "Set top-N results  (e.g. /n 10)"),
        ("/book <NAME>",        "Filter by one book  (e.g. /book JOHN)"),
        ("/books <A,B,...>",    "Filter by book list (e.g. /books PSALMS,PROVERBS)"),
        ("/group <NAME>",       "Filter by group     (e.g. /group Gospels)"),
        ("/ot  /nt",            "Filter to Old / New Testament"),
        ("/clear",              "Clear all filters"),
        ("/scope",              "Show current filter"),
        ("/groups",             "List all group names"),
        ("/verbose  /brief",    "Toggle full verse text on/off"),
        ("/help",               "Show this message"),
        ("/quit  or Ctrl-C",    "Exit"),
    ]
    for cmd, desc in rows:
        print(f"    {_c(_CYAN, f'{cmd:<24}')} {_c(_DIM, desc)}")
    print()

def _print_groups() -> None:
    from src.retrieval import AVAILABLE_GROUPS
    print()
    print(_c(_BOLD, "  Available groups:"))
    cols = 3
    items = sorted(AVAILABLE_GROUPS)
    for i in range(0, len(items), cols):
        row = items[i:i + cols]
        print("    " + "   ".join(f"{_c(_CYAN, g):<30}" for g in row))
    print()

# ---------------------------------------------------------------------------
# REPL
# ---------------------------------------------------------------------------

def _run_repl(retriever, n: int, mode: str) -> None:
    from src.retrieval import AVAILABLE_GROUPS

    scope_label   = "ALL"
    scope_filter  = None
    verbose       = True

    def _search(query: str):
        if scope_filter is None:
            return retriever.search(query, n=n)
        elif scope_filter["type"] == "books":
            return retriever.search_books(query, scope_filter["books"], n=n)
        elif scope_filter["type"] == "group":
            return retriever.search_group(query, scope_filter["group"], n=n)

    _print_header("(loaded)", "(loaded)", n, scope_label, mode)
    _print_help()

    while True:
        try:
            raw = input(_c(_BOLD, "Query: ")).strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{_c(_DIM, 'Goodbye.')}")
            break

        if not raw:
            continue

        # ---- commands ----
        if raw.startswith("/"):
            parts = raw.split(None, 1)
            cmd   = parts[0].lower()
            arg   = parts[1].strip() if len(parts) > 1 else ""

            if cmd in ("/quit", "/exit", "/q"):
                print(_c(_DIM, "Goodbye."))
                break

            elif cmd == "/help":
                _print_help()

            elif cmd == "/groups":
                _print_groups()

            elif cmd == "/n":
                if arg.isdigit() and int(arg) > 0:
                    n = int(arg)
                    print(_c(_DIM, f"  Top-N set to {n}"))
                else:
                    print(_c(_RED, f"  Usage: /n <positive integer>"))

            elif cmd == "/book":
                if arg:
                    scope_filter = {"type": "books", "books": [arg.upper()]}
                    scope_label  = f"book={arg.upper()}"
                    print(_c(_DIM, f"  Scope: {scope_label}"))
                else:
                    print(_c(_RED, "  Usage: /book <BOOK_NAME>"))

            elif cmd == "/books":
                if arg:
                    names = [b.strip().upper() for b in arg.split(",") if b.strip()]
                    scope_filter = {"type": "books", "books": names}
                    scope_label  = f"books={','.join(names)}"
                    print(_c(_DIM, f"  Scope: {scope_label}"))
                else:
                    print(_c(_RED, "  Usage: /books <BOOK1,BOOK2,...>"))

            elif cmd == "/ot":
                scope_filter = {"type": "group", "group": "OT"}
                scope_label  = "Old Testament"
                print(_c(_DIM, f"  Scope: {scope_label}"))

            elif cmd == "/nt":
                scope_filter = {"type": "group", "group": "NT"}
                scope_label  = "New Testament"
                print(_c(_DIM, f"  Scope: {scope_label}"))

            elif cmd == "/group":
                if arg:
                    try:
                        from src.retrieval.retriever import _resolve_group
                        _resolve_group(arg)
                        scope_filter = {"type": "group", "group": arg}
                        scope_label  = arg.title()
                        print(_c(_DIM, f"  Scope: {scope_label}"))
                    except ValueError as e:
                        print(_c(_RED, f"  {e}"))
                else:
                    print(_c(_RED, "  Usage: /group <name>  (try /groups for the list)"))

            elif cmd == "/clear":
                scope_filter = None
                scope_label  = "ALL"
                print(_c(_DIM, "  Scope cleared — searching all verses"))

            elif cmd == "/scope":
                print(_c(_DIM, f"  Current scope: {scope_label}  |  top-N: {n}"))

            elif cmd == "/verbose":
                verbose = True
                print(_c(_DIM, "  Verbose mode on"))

            elif cmd == "/brief":
                verbose = False
                print(_c(_DIM, "  Brief mode on (references only)"))

            else:
                print(_c(_RED, f"  Unknown command '{cmd}'. Type /help for options."))

            continue

        # ---- semantic search ----
        print()
        try:
            results = _search(raw)
        except Exception as e:
            print(_c(_RED, f"  Error: {e}"))
            continue

        if not results:
            print(_c(_DIM, "  No results found."))
            continue

        scope_tag = f"  {_c(_DIM, f'[ {scope_label} ]')}" if scope_label != "ALL" else ""
        print(
            _c(_BOLD, f"  Top {len(results)} results for ")
            + _c(_YELLOW, f'"{raw}"')
            + scope_tag
        )
        print()
        for i, r in enumerate(results, 1):
            _print_result(i, r, verbose=verbose)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Interactive Bible semantic search.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--chroma-path", default="./chroma_db")
    p.add_argument("--collection",  default="bible")
    p.add_argument(
        "--model-path",
        default=os.environ.get(
            "EMBED_MODEL_PATH",
            str(Path.home() / ".cache/llama.cpp/Qwen3-Embedding-4B-Q8_0.gguf"),
        ),
    )
    p.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("EMBED_SERVER_PORT", "8081")),
    )
    p.add_argument("--n", type=int, default=5, help="Default top-N results")

    # Hybrid pipeline options
    p.add_argument("--bm25-path",  default="./bm25_index",
                   help="Directory for the BM25 index")
    p.add_argument("--candidates", type=int, default=50,
                   help="Candidate pool size for fusion / reranking")
    p.add_argument("--reranker",   default="microsoft/harrier-oss-v1-0.6b",
                   help="HuggingFace cross-encoder model for reranking")
    p.add_argument("--no-bm25",      action="store_true",
                   help="Disable BM25 sparse retrieval (pure cosine)")
    p.add_argument("--no-reranker",  action="store_true",
                   help="Disable cross-encoder reranking")
    return p


def main() -> None:
    args = _build_parser().parse_args()

    model_path = Path(args.model_path)
    if not model_path.is_file():
        print(_c(_RED, f"ERROR: model not found at {model_path}"), file=sys.stderr)
        sys.exit(1)

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.embedding.run import _find_llama_server
    binary = _find_llama_server()
    if binary is None:
        print(_c(_RED, "ERROR: llama-server not found. Set LLAMA_SERVER_BIN or add to PATH."), file=sys.stderr)
        sys.exit(1)

    from src.embedding.embedder import ChromaConfig, EmbeddingConfig, EmbeddingService
    from src.retrieval import BibleRetriever, RetrievalConfig

    use_bm25     = not args.no_bm25
    use_reranker = not args.no_reranker

    mode_parts = []
    if use_bm25:
        mode_parts.append("BM25")
    mode_parts.append("dense")
    if use_reranker and use_bm25:
        mode_parts.append("rerank")
    elif use_reranker:
        mode_parts.append("rerank")
    mode = " + ".join(mode_parts) if use_bm25 else "dense-only"
    if use_reranker:
        mode += " + rerank"
    # Clean up the mode string
    if use_bm25 and use_reranker:
        mode = "BM25 + dense → rerank"
    elif use_bm25:
        mode = "BM25 + dense (no rerank)"
    elif use_reranker:
        mode = "dense + rerank"
    else:
        mode = "dense-only"

    retrieval_cfg = RetrievalConfig(
        candidates=args.candidates,
        use_bm25=use_bm25,
        bm25_index_path=args.bm25_path,
        use_reranker=use_reranker,
        reranker_model=args.reranker,
    )

    config = EmbeddingConfig(model_path=str(model_path), port=args.port)
    chroma = ChromaConfig(path=args.chroma_path, collection_name=args.collection)

    print(_c(_DIM, f"Connecting to llama-server on port {args.port}…"))

    with EmbeddingService(config, chroma=chroma, binary=binary) as svc:
        retriever = BibleRetriever(svc, config=retrieval_cfg)
        _run_repl(retriever, n=args.n, mode=mode)


if __name__ == "__main__":
    main()
