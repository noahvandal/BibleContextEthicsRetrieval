"""
Run local-model inference on the Hendrycks ETHICS benchmark.

Examples
--------
Plain prompt over a small validation slice:
    uv run python -m src.inference.run_ethics \
        --generation-model-path ~/.cache/llama.cpp/Qwen3-4B-Q4_K_M.gguf \
        --dataset commonsense \
        --split validation \
        --limit 25

Same slice with Bible retrieval context:
    uv run python -m src.inference.run_ethics \
        --generation-model-path ~/.cache/llama.cpp/Qwen3-4B-Q4_K_M.gguf \
        --mode bible \
        --dataset commonsense \
        --split validation \
        --limit 25 \
        --chroma-path ./chroma_db \
        --collection bible
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

import httpx

from src.datasets.hendrycks_dataset.hendrycks_handler import HendrycksHandler
from src.inference.ethics_inference import (
    ETHICS_CONFIGS,
    EthicsInferenceConfig,
    EthicsInferenceRunner,
    LlamaCppGenerationClient,
    LlamaCppGenerationConfig,
    find_llama_server_binary,
)

if TYPE_CHECKING:
    from src.embedding import EmbeddingService


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run ETHICS inference with a local llama.cpp model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--generation-model-path",
        default=os.environ.get(
            "GEN_MODEL_PATH",
            str(Path.home() / ".cache/llama.cpp/Qwen3-4B-Q4_K_M.gguf"),
        ),
        metavar="PATH",
        help="Path to the GGUF generation model.",
    )
    p.add_argument(
        "--generation-port",
        type=int,
        default=int(os.environ.get("GEN_SERVER_PORT", "8081")),
        metavar="PORT",
        help="Port for the local generation llama-server.",
    )
    p.add_argument(
        "--dataset",
        choices=("all", *ETHICS_CONFIGS),
        default="all",
        help="ETHICS config to evaluate. Default runs all subsets.",
    )
    p.add_argument(
        "--split",
        choices=("all", "train", "validation", "test"),
        default="all",
        help="Dataset split to evaluate. Default uses all splits.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=10,
        metavar="N",
        help="Maximum examples to evaluate. Keep small for smoke tests.",
    )
    p.add_argument(
        "--mode",
        choices=("plain", "bible", "both"),
        default="plain",
        help="Prompting mode to run.",
    )
    p.add_argument(
        "--top-k-verses",
        type=int,
        default=5,
        metavar="N",
        help="How many retrieved Bible verses to prepend in bible mode.",
    )
    p.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        metavar="N",
        help="Maximum completion tokens per example.",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        metavar="T",
        help="Sampling temperature for the local model.",
    )
    p.add_argument(
        "--output-jsonl",
        default=None,
        metavar="PATH",
        help="Optional JSONL file for per-example results.",
    )
    p.add_argument(
        "--embed-model-path",
        default=os.environ.get(
            "EMBED_MODEL_PATH",
            str(Path.home() / ".cache/llama.cpp/Qwen3-Embedding-4B-Q8_0.gguf"),
        ),
        metavar="PATH",
        help="Path to the GGUF embedding model used for Bible retrieval.",
    )
    p.add_argument(
        "--embed-port",
        type=int,
        default=int(os.environ.get("EMBED_SERVER_PORT", "8080")),
        metavar="PORT",
        help="Port for the embedding llama-server.",
    )
    p.add_argument(
        "--chroma-path",
        default="./chroma_db",
        metavar="PATH",
        help="Directory for the Bible ChromaDB persistent store.",
    )
    p.add_argument(
        "--collection",
        default="bible",
        metavar="NAME",
        help="Bible ChromaDB collection name.",
    )
    p.add_argument(
        "--bm25-index-path",
        default="./bm25_index",
        metavar="PATH",
        help="BM25 index path for hybrid Bible retrieval.",
    )
    p.add_argument(
        "--dense-only",
        action="store_true",
        help="Disable BM25 and reranking; use dense retrieval only.",
    )
    p.add_argument(
        "--use-dense-retrieval",
        action="store_true",
        help="Enable dense retrieval for Bible mode. Default is BM25 + reranker only.",
    )
    p.add_argument(
        "--no-reranker",
        action="store_true",
        help="Disable the reranker while keeping dense/BM25 retrieval.",
    )
    p.add_argument(
        "--reranker-device",
        default="cuda",
        metavar="DEVICE",
        help="Device for the reranker model.",
    )
    return p


def _require_file(path_str: str, *, label: str) -> Path:
    path = Path(path_str).expanduser()
    if not path.is_file():
        print(f"ERROR: {label} not found at {path}", file=sys.stderr)
        sys.exit(1)
    return path


def _print_summary(mode: str, results: list) -> None:
    total = len(results)
    comparable = [r for r in results if r.correct is not None]
    correct = sum(1 for r in comparable if r.correct)
    accuracy = (correct / len(comparable)) if comparable else 0.0

    print(f"\n[{mode}]")
    print(f"Examples   : {total}")
    print(f"Comparable : {len(comparable)}")
    print(f"Correct    : {correct}")
    print(f"Accuracy   : {accuracy:.3f}")

    preview = results[: min(3, len(results))]
    for item in preview:
        print(
            f"- idx={item.index} expected={item.expected!r} "
            f"pred={item.parsed_prediction!r} correct={item.correct}"
        )


def _selected_datasets(dataset: str) -> list[str]:
    return list(ETHICS_CONFIGS) if dataset == "all" else [dataset]


def _format_duration(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _count_scope_examples(
    handler: HendrycksHandler,
    datasets: list[str],
    *,
    split: str,
) -> int:
    total = 0
    for dataset_name in datasets:
        total += len(handler.load_dataset(dataset_name, split=split))
    return total


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def _is_port_free(port: int, *, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        return sock.connect_ex((host, port)) != 0


def _is_healthy_llama_server(port: int, *, host: str = "127.0.0.1") -> bool:
    try:
        response = httpx.get(f"http://{host}:{port}/health", timeout=2.0)
        if response.status_code != 200:
            return False
        return response.json().get("status") == "ok"
    except Exception:
        return False


def _select_port(
    preferred: int,
    *,
    host: str = "127.0.0.1",
    avoid: set[int] | None = None,
    search_span: int = 50,
) -> int:
    blocked = avoid or set()

    if preferred not in blocked and (
        _is_healthy_llama_server(preferred, host=host) or _is_port_free(preferred, host=host)
    ):
        return preferred

    for port in range(preferred + 1, preferred + search_span + 1):
        if port in blocked:
            continue
        if _is_healthy_llama_server(port, host=host) or _is_port_free(port, host=host):
            return port

    raise RuntimeError(
        f"Could not find an available llama.cpp port in range "
        f"{preferred}-{preferred + search_span}."
    )


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    gen_model = _require_file(args.generation_model_path, label="generation model")
    binary = find_llama_server_binary()
    if binary is None:
        print(
            "ERROR: llama-server not found. Add it to PATH or set LLAMA_SERVER_BIN.",
            file=sys.stderr,
        )
        sys.exit(1)

    generation_port = _select_port(args.generation_port)
    if args.mode in {"bible", "both"} and (args.dense_only or args.use_dense_retrieval):
        embed_port = _select_port(args.embed_port, avoid={generation_port})
    else:
        embed_port = args.embed_port

    gen_cfg = LlamaCppGenerationConfig(
        model_path=str(gen_model),
        port=generation_port,
    )
    gen_client = LlamaCppGenerationClient(gen_cfg, binary=binary)

    runner_kwargs = {}
    embed_service: EmbeddingService | None = None

    if args.mode in {"bible", "both"}:
        from src.embedding import ChromaConfig, EmbeddingConfig, EmbeddingService
        from src.retrieval import BibleRetriever, RetrievalConfig

        chroma = ChromaConfig(path=args.chroma_path, collection_name=args.collection)
        dense_enabled = args.dense_only or args.use_dense_retrieval
        if dense_enabled:
            embed_model = _require_file(args.embed_model_path, label="embedding model")
            embed_cfg = EmbeddingConfig(model_path=str(embed_model), port=embed_port)
            embed_service = EmbeddingService(embed_cfg, chroma=chroma, binary=binary)
        else:
            embed_model = None
            embed_service = EmbeddingService(
                EmbeddingConfig(model_path="unused-in-bm25-only", port=embed_port),
                chroma=chroma,
                binary=binary,
            )

        if embed_service.collection_count() == 0:
            print(
                "ERROR: Bible collection is empty. Run the embedding pipeline first.",
                file=sys.stderr,
            )
            sys.exit(1)

        retrieval_cfg = RetrievalConfig(
            bm25_index_path=args.bm25_index_path,
            use_dense=dense_enabled,
            use_bm25=not args.dense_only,
            use_reranker=not args.dense_only and not args.no_reranker,
            reranker_device=args.reranker_device,
        )
        runner_kwargs["retriever"] = BibleRetriever(embed_service, config=retrieval_cfg)

    runner = EthicsInferenceRunner(
        gen_client,
        config=None if args.top_k_verses == 5 and args.max_tokens == 128 and args.temperature == 0.0
        else EthicsInferenceConfig(
            top_k_verses=args.top_k_verses,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        ),
        **runner_kwargs,
    )

    print(f"Generation model : {gen_model}")
    print(f"llama-server     : {binary}")
    selected_datasets = _selected_datasets(args.dataset)
    count_handler = HendrycksHandler()
    selected_scope_total = _count_scope_examples(
        count_handler,
        selected_datasets,
        split=args.split,
    )
    full_scope_total = _count_scope_examples(
        count_handler,
        selected_datasets,
        split="all",
    )
    print(
        "Dataset          : "
        f"{', '.join(selected_datasets) if args.dataset == 'all' else args.dataset}"
        f"/{args.split}"
    )
    print(f"Examples in scope: {selected_scope_total:,}")
    print(f"Examples full run: {full_scope_total:,}")
    print(f"Mode             : {args.mode}")
    print(f"Limit            : {args.limit}")
    print(f"Generation port  : {generation_port}")
    if embed_service is not None:
        print(f"Dense retrieval  : {dense_enabled}")
        print(f"Embedding model  : {args.embed_model_path if dense_enabled else 'unused (BM25-only mode)'}")
        if dense_enabled:
            print(f"Embedding port   : {embed_port}")
        print(f"Chroma           : {args.chroma_path} / {args.collection}")
        print(f"BM25 index       : {args.bm25_index_path}")
        print(f"Reranker         : {not args.dense_only and not args.no_reranker}")
        if not args.no_reranker and not args.dense_only:
            print(f"Reranker device  : {args.reranker_device}")
    print()

    jsonl_rows: list[dict] = []
    try:
        if args.mode in {"plain", "both"}:
            plain_start = time.perf_counter()
            plain_processed = 0
            for dataset_name in selected_datasets:
                plain_results = runner.run_dataset(
                    dataset_name,
                    split=args.split,
                    limit=args.limit,
                    with_bible_context=False,
                )
                plain_processed += len(plain_results)
                _print_summary(f"plain:{dataset_name}", plain_results)
                jsonl_rows.extend(
                    [{"mode": "plain", **asdict(r)} for r in plain_results]
                )
            plain_elapsed = time.perf_counter() - plain_start
            if plain_processed:
                print(
                    "Estimated full plain runtime"
                    f" ({full_scope_total:,} examples): "
                    f"{_format_duration((plain_elapsed / plain_processed) * full_scope_total)}"
                )

        if args.mode in {"bible", "both"}:
            bible_start = time.perf_counter()
            bible_processed = 0
            for dataset_name in selected_datasets:
                bible_results = runner.run_dataset(
                    dataset_name,
                    split=args.split,
                    limit=args.limit,
                    with_bible_context=True,
                )
                bible_processed += len(bible_results)
                _print_summary(f"bible:{dataset_name}", bible_results)
                jsonl_rows.extend(
                    [{"mode": "bible", **asdict(r)} for r in bible_results]
                )
            bible_elapsed = time.perf_counter() - bible_start
            if bible_processed:
                print(
                    "Estimated full bible runtime"
                    f" ({full_scope_total:,} examples): "
                    f"{_format_duration((bible_elapsed / bible_processed) * full_scope_total)}"
                )
    finally:
        gen_client.stop()
        if embed_service is not None:
            embed_service.stop()

    if args.output_jsonl:
        out_path = Path(args.output_jsonl).expanduser()
        _write_jsonl(out_path, jsonl_rows)
        print(f"\nWrote results to {out_path}")


if __name__ == "__main__":
    main()
