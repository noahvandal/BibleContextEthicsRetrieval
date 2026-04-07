from __future__ import annotations

from pathlib import Path

from src.inference.run_ethics import (
    _build_parser,
    _format_duration,
    _select_port,
    _write_jsonl,
)


def test_parser_defaults_to_small_smoke_test_mode():
    args = _build_parser().parse_args([])

    assert args.dataset == "all"
    assert args.split == "all"
    assert args.limit == 10
    assert args.mode == "plain"
    assert args.generation_port == 8081
    assert args.embed_port == 8080
    assert args.use_dense_retrieval is False
    assert args.reranker_device == "cuda"


def test_write_jsonl_writes_one_json_object_per_line(tmp_path: Path):
    out = tmp_path / "results.jsonl"
    _write_jsonl(out, [{"mode": "plain", "index": 0}, {"mode": "bible", "index": 1}])

    lines = out.read_text(encoding="utf-8").splitlines()
    assert lines == [
        '{"mode": "plain", "index": 0}',
        '{"mode": "bible", "index": 1}',
    ]


def test_select_port_uses_preferred_when_free(monkeypatch):
    monkeypatch.setattr("src.inference.run_ethics._is_healthy_llama_server", lambda *args, **kwargs: False)
    monkeypatch.setattr("src.inference.run_ethics._is_port_free", lambda port, **kwargs: port == 8081)

    assert _select_port(8081) == 8081


def test_select_port_skips_blocked_or_busy_port(monkeypatch):
    monkeypatch.setattr("src.inference.run_ethics._is_healthy_llama_server", lambda port, **kwargs: False)
    monkeypatch.setattr("src.inference.run_ethics._is_port_free", lambda port, **kwargs: port in {8082, 8083})

    assert _select_port(8081, avoid={8082}) == 8083


def test_format_duration_renders_hms():
    assert _format_duration(3661.2) == "01:01:01"
