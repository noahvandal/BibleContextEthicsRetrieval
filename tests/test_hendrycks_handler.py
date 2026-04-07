from __future__ import annotations

from src.datasets.hendrycks_dataset.hendrycks_handler import HendrycksHandler


def test_load_dataset_passes_through_repo_config_and_split():
    calls: list[tuple[str, str, dict[str, object]]] = []

    def fake_load_dataset(name: str, config: str, **kwargs: object):
        calls.append((name, config, kwargs))
        return [{"label": "1", "input": "help a stranger"}]

    handler = HendrycksHandler(load_dataset_fn=fake_load_dataset, cache_dir="/tmp/ethics-cache")
    dataset = handler.load_dataset("commonsense", split="train")

    assert dataset == [{"label": "1", "input": "help a stranger"}]
    assert calls == [
        (
            "hendrycks/ethics",
            "commonsense",
            {"cache_dir": "/tmp/ethics-cache", "split": "train"},
        )
    ]


def test_get_example_normalizes_commonsense_record():
    handler = HendrycksHandler(load_dataset_fn=lambda *_args, **_kwargs: None)
    dataset = [{"label": "1", "input": "You return a lost wallet."}]

    record = handler.get_example("commonsense", "train", 0, dataset=dataset)

    assert record.dataset == "commonsense"
    assert record.split == "train"
    assert record.index == 0
    assert record.label == 1
    assert record.input == "You return a lost wallet."
    assert record.text == "You return a lost wallet."


def test_get_example_normalizes_utilitarianism_record():
    handler = HendrycksHandler(load_dataset_fn=lambda *_args, **_kwargs: None)
    dataset = [{"baseline": "Attend a quiet dinner.", "less_pleasant": "Attend a loud dinner."}]

    record = handler.get_example("utilitarianism", "validation", 0, dataset=dataset)

    assert record.dataset == "utilitarianism"
    assert record.split == "validation"
    assert record.label is None
    assert record.baseline == "Attend a quiet dinner."
    assert record.less_pleasant == "Attend a loud dinner."
    assert record.text == (
        "Baseline: Attend a quiet dinner.\n"
        "Less pleasant: Attend a loud dinner."
    )


def test_iter_examples_batches_records():
    handler = HendrycksHandler(load_dataset_fn=lambda *_args, **_kwargs: None)
    dataset = [
        {"label": "1", "scenario": "Share food."},
        {"label": "0", "scenario": "Cut the line."},
        {"label": "1", "scenario": "Return the book."},
    ]

    batches = list(handler.iter_examples("justice", split="test", batch_size=2, dataset=dataset))

    assert len(batches) == 2
    assert [record.text for record in batches[0]] == ["Share food.", "Cut the line."]
    assert [record.label for record in batches[1]] == [1]
