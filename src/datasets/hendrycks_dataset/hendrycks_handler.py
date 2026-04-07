"""
HendrycksHandler — thin, typed wrapper around the Hugging Face ETHICS dataset.

Provides:
  - Direct access to ``hendrycks/ethics`` via ``datasets.load_dataset``
  - Structured example retrieval (HendrycksRecord dataclass)
  - iter_examples() generator for streaming one config / split into a pipeline
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generator, Literal, Mapping


EthicsConfig = Literal[
    "commonsense",
    "deontology",
    "justice",
    "utilitarianism",
    "virtue",
]
EthicsConfigSelector = EthicsConfig | Literal["all"]
EthicsSplit = Literal["train", "validation", "test"]
EthicsSplitSelector = EthicsSplit | Literal["all"]

ETHICS_CONFIGS: tuple[EthicsConfig, ...] = (
    "commonsense",
    "deontology",
    "justice",
    "utilitarianism",
    "virtue",
)

_HF_SPLIT_ALIASES: dict[str, EthicsSplit] = {
    "train": "train",
    "test": "validation",
    "test_hard": "test",
    "validation": "validation",
}

_ETHICS_FILES: dict[EthicsSplit, str] = {
    "train": "train.csv",
    "validation": "test.csv",
    "test": "test_hard.csv",
}

_URL_BASE = "https://huggingface.co/datasets/hendrycks/ethics/resolve/main/data"


@dataclass(frozen=True, slots=True)
class HendrycksRecord:
    """A normalized ETHICS example with both canonical and raw fields."""

    dataset: EthicsConfig
    split: EthicsSplit
    index: int
    text: str
    label: int | None = None
    input: str | None = None
    scenario: str | None = None
    excuse: str | None = None
    baseline: str | None = None
    less_pleasant: str | None = None


class HendrycksHandler:
    """Load and normalize examples from the Hendrycks ETHICS benchmark."""

    def __init__(
        self,
        dataset_name: str = "hendrycks/ethics",
        *,
        cache_dir: str | None = None,
        load_dataset_fn: Any | None = None,
    ) -> None:
        self.dataset_name = dataset_name
        self.cache_dir = cache_dir
        self._load_dataset_fn = load_dataset_fn

    def get_configs(self) -> tuple[EthicsConfig, ...]:
        """Return the supported ETHICS dataset configs."""
        return ETHICS_CONFIGS

    def load_dataset(
        self,
        config: EthicsConfig = "commonsense",
        *,
        split: EthicsSplitSelector | None = None,
        **kwargs: Any,
    ) -> Any:
        """Load a full config or one split from Hugging Face datasets."""
        if config not in ETHICS_CONFIGS:
            raise ValueError(f"Unsupported ETHICS config: {config}")

        if split == "all":
            return self._load_all_splits(config, **kwargs)

        if self._load_dataset_fn is None:
            return self._load_ethics_csv_dataset(config, split=split, **kwargs)

        load_dataset_fn = self._load_dataset_fn
        load_kwargs: dict[str, Any] = dict(kwargs)
        if self.cache_dir is not None:
            load_kwargs.setdefault("cache_dir", self.cache_dir)
        if split is not None:
            load_kwargs["split"] = split

        return load_dataset_fn(self.dataset_name, config, **load_kwargs)

    def get_example(
        self,
        config: EthicsConfig,
        split: EthicsSplit,
        index: int,
        *,
        dataset: Any | None = None,
    ) -> HendrycksRecord:
        """Return one normalized example from a config / split."""
        split_dataset = dataset if dataset is not None else self.load_dataset(config, split=split)
        row = split_dataset[index]
        return self._build_record(config, split, index, row)

    def iter_examples(
        self,
        config: EthicsConfig = "commonsense",
        *,
        split: EthicsSplit = "train",
        batch_size: int = 1,
        dataset: Any | None = None,
    ) -> Generator[HendrycksRecord | list[HendrycksRecord], None, None]:
        """Iterate over normalized ETHICS examples, optionally in batches."""
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")

        split_dataset = dataset if dataset is not None else self.load_dataset(config, split=split)
        batch: list[HendrycksRecord] = []

        for index, row in enumerate(split_dataset):
            record = self._build_record(config, split, index, row)
            if batch_size == 1:
                yield record
            else:
                batch.append(record)
                if len(batch) == batch_size:
                    yield batch
                    batch = []

        if batch:
            yield batch

    def _build_record(
        self,
        config: EthicsConfig,
        split: EthicsSplit,
        index: int,
        row: Mapping[str, Any],
    ) -> HendrycksRecord:
        """Normalize one raw ETHICS row into a single typed record."""
        label = self._coerce_optional_int(row.get("label"))
        input_text = self._coerce_optional_str(row.get("input"))
        scenario = self._coerce_optional_str(row.get("scenario"))
        excuse = self._coerce_optional_str(row.get("excuse"))
        baseline = self._coerce_optional_str(row.get("baseline"))
        less_pleasant = self._coerce_optional_str(row.get("less_pleasant"))

        if config == "commonsense":
            text = input_text or ""
        elif config == "deontology":
            text = self._join_parts(
                ("Scenario", scenario),
                ("Excuse", excuse),
            )
        elif config in {"justice", "virtue"}:
            text = scenario or ""
        elif config == "utilitarianism":
            text = self._join_parts(
                ("Baseline", baseline),
                ("Less pleasant", less_pleasant),
            )
        else:
            raise ValueError(f"Unsupported ETHICS config: {config}")

        return HendrycksRecord(
            dataset=config,
            split=split,
            index=index,
            text=text,
            label=label,
            input=input_text,
            scenario=scenario,
            excuse=excuse,
            baseline=baseline,
            less_pleasant=less_pleasant,
        )

    @staticmethod
    def hf_split_name(split: str) -> str:
        """Map raw ETHICS file split names to Hugging Face split names."""
        try:
            return _HF_SPLIT_ALIASES[split]
        except KeyError as exc:
            raise ValueError(f"Unsupported ETHICS split: {split}") from exc

    @staticmethod
    def _join_parts(*parts: tuple[str, str | None]) -> str:
        lines = [f"{label}: {value}" for label, value in parts if value]
        return "\n".join(lines)

    @staticmethod
    def _coerce_optional_int(value: Any) -> int | None:
        if value is None or value == "":
            return None
        return int(value)

    @staticmethod
    def _coerce_optional_str(value: Any) -> str | None:
        if value is None:
            return None
        return str(value)

    @staticmethod
    def _import_load_dataset() -> Any:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError(
                "HendrycksHandler requires the `datasets` package to load "
                "`hendrycks/ethics`."
            ) from exc
        return load_dataset

    def _load_ethics_csv_dataset(
        self,
        config: EthicsConfig,
        *,
        split: EthicsSplit | None,
        **kwargs: Any,
    ) -> Any:
        load_dataset = self._import_load_dataset()
        data_files = {
            split_name: f"{_URL_BASE}/{config}/{filename}"
            for split_name, filename in _ETHICS_FILES.items()
        }
        load_kwargs: dict[str, Any] = dict(kwargs)
        if self.cache_dir is not None:
            load_kwargs.setdefault("cache_dir", self.cache_dir)
        return load_dataset("csv", data_files=data_files, split=split, **load_kwargs)

    def _load_all_splits(self, config: EthicsConfig, **kwargs: Any) -> Any:
        try:
            from datasets import concatenate_datasets
        except ImportError as exc:
            raise ImportError(
                "HendrycksHandler requires the `datasets` package to concatenate "
                "ETHICS splits."
            ) from exc

        datasets = [
            self._load_ethics_csv_dataset(config, split=split_name, **kwargs)
            for split_name in ("train", "validation", "test")
        ]
        return concatenate_datasets(datasets)
