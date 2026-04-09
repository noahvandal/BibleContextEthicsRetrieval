from __future__ import annotations

from dataclasses import dataclass

from src.datasets.hendrycks_dataset.hendrycks_handler import HendrycksHandler
from src.inference.ethics_inference import EthicsInferenceConfig, EthicsInferenceRunner


class FakeGenerationClient:
    def __init__(self, responses: list[str]) -> None:
        self._responses = responses
        self.prompts: list[str] = []
        self.ensure_calls = 0

    def ensure_running(self, startup_timeout: float = 60.0) -> None:
        self.ensure_calls += 1

    def generate(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        self.prompts.append(prompt)
        return self._responses[len(self.prompts) - 1]


@dataclass(frozen=True)
class FakeRetrievalResult:
    reference: str
    text: str


class FakeRetriever:
    def __init__(self) -> None:
        self.queries: list[tuple[str, int]] = []

    def search(self, query: str, *, n: int = 5) -> list[FakeRetrievalResult]:
        self.queries.append((query, n))
        return [
            FakeRetrievalResult("John 3:16", "For God so loved the world"),
            FakeRetrievalResult("Micah 6:8", "Do justice, love mercy, walk humbly"),
        ]


@dataclass(frozen=True)
class FakeVerse:
    verse_id: int
    reference: str
    text: str
    chapter: int
    verse: int
    book: str


class FakeBibleHandler:
    def __init__(self) -> None:
        self._by_id = {
            100: FakeVerse(100, "John 3:15", "Verse 15", 3, 15, "JOHN"),
            101: FakeVerse(101, "John 3:16", "Verse 16", 3, 16, "JOHN"),
            102: FakeVerse(102, "John 3:17", "Verse 17", 3, 17, "JOHN"),
        }
        self._by_loc = {
            ("JOHN", 3, 15): self._by_id[100],
            ("JOHN", 3, 16): self._by_id[101],
            ("JOHN", 3, 17): self._by_id[102],
        }

    def get_verse(self, verse_id: int):
        return self._by_id[verse_id]

    def get_verse_count(self, book, chapter: int) -> int:
        return 17

    def get_verse_by_location(self, book, chapter: int, verse: int):
        book_name = book if isinstance(book, str) else book.name
        return self._by_loc[(book_name, chapter, verse)]


def test_run_dataset_plain_prompt_respects_limit_and_parses_binary_labels():
    dataset = [
        {"label": "1", "input": "Return a lost wallet."},
        {"label": "0", "input": "Steal cash from a friend."},
        {"label": "1", "input": "Help an injured stranger."},
    ]
    client = FakeGenerationClient(["1", "0"])
    runner = EthicsInferenceRunner(
        client,
        handler=HendrycksHandler(load_dataset_fn=lambda *_args, **_kwargs: None),
    )

    results = runner.run_dataset(
        "commonsense",
        split="validation",
        limit=2,
        dataset_split=dataset,
    )

    assert client.ensure_calls == 1
    assert len(results) == 2
    assert [r.parsed_prediction for r in results] == [1, 0]
    assert [r.correct for r in results] == [True, True]
    assert "Scenario: Return a lost wallet." in client.prompts[0]


def test_run_dataset_with_bible_context_prepends_top_verses():
    dataset = [{"label": "1", "scenario": "Show mercy to an enemy.", "excuse": "They apologized."}]
    client = FakeGenerationClient(["1"])
    retriever = FakeRetriever()
    runner = EthicsInferenceRunner(
        client,
        handler=HendrycksHandler(load_dataset_fn=lambda *_args, **_kwargs: None),
        retriever=retriever,
    )

    results = runner.run_dataset(
        "deontology",
        split="validation",
        limit=1,
        with_bible_context=True,
        dataset_split=dataset,
    )

    assert retriever.queries == [("Scenario: Show mercy to an enemy.\nExcuse: They apologized.", 5)] or retriever.queries == [("Scenario: Show mercy to an enemy.\nExcuse: They apologized.", 5)]
    assert len(results[0].retrieved_context) == 2
    assert "Bible context:" in results[0].prompt
    assert "John 3:16 For God so loved the world" in results[0].prompt


def test_utilitarianism_prediction_is_parsed_as_a_or_b():
    dataset = [{"baseline": "Read a good book.", "less_pleasant": "Read spam email."}]
    client = FakeGenerationClient(["A"])
    runner = EthicsInferenceRunner(
        client,
        handler=HendrycksHandler(load_dataset_fn=lambda *_args, **_kwargs: None),
    )

    results = runner.run_dataset(
        "utilitarianism",
        split="validation",
        limit=1,
        dataset_split=dataset,
    )

    assert results[0].parsed_prediction == "A"
    assert results[0].expected == "A"
    assert results[0].correct is True


def test_bible_context_can_expand_surrounding_verses():
    dataset = [{"label": "1", "input": "Show mercy."}]
    client = FakeGenerationClient(["1"])

    class CenterOnlyRetriever:
        def search(self, query: str, *, n: int = 5):
            return [
                type(
                    "R",
                    (),
                    {
                        "reference": "John 3:16",
                        "text": "Verse 16",
                        "verse_id": 101,
                        "chapter": 3,
                        "verse": 16,
                    },
                )()
            ]

    runner = EthicsInferenceRunner(
        client,
        handler=HendrycksHandler(load_dataset_fn=lambda *_args, **_kwargs: None),
        retriever=CenterOnlyRetriever(),
        bible_handler=FakeBibleHandler(),
        config=EthicsInferenceConfig(surrounding_verses=1),
    )

    results = runner.run_dataset(
        "commonsense",
        split="validation",
        limit=1,
        with_bible_context=True,
        dataset_split=dataset,
    )

    assert results[0].retrieved_context == (
        "John 3:15 Verse 15",
        "John 3:16 Verse 16",
        "John 3:17 Verse 17",
    )
