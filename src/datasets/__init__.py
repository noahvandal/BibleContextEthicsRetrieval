from src.datasets.hendrycks_dataset.hendrycks_handler import (
    ETHICS_CONFIGS,
    HendrycksHandler,
    HendrycksRecord,
)

__all__ = ["ETHICS_CONFIGS", "HendrycksHandler", "HendrycksRecord"]

try:
    from src.datasets.bible_dataset.bible_handler import BibleHandler, VerseRecord
except ModuleNotFoundError:
    BibleHandler = None
    VerseRecord = None
else:
    __all__.extend(["BibleHandler", "VerseRecord"])
