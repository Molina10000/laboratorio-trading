from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from database import DB_PATH, load_journal_entries, save_journal_entry


@dataclass(slots=True)
class JournalEntry:
    date: str
    asset: str
    timeframe: str
    direction: str
    pattern: str
    entry_planned: float
    stop_loss: float
    take_profit: float
    result: str
    entry_reason: str
    exit_reason: str
    respected_plan: bool
    psychological_note: str


def record_trade(entry: JournalEntry, db_path: str | Path = DB_PATH) -> None:
    save_journal_entry(asdict(entry), db_path=db_path)


def get_journal(db_path: str | Path = DB_PATH, limit: int = 200) -> pd.DataFrame:
    return load_journal_entries(db_path=db_path, limit=limit)
