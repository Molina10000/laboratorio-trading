from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import pandas as pd


DB_PATH = Path(__file__).with_name("analysis_lab.db")


@contextmanager
def get_connection(db_path: str | Path = DB_PATH) -> Iterator[sqlite3.Connection]:
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    try:
        yield connection
    finally:
        connection.close()


def init_database(db_path: str | Path = DB_PATH) -> None:
    with get_connection(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS market_data (
                asset TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume REAL,
                PRIMARY KEY (asset, timeframe, timestamp)
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS journal_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                asset TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                pattern TEXT NOT NULL,
                entry_planned REAL NOT NULL,
                stop_loss REAL NOT NULL,
                take_profit REAL NOT NULL,
                result TEXT,
                entry_reason TEXT,
                exit_reason TEXT,
                respected_plan INTEGER NOT NULL,
                psychological_note TEXT
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS app_settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS backtest_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                configuration_name TEXT NOT NULL,
                executed_at TEXT NOT NULL,
                asset TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                mode TEXT NOT NULL,
                data_segment TEXT NOT NULL,
                start_date TEXT,
                end_date TEXT,
                parameter_hash TEXT NOT NULL,
                parameters_json TEXT NOT NULL,
                statistics_json TEXT NOT NULL,
                data_quality_json TEXT,
                trade_count INTEGER NOT NULL DEFAULT 0,
                blocked_signals INTEGER NOT NULL DEFAULT 0,
                expectancy_gross REAL NOT NULL DEFAULT 0,
                expectancy_net REAL NOT NULL DEFAULT 0,
                max_drawdown_r REAL NOT NULL DEFAULT 0,
                profit_factor REAL NOT NULL DEFAULT 0,
                win_rate REAL NOT NULL DEFAULT 0
            )
            """
        )
        _ensure_column(connection, "journal_entries", "direction", "TEXT")
        connection.commit()


def _table_columns(connection: sqlite3.Connection, table_name: str) -> set[str]:
    rows = connection.execute(f"PRAGMA table_info({table_name})").fetchall()
    return {str(row["name"]) for row in rows}


def _ensure_column(
    connection: sqlite3.Connection,
    table_name: str,
    column_name: str,
    column_definition: str,
) -> None:
    if column_name in _table_columns(connection, table_name):
        return
    connection.execute(
        f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_definition}"
    )


def _normalize_timestamp(value: pd.Timestamp) -> str:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert("UTC").tz_localize(None)
    return timestamp.isoformat()


def upsert_market_data(
    asset: str,
    timeframe: str,
    frame: pd.DataFrame,
    db_path: str | Path = DB_PATH,
) -> int:
    if frame.empty:
        return 0

    rows = []
    for timestamp, row in frame.iterrows():
        rows.append(
            (
                asset,
                timeframe,
                _normalize_timestamp(pd.Timestamp(timestamp)),
                float(row["open"]),
                float(row["high"]),
                float(row["low"]),
                float(row["close"]),
                None if pd.isna(row.get("volume")) else float(row.get("volume")),
            )
        )

    with get_connection(db_path) as connection:
        connection.executemany(
            """
            INSERT INTO market_data (
                asset, timeframe, timestamp, open, high, low, close, volume
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(asset, timeframe, timestamp) DO UPDATE SET
                open = excluded.open,
                high = excluded.high,
                low = excluded.low,
                close = excluded.close,
                volume = excluded.volume
            """,
            rows,
        )
        connection.commit()
    return len(rows)


def load_market_data(
    asset: str,
    timeframe: str,
    db_path: str | Path = DB_PATH,
    limit: int | None = None,
) -> pd.DataFrame:
    query = """
        SELECT timestamp, open, high, low, close, volume
        FROM market_data
        WHERE asset = ? AND timeframe = ?
        ORDER BY timestamp ASC
    """
    parameters: list[object] = [asset, timeframe]

    if limit is not None:
        query = """
            SELECT timestamp, open, high, low, close, volume
            FROM (
                SELECT timestamp, open, high, low, close, volume
                FROM market_data
                WHERE asset = ? AND timeframe = ?
                ORDER BY timestamp DESC
                LIMIT ?
            )
            ORDER BY timestamp ASC
        """
        parameters.append(limit)

    with get_connection(db_path) as connection:
        frame = pd.read_sql_query(
            query,
            connection,
            params=parameters,
            parse_dates=["timestamp"],
        )

    if frame.empty:
        return frame

    frame = frame.set_index("timestamp")
    frame.index = pd.to_datetime(frame.index)
    return frame


def count_market_rows(db_path: str | Path = DB_PATH) -> int:
    with get_connection(db_path) as connection:
        row = connection.execute("SELECT COUNT(*) AS total FROM market_data").fetchone()
    return int(row["total"])


def save_journal_entry(entry: dict[str, object], db_path: str | Path = DB_PATH) -> None:
    with get_connection(db_path) as connection:
        connection.execute(
            """
            INSERT INTO journal_entries (
                date,
                asset,
                timeframe,
                direction,
                pattern,
                entry_planned,
                stop_loss,
                take_profit,
                result,
                entry_reason,
                exit_reason,
                respected_plan,
                psychological_note
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                entry["date"],
                entry["asset"],
                entry["timeframe"],
                entry.get("direction"),
                entry["pattern"],
                entry["entry_planned"],
                entry["stop_loss"],
                entry["take_profit"],
                entry.get("result"),
                entry.get("entry_reason"),
                entry.get("exit_reason"),
                1 if entry.get("respected_plan") else 0,
                entry.get("psychological_note"),
            ),
        )
        connection.commit()


def load_journal_entries(
    db_path: str | Path = DB_PATH,
    limit: int = 200,
) -> pd.DataFrame:
    with get_connection(db_path) as connection:
        frame = pd.read_sql_query(
            """
            SELECT
                id,
                date,
                asset,
                timeframe,
                direction,
                pattern,
                entry_planned,
                stop_loss,
                take_profit,
                result,
                entry_reason,
                exit_reason,
                respected_plan,
                psychological_note
            FROM journal_entries
            ORDER BY date DESC, id DESC
            LIMIT ?
            """,
            connection,
            params=[limit],
        )

    if frame.empty:
        return frame

    frame["respected_plan"] = frame["respected_plan"].astype(bool)
    return frame


def save_settings_store(
    settings: dict[str, object],
    db_path: str | Path = DB_PATH,
) -> None:
    rows = [
        (key, json.dumps(value, ensure_ascii=False))
        for key, value in settings.items()
    ]
    with get_connection(db_path) as connection:
        connection.executemany(
            """
            INSERT INTO app_settings (key, value)
            VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            rows,
        )
        connection.commit()


def load_settings_store(db_path: str | Path = DB_PATH) -> dict[str, object]:
    with get_connection(db_path) as connection:
        rows = connection.execute(
            "SELECT key, value FROM app_settings ORDER BY key ASC"
        ).fetchall()

    parsed: dict[str, object] = {}
    for row in rows:
        try:
            parsed[str(row["key"])] = json.loads(str(row["value"]))
        except json.JSONDecodeError:
            parsed[str(row["key"])] = row["value"]
    return parsed


def save_backtest_run(
    run_record: dict[str, object],
    db_path: str | Path = DB_PATH,
) -> int:
    execution = dict(run_record.get("execution", {}))
    statistics = dict(run_record.get("statistics", {}))
    data_quality_log = run_record.get("data_quality_log")
    if isinstance(data_quality_log, pd.DataFrame):
        quality_payload = data_quality_log.to_dict(orient="records")
    else:
        quality_payload = data_quality_log or []

    with get_connection(db_path) as connection:
        cursor = connection.execute(
            """
            INSERT INTO backtest_runs (
                configuration_name,
                executed_at,
                asset,
                timeframe,
                mode,
                data_segment,
                start_date,
                end_date,
                parameter_hash,
                parameters_json,
                statistics_json,
                data_quality_json,
                trade_count,
                blocked_signals,
                expectancy_gross,
                expectancy_net,
                max_drawdown_r,
                profit_factor,
                win_rate
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                execution.get("nombre_configuracion"),
                execution.get("fecha_ejecucion"),
                execution.get("asset"),
                execution.get("timeframe"),
                execution.get("parametros_usados", {}).get("mode"),
                execution.get("parametros_usados", {}).get("data_segment"),
                execution.get("parametros_usados", {}).get("start_date"),
                execution.get("parametros_usados", {}).get("end_date"),
                execution.get("parameter_hash"),
                json.dumps(execution.get("parametros_usados", {}), ensure_ascii=False),
                json.dumps(statistics, ensure_ascii=False),
                json.dumps(quality_payload, ensure_ascii=False),
                int(statistics.get("numero_senales", 0)),
                int(statistics.get("senales_bloqueadas", 0)),
                float(statistics.get("expectativa_bruta", 0.0)),
                float(statistics.get("expectativa_neta", 0.0)),
                float(statistics.get("max_drawdown_R", 0.0)),
                float(statistics.get("profit_factor", 0.0)),
                float(statistics.get("win_rate", 0.0)),
            ),
        )
        connection.commit()
        return int(cursor.lastrowid)


def load_backtest_runs(
    db_path: str | Path = DB_PATH,
    limit: int = 50,
) -> pd.DataFrame:
    with get_connection(db_path) as connection:
        frame = pd.read_sql_query(
            """
            SELECT
                id,
                configuration_name,
                executed_at,
                asset,
                timeframe,
                mode,
                data_segment,
                start_date,
                end_date,
                parameter_hash,
                trade_count,
                blocked_signals,
                expectancy_gross,
                expectancy_net,
                max_drawdown_r,
                profit_factor,
                win_rate
            FROM backtest_runs
            ORDER BY id DESC
            LIMIT ?
            """,
            connection,
            params=[limit],
        )
    return frame
