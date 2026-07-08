from __future__ import annotations

from datetime import datetime
from typing import Any

from data_loader import ASSET_CATALOG, DEFAULT_ACTIVE_ASSETS, VISIBLE_TIMEFRAMES
from database import init_database, load_settings_store, save_settings_store


REFRESH_OPTIONS = {
    "manual": {"label": "Manual", "minutes": None},
    "15m": {"label": "Cada 15 minutos", "minutes": 15},
    "30m": {"label": "Cada 30 minutos", "minutes": 30},
    "60m": {"label": "Cada 60 minutos", "minutes": 60},
}

DEFAULT_SETTINGS = {
    "active_assets": DEFAULT_ACTIVE_ASSETS,
    "visible_timeframes": list(VISIBLE_TIMEFRAMES),
    "auto_refresh": "manual",
    "reference_capital": 10000.0,
    "risk_per_trade": 1.0,
    "last_refresh_at": None,
    "last_refresh_attempt_at": None,
    "backtest_frozen_signature": None,
    "backtest_frozen_at": None,
}


def _ordered_asset_names(asset_names: list[str]) -> list[str]:
    catalog_order = list(ASSET_CATALOG.keys())
    return [asset for asset in catalog_order if asset in asset_names]


def _ordered_timeframes(timeframes: list[str]) -> list[str]:
    return [timeframe for timeframe in VISIBLE_TIMEFRAMES if timeframe in timeframes]


def normalize_settings(raw_settings: dict[str, Any] | None) -> dict[str, Any]:
    raw_settings = raw_settings or {}

    active_assets = raw_settings.get("active_assets", DEFAULT_SETTINGS["active_assets"])
    if not isinstance(active_assets, list):
        active_assets = list(DEFAULT_SETTINGS["active_assets"])
    active_assets = _ordered_asset_names(
        [asset for asset in active_assets if asset in ASSET_CATALOG]
    )
    if not active_assets:
        active_assets = list(DEFAULT_SETTINGS["active_assets"])

    visible_timeframes = raw_settings.get(
        "visible_timeframes",
        DEFAULT_SETTINGS["visible_timeframes"],
    )
    if not isinstance(visible_timeframes, list):
        visible_timeframes = list(DEFAULT_SETTINGS["visible_timeframes"])
    visible_timeframes = _ordered_timeframes(
        [timeframe for timeframe in visible_timeframes if timeframe in VISIBLE_TIMEFRAMES]
    )
    if not visible_timeframes:
        visible_timeframes = list(DEFAULT_SETTINGS["visible_timeframes"])

    auto_refresh = str(raw_settings.get("auto_refresh", DEFAULT_SETTINGS["auto_refresh"]))
    if auto_refresh not in REFRESH_OPTIONS:
        auto_refresh = str(DEFAULT_SETTINGS["auto_refresh"])

    try:
        reference_capital = float(
            raw_settings.get("reference_capital", DEFAULT_SETTINGS["reference_capital"])
        )
    except (TypeError, ValueError):
        reference_capital = float(DEFAULT_SETTINGS["reference_capital"])

    try:
        risk_per_trade = float(
            raw_settings.get("risk_per_trade", DEFAULT_SETTINGS["risk_per_trade"])
        )
    except (TypeError, ValueError):
        risk_per_trade = float(DEFAULT_SETTINGS["risk_per_trade"])

    last_refresh_at = raw_settings.get("last_refresh_at")
    if last_refresh_at is not None:
        last_refresh_at = str(last_refresh_at)

    last_refresh_attempt_at = raw_settings.get("last_refresh_attempt_at")
    if last_refresh_attempt_at is not None:
        last_refresh_attempt_at = str(last_refresh_attempt_at)

    backtest_frozen_signature = raw_settings.get("backtest_frozen_signature")
    if backtest_frozen_signature is not None:
        backtest_frozen_signature = str(backtest_frozen_signature)

    backtest_frozen_at = raw_settings.get("backtest_frozen_at")
    if backtest_frozen_at is not None:
        backtest_frozen_at = str(backtest_frozen_at)

    return {
        "active_assets": active_assets,
        "visible_timeframes": visible_timeframes,
        "auto_refresh": auto_refresh,
        "reference_capital": max(reference_capital, 1.0),
        "risk_per_trade": min(max(risk_per_trade, 0.1), 100.0),
        "last_refresh_at": last_refresh_at,
        "last_refresh_attempt_at": last_refresh_attempt_at,
        "backtest_frozen_signature": backtest_frozen_signature,
        "backtest_frozen_at": backtest_frozen_at,
    }


def load_app_settings() -> dict[str, Any]:
    init_database()
    return normalize_settings(load_settings_store())


def save_app_settings(settings: dict[str, Any]) -> dict[str, Any]:
    init_database()
    normalized = normalize_settings(settings)
    save_settings_store(normalized)
    return normalized


def get_refresh_interval_minutes(settings: dict[str, Any]) -> int | None:
    option_key = str(settings.get("auto_refresh", "manual"))
    option = REFRESH_OPTIONS.get(option_key, REFRESH_OPTIONS["manual"])
    minutes = option.get("minutes")
    return int(minutes) if minutes is not None else None


def get_refresh_label(settings: dict[str, Any]) -> str:
    option_key = str(settings.get("auto_refresh", "manual"))
    option = REFRESH_OPTIONS.get(option_key, REFRESH_OPTIONS["manual"])
    return str(option["label"])


def parse_refresh_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def format_refresh_timestamp(value: str | None, fallback: str = "Sin actualizaciones") -> str:
    timestamp = parse_refresh_timestamp(value)
    if timestamp is None:
        return fallback
    return timestamp.strftime("%Y-%m-%d %H:%M")


def now_storage_timestamp() -> str:
    return datetime.now().replace(second=0, microsecond=0).isoformat()
