from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd
import yfinance as yf

from database import DB_PATH, init_database, load_market_data, upsert_market_data


ASSET_CATALOG = {
    "EUR/USD": {
        "ticker": "EURUSD=X",
        "category": "forex",
        "default_active": True,
        "cost_unit_label": "puntos",
        "cost_point_size": 0.00001,
    },
    "Oro": {
        "ticker": "GC=F",
        "category": "commodity",
        "default_active": True,
        "cost_unit_label": "puntos",
        "cost_point_size": 0.1,
    },
    "Petróleo WTI": {
        "ticker": "CL=F",
        "category": "commodity",
        "default_active": True,
        "cost_unit_label": "puntos",
        "cost_point_size": 0.01,
    },
    "Apple": {
        "ticker": "AAPL",
        "category": "equity",
        "default_active": False,
        "cost_unit_label": "centavos",
        "cost_point_size": 0.01,
    },
    "S&P 500 ETF": {
        "ticker": "SPY",
        "category": "etf",
        "default_active": False,
        "cost_unit_label": "centavos",
        "cost_point_size": 0.01,
    },
    "Nasdaq 100 ETF": {
        "ticker": "QQQ",
        "category": "etf",
        "default_active": False,
        "cost_unit_label": "centavos",
        "cost_point_size": 0.01,
    },
}

ASSETS = {
    asset_name: str(metadata["ticker"])
    for asset_name, metadata in ASSET_CATALOG.items()
}
DEFAULT_ACTIVE_ASSETS = [
    asset_name
    for asset_name, metadata in ASSET_CATALOG.items()
    if bool(metadata.get("default_active"))
]
TIMEFRAMES = ("1h", "4h", "1d", "1wk")
VISIBLE_TIMEFRAMES = ("1h", "4h", "1d", "1wk")


class DataDownloadError(RuntimeError):
    """Raised when market data cannot be downloaded or normalized."""


@dataclass(slots=True)
class DownloadResult:
    asset: str
    ticker: str
    timeframe: str
    rows: int
    status: str
    message: str


def download_ohlc(
    ticker: str,
    period: str = "730d",
    interval: str = "60m",
) -> pd.DataFrame:
    try:
        frame = yf.download(
            tickers=ticker,
            period=period,
            interval=interval,
            auto_adjust=False,
            progress=False,
            threads=False,
        )
    except Exception as exc:
        raise DataDownloadError(f"yfinance devolvió un error para {ticker}: {exc}") from exc

    return _normalize_downloaded_frame(frame, ticker=ticker)


def download_ohlc_range(
    ticker: str,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    interval: str,
) -> pd.DataFrame:
    start_timestamp = pd.Timestamp(start)
    end_timestamp = pd.Timestamp(end)
    if end_timestamp <= start_timestamp:
        raise DataDownloadError("El rango solicitado para descargar datos es inválido.")

    try:
        frame = yf.download(
            tickers=ticker,
            start=start_timestamp.to_pydatetime(),
            end=end_timestamp.to_pydatetime(),
            interval=interval,
            auto_adjust=False,
            progress=False,
            threads=False,
        )
    except Exception as exc:
        raise DataDownloadError(f"yfinance devolvió un error para {ticker}: {exc}") from exc

    return _normalize_downloaded_frame(frame, ticker=ticker)


def _normalize_downloaded_frame(frame: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if frame.empty:
        raise DataDownloadError(f"No se recibieron datos para {ticker}.")

    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)

    required_columns = ["Open", "High", "Low", "Close", "Volume"]
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise DataDownloadError(
            f"Faltan columnas requeridas para {ticker}: {', '.join(missing)}."
        )

    frame = (
        frame[required_columns]
        .rename(columns=str.lower)
        .dropna(subset=["open", "high", "low", "close"])
        .sort_index()
    )

    if frame.empty:
        raise DataDownloadError(f"Los datos válidos para {ticker} quedaron vacíos.")

    return frame


def resample_to_4h(frame: pd.DataFrame) -> pd.DataFrame:
    resampled = frame.resample("4h").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )
    return resampled.dropna(subset=["open", "high", "low", "close"])


def resample_to_1w(frame: pd.DataFrame) -> pd.DataFrame:
    resampled = frame.resample("W-FRI").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )
    return resampled.dropna(subset=["open", "high", "low", "close"])


def refresh_market_data(
    db_path: str = str(DB_PATH),
    assets: dict[str, str] | None = None,
) -> list[DownloadResult]:
    init_database(db_path)
    selected_assets = assets or ASSETS
    results: list[DownloadResult] = []

    for asset, ticker in selected_assets.items():
        try:
            frame_1h = download_ohlc(ticker=ticker, period="730d", interval="60m")
            rows_1h = upsert_market_data(asset, "1h", frame_1h, db_path=db_path)
            results.append(
                DownloadResult(
                    asset=asset,
                    ticker=ticker,
                    timeframe="1h",
                    rows=rows_1h,
                    status="ok",
                    message="Datos 1h actualizados.",
                )
            )

            frame_4h = resample_to_4h(frame_1h)
            rows_4h = upsert_market_data(asset, "4h", frame_4h, db_path=db_path)
            results.append(
                DownloadResult(
                    asset=asset,
                    ticker=ticker,
                    timeframe="4h",
                    rows=rows_4h,
                    status="ok",
                    message="Datos 4h re-muestreados y actualizados.",
                )
            )
        except DataDownloadError as exc:
            for timeframe in ("1h", "4h"):
                results.append(
                    DownloadResult(
                        asset=asset,
                        ticker=ticker,
                        timeframe=timeframe,
                        rows=0,
                        status="error",
                        message=str(exc),
                    )
                )

        try:
            frame_1d = download_ohlc(ticker=ticker, period="10y", interval="1d")
            rows_1d = upsert_market_data(asset, "1d", frame_1d, db_path=db_path)
            results.append(
                DownloadResult(
                    asset=asset,
                    ticker=ticker,
                    timeframe="1d",
                    rows=rows_1d,
                    status="ok",
                    message="Datos 1d actualizados.",
                )
            )

            frame_1w = resample_to_1w(frame_1d)
            rows_1w = upsert_market_data(asset, "1wk", frame_1w, db_path=db_path)
            results.append(
                DownloadResult(
                    asset=asset,
                    ticker=ticker,
                    timeframe="1wk",
                    rows=rows_1w,
                    status="ok",
                    message="Datos 1wk re-muestreados y actualizados.",
                )
            )
        except DataDownloadError as exc:
            for timeframe in ("1d", "1wk"):
                results.append(
                    DownloadResult(
                        asset=asset,
                        ticker=ticker,
                        timeframe=timeframe,
                        rows=0,
                        status="error",
                        message=str(exc),
                    )
                )

    return results


def load_asset_data(
    asset: str,
    timeframe: str,
    db_path: str = str(DB_PATH),
    limit: int | None = 1500,
) -> pd.DataFrame:
    return load_market_data(asset=asset, timeframe=timeframe, db_path=db_path, limit=limit)


def summarize_download_results(results: Iterable[DownloadResult]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "activo": result.asset,
                "ticker": result.ticker,
                "temporalidad": result.timeframe,
                "filas": result.rows,
                "estado": result.status,
                "mensaje": result.message,
            }
            for result in results
        ]
    )


def get_asset_metadata(asset: str) -> dict[str, object]:
    return dict(ASSET_CATALOG.get(asset, {}))


def get_cost_point_size(asset: str) -> float:
    metadata = get_asset_metadata(asset)
    try:
        point_size = float(metadata.get("cost_point_size", 1.0))
    except (TypeError, ValueError):
        point_size = 1.0
    return point_size if point_size > 0 else 1.0


def get_cost_unit_label(asset: str) -> str:
    metadata = get_asset_metadata(asset)
    return str(metadata.get("cost_unit_label", "puntos"))
