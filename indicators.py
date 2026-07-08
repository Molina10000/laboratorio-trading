from __future__ import annotations

import numpy as np
import pandas as pd


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)

    average_gain = gains.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    average_loss = losses.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    relative_strength = average_gain / average_loss.replace(0, np.nan)

    oscillator = 100 - (100 / (1 + relative_strength))
    return oscillator.fillna(50.0)


def atr(frame: pd.DataFrame, period: int = 14) -> pd.Series:
    previous_close = frame["close"].shift(1)
    true_range = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - previous_close).abs(),
            (frame["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def add_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()
    enriched["ema_50"] = ema(enriched["close"], 50)
    enriched["ema_200"] = ema(enriched["close"], 200)
    enriched["rsi_14"] = rsi(enriched["close"], 14)
    enriched["atr_14"] = atr(enriched, 14)
    return enriched


def detect_general_trend(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "sin datos"

    latest = frame.iloc[-1]
    required_values = ["close", "ema_50", "ema_200"]
    if any(pd.isna(latest[value]) for value in required_values):
        return "sin datos suficientes"

    close_price = float(latest["close"])
    ema_50_value = float(latest["ema_50"])
    ema_200_value = float(latest["ema_200"])
    distance_ratio = abs(ema_50_value - ema_200_value) / max(close_price, 1e-9)

    if close_price > ema_50_value > ema_200_value and distance_ratio >= 0.002:
        return "alcista"
    if close_price < ema_50_value < ema_200_value and distance_ratio >= 0.002:
        return "bajista"
    return "lateral"
