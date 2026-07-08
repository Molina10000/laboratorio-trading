from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema

from core.indicators import detect_general_trend


def _price_precision(value: float) -> int:
    return 5 if abs(value) < 10 else 2


def _round_price(value: float) -> float:
    return round(float(value), _price_precision(float(value)))


def _swing_order(frame: pd.DataFrame) -> int:
    return max(3, min(8, len(frame) // 25))


def find_relevant_swings(frame: pd.DataFrame, order: int | None = None) -> dict[str, pd.DataFrame]:
    if len(frame) < 15:
        empty = pd.DataFrame(columns=["bar", "timestamp", "price"])
        return {"highs": empty, "lows": empty}

    swing_order = order or _swing_order(frame)
    highs_idx = argrelextrema(frame["high"].to_numpy(), np.greater_equal, order=swing_order)[0]
    lows_idx = argrelextrema(frame["low"].to_numpy(), np.less_equal, order=swing_order)[0]

    highs = pd.DataFrame(
        {
            "bar": highs_idx,
            "timestamp": frame.index.take(highs_idx),
            "price": frame["high"].iloc[highs_idx].to_numpy(),
        }
    )
    lows = pd.DataFrame(
        {
            "bar": lows_idx,
            "timestamp": frame.index.take(lows_idx),
            "price": frame["low"].iloc[lows_idx].to_numpy(),
        }
    )
    return {"highs": highs, "lows": lows}


def _cluster_levels(levels: pd.Series, tolerance: float) -> list[float]:
    cleaned = sorted(float(level) for level in levels.dropna().tolist())
    if not cleaned:
        return []

    clusters: list[list[float]] = [[cleaned[0]]]
    for level in cleaned[1:]:
        if abs(level - clusters[-1][-1]) <= tolerance:
            clusters[-1].append(level)
        else:
            clusters.append([level])

    return [float(np.mean(cluster)) for cluster in clusters]


def _market_structure_trend(swings: dict[str, pd.DataFrame], fallback: str) -> str:
    highs = swings["highs"]
    lows = swings["lows"]

    if len(highs) < 2 or len(lows) < 2:
        return fallback

    last_highs = highs["price"].tail(2).tolist()
    last_lows = lows["price"].tail(2).tolist()

    if last_highs[-1] > last_highs[-2] and last_lows[-1] > last_lows[-2]:
        return "alcista"
    if last_highs[-1] < last_highs[-2] and last_lows[-1] < last_lows[-2]:
        return "bajista"
    return "lateral"


def summarize_market_structure(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {
            "trend": "sin datos",
            "relevant_highs": [],
            "relevant_lows": [],
            "supports": [],
            "resistances": [],
        }

    window = frame.tail(220)
    swings = find_relevant_swings(window)
    atr_value = float(window["atr_14"].dropna().iloc[-1]) if "atr_14" in window and not window["atr_14"].dropna().empty else float(window["close"].iloc[-1] * 0.002)
    fallback_trend = detect_general_trend(window)
    structure_trend = _market_structure_trend(swings, fallback_trend)

    support_levels = _cluster_levels(swings["lows"]["price"].tail(10), max(atr_value * 0.75, 1e-6))
    resistance_levels = _cluster_levels(swings["highs"]["price"].tail(10), max(atr_value * 0.75, 1e-6))

    relevant_highs = [
        {"fecha": pd.Timestamp(row["timestamp"]).strftime("%Y-%m-%d %H:%M"), "precio": _round_price(row["price"])}
        for _, row in swings["highs"].tail(5).iterrows()
    ]
    relevant_lows = [
        {"fecha": pd.Timestamp(row["timestamp"]).strftime("%Y-%m-%d %H:%M"), "precio": _round_price(row["price"])}
        for _, row in swings["lows"].tail(5).iterrows()
    ]

    return {
        "trend": structure_trend,
        "relevant_highs": relevant_highs,
        "relevant_lows": relevant_lows,
        "supports": [_round_price(level) for level in support_levels[-3:]],
        "resistances": [_round_price(level) for level in resistance_levels[-3:]],
    }


def _build_pattern_signal(
    asset: str,
    timeframe: str,
    pattern_name: str,
    direction: str,
    entry: float,
    stop: float,
    target: float,
    explanation: str,
    signal_index: pd.Timestamp,
) -> dict[str, Any] | None:
    risk = abs(entry - stop)
    reward = abs(target - entry)
    if risk <= 0 or reward <= 0:
        return None

    return {
        "activo": asset,
        "temporalidad": timeframe,
        "patron": pattern_name,
        "direccion": direction,
        "precio_entrada": _round_price(entry),
        "stop_loss": _round_price(stop),
        "take_profit": _round_price(target),
        "relacion_riesgo_beneficio": round(reward / risk, 2),
        "explicacion": explanation,
        "signal_index": pd.Timestamp(signal_index),
    }


def _build_double_pattern_context(frame: pd.DataFrame) -> dict[str, Any] | None:
    if len(frame) < 60 or "atr_14" not in frame:
        return None

    window = frame.tail(220)
    swings = find_relevant_swings(window)
    latest = window.iloc[-1]
    previous = window.iloc[-2]
    atr_value = float(latest["atr_14"]) if not pd.isna(latest["atr_14"]) else float(window["close"].iloc[-1] * 0.002)

    return {
        "window": window,
        "swings": swings,
        "latest": latest,
        "previous": previous,
        "atr_value": atr_value,
    }


def _detect_double_bottom_from_context(
    *,
    context: dict[str, Any],
    asset: str,
    timeframe: str,
) -> dict[str, Any] | None:
    window = context["window"]
    swings = context["swings"]
    lows = swings["lows"]
    if len(lows) < 2:
        return None

    latest = context["latest"]
    previous = context["previous"]
    atr_value = float(context["atr_value"])

    second_low = lows.iloc[-1]
    first_low = lows.iloc[-2]
    if second_low["bar"] - first_low["bar"] < 5:
        return None

    lows_distance = abs(float(second_low["price"]) - float(first_low["price"]))
    if lows_distance > max(atr_value * 0.6, float(window["close"].iloc[-1]) * 0.002):
        return None

    neckline = float(window.iloc[int(first_low["bar"]) : int(second_low["bar"]) + 1]["high"].max())
    if not (float(previous["close"]) <= neckline < float(latest["close"])):
        return None

    lowest_point = min(float(first_low["price"]), float(second_low["price"]))
    entry = float(latest["close"])
    stop = lowest_point - atr_value * 0.5
    pattern_height = neckline - lowest_point
    target = max(entry + pattern_height, entry + 2 * abs(entry - stop))
    explanation = (
        "Dos mínimos similares con confirmación por ruptura de neckline en la última vela."
    )

    return _build_pattern_signal(
        asset=asset,
        timeframe=timeframe,
        pattern_name="doble suelo",
        direction="alcista",
        entry=entry,
        stop=stop,
        target=target,
        explanation=explanation,
        signal_index=window.index[-1],
    )


def detect_double_bottom(frame: pd.DataFrame, asset: str, timeframe: str) -> dict[str, Any] | None:
    context = _build_double_pattern_context(frame)
    if context is None:
        return None
    return _detect_double_bottom_from_context(context=context, asset=asset, timeframe=timeframe)


def _detect_double_top_from_context(
    *,
    context: dict[str, Any],
    asset: str,
    timeframe: str,
) -> dict[str, Any] | None:
    window = context["window"]
    swings = context["swings"]
    highs = swings["highs"]
    if len(highs) < 2:
        return None

    latest = context["latest"]
    previous = context["previous"]
    atr_value = float(context["atr_value"])

    second_high = highs.iloc[-1]
    first_high = highs.iloc[-2]
    if second_high["bar"] - first_high["bar"] < 5:
        return None

    highs_distance = abs(float(second_high["price"]) - float(first_high["price"]))
    if highs_distance > max(atr_value * 0.6, float(window["close"].iloc[-1]) * 0.002):
        return None

    neckline = float(window.iloc[int(first_high["bar"]) : int(second_high["bar"]) + 1]["low"].min())
    if not (float(previous["close"]) >= neckline > float(latest["close"])):
        return None

    highest_point = max(float(first_high["price"]), float(second_high["price"]))
    entry = float(latest["close"])
    stop = highest_point + atr_value * 0.5
    pattern_height = highest_point - neckline
    target = min(entry - pattern_height, entry - 2 * abs(entry - stop))
    explanation = (
        "Dos máximos similares con confirmación por ruptura bajista del neckline en la última vela."
    )

    return _build_pattern_signal(
        asset=asset,
        timeframe=timeframe,
        pattern_name="doble techo",
        direction="bajista",
        entry=entry,
        stop=stop,
        target=target,
        explanation=explanation,
        signal_index=window.index[-1],
    )


def detect_double_top(frame: pd.DataFrame, asset: str, timeframe: str) -> dict[str, Any] | None:
    context = _build_double_pattern_context(frame)
    if context is None:
        return None
    return _detect_double_top_from_context(context=context, asset=asset, timeframe=timeframe)


def detect_range_breakout(frame: pd.DataFrame, asset: str, timeframe: str) -> dict[str, Any] | None:
    if len(frame) < 40 or "atr_14" not in frame:
        return None

    latest = frame.iloc[-1]
    previous = frame.iloc[-2]
    atr_value = float(latest["atr_14"]) if not pd.isna(latest["atr_14"]) else float(frame["close"].iloc[-1] * 0.002)

    recent_high = float(frame["high"].shift(1).rolling(20).max().iloc[-1])
    recent_low = float(frame["low"].shift(1).rolling(20).min().iloc[-1])
    if np.isnan(recent_high) or np.isnan(recent_low):
        return None

    range_width = recent_high - recent_low
    if range_width > atr_value * 4:
        return None

    latest_close = float(latest["close"])
    previous_close = float(previous["close"])

    if previous_close <= recent_high < latest_close:
        entry = latest_close
        stop = recent_low - atr_value * 0.5
        target = entry + 2 * abs(entry - stop)
        explanation = "El precio rompió por arriba un rango compacto de las últimas 20 velas."
        return _build_pattern_signal(
            asset=asset,
            timeframe=timeframe,
            pattern_name="ruptura de rango alcista",
            direction="alcista",
            entry=entry,
            stop=stop,
            target=target,
            explanation=explanation,
            signal_index=frame.index[-1],
        )

    if previous_close >= recent_low > latest_close:
        entry = latest_close
        stop = recent_high + atr_value * 0.5
        target = entry - 2 * abs(entry - stop)
        explanation = "El precio rompió por abajo un rango compacto de las últimas 20 velas."
        return _build_pattern_signal(
            asset=asset,
            timeframe=timeframe,
            pattern_name="ruptura de rango bajista",
            direction="bajista",
            entry=entry,
            stop=stop,
            target=target,
            explanation=explanation,
            signal_index=frame.index[-1],
        )

    return None


def detect_patterns(frame: pd.DataFrame, asset: str, timeframe: str) -> list[dict[str, Any]]:
    signals: list[dict[str, Any]] = []

    double_pattern_context = _build_double_pattern_context(frame)
    if double_pattern_context is not None:
        bottom_signal = _detect_double_bottom_from_context(
            context=double_pattern_context,
            asset=asset,
            timeframe=timeframe,
        )
        if bottom_signal is not None:
            signals.append(bottom_signal)

        top_signal = _detect_double_top_from_context(
            context=double_pattern_context,
            asset=asset,
            timeframe=timeframe,
        )
        if top_signal is not None:
            signals.append(top_signal)

    breakout_signal = detect_range_breakout(frame, asset, timeframe)
    if breakout_signal is not None:
        signals.append(breakout_signal)

    return signals
