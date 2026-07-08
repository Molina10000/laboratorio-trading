from __future__ import annotations

from typing import Any

import pandas as pd


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_direction(raw_direction: str | None) -> str:
    if raw_direction == "alcista":
        return "ALCISTA"
    if raw_direction == "bajista":
        return "BAJISTA"
    return "MIXTO"


def _ema_bias(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "MIXTO"

    latest = frame.iloc[-1]
    close_price = _safe_float(latest.get("close"))
    ema_50 = _safe_float(latest.get("ema_50"), default=close_price)
    ema_200 = _safe_float(latest.get("ema_200"), default=close_price)

    if close_price > ema_50 > ema_200:
        return "ALCISTA"
    if close_price < ema_50 < ema_200:
        return "BAJISTA"
    return "MIXTO"


def _rsi_bias(frame: pd.DataFrame) -> tuple[str, float]:
    if frame.empty:
        return "MIXTO", 50.0

    latest = frame.iloc[-1]
    rsi_value = _safe_float(latest.get("rsi_14"), default=50.0)
    if rsi_value >= 55:
        return "ALCISTA", rsi_value
    if rsi_value <= 45:
        return "BAJISTA", rsi_value
    return "MIXTO", rsi_value


def _pattern_bias(signals: list[dict[str, Any]]) -> tuple[str, list[str]]:
    if not signals:
        return "MIXTO", []

    directions = {_normalize_direction(signal.get("direccion")) for signal in signals}
    patterns = [str(signal.get("patron", "")) for signal in signals if signal.get("patron")]
    if directions == {"ALCISTA"}:
        return "ALCISTA", patterns
    if directions == {"BAJISTA"}:
        return "BAJISTA", patterns
    return "MIXTO", patterns


def _context_from_components(
    trend: str,
    ema_bias: str,
    rsi_bias: str,
    pattern_bias: str,
) -> str:
    bullish_score = 0
    bearish_score = 0

    if trend == "alcista":
        bullish_score += 2
    elif trend == "bajista":
        bearish_score += 2

    if ema_bias == "ALCISTA":
        bullish_score += 2
    elif ema_bias == "BAJISTA":
        bearish_score += 2

    if rsi_bias == "ALCISTA":
        bullish_score += 1
    elif rsi_bias == "BAJISTA":
        bearish_score += 1

    if pattern_bias == "ALCISTA":
        bullish_score += 1
    elif pattern_bias == "BAJISTA":
        bearish_score += 1

    if bullish_score >= bearish_score + 2:
        return "ALCISTA"
    if bearish_score >= bullish_score + 2:
        return "BAJISTA"
    return "MIXTO"


def _summarize_timeframe(analysis: dict[str, Any], label: str) -> dict[str, Any]:
    frame = analysis.get("data", pd.DataFrame())
    trend = str(analysis.get("general_trend", "sin datos"))
    signals = analysis.get("signals", []) or []

    ema_bias = _ema_bias(frame)
    rsi_bias, rsi_value = _rsi_bias(frame)
    pattern_bias, pattern_names = _pattern_bias(signals)
    context = _context_from_components(trend, ema_bias, rsi_bias, pattern_bias)

    return {
        "label": label,
        "trend": trend.upper() if trend in {"alcista", "bajista", "lateral"} else "MIXTO",
        "ema_bias": ema_bias,
        "rsi_bias": rsi_bias,
        "rsi_value": round(rsi_value, 2),
        "pattern_bias": pattern_bias,
        "pattern_names": pattern_names,
        "context": context,
        "has_data": not frame.empty,
    }


def compare_multi_timeframe(
    analyses: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    analysis_4h = _summarize_timeframe(analyses.get("4h", {}), "4H")
    analysis_1h = _summarize_timeframe(analyses.get("1h", {}), "1H")

    if not analysis_4h["has_data"] or not analysis_1h["has_data"]:
        return {
            "alineacion": False,
            "contexto": "MIXTO",
            "sesgo_permitido": "ESPERAR",
            "explicacion": [
                "No hay datos suficientes en 4H o 1H para confirmar la lectura multi-temporal."
            ],
            "detalle": {"4H": analysis_4h, "1H": analysis_1h},
        }

    context_4h = analysis_4h["context"]
    context_1h = analysis_1h["context"]

    alignment = context_4h in {"ALCISTA", "BAJISTA"} and context_4h == context_1h

    if alignment:
        contexto = context_4h
        sesgo = "COMPRA" if contexto == "ALCISTA" else "VENTA"
    elif context_4h in {"ALCISTA", "BAJISTA"} and context_1h == "MIXTO":
        contexto = context_4h
        sesgo = "ESPERAR"
    else:
        contexto = "MIXTO"
        sesgo = "ESPERAR"

    explanation: list[str] = [
        f"4H marca un contexto principal {context_4h.lower()} con tendencia {analysis_4h['trend'].lower()} y EMA {analysis_4h['ema_bias'].lower()}.",
        f"1H muestra una lectura {context_1h.lower()} con tendencia {analysis_1h['trend'].lower()} y EMA {analysis_1h['ema_bias'].lower()}.",
    ]

    if alignment:
        explanation.append(
            f"Alineacion confirmada entre 4H y 1H. El sesgo permitido es {sesgo.lower()}."
        )
    elif context_4h in {"ALCISTA", "BAJISTA"} and context_1h in {"ALCISTA", "BAJISTA"} and context_4h != context_1h:
        explanation.append(
            f"4H y 1H muestran direcciones opuestas. El sesgo permitido permanece en {sesgo.lower()}."
        )
    else:
        explanation.append(
            f"No hay alineacion completa entre 4H y 1H. El sesgo permitido permanece en {sesgo.lower()}."
        )

    if analysis_1h["pattern_names"]:
        explanation.append(
            "Patrones 1H detectados: " + ", ".join(analysis_1h["pattern_names"]) + "."
        )
    elif analysis_4h["pattern_names"]:
        explanation.append(
            "Patrones 4H detectados: " + ", ".join(analysis_4h["pattern_names"]) + "."
        )

    return {
        "alineacion": alignment,
        "contexto": contexto,
        "sesgo_permitido": sesgo,
        "explicacion": explanation,
        "detalle": {"4H": analysis_4h, "1H": analysis_1h},
    }
