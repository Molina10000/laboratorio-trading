from __future__ import annotations

from typing import Any

import pandas as pd

from data_loader import get_asset_metadata
from decision_engine import POSIBLE_OPERACION, VIGILAR


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _resolved_direction(analysis: dict[str, Any]) -> str | None:
    general_trend = str(analysis.get("general_trend", ""))
    structure_trend = str((analysis.get("market_structure", {}) or {}).get("trend", ""))

    directional = {"alcista", "bajista"}
    if general_trend in directional and structure_trend in {"", "lateral", "sin datos", "sin datos suficientes"}:
        return general_trend
    if structure_trend in directional and general_trend in {"", "lateral", "sin datos", "sin datos suficientes"}:
        return structure_trend
    if general_trend == structure_trend and general_trend in directional:
        return general_trend
    return None


def _latest_atr_ratio(analysis: dict[str, Any]) -> float:
    frame = analysis.get("data", pd.DataFrame())
    if frame.empty:
        return 0.0
    latest = frame.iloc[-1]
    close_price = max(_safe_float(latest.get("close")), 1e-9)
    atr_value = _safe_float(latest.get("atr_14"))
    return atr_value / close_price


def _ema_alignment(analysis: dict[str, Any]) -> bool:
    frame = analysis.get("data", pd.DataFrame())
    if frame.empty:
        return False
    latest = frame.iloc[-1]
    close_price = _safe_float(latest.get("close"))
    ema_50 = _safe_float(latest.get("ema_50"), default=close_price)
    ema_200 = _safe_float(latest.get("ema_200"), default=close_price)
    return (close_price >= ema_50 >= ema_200) or (close_price <= ema_50 <= ema_200)


def _decision_state(analysis: dict[str, Any]) -> str:
    return str((analysis.get("decision", {}) or {}).get("estado", "NO OPERAR"))


def _signal_count(analysis: dict[str, Any]) -> int:
    return len(analysis.get("signals", []) or [])


def _short_term_candidate(asset: str, analyses: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    analysis_4h = analyses.get("4h", {})
    analysis_1h = analyses.get("1h", {})
    trend_4h = _resolved_direction(analysis_4h)
    trend_1h = _resolved_direction(analysis_1h)

    if not trend_4h or trend_4h != trend_1h:
        return None

    metadata = get_asset_metadata(asset)
    reasons = [
        "4H y 1H están alineadas dentro de la misma estructura técnica.",
    ]
    warnings = [
        "Requiere stop loss, take profit y validación manual antes de cualquier simulación.",
    ]
    score = 55

    signals_1h = _signal_count(analysis_1h)
    if signals_1h:
        score += 15
        reasons.append("Hay patrón técnico reciente en 1H para afinar la entrada.")
    else:
        warnings.append("No hay patrón confirmado todavía.")

    atr_ratio = max(_latest_atr_ratio(analysis_1h), _latest_atr_ratio(analysis_4h))
    if atr_ratio >= 0.004:
        score += 10
        reasons.append("La volatilidad actual es suficiente según ATR.")
    else:
        warnings.append("La volatilidad actual es moderada y exige paciencia.")

    if _decision_state(analysis_1h) in {VIGILAR, POSIBLE_OPERACION}:
        score += 10
        reasons.append("La lectura operativa en 1H sigue activa para vigilancia educativa.")

    if metadata.get("category") in {"forex", "commodity"}:
        score += 10
        reasons.append("El activo suele adaptarse mejor a recorridos de horas a pocos días.")

    return {
        "score": score,
        "horizonte": "CORTO PLAZO",
        "duracion_estimada": "horas a pocos días",
        "razones": reasons,
        "advertencias": warnings,
        "temporalidades_usadas": ["4H", "1H"],
    }


def _medium_term_candidate(asset: str, analyses: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    analysis_1d = analyses.get("1d", {})
    analysis_4h = analyses.get("4h", {})
    trend_1d = _resolved_direction(analysis_1d)
    trend_4h = _resolved_direction(analysis_4h)

    if not trend_1d or trend_1d != trend_4h:
        return None

    metadata = get_asset_metadata(asset)
    reasons = [
        "1D marca una tendencia clara y 4H acompaña el contexto de entrada.",
    ]
    warnings: list[str] = []
    score = 60

    if _signal_count(analysis_4h):
        score += 10
        reasons.append("4H ofrece estructura técnica útil para un retroceso o continuación.")
    else:
        warnings.append("Todavía falta un patrón confirmado en 4H.")

    if _decision_state(analysis_4h) in {VIGILAR, POSIBLE_OPERACION}:
        score += 10
        reasons.append("El contexto de 4H permite vigilar una posible continuación educativa.")

    if _latest_atr_ratio(analysis_1d) >= 0.012:
        score += 5
        reasons.append("El desplazamiento diario sugiere recorrido para varias sesiones.")

    if metadata.get("category") in {"equity", "etf"}:
        score += 15
        reasons.append("El activo encaja mejor en ventanas de varios días a semanas.")
    elif metadata.get("category") == "commodity":
        score += 5

    warnings.append("Validar manualmente que exista retroceso o continuación antes de operar.")

    return {
        "score": score,
        "horizonte": "MEDIO PLAZO",
        "duracion_estimada": "1 a 3 semanas",
        "razones": reasons,
        "advertencias": warnings,
        "temporalidades_usadas": ["1D", "4H"],
    }


def _long_term_candidate(asset: str, analyses: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    analysis_1wk = analyses.get("1wk", {})
    analysis_1d = analyses.get("1d", {})
    trend_1wk = _resolved_direction(analysis_1wk)
    trend_1d = _resolved_direction(analysis_1d)

    if not trend_1wk or trend_1wk != trend_1d:
        return None

    metadata = get_asset_metadata(asset)
    reasons = [
        "1W y 1D sostienen una misma tendencia dominante.",
    ]
    warnings = [
        "Solo análisis técnico preliminar.",
        "El largo plazo requiere análisis fundamental adicional.",
    ]
    score = 65

    if _ema_alignment(analysis_1wk) and _ema_alignment(analysis_1d):
        score += 10
        reasons.append("Precio y medias móviles mantienen una estructura consistente de largo plazo.")
    else:
        warnings.append("La estructura con medias móviles todavía no es completamente limpia.")

    if metadata.get("category") in {"equity", "etf"}:
        score += 15
        reasons.append("El activo suele responder mejor a marcos de inversión más amplios.")
    else:
        warnings.append("Este activo no es el candidato más natural para una lectura de largo plazo.")

    if _latest_atr_ratio(analysis_1wk) >= 0.02:
        score += 5
        reasons.append("La volatilidad semanal sugiere que el movimiento sigue activo.")

    return {
        "score": score,
        "horizonte": "LARGO PLAZO",
        "duracion_estimada": "meses a años",
        "razones": reasons,
        "advertencias": warnings,
        "temporalidades_usadas": ["1W", "1D"],
    }


def classify_time_horizon(asset: str, analyses: dict[str, dict[str, Any]]) -> dict[str, Any]:
    candidates = [
        candidate
        for candidate in [
            _short_term_candidate(asset, analyses),
            _medium_term_candidate(asset, analyses),
            _long_term_candidate(asset, analyses),
        ]
        if candidate is not None
    ]

    if not candidates:
        return {
            "horizonte": "SIN CLASIFICAR",
            "duracion_estimada": "pendiente de confirmación",
            "razones": [
                "Falta alineación clara entre 4H/1H, 1D/4H o 1W/1D.",
            ],
            "advertencias": [
                "No hay suficiente estructura para sugerir un horizonte operativo consistente.",
                "Validar manualmente antes de considerar cualquier operación simulada.",
            ],
            "temporalidades_usadas": [],
        }

    best_candidate = max(candidates, key=lambda candidate: int(candidate["score"]))
    return {
        "horizonte": best_candidate["horizonte"],
        "duracion_estimada": best_candidate["duracion_estimada"],
        "razones": best_candidate["razones"],
        "advertencias": best_candidate["advertencias"],
        "temporalidades_usadas": best_candidate["temporalidades_usadas"],
    }
