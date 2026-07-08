from __future__ import annotations

from typing import Any, Iterable

import pandas as pd

from core.data_loader import ASSETS, TIMEFRAMES, load_asset_data
from core.indicators import add_indicators, detect_general_trend
from core.patterns import detect_patterns, summarize_market_structure
from core.semaforo import (
    NO_OPERAR,
    POSIBLE_OPERACION,
    VIGILAR,
    evaluate_trade_decision,
    public_entry_status,
    public_semaforo,
)
from core.utils import dedupe_keep_order
from multi_timeframe import compare_multi_timeframe
from time_horizon import classify_time_horizon
from watchlist import build_watchlist


def _empty_analysis(asset: str, timeframe: str) -> dict[str, Any]:
    empty_decision = {
        "estado": "NO OPERAR",
        "direccion": "NINGUNA",
        "score": 0,
        "motivos": [f"No hay datos disponibles para {asset} en {timeframe}."],
        "advertencias": ["Actualizar datos antes de analizar esta temporalidad."],
    }
    return {
        "timeframe": timeframe,
        "data": pd.DataFrame(),
        "market_structure": {
            "trend": "sin datos",
            "relevant_highs": [],
            "relevant_lows": [],
            "supports": [],
            "resistances": [],
        },
        "general_trend": "sin datos",
        "signals": [],
        "decision": empty_decision,
    }


def _prepare_timeframe_analysis(
    asset: str,
    timeframe: str,
    *,
    limit: int | None,
) -> dict[str, Any]:
    data = load_asset_data(asset, timeframe, limit=limit)
    if data.empty:
        return _empty_analysis(asset, timeframe)

    enriched = add_indicators(data)
    market_structure = summarize_market_structure(enriched)
    general_trend = detect_general_trend(enriched)
    signals = detect_patterns(enriched, asset, timeframe)
    decision = evaluate_trade_decision(
        frame=enriched,
        asset=asset,
        timeframe=timeframe,
        market_structure=market_structure,
        signals=signals,
        general_trend=general_trend,
    )
    return {
        "timeframe": timeframe,
        "data": enriched,
        "market_structure": market_structure,
        "general_trend": general_trend,
        "signals": signals,
        "decision": decision,
    }


def prepare_timeframe_analysis(asset: str, timeframe: str) -> dict[str, Any]:
    return _prepare_timeframe_analysis(asset, timeframe, limit=1500)


def build_asset_analyses(
    asset: str,
    timeframes: Iterable[str] | None = None,
    *,
    limit: int | None = 1500,
) -> dict[str, dict[str, Any]]:
    selected_timeframes = tuple(timeframes or TIMEFRAMES)
    return {
        timeframe: _prepare_timeframe_analysis(asset, timeframe, limit=limit)
        for timeframe in selected_timeframes
    }


def _best_available_analysis(
    analyses: dict[str, dict[str, Any]],
    preferred_order: Iterable[str],
) -> dict[str, Any]:
    for timeframe in preferred_order:
        analysis = analyses.get(timeframe)
        if analysis and not analysis.get("data", pd.DataFrame()).empty:
            return analysis
    return next(iter(analyses.values()), _empty_analysis("", "1h"))


def get_latest_timestamp(analyses: dict[str, dict[str, Any]]) -> pd.Timestamp | None:
    timestamps: list[pd.Timestamp] = []
    for analysis in analyses.values():
        frame = analysis.get("data", pd.DataFrame())
        if frame.empty:
            continue
        timestamps.append(pd.Timestamp(frame.index[-1]))
    if not timestamps:
        return None
    return max(timestamps)


def format_price(value: float) -> str:
    return f"{value:.5f}" if abs(value) < 10 else f"{value:.2f}"


def _describe_timeframe_state(detail: dict[str, Any]) -> str:
    context = str(detail.get("context", "MIXTO")).upper()
    trend = str(detail.get("trend", "MIXTO")).upper()

    context_label = context.lower() if context in {"ALCISTA", "BAJISTA", "MIXTO"} else "sin datos"
    trend_label = trend.lower() if trend in {"ALCISTA", "BAJISTA", "LATERAL"} else "sin datos"

    if context == "MIXTO" and trend == "LATERAL":
        return "mixto/lateral"
    if context in {"ALCISTA", "BAJISTA"} and trend == "LATERAL":
        return f"{context_label} con tendencia lateral"
    if context == "MIXTO" and trend in {"ALCISTA", "BAJISTA"}:
        return f"mixto con tendencia {trend_label}"
    if context in {"ALCISTA", "BAJISTA"} and trend in {"ALCISTA", "BAJISTA"} and context != trend:
        return f"{context_label} con tendencia {trend_label}"
    if context in {"ALCISTA", "BAJISTA"}:
        return context_label
    if trend in {"ALCISTA", "BAJISTA", "LATERAL"}:
        return trend_label
    if context == "MIXTO":
        return context_label

    return "sin datos"


def _build_alignment_reason(multi_timeframe: dict[str, Any]) -> str:
    detail = multi_timeframe.get("detalle", {}) or {}
    state_4h = _describe_timeframe_state(detail.get("4H", {}))
    state_1h = _describe_timeframe_state(detail.get("1H", {}))

    if state_4h in {"alcista", "bajista"} and state_1h in {"alcista", "bajista"}:
        if state_4h == state_1h:
            return f"4H {state_4h} + 1H {state_1h} = alineacion confirmada."
        return f"4H {state_4h} + 1H {state_1h} = conflicto entre temporalidades; esperar confirmacion."

    return f"4H {state_4h} + 1H {state_1h} = no hay alineacion completa; esperar confirmacion."


def _build_operational_summary(
    decision: dict[str, Any],
    multi_timeframe: dict[str, Any],
) -> str:
    bias = str(multi_timeframe.get("sesgo_permitido", "ESPERAR"))
    direction = str(decision.get("direccion", "NINGUNA"))

    if bias not in {"COMPRA", "VENTA"}:
        return "No existe alineacion suficiente entre 4H y 1H; el sesgo permitido es esperar y la entrada permanece no habilitada."

    if direction not in {"COMPRA", "VENTA"}:
        return f"El sesgo permitido es {bias.lower()}, pero la señal actual de 1H no esta alineada; esperar confirmacion."

    if bool(decision.get("entrada_habilitada")):
        return f"La entrada de {direction.lower()} quedo habilitada porque 4H y 1H estan alineadas y la senal cumple patron, niveles validos y riesgo/beneficio aceptable."
    if not bool(decision.get("patron_confirmado")):
        return f"4H y 1H estan alineadas con sesgo {bias.lower()}, pero todavia falta patron confirmado en 1H."
    if not bool(decision.get("sl_tp_validos")):
        return "Hay contexto alineado, pero faltan stop loss y take profit validos."
    if not bool(decision.get("riesgo_beneficio_valido")):
        return "Hay contexto alineado, pero la relacion riesgo/beneficio todavia no es valida."
    return f"4H y 1H estan alineadas con sesgo {bias.lower()}, pero la entrada aun no cumple todos los filtros operativos."


def _neutralize_wait_warnings(warnings: Iterable[str]) -> list[str]:
    normalized: list[str] = []
    for warning in warnings:
        cleaned = str(warning).strip()
        if cleaned == "La tendencia actual va en contra del sesgo operativo.":
            normalized.append(
                "La lectura 1H aun no esta suficientemente alineada con 4H para habilitar entrada."
            )
            continue
        normalized.append(cleaned)
    return normalized


def _align_operative_decision(
    decision: dict[str, Any],
    multi_timeframe: dict[str, Any],
) -> dict[str, Any]:
    aligned = dict(decision)
    aligned["motivos"] = list(decision.get("motivos", []) or [])
    aligned["advertencias"] = list(decision.get("advertencias", []) or [])

    bias = str(multi_timeframe.get("sesgo_permitido", "ESPERAR"))
    raw_direction = str(decision.get("direccion", "NINGUNA"))
    allowed_direction = bias if bool(multi_timeframe.get("alineacion")) and bias in {"COMPRA", "VENTA"} else "NINGUNA"
    alignment_reason = _build_alignment_reason(multi_timeframe)
    score = int(aligned.get("score", 0))

    if allowed_direction == "NINGUNA":
        aligned["advertencias"] = _neutralize_wait_warnings(aligned["advertencias"])
        aligned["direccion"] = "NINGUNA"
        aligned["entrada_habilitada"] = False
        aligned["score"] = min(score, 69)
        if str(aligned.get("estado", NO_OPERAR)) == POSIBLE_OPERACION:
            aligned["estado"] = VIGILAR
        aligned["motivos"] = dedupe_keep_order(
            [
                alignment_reason,
                "El sesgo permitido permanece en esperar; no se habilita ninguna entrada.",
                (
                    "1H todavia no confirma un patron operativo valido."
                    if not bool(decision.get("patron_confirmado"))
                    else "Aunque exista una lectura local en 1H, no es operable mientras la multi-temporalidad siga en esperar."
                ),
            ]
        )
        aligned["advertencias"] = dedupe_keep_order(
            [
                "El contexto multi-temporal no autoriza una compra ni una venta en este momento.",
                *aligned["advertencias"],
            ]
        )
        aligned["resumen_operativo"] = _build_operational_summary(aligned, multi_timeframe)
        return aligned

    if raw_direction != allowed_direction:
        aligned["direccion"] = "NINGUNA"
        aligned["entrada_habilitada"] = False
        aligned["score"] = min(score, 69)
        if str(aligned.get("estado", NO_OPERAR)) == POSIBLE_OPERACION:
            aligned["estado"] = VIGILAR
        aligned["motivos"] = dedupe_keep_order(
            [
                alignment_reason,
                f"El sesgo permitido es {bias.lower()}, pero la lectura actual de 1H no coincide con ese sesgo.",
                "Hasta que la senal de entrada vuelva a alinearse con la multi-temporalidad, la entrada permanece en espera.",
            ]
        )
        aligned["advertencias"] = dedupe_keep_order(
            [
                "La senal local y el sesgo multi-temporal no coinciden; evitar operar contra el contexto permitido.",
                *aligned["advertencias"],
            ]
        )
        aligned["resumen_operativo"] = _build_operational_summary(aligned, multi_timeframe)
        return aligned

    aligned["direccion"] = allowed_direction
    aligned["resumen_operativo"] = _build_operational_summary(aligned, multi_timeframe)
    return aligned


def build_asset_snapshot(
    asset: str,
    analyses: dict[str, dict[str, Any]],
    selected_timeframe: str,
) -> dict[str, Any]:
    selected_analysis = analyses.get(selected_timeframe) or _empty_analysis(asset, selected_timeframe)
    entry_analysis = analyses.get("1h") or _empty_analysis(asset, "1h")
    context_analysis = analyses.get("4h") or _empty_analysis(asset, "4h")
    multi_timeframe = compare_multi_timeframe(analyses)
    operative_decision = _align_operative_decision(entry_analysis["decision"], multi_timeframe)
    watchlist = build_watchlist(
        asset=asset,
        decision=operative_decision,
        multi_timeframe=multi_timeframe,
        context_analysis=context_analysis,
        entry_analysis=entry_analysis,
    )
    horizon = classify_time_horizon(asset, analyses)
    price_source = _best_available_analysis(analyses, ["1h", "4h", "1d", "1wk", selected_timeframe])
    price_frame = price_source.get("data", pd.DataFrame())
    latest_timestamp = get_latest_timestamp(analyses)

    latest_price = None
    if not price_frame.empty:
        latest_price = float(price_frame["close"].iloc[-1])

    return {
        "asset": asset,
        "selected_timeframe": selected_timeframe,
        "selected_analysis": selected_analysis,
        "entry_analysis": entry_analysis,
        "context_analysis": context_analysis,
        "operative_decision": operative_decision,
        "multi_timeframe": multi_timeframe,
        "watchlist": watchlist,
        "time_horizon": horizon,
        "latest_price": latest_price,
        "latest_timestamp": latest_timestamp,
    }


def build_scanner_row(asset: str, analyses: dict[str, dict[str, Any]]) -> dict[str, Any]:
    snapshot = build_asset_snapshot(asset, analyses, selected_timeframe="1h")
    analysis_4h = analyses.get("4h") or _empty_analysis(asset, "4h")
    analysis_1h = analyses.get("1h") or _empty_analysis(asset, "1h")
    price = snapshot.get("latest_price")
    latest_timestamp = snapshot.get("latest_timestamp")
    decision = snapshot["operative_decision"]
    multi_timeframe = snapshot["multi_timeframe"]
    horizon = snapshot["time_horizon"]

    return {
        "activo": asset,
        "precio": format_price(float(price)) if price is not None else "sin datos",
        "tendencia 4H": str(analysis_4h.get("general_trend", "sin datos")).upper(),
        "tendencia 1H": str(analysis_1h.get("general_trend", "sin datos")).upper(),
        "alineación": "Sí" if bool(multi_timeframe.get("alineacion")) else "Esperar",
        "estado semáforo": str(decision.get("estado", "NO OPERAR")),
        "dirección": str(decision.get("direccion", "NINGUNA")),
        "score": int(decision.get("score", 0)),
        "horizonte sugerido": str(horizon.get("horizonte", "SIN CLASIFICAR")),
        "última actualización": (
            pd.Timestamp(latest_timestamp).strftime("%Y-%m-%d %H:%M")
            if latest_timestamp is not None
            else "sin datos"
        ),
    }


def build_scanner_table(
    assets: Iterable[str],
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    cache: dict[str, dict[str, Any]] = {}

    for asset in assets:
        analyses = build_asset_analyses(asset)
        cache[asset] = analyses
        rows.append(build_scanner_row(asset, analyses))

    return pd.DataFrame(rows), cache


def asset_has_required_data(asset: str, timeframes: Iterable[str] | None = None) -> bool:
    for timeframe in tuple(timeframes or TIMEFRAMES):
        if load_asset_data(asset, timeframe, limit=5).empty:
            return False
    return True


def _timeframe_label(timeframe: str) -> str:
    return timeframe.upper().replace("WK", "W")


def _trend_label(raw_trend: str | None) -> str:
    normalized = str(raw_trend or "").lower()
    if normalized == "alcista":
        return "Alcista"
    if normalized == "bajista":
        return "Bajista"
    if normalized == "lateral":
        return "Lateral"
    return "Sin datos"


def _resolve_asset(symbol: str | None, nombre: str | None) -> tuple[str, str]:
    if nombre and nombre in ASSETS:
        return nombre, ASSETS[nombre]

    if symbol:
        for asset_name, ticker in ASSETS.items():
            if ticker == symbol:
                return asset_name, ticker

    raise ValueError("No se pudo resolver el activo solicitado.")


def _analysis_to_public_dict(
    asset: str,
    ticker: str,
    snapshot: dict[str, Any],
) -> dict[str, Any]:
    decision = snapshot["operative_decision"]
    multi_timeframe = snapshot["multi_timeframe"]
    selected_analysis = snapshot["selected_analysis"]
    detail = multi_timeframe.get("detalle", {}) or {}
    detail_4h = detail.get("4H", {}) or {}
    detail_1h = detail.get("1H", {}) or {}
    latest_timestamp = snapshot.get("latest_timestamp")
    latest_price = snapshot.get("latest_price")
    public_bias = str(multi_timeframe.get("sesgo_permitido", "ESPERAR"))
    if public_bias not in {"COMPRA", "VENTA"}:
        public_bias = "ESPERAR"

    return {
        "activo": asset,
        "symbol": ticker,
        "precio": float(latest_price) if latest_price is not None else None,
        "precio_texto": format_price(float(latest_price)) if latest_price is not None else "sin datos",
        "temporalidad": _timeframe_label(str(snapshot.get("selected_timeframe", "1h"))),
        "tendencia": _trend_label(str(selected_analysis.get("general_trend", "sin datos"))),
        "contexto_4h": str(detail_4h.get("context", "MIXTO")).upper(),
        "contexto_1h": str(detail_1h.get("context", "MIXTO")).upper(),
        "alineacion": "Confirmada" if bool(multi_timeframe.get("alineacion")) else "No confirmada",
        "semaforo": public_semaforo(str(decision.get("estado", NO_OPERAR))),
        "sesgo": public_bias,
        "entrada": public_entry_status(bool(decision.get("entrada_habilitada"))),
        "score": int(decision.get("score", 0)),
        "motivos": list(decision.get("motivos", []) or []),
        "advertencias": list(decision.get("advertencias", []) or []),
        "timestamp": (
            pd.Timestamp(latest_timestamp).isoformat()
            if latest_timestamp is not None
            else None
        ),
        "snapshot": snapshot,
    }


def analizar_activo(
    symbol: str | None = None,
    nombre: str | None = None,
    *,
    timeframe: str = "1h",
    analyses: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    asset, ticker = _resolve_asset(symbol, nombre)
    local_analyses = analyses or build_asset_analyses(asset)
    snapshot = build_asset_snapshot(asset, local_analyses, selected_timeframe=timeframe)
    return _analysis_to_public_dict(asset, ticker, snapshot)


def analizar_activos(
    assets: dict[str, str],
    *,
    timeframe: str = "1h",
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for asset_name, ticker in assets.items():
        try:
            results.append(analizar_activo(symbol=ticker, nombre=asset_name, timeframe=timeframe))
        except Exception as exc:
            results.append(
                {
                    "activo": asset_name,
                    "symbol": ticker,
                    "precio": None,
                    "precio_texto": "sin datos",
                    "temporalidad": _timeframe_label(timeframe),
                    "tendencia": "Sin datos",
                    "contexto_4h": "MIXTO",
                    "contexto_1h": "MIXTO",
                    "alineacion": "No confirmada",
                    "semaforo": "ESPERAR",
                    "sesgo": "ESPERAR",
                    "entrada": "NO HABILITADA",
                    "score": 0,
                    "motivos": [f"No fue posible analizar {asset_name}: {exc}"],
                    "advertencias": ["Revisar la descarga de datos y los logs del watcher."],
                    "timestamp": None,
                }
            )
    return results
