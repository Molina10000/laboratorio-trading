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


def _opposite_direction(direction: str) -> str:
    if direction == "COMPRA":
        return "VENTA"
    if direction == "VENTA":
        return "COMPRA"
    return "NINGUNA"


def _directional_patterns(signals: list[dict[str, Any]], direction: str) -> list[dict[str, Any]]:
    if direction not in {"COMPRA", "VENTA"}:
        return []
    expected = "alcista" if direction == "COMPRA" else "bajista"
    return [signal for signal in signals if signal.get("direccion") == expected]


def _continuation_patterns(signals: list[dict[str, Any]], direction: str) -> list[dict[str, Any]]:
    return [
        signal
        for signal in _directional_patterns(signals, direction)
        if "ruptura" in str(signal.get("patron", "")).lower()
    ]


def _nearest_level(levels: list[float], price: float, prefer_above: bool | None = None) -> float | None:
    if not levels:
        return None

    cleaned = sorted(float(level) for level in levels)
    if prefer_above is True:
        above = [level for level in cleaned if level >= price]
        if above:
            return min(above, key=lambda level: abs(level - price))
    if prefer_above is False:
        below = [level for level in cleaned if level <= price]
        if below:
            return min(below, key=lambda level: abs(level - price))
    return min(cleaned, key=lambda level: abs(level - price))


def _format_level(level: float | None, reference_price: float) -> str:
    if level is None:
        return ""
    precision = 5 if abs(reference_price) < 10 else 2
    return f"{level:.{precision}f}"


def _collect_levels(
    entry_structure: dict[str, Any],
    context_structure: dict[str, Any],
    key: str,
) -> list[float]:
    entry_levels = [float(level) for level in entry_structure.get(key, [])]
    context_levels = [float(level) for level in context_structure.get(key, [])]
    return sorted({*entry_levels, *context_levels})


def _is_near_level(price: float, level: float | None, threshold: float) -> bool:
    return level is not None and abs(price - level) <= threshold


def _principal_direction(
    multi_timeframe: dict[str, Any],
    context_analysis: dict[str, Any],
) -> str:
    _ = context_analysis
    allowed_bias = str(multi_timeframe.get("sesgo_permitido", "ESPERAR"))
    if bool(multi_timeframe.get("alineacion")) and allowed_bias in {"COMPRA", "VENTA"}:
        return allowed_bias
    return "NINGUNA"


def _reward_risk_available(signals: list[dict[str, Any]], direction: str) -> bool:
    return any(
        _safe_float(signal.get("relacion_riesgo_beneficio")) >= 2
        for signal in _directional_patterns(signals, direction)
    )


def _build_principal_scenario(
    asset: str,
    decision: dict[str, Any],
    multi_timeframe: dict[str, Any],
    context_analysis: dict[str, Any],
    entry_analysis: dict[str, Any],
) -> dict[str, Any]:
    direction = _principal_direction(multi_timeframe, context_analysis)
    entry_frame = entry_analysis.get("data", pd.DataFrame())
    context_frame = context_analysis.get("data", pd.DataFrame())
    entry_structure = entry_analysis.get("market_structure", {}) or {}
    context_structure = context_analysis.get("market_structure", {}) or {}
    active_watch = str(decision.get("estado", "")) == "VIGILAR"

    if entry_frame.empty:
        return {
            "titulo": "Plan principal (a favor de tendencia)",
            "direccion": direction,
            "condiciones_cumplidas": [],
            "condiciones_pendientes": ["No hay datos 1H suficientes para construir el escenario principal."],
            "nivel_a_vigilar": "",
            "mensaje": "Actualizar datos antes de definir un plan principal.",
        }

    latest = entry_frame.iloc[-1]
    latest_price = _safe_float(latest.get("close"))
    latest_atr = max(_safe_float(latest.get("atr_14"), default=latest_price * 0.002), latest_price * 0.0015, 0.0001)
    zone_threshold = max(latest_atr * 0.75, latest_price * 0.0025)

    entry_trend = str(entry_analysis.get("general_trend", ""))
    entry_signals = entry_analysis.get("signals", []) or []
    supports = _collect_levels(entry_structure, context_structure, "supports")
    resistances = _collect_levels(entry_structure, context_structure, "resistances")

    conditions_met: list[str] = []
    conditions_pending: list[str] = []

    if direction == "COMPRA":
        breakout_level = _nearest_level(resistances, latest_price, prefer_above=True)
        pullback_level = _nearest_level(supports, latest_price, prefer_above=False)
        breakout_confirmed = breakout_level is not None and latest_price > breakout_level
        pullback_available = _is_near_level(latest_price, pullback_level, zone_threshold)
        continuation_confirmed = bool(_continuation_patterns(entry_signals, direction))
        close_confirmed = breakout_confirmed or continuation_confirmed
        watched_level = pullback_level if pullback_available and pullback_level is not None else breakout_level
        watched_text = _format_level(watched_level, latest_price)

        if str(context_analysis.get("general_trend", "")) == "alcista":
            conditions_met.append("Tendencia 4H alcista.")
        else:
            conditions_pending.append("Que 4H mantenga una tendencia alcista clara.")

        if bool(multi_timeframe.get("alineacion")) and str(multi_timeframe.get("sesgo_permitido")) == "COMPRA":
            conditions_met.append("1H alineada con el contexto 4H.")
        else:
            conditions_pending.append("Alineación 1H con el sesgo alcista de 4H.")

        if pullback_available:
            conditions_met.append("Retroceso a soporte relevante.")
        else:
            conditions_pending.append("Retroceso a soporte.")

        if breakout_confirmed:
            conditions_met.append("Ruptura de resistencia.")
        else:
            conditions_pending.append("Ruptura de resistencia.")

        if continuation_confirmed:
            conditions_met.append("Patrón de continuación alcista.")
        else:
            conditions_pending.append("Patrón de continuación.")

        if close_confirmed:
            conditions_met.append("Cierre confirmado a favor de la continuación.")
        else:
            conditions_pending.append("Cierre confirmado.")

        if _reward_risk_available(entry_signals, direction):
            conditions_met.append("Riesgo/beneficio calculado con mínimo 1:2.")
        else:
            conditions_pending.append("R/R calculado.")

        if not active_watch:
            message = "No hay plan principal activo porque el semáforo actual no está en VIGILAR."
        elif pullback_available and watched_text:
            message = (
                f"Si el precio retrocede a soporte {watched_text} y aparece patrón de continuación alcista, revisar posible compra."
            )
        elif watched_text:
            message = (
                f"Si el precio rompe resistencia y cierra encima de {watched_text}, revisar posible compra."
            )
        else:
            message = "Si aparece continuación alcista con cierre válido en 1H, revisar posible compra."

    elif direction == "VENTA":
        breakout_level = _nearest_level(supports, latest_price, prefer_above=False)
        pullback_level = _nearest_level(resistances, latest_price, prefer_above=True)
        breakout_confirmed = breakout_level is not None and latest_price < breakout_level
        pullback_available = _is_near_level(latest_price, pullback_level, zone_threshold)
        continuation_confirmed = bool(_continuation_patterns(entry_signals, direction))
        close_confirmed = breakout_confirmed or continuation_confirmed
        watched_level = pullback_level if pullback_available and pullback_level is not None else breakout_level
        watched_text = _format_level(watched_level, latest_price)

        if str(context_analysis.get("general_trend", "")) == "bajista":
            conditions_met.append("Tendencia 4H bajista.")
        else:
            conditions_pending.append("Que 4H mantenga una tendencia bajista clara.")

        if bool(multi_timeframe.get("alineacion")) and str(multi_timeframe.get("sesgo_permitido")) == "VENTA":
            conditions_met.append("1H alineada con el contexto 4H.")
        else:
            conditions_pending.append("Alineación 1H con el sesgo bajista de 4H.")

        if pullback_available:
            conditions_met.append("Retroceso a resistencia relevante.")
        else:
            conditions_pending.append("Retroceso a resistencia.")

        if breakout_confirmed:
            conditions_met.append("Ruptura de soporte.")
        else:
            conditions_pending.append("Ruptura de soporte.")

        if continuation_confirmed:
            conditions_met.append("Patrón de continuación bajista.")
        else:
            conditions_pending.append("Patrón de continuación.")

        if close_confirmed:
            conditions_met.append("Cierre confirmado a favor de la continuación.")
        else:
            conditions_pending.append("Cierre confirmado.")

        if _reward_risk_available(entry_signals, direction):
            conditions_met.append("Riesgo/beneficio calculado con mínimo 1:2.")
        else:
            conditions_pending.append("R/R calculado.")

        if not active_watch:
            message = "No hay plan principal activo porque el semáforo actual no está en VIGILAR."
        elif pullback_available and watched_text:
            message = (
                f"Si el precio retrocede a resistencia {watched_text} y aparece patrón de continuación bajista, revisar posible venta."
            )
        elif watched_text:
            message = (
                f"Si el precio rompe soporte y cierra debajo de {watched_text}, revisar posible venta."
            )
        else:
            message = "Si aparece continuación bajista con cierre válido en 1H, revisar posible venta."

    else:
        watched_text = ""
        conditions_pending.extend(
            [
                "Definir una tendencia principal clara en 4H.",
                "Esperar alineación de 1H con el contexto 4H.",
            ]
        )
        message = "Sin sesgo principal claro. Esperar una dirección dominante en multi-temporalidad."

    return {
        "titulo": "Plan principal (a favor de tendencia)",
        "direccion": direction,
        "condiciones_cumplidas": conditions_met,
        "condiciones_pendientes": conditions_pending,
        "nivel_a_vigilar": watched_text,
        "mensaje": message,
    }


def _build_secondary_scenario(
    principal_direction: str,
    context_analysis: dict[str, Any],
    entry_analysis: dict[str, Any],
) -> dict[str, Any]:
    secondary_direction = _opposite_direction(principal_direction)
    entry_frame = entry_analysis.get("data", pd.DataFrame())
    entry_structure = entry_analysis.get("market_structure", {}) or {}
    entry_signals = entry_analysis.get("signals", []) or []

    if entry_frame.empty or secondary_direction == "NINGUNA":
        return {
            "titulo": "Escenario secundario (reversión contra tendencia)",
            "direccion": secondary_direction,
            "advertencia": "Mayor riesgo porque contradice la tendencia principal.",
            "condiciones_cumplidas": [],
            "condiciones_pendientes": ["Esperar una tendencia principal clara antes de definir una reversión."],
            "nivel_a_vigilar": "",
            "mensaje": "Sin escenario secundario válido por ahora.",
        }

    latest = entry_frame.iloc[-1]
    latest_price = _safe_float(latest.get("close"))
    latest_rsi = _safe_float(latest.get("rsi_14"), default=50.0)
    ema_50 = _safe_float(latest.get("ema_50"), default=latest_price)
    latest_atr = max(_safe_float(latest.get("atr_14"), default=latest_price * 0.002), latest_price * 0.0015, 0.0001)
    zone_threshold = max(latest_atr * 0.75, latest_price * 0.0025)

    support_levels = [float(level) for level in entry_structure.get("supports", [])]
    resistance_levels = [float(level) for level in entry_structure.get("resistances", [])]
    structure_trend = str(entry_structure.get("trend", ""))
    reversal_signals = _directional_patterns(entry_signals, secondary_direction)

    conditions_met: list[str] = []
    conditions_pending: list[str] = []

    if secondary_direction == "COMPRA":
        watched_level = _nearest_level(support_levels, latest_price, prefer_above=False)
        watched_text = _format_level(watched_level, latest_price)
        if latest_rsi <= 35:
            conditions_met.append("RSI en sobreventa como confirmación de posible rebote.")
        else:
            conditions_pending.append("RSI en sobreventa para apoyar la reversión.")

        if _is_near_level(latest_price, watched_level, zone_threshold):
            conditions_met.append("Precio cerca de soporte relevante.")
        else:
            conditions_pending.append("Llegada o reacción clara en soporte.")

        if structure_trend == "alcista":
            conditions_met.append("Cambio de estructura hacia máximos y mínimos crecientes.")
        else:
            conditions_pending.append("Cambio de estructura.")

        if latest_price >= ema_50:
            conditions_met.append("Recuperación de EMA 50.")
        else:
            conditions_pending.append("Recuperación EMA.")

        if reversal_signals:
            conditions_met.append("Patrón alcista confirmado.")
        else:
            conditions_pending.append("Patrón confirmado.")

        if watched_text:
            message = (
                f"Si el precio rebota en soporte {watched_text}, recupera EMA 50 y confirma patrón alcista, revisar posible rebote contra tendencia."
            )
        else:
            message = "Si aparece cambio de estructura, recuperación de EMA y patrón alcista, revisar posible rebote contra tendencia."

    else:
        watched_level = _nearest_level(resistance_levels, latest_price, prefer_above=True)
        watched_text = _format_level(watched_level, latest_price)
        if latest_rsi >= 65:
            conditions_met.append("RSI en sobrecompra como confirmación de posible reversión.")
        else:
            conditions_pending.append("RSI en sobrecompra para apoyar la reversión.")

        if _is_near_level(latest_price, watched_level, zone_threshold):
            conditions_met.append("Precio cerca de resistencia relevante.")
        else:
            conditions_pending.append("Llegada o rechazo claro en resistencia.")

        if structure_trend == "bajista":
            conditions_met.append("Cambio de estructura hacia máximos y mínimos decrecientes.")
        else:
            conditions_pending.append("Cambio de estructura.")

        if latest_price <= ema_50:
            conditions_met.append("Pérdida de EMA 50.")
        else:
            conditions_pending.append("Pérdida EMA.")

        if reversal_signals:
            conditions_met.append("Patrón bajista confirmado.")
        else:
            conditions_pending.append("Patrón confirmado.")

        if watched_text:
            message = (
                f"Si el precio rechaza resistencia {watched_text}, pierde EMA 50 y confirma patrón bajista, revisar posible reversión contra tendencia."
            )
        else:
            message = "Si aparece cambio de estructura, pérdida de EMA y patrón bajista, revisar posible reversión contra tendencia."

    return {
        "titulo": "Escenario secundario (reversión contra tendencia)",
        "direccion": secondary_direction,
        "advertencia": "Mayor riesgo porque contradice la tendencia principal.",
        "condiciones_cumplidas": conditions_met,
        "condiciones_pendientes": conditions_pending,
        "nivel_a_vigilar": watched_text,
        "mensaje": message,
    }


def build_watchlist(
    asset: str,
    decision: dict[str, Any],
    multi_timeframe: dict[str, Any],
    context_analysis: dict[str, Any],
    entry_analysis: dict[str, Any],
) -> dict[str, Any]:
    principal = _build_principal_scenario(
        asset=asset,
        decision=decision,
        multi_timeframe=multi_timeframe,
        context_analysis=context_analysis,
        entry_analysis=entry_analysis,
    )
    secondary = _build_secondary_scenario(
        principal_direction=str(principal.get("direccion", "NINGUNA")),
        context_analysis=context_analysis,
        entry_analysis=entry_analysis,
    )

    return {
        "activo": asset,
        "direccion_observada": principal.get("direccion", "NINGUNA"),
        "escenario_principal": principal,
        "escenario_secundario": secondary,
        "condiciones_cumplidas": principal.get("condiciones_cumplidas", []),
        "condiciones_pendientes": principal.get("condiciones_pendientes", []),
        "nivel_a_vigilar": principal.get("nivel_a_vigilar", ""),
        "mensaje": principal.get("mensaje", ""),
    }
