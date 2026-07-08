from __future__ import annotations

from typing import Any

import pandas as pd

from indicators import detect_general_trend


NO_OPERAR = "NO OPERAR"
VIGILAR = "VIGILAR"
POSIBLE_OPERACION = "POSIBLE OPERACIÓN"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp_score(score: float) -> int:
    return int(max(0, min(100, round(score))))


def _normalize_direction(raw_direction: str | None) -> str:
    if raw_direction == "alcista":
        return "COMPRA"
    if raw_direction == "bajista":
        return "VENTA"
    return "NINGUNA"


def _resolve_trend(general_trend: str, structure_trend: str) -> str:
    directional = {"alcista", "bajista"}

    if general_trend == structure_trend and general_trend in directional:
        return general_trend
    if general_trend in directional and structure_trend in {"lateral", "sin datos", "sin datos suficientes"}:
        return general_trend
    if structure_trend in directional and general_trend in {"lateral", "sin datos", "sin datos suficientes"}:
        return structure_trend
    if general_trend in directional and structure_trend in directional and general_trend != structure_trend:
        return "lateral"
    if general_trend in directional:
        return general_trend
    if structure_trend in directional:
        return structure_trend
    return "lateral"


def _trend_supports_direction(trend: str, direction: str) -> bool:
    return (trend == "alcista" and direction == "COMPRA") or (
        trend == "bajista" and direction == "VENTA"
    )


def _closest_level_distance(price: float, levels: list[float]) -> float | None:
    if not levels:
        return None
    return min(abs(price - float(level)) for level in levels)


def _has_valid_trade_levels(signal: dict[str, Any] | None) -> bool:
    if not signal:
        return False

    entry = _safe_float(signal.get("precio_entrada"))
    stop = _safe_float(signal.get("stop_loss"))
    target = _safe_float(signal.get("take_profit"))
    return all(value > 0 for value in [entry, stop, target]) and stop != entry and target != entry


def _select_primary_signal(
    signals: list[dict[str, Any]],
    resolved_trend: str,
) -> dict[str, Any] | None:
    if not signals:
        return None

    def score_signal(signal: dict[str, Any]) -> tuple[float, float]:
        rr = _safe_float(signal.get("relacion_riesgo_beneficio"))
        direction = _normalize_direction(signal.get("direccion"))
        alignment = 1.0 if _trend_supports_direction(resolved_trend, direction) else 0.0
        return (rr, alignment)

    return max(signals, key=score_signal)


def _has_opposite_signals(signals: list[dict[str, Any]]) -> bool:
    directions = {_normalize_direction(signal.get("direccion")) for signal in signals}
    return "COMPRA" in directions and "VENTA" in directions


def _detect_pending_breakout(
    frame: pd.DataFrame,
    atr_value: float,
) -> dict[str, str] | None:
    if len(frame) < 25:
        return None

    recent_high = _safe_float(frame["high"].shift(1).rolling(20).max().iloc[-1], default=float("nan"))
    recent_low = _safe_float(frame["low"].shift(1).rolling(20).min().iloc[-1], default=float("nan"))
    if pd.isna(recent_high) or pd.isna(recent_low):
        return None

    close_price = _safe_float(frame["close"].iloc[-1])
    latest_high = _safe_float(frame["high"].iloc[-1])
    latest_low = _safe_float(frame["low"].iloc[-1])
    range_width = recent_high - recent_low

    if range_width <= 0 or range_width > atr_value * 4:
        return None

    threshold = max(atr_value * 0.35, close_price * 0.0015)

    if latest_high >= recent_high - threshold and close_price < recent_high:
        return {
            "direction": "COMPRA",
            "message": "Posible ruptura alcista de rango, pero falta cierre de confirmación.",
        }

    if latest_low <= recent_low + threshold and close_price > recent_low:
        return {
            "direction": "VENTA",
            "message": "Posible ruptura bajista de rango, pero falta cierre de confirmación.",
        }

    return None


def _infer_context_direction(
    resolved_trend: str,
    near_support: bool,
    near_resistance: bool,
    rsi_value: float,
    near_ema_50: bool,
    near_ema_200: bool,
    pending_breakout: dict[str, str] | None,
) -> str:
    if pending_breakout is not None:
        return pending_breakout["direction"]

    if near_support and not near_resistance:
        if resolved_trend != "bajista" or rsi_value < 35:
            return "COMPRA"

    if near_resistance and not near_support:
        if resolved_trend != "alcista" or rsi_value > 65:
            return "VENTA"

    if resolved_trend == "alcista" and (near_ema_50 or near_ema_200):
        return "COMPRA"

    if resolved_trend == "bajista" and (near_ema_50 or near_ema_200):
        return "VENTA"

    if rsi_value < 35:
        return "COMPRA"

    if rsi_value > 65:
        return "VENTA"

    return "NINGUNA"


def _zone_supports_direction(
    direction: str,
    support_distance: float | None,
    resistance_distance: float | None,
    threshold: float,
    signal: dict[str, Any] | None,
) -> tuple[bool, str]:
    pattern_name = str(signal.get("patron", "")) if signal else ""

    if direction == "COMPRA":
        if support_distance is not None and support_distance <= threshold:
            return True, "El precio está cerca de un soporte relevante."
        if "ruptura" in pattern_name and resistance_distance is not None and resistance_distance <= threshold:
            return True, "El precio está rompiendo una resistencia con confirmación."

    if direction == "VENTA":
        if resistance_distance is not None and resistance_distance <= threshold:
            return True, "El precio está cerca de una resistencia relevante."
        if "ruptura" in pattern_name and support_distance is not None and support_distance <= threshold:
            return True, "El precio está rompiendo un soporte con confirmación."

    return False, ""


def _rsi_supports_direction(direction: str, rsi_value: float) -> tuple[bool, str]:
    if direction == "COMPRA" and rsi_value < 75:
        if rsi_value < 35:
            return True, "RSI en zona de sobreventa o recuperación para compras."
        if 35 <= rsi_value <= 60:
            return True, "RSI acompaña la idea alcista sin sobrecompra extrema."

    if direction == "VENTA" and rsi_value > 25:
        if rsi_value > 65:
            return True, "RSI en zona de sobrecompra o agotamiento para ventas."
        if 40 <= rsi_value <= 65:
            return True, "RSI acompaña la idea bajista sin sobreventa extrema."

    return False, ""


def _rsi_extreme_against_direction(direction: str, rsi_value: float) -> tuple[bool, str]:
    if direction == "COMPRA" and rsi_value >= 75:
        return True, "RSI demasiado sobrecomprado para considerar una compra conservadora."
    if direction == "VENTA" and rsi_value <= 25:
        return True, "RSI demasiado sobrevendido para considerar una venta conservadora."
    return False, ""


def _ema_supports_direction(
    direction: str,
    price: float,
    ema_50: float,
    ema_200: float,
    near_ema_50: bool,
    near_ema_200: bool,
) -> tuple[bool, str]:
    if direction == "COMPRA":
        if price >= ema_50 or (near_ema_50 and ema_50 >= ema_200) or (near_ema_200 and price >= ema_200):
            return True, "El precio está por encima de EMA 50/200 o reaccionando sobre ellas."

    if direction == "VENTA":
        if price <= ema_50 or (near_ema_50 and ema_50 <= ema_200) or (near_ema_200 and price <= ema_200):
            return True, "El precio está por debajo de EMA 50/200 o reaccionando bajo ellas."

    return False, ""


def _dedupe_keep_order(messages: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for message in messages:
        if not message or message in seen:
            continue
        seen.add(message)
        ordered.append(message)
    return ordered


def evaluate_trade_decision(
    frame: pd.DataFrame,
    asset: str,
    timeframe: str,
    market_structure: dict[str, Any],
    signals: list[dict[str, Any]],
    general_trend: str | None = None,
) -> dict[str, Any]:
    if frame.empty:
        return {
            "estado": NO_OPERAR,
            "direccion": "NINGUNA",
            "score": 0,
            "motivos": [f"No hay datos suficientes para evaluar {asset} en {timeframe}."],
            "advertencias": ["Verificar la descarga de datos antes de analizar."],
            "patron_confirmado": False,
            "sl_tp_validos": False,
            "riesgo_beneficio_valido": False,
            "entrada_habilitada": False,
        }

    latest = frame.iloc[-1]
    close_price = _safe_float(latest.get("close"))
    rsi_value = _safe_float(latest.get("rsi_14"), default=50.0)
    ema_50 = _safe_float(latest.get("ema_50"), default=close_price)
    ema_200 = _safe_float(latest.get("ema_200"), default=close_price)
    atr_value = _safe_float(latest.get("atr_14"), default=max(close_price * 0.002, 0.0001))
    atr_value = max(atr_value, close_price * 0.0015, 0.0001)

    computed_general_trend = general_trend or detect_general_trend(frame)
    structure_trend = str(market_structure.get("trend", "lateral"))
    resolved_trend = _resolve_trend(computed_general_trend, structure_trend)

    supports = [float(level) for level in market_structure.get("supports", [])]
    resistances = [float(level) for level in market_structure.get("resistances", [])]
    support_distance = _closest_level_distance(close_price, supports)
    resistance_distance = _closest_level_distance(close_price, resistances)

    zone_threshold = max(atr_value * 0.75, close_price * 0.0025)
    ema_threshold = max(atr_value * 0.5, close_price * 0.0015)
    near_support = support_distance is not None and support_distance <= zone_threshold
    near_resistance = resistance_distance is not None and resistance_distance <= zone_threshold
    near_ema_50 = abs(close_price - ema_50) <= ema_threshold
    near_ema_200 = abs(close_price - ema_200) <= ema_threshold
    ema_distance_ratio = abs(ema_50 - ema_200) / max(close_price, 1e-9)
    ema_compact = ema_distance_ratio <= 0.0015

    primary_signal = _select_primary_signal(signals, resolved_trend)
    has_pattern = primary_signal is not None
    direction = (
        _normalize_direction(primary_signal.get("direccion"))
        if primary_signal is not None
        else "NINGUNA"
    )
    pending_breakout = _detect_pending_breakout(frame, atr_value)

    if direction == "NINGUNA":
        direction = _infer_context_direction(
            resolved_trend=resolved_trend,
            near_support=near_support,
            near_resistance=near_resistance,
            rsi_value=rsi_value,
            near_ema_50=near_ema_50,
            near_ema_200=near_ema_200,
            pending_breakout=pending_breakout,
        )

    score = 0.0
    motives: list[str] = []
    warnings: list[str] = []

    if has_pattern:
        score += 30
        motives.append(f"Patrón confirmado: {primary_signal['patron']}.")
    else:
        warnings.append("No hay patrón confirmado en la última vela analizada.")

    if direction != "NINGUNA":
        if _trend_supports_direction(resolved_trend, direction):
            score += 20
            motives.append(
                f"La tendencia {resolved_trend} está alineada con el sesgo de {direction.lower()}."
            )
        elif resolved_trend in {"alcista", "bajista"}:
            score -= 20
            warnings.append("La tendencia actual va en contra del sesgo operativo.")

    zone_match, zone_message = _zone_supports_direction(
        direction=direction,
        support_distance=support_distance,
        resistance_distance=resistance_distance,
        threshold=zone_threshold,
        signal=primary_signal,
    )
    if zone_match:
        score += 15
        motives.append(zone_message)

    rsi_match, rsi_message = _rsi_supports_direction(direction, rsi_value)
    if rsi_match:
        score += 10
        motives.append(rsi_message)

    rsi_extreme, rsi_warning = _rsi_extreme_against_direction(direction, rsi_value)
    if rsi_extreme:
        score -= 15
        warnings.append(rsi_warning)

    has_valid_levels = _has_valid_trade_levels(primary_signal)
    reward_risk_ratio = (
        _safe_float(primary_signal.get("relacion_riesgo_beneficio"))
        if primary_signal is not None
        else 0.0
    )
    if has_pattern:
        if has_valid_levels and reward_risk_ratio >= 2:
            score += 15
            motives.append(
                f"Relación riesgo/beneficio estimada aceptable ({reward_risk_ratio}:1)."
            )
        elif has_valid_levels and reward_risk_ratio < 2:
            score -= 25
            warnings.append(
                f"Relación riesgo/beneficio insuficiente ({reward_risk_ratio}:1)."
            )
        else:
            score -= 30
            warnings.append("Faltan stop loss o take profit calculados.")

    ema_match, ema_message = _ema_supports_direction(
        direction=direction,
        price=close_price,
        ema_50=ema_50,
        ema_200=ema_200,
        near_ema_50=near_ema_50,
        near_ema_200=near_ema_200,
    )
    if ema_match:
        score += 10
        motives.append(ema_message)

    opposite_signals = _has_opposite_signals(signals)
    contradictory_signals = False
    if computed_general_trend in {"alcista", "bajista"} and structure_trend in {"alcista", "bajista"}:
        contradictory_signals = computed_general_trend != structure_trend
    contradictory_signals = contradictory_signals or ema_compact or opposite_signals

    if contradictory_signals:
        score -= 20
        if ema_compact:
            warnings.append("EMA 50 y EMA 200 están muy juntas y sin dirección clara.")
        if computed_general_trend != structure_trend and computed_general_trend in {"alcista", "bajista"} and structure_trend in {"alcista", "bajista"}:
            warnings.append("La tendencia por medias y la estructura de mercado no coinciden.")
        if opposite_signals:
            warnings.append("Hay señales alcistas y bajistas a la vez en la misma lectura.")

    if pending_breakout is not None and not has_pattern:
        motives.append(pending_breakout["message"])

    if resolved_trend in {"alcista", "bajista"} and not has_pattern:
        motives.append("Hay tendencia clara, pero todavía no hay patrón confirmado.")

    if (near_ema_50 or near_ema_200) and not has_pattern:
        motives.append("El precio está cerca de EMA 50 o EMA 200 y podría reaccionar.")

    if 40 <= rsi_value <= 60 and not has_pattern:
        warnings.append("RSI en zona neutral entre 40 y 60, sin impulso claro.")

    watch_conditions = any(
        [
            near_support,
            near_resistance,
            rsi_value < 35,
            rsi_value > 65,
            pending_breakout is not None,
            (resolved_trend in {"alcista", "bajista"} and not has_pattern),
            near_ema_50,
            near_ema_200,
        ]
    )

    hard_no_trade = (
        not has_pattern
        and resolved_trend == "lateral"
        and not near_support
        and not near_resistance
        and 40 <= rsi_value <= 60
        and ema_compact
    )

    final_score = _clamp_score(score)
    if final_score <= 39:
        state = NO_OPERAR
    elif final_score <= 69:
        state = VIGILAR
    else:
        state = POSIBLE_OPERACION

    if hard_no_trade:
        state = NO_OPERAR
        final_score = min(final_score, 35)
        warnings.append(
            "Mercado lateral, RSI neutral y lejos de soportes o resistencias relevantes."
        )

    if not has_pattern:
        if state == POSIBLE_OPERACION:
            state = VIGILAR
            final_score = min(final_score, 69)
        if state == NO_OPERAR and watch_conditions and not hard_no_trade:
            state = VIGILAR
            final_score = max(final_score, 45)
        warnings.append("Sin patrón confirmado, no se habilita una posible operación.")

    if has_pattern and not has_valid_levels:
        if state == POSIBLE_OPERACION:
            state = VIGILAR
            final_score = min(final_score, 69)
        warnings.append(
            "Sin stop loss y take profit calculados, no se habilita una posible operación."
        )

    if has_pattern and has_valid_levels and reward_risk_ratio < 2:
        if state == POSIBLE_OPERACION:
            state = VIGILAR
            final_score = min(final_score, 69)
        warnings.append(
            "Con riesgo/beneficio menor a 1:2, no se habilita una posible operación."
        )

    if direction == "NINGUNA" and state == POSIBLE_OPERACION:
        state = VIGILAR
        final_score = min(final_score, 69)
        warnings.append("No hay dirección operativa clara para una validación manual seria.")

    if not motives:
        motives = ["No operar. No hay una señal suficientemente clara según las reglas actuales."]
    if state == VIGILAR and not watch_conditions:
        warnings.append("La lectura quedó en vigilancia por score, pero el contexto sigue siendo débil.")

    motives = _dedupe_keep_order(motives)
    warnings = _dedupe_keep_order(warnings)
    rr_valid = has_valid_levels and reward_risk_ratio >= 2
    entry_enabled = (
        has_pattern
        and has_valid_levels
        and rr_valid
        and direction in {"COMPRA", "VENTA"}
        and state == POSIBLE_OPERACION
    )

    return {
        "estado": state,
        "direccion": direction,
        "score": final_score,
        "motivos": motives,
        "advertencias": warnings,
        "patron_confirmado": has_pattern,
        "sl_tp_validos": has_valid_levels,
        "riesgo_beneficio_valido": rr_valid,
        "entrada_habilitada": entry_enabled,
    }
