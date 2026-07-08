from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

import pandas as pd

from data_loader import (
    DataDownloadError,
    download_ohlc_range,
    get_asset_metadata,
    get_cost_point_size,
    get_cost_unit_label,
)
from indicators import add_indicators
from patterns import detect_patterns


MODE_OPERATIVO = "operativo"
MODE_DIAGNOSTICO = "diagnostico"
SEGMENT_ALL = "all"
SEGMENT_DEVELOPMENT = "development"
SEGMENT_VALIDATION = "validation"
SEGMENT_TEST = "test"

TIMEFRAME_DELTAS = {
    "1h": pd.Timedelta(hours=1),
    "4h": pd.Timedelta(hours=4),
    "1d": pd.Timedelta(days=1),
    "1wk": pd.Timedelta(weeks=1),
}

INTRABAR_HISTORY_LIMITS = {
    "5m": pd.Timedelta(days=60),
    "15m": pd.Timedelta(days=60),
}
PATTERN_LOOKBACK_BARS = 260


@dataclass(slots=True)
class TradingCosts:
    spread: float = 0.0
    commission: float = 0.0
    commission_mode: str = "fixed"
    slippage: float = 0.0
    swap_per_night: float = 0.0

    def commission_in_price_units(self, entry_price: float, point_size: float) -> float:
        if self.commission_mode == "percent":
            return entry_price * (self.commission / 100.0)
        return self.commission * point_size


@dataclass(slots=True)
class BacktestConfig:
    configuration_name: str
    max_holding_bars: int = 24
    mode: str = MODE_OPERATIVO
    start_date: str | None = None
    end_date: str | None = None
    data_segment: str = SEGMENT_ALL
    warmup_bars: int = 220
    gap_threshold_pct: float = 3.0
    costs: TradingCosts = field(default_factory=TradingCosts)
    allow_intrabar_lookup: bool = True
    lower_timeframe_priority: tuple[str, ...] = ("15m", "5m")

    def parameter_snapshot(
        self,
        *,
        asset: str,
        timeframe: str,
    ) -> dict[str, Any]:
        point_size = get_cost_point_size(asset)
        cost_unit_label = get_cost_unit_label(asset)
        return {
            "asset": asset,
            "timeframe": timeframe,
            "max_holding_bars": int(self.max_holding_bars),
            "mode": self.mode,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "data_segment": self.data_segment,
            "warmup_bars": int(self.warmup_bars),
            "gap_threshold_pct": float(self.gap_threshold_pct),
            "spread": float(self.costs.spread),
            "commission": float(self.costs.commission),
            "commission_mode": self.costs.commission_mode,
            "slippage": float(self.costs.slippage),
            "swap_per_night": float(self.costs.swap_per_night),
            "cost_unit_label": cost_unit_label,
            "cost_point_size": point_size,
            "allow_intrabar_lookup": bool(self.allow_intrabar_lookup),
            "lower_timeframe_priority": list(self.lower_timeframe_priority),
        }

    def parameter_hash(self, *, asset: str, timeframe: str) -> str:
        payload = json.dumps(
            self.parameter_snapshot(asset=asset, timeframe=timeframe),
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _expected_delta(timeframe: str) -> pd.Timedelta | None:
    return TIMEFRAME_DELTAS.get(timeframe)


def _append_issue(
    issues: list[dict[str, Any]],
    *,
    severity: str,
    code: str,
    message: str,
    details: str = "",
) -> None:
    issues.append(
        {
            "severity": severity,
            "code": code,
            "message": message,
            "details": details,
        }
    )


def _serialize_timestamps(index: pd.Index, limit: int = 5) -> str:
    timestamps = [pd.Timestamp(value).strftime("%Y-%m-%d %H:%M") for value in index[:limit]]
    return ", ".join(timestamps)


def _is_expected_market_gap(
    previous_timestamp: pd.Timestamp,
    current_timestamp: pd.Timestamp,
    timeframe: str,
    diff: pd.Timedelta,
) -> bool:
    expected = _expected_delta(timeframe)
    if expected is None:
        return False

    if timeframe in {"1h", "4h"}:
        if previous_timestamp.weekday() >= 4 and current_timestamp.weekday() <= 1:
            return diff >= expected
    if timeframe == "1d":
        if previous_timestamp.weekday() == 4 and current_timestamp.weekday() == 0:
            return diff >= pd.Timedelta(days=3)
    if timeframe == "1wk":
        return False
    return False


def validate_market_data(
    frame: pd.DataFrame,
    timeframe: str,
    *,
    gap_threshold_pct: float,
    label: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    issues: list[dict[str, Any]] = []
    if frame.empty:
        _append_issue(
            issues,
            severity="error",
            code="empty_frame",
            message=f"{label}: no hay datos para validar.",
        )
        return frame.copy(), pd.DataFrame(issues)

    working = frame.copy()
    raw_index = pd.to_datetime(working.index)
    if not raw_index.is_monotonic_increasing:
        _append_issue(
            issues,
            severity="error",
            code="out_of_order",
            message=f"{label}: se detectaron velas fuera de orden cronológico.",
        )

    duplicates = raw_index.duplicated(keep=False)
    if duplicates.any():
        duplicated_index = raw_index[duplicates]
        _append_issue(
            issues,
            severity="error",
            code="duplicates",
            message=f"{label}: se detectaron velas duplicadas.",
            details=_serialize_timestamps(duplicated_index),
        )

    if getattr(raw_index, "tz", None) is not None:
        _append_issue(
            issues,
            severity="warning",
            code="timezone_aware",
            message=f"{label}: el índice conserva zona horaria. Se normalizará para la simulación.",
            details=str(raw_index.tz),
        )

    working.index = raw_index
    working = working.sort_index()
    if working.index.duplicated().any():
        working = working[~working.index.duplicated(keep="last")]

    expected_delta = _expected_delta(timeframe)
    if expected_delta is not None and len(working.index) > 1:
        diffs = working.index.to_series().diff().dropna()
        missing_ranges: list[str] = []
        missing_candles = 0

        for current_timestamp, diff in diffs.items():
            previous_timestamp = pd.Timestamp(current_timestamp) - diff
            if diff <= expected_delta:
                continue
            if _is_expected_market_gap(previous_timestamp, pd.Timestamp(current_timestamp), timeframe, diff):
                continue
            missing_here = max(int(diff / expected_delta) - 1, 0)
            if missing_here > 0:
                missing_candles += missing_here
                missing_ranges.append(
                    f"{previous_timestamp.strftime('%Y-%m-%d %H:%M')} -> {pd.Timestamp(current_timestamp).strftime('%Y-%m-%d %H:%M')}"
                )

        if missing_candles > 0:
            _append_issue(
                issues,
                severity="warning",
                code="missing_candles",
                message=f"{label}: se detectaron velas faltantes.",
                details=f"faltantes={missing_candles}; ejemplos={'; '.join(missing_ranges[:5])}",
            )

        if timeframe in {"1h", "4h"}:
            dst_like = diffs[
                diffs.isin(
                    [
                        expected_delta - pd.Timedelta(hours=1),
                        expected_delta + pd.Timedelta(hours=1),
                    ]
                )
            ]
            if not dst_like.empty:
                _append_issue(
                    issues,
                    severity="warning",
                    code="possible_dst",
                    message=f"{label}: se detectaron saltos compatibles con horario de verano o inconsistencia horaria.",
                    details=_serialize_timestamps(dst_like.index),
                )

    previous_close = working["close"].shift(1)
    gap_pct = ((working["open"] - previous_close).abs() / previous_close.replace(0, pd.NA)) * 100
    anomalies = gap_pct[gap_pct > gap_threshold_pct].dropna()
    if not anomalies.empty:
        _append_issue(
            issues,
            severity="warning",
            code="anomalous_gaps",
            message=f"{label}: se detectaron gaps superiores al umbral configurado.",
            details=", ".join(
                f"{pd.Timestamp(index).strftime('%Y-%m-%d %H:%M')} ({value:.2f}%)"
                for index, value in anomalies.head(5).items()
            ),
        )

    price_shape_errors = working[
        (working["high"] < working["low"])
        | (working["high"] < working["open"])
        | (working["high"] < working["close"])
        | (working["low"] > working["open"])
        | (working["low"] > working["close"])
    ]
    if not price_shape_errors.empty:
        _append_issue(
            issues,
            severity="error",
            code="invalid_ohlc",
            message=f"{label}: se detectaron velas con OHLC inconsistente.",
            details=_serialize_timestamps(price_shape_errors.index),
        )

    return working, pd.DataFrame(issues)


def _trend_from_row(row: pd.Series) -> str:
    close_price = _safe_float(row.get("close"))
    ema_50 = _safe_float(row.get("ema_50"), default=close_price)
    ema_200 = _safe_float(row.get("ema_200"), default=close_price)
    distance_ratio = abs(ema_50 - ema_200) / max(close_price, 1e-9)
    if close_price > ema_50 > ema_200 and distance_ratio >= 0.002:
        return "alcista"
    if close_price < ema_50 < ema_200 and distance_ratio >= 0.002:
        return "bajista"
    return "lateral"


def _build_context_lookup(context_frame: pd.DataFrame | None) -> pd.DataFrame:
    if context_frame is None or context_frame.empty:
        return pd.DataFrame(columns=["trend_4h"])

    enriched_context = add_indicators(context_frame.copy())
    lookup = pd.DataFrame(index=enriched_context.index.copy())
    lookup["trend_4h"] = enriched_context.apply(_trend_from_row, axis=1)
    return lookup


def _resolve_context_for_signal(
    context_lookup: pd.DataFrame,
    signal_time: pd.Timestamp,
    direction: str,
) -> tuple[str, str]:
    if context_lookup.empty:
        return "sin_contexto", "sin datos"

    position = context_lookup.index.searchsorted(signal_time, side="right") - 1
    if position < 0:
        return "sin_contexto", "sin datos"

    trend_4h = str(context_lookup.iloc[position]["trend_4h"])
    if trend_4h == "alcista" and direction == "alcista":
        return "a_favor", trend_4h
    if trend_4h == "bajista" and direction == "bajista":
        return "a_favor", trend_4h
    if trend_4h in {"alcista", "bajista"}:
        return "en_contra", trend_4h
    return "sin_contexto", trend_4h


def _nights_held(entry_time: pd.Timestamp, exit_time: pd.Timestamp) -> int:
    entry_date = pd.Timestamp(entry_time).normalize()
    exit_date = pd.Timestamp(exit_time).normalize()
    delta_days = (exit_date - entry_date).days
    return max(delta_days, 0)


def _segment_ranges(index: pd.Index) -> dict[str, tuple[int, int]]:
    total = len(index)
    if total <= 0:
        return {
            SEGMENT_ALL: (0, -1),
            SEGMENT_DEVELOPMENT: (0, -1),
            SEGMENT_VALIDATION: (0, -1),
            SEGMENT_TEST: (0, -1),
        }

    dev_end_exclusive = max(1, int(total * 0.6))
    val_end_exclusive = max(dev_end_exclusive + 1, int(total * 0.8))
    val_end_exclusive = min(val_end_exclusive, total)

    return {
        SEGMENT_ALL: (0, total - 1),
        SEGMENT_DEVELOPMENT: (0, dev_end_exclusive - 1),
        SEGMENT_VALIDATION: (dev_end_exclusive, val_end_exclusive - 1),
        SEGMENT_TEST: (val_end_exclusive, total - 1),
    }


def _prepare_execution_window(
    frame: pd.DataFrame,
    config: BacktestConfig,
) -> tuple[pd.DataFrame, int, int, dict[str, Any]]:
    if frame.empty:
        return frame.copy(), 0, -1, {}

    start_bound = pd.Timestamp(config.start_date) if config.start_date else pd.Timestamp(frame.index[0])
    end_bound = (
        pd.Timestamp(config.end_date) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
        if config.end_date
        else pd.Timestamp(frame.index[-1])
    )

    eligible_mask = (frame.index >= start_bound) & (frame.index <= end_bound)
    if not eligible_mask.any():
        return frame.iloc[0:0].copy(), 0, -1, {}

    eligible_index = frame.index[eligible_mask]
    segment_map = _segment_ranges(eligible_index)
    segment_start_rel, segment_end_rel = segment_map.get(config.data_segment, segment_map[SEGMENT_ALL])
    if segment_start_rel > segment_end_rel:
        return frame.iloc[0:0].copy(), 0, -1, {}

    segment_start_label = eligible_index[segment_start_rel]
    segment_end_label = eligible_index[segment_end_rel]
    absolute_start_pos = int(frame.index.get_indexer([segment_start_label])[0])
    absolute_end_pos = int(frame.index.get_indexer([segment_end_label])[0])

    working_start_pos = max(0, absolute_start_pos - int(config.warmup_bars))
    working_frame = frame.iloc[working_start_pos : absolute_end_pos + 1].copy()
    loop_start_pos = absolute_start_pos - working_start_pos
    loop_end_pos = absolute_end_pos - working_start_pos

    segment_info = {
        "segment": config.data_segment,
        "requested_start": start_bound.strftime("%Y-%m-%d"),
        "requested_end": end_bound.strftime("%Y-%m-%d"),
        "effective_start": pd.Timestamp(segment_start_label).strftime("%Y-%m-%d %H:%M"),
        "effective_end": pd.Timestamp(segment_end_label).strftime("%Y-%m-%d %H:%M"),
    }
    return working_frame, loop_start_pos, loop_end_pos, segment_info


def _download_intrabar_slice(
    asset: str,
    interval: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    metadata = get_asset_metadata(asset)
    ticker = str(metadata.get("ticker", ""))
    if not ticker:
        raise DataDownloadError(f"No se encontró ticker para {asset}.")

    frame = download_ohlc_range(ticker=ticker, start=start, end=end, interval=interval)
    filtered = frame[(frame.index >= start) & (frame.index < end)].copy()
    return filtered


def _intrabar_lookup_is_available(
    interval: str,
    bar_start: pd.Timestamp,
) -> bool:
    max_lookback = INTRABAR_HISTORY_LIMITS.get(interval)
    if max_lookback is None:
        return True

    reference_now = pd.Timestamp.utcnow().tz_localize(None)
    return bar_start >= (reference_now - max_lookback)


def _resolve_intrabar_conflict(
    *,
    asset: str,
    timeframe: str,
    bar_start: pd.Timestamp,
    direction_multiplier: int,
    stop: float,
    target: float,
    config: BacktestConfig,
) -> tuple[float, str, str]:
    if not config.allow_intrabar_lookup:
        return stop, "stop_loss", "conflicto_conservador_sin_lookup"

    bar_end = bar_start + _expected_delta(timeframe)
    for interval in config.lower_timeframe_priority:
        if not _intrabar_lookup_is_available(interval, bar_start):
            continue
        try:
            intrabar = _download_intrabar_slice(asset, interval, bar_start, bar_end)
        except DataDownloadError:
            continue

        if intrabar.empty:
            continue

        for _, row in intrabar.iterrows():
            high = _safe_float(row["high"])
            low = _safe_float(row["low"])
            if direction_multiplier == 1:
                hit_stop = low <= stop
                hit_target = high >= target
            else:
                hit_stop = high >= stop
                hit_target = low <= target

            if hit_stop and hit_target:
                break
            if hit_stop:
                return stop, "stop_loss", f"intrabar_{interval}"
            if hit_target:
                return target, "take_profit", f"intrabar_{interval}"

    return stop, "stop_loss", "conflicto_conservador"


def _build_trade_record(
    *,
    frame: pd.DataFrame,
    asset: str,
    timeframe: str,
    signal_position: int,
    signal: dict[str, Any],
    config: BacktestConfig,
    context_lookup: pd.DataFrame,
) -> dict[str, Any] | None:
    entry_position = signal_position + 1
    if entry_position >= len(frame):
        return None

    raw_entry = _safe_float(frame.iloc[entry_position]["open"])
    stop = _safe_float(signal.get("stop_loss"))
    target = _safe_float(signal.get("take_profit"))
    if raw_entry <= 0 or stop <= 0 or target <= 0:
        return None

    risk_points = abs(raw_entry - stop)
    if risk_points <= 0:
        return None

    direction_multiplier = 1 if str(signal.get("direccion")) == "alcista" else -1
    planned_end_position = min(entry_position + int(config.max_holding_bars) - 1, len(frame) - 1)
    future = frame.iloc[entry_position : planned_end_position + 1]
    if future.empty:
        return None

    gross_exit = _safe_float(future["close"].iloc[-1], default=raw_entry)
    exit_time = pd.Timestamp(future.index[-1])
    exit_position = int(frame.index.get_indexer([exit_time])[0])
    exit_reason = "fin_de_ventana"
    exit_source = "cierre_ultima_vela"

    mae_points = 0.0
    mfe_points = 0.0

    for timestamp, row in future.iterrows():
        high = _safe_float(row["high"])
        low = _safe_float(row["low"])
        if direction_multiplier == 1:
            mae_points = max(mae_points, max(raw_entry - low, 0.0))
            mfe_points = max(mfe_points, max(high - raw_entry, 0.0))
            hit_stop = low <= stop
            hit_target = high >= target
        else:
            mae_points = max(mae_points, max(high - raw_entry, 0.0))
            mfe_points = max(mfe_points, max(raw_entry - low, 0.0))
            hit_stop = high >= stop
            hit_target = low <= target

        if hit_stop and hit_target:
            gross_exit, exit_reason, exit_source = _resolve_intrabar_conflict(
                asset=asset,
                timeframe=timeframe,
                bar_start=pd.Timestamp(timestamp),
                direction_multiplier=direction_multiplier,
                stop=stop,
                target=target,
                config=config,
            )
            exit_time = pd.Timestamp(timestamp)
            exit_position = int(frame.index.get_indexer([timestamp])[0])
            break
        if hit_stop:
            gross_exit = stop
            exit_time = pd.Timestamp(timestamp)
            exit_position = int(frame.index.get_indexer([timestamp])[0])
            exit_reason = "stop_loss"
            exit_source = "vela_principal"
            break
        if hit_target:
            gross_exit = target
            exit_time = pd.Timestamp(timestamp)
            exit_position = int(frame.index.get_indexer([timestamp])[0])
            exit_reason = "take_profit"
            exit_source = "vela_principal"
            break

    gross_points = direction_multiplier * (gross_exit - raw_entry)
    gross_r = gross_points / risk_points

    point_size = get_cost_point_size(asset)
    cost_unit_label = get_cost_unit_label(asset)
    cost_unit_code = "points"
    nights = _nights_held(pd.Timestamp(frame.index[entry_position]), exit_time)
    spread_cost = float(config.costs.spread) * point_size
    slippage_cost = float(config.costs.slippage) * point_size
    commission_cost = float(config.costs.commission_in_price_units(raw_entry, point_size))
    swap_cost = float(config.costs.swap_per_night) * nights * point_size
    total_cost_points = spread_cost + slippage_cost + commission_cost + swap_cost
    total_cost_r = total_cost_points / risk_points
    net_r = (gross_points - total_cost_points) / risk_points

    context_label, trend_4h = _resolve_context_for_signal(
        context_lookup=context_lookup,
        signal_time=pd.Timestamp(frame.index[signal_position]),
        direction=str(signal.get("direccion")),
    )

    exit_family = "tiempo"
    if exit_reason.startswith("take_profit"):
        exit_family = "tp"
    elif exit_reason.startswith("stop_loss"):
        exit_family = "sl"

    outcome = "neutral"
    if net_r > 0:
        outcome = "ganadora"
    elif net_r < 0:
        outcome = "perdedora"

    return {
        "fecha_senal": pd.Timestamp(frame.index[signal_position]),
        "fecha_entrada": pd.Timestamp(frame.index[entry_position]),
        "fecha_salida": exit_time,
        "activo": asset,
        "temporalidad": timeframe,
        "patron": str(signal.get("patron")),
        "direccion": "COMPRA" if direction_multiplier == 1 else "VENTA",
        "contexto_4h": context_label,
        "tendencia_4h": trend_4h,
        "entrada_sugerida": _safe_float(signal.get("precio_entrada")),
        "entrada": raw_entry,
        "stop_loss": stop,
        "take_profit": target,
        "salida": gross_exit,
        "tipo_salida": exit_reason,
        "familia_salida": exit_family,
        "fuente_salida": exit_source,
        "resultado": outcome,
        "resultado_R": round(gross_r, 4),
        "resultado_neto_R": round(net_r, 4),
        "duracion_velas": int(exit_position - entry_position + 1),
        "risk_price": round(risk_points, 8),
        "MAE": round(mae_points / risk_points, 4),
        "MFE": round(mfe_points / risk_points, 4),
        "unidad_coste": cost_unit_label,
        "tamano_unidad_coste": point_size,
        "spread_input": round(float(config.costs.spread), 4),
        "spread_unit": cost_unit_code,
        "spread_price": round(spread_cost, 8),
        "slippage_input": round(float(config.costs.slippage), 4),
        "slippage_unit": cost_unit_code,
        "slippage_price": round(slippage_cost, 8),
        "comision_input": round(float(config.costs.commission), 4),
        "comision_price": round(commission_cost, 8),
        "swap_input": round(float(config.costs.swap_per_night), 4),
        "swap_price": round(swap_cost, 8),
        "total_cost_price": round(total_cost_points, 8),
        "cost_R": round(total_cost_r, 4),
        "spread_coste": round(spread_cost / risk_points, 4),
        "slippage_coste": round(slippage_cost / risk_points, 4),
        "comision_coste": round(commission_cost / risk_points, 4),
        "swap_coste": round(swap_cost / risk_points, 4),
        "coste_total_R": round(total_cost_r, 4),
        "noches_swap": int(nights),
        "exit_position": int(exit_position),
        "signal_position": int(signal_position),
        "rr_estimado": round(_safe_float(signal.get("relacion_riesgo_beneficio")), 2),
    }


def _collect_blocked_signals(
    frame: pd.DataFrame,
    asset: str,
    timeframe: str,
    start_signal_position: int,
    end_signal_position: int,
) -> list[dict[str, Any]]:
    blocked: list[dict[str, Any]] = []
    if end_signal_position < start_signal_position:
        return blocked

    for signal_position in range(start_signal_position, end_signal_position + 1):
        if signal_position >= len(frame) - 1:
            break
        window_start = max(0, signal_position - PATTERN_LOOKBACK_BARS + 1)
        window = frame.iloc[window_start : signal_position + 1]
        signals = detect_patterns(window, asset=asset, timeframe=timeframe)
        for signal in signals:
            blocked.append(
                {
                    "fecha_senal": pd.Timestamp(frame.index[signal_position]),
                    "activo": asset,
                    "temporalidad": timeframe,
                    "patron": str(signal.get("patron")),
                    "direccion": "COMPRA" if signal.get("direccion") == "alcista" else "VENTA",
                    "motivo": "operacion_activa",
                }
            )
    return blocked


def _longest_losing_streak(values: pd.Series) -> int:
    streak = 0
    max_streak = 0
    for value in values.tolist():
        if float(value) < 0:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0
    return max_streak


def _build_breakdown(
    trades: pd.DataFrame,
    group_column: str,
    value_column: str = "resultado_neto_R",
) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()

    grouped = trades.groupby(group_column, dropna=False)
    rows: list[dict[str, Any]] = []
    for key, group in grouped:
        wins = group[group[value_column] > 0]
        rows.append(
            {
                group_column: key,
                "operaciones": int(len(group)),
                "ganancia_neta_R": round(float(group[value_column].sum()), 4),
                "expectativa": round(float(group[value_column].mean()), 4),
                "win_rate": round((len(wins) / len(group)) * 100, 2),
            }
        )
    return pd.DataFrame(rows)


def _build_monthly_breakdown(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()

    monthly = trades.copy()
    monthly["periodo"] = pd.to_datetime(monthly["fecha_salida"]).dt.to_period("M").astype(str)
    result = (
        monthly.groupby("periodo", sort=True)["resultado_neto_R"]
        .sum()
        .reset_index(name="ganancia_neta_R")
    )
    return result


def _empty_statistics(blocked_signals: int = 0) -> dict[str, Any]:
    return {
        "numero_senales": 0,
        "operaciones_ganadoras": 0,
        "operaciones_perdedoras": 0,
        "ganancia_neta_R": 0.0,
        "expectativa_bruta": 0.0,
        "expectativa_neta": 0.0,
        "profit_factor": 0.0,
        "win_rate": 0.0,
        "promedio_ganador_R": 0.0,
        "promedio_perdedor_R": 0.0,
        "duracion_media_velas": 0.0,
        "salidas_por_tp": {"count": 0, "pct": 0.0},
        "salidas_por_sl": {"count": 0, "pct": 0.0},
        "salidas_por_tiempo": {"count": 0, "pct": 0.0},
        "max_drawdown_R": 0.0,
        "max_racha_perdedora": 0,
        "MAE_promedio": 0.0,
        "MFE_promedio": 0.0,
        "senales_bloqueadas": int(blocked_signals),
        "expectativa_matematica": 0.0,
        "drawdown_aproximado": 0.0,
    }


def _summarize_statistics(trades: pd.DataFrame, blocked_signals: int) -> dict[str, Any]:
    if trades.empty:
        return _empty_statistics(blocked_signals)

    winners = trades[trades["resultado_neto_R"] > 0]
    losers = trades[trades["resultado_neto_R"] < 0]
    equity_curve = trades["resultado_neto_R"].cumsum()
    drawdown = equity_curve - equity_curve.cummax()

    gross_profit = float(winners["resultado_neto_R"].sum())
    gross_loss = abs(float(losers["resultado_neto_R"].sum()))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0

    total = len(trades)
    tp_count = int((trades["familia_salida"] == "tp").sum())
    sl_count = int((trades["familia_salida"] == "sl").sum())
    time_count = int((trades["familia_salida"] == "tiempo").sum())

    stats = {
        "numero_senales": int(total),
        "operaciones_ganadoras": int(len(winners)),
        "operaciones_perdedoras": int(len(losers)),
        "ganancia_neta_R": round(float(trades["resultado_neto_R"].sum()), 4),
        "expectativa_bruta": round(float(trades["resultado_R"].mean()), 4),
        "expectativa_neta": round(float(trades["resultado_neto_R"].mean()), 4),
        "profit_factor": round(float(profit_factor), 4),
        "win_rate": round((len(winners) / total) * 100, 2),
        "promedio_ganador_R": round(float(winners["resultado_neto_R"].mean()) if not winners.empty else 0.0, 4),
        "promedio_perdedor_R": round(float(losers["resultado_neto_R"].mean()) if not losers.empty else 0.0, 4),
        "duracion_media_velas": round(float(trades["duracion_velas"].mean()), 2),
        "salidas_por_tp": {"count": tp_count, "pct": round((tp_count / total) * 100, 2)},
        "salidas_por_sl": {"count": sl_count, "pct": round((sl_count / total) * 100, 2)},
        "salidas_por_tiempo": {"count": time_count, "pct": round((time_count / total) * 100, 2)},
        "max_drawdown_R": round(abs(float(drawdown.min())) if not drawdown.empty else 0.0, 4),
        "max_racha_perdedora": int(_longest_losing_streak(trades["resultado_neto_R"])),
        "MAE_promedio": round(float(trades["MAE"].mean()), 4),
        "MFE_promedio": round(float(trades["MFE"].mean()), 4),
        "senales_bloqueadas": int(blocked_signals),
        "expectativa_matematica": round(float(trades["resultado_neto_R"].mean()), 4),
        "drawdown_aproximado": round(abs(float(drawdown.min())) if not drawdown.empty else 0.0, 4),
    }
    return stats


def _finalize_result(
    *,
    asset: str,
    timeframe: str,
    config: BacktestConfig,
    trades: list[dict[str, Any]],
    blocked_signals: list[dict[str, Any]],
    data_quality_log: pd.DataFrame,
    segment_info: dict[str, Any],
) -> dict[str, Any]:
    trades_frame = pd.DataFrame(trades)
    blocked_frame = pd.DataFrame(blocked_signals)

    if not trades_frame.empty:
        trades_frame = trades_frame.sort_values("fecha_entrada").reset_index(drop=True)
        trades_frame["equity_curve_bruta_R"] = trades_frame["resultado_R"].cumsum()
        trades_frame["equity_curve_neta_R"] = trades_frame["resultado_neto_R"].cumsum()
        trades_frame["fecha_entrada"] = pd.to_datetime(trades_frame["fecha_entrada"])
        trades_frame["fecha_salida"] = pd.to_datetime(trades_frame["fecha_salida"])
        trades_frame["fecha_senal"] = pd.to_datetime(trades_frame["fecha_senal"])

    statistics = _summarize_statistics(trades_frame, blocked_signals=len(blocked_signals))
    result_by_month = _build_monthly_breakdown(trades_frame)
    result_by_pattern = _build_breakdown(trades_frame, "patron")
    result_by_direction = _build_breakdown(trades_frame, "direccion")
    result_by_context = _build_breakdown(trades_frame, "contexto_4h")

    execution = {
        "nombre_configuracion": config.configuration_name,
        "fecha_ejecucion": pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        "asset": asset,
        "timeframe": timeframe,
        "parametros_usados": config.parameter_snapshot(asset=asset, timeframe=timeframe),
        "parameter_hash": config.parameter_hash(asset=asset, timeframe=timeframe),
        "segment_info": segment_info,
    }

    export_frame = trades_frame.copy()
    if not export_frame.empty:
        export_frame = export_frame.rename(
            columns={
                "resultado_R": "resultado_R",
                "resultado_neto_R": "resultado_neto_R",
                "tipo_salida": "tipo_salida",
                "duracion_velas": "duracion_velas",
                "contexto_4h": "contexto_4h",
                "MAE": "MAE",
                "MFE": "MFE",
            }
        )

    return {
        "statistics": statistics,
        "trades": trades_frame.drop(columns=["exit_position", "signal_position"], errors="ignore"),
        "export_trades": export_frame.drop(columns=["exit_position", "signal_position"], errors="ignore"),
        "data_quality_log": data_quality_log,
        "blocked_signals": blocked_frame,
        "result_by_month": result_by_month,
        "result_by_pattern": result_by_pattern,
        "result_by_direction": result_by_direction,
        "result_by_context_4h": result_by_context,
        "execution": execution,
    }


def backtest_patterns(
    frame: pd.DataFrame,
    asset: str,
    timeframe: str,
    *,
    config: BacktestConfig,
    context_frame: pd.DataFrame | None = None,
) -> dict[str, Any]:
    validated_frame, frame_issues = validate_market_data(
        frame=frame,
        timeframe=timeframe,
        gap_threshold_pct=config.gap_threshold_pct,
        label=f"Datos {asset} {timeframe}",
    )

    context_issues = pd.DataFrame()
    if context_frame is not None and not context_frame.empty:
        validated_context, context_issues = validate_market_data(
            frame=context_frame,
            timeframe="4h",
            gap_threshold_pct=config.gap_threshold_pct,
            label=f"Contexto 4H {asset}",
        )
    else:
        validated_context = pd.DataFrame()

    combined_issues = pd.concat([frame_issues, context_issues], ignore_index=True)

    if validated_frame.empty:
        return _finalize_result(
            asset=asset,
            timeframe=timeframe,
            config=config,
            trades=[],
            blocked_signals=[],
            data_quality_log=combined_issues,
            segment_info={},
        )

    enriched = add_indicators(validated_frame.copy())
    working_frame, loop_start_pos, loop_end_pos, segment_info = _prepare_execution_window(
        enriched,
        config,
    )
    if working_frame.empty or loop_end_pos <= loop_start_pos or len(working_frame) <= 1:
        return _finalize_result(
            asset=asset,
            timeframe=timeframe,
            config=config,
            trades=[],
            blocked_signals=[],
            data_quality_log=combined_issues,
            segment_info=segment_info,
        )

    working_context = validated_context.copy() if not validated_context.empty else pd.DataFrame()
    context_lookup = _build_context_lookup(working_context)

    trades: list[dict[str, Any]] = []
    blocked_signals: list[dict[str, Any]] = []

    signal_position = max(loop_start_pos, int(config.warmup_bars))
    while signal_position <= min(loop_end_pos, len(working_frame) - 2):
        window_start = max(0, signal_position - PATTERN_LOOKBACK_BARS + 1)
        window = working_frame.iloc[window_start : signal_position + 1]
        signals = detect_patterns(window, asset=asset, timeframe=timeframe)

        if not signals:
            signal_position += 1
            continue

        if config.mode == MODE_DIAGNOSTICO:
            for signal in signals:
                trade = _build_trade_record(
                    frame=working_frame,
                    asset=asset,
                    timeframe=timeframe,
                    signal_position=signal_position,
                    signal=signal,
                    config=config,
                    context_lookup=context_lookup,
                )
                if trade is not None:
                    trades.append(trade)
            signal_position += 1
            continue

        primary_signal = signals[0]
        if len(signals) > 1:
            for extra_signal in signals[1:]:
                blocked_signals.append(
                    {
                        "fecha_senal": pd.Timestamp(working_frame.index[signal_position]),
                        "activo": asset,
                        "temporalidad": timeframe,
                        "patron": str(extra_signal.get("patron")),
                        "direccion": "COMPRA" if extra_signal.get("direccion") == "alcista" else "VENTA",
                        "motivo": "multiple_signal_same_bar",
                    }
                )

        trade = _build_trade_record(
            frame=working_frame,
            asset=asset,
            timeframe=timeframe,
            signal_position=signal_position,
            signal=primary_signal,
            config=config,
            context_lookup=context_lookup,
        )
        if trade is None:
            signal_position += 1
            continue

        trades.append(trade)
        blocked_signals.extend(
            _collect_blocked_signals(
                frame=working_frame,
                asset=asset,
                timeframe=timeframe,
                start_signal_position=signal_position + 1,
                end_signal_position=max(trade["exit_position"] - 1, signal_position),
            )
        )
        signal_position = max(trade["exit_position"], signal_position + 1)

    return _finalize_result(
        asset=asset,
        timeframe=timeframe,
        config=config,
        trades=trades,
        blocked_signals=blocked_signals,
        data_quality_log=combined_issues,
        segment_info=segment_info,
    )
