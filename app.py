from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from core.analysis_engine import (
    analizar_activo,
    asset_has_required_data,
    build_asset_analyses,
    build_asset_snapshot,
    build_scanner_table,
    format_price,
)
from app_settings import (
    REFRESH_OPTIONS,
    format_refresh_timestamp,
    get_refresh_interval_minutes,
    get_refresh_label,
    load_app_settings,
    now_storage_timestamp,
    parse_refresh_timestamp,
    save_app_settings,
)
from backtester import (
    MODE_DIAGNOSTICO,
    MODE_OPERATIVO,
    SEGMENT_ALL,
    SEGMENT_DEVELOPMENT,
    SEGMENT_TEST,
    SEGMENT_VALIDATION,
    BacktestConfig,
    TradingCosts,
    backtest_patterns,
)
from data_loader import (
    ASSETS,
    DEFAULT_ACTIVE_ASSETS,
    get_cost_point_size,
    get_cost_unit_label,
    refresh_market_data,
    summarize_download_results,
)
from database import count_market_rows, init_database, load_backtest_runs, save_backtest_run
from journal import JournalEntry, get_journal, record_trade
from risk import EDUCATIONAL_WARNING, calculate_risk
from ui_components import (
    build_price_chart,
    inject_global_styles,
    render_decision_card,
    render_multi_timeframe_card,
    render_time_horizon_card,
    render_watchlist_card,
    show_download_status,
)


st.set_page_config(page_title="Laboratorio de Análisis Técnico", layout="wide")
inject_global_styles()


PAGE_OPTIONS = [
    "Dashboard",
    "Escáner de activos",
    "Calculadora de riesgo",
    "Bitácora",
    "Backtesting",
    "Configuración",
]
AUTO_REFRESH_PAGES = {"Dashboard", "Escáner de activos"}


def _all_asset_names() -> list[str]:
    return list(ASSETS.keys())


def _active_asset_mapping(settings: dict[str, Any]) -> dict[str, str]:
    active_assets = settings.get("active_assets", DEFAULT_ACTIVE_ASSETS)
    return {asset: ASSETS[asset] for asset in active_assets if asset in ASSETS}


def _is_asset_active(settings: dict[str, Any], asset: str) -> bool:
    return asset in settings.get("active_assets", DEFAULT_ACTIVE_ASSETS)


def _update_refresh_feedback(kind: str, text: str) -> None:
    st.session_state["refresh_feedback"] = {"kind": kind, "text": text}


def _run_market_refresh(settings: dict[str, Any], trigger_label: str) -> dict[str, Any]:
    asset_mapping = _active_asset_mapping(settings)
    if not asset_mapping:
        _update_refresh_feedback("error", "No hay activos activos para actualizar.")
        return settings

    with st.spinner(f"Actualizando datos de mercado ({trigger_label})..."):
        results = refresh_market_data(assets=asset_mapping)

    results_frame = summarize_download_results(results)
    st.session_state["last_refresh_results"] = results_frame

    updated_settings = dict(settings)
    updated_settings["last_refresh_attempt_at"] = now_storage_timestamp()

    has_success = not results_frame.empty and (results_frame["estado"] == "ok").any()
    has_error = not results_frame.empty and (results_frame["estado"] == "error").any()

    if has_success:
        updated_settings["last_refresh_at"] = updated_settings["last_refresh_attempt_at"]
        save_app_settings(updated_settings)
        if has_error:
            _update_refresh_feedback(
                "warning",
                "La actualización terminó con errores parciales. Se conservan los últimos datos válidos.",
            )
        else:
            _update_refresh_feedback("success", "Datos actualizados correctamente.")
        return updated_settings

    save_app_settings(updated_settings)
    _update_refresh_feedback(
        "error",
        "La descarga falló. Se mantienen los últimos datos válidos guardados.",
    )
    return updated_settings


def _show_refresh_feedback() -> None:
    feedback = st.session_state.get("refresh_feedback")
    if not feedback:
        return

    kind = str(feedback.get("kind", "info"))
    text = str(feedback.get("text", ""))
    if kind == "success":
        st.success(text)
    elif kind == "warning":
        st.warning(text)
    elif kind == "error":
        st.error(text)
    else:
        st.info(text)


def _activate_asset(settings: dict[str, Any], asset: str) -> dict[str, Any]:
    updated_assets = list(settings.get("active_assets", DEFAULT_ACTIVE_ASSETS))
    if asset not in updated_assets:
        updated_assets.append(asset)
    ordered_assets = [asset_name for asset_name in _all_asset_names() if asset_name in updated_assets]
    updated_settings = save_app_settings(
        {
            **settings,
            "active_assets": ordered_assets,
        }
    )
    st.session_state["ui_asset"] = asset
    return updated_settings


def _render_asset_availability_actions(
    settings: dict[str, Any],
    selected_asset: str,
    has_data: bool,
    *,
    button_scope: str,
) -> None:
    asset_is_active = _is_asset_active(settings, selected_asset)

    if asset_is_active and has_data:
        return

    if not asset_is_active:
        st.info(
            f"{selected_asset} está disponible en el catálogo, pero ahora mismo está inactivo para el escáner y la actualización automática."
        )
        if st.button(
            f"Activar {selected_asset} y descargar datos",
            key=f"activate_{button_scope}_{selected_asset}",
            use_container_width=True,
        ):
            updated_settings = _activate_asset(settings, selected_asset)
            _run_market_refresh(updated_settings, f"activación de {selected_asset}")
            st.rerun()
        if not has_data:
            st.warning(
                "Todavía no hay datos descargados para este activo. Actívalo y actualiza para analizarlo."
            )
        return

    if not has_data:
        st.warning(
            f"No hay datos locales disponibles para {selected_asset}. Puedes actualizar los activos activos para descargar su histórico."
        )
        if st.button(
            "Actualizar datos de activos activos",
            key=f"refresh_{button_scope}_{selected_asset}",
            use_container_width=True,
        ):
            _run_market_refresh(settings, f"recarga de {selected_asset}")
            st.rerun()


def _refresh_reference_timestamp(settings: dict[str, Any]) -> datetime | None:
    return parse_refresh_timestamp(
        settings.get("last_refresh_attempt_at") or settings.get("last_refresh_at")
    )


def _next_refresh_timestamp(settings: dict[str, Any]) -> datetime | None:
    interval_minutes = get_refresh_interval_minutes(settings)
    reference = _refresh_reference_timestamp(settings)
    if interval_minutes is None or reference is None:
        return None
    return reference + timedelta(minutes=interval_minutes)


def _bootstrap_data(settings: dict[str, Any]) -> dict[str, Any]:
    init_database()
    if st.session_state.get("bootstrap_done"):
        return settings

    st.session_state["bootstrap_done"] = True
    active_assets = settings.get("active_assets", DEFAULT_ACTIVE_ASSETS)
    requires_seed = count_market_rows() == 0
    if not requires_seed:
        requires_seed = any(not asset_has_required_data(asset) for asset in active_assets)

    if requires_seed:
        return _run_market_refresh(settings, "inicial")
    return settings


def _handle_auto_refresh(current_page: str, settings: dict[str, Any]) -> dict[str, Any]:
    interval_minutes = get_refresh_interval_minutes(settings)
    if interval_minutes is None or current_page not in AUTO_REFRESH_PAGES:
        return settings

    interval_ms = interval_minutes * 60 * 1000
    components.html(
        f"""
        <script>
            window.setTimeout(function() {{
                window.parent.location.reload();
            }}, {interval_ms});
        </script>
        """,
        height=0,
    )

    next_refresh = _next_refresh_timestamp(settings)
    if next_refresh is None or datetime.now() < next_refresh:
        return settings

    refreshed_settings = _run_market_refresh(settings, "automática")
    st.rerun()
    return refreshed_settings


def _trade_defaults(snapshot: dict[str, Any]) -> dict[str, float | str]:
    candidate_analyses = [
        snapshot.get("selected_analysis", {}),
        snapshot.get("entry_analysis", {}),
        snapshot.get("context_analysis", {}),
    ]

    signal: dict[str, Any] | None = None
    latest_row = None
    direction = str(snapshot.get("operative_decision", {}).get("direccion", "NINGUNA"))

    for analysis in candidate_analyses:
        signals = analysis.get("signals", []) or []
        data = analysis.get("data", pd.DataFrame())
        if signal is None and signals:
            signal = signals[0]
        if latest_row is None and not data.empty:
            latest_row = data.iloc[-1]

    if latest_row is None:
        return {
            "entry": 0.0,
            "stop": 0.0,
            "take_profit": 0.0,
            "pattern": "",
            "direction": direction,
            "reason": "",
        }

    close_price = float(latest_row["close"])
    atr_value = float(latest_row["atr_14"]) if pd.notna(latest_row["atr_14"]) else max(close_price * 0.002, 0.0001)

    if signal is not None:
        return {
            "entry": float(signal["precio_entrada"]),
            "stop": float(signal["stop_loss"]),
            "take_profit": float(signal["take_profit"]),
            "pattern": str(signal["patron"]),
            "direction": direction,
            "reason": str(signal.get("explicacion", "")),
        }

    if direction == "VENTA":
        stop_loss = close_price + atr_value
        take_profit = close_price - 2 * abs(close_price - stop_loss)
    else:
        stop_loss = close_price - atr_value
        take_profit = close_price + 2 * abs(close_price - stop_loss)

    return {
        "entry": close_price,
        "stop": stop_loss,
        "take_profit": take_profit,
        "pattern": "",
        "direction": direction,
        "reason": "",
    }


def _render_dashboard(
    settings: dict[str, Any],
    selected_asset: str,
    selected_timeframe: str,
    selected_chart: str,
) -> None:
    analysis_result = analizar_activo(nombre=selected_asset, timeframe=selected_timeframe)
    snapshot = analysis_result["snapshot"]
    selected_analysis = snapshot["selected_analysis"]
    data = selected_analysis.get("data", pd.DataFrame())

    _render_asset_availability_actions(
        settings,
        selected_asset,
        has_data=not data.empty,
        button_scope="dashboard",
    )
    if data.empty:
        st.error(
            "No hay datos disponibles para este activo y temporalidad. Actualiza o activa el activo para descargar su histórico."
        )
        return

    latest = data.iloc[-1]
    latest_price = float(latest["close"])
    latest_atr = (
        float(latest["atr_14"])
        if pd.notna(latest["atr_14"]) and latest["atr_14"] > 0
        else max(latest_price * 0.002, 0.0001)
    )

    metric_1, metric_2, metric_3, metric_4, metric_5, metric_6 = st.columns(6)
    metric_1.metric("Activo", selected_asset)
    metric_2.metric("Temporalidad", selected_timeframe.upper())
    metric_3.metric("Precio actual", str(analysis_result["precio_texto"]))
    metric_4.metric("Tendencia", str(analysis_result["tendencia"]))
    metric_5.metric("RSI 14", f"{latest['rsi_14']:.2f}")
    metric_6.metric("ATR 14", format_price(float(latest_atr)))

    render_multi_timeframe_card(snapshot["multi_timeframe"])
    render_time_horizon_card(snapshot["time_horizon"])
    render_decision_card(snapshot["operative_decision"], snapshot["multi_timeframe"])
    render_watchlist_card(snapshot["watchlist"], snapshot["operative_decision"])

    market_structure = selected_analysis["market_structure"]
    chart_figure = build_price_chart(
        frame=data.tail(250),
        chart_type=selected_chart,
        asset=selected_asset,
        timeframe=selected_timeframe,
        supports=market_structure["supports"],
        resistances=market_structure["resistances"],
    )
    st.plotly_chart(chart_figure, use_container_width=True)

    left_col, right_col = st.columns([1.25, 1])
    with left_col:
        st.subheader("Estructura de mercado")
        st.write(f"Tendencia por estructura: **{market_structure['trend']}**")
        st.write(
            "Soportes aproximados: "
            + (", ".join(str(level) for level in market_structure["supports"]) or "sin niveles claros")
        )
        st.write(
            "Resistencias aproximadas: "
            + (", ".join(str(level) for level in market_structure["resistances"]) or "sin niveles claros")
        )
        st.markdown("**Máximos relevantes**")
        st.dataframe(
            pd.DataFrame(market_structure["relevant_highs"]),
            use_container_width=True,
            hide_index=True,
        )
        st.markdown("**Mínimos relevantes**")
        st.dataframe(
            pd.DataFrame(market_structure["relevant_lows"]),
            use_container_width=True,
            hide_index=True,
        )

    with right_col:
        st.subheader("Patrones detectados")
        signals = selected_analysis["signals"]
        if signals:
            patterns_frame = pd.DataFrame(signals).drop(columns=["signal_index"])
            st.dataframe(patterns_frame, use_container_width=True, hide_index=True)
        else:
            st.info("No se detectaron patrones simples en la última vela analizada.")


def _render_scanner(settings: dict[str, Any]) -> None:
    scanner_frame, _ = build_scanner_table(settings["active_assets"])
    if scanner_frame.empty:
        st.warning("No hay activos activos con datos disponibles para el escáner.")
        inactive_assets = [asset for asset in _all_asset_names() if asset not in settings["active_assets"]]
        if inactive_assets:
            st.info("Activos disponibles para activar en Configuración: " + ", ".join(inactive_assets))
        return

    possible_count = int((scanner_frame["estado semáforo"] == "POSIBLE OPERACIÓN").sum())
    watch_count = int((scanner_frame["estado semáforo"] == "VIGILAR").sum())
    no_trade_count = int((scanner_frame["estado semáforo"] == "NO OPERAR").sum())

    metric_1, metric_2, metric_3, metric_4 = st.columns(4)
    metric_1.metric("Activos analizados", str(len(scanner_frame)))
    metric_2.metric("Posibles operaciones", str(possible_count))
    metric_3.metric("En vigilancia", str(watch_count))
    metric_4.metric("No operar", str(no_trade_count))

    inactive_assets = [asset for asset in _all_asset_names() if asset not in settings["active_assets"]]
    if inactive_assets:
        st.caption("Disponibles para activar y agregar al escáner: " + ", ".join(inactive_assets))

    st.dataframe(scanner_frame, use_container_width=True, hide_index=True)


def _render_risk_calculator(
    settings: dict[str, Any],
    selected_asset: str,
    selected_timeframe: str,
) -> None:
    analyses = build_asset_analyses(selected_asset)
    snapshot = build_asset_snapshot(selected_asset, analyses, selected_timeframe)
    defaults = _trade_defaults(snapshot)
    has_data = bool(defaults["entry"] > 0)

    st.subheader("Calculadora de riesgo educativa")
    st.caption(
        "Esta página no envía órdenes reales. Solo estima riesgo monetario, tamaño de posición y relación riesgo/beneficio."
    )
    _render_asset_availability_actions(
        settings,
        selected_asset,
        has_data=has_data,
        button_scope="risk",
    )

    col_1, col_2, col_3 = st.columns(3)
    col_1.text_input("Activo", value=selected_asset, disabled=True)
    capital = col_2.number_input(
        "Capital",
        min_value=1.0,
        value=float(settings["reference_capital"]),
        step=100.0,
    )
    risk_percent = col_3.number_input(
        "Riesgo %",
        min_value=0.1,
        value=float(settings["risk_per_trade"]),
        step=0.1,
    )

    col_4, col_5, col_6 = st.columns(3)
    entry_price = col_4.number_input(
        "Entrada",
        min_value=0.0,
        value=float(defaults["entry"]),
        format="%.5f",
    )
    stop_loss = col_5.number_input(
        "Stop loss",
        min_value=0.0,
        value=float(defaults["stop"]),
        format="%.5f",
    )
    take_profit = col_6.number_input(
        "Take profit",
        min_value=0.0,
        value=float(defaults["take_profit"]),
        format="%.5f",
    )

    try:
        risk_result = calculate_risk(
            account_capital=float(capital),
            risk_percent=float(risk_percent),
            entry_price=float(entry_price),
            stop_loss=float(stop_loss),
            take_profit=float(take_profit),
        )
        out_col_1, out_col_2, out_col_3 = st.columns(3)
        out_col_1.metric(
            "Riesgo monetario",
            f"{risk_result['perdida_maxima_estimada']}",
        )
        rr_value = risk_result.get("relacion_riesgo_beneficio")
        out_col_2.metric(
            "Relación R/R",
            f"{rr_value}:1" if rr_value is not None else "sin TP",
        )
        out_col_3.metric(
            "Tamaño aprox. de posición",
            f"{risk_result['tamano_posicion_aproximado']}",
        )
        st.warning(EDUCATIONAL_WARNING)
        if rr_value is not None and float(rr_value) < 2:
            st.info("Con R/R menor a 1:2 no debería mostrarse como posible operación.")
    except ValueError as exc:
        st.error(str(exc))


def _render_journal(
    selected_asset: str,
    selected_timeframe: str,
) -> None:
    analyses = build_asset_analyses(selected_asset)
    snapshot = build_asset_snapshot(selected_asset, analyses, selected_timeframe)
    defaults = _trade_defaults(snapshot)
    operative_direction = str(snapshot["operative_decision"].get("direccion", "NINGUNA"))

    st.subheader("Bitácora de operaciones simuladas")
    st.caption("Registro educativo para revisar disciplina, contexto técnico y ejecución del plan.")

    with st.form("journal_form", clear_on_submit=False):
        form_col_1, form_col_2, form_col_3, form_col_4 = st.columns(4)
        journal_date = form_col_1.text_input(
            "Fecha",
            value=datetime.now().strftime("%Y-%m-%d %H:%M"),
        )
        journal_asset = form_col_2.text_input("Activo", value=selected_asset, disabled=True)
        journal_timeframe = form_col_3.text_input("Temporalidad", value=selected_timeframe.upper(), disabled=True)
        journal_direction = form_col_4.selectbox(
            "Dirección",
            ["COMPRA", "VENTA", "NINGUNA"],
            index=["COMPRA", "VENTA", "NINGUNA"].index(
                operative_direction if operative_direction in {"COMPRA", "VENTA"} else "NINGUNA"
            ),
        )

        form_col_5, form_col_6, form_col_7, form_col_8 = st.columns(4)
        journal_pattern = form_col_5.text_input("Patrón", value=str(defaults["pattern"]))
        journal_entry_price = form_col_6.number_input(
            "Entrada",
            min_value=0.0,
            value=float(defaults["entry"]),
            format="%.5f",
        )
        journal_stop = form_col_7.number_input(
            "SL",
            min_value=0.0,
            value=float(defaults["stop"]),
            format="%.5f",
        )
        journal_take_profit = form_col_8.number_input(
            "TP",
            min_value=0.0,
            value=float(defaults["take_profit"]),
            format="%.5f",
        )

        form_col_9, form_col_10 = st.columns(2)
        journal_result = form_col_9.selectbox(
            "Resultado",
            ["pendiente", "ganadora", "perdedora", "neutral"],
        )
        respected_plan = form_col_10.checkbox("¿Respeté el plan?", value=True)

        entry_reason = st.text_area("Motivo", value=str(defaults["reason"]))
        exit_reason = st.text_area("Nota de salida (opcional)")
        psychological_note = st.text_area("Nota psicológica")

        submitted = st.form_submit_button("Guardar operación simulada")
        if submitted:
            journal_entry = JournalEntry(
                date=journal_date,
                asset=journal_asset,
                timeframe=selected_timeframe,
                direction=journal_direction,
                pattern=journal_pattern or "sin patrón",
                entry_planned=float(journal_entry_price),
                stop_loss=float(journal_stop),
                take_profit=float(journal_take_profit),
                result=journal_result,
                entry_reason=entry_reason,
                exit_reason=exit_reason,
                respected_plan=bool(respected_plan),
                psychological_note=psychological_note,
            )
            record_trade(journal_entry)
            st.success("Operación simulada guardada en la bitácora.")

    journal_frame = get_journal(limit=200)
    if journal_frame.empty:
        st.info("Todavía no hay operaciones guardadas en la bitácora.")
    else:
        st.dataframe(journal_frame, use_container_width=True, hide_index=True)


def _format_exit_stats(exit_stats: dict[str, Any]) -> str:
    return f"{int(exit_stats['count'])} ({float(exit_stats['pct']):.2f}%)"


def _render_breakdown_table(title: str, frame: pd.DataFrame) -> None:
    st.markdown(f"**{title}**")
    if frame.empty:
        st.info(f"No hay datos disponibles para {title.lower()}.")
    else:
        st.dataframe(frame, use_container_width=True, hide_index=True)


def _render_backtesting(
    settings: dict[str, Any],
    selected_asset: str,
    selected_timeframe: str,
) -> None:
    requested_timeframes = list(dict.fromkeys([selected_timeframe, "4h"]))
    analyses = build_asset_analyses(selected_asset, requested_timeframes, limit=None)
    selected_analysis = analyses[selected_timeframe]
    data = selected_analysis.get("data", pd.DataFrame())
    context_frame = analyses.get("4h", {}).get("data", pd.DataFrame())

    st.subheader("Backtesting educativo")
    st.caption(
        "Motor auditado para análisis, diagnóstico y disciplina operativa. "
        "No ejecuta órdenes reales ni optimiza parámetros automáticamente."
    )
    _render_asset_availability_actions(
        settings,
        selected_asset,
        has_data=not data.empty,
        button_scope="backtest",
    )

    if data.empty:
        st.info("Selecciona un activo con datos disponibles para preparar el backtest.")
        return

    min_date = pd.Timestamp(data.index.min()).date()
    max_date = pd.Timestamp(data.index.max()).date()
    cost_unit_label = get_cost_unit_label(selected_asset)
    cost_point_size = get_cost_point_size(selected_asset)
    cost_unit_singular = cost_unit_label[:-1] if cost_unit_label.endswith("s") else cost_unit_label

    st.markdown("**Configuración de la ejecución**")
    config_col_1, config_col_2 = st.columns(2)
    configuration_name = config_col_1.text_input(
        "Nombre de configuración",
        value=f"Baseline_v0.1_{selected_asset}_{selected_timeframe}_24velas",
        key=f"backtest_name_{selected_asset}_{selected_timeframe}",
    )
    execution_mode = config_col_2.selectbox(
        "Modo de ejecución",
        options=[MODE_OPERATIVO, MODE_DIAGNOSTICO],
        format_func=lambda value: "Operativo" if value == MODE_OPERATIVO else "Diagnóstico",
        key=f"backtest_mode_{selected_asset}_{selected_timeframe}",
    )

    config_col_3, config_col_4, config_col_5 = st.columns(3)
    max_holding_bars = int(
        config_col_3.number_input(
            "Máx. velas por operación",
            min_value=4,
            max_value=160,
            value=24,
            step=4,
            key=f"backtest_maxbars_{selected_asset}_{selected_timeframe}",
        )
    )
    data_segment = config_col_4.selectbox(
        "Segmento cronológico",
        options=[SEGMENT_ALL, SEGMENT_DEVELOPMENT, SEGMENT_VALIDATION, SEGMENT_TEST],
        format_func=lambda value: {
            SEGMENT_ALL: "Todo el rango",
            SEGMENT_DEVELOPMENT: "Desarrollo (60%)",
            SEGMENT_VALIDATION: "Validación (20%)",
            SEGMENT_TEST: "Prueba final (20%)",
        }[value],
        key=f"backtest_segment_{selected_asset}_{selected_timeframe}",
    )
    gap_threshold_pct = float(
        config_col_5.number_input(
            "Gap anómalo (%)",
            min_value=0.1,
            max_value=100.0,
            value=3.0,
            step=0.1,
            key=f"backtest_gap_{selected_asset}_{selected_timeframe}",
        )
    )

    range_col_1, range_col_2 = st.columns(2)
    start_date = range_col_1.date_input(
        "Fecha inicio",
        value=min_date,
        min_value=min_date,
        max_value=max_date,
        key=f"backtest_start_{selected_asset}_{selected_timeframe}",
    )
    end_date = range_col_2.date_input(
        "Fecha fin",
        value=max_date,
        min_value=min_date,
        max_value=max_date,
        key=f"backtest_end_{selected_asset}_{selected_timeframe}",
    )

    with st.expander("Costes y fricciones de mercado", expanded=True):
        st.caption(
            f"Para {selected_asset}, 1 {cost_unit_singular} = {cost_point_size} unidades de precio. "
            "Spread, slippage, swap y comisión fija se introducen en esa unidad; la comisión porcentual se aplica sobre el precio de entrada."
        )
        cost_col_1, cost_col_2, cost_col_3, cost_col_4, cost_col_5 = st.columns(5)
        spread = float(
            cost_col_1.number_input(
                f"Spread ({cost_unit_label})",
                min_value=0.0,
                value=0.0,
                step=0.01,
                key=f"backtest_spread_{selected_asset}_{selected_timeframe}",
            )
        )
        commission = float(
            cost_col_2.number_input(
                "Comisión",
                min_value=0.0,
                value=0.0,
                step=0.01,
                key=f"backtest_commission_{selected_asset}_{selected_timeframe}",
            )
        )
        commission_mode = cost_col_3.selectbox(
            "Tipo de comisión",
            options=["fixed", "percent"],
            format_func=lambda value: "Fija" if value == "fixed" else "Porcentaje",
            key=f"backtest_commission_mode_{selected_asset}_{selected_timeframe}",
        )
        slippage = float(
            cost_col_4.number_input(
                f"Slippage ({cost_unit_label})",
                min_value=0.0,
                value=0.0,
                step=0.01,
                key=f"backtest_slippage_{selected_asset}_{selected_timeframe}",
            )
        )
        swap = float(
            cost_col_5.number_input(
                f"Swap por noche ({cost_unit_label})",
                value=0.0,
                step=0.01,
                key=f"backtest_swap_{selected_asset}_{selected_timeframe}",
            )
        )

    if start_date > end_date:
        st.error("La fecha de inicio no puede ser posterior a la fecha final.")
        return

    backtest_config = BacktestConfig(
        configuration_name=configuration_name.strip() or f"{selected_asset}_{selected_timeframe}_backtest",
        max_holding_bars=max_holding_bars,
        mode=execution_mode,
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d"),
        data_segment=data_segment,
        gap_threshold_pct=gap_threshold_pct,
        costs=TradingCosts(
            spread=spread,
            commission=commission,
            commission_mode=commission_mode,
            slippage=slippage,
            swap_per_night=swap,
        ),
    )
    current_signature = backtest_config.parameter_hash(asset=selected_asset, timeframe=selected_timeframe)
    frozen_signature = settings.get("backtest_frozen_signature")
    frozen_at = settings.get("backtest_frozen_at")
    final_segment_blocked = data_segment == SEGMENT_TEST and current_signature != frozen_signature

    freeze_col_1, freeze_col_2 = st.columns([2, 1])
    freeze_col_1.caption(
        "Hash activo: "
        + current_signature
        + (
            f" | Configuración congelada: {frozen_signature} ({frozen_at})"
            if frozen_signature
            else " | No hay configuración congelada para prueba final."
        )
    )
    if freeze_col_2.button("Congelar configuración actual", use_container_width=True):
        updated_settings = save_app_settings(
            {
                **settings,
                "backtest_frozen_signature": current_signature,
                "backtest_frozen_at": now_storage_timestamp(),
            }
        )
        st.session_state["refresh_feedback"] = {
            "kind": "success",
            "text": "Configuración congelada para permitir la prueba final cuando estés listo.",
        }
        settings.update(updated_settings)
        st.rerun()

    if final_segment_blocked:
        st.warning(
            "La prueba final está bloqueada: la configuración actual no coincide con la versión congelada."
        )

    if st.button("Ejecutar backtest", use_container_width=True):
        if final_segment_blocked:
            st.error(
                "No se ejecutó la prueba final porque la configuración cambió respecto a la versión congelada."
            )
        else:
            with st.spinner("Ejecutando backtest auditado..."):
                result = backtest_patterns(
                    frame=data,
                    asset=selected_asset,
                    timeframe=selected_timeframe,
                    config=backtest_config,
                    context_frame=context_frame if not context_frame.empty else None,
                )
                run_id = save_backtest_run(result)
            result["run_id"] = run_id
            st.session_state["backtest_result"] = {
                "asset": selected_asset,
                "timeframe": selected_timeframe,
                "result": result,
            }

    backtest_result = st.session_state.get("backtest_result")
    if not backtest_result:
        st.info("Ejecuta el backtest para ver resultados.")
    elif (
        backtest_result.get("asset") != selected_asset
        or backtest_result.get("timeframe") != selected_timeframe
    ):
        st.info("Ejecuta de nuevo el backtest para esta combinación de activo y temporalidad.")
    else:
        result = backtest_result["result"]
        execution = result["execution"]
        statistics = result["statistics"]
        trades = result["trades"]
        data_quality_log = result["data_quality_log"]
        blocked_signals = result["blocked_signals"]

        st.caption(
            f"Configuración: {execution['nombre_configuracion']} | "
            f"Ejecutado: {execution['fecha_ejecucion']} | "
            f"Hash: {execution['parameter_hash']}"
        )
        if execution.get("segment_info"):
            segment_info = execution["segment_info"]
            st.caption(
                "Rango efectivo: "
                f"{segment_info.get('effective_start', 'n/a')} -> {segment_info.get('effective_end', 'n/a')}"
            )

        stats_col_1, stats_col_2, stats_col_3, stats_col_4 = st.columns(4)
        stats_col_1.metric("Operaciones", statistics["numero_senales"])
        stats_col_2.metric("Win rate", f"{statistics['win_rate']}%")
        stats_col_3.metric("Ganancia neta R", statistics["ganancia_neta_R"])
        stats_col_4.metric("Profit factor", statistics["profit_factor"])

        stats_col_5, stats_col_6, stats_col_7, stats_col_8 = st.columns(4)
        stats_col_5.metric("Expectativa bruta", statistics["expectativa_bruta"])
        stats_col_6.metric("Expectativa neta", statistics["expectativa_neta"])
        stats_col_7.metric("Max drawdown R", statistics["max_drawdown_R"])
        stats_col_8.metric("Señales bloqueadas", statistics["senales_bloqueadas"])

        stats_col_9, stats_col_10, stats_col_11, stats_col_12 = st.columns(4)
        stats_col_9.metric("Promedio ganador R", statistics["promedio_ganador_R"])
        stats_col_10.metric("Promedio perdedor R", statistics["promedio_perdedor_R"])
        stats_col_11.metric("Duración media", statistics["duracion_media_velas"])
        stats_col_12.metric("Máx. racha perdedora", statistics["max_racha_perdedora"])

        stats_col_13, stats_col_14, stats_col_15, stats_col_16 = st.columns(4)
        stats_col_13.metric("Salidas TP", _format_exit_stats(statistics["salidas_por_tp"]))
        stats_col_14.metric("Salidas SL", _format_exit_stats(statistics["salidas_por_sl"]))
        stats_col_15.metric("Salidas tiempo", _format_exit_stats(statistics["salidas_por_tiempo"]))
        stats_col_16.metric("MAE / MFE", f"{statistics['MAE_promedio']} / {statistics['MFE_promedio']}")

        st.markdown("**Log de calidad de datos**")
        if data_quality_log.empty:
            st.success("No se detectaron incidencias de calidad de datos en esta ejecución.")
        else:
            st.dataframe(data_quality_log, use_container_width=True, hide_index=True)

        if not trades.empty:
            st.markdown("**Curva de equity (bruta vs neta)**")
            st.line_chart(
                trades.set_index("fecha_salida")[["equity_curve_bruta_R", "equity_curve_neta_R"]]
            )

            export_columns = [
                "fecha_entrada",
                "fecha_salida",
                "patron",
                "direccion",
                "entrada",
                "stop_loss",
                "take_profit",
                "resultado_R",
                "resultado_neto_R",
                "risk_price",
                "spread_input",
                "spread_unit",
                "spread_price",
                "slippage_input",
                "slippage_unit",
                "slippage_price",
                "total_cost_price",
                "cost_R",
                "tipo_salida",
                "duracion_velas",
                "contexto_4h",
                "MAE",
                "MFE",
            ]
            csv_bytes = (
                result["export_trades"][export_columns]
                .to_csv(index=False)
                .encode("utf-8")
            )
            st.download_button(
                "Exportar operaciones CSV",
                data=csv_bytes,
                file_name=f"{execution['nombre_configuracion']}.csv",
                mime="text/csv",
                use_container_width=True,
            )

            _render_breakdown_table("Resultado por mes", result["result_by_month"])
            _render_breakdown_table("Resultado por patrón", result["result_by_pattern"])
            _render_breakdown_table("Resultado por dirección", result["result_by_direction"])
            _render_breakdown_table("Resultado por contexto 4H", result["result_by_context_4h"])

            if not blocked_signals.empty:
                st.markdown("**Señales bloqueadas**")
                st.dataframe(blocked_signals, use_container_width=True, hide_index=True)

            st.markdown("**Detalle de operaciones simuladas**")
            st.dataframe(trades, use_container_width=True, hide_index=True)
        else:
            st.info("No hubo operaciones ejecutadas con la configuración actual.")

    st.markdown("**Historial de ejecuciones registradas**")
    history = load_backtest_runs(limit=20)
    if history.empty:
        st.info("Todavía no hay ejecuciones registradas.")
    else:
        st.dataframe(history, use_container_width=True, hide_index=True)


def _render_settings(settings: dict[str, Any]) -> dict[str, Any]:
    st.subheader("Configuración")
    st.caption("Los cambios modifican qué activos se analizan y cómo se refresca la información.")

    current_settings = dict(settings)
    with st.form("settings_form"):
        active_assets = st.multiselect(
            "Activos activos/inactivos",
            options=_all_asset_names(),
            default=current_settings["active_assets"],
        )
        visible_timeframes = st.multiselect(
            "Temporalidades visibles para dashboard y backtesting",
            options=["1h", "4h", "1d", "1wk"],
            default=current_settings["visible_timeframes"],
        )
        auto_refresh = st.selectbox(
            "Intervalo de actualización automática",
            options=list(REFRESH_OPTIONS.keys()),
            index=list(REFRESH_OPTIONS.keys()).index(current_settings["auto_refresh"]),
            format_func=lambda key: str(REFRESH_OPTIONS[key]["label"]),
        )
        col_1, col_2 = st.columns(2)
        reference_capital = col_1.number_input(
            "Capital de referencia",
            min_value=1.0,
            value=float(current_settings["reference_capital"]),
            step=100.0,
        )
        risk_per_trade = col_2.number_input(
            "Riesgo por operación (%)",
            min_value=0.1,
            max_value=100.0,
            value=float(current_settings["risk_per_trade"]),
            step=0.1,
        )

        submitted = st.form_submit_button("Guardar configuración")
        if submitted:
            if not active_assets:
                st.error("Debes mantener al menos un activo activo.")
            elif not visible_timeframes:
                st.error("Debes mantener al menos una temporalidad visible.")
            else:
                updated_settings = save_app_settings(
                    {
                        **current_settings,
                        "active_assets": active_assets,
                        "visible_timeframes": visible_timeframes,
                        "auto_refresh": auto_refresh,
                        "reference_capital": reference_capital,
                        "risk_per_trade": risk_per_trade,
                    }
                )
                if st.session_state.get("ui_timeframe") not in updated_settings["visible_timeframes"]:
                    st.session_state["ui_timeframe"] = updated_settings["visible_timeframes"][0]
                _update_refresh_feedback("success", "Configuración guardada.")
                st.rerun()

    assets_frame = pd.DataFrame(
        [
            {
                "activo": asset,
                "ticker": ASSETS[asset],
                "estado": "Activo" if asset in current_settings["active_assets"] else "Inactivo",
            }
            for asset in _all_asset_names()
        ]
    )
    st.markdown("**Activos disponibles**")
    st.dataframe(assets_frame, use_container_width=True, hide_index=True)

    st.markdown("**Estado de la última descarga**")
    show_download_status(st.session_state.get("last_refresh_results", pd.DataFrame()))
    return current_settings


init_database()
settings = load_app_settings()
settings = _bootstrap_data(settings)

with st.sidebar:
    st.header("Laboratorio")
    current_page = st.radio("Módulo", PAGE_OPTIONS, key="ui_page")

    all_assets = _all_asset_names()
    default_asset = st.session_state.get(
        "ui_asset",
        settings.get("active_assets", DEFAULT_ACTIVE_ASSETS)[0] if settings.get("active_assets") else all_assets[0],
    )
    if default_asset not in all_assets:
        default_asset = all_assets[0]
    selected_asset = st.selectbox(
        "Activo seleccionado",
        all_assets,
        index=all_assets.index(default_asset),
        format_func=lambda asset: asset if _is_asset_active(settings, asset) else f"{asset} (inactivo)",
    )
    st.session_state["ui_asset"] = selected_asset
    st.caption(
        "Estado del activo: "
        + ("Activo" if _is_asset_active(settings, selected_asset) else "Inactivo, disponible para activar")
    )

    visible_timeframes = settings.get("visible_timeframes", ["1h", "4h", "1d", "1wk"])
    default_timeframe = st.session_state.get("ui_timeframe", visible_timeframes[0])
    if default_timeframe not in visible_timeframes:
        default_timeframe = visible_timeframes[0]
    selected_timeframe = st.selectbox(
        "Temporalidad",
        visible_timeframes,
        index=visible_timeframes.index(default_timeframe),
    )
    st.session_state["ui_timeframe"] = selected_timeframe

    selected_chart = st.selectbox("Tipo de gráfico", ["Velas", "Línea"], key="ui_chart")

    if st.button("Actualizar datos", use_container_width=True):
        settings = _run_market_refresh(settings, "manual")
        st.rerun()

    st.markdown("---")
    st.caption(f"Actualización automática: {get_refresh_label(settings)}")
    st.caption(
        f"Última actualización automática: {format_refresh_timestamp(settings.get('last_refresh_at'))}"
    )
    next_refresh = _next_refresh_timestamp(settings)
    st.caption(
        "Próxima actualización estimada: "
        + (next_refresh.strftime("%H:%M") if next_refresh is not None else "manual")
    )
    _show_refresh_feedback()

settings = _handle_auto_refresh(current_page, settings)

st.title("Laboratorio de Análisis Técnico Financiero")
st.caption(
    "Herramienta educativa para análisis, práctica de riesgo, bitácora y backtesting. No ejecuta órdenes reales."
)
st.warning("No es asesoría financiera. Todas las lecturas deben validarse manualmente.")

if current_page == "Dashboard":
    _render_dashboard(settings, selected_asset, selected_timeframe, selected_chart)
elif current_page == "Escáner de activos":
    _render_scanner(settings)
elif current_page == "Calculadora de riesgo":
    _render_risk_calculator(settings, selected_asset, selected_timeframe)
elif current_page == "Bitácora":
    _render_journal(selected_asset, selected_timeframe)
elif current_page == "Backtesting":
    _render_backtesting(settings, selected_asset, selected_timeframe)
else:
    settings = _render_settings(settings)
