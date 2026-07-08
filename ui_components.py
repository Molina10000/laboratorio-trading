from __future__ import annotations

from html import escape
from textwrap import dedent

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


def inject_global_styles() -> None:
    st.markdown(
        """
        <style>
            .lab-section-label {
                font-size: 0.82rem;
                color: #6b7280;
                text-transform: uppercase;
                letter-spacing: 0.06em;
                font-weight: 700;
            }
            .lab-note {
                color: #4b5563;
                font-size: 0.95rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def build_price_chart(
    frame: pd.DataFrame,
    chart_type: str,
    asset: str,
    timeframe: str,
    supports: list[float],
    resistances: list[float],
) -> go.Figure:
    figure = go.Figure()

    if chart_type == "Velas":
        figure.add_trace(
            go.Candlestick(
                x=frame.index,
                open=frame["open"],
                high=frame["high"],
                low=frame["low"],
                close=frame["close"],
                name="OHLC",
            )
        )
    else:
        figure.add_trace(
            go.Scatter(
                x=frame.index,
                y=frame["close"],
                mode="lines",
                name="Cierre",
                line={"color": "#1f77b4", "width": 2},
            )
        )

    figure.add_trace(
        go.Scatter(
            x=frame.index,
            y=frame["ema_50"],
            mode="lines",
            name="EMA 50",
            line={"color": "#ff7f0e", "width": 1.5},
        )
    )
    figure.add_trace(
        go.Scatter(
            x=frame.index,
            y=frame["ema_200"],
            mode="lines",
            name="EMA 200",
            line={"color": "#2ca02c", "width": 1.5},
        )
    )

    for level in supports:
        figure.add_hline(y=level, line_dash="dot", line_color="#17becf")
    for level in resistances:
        figure.add_hline(y=level, line_dash="dot", line_color="#d62728")

    figure.update_layout(
        title=f"{asset} - {timeframe}",
        xaxis_title="Fecha",
        yaxis_title="Precio",
        height=520,
        xaxis_rangeslider_visible=False,
        template="plotly_white",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
    )
    return figure


def show_download_status(results_frame: pd.DataFrame) -> None:
    if results_frame.empty:
        return

    if (results_frame["estado"] == "error").any():
        st.warning("Hubo errores de descarga. Se conservan los últimos datos válidos.")
    else:
        st.success("Datos actualizados correctamente.")

    st.dataframe(results_frame, use_container_width=True, hide_index=True)


def render_multi_timeframe_card(multi_timeframe: dict[str, object]) -> None:
    detail = multi_timeframe.get("detalle", {})
    detail_4h = detail.get("4H", {})
    detail_1h = detail.get("1H", {})
    alignment = bool(multi_timeframe.get("alineacion"))
    context = str(multi_timeframe.get("contexto", "MIXTO"))
    bias = str(multi_timeframe.get("sesgo_permitido", "ESPERAR"))
    explanations = multi_timeframe.get("explicacion", []) or []

    border_color = "#15803d" if alignment else "#c2410c"
    background = "#f0fdf4" if alignment else "#fff7ed"
    alignment_text = "Confirmada" if alignment else "Esperar confirmación"

    explanation_html = "".join(
        f"<li>{escape(str(item))}</li>" for item in explanations
    )

    st.subheader("Análisis Multi Temporalidad")
    st.markdown(
        f"""
        <div style="
            background: {background};
            border: 1px solid #e5e7eb;
            border-left: 8px solid {border_color};
            border-radius: 18px;
            padding: 1.2rem 1.35rem;
            margin: 0.35rem 0 1rem 0;
        ">
            <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 0.8rem;">
                <div>
                    <div class="lab-section-label">4H</div>
                    <div style="font-size:1.15rem; font-weight:800; color:#111827;">{escape(str(detail_4h.get("context", "MIXTO")))}</div>
                    <div class="lab-note">Trend: {escape(str(detail_4h.get("trend", "MIXTO")))}</div>
                </div>
                <div>
                    <div class="lab-section-label">1H</div>
                    <div style="font-size:1.15rem; font-weight:800; color:#111827;">{escape(str(detail_1h.get("context", "MIXTO")))}</div>
                    <div class="lab-note">Trend: {escape(str(detail_1h.get("trend", "MIXTO")))}</div>
                </div>
                <div>
                    <div class="lab-section-label">Alineación</div>
                    <div style="font-size:1.15rem; font-weight:800; color:#111827;">{alignment_text}</div>
                    <div class="lab-note">Contexto: {escape(context)}</div>
                </div>
                <div>
                    <div class="lab-section-label">Sesgo permitido</div>
                    <div style="font-size:1.15rem; font-weight:800; color:#111827;">{escape(bias)}</div>
                    <div class="lab-note">4H suma contexto, 1H afina entrada</div>
                </div>
            </div>
            <div style="margin-top:0.95rem;">
                <div style="font-weight:700; color:#111827; margin-bottom:0.35rem;">Explicación</div>
                <ul style="margin:0; padding-left:1.1rem; color:#1f2937;">
                    {explanation_html}
                </ul>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _display_operational_bias(multi_timeframe: dict[str, object]) -> str:
    bias = str(multi_timeframe.get("sesgo_permitido", "ESPERAR"))
    if bias in {"COMPRA", "VENTA"}:
        return bias
    return "ESPERAR"


def _entry_status(decision: dict[str, object]) -> str:
    return "HABILITADA" if bool(decision.get("entrada_habilitada")) else "NO HABILITADA"


def _decision_summary(decision: dict[str, object], multi_timeframe: dict[str, object]) -> str:
    summary = str(decision.get("resumen_operativo", "") or "").strip()
    if summary:
        return summary

    bias = str(multi_timeframe.get("sesgo_permitido", "ESPERAR"))
    if bias in {"COMPRA", "VENTA"} and not bool(decision.get("patron_confirmado")):
        return f"Tendencia alineada con sesgo {bias.lower()}, pero falta patrón confirmado."
    if bias in {"COMPRA", "VENTA"} and not bool(decision.get("sl_tp_validos")):
        return "Hay contexto alineado, pero faltan stop loss y take profit válidos."
    if bias in {"COMPRA", "VENTA"} and not bool(decision.get("riesgo_beneficio_valido")):
        return "Hay contexto alineado, pero la relación riesgo/beneficio todavía no es válida."
    if bool(decision.get("entrada_habilitada")):
        return "La entrada quedó habilitada por patrón confirmado, niveles válidos y riesgo/beneficio aceptable."
    return "No existe alineación suficiente ni contexto claro para habilitar una entrada."


def render_decision_card(
    decision: dict[str, object],
    multi_timeframe: dict[str, object],
) -> None:
    color_map = {
        "NO OPERAR": {
            "background": "#f3f4f6",
            "border": "#9ca3af",
            "accent": "#b91c1c",
        },
        "VIGILAR": {
            "background": "#fff7ed",
            "border": "#fdba74",
            "accent": "#c2410c",
        },
        "POSIBLE OPERACIÓN": {
            "background": "#ecfdf5",
            "border": "#86efac",
            "accent": "#15803d",
        },
    }
    palette = color_map[str(decision["estado"])]
    motives = decision.get("motivos", []) or ["Sin motivos registrados."]
    warnings = decision.get("advertencias", []) or ["Sin advertencias adicionales."]
    bias_label = _display_operational_bias(multi_timeframe)
    entry_label = _entry_status(decision)
    summary_text = _decision_summary(decision, multi_timeframe)

    motives_html = "".join(f"<li>{escape(str(item))}</li>" for item in motives)
    warnings_html = "".join(f"<li>{escape(str(item))}</li>" for item in warnings)

    st.markdown(
        f"""
        <div style="
            background: {palette['background']};
            border: 1px solid {palette['border']};
            border-left: 8px solid {palette['accent']};
            border-radius: 18px;
            padding: 1.25rem 1.5rem;
            margin: 0.5rem 0 1.25rem 0;
        ">
            <div style="display:flex; justify-content:space-between; gap:1rem; align-items:flex-start; flex-wrap:wrap;">
                <div>
                    <div style="font-size:0.85rem; font-weight:700; letter-spacing:0.06em; color:{palette['accent']};">
                        SEMÁFORO OPERATIVO
                    </div>
                    <div style="font-size:1.9rem; font-weight:800; color:#111827; margin-top:0.2rem;">
                        {escape(str(decision["estado"]))}
                    </div>
                    <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap:0.75rem; margin-top:0.55rem;">
                        <div>
                            <div class="lab-section-label">Sesgo operativo</div>
                            <div style="font-size:1rem; font-weight:800; color:#111827;">{escape(bias_label)}</div>
                        </div>
                        <div>
                            <div class="lab-section-label">Entrada</div>
                            <div style="font-size:1rem; font-weight:800; color:#111827;">{escape(entry_label)}</div>
                        </div>
                    </div>
                    <div style="font-size:0.95rem; color:#374151; margin-top:0.7rem;">
                        {escape(summary_text)}
                    </div>
                </div>
                <div style="min-width:120px; text-align:right;">
                    <div class="lab-section-label">Score</div>
                    <div style="font-size:2.6rem; line-height:1; font-weight:800; color:{palette['accent']};">
                        {escape(str(decision["score"]))}
                    </div>
                </div>
            </div>
            <div style="margin-top:1rem;">
                <div style="font-weight:700; color:#111827; margin-bottom:0.35rem;">Motivos principales</div>
                <ul style="margin:0; padding-left:1.1rem; color:#1f2937;">
                    {motives_html}
                </ul>
            </div>
            <div style="margin-top:0.85rem;">
                <div style="font-weight:700; color:#111827; margin-bottom:0.35rem;">Advertencias</div>
                <ul style="margin:0; padding-left:1.1rem; color:#4b5563;">
                    {warnings_html}
                </ul>
            </div>
            <div style="margin-top:1rem; font-size:0.95rem; font-weight:700; color:#7c2d12;">
                Esto no es asesoría financiera. Validar manualmente antes de operar.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_watchlist_card(
    watchlist: dict[str, object],
    decision: dict[str, object],
) -> None:
    principal = watchlist.get("escenario_principal", {}) or {}
    secondary = watchlist.get("escenario_secundario", {}) or {}
    direction = str(watchlist.get("direccion_observada", "NINGUNA"))
    active_watch = str(decision.get("estado", "")) == "VIGILAR"

    border_color = "#c2410c" if active_watch else "#6b7280"
    background = "#fff7ed" if active_watch else "#f9fafb"
    status_text = (
        f"Vigilar {direction.lower()}"
        if active_watch and direction != "NINGUNA"
        else "Esperar confirmacion"
        if active_watch
        else "Sin plan de espera activo"
    )

    def render_scenario_block(
        scenario: dict[str, object],
        block_background: str,
        accent: str,
        warning_color: str,
    ) -> str:
        fulfilled = scenario.get("condiciones_cumplidas", []) or []
        pending = scenario.get("condiciones_pendientes", []) or []
        fulfilled_html = "".join(
            f"<li>&#10003; {escape(str(item))}</li>" for item in fulfilled
        ) or "<li>&#10003; Sin condiciones cumplidas registradas.</li>"
        pending_html = "".join(
            f"<li>&#9633; {escape(str(item))}</li>" for item in pending
        ) or "<li>&#9633; Sin condiciones pendientes registradas.</li>"
        warning = str(scenario.get("advertencia", "") or "")
        warning_html = (
            f'<div style="margin-top:0.75rem; font-weight:700; color:{warning_color};">{escape(warning)}</div>'
            if warning
            else ""
        )
        level = str(scenario.get("nivel_a_vigilar", "") or "sin nivel definido")
        message = str(scenario.get("mensaje", "") or "")
        title = str(scenario.get("titulo", "Escenario"))
        scenario_direction = str(scenario.get("direccion", "NINGUNA"))
        waiting_html = (
            '<div style="margin-top:0.85rem; font-weight:700; color:#111827;">¿Qué estoy esperando?</div>'
            f'<div style="margin-top:0.35rem; color:#1f2937;">{escape(message)}</div>'
        )

        return "".join(
            [
                f'<div style="background:{block_background}; border:1px solid #e5e7eb; border-left:6px solid {accent}; border-radius:14px; padding:1rem 1.05rem;">',
                '<div style="display:flex; justify-content:space-between; gap:1rem; align-items:flex-start; flex-wrap:wrap;">',
                "<div>",
                f'<div style="font-size:1rem; font-weight:800; color:#111827;">{escape(title)}</div>',
                f'<div class="lab-note" style="margin-top:0.2rem;">Dirección observada: <strong>{escape(scenario_direction)}</strong></div>',
                "</div>",
                '<div style="text-align:right;">',
                '<div class="lab-section-label">Nivel a vigilar</div>',
                f'<div style="font-size:1.1rem; font-weight:800; color:#111827;">{escape(level)}</div>',
                "</div>",
                "</div>",
                '<div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap:1rem; margin-top:0.9rem;">',
                "<div>",
                '<div style="font-weight:700; color:#111827; margin-bottom:0.35rem;">Condiciones cumplidas</div>',
                f'<ul style="margin:0; padding-left:1.1rem; color:#166534;">{fulfilled_html}</ul>',
                "</div>",
                "<div>",
                '<div style="font-weight:700; color:#111827; margin-bottom:0.35rem;">Condiciones pendientes</div>',
                f'<ul style="margin:0; padding-left:1.1rem; color:#92400e;">{pending_html}</ul>',
                "</div>",
                "</div>",
                warning_html,
                waiting_html,
                "</div>",
            ]
        )

    st.subheader("Plan de Espera")
    html_card = dedent(
        f"""
        <div style="
            background: {background};
            border: 1px solid #e5e7eb;
            border-left: 8px solid {border_color};
            border-radius: 18px;
            padding: 1.2rem 1.35rem;
            margin: 0.35rem 0 1rem 0;
        ">
            <div style="display:flex; justify-content:space-between; gap:1rem; align-items:flex-start; flex-wrap:wrap;">
                <div>
                    <div class="lab-section-label">Estado</div>
                    <div style="font-size:1.35rem; font-weight:800; color:#111827;">{escape(status_text)}</div>
                    <div class="lab-note" style="margin-top:0.35rem;">Activo: {escape(str(watchlist.get("activo", "")))}</div>
                </div>
                <div style="text-align:right;">
                    <div class="lab-section-label">Activo</div>
                    <div style="font-size:1.1rem; font-weight:800; color:#111827;">{escape(str(watchlist.get("activo", "")))}</div>
                </div>
            </div>
            <div style="display:grid; grid-template-columns: repeat(1, minmax(0, 1fr)); gap: 0.9rem; margin-top:1rem;">
                {render_scenario_block(principal, "#ffffff", "#1d4ed8", "#1d4ed8")}
                {render_scenario_block(secondary, "#fffaf0", "#b45309", "#b91c1c")}
            </div>
        </div>
        """
    ).strip()
    st.markdown(html_card, unsafe_allow_html=True)


def render_time_horizon_card(time_horizon: dict[str, object]) -> None:
    horizon = str(time_horizon.get("horizonte", "SIN CLASIFICAR"))
    duration = str(time_horizon.get("duracion_estimada", "pendiente"))
    reasons = time_horizon.get("razones", []) or ["Sin razones disponibles."]
    warnings = time_horizon.get("advertencias", []) or ["Sin advertencias adicionales."]
    timeframes = ", ".join(time_horizon.get("temporalidades_usadas", []) or ["Sin datos"])

    palette = {
        "CORTO PLAZO": ("#eff6ff", "#2563eb"),
        "MEDIO PLAZO": ("#fefce8", "#ca8a04"),
        "LARGO PLAZO": ("#f0fdf4", "#15803d"),
        "SIN CLASIFICAR": ("#f3f4f6", "#6b7280"),
    }
    background, accent = palette.get(horizon, palette["SIN CLASIFICAR"])
    reasons_html = "".join(f"<li>{escape(str(item))}</li>" for item in reasons)
    warnings_html = "".join(f"<li>{escape(str(item))}</li>" for item in warnings)

    st.subheader("Horizonte Sugerido")
    st.markdown(
        f"""
        <div style="
            background:{background};
            border:1px solid #e5e7eb;
            border-left:8px solid {accent};
            border-radius:18px;
            padding:1.2rem 1.35rem;
            margin:0.35rem 0 1rem 0;
        ">
            <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 0.8rem;">
                <div>
                    <div class="lab-section-label">Horizonte</div>
                    <div style="font-size:1.45rem; font-weight:800; color:#111827;">{escape(horizon)}</div>
                </div>
                <div>
                    <div class="lab-section-label">Duración estimada</div>
                    <div style="font-size:1.1rem; font-weight:700; color:#111827;">{escape(duration)}</div>
                </div>
                <div>
                    <div class="lab-section-label">Temporalidades usadas</div>
                    <div style="font-size:1rem; font-weight:700; color:#111827;">{escape(timeframes)}</div>
                </div>
            </div>
            <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 1rem; margin-top:1rem;">
                <div>
                    <div style="font-weight:700; color:#111827; margin-bottom:0.35rem;">Razones</div>
                    <ul style="margin:0; padding-left:1.1rem; color:#1f2937;">
                        {reasons_html}
                    </ul>
                </div>
                <div>
                    <div style="font-weight:700; color:#111827; margin-bottom:0.35rem;">Advertencias</div>
                    <ul style="margin:0; padding-left:1.1rem; color:#4b5563;">
                        {warnings_html}
                    </ul>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
