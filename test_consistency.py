from __future__ import annotations

import unittest

import pandas as pd

from analysis_service import _align_operative_decision
from decision_engine import POSIBLE_OPERACION, VIGILAR
from multi_timeframe import compare_multi_timeframe
from watchlist import build_watchlist


def _frame(
    close: float,
    ema_50: float,
    ema_200: float,
    rsi_14: float,
    atr_14: float = 1.0,
) -> pd.DataFrame:
    index = pd.date_range("2025-01-01", periods=3, freq="h")
    return pd.DataFrame(
        {
            "close": [close, close, close],
            "ema_50": [ema_50, ema_50, ema_50],
            "ema_200": [ema_200, ema_200, ema_200],
            "rsi_14": [rsi_14, rsi_14, rsi_14],
            "atr_14": [atr_14, atr_14, atr_14],
        },
        index=index,
    )


def _analysis(
    trend: str,
    frame: pd.DataFrame,
    *,
    signals: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    latest_close = float(frame["close"].iloc[-1])
    return {
        "data": frame,
        "general_trend": trend,
        "signals": list(signals or []),
        "market_structure": {
            "trend": trend,
            "supports": [latest_close * 0.98],
            "resistances": [latest_close * 1.02],
            "relevant_highs": [],
            "relevant_lows": [],
        },
    }


def _decision(
    direction: str,
    *,
    warnings: list[str] | None = None,
) -> dict[str, object]:
    alignment_word = "compra" if direction == "COMPRA" else "venta"
    trend_word = "alcista" if direction == "COMPRA" else "bajista"
    return {
        "estado": POSIBLE_OPERACION,
        "direccion": direction,
        "score": 82,
        "motivos": [f"La tendencia {trend_word} esta alineada con el sesgo de {alignment_word}."],
        "advertencias": list(warnings or []),
        "patron_confirmado": True,
        "sl_tp_validos": True,
        "riesgo_beneficio_valido": True,
        "entrada_habilitada": True,
    }


class ConsistencyTests(unittest.TestCase):
    def test_case_a_wait_blocks_buy_language(self) -> None:
        analyses = {
            "4h": _analysis("bajista", _frame(close=90, ema_50=95, ema_200=100, rsi_14=35)),
            "1h": _analysis("lateral", _frame(close=100, ema_50=100, ema_200=100, rsi_14=50)),
        }
        multi_timeframe = compare_multi_timeframe(analyses)

        self.assertEqual(multi_timeframe["sesgo_permitido"], "ESPERAR")
        self.assertFalse(multi_timeframe["alineacion"])

        adjusted = _align_operative_decision(_decision("COMPRA"), multi_timeframe)

        self.assertEqual(adjusted["estado"], VIGILAR)
        self.assertEqual(adjusted["direccion"], "NINGUNA")
        self.assertFalse(adjusted["entrada_habilitada"])
        self.assertTrue(any("esperar" in reason.lower() for reason in adjusted["motivos"]))
        self.assertFalse(any("sesgo de compra" in reason.lower() for reason in adjusted["motivos"]))

    def test_mixed_and_lateral_labels_are_preserved_in_wait_reason(self) -> None:
        analyses = {
            "4h": _analysis("lateral", _frame(close=100, ema_50=100, ema_200=100, rsi_14=50)),
            "1h": _analysis("lateral", _frame(close=104, ema_50=102, ema_200=100, rsi_14=60)),
        }
        multi_timeframe = compare_multi_timeframe(analyses)

        self.assertEqual(multi_timeframe["sesgo_permitido"], "ESPERAR")
        self.assertFalse(multi_timeframe["alineacion"])

        adjusted = _align_operative_decision(
            _decision(
                "COMPRA",
                warnings=["La tendencia actual va en contra del sesgo operativo."],
            ),
            multi_timeframe,
        )

        self.assertTrue(
            any(
                reason
                == "4H mixto/lateral + 1H alcista con tendencia lateral = no hay alineacion completa; esperar confirmacion."
                for reason in adjusted["motivos"]
            )
        )
        self.assertFalse(
            any(
                warning == "La tendencia actual va en contra del sesgo operativo."
                for warning in adjusted["advertencias"]
            )
        )
        self.assertTrue(
            any(
                warning == "La lectura 1H aun no esta suficientemente alineada con 4H para habilitar entrada."
                for warning in adjusted["advertencias"]
            )
        )

    def test_case_b_aligned_buy_is_preserved(self) -> None:
        analyses = {
            "4h": _analysis("alcista", _frame(close=110, ema_50=105, ema_200=100, rsi_14=62)),
            "1h": _analysis("alcista", _frame(close=108, ema_50=104, ema_200=101, rsi_14=58)),
        }
        multi_timeframe = compare_multi_timeframe(analyses)

        self.assertEqual(multi_timeframe["sesgo_permitido"], "COMPRA")
        self.assertTrue(multi_timeframe["alineacion"])

        adjusted = _align_operative_decision(_decision("COMPRA"), multi_timeframe)

        self.assertEqual(adjusted["direccion"], "COMPRA")
        self.assertTrue(adjusted["entrada_habilitada"])
        self.assertIn("compra", str(adjusted["resumen_operativo"]).lower())
        self.assertTrue(any("sesgo de compra" in reason.lower() for reason in adjusted["motivos"]))

    def test_case_c_aligned_sell_is_preserved(self) -> None:
        analyses = {
            "4h": _analysis("bajista", _frame(close=90, ema_50=95, ema_200=100, rsi_14=38)),
            "1h": _analysis("bajista", _frame(close=92, ema_50=96, ema_200=101, rsi_14=42)),
        }
        multi_timeframe = compare_multi_timeframe(analyses)

        self.assertEqual(multi_timeframe["sesgo_permitido"], "VENTA")
        self.assertTrue(multi_timeframe["alineacion"])

        adjusted = _align_operative_decision(_decision("VENTA"), multi_timeframe)

        self.assertEqual(adjusted["direccion"], "VENTA")
        self.assertTrue(adjusted["entrada_habilitada"])
        self.assertIn("venta", str(adjusted["resumen_operativo"]).lower())
        self.assertTrue(any("sesgo de venta" in reason.lower() for reason in adjusted["motivos"]))

    def test_case_d_conflict_forces_wait_language(self) -> None:
        analyses = {
            "4h": _analysis("alcista", _frame(close=110, ema_50=105, ema_200=100, rsi_14=60)),
            "1h": _analysis("bajista", _frame(close=92, ema_50=96, ema_200=101, rsi_14=40)),
        }
        multi_timeframe = compare_multi_timeframe(analyses)

        self.assertEqual(multi_timeframe["sesgo_permitido"], "ESPERAR")
        self.assertFalse(multi_timeframe["alineacion"])

        adjusted = _align_operative_decision(_decision("VENTA"), multi_timeframe)

        self.assertEqual(adjusted["direccion"], "NINGUNA")
        self.assertFalse(adjusted["entrada_habilitada"])
        self.assertTrue(any("conflicto" in reason.lower() for reason in adjusted["motivos"]))
        self.assertIn("esperar", str(adjusted["resumen_operativo"]).lower())

    def test_watchlist_stays_neutral_when_bias_is_wait(self) -> None:
        analyses = {
            "4h": _analysis("bajista", _frame(close=90, ema_50=95, ema_200=100, rsi_14=35)),
            "1h": _analysis("lateral", _frame(close=100, ema_50=100, ema_200=100, rsi_14=50)),
        }
        multi_timeframe = compare_multi_timeframe(analyses)
        adjusted = _align_operative_decision(_decision("COMPRA"), multi_timeframe)

        watchlist = build_watchlist(
            asset="TEST",
            decision=adjusted,
            multi_timeframe=multi_timeframe,
            context_analysis=analyses["4h"],
            entry_analysis=analyses["1h"],
        )

        self.assertEqual(watchlist["direccion_observada"], "NINGUNA")
        self.assertIn("Esperar", str(watchlist["mensaje"]))


if __name__ == "__main__":
    unittest.main()
