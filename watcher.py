from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.analysis_engine import analizar_activos
from core.data_loader import refresh_market_data, summarize_download_results
from core.utils import (
    PROJECT_ROOT,
    build_telegram_message,
    load_json_file,
    public_state_for_storage,
    save_json_file,
    send_telegram_message,
)


WATCHED_ASSETS = {
    "EUR/USD": "EURUSD=X",
    "Oro": "GC=F",
    "Petróleo WTI": "CL=F",
}
LAST_STATE_PATH = PROJECT_ROOT / "last_state.json"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Watcher educativo de semaforo operativo.")
    parser.add_argument(
        "--skip-refresh",
        action="store_true",
        help="Usa los ultimos datos guardados y no descarga nuevos datos.",
    )
    parser.add_argument(
        "--test-telegram",
        action="store_true",
        help="Envia un mensaje de prueba a Telegram y termina.",
    )
    return parser.parse_args()


def _require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Falta la variable de entorno requerida: {name}")
    return value


def _load_previous_state(path: Path) -> dict[str, Any]:
    raw_state = load_json_file(path, {"checked_at": None, "assets": {}})
    if not isinstance(raw_state, dict):
        return {"checked_at": None, "assets": {}}
    assets = raw_state.get("assets", {})
    if not isinstance(assets, dict):
        assets = {}
    return {
        "checked_at": raw_state.get("checked_at"),
        "assets": assets,
    }


def _save_current_state(path: Path, analyses: list[dict[str, Any]]) -> None:
    save_json_file(
        path,
        {
            "checked_at": datetime.now(timezone.utc).isoformat(),
            "assets": {
                str(item["activo"]): public_state_for_storage(item)
                for item in analyses
            },
        },
    )


def _should_alert(previous: dict[str, Any], current: dict[str, Any]) -> bool:
    entry_trigger = (
        previous.get("entrada") != "HABILITADA"
        and current.get("entrada") == "HABILITADA"
    )
    semaforo_trigger = (
        previous.get("semaforo") != "OPERATIVO"
        and current.get("semaforo") == "OPERATIVO"
    )
    return entry_trigger or semaforo_trigger


def _send_alert(analysis: dict[str, Any]) -> None:
    bot_token = _require_env("TELEGRAM_BOT_TOKEN")
    chat_id = _require_env("TELEGRAM_CHAT_ID")
    send_telegram_message(
        bot_token=bot_token,
        chat_id=chat_id,
        text=build_telegram_message(analysis),
    )


def _send_test_message() -> None:
    bot_token = _require_env("TELEGRAM_BOT_TOKEN")
    chat_id = _require_env("TELEGRAM_CHAT_ID")
    send_telegram_message(
        bot_token=bot_token,
        chat_id=chat_id,
        text=(
            "🚨 Prueba de Telegram del watcher\n\n"
            "Si recibes este mensaje, la configuracion basica del bot funciona.\n\n"
            "Validar manualmente en iFOREX antes de operar. Esta app no ejecuta ordenes reales."
        ),
    )
    print("Mensaje de prueba enviado correctamente.")


def _refresh_market_snapshot() -> None:
    results = refresh_market_data(assets=WATCHED_ASSETS)
    summary = summarize_download_results(results)
    if summary.empty:
        print("No hubo resultados de descarga.")
        return

    print("Resumen de descarga:")
    for row in summary.to_dict(orient="records"):
        print(
            f"- {row['activo']} {row['temporalidad']}: {row['estado']} | {row['mensaje']}"
        )


def main() -> int:
    args = _parse_args()
    if args.test_telegram:
        _send_test_message()
        return 0

    if not args.skip_refresh:
        _refresh_market_snapshot()

    previous_state = _load_previous_state(LAST_STATE_PATH)
    analyses = analizar_activos(WATCHED_ASSETS)
    notification_errors: list[str] = []

    for analysis in analyses:
        asset_name = str(analysis["activo"])
        previous_asset_state = previous_state["assets"].get(
            asset_name,
            {"entrada": "NO HABILITADA", "semaforo": "ESPERAR"},
        )

        if analysis.get("entrada") != "HABILITADA" and analysis.get("semaforo") != "OPERATIVO":
            print(f"{asset_name}: sin alerta activa.")
            continue

        if not _should_alert(previous_asset_state, analysis):
            print(f"{asset_name}: la condicion sigue activa, no se repite alerta.")
            continue

        try:
            _send_alert(analysis)
            print(f"{asset_name}: alerta enviada.")
        except Exception as exc:
            notification_errors.append(f"{asset_name}: {exc}")
            print(f"{asset_name}: no fue posible enviar la alerta.")

    _save_current_state(LAST_STATE_PATH, analyses)
    print(f"Estado actualizado en {LAST_STATE_PATH.name}.")

    if notification_errors:
        print("Errores de notificacion detectados:")
        for error in notification_errors:
            print(f"- {error}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
