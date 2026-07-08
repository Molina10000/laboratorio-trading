from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import requests


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def dedupe_keep_order(messages: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for message in messages:
        cleaned = str(message).strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        ordered.append(cleaned)
    return ordered


def load_json_file(path: str | Path, default: Any) -> Any:
    file_path = Path(path)
    if not file_path.exists():
        return default

    try:
        with file_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return default


def save_json_file(path: str | Path, data: Any) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")


def bullet_lines(items: Iterable[str], fallback: str) -> str:
    values = dedupe_keep_order(items)
    if not values:
        return f"- {fallback}"
    return "\n".join(f"- {value}" for value in values)


def build_telegram_message(analysis: dict[str, Any]) -> str:
    motivos = bullet_lines(analysis.get("motivos", []), "Sin motivos registrados.")
    advertencias = bullet_lines(
        analysis.get("advertencias", []),
        "Sin advertencias adicionales.",
    )
    return "\n".join(
        [
            "🚨 Semaforo operativo detectado",
            "",
            f"Activo: {analysis.get('activo', 'N/D')}",
            f"Precio: {analysis.get('precio_texto', analysis.get('precio', 'N/D'))}",
            f"Semaforo: {analysis.get('semaforo', 'N/D')}",
            f"Sesgo: {analysis.get('sesgo', 'N/D')}",
            f"Entrada: {analysis.get('entrada', 'N/D')}",
            f"Score: {analysis.get('score', 'N/D')}",
            f"4H: {analysis.get('contexto_4h', 'N/D')}",
            f"1H: {analysis.get('contexto_1h', 'N/D')}",
            f"Alineacion: {analysis.get('alineacion', 'N/D')}",
            "Motivos:",
            motivos,
            "Advertencias:",
            advertencias,
            "",
            "Validar manualmente en iFOREX antes de operar. Esta app no ejecuta ordenes reales.",
        ]
    )


def send_telegram_message(
    bot_token: str,
    chat_id: str,
    text: str,
    *,
    timeout: int = 20,
) -> None:
    response = requests.post(
        f"https://api.telegram.org/bot{bot_token}/sendMessage",
        json={
            "chat_id": chat_id,
            "text": text,
            "disable_web_page_preview": True,
        },
        timeout=timeout,
    )
    response.raise_for_status()


def public_state_for_storage(analysis: dict[str, Any]) -> dict[str, Any]:
    return {
        "activo": analysis.get("activo"),
        "symbol": analysis.get("symbol"),
        "precio": analysis.get("precio"),
        "precio_texto": analysis.get("precio_texto"),
        "temporalidad": analysis.get("temporalidad"),
        "tendencia": analysis.get("tendencia"),
        "contexto_4h": analysis.get("contexto_4h"),
        "contexto_1h": analysis.get("contexto_1h"),
        "alineacion": analysis.get("alineacion"),
        "semaforo": analysis.get("semaforo"),
        "sesgo": analysis.get("sesgo"),
        "entrada": analysis.get("entrada"),
        "score": analysis.get("score"),
        "motivos": list(analysis.get("motivos", []) or []),
        "advertencias": list(analysis.get("advertencias", []) or []),
        "timestamp": analysis.get("timestamp"),
    }
