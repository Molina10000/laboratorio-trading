from __future__ import annotations


EDUCATIONAL_WARNING = (
    "Esto es educativo, verificar manualmente en el broker antes de operar."
)


def calculate_risk(
    account_capital: float,
    risk_percent: float,
    entry_price: float,
    stop_loss: float,
    take_profit: float | None = None,
) -> dict[str, float | str]:
    if account_capital <= 0:
        raise ValueError("El capital de la cuenta debe ser mayor que cero.")
    if risk_percent <= 0:
        raise ValueError("El porcentaje de riesgo debe ser mayor que cero.")
    if entry_price <= 0 or stop_loss <= 0:
        raise ValueError("El precio de entrada y el stop loss deben ser mayores que cero.")
    if entry_price == stop_loss:
        raise ValueError("La entrada y el stop loss no pueden ser iguales.")

    maximum_loss = account_capital * (risk_percent / 100)
    unit_risk = abs(entry_price - stop_loss)
    position_size = maximum_loss / unit_risk
    reward_risk_ratio: float | None = None

    if take_profit is not None and take_profit > 0 and take_profit != entry_price:
        reward_risk_ratio = round(abs(take_profit - entry_price) / unit_risk, 2)

    return {
        "perdida_maxima_estimada": round(maximum_loss, 2),
        "riesgo_por_unidad": round(unit_risk, 5 if unit_risk < 10 else 2),
        "tamano_posicion_aproximado": round(position_size, 4),
        "relacion_riesgo_beneficio": reward_risk_ratio,
        "advertencia": EDUCATIONAL_WARNING,
    }
