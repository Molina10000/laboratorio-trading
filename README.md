# Laboratorio de Analisis Tecnico Financiero

Aplicacion educativa en Python y Streamlit para analizar `EUR/USD`, `Oro` y `Petroleo WTI` con contexto multi temporal `4H/1H`, semaforo operativo, sesgo, score, motivos y estado de entrada.

La app no ejecuta ordenes reales. Todo resultado debe validarse manualmente antes de operar.

## Que hace

- Dashboard educativo en `Streamlit`.
- Descarga datos desde `yfinance`.
- Guarda OHLCV en `SQLite` local.
- Calcula `EMA 50`, `EMA 200`, `RSI 14` y `ATR 14`.
- Evalua contexto `4H/1H` y construye un semaforo operativo.
- Expone la misma logica para la UI y para el watcher automatico.
- Envia alertas por `Telegram` solo cuando una condicion pasa a operativa.

## Arquitectura

```text
laboratorio-trading/
|-- app.py
|-- watcher.py
|-- requirements.txt
|-- .gitignore
|-- .env.example
|-- last_state.json
|-- core/
|   |-- __init__.py
|   |-- analysis_engine.py
|   |-- data_loader.py
|   |-- indicators.py
|   |-- patterns.py
|   |-- semaforo.py
|   `-- utils.py
`-- .github/
    `-- workflows/
        `-- market_watcher.yml
```

## Requisitos

- Python `3.11`
- Acceso a internet para `yfinance` y Telegram
- Un bot de Telegram
- Un repositorio de GitHub para desplegar y ejecutar el workflow

## Instalacion local

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Ejecutar la app

```powershell
streamlit run app.py
```

La app usa `app.py` como punto de entrada y no depende de rutas absolutas.

## Ejecutar el watcher manualmente

1. Define las variables de entorno:

```powershell
$env:TELEGRAM_BOT_TOKEN="tu_token"
$env:TELEGRAM_CHAT_ID="tu_chat_id"
```

2. Prueba Telegram sin analizar mercado:

```powershell
python watcher.py --test-telegram
```

3. Ejecuta el watcher completo:

```powershell
python watcher.py
```

4. Si quieres reutilizar la base local sin volver a descargar datos:

```powershell
python watcher.py --skip-refresh
```

## Logica de alertas

El watcher analiza:

- `EUR/USD` -> `EURUSD=X`
- `Oro` -> `GC=F`
- `Petroleo WTI` -> `CL=F`

Solo envia alerta cuando ocurre al menos una de estas transiciones:

```python
if estado_anterior["entrada"] != "HABILITADA" and estado_actual["entrada"] == "HABILITADA":
    enviar_alerta()

if estado_anterior["semaforo"] != "OPERATIVO" and estado_actual["semaforo"] == "OPERATIVO":
    enviar_alerta()
```

No alerta solo por sesgo `COMPRA` o `VENTA`.

## Crear el bot de Telegram

1. Abre Telegram y busca `@BotFather`.
2. Ejecuta `/newbot`.
3. Define nombre y username del bot.
4. Copia el token que entrega BotFather.
5. Escribe cualquier mensaje a tu bot.
6. Abre en el navegador:

```text
https://api.telegram.org/botTU_TOKEN/getUpdates
```

7. Busca el `chat.id` de tu conversacion y guardalo como `TELEGRAM_CHAT_ID`.

## GitHub Secrets

En tu repositorio de GitHub ve a:

`Settings` -> `Secrets and variables` -> `Actions` -> `New repository secret`

Crea estos dos secrets:

- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_CHAT_ID`

## GitHub Actions

El workflow `market_watcher.yml`:

- corre cada 15 minutos con `7,22,37,52 * * * *`
- permite ejecucion manual con `workflow_dispatch`
- instala dependencias
- ejecuta `python watcher.py`
- hace commit de `last_state.json` si cambio

### Ejecutarlo manualmente

1. En GitHub abre la pestaña `Actions`.
2. Entra a `Market Watcher`.
3. Pulsa `Run workflow`.

### Cambiar frecuencia

Edita `.github/workflows/market_watcher.yml` y modifica:

```yaml
schedule:
  - cron: "7,22,37,52 * * * *"
```

## Despliegue en Streamlit Community Cloud

1. Sube el proyecto a GitHub.
2. Entra a https://share.streamlit.io/
3. Pulsa `New app`.
4. Selecciona tu repositorio.
5. Branch: la rama principal.
6. Main file path: `app.py`
7. Deploy.

La app descargara datos automaticamente cuando no exista una base local valida.

## Subir el proyecto a GitHub desde cero

Como este workspace no esta inicializado como repo, estos son los pasos exactos:

```powershell
git init
git branch -M main
git add .
git commit -m "Initial trading lab setup"
git remote add origin https://github.com/TU_USUARIO/TU_REPO.git
git push -u origin main
```

Si algun archivo `*.db` ya estaba siendo seguido en otro repo anterior, deja de trackearlo antes del push:

```powershell
git rm --cached analysis_lab.db
git commit -m "Stop tracking local database"
```

## Prueba completa paso a paso

1. Instala dependencias.
2. Ejecuta `streamlit run app.py`.
3. Verifica que el dashboard cargue y que no abra Streamlit al correr `python watcher.py`.
4. Define `TELEGRAM_BOT_TOKEN` y `TELEGRAM_CHAT_ID`.
5. Ejecuta `python watcher.py --test-telegram`.
6. Ejecuta `python watcher.py`.
7. Revisa que `last_state.json` se haya actualizado.
8. Sube el proyecto a GitHub.
9. Crea los dos `Secrets` en GitHub.
10. Ejecuta `Market Watcher` manualmente desde `Actions`.
11. Despliega `app.py` en Streamlit Community Cloud.

## Seguridad

- No subas `.env`.
- No pongas tokens en `app.py` ni en `watcher.py`.
- No imprimas el token completo en logs.
- `analysis_lab.db` queda ignorado por `.gitignore`.

## Recordatorio final

Proyecto educativo. No ejecuta ordenes reales y no debe usarse para automatizar trading en vivo.
