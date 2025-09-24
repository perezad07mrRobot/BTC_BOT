# BTC Bot v0.1 — Breakout 10/−5 (Python/ccxt)

Bot de trading long-only sobre Kraken spot (par **XBT/USDT**) con objetivo de +10% y stop -5%. Incluye backtest, paper/live trading con `DRY_RUN` y reporte de profit split.

## Requisitos

- Python 3.10+
- macOS probado en Apple Silicon / Intel
- Paquetes Python: `pip install -r requirements.txt`
  - Para ejecutar `make lint` instala también `pip install ruff`

## Configuración

1. Crea un entorno virtual (recomendado) y ejecuta `pip install -r requirements.txt`.
2. Copia `.env.example` (si no existe, crea uno nuevo) con al menos:
   ```env
   EXCHANGE=kraken
   API_KEY=tu_api_key
   API_SECRET=tu_api_secret
   DRY_RUN=true
   TELEGRAM_BOT_TOKEN=123456:ABC...
   TELEGRAM_CHAT_ID=123456789
   TIMEFRAME=1h
   ```
3. **No otorgues permisos de retiro** a las llaves API. El bot no realiza retiros.
4. Ajusta parámetros opcionales (`LOOKBACK`, `MA_LEN`, etc.) mediante variables de entorno.

## Uso

### Makefile

- `make backtest` → `python btc_bot.py backtest`
- `make trade` → `python btc_bot.py trade`
- `make lint` → `ruff .`

### Comandos directos

- Backtest: `python btc_bot.py backtest`
- Trading (paper por defecto): `python btc_bot.py trade`
  - `DRY_RUN=true` garantiza que **no** se envían órdenes reales.
  - Para operar con dinero real, configura `DRY_RUN=false` bajo tu propio riesgo.

### Reporte semanal de profit split

1. Guarda tus CSV de operaciones en `./reports/` (por ejemplo, exportes del bot).
2. Ejecuta `python profit_split_report.py`.
3. Se genera `reports/profit_split_YYYYMMDD.csv` con el PnL semanal y una propuesta 50% retiro / 50% reinversión. El resumen se imprime en consola.

## Seguridad

- Las llaves API deben tener únicamente permisos de **lectura y trading**, nunca de retiro.
- Protege tu archivo `.env` y evita subirlo a repositorios públicos.
- Limita la IP de acceso cuando el exchange lo permita.

## Logs

`Trader.loop()` registra cada iteración con timestamp, precio, señal (True/False) y estado de la posición (`flat`/`open`). Verifica la consola para monitorear la operación del bot.
