# BTCC High-Leverage AI Trading Agent System

Built for **Crypto (`ETHUSDT`, `BTCUSDT`, `SOLUSDT`)** and **Tokenized RWAs (`XAUUSDT` Gold, `NVDAUSDT`)** on **BTCC** using **20x–500x isolated leverage** with a deep account reserve buffer.

---

## Core Strategy & Agent Architecture

1. **Impulse + Flatline Reversal Detector ([`detector.py`](./detector.py))**
   - **Phase 1 (Impulse):** Detects multi-ATR spikes (`UP_SPIKE`) or liquidation dumps (`DOWN_DUMP`).
   - **Phase 2 (Flatline Exhaustion):** Detects when price compresses into a tight horizontal band (`flatline_range_atr <= 0.95–1.25x ATR`) with decaying candle bodies and volume, failing to continue in the impulse direction.
   - **Phase 3 (Dynamic Isolated Leverage 20x–500x):** Automatically scales leverage so the stop-loss (placed just beyond the impulse/flatline wick) only risks ~50% of the isolated per-trade margin, preventing exchange liquidations while keeping your main account balance untouched.

2. **Active Position Profit-Flatline Stall Monitor ([`detector.py`](./detector.py) & [`execution_engine.py`](./execution_engine.py))**
   - Solves the *"missing profits from not paying attention"* problem:
   - Automatically locks in **50% partial profit at TP1 (38.2% Fib retracement)** and moves the Stop-Loss to **Breakeven**.
   - Continuously watches the open reversal trade—the moment the reversal is in profit (`>= 0.8R`) and **flatlines/stalls for 3 bars**, it immediately fires a **`PROFIT_FLATLINE_STALL`** alert to your phone on Telegram and **auto-closes the position** before ETH bounces back.

3. **Gemini Sentiment & Catalyst Analyst Agent ([`sentiment_agent.py`](./sentiment_agent.py))**
   - Uses **Gemini + Google Search Grounding** alongside live perpetual funding rates to answer one question before taking a reversal:
     - *Was the spike/drop caused by a breaking fundamental/macro catalyst (Fed shock, ETF news, exploit, CPI)?* → **`VETO_CONTINUATION_RISK`** (flatline is a bull/bear flag—do NOT fade it!).
     - *Or was it a pure liquidity sweep / stop hunt with decaying volume and no structural news?* → **`APPROVE_REVERSAL`** (fade it!).

4. **Automated Backtest Researcher ([`backtest_researcher.py`](./backtest_researcher.py))**
   - Sweeps 108+ parameter combinations across historical 1m/5m candles and runs an A/B test comparing **Profit-Flatline Stall Auto-Exit ON vs. OFF**.

---

## Quick Start

### 1. Configure `.env`
```bash
cp btcc_trading_agents/.env.example btcc_trading_agents/.env
```
Fill in:
- `GEMINI_API_KEY`: From Google AI Studio
- `TELEGRAM_BOT_TOKEN`: From `@BotFather` on Telegram
- `TELEGRAM_CHAT_ID`: Your Telegram user/chat ID
- `ACCOUNT_BALANCE_USD=10000.0` & `MARGIN_PER_TRADE_USD=100.0`

### 2. Run the Automated Backtest Researcher
```bash
# Optimize on ETH 5m or 1m candles
./btcc_trading_agents/.venv/bin/python btcc_trading_agents/main.py --mode backtest --symbol ETHUSDT --timeframe 5m --limit 1000

# Optimize on Tokenized Gold (XAUUSDT)
./btcc_trading_agents/.venv/bin/python btcc_trading_agents/main.py --mode backtest --symbol XAUUSDT --timeframe 5m --limit 1000
```

### 3. Run the 24/7 Live Alert & Full-Auto Testing Daemon
```bash
./btcc_trading_agents/.venv/bin/python btcc_trading_agents/main.py --mode daemon --timeframe 5m --poll-seconds 20
```

### 4. Telegram Bot Commands (from your phone)
- `/status` — Check account reserve, daily realized PnL, and active watchlist
- `/positions` — View open high-leverage positions, unrealized ROE %, and breakeven stop status
- `/scan` — Trigger an immediate scan across ETH, BTC, XAU (Gold), and SOL
- `/backtest ETHUSDT 5m` — Run the Automated Backtest Researcher from your phone
