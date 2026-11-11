# BTCC High-Leverage AI Trading Agent System

Built for **Crypto (`ETHUSDT`, `BTCUSDT`, `SOLUSDT`)** and **Tokenized RWAs (`XAUUSDT` Gold, `NVDAUSDT`)** on **BTCC** using **20x–500x isolated leverage** with a deep account reserve buffer.

---

## Live Engine: Krypton V3 (default since 2026-11-11)

The live scanner, GitHub cloud runner, web dashboard and Telegram bot now run **Krypton V3**
([`krypton_engine.py`](./krypton_engine.py)), a volatility-normalised trend-following engine:

- **Entry:** first candle *close* beyond the prior 20-bar (4h) / 160-bar (1h) Donchian channel, only in the direction of the 50/200 EMA trend.
- **Risk:** initial stop = 2 ATR (4h) / 3 ATR (1h) = 1R; leverage sized from the stop as before.
- **Exit:** no fixed take-profit. Chandelier trail (the bot tells you when to raise your stop) + close when a candle closes through the 10-bar (4h) / 80-bar (1h) exit channel.
- **Portfolio rule:** max 4 open positions: up to 3 longs and 1 short (breakouts beyond the limit are shown as `SKIP`).
  Override with `MAX_CONCURRENT_POSITIONS` / `MAX_LONG_POSITIONS` / `MAX_SHORT_POSITIONS`. (The head-to-head benchmark keeps the
  original 3-open / 2-same-side rule so Krypton, Carbon and Argon stay directly comparable.)
- **Evidence:** [`benchmark_krypton.py`](./benchmark_krypton.py) re-uses the exact Argon/Carbon simulator and OOS windows.
  Hold-out: 4h PF 1.29 / +$592 (Carbon V2: PF 1.26 / +$247), 1h PF 1.80 / +$821 (Carbon V2: PF 0.82 / −$224).
  Expect a ~25–35% win rate: the edge comes from a few large trend winners.

Switch back to the previous engine with `ENGINE_VERSION=CARBON_V2` (env var / GitHub workflow).

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
