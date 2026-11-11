"""
Configuration and Asset Profiles for the BTCC AI Trading Agent System.
Supports Crypto (ETH, BTC, SOL) and Tokenized RWAs (Gold/XAU, Silver/XAG, NVDA, TSLA)
with high-leverage (20x-500x) isolated margin and deep account reserve protection.
"""

from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import Dict, List
from dotenv import load_dotenv

load_dotenv()


@dataclass
class StrategyParams:
    """Parameters defining the 'Impulse + Flatline -> Reversal' setup and Profit-Stall exit."""
    timeframe: str = "4h"                  # "1m", "5m", "15m", "1h", "4h", or "1d"
    atr_period: int = 14

    # Phase 1: Huge Jump or Fall (Impulse)
    impulse_window_bars: int = 6           # Optimized: 6 bars (30m on 5m chart, or 6m on 1m chart)
    impulse_min_atr_mult: float = 1.8      # Optimized: Impulse net move >= 1.8x ATR
    impulse_min_pct: float = 0.42          # Minimum absolute % move for the impulse on ETH 5m

    # Phase 2: Flatline (Exhaustion Consolidation)
    flatline_bars: int = 2                 # Optimized: 2-3 bars of horizontal stall after impulse
    flatline_max_retracement_ratio: float = 0.38  # Flatline stays near the extreme (within top/bottom 38% of impulse range)
    flatline_max_range_atr: float = 1.15   # Total high-low range of the flatline window <= 1.15x ATR
    flatline_body_decay_ratio: float = 0.55 # Average candle body in flatline vs impulse candle body

    # Phase 3: Reversal Entry, Stop-Loss & Profit-Stall Exit
    stop_loss_atr_buffer: float = 0.35     # Tight invalidation buffer just outside the flatline high/low
    tp1_retracement_ratio: float = 0.382   # Take partial/target 1 at 38.2% retracement of the impulse move
    tp2_retracement_ratio: float = 0.618   # Full target at 61.8% retracement of the impulse move
    exit_on_profit_flatline: bool = True   # Auto-close/alert if trade is in profit and stalls for `exit_flatline_bars`
    exit_flatline_bars: int = 3            # Bars of tight stall while in profit before locking in gains
    min_profit_r_for_stall_exit: float = 0.8  # Minimum R-multiple in profit before flatline-stall exit activates
    max_hold_bars: int = 24                # Time stop if reversal doesn't play out


@dataclass
class LeverageRiskConfig:
    """
    Models the user's BTCC isolated-margin style:
    - Configurable micro-margin ($5/trade) or standard ($20-$100/trade) isolated positions
    - Dynamic leverage (3x to 500x) scaled so Stop-Loss only risks ~25% of the isolated trade margin
    """
    total_account_balance_usd: float = float(os.getenv("ACCOUNT_BALANCE_USD", "100.0"))
    margin_per_trade_usd: float = float(os.getenv("MARGIN_PER_TRADE_USD", "5.0"))    # $5 minimum margin per trade
    default_leverage: int = int(os.getenv("DEFAULT_LEVERAGE", "5"))                  # 5x lowest multiplier ceiling
    max_leverage: int = 500
    min_leverage: int = int(os.getenv("MIN_LEVERAGE", "2"))                          # 2x minimum leverage
    target_sl_margin_risk_pct: float = float(os.getenv("TARGET_SL_MARGIN_RISK_PCT", "15.0"))  # ~15% of $5 = $0.75 risk at SL
    auto_adjust_leverage_to_stop: bool = True  # Dynamically scale leverage so SL triggers well before isolated liquidation
    # Portfolio limits (risk guard, not an exchange limit). Re-tested for Krypton V3 on 2019-2026 data:
    # longs carry most of the edge, so allow 3 concurrent longs but only 1 short, 4 positions in total.
    max_concurrent_positions: int = int(os.getenv("MAX_CONCURRENT_POSITIONS", "4"))
    max_long_positions: int = int(os.getenv("MAX_LONG_POSITIONS", "3"))
    max_short_positions: int = int(os.getenv("MAX_SHORT_POSITIONS", "1"))
    max_same_side_positions: int = 2           # Legacy symmetric correlation guard (used only if per-side limits are absent)
    daily_max_loss_usd: float = float(os.getenv("DAILY_MAX_LOSS_USD", "15.0"))       # Kill-switch if daily losses exceed 3x trade margin
    btcc_taker_fee_pct: float = 0.045          # 0.045% per side typical BTCC VIP/standard perp taker fee
    btcc_maker_fee_pct: float = 0.020          # 0.020% per side post-only / maker limit fee
    prefer_maker_limit_entries: bool = True    # Route entries as Maker/Pullback limits to cut fee+spread drag
    engine_version: str = os.getenv("ENGINE_VERSION", "KRYPTON_V3").upper()  # Active alpha engine: KRYPTON_V3 (default) | CARBON_V2


@dataclass
class AssetProfile:
    symbol: str                  # Canonical symbol, e.g., "ETHUSDT"
    btcc_symbol: str             # BTCC contract name, e.g., "ETHUSDT"
    category: str                # "crypto" or "rwa"
    binance_symbol: str          # Public futures/spot feed symbol
    yfinance_ticker: str         # Yahoo Finance fallback for tokenized RWAs (Gold, Silver, NVDA, TSLA)
    max_btcc_leverage: int       # e.g., ETH 500x, Gold 150x-200x, Stocks 50x
    default_impulse_pct: float   # Asset-specific minimum impulse % on 5m
    min_lot_label: str = "0.01"  # Minimum contract lot on BTCC
    min_notional_usd: float = 15.0  # Approximate minimum notional value ($) per order


DEFAULT_ASSETS: Dict[str, AssetProfile] = {
    "ETHUSDT": AssetProfile(
        symbol="ETHUSDT",
        btcc_symbol="ETHUSDT",
        category="crypto",
        binance_symbol="ETHUSDT",
        yfinance_ticker="ETH-USD",
        max_btcc_leverage=500,
        default_impulse_pct=0.70,
        min_lot_label="0.01 ETH",
        min_notional_usd=21.0,
    ),
    "BTCUSDT": AssetProfile(
        symbol="BTCUSDT",
        btcc_symbol="BTCUSDT",
        category="crypto",
        binance_symbol="BTCUSDT",
        yfinance_ticker="BTC-USD",
        max_btcc_leverage=500,
        default_impulse_pct=0.55,
        min_lot_label="0.001 BTC",
        min_notional_usd=76.0,
    ),
    "SOLUSDT": AssetProfile(
        symbol="SOLUSDT",
        btcc_symbol="SOLUSDT",
        category="crypto",
        binance_symbol="SOLUSDT",
        yfinance_ticker="SOL-USD",
        max_btcc_leverage=200,
        default_impulse_pct=1.00,
        min_lot_label="0.1 SOL",
        min_notional_usd=9.0,
    ),
    "XAUUSDT": AssetProfile(
        symbol="XAUUSDT",
        btcc_symbol="GOLDUSDT",
        category="rwa",
        binance_symbol="PAXGUSDT",  # Tokenized gold 24/7 feed + GC=F
        yfinance_ticker="GC=F",
        max_btcc_leverage=200,
        default_impulse_pct=0.35,
        min_lot_label="0.01 oz",
        min_notional_usd=45.0,
    ),
    "NVDAUSDT": AssetProfile(
        symbol="NVDAUSDT",
        btcc_symbol="NVDAUSDT",
        category="rwa",
        binance_symbol="",
        yfinance_ticker="NVDA",
        max_btcc_leverage=50,
        default_impulse_pct=0.85,
        min_lot_label="0.1 NVDA",
        min_notional_usd=18.0,
    ),
    "XRPUSDT": AssetProfile(
        symbol="XRPUSDT",
        btcc_symbol="XRPUSDT",
        category="crypto",
        binance_symbol="XRPUSDT",
        yfinance_ticker="XRP-USD",
        max_btcc_leverage=200,
        default_impulse_pct=0.45,
        min_lot_label="1 XRP",
        min_notional_usd=1.5,
    ),
    "DOGEUSDT": AssetProfile(
        symbol="DOGEUSDT",
        btcc_symbol="DOGEUSDT",
        category="crypto",
        binance_symbol="DOGEUSDT",
        yfinance_ticker="DOGE-USD",
        max_btcc_leverage=200,
        default_impulse_pct=0.50,
        min_lot_label="10 DOGE",
        min_notional_usd=1.2,
    ),
    "SUIUSDT": AssetProfile(
        symbol="SUIUSDT",
        btcc_symbol="SUIUSDT",
        category="crypto",
        binance_symbol="SUIUSDT",
        yfinance_ticker="SUI20947-USD",
        max_btcc_leverage=100,
        default_impulse_pct=0.60,
        min_lot_label="1 SUI",
        min_notional_usd=1.2,
    ),
    "XAGUSDT": AssetProfile(
        symbol="XAGUSDT",
        btcc_symbol="SILVERUSDT",
        category="rwa",
        binance_symbol="",
        yfinance_ticker="SI=F",
        max_btcc_leverage=150,
        default_impulse_pct=0.45,
        min_lot_label="0.1 oz",
        min_notional_usd=10.0,
    ),
    "PEPEUSDT": AssetProfile(
        symbol="PEPEUSDT",
        btcc_symbol="1000PEPEUSDT",
        category="crypto",
        binance_symbol="PEPEUSDT",
        yfinance_ticker="PEPE24478-USD",
        max_btcc_leverage=100,
        default_impulse_pct=0.70,
        min_lot_label="100k PEPE",
        min_notional_usd=5.0,
    ),
    "AVAXUSDT": AssetProfile(
        symbol="AVAXUSDT",
        btcc_symbol="AVAXUSDT",
        category="crypto",
        binance_symbol="AVAXUSDT",
        yfinance_ticker="AVAX-USD",
        max_btcc_leverage=150,
        default_impulse_pct=0.60,
        min_lot_label="0.1 AVAX",
        min_notional_usd=5.0,
    ),
    "WIFUSDT": AssetProfile(
        symbol="WIFUSDT",
        btcc_symbol="WIFUSDT",
        category="crypto",
        binance_symbol="WIFUSDT",
        yfinance_ticker="WIF-USD",
        max_btcc_leverage=100,
        default_impulse_pct=0.70,
        min_lot_label="1 WIF",
        min_notional_usd=5.0,
    ),
    "INJUSDT": AssetProfile(
        symbol="INJUSDT",
        btcc_symbol="INJUSDT",
        category="crypto",
        binance_symbol="INJUSDT",
        yfinance_ticker="INJ-USD",
        max_btcc_leverage=100,
        default_impulse_pct=0.65,
        min_lot_label="0.1 INJ",
        min_notional_usd=5.0,
    ),
}


@dataclass
class AppConfig:
    gemini_api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    gemini_model: str = field(default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-3.5-flash"))
    telegram_bot_token: str = field(default_factory=lambda: os.getenv("TELEGRAM_BOT_TOKEN", ""))
    telegram_chat_id: str = field(default_factory=lambda: os.getenv("TELEGRAM_CHAT_ID", ""))
    btcc_api_key: str = field(default_factory=lambda: os.getenv("BTCC_API_KEY", ""))
    btcc_api_secret: str = field(default_factory=lambda: os.getenv("BTCC_API_SECRET", ""))
    live_trading_enabled: bool = field(
        default_factory=lambda: os.getenv("LIVE_TRADING_ENABLED", "false").lower() == "true"
    )
    watchlist: List[str] = field(
        default_factory=lambda: [
            "XRPUSDT",
            "DOGEUSDT",
            "SUIUSDT",
            "SOLUSDT",
            "ETHUSDT",
            "BTCUSDT",
            "XAUUSDT",
            "XAGUSDT",
            "PEPEUSDT",
            "AVAXUSDT",
            "WIFUSDT",
            "INJUSDT",
            "NVDAUSDT",
        ]
    )
    strategy: StrategyParams = field(default_factory=StrategyParams)
    risk: LeverageRiskConfig = field(default_factory=LeverageRiskConfig)
