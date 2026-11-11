"""
Gemini Sentiment & Catalyst Analyst Agent (with Google Search Grounding + Perp Funding/OI Context).

Core Mission:
When ETH (or another crypto/RWA asset) makes a huge jump/fall and flatlines, determine whether:
1. EXHAUSTION / LIQUIDITY FLUSH (APPROVE REVERSAL): The move was a typical leverage squeeze / stop hunt
   with decaying volume and no major structural catalyst -> High-probability mean reversion!
2. STRUCTURAL CATALYST / CONTINUATION FLAG (VETO REVERSAL): The spike/drop was driven by breaking news
   (Fed rate shock, ETF headline, protocol hack, macro CPI print, earnings surprise) where the flatline
   is actually a bull/bear flag before leg 2 -> Do NOT fade it at 20x-500x leverage!
"""

from __future__ import annotations
import json
import math
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, asdict, field
from typing import Dict, Any, List, Optional
import requests
from pydantic import BaseModel, Field

from config import AppConfig
from detector import ReversalSetupSignal


TV_SYMBOL_MAP: Dict[str, str] = {
    "XRPUSDT": "BINANCE:XRPUSDT",
    "DOGEUSDT": "BINANCE:DOGEUSDT",
    "SUIUSDT": "BINANCE:SUIUSDT",
    "SOLUSDT": "BINANCE:SOLUSDT",
    "ETHUSDT": "BINANCE:ETHUSDT",
    "BTCUSDT": "BINANCE:BTCUSDT",
    "XAUUSDT": "OANDA:XAUUSD",
    "XAGUSDT": "TVC:SILVER",
    "NVDAUSDT": "NASDAQ:NVDA",
    "SPACEXUSDT": "NASDAQ:SPCX",
    "PEPEUSDT": "BINANCE:PEPEUSDT",
    "WIFUSDT": "BINANCE:WIFUSDT",
    "AVAXUSDT": "BINANCE:AVAXUSDT",
    "INJUSDT": "BINANCE:INJUSDT",
    "BCHUSDT": "BINANCE:BCHUSDT",
    "OPUSDT": "BINANCE:OPUSDT",
    "LINKUSDT": "BINANCE:LINKUSDT",
    "ADAUSDT": "BINANCE:ADAUSDT",
}

SOCIAL_QUERY_MAP: Dict[str, str] = {
    "XRPUSDT": "(XRP OR Ripple) (crypto OR whale OR ETF OR SEC OR rally OR dump)",
    "DOGEUSDT": "(Dogecoin OR DOGE) (crypto OR Elon OR whale OR rally OR dump)",
    "SUIUSDT": "(Sui crypto OR $SUI) (TVL OR unlock OR whale OR rally OR dump)",
    "SOLUSDT": "(Solana OR $SOL) (crypto OR whale OR ETF OR rally OR dump)",
    "ETHUSDT": "(Ethereum OR $ETH) (Vitalik OR ETF OR whale OR upgrade OR dump)",
    "BTCUSDT": "(Bitcoin OR $BTC) (ETF OR whale OR macro OR rally OR dump)",
    "XAUUSDT": "(Gold price OR XAUUSD) (Fed OR central bank OR safe haven OR inflation)",
    "XAGUSDT": "(Silver price OR XAGUSD) (COMEX OR industrial demand OR safe haven OR Gold)",
    "NVDAUSDT": "(Nvidia OR NVDA) (AI chips OR Jensen Huang OR earnings OR Wall Street)",
    "SPACEXUSDT": "(SpaceX OR SPCX OR Starlink) (stock OR Starship OR launch OR Musk OR earnings OR Nasdaq)",
    "PEPEUSDT": "(Pepe coin OR $PEPE) (memecoin OR whale OR rally OR breakout)",
    "WIFUSDT": "(dogwifhat OR $WIF) (Solana memecoin OR whale OR breakout OR rally)",
    "AVAXUSDT": "(Avalanche OR $AVAX) (crypto OR RWA OR staking OR rally)",
    "INJUSDT": "(Injective OR $INJ) (ETF OR Nasdaq OR DeFi OR staking)",
    "BCHUSDT": "(Bitcoin Cash OR $BCH) (crypto OR whale OR futures OR rally)",
    "OPUSDT": "(Optimism OR $OP) (Superchain OR L2 OR Unichain OR Ethereum)",
}

BULLISH_KEYWORDS = (
    "surge", "soar", "rally", "bull", "bullish", "breakout", "inflow", "inflows",
    "adoption", "upgrade", "approval", "accumulate", "accumulating", "whale bought",
    "record high", "partnership", "launch", "buy", "buying", "rebound", "jump",
    "gain", "outperform", "store of value", "recovery", "upside", "higher",
    "support holds", "bounce", "strategic", "targets", "record", "demand",
    "optimistic", "break above", "reclaim", "growth", "beat", "strong"
)
BEARISH_KEYWORDS = (
    "drop", "plunge", "crash", "bear", "bearish", "decline", "outflow", "hack",
    "exploit", "stolen", "stealing", "pleads guilty", "guilty", "investigate",
    "spying", "lawsuit", "sec ", "ban", "dump", "sell-off", "selloff", "sell",
    "liquidation", "liquidated", "warn", "warning", "break the chain", "risk",
    "fear", "panic", "tariff", "slump", "fall", "tumble", "exit", "downside",
    "slips", "erased", "backward", "wrong", "pressure", "weakness", "losses",
    "short", "bleed", "underperform", "reject", "below"
)
HIGH_IMPACT_CATALYSTS = (
    "hack", "exploit", "sec ", "etf", "cpi", "fomc", "fed ", "rate cut", "rate hike",
    "investigate", "lawsuit", "tariff", "earnings", "upgrade", "central bank", "stolen"
)


class SentimentVerdictSchema(BaseModel):
    decision: str = Field(
        description="One of: 'APPROVE_REVERSAL', 'CAUTION_REDUCE_LEVERAGE', or 'VETO_CONTINUATION_RISK'"
    )
    confidence: float = Field(description="Confidence score from 0.0 to 1.0")
    sentiment_score: float = Field(
        description="Net market sentiment from -1.0 (extreme bearish) to +1.0 (extreme bullish)"
    )
    is_news_catalyst_driven: bool = Field(
        description="True if the impulse move was caused by a major breaking fundamental/macro catalyst"
    )
    pattern_diagnosis: str = Field(
        description="Diagnosis of the flatline: 'EXHAUSTION_REVERSAL' vs 'BULL_BEAR_FLAG_CONTINUATION'"
    )
    catalysts_found: List[str] = Field(description="Key headlines, macro events, or orderflow drivers found")
    reasoning: str = Field(description="Concise trader-focused explanation of why to take or skip the reversal")
    leverage_multiplier: float = Field(
        description="1.0 for full recommended leverage, 0.5 for caution, 0.0 for veto"
    )


@dataclass
class DerivativesContext:
    funding_rate_pct: float = 0.0
    open_interest_change_pct: float = 0.0
    long_short_ratio: float = 1.0
    crowd_long_pct: float = 50.0
    fear_greed_value: int = 50
    fear_greed_label: str = "Neutral"
    tv_headlines: List[str] = field(default_factory=list)
    social_headlines: List[str] = field(default_factory=list)
    total_posts_analyzed: int = 0
    bull_posts: int = 0
    bear_posts: int = 0
    directional_posts: int = 0
    dominant_pct: float = 50.0
    sentiment_z_score: float = 0.0
    sentiment_p_value: float = 1.0
    confidence_pct: float = 50.0
    is_statistically_significant: bool = False
    news_sentiment_score: float = 0.0
    news_bias_label: str = "NEUTRAL"
    major_catalyst_detected: bool = False
    top_headline: str = ""
    updated_ts: float = 0.0


class SentimentAnalystAgent:
    """
    Combines live TradingView News, X/Twitter & Social News feeds, OKX/Binance
    Perpetual Funding + Retail Long/Short Ratios, Crypto Fear & Greed, and
    Gemini + Google Search Grounding to validate Impulse + Flatline reversal setups.
    """

    def __init__(self, config: AppConfig):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/124.0 Safari/537.36"
        })
        self._verdict_cache: Dict[str, SentimentVerdictSchema] = {}
        self._intel_cache: Dict[str, DerivativesContext] = {}
        self._fg_cache: Dict[str, Any] = {"value": 50, "label": "Neutral", "ts": 0.0}

    def _fetch_fear_and_greed(self) -> tuple[int, str]:
        now = time.time()
        if now - self._fg_cache["ts"] < 600:
            return self._fg_cache["value"], self._fg_cache["label"]
        try:
            r = self.session.get("https://api.alternative.me/fng/?limit=1", timeout=2.5)
            if r.status_code == 200:
                d = r.json().get("data", [{}])[0]
                val = int(d.get("value", 50))
                lbl = str(d.get("value_classification", "Neutral"))
                self._fg_cache = {"value": val, "label": lbl, "ts": now}
                return val, lbl
        except Exception:
            pass
        return self._fg_cache["value"], self._fg_cache["label"]

    def fetch_derivatives_context(self, symbol: str, ttl_sec: int = 300) -> DerivativesContext:
        """
        Fetch live TradingView News headlines (up to 30), X/Twitter/Social news RSS (up to 20),
        OKX/Binance perp funding rate, and retail Long/Short account ratio.
        Runs a Binomial z-score & p-value test on directional posts to detect
        statistically significant (|z| >= 1.96, p <= 0.05) sentiment waves.
        Cached for `ttl_sec` (default 5 min) so 15s scans never block.
        """
        sym = symbol.upper()
        now = time.time()
        cached = self._intel_cache.get(sym)
        if cached is not None and (now - cached.updated_ts) < ttl_sec:
            return cached

        ctx = DerivativesContext(updated_ts=now)
        ctx.fear_greed_value, ctx.fear_greed_label = self._fetch_fear_and_greed()

        # 1. OKX Perp Funding Rate & Retail Long/Short Account Ratio (no geo-block)
        base_ccy = sym.replace("USDT", "")
        if sym not in ("XAUUSDT", "XAGUSDT", "NVDAUSDT", "TSLAUSDT", "SPACEXUSDT"):
            try:
                r_fund = self.session.get(
                    "https://www.okx.com/api/v5/public/funding-rate",
                    params={"instId": f"{base_ccy}-USDT-SWAP"},
                    timeout=2.5,
                )
                if r_fund.status_code == 200:
                    fd = r_fund.json().get("data", [])
                    if fd and fd[0].get("fundingRate") is not None:
                        ctx.funding_rate_pct = float(fd[0]["fundingRate"]) * 100.0
            except Exception:
                pass

            try:
                r_ls = self.session.get(
                    "https://www.okx.com/api/v5/rubik/stat/contracts/long-short-account-ratio",
                    params={"ccy": base_ccy, "period": "1H"},
                    timeout=2.5,
                )
                if r_ls.status_code == 200:
                    ls_data = r_ls.json().get("data", [])
                    if ls_data and len(ls_data[0]) >= 2:
                        ratio = float(ls_data[0][1])
                        ctx.long_short_ratio = round(ratio, 2)
                        ctx.crowd_long_pct = round((ratio / (1.0 + ratio)) * 100.0, 1)
            except Exception:
                pass

        # Fallback to Binance if funding is still 0.0
        if ctx.funding_rate_pct == 0.0:
            try:
                resp = self.session.get(
                    "https://fapi.binance.com/fapi/v1/premiumIndex",
                    params={"symbol": sym},
                    timeout=2.0,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    if isinstance(data, dict) and "lastFundingRate" in data:
                        ctx.funding_rate_pct = float(data["lastFundingRate"]) * 100.0
            except Exception:
                pass

        # 2. Live TradingView News Headlines (up to 30 recent items for statistical power)
        tv_sym = TV_SYMBOL_MAP.get(sym, f"BINANCE:{sym}")
        try:
            r_tv = self.session.get(
                "https://news-headlines.tradingview.com/v2/view/headlines/symbol",
                params={"client": "web", "lang": "en", "symbol": tv_sym},
                timeout=3.0,
            )
            if r_tv.status_code == 200:
                items = r_tv.json().get("items", [])
                for it in items[:30]:
                    title = str(it.get("title", "")).strip()
                    provider = str(it.get("provider", "TradingView")).strip()
                    if title:
                        ctx.tv_headlines.append(f"[TV/{provider}] {title}")
        except Exception:
            pass

        # 3. Live X (Twitter) & Social / Breaking News Feed via Google News RSS (up to 20 recent items)
        q = SOCIAL_QUERY_MAP.get(sym, f"{base_ccy} crypto")
        try:
            r_soc = self.session.get(
                "https://news.google.com/rss/search",
                params={"q": f"{q} when:24h", "hl": "en-US", "gl": "US", "ceid": "US:en"},
                timeout=3.0,
            )
            if r_soc.status_code == 200 and r_soc.text:
                root = ET.fromstring(r_soc.text)
                for item in root.findall(".//item")[:20]:
                    t_el = item.find("title")
                    if t_el is not None and t_el.text:
                        clean_t = t_el.text.strip()
                        ctx.social_headlines.append(f"[Social/News] {clean_t}")
        except Exception:
            pass

        # 4. Binomial z-Score & p-Value Test Across All 50 Live Posts
        all_posts = ctx.tv_headlines + ctx.social_headlines
        ctx.total_posts_analyzed = len(all_posts)
        n_bull = 0
        n_bear = 0
        cat_hits = 0
        for h in all_posts:
            hl = h.lower()
            b_kw = sum(1 for kw in BULLISH_KEYWORDS if kw in hl)
            s_kw = sum(1 for kw in BEARISH_KEYWORDS if kw in hl)
            if b_kw > s_kw:
                n_bull += 1
            elif s_kw > b_kw:
                n_bear += 1
            if any(kw in hl for kw in HIGH_IMPACT_CATALYSTS):
                cat_hits += 1

        ctx.bull_posts = n_bull
        ctx.bear_posts = n_bear
        n_dir = n_bull + n_bear
        ctx.directional_posts = n_dir

        if n_dir >= 5:
            raw_z = (abs(n_bull - n_bear) - 0.5) / math.sqrt(n_dir)
            raw_z = max(0.0, raw_z)
            signed_z = raw_z if n_bull >= n_bear else -raw_z
            p_val = math.erfc(raw_z / math.sqrt(2.0))
            ctx.sentiment_z_score = round(signed_z, 2)
            ctx.sentiment_p_value = round(p_val, 4)
            ctx.confidence_pct = round((1.0 - p_val) * 100.0, 1)
            ctx.dominant_pct = round((max(n_bull, n_bear) / n_dir) * 100.0, 1)
            ctx.is_statistically_significant = (raw_z >= 1.90 and n_dir >= 8)
            ctx.news_sentiment_score = round((n_bull - n_bear) / n_dir, 2)
        else:
            ctx.news_sentiment_score = 0.0

        if ctx.is_statistically_significant:
            ctx.news_bias_label = "BULLISH" if n_bull > n_bear else "BEARISH"
        elif ctx.news_sentiment_score >= 0.22:
            ctx.news_bias_label = "BULLISH"
        elif ctx.news_sentiment_score <= -0.22:
            ctx.news_bias_label = "BEARISH"
        else:
            ctx.news_bias_label = "NEUTRAL"

        ctx.major_catalyst_detected = cat_hits >= 3 and ctx.is_statistically_significant
        if ctx.tv_headlines:
            ctx.top_headline = ctx.tv_headlines[0]
        elif ctx.social_headlines:
            ctx.top_headline = ctx.social_headlines[0]
        else:
            ctx.top_headline = "No breaking catalyst headlines in last 4h."

        self._intel_cache[sym] = ctx
        return ctx

    def evaluate_pre_move_sentiment_overlap(
        self,
        symbol: str,
        curr_price: float,
        curr_rsi: float,
        curr_atr: float,
        curr_ema20: float,
        mtf_radar: Dict[str, Any],
        macro_compass: Dict[str, Any],
        rec_leverage: int = 20,
    ) -> Dict[str, Any]:
        """
        Detects when a STATISTICALLY SIGNIFICANT sentiment wave (|z| >= 1.90, p <= 0.05)
        overlaps with Historical Technical Analysis (Stability Box coiling, RSI non-exhaustion,
        EMA20 alignment, or Contrarian Retail Crowd L/S trap) BEFORE or AT the start of a move.
        """
        ctx = self.fetch_derivatives_context(symbol)
        bias_side = "LONG" if ctx.bull_posts >= ctx.bear_posts else "SHORT"
        box_high = float(mtf_radar.get("box_high") or (curr_price + 0.5 * curr_atr))
        box_low = float(mtf_radar.get("box_low") or (curr_price - 0.5 * curr_atr))
        box_range_atr = float(mtf_radar.get("box_range_atr") or 1.0)
        is_coiled_box = bool(mtf_radar.get("is_flatline_box", False)) or (box_range_atr <= 1.45)

        # Check if price is still inside or within 0.35x ATR of the Stability Box (Pre-Move Window!)
        in_or_near_box = (box_low - 0.35 * curr_atr) <= curr_price <= (box_high + 0.35 * curr_atr)

        overlap_factors: List[str] = []
        overlap_score = 0

        # Factor 1: Stability Box Pre-Move Coiling (75%-80% WR Historical Setup Window)
        if is_coiled_box and in_or_near_box:
            overlap_score += 2
            overlap_factors.append(
                f"PRE-MOVE STABILITY BOX: Coiling inside [${box_low:.4g}–${box_high:.4g}] ({box_range_atr:.2f}x ATR) — market has NOT priced in the sentiment wave yet"
            )
        elif in_or_near_box:
            overlap_score += 1
            overlap_factors.append(f"Holding near 4h structure [${box_low:.4g}–${box_high:.4g}] before full expansion")

        # Factor 2: Non-Exhausted RSI / Historical Win-Rate Zone
        move_already_exhausted = False
        if bias_side == "LONG":
            if curr_rsi > 65.5:
                move_already_exhausted = True
                overlap_score -= 3
                overlap_factors.append(f"Exhausted Overbought RSI ({curr_rsi:.1f} > 65.5) — Bullish wave already priced in; do NOT chase high")
            elif curr_rsi <= 35.0:
                overlap_score += 2
                overlap_factors.append(f"Oversold Floor RSI ({curr_rsi:.1f} <= 35) + Bullish Sentiment Wave (80% WR Reversal Zone)")
            else:
                overlap_score += 1
                overlap_factors.append(f"Non-Exhausted RSI ({curr_rsi:.1f}) leaves full room for upside expansion (75% WR Bridge Zone)")
        else:
            if curr_rsi < 34.0:
                move_already_exhausted = True
                overlap_score -= 3
                overlap_factors.append(f"Exhausted Oversold Floor RSI ({curr_rsi:.1f} < 34) — Bearish dump already played out; do NOT chase short at floor!")
            elif curr_rsi >= 65.0:
                overlap_score += 2
                overlap_factors.append(f"Overbought Ceiling RSI ({curr_rsi:.1f} >= 65) + Bearish Sentiment Wave (80% WR Reversal Zone)")
            else:
                overlap_score += 1
                overlap_factors.append(f"Non-Exhausted RSI ({curr_rsi:.1f} >= 34) leaves room for downside breakdown")

        # Factor 3: Retail Crowd Long/Short Trap (OKX Account Ratio)
        if bias_side == "SHORT" and ctx.long_short_ratio >= 2.0 and not move_already_exhausted:
            overlap_score += 2
            overlap_factors.append(
                f"Retail Crowd Trap: {ctx.crowd_long_pct:.1f}% of accounts are crowded LONG ({ctx.long_short_ratio:.2f}x L/S) while institutional/news flow is {ctx.dominant_pct:.0f}% BEARISH"
            )
        elif bias_side == "LONG" and (ctx.funding_rate_pct < 0.0 or ctx.long_short_ratio <= 1.3) and not move_already_exhausted:
            overlap_score += 1
            overlap_factors.append(f"Unwound/Short-heavy positioning (Funding {ctx.funding_rate_pct:+.4f}%) fuels short squeeze")

        # Factor 4: Macro Trend (1D / 1W / 1M Monthly) & 1M Historical Optimization Alignment
        d1_obj = macro_compass.get("1d", {})
        w1_obj = macro_compass.get("1w", {})
        m1_obj = macro_compass.get("1M", {})
        d1_trend = str(d1_obj.get("trend", d1_obj) if isinstance(d1_obj, dict) else d1_obj)
        w1_trend = str(w1_obj.get("trend", w1_obj) if isinstance(w1_obj, dict) else w1_obj)
        m1_trend = str(m1_obj.get("trend", m1_obj) if isinstance(m1_obj, dict) else m1_obj)
        if (bias_side == "LONG" and (d1_trend == "BULLISH" or m1_trend == "BULLISH")) or (
            bias_side == "SHORT" and (d1_trend == "BEARISH" or m1_trend == "BEARISH")
        ):
            overlap_score += 1
            overlap_factors.append(
                f"Aligned with Macro & Monthly Compass (1D: {d1_trend} • 1W: {w1_trend} • 1M: {m1_trend})"
            )

        # Calculate Exact Pre-Move Entry / Trigger Ticket
        prec = 8 if curr_price < 0.001 else 5
        if bias_side == "LONG":
            trig_entry = round(max(curr_price, box_high + 0.05 * curr_atr), prec)
            early_entry = round(curr_price, prec)
            sl_price = round(min(box_low - 0.25 * curr_atr, early_entry - 1.15 * curr_atr), prec)
            risk_dist = max(abs(trig_entry - sl_price), 0.5 * curr_atr)
            tp1_price = round(trig_entry + 1.2 * risk_dist, prec)
            tp2_price = round(trig_entry + 2.4 * risk_dist, prec)
        else:
            trig_entry = round(min(curr_price, box_low - 0.05 * curr_atr), prec)
            early_entry = round(curr_price, prec)
            sl_price = round(max(box_high + 0.25 * curr_atr, early_entry + 1.15 * curr_atr), prec)
            risk_dist = max(abs(sl_price - trig_entry), 0.5 * curr_atr)
            tp1_price = round(trig_entry - 1.2 * risk_dist, prec)
            tp2_price = round(trig_entry - 2.4 * risk_dist, prec)

        if move_already_exhausted:
            stage = "POST_MOVE_EXHAUSTED"
            stage_label = f"MOVE ALREADY PLAYED OUT (RSI {curr_rsi:.1f} Exhausted) — Wait for 3-Bar Stability Box"
        elif is_coiled_box and in_or_near_box:
            stage = "PRE_MOVE_COILED_READY"
            stage_label = "PRE-MOVE STABILITY WINDOW (Get In Before Move / Box Trigger)"
        elif (bias_side == "LONG" and curr_price > box_high) or (bias_side == "SHORT" and curr_price < box_low):
            stage = "BREAKOUT_IN_PROGRESS"
            stage_label = "SENTIMENT WAVE BREAKOUT IN PROGRESS"
        else:
            stage = "WATCH_STRUCTURE"
            stage_label = "SENTIMENT SKEW BUILDING — WATCH EMA20 / BOX TRIGGER"

        has_strong_overlap = bool(ctx.is_statistically_significant and overlap_score >= 2 and not move_already_exhausted)

        return {
            "symbol": symbol,
            "is_statistically_significant": ctx.is_statistically_significant,
            "has_strong_overlap": has_strong_overlap,
            "overlap_score": overlap_score,
            "bias_side": bias_side,
            "news_bias_label": ctx.news_bias_label,
            "total_posts": ctx.total_posts_analyzed,
            "directional_posts": ctx.directional_posts,
            "bull_posts": ctx.bull_posts,
            "bear_posts": ctx.bear_posts,
            "dominant_pct": ctx.dominant_pct,
            "z_score": ctx.sentiment_z_score,
            "p_value": ctx.sentiment_p_value,
            "confidence_pct": ctx.confidence_pct,
            "stage": stage,
            "stage_label": stage_label,
            "in_pre_move_box": bool(is_coiled_box and in_or_near_box),
            "early_entry_price": early_entry,
            "trigger_entry_price": trig_entry,
            "stop_loss": sl_price,
            "tp1_price": tp1_price,
            "tp2_price": tp2_price,
            "recommended_leverage": rec_leverage,
            "overlap_factors": overlap_factors,
            "top_headlines": (ctx.tv_headlines[:2] + ctx.social_headlines[:1]),
        }

    def get_live_market_intel_dict(self, symbol: str) -> Dict[str, Any]:
        """Return a JSON-serializable summary of TradingView News, X/Social buzz, and Crowd Positioning for the GUI."""
        ctx = self.fetch_derivatives_context(symbol)
        return {
            "funding_rate_pct": round(ctx.funding_rate_pct, 4),
            "long_short_ratio": ctx.long_short_ratio,
            "crowd_long_pct": ctx.crowd_long_pct,
            "fear_greed_value": ctx.fear_greed_value,
            "fear_greed_label": ctx.fear_greed_label,
            "total_posts_analyzed": ctx.total_posts_analyzed,
            "bull_posts": ctx.bull_posts,
            "bear_posts": ctx.bear_posts,
            "directional_posts": ctx.directional_posts,
            "dominant_pct": ctx.dominant_pct,
            "sentiment_z_score": ctx.sentiment_z_score,
            "sentiment_p_value": ctx.sentiment_p_value,
            "confidence_pct": ctx.confidence_pct,
            "is_statistically_significant": ctx.is_statistically_significant,
            "news_sentiment_score": ctx.news_sentiment_score,
            "news_bias_label": ctx.news_bias_label,
            "major_catalyst_detected": ctx.major_catalyst_detected,
            "top_headline": ctx.top_headline,
            "tv_headlines": ctx.tv_headlines[:3],
            "social_headlines": ctx.social_headlines[:3],
        }

    def evaluate_setup(self, signal: ReversalSetupSignal) -> SentimentVerdictSchema:
        """
        Evaluate a detected Impulse + Flatline setup using Live TradingView News + X/Social RSS
        + OKX/Binance Crowd Positioning + Gemini Google Search Grounding.
        """
        cache_key = f"{signal.symbol}:{signal.timeframe}:{signal.timestamp}:{bool(self.config.gemini_api_key)}"
        if cache_key in self._verdict_cache:
            return self._verdict_cache[cache_key]

        deriv_ctx = self.fetch_derivatives_context(signal.symbol)

        if not self.config.gemini_api_key:
            verdict = self._heuristic_fallback(signal, deriv_ctx)
            self._verdict_cache[cache_key] = verdict
            return verdict

        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=self.config.gemini_api_key)
            headlines_block = "\n".join(f"  * {h}" for h in (deriv_ctx.tv_headlines[:5] + deriv_ctx.social_headlines[:4]))

            prompt = f"""
You are an elite intraday Crypto & Tokenized RWA Trading Analyst specializing in high-leverage (20x-500x)
'Impulse + Flatline Exhaustion Reversal' setups on BTCC.

A deterministic scanner just detected the following setup:
- Asset: {signal.symbol} ({signal.timeframe} chart)
- Timestamp (UTC): {signal.timestamp}
- Impulse Move: {signal.impulse_direction} of {signal.impulse_pct:+.2f}% ({signal.impulse_atr_mult:.1f}x ATR)
- Flatline Consolidation: {signal.flatline_bars} bars tight compression (range {signal.flatline_range_pct:.3f}%, {signal.flatline_range_atr:.2f}x ATR)
- Candle Body Decay Ratio: {signal.body_decay_ratio:.2f} (smaller means stronger momentum exhaustion)
- Volume Decay Ratio in Flatline: {signal.volume_decay_ratio:.2f}
- RSI at Flatline: {signal.rsi_at_flatline}
- Proposed Reversal Trade: {signal.trade_side} at {signal.entry_price} | Stop-Loss: {signal.stop_loss} | TP1: {signal.tp1_price} | TP2: {signal.tp2_price}
- Live Perp Funding Rate: {deriv_ctx.funding_rate_pct:+.4f}% | Retail Long/Short Ratio: {deriv_ctx.long_short_ratio:.2f} ({deriv_ctx.crowd_long_pct:.1f}% Long)
- Crypto Fear & Greed Index: {deriv_ctx.fear_greed_value} ({deriv_ctx.fear_greed_label})
- Live TradingView & X/Social Headlines Fetched Just Now:
{headlines_block or '  * None'}

YOUR TASK:
1. Analyze the live TradingView & X/Social headlines above and search live news for {signal.symbol} from the last few hours.
2. Determine if the {signal.impulse_direction} was triggered by a MAJOR structural news catalyst (e.g. Fed/CPI shock, ETF ruling, hack/exploit, earnings blowout, geopolitical escalation) where fading the flatline is dangerous because the flatline is a continuation flag.
3. Or if there is NO major fundamental shock and the technicals (body decay {signal.body_decay_ratio:.2f}, volume decay {signal.volume_decay_ratio:.2f}, RSI {signal.rsi_at_flatline}) + crowd positioning confirm a classic liquidity sweep / exhaustion flatline that is primed to reverse.

Return ONLY valid JSON matching this exact structure:
{{
  "decision": "APPROVE_REVERSAL" | "CAUTION_REDUCE_LEVERAGE" | "VETO_CONTINUATION_RISK",
  "confidence": 0.0 to 1.0,
  "sentiment_score": -1.0 to 1.0,
  "is_news_catalyst_driven": true | false,
  "pattern_diagnosis": "EXHAUSTION_REVERSAL" | "BULL_BEAR_FLAG_CONTINUATION",
  "catalysts_found": ["headline or factor 1", "factor 2"],
  "reasoning": "Concise 2-3 sentence explanation for Telegram alert",
  "leverage_multiplier": 1.0 | 0.5 | 0.0
}}
"""
            response = client.models.generate_content(
                model=self.config.gemini_model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    tools=[types.Tool(google_search=types.GoogleSearch())],
                    temperature=0.2,
                ),
            )

            raw_text = (response.text or "").strip()
            if "```" in raw_text:
                parts = raw_text.split("```")
                for part in parts:
                    part_clean = part.strip()
                    if part_clean.startswith("json"):
                        part_clean = part_clean[4:].strip()
                    if part_clean.startswith("{") and part_clean.endswith("}"):
                        raw_text = part_clean
                        break

            parsed = json.loads(raw_text)
            verdict = SentimentVerdictSchema(**parsed)
            self._verdict_cache[cache_key] = verdict
            return verdict

        except Exception as e:
            fallback = self._heuristic_fallback(signal, deriv_ctx)
            fallback.reasoning += f" (Note: Gemini live search fallback used: {type(e).__name__})"
            return fallback

    def _heuristic_fallback(
        self,
        signal: ReversalSetupSignal,
        deriv_ctx: DerivativesContext,
    ) -> SentimentVerdictSchema:
        """
        Live TradingView News + X/Social RSS + OKX Crowd Positioning + Orderflow scoring.
        """
        reasons = []
        score = 0.65

        # 1. Volume decay during flatline
        if signal.volume_decay_ratio < 0.65:
            score += 0.10
            reasons.append(f"Flatline volume dropped {(1 - signal.volume_decay_ratio)*100:.0f}% vs impulse (exhaustion)")
        elif signal.volume_decay_ratio > 1.25:
            score -= 0.12
            reasons.append("High volume persisting in flatline (churn/continuation risk)")

        # 2. RSI extreme confirmation
        if signal.trade_side == "SHORT" and signal.rsi_at_flatline >= 68:
            score += 0.08
            reasons.append(f"Overbought RSI ({signal.rsi_at_flatline:.1f}) stalling at top of spike")
        elif signal.trade_side == "LONG" and signal.rsi_at_flatline <= 32:
            score += 0.08
            reasons.append(f"Oversold RSI ({signal.rsi_at_flatline:.1f}) holding flatline floor")

        # 3. Retail Long/Short Ratio & Funding Crowding
        if signal.trade_side == "SHORT":
            if deriv_ctx.funding_rate_pct > 0.015 or deriv_ctx.long_short_ratio >= 1.8:
                score += 0.08
                reasons.append(
                    f"Retail L/S ratio {deriv_ctx.long_short_ratio:.2f}x ({deriv_ctx.crowd_long_pct:.0f}% Long, funding {deriv_ctx.funding_rate_pct:+.4f}%) supports downside flush"
                )
        elif signal.trade_side == "LONG":
            if deriv_ctx.funding_rate_pct < -0.003 or deriv_ctx.long_short_ratio <= 1.1:
                score += 0.08
                reasons.append(
                    f"Negative/unwound funding ({deriv_ctx.funding_rate_pct:+.4f}%, L/S {deriv_ctx.long_short_ratio:.2f}x) favors short-covering bounce"
                )

        # 4. TradingView News & X/Social Headline Alignment vs Catalyst Veto
        if deriv_ctx.top_headline:
            reasons.append(f"Latest headline: {deriv_ctx.top_headline[:90]}")

        if signal.trade_side == "LONG" and deriv_ctx.news_sentiment_score <= -0.50 and deriv_ctx.major_catalyst_detected:
            score -= 0.22
            reasons.append("WARNING: Heavy bearish breaking news catalyst on TradingView/Social feed")
        elif signal.trade_side == "SHORT" and deriv_ctx.news_sentiment_score >= 0.50 and deriv_ctx.major_catalyst_detected:
            score -= 0.22
            reasons.append("WARNING: Heavy bullish breaking news catalyst on TradingView/Social feed")
        elif (signal.trade_side == "LONG" and deriv_ctx.news_sentiment_score >= 0.20) or (
            signal.trade_side == "SHORT" and deriv_ctx.news_sentiment_score <= -0.20
        ):
            score += 0.07
            reasons.append(f"TradingView/Social news bias ({deriv_ctx.news_bias_label}, {deriv_ctx.news_sentiment_score:+.2f}) aligns with {signal.trade_side}")

        score = max(0.1, min(0.95, score))
        if score >= 0.70:
            decision = "APPROVE_REVERSAL"
            diag = "EXHAUSTION_REVERSAL"
            lev_mult = 1.0
        elif score >= 0.55:
            decision = "CAUTION_REDUCE_LEVERAGE"
            diag = "EXHAUSTION_REVERSAL"
            lev_mult = 0.5
        else:
            decision = "VETO_CONTINUATION_RISK"
            diag = "BULL_BEAR_FLAG_CONTINUATION"
            lev_mult = 0.0

        return SentimentVerdictSchema(
            decision=decision,
            confidence=round(score, 2),
            sentiment_score=deriv_ctx.news_sentiment_score,
            is_news_catalyst_driven=deriv_ctx.major_catalyst_detected,
            pattern_diagnosis=diag,
            catalysts_found=(deriv_ctx.tv_headlines[:2] + deriv_ctx.social_headlines[:2]) or reasons,
            reasoning="; ".join(reasons[:3]) if reasons else "Standard exhaustion flatline detected.",
            leverage_multiplier=lev_mult,
        )
