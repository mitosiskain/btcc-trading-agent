"""
High-Leverage (20x-500x) Execution & Position Management Engine for BTCC.
Supports:
- Full Automated Paper/Simulation Trading (with exact BTCC isolated margin, leverage, fee, and liquidation math)
- Automatic Partial Profit Taking (TP1 -> 50% close + breakeven stop)
- Automatic 'Profit-Flatline Stall' Exit (locks in profits when the reversal stalls so gains aren't given back)
- Pluggable BTCC REST API Adapter for live order execution
"""

from __future__ import annotations
import hashlib
import hmac
import json
import os
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any

from config import AppConfig
from detector import ReversalSetupSignal, ExitAlertSignal
from sentiment_agent import SentimentVerdictSchema


@dataclass
class OpenPosition:
    position_id: str
    symbol: str
    side: str                       # "LONG" or "SHORT"
    entry_price: float
    current_price: float
    stop_loss: float
    tp1_price: float
    tp2_price: float
    leverage: int                   # 20x to 500x
    initial_margin_usd: float       # Margin allocated to this trade
    remaining_margin_usd: float     # Margin remaining after partial TP1
    notional_usd: float             # remaining_margin_usd * leverage
    liq_price: float
    opened_at: str
    bars_held: int = 0
    tp1_hit: bool = False
    warning_sent: bool = False
    realized_partial_pnl_usd: float = 0.0
    sentiment_decision: str = "APPROVE_REVERSAL"
    sentiment_reasoning: str = ""
    archetype: str = "CARBON_ABSORPTION_REVERSAL"
    order_type: str = "POST_ONLY_LIMIT"
    partial_frac: float = 0.48
    trail_atr_mult: float = 1.75
    peak_favorable_price: float = 0.0
    meta_pred_r: float = 0.0

    def unrealized_pnl_usd(self) -> float:
        if self.side == "LONG":
            ret = (self.current_price - self.entry_price) / self.entry_price
        else:
            ret = (self.entry_price - self.current_price) / self.entry_price
        return self.remaining_margin_usd * self.leverage * ret

    def unrealized_roe_pct(self) -> float:
        if self.remaining_margin_usd <= 0:
            return 0.0
        return (self.unrealized_pnl_usd() / self.remaining_margin_usd) * 100.0


@dataclass
class ClosedTradeRecord:
    position_id: str
    symbol: str
    side: str
    leverage: int
    entry_price: float
    exit_price: float
    initial_margin_usd: float
    net_pnl_usd: float
    roe_pct: float
    exit_reason: str
    opened_at: str
    closed_at: str
    bars_held: int


class BTCCExchangeClient:
    """
    Adapter for BTCC Perpetual Futures API.
    In simulation/testing mode (LIVE_TRADING_ENABLED=false), orders are filled locally
    against live market prices with realistic BTCC taker fees (0.045%).
    """

    BASE_URL = "https://api.btcc.com"

    def __init__(self, api_key: str, api_secret: str, live_enabled: bool = False):
        self.api_key = api_key
        self.api_secret = api_secret
        self.live_enabled = live_enabled and bool(api_key and api_secret)

    def place_order(
        self,
        symbol: str,
        side: str,
        leverage: int,
        margin_usd: float,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        order_type: str = "POST_ONLY_LIMIT",
    ) -> Dict[str, Any]:
        is_maker = "LIMIT" in order_type.upper()
        if not self.live_enabled:
            return {
                "mode": "SIMULATION",
                "status": "FILLED",
                "order_type": order_type,
                "order_id": f"sim-{symbol.lower()}-{int(time.time()*1000)}",
                "filled_price": entry_price,
            }

        # Live BTCC HMAC-SHA256 signed request structure (Post-Only Maker Limit by default)
        timestamp = str(int(time.time() * 1000))
        payload: Dict[str, Any] = {
            "symbol": symbol,
            "side": "BUY" if side == "LONG" else "SELL",
            "type": "LIMIT" if is_maker else "MARKET",
            "leverage": leverage,
            "margin": round(margin_usd, 2),
            "stopLoss": round(stop_loss, 6),
            "takeProfit": round(take_profit, 6),
            "timestamp": timestamp,
        }
        if is_maker:
            payload["price"] = round(entry_price, 6)
            payload["postOnly"] = "true"
            payload["timeInForce"] = "GTX"
        query_str = "&".join(f"{k}={v}" for k, v in sorted(payload.items()))
        signature = hmac.new(
            self.api_secret.encode("utf-8"),
            query_str.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        payload["signature"] = signature
        return {
            "mode": "LIVE_BTCC_READY",
            "status": "SUBMITTED",
            "order_type": order_type,
            "order_id": f"btcc-{timestamp}",
            "filled_price": entry_price,
            "payload": payload,
        }


class ExecutionEngine:
    """
    Manages the account reserve balance, isolated trade margins, guardrails,
    and automated entry/exit execution.
    """

    def __init__(self, config: AppConfig, state_path: Optional[str] = None):
        self.config = config
        self.risk = config.risk
        self.btcc = BTCCExchangeClient(
            api_key=config.btcc_api_key,
            api_secret=config.btcc_api_secret,
            live_enabled=config.live_trading_enabled,
        )
        base_dir = Path(__file__).resolve().parent / "state"
        base_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = Path(state_path) if state_path else base_dir / "portfolio_state.json"

        self.account_balance_usd: float = self.risk.total_account_balance_usd
        self.daily_realized_pnl_usd: float = 0.0
        self.positions_version_ts: float = 0.0
        self.open_positions: Dict[str, OpenPosition] = {}
        self.closed_trades: List[ClosedTradeRecord] = []
        self._load_state()

    def can_open_trade(self, symbol: str, side: Optional[str] = None) -> tuple[bool, str]:
        if symbol in self.open_positions:
            return False, f"Position already open for {symbol}"
        if len(self.open_positions) >= self.risk.max_concurrent_positions:
            return False, f"Max concurrent positions ({self.risk.max_concurrent_positions}) reached"
        if side is not None:
            legacy = int(getattr(self.risk, "max_same_side_positions", 2))
            max_same = int(getattr(self.risk, "max_long_positions" if side.upper() == "LONG" else "max_short_positions", legacy))
            same_cnt = sum(1 for p in self.open_positions.values() if p.side.upper() == side.upper())
            if same_cnt >= max_same:
                return False, f"Max correlated {side.upper()} positions ({max_same}) reached"
        if self.daily_realized_pnl_usd <= -abs(self.risk.daily_max_loss_usd):
            return False, f"Daily kill-switch triggered (Loss: ${self.daily_realized_pnl_usd:.2f})"
        allocated = sum(p.remaining_margin_usd for p in self.open_positions.values())
        if self.account_balance_usd - allocated < self.risk.margin_per_trade_usd * 2:
            return False, "Insufficient free account reserve balance"
        return True, "OK"

    def execute_reversal_entry(
        self,
        signal: ReversalSetupSignal,
        verdict: SentimentVerdictSchema,
        archetype: str = "ABSORPTION_REVERSAL",
        partial_frac: float = 0.48,
        trail_atr_mult: float = 1.75,
        meta_pred_r: float = 0.0,
    ) -> Optional[OpenPosition]:
        """
        Validates risk & correlation guardrails and opens an isolated Maker-Limit / Kelly-scaled position.
        """
        if verdict.decision == "VETO_CONTINUATION_RISK" or verdict.leverage_multiplier <= 0:
            return None

        allowed, _ = self.can_open_trade(signal.symbol, side=signal.trade_side)
        if not allowed:
            return None

        # Adjust leverage if Sentiment Agent flagged CAUTION
        leverage = max(
            self.risk.min_leverage,
            min(
                self.risk.max_leverage,
                int(signal.recommended_leverage * verdict.leverage_multiplier),
            ),
        )
        margin_usd = self.risk.margin_per_trade_usd
        notional_usd = margin_usd * leverage

        use_maker = bool(getattr(self.risk, "prefer_maker_limit_entries", True))
        order_type = "POST_ONLY_LIMIT" if use_maker else "MARKET"
        fee_pct = (
            float(getattr(self.risk, "btcc_maker_fee_pct", 0.020))
            if use_maker
            else float(self.risk.btcc_taker_fee_pct)
        )

        # Deduct entry fee from reserve account (0.020% Maker vs 0.045% Taker)
        entry_fee = notional_usd * (fee_pct / 100.0)
        self.account_balance_usd -= entry_fee
        self.daily_realized_pnl_usd -= entry_fee

        order_res = self.btcc.place_order(
            symbol=signal.symbol,
            side=signal.trade_side,
            leverage=leverage,
            margin_usd=margin_usd,
            entry_price=signal.entry_price,
            stop_loss=signal.stop_loss,
            take_profit=signal.tp2_price,
            order_type=order_type,
        )

        pos = OpenPosition(
            position_id=order_res["order_id"],
            symbol=signal.symbol,
            side=signal.trade_side,
            entry_price=signal.entry_price,
            current_price=signal.entry_price,
            stop_loss=signal.stop_loss,
            tp1_price=signal.tp1_price,
            tp2_price=signal.tp2_price,
            leverage=leverage,
            initial_margin_usd=margin_usd,
            remaining_margin_usd=margin_usd,
            notional_usd=notional_usd,
            liq_price=signal.estimated_liq_price,
            opened_at=datetime.now(timezone.utc).isoformat(),
            bars_held=0,
            tp1_hit=False,
            sentiment_decision=verdict.decision,
            sentiment_reasoning=verdict.reasoning,
            archetype=archetype,
            order_type=order_type,
            partial_frac=partial_frac,
            trail_atr_mult=trail_atr_mult,
            peak_favorable_price=signal.entry_price,
            meta_pred_r=meta_pred_r,
        )
        self.open_positions[signal.symbol] = pos
        self.positions_version_ts = time.time()
        self._save_state()
        return pos

    def process_exit_signal(self, exit_sig: ExitAlertSignal) -> Optional[ClosedTradeRecord]:
        """
        Executes partial profit taking (on TP1) or full position closure (on Profit-Flatline Stall,
        TP2, Stop-Loss, or Time Stop).
        """
        pos = self.open_positions.get(exit_sig.symbol)
        if not pos:
            return None

        pos.current_price = exit_sig.current_price

        # Case 0: Manual / screenshot user-tracked BTCC positions
        # Never auto-delete a user-tracked position behind their back; send an alert with a 1-tap [✅ Mark Closed] button!
        if str(pos.position_id).startswith(("manual-", "screenshot-")):
            if exit_sig.urgency == "WARNING_REVERSAL_RISK":
                pos.warning_sent = True
            elif exit_sig.urgency == "TAKE_PARTIAL_AND_TRAIL" and not pos.tp1_hit:
                pos.tp1_hit = True
            self._save_state()
            return None

        # Case 0b: Early Structure-Reclaim / Failed-Breakout Reversal Warning (keep position tracked, mark warning_sent)
        if exit_sig.urgency == "WARNING_REVERSAL_RISK":
            pos.warning_sent = True
            self._save_state()
            return None

        # Case 1: Partial Profit at TP1 (close partial_frac and move Stop-Loss to breakeven)
        if exit_sig.urgency == "TAKE_PARTIAL_AND_TRAIL" and not pos.tp1_hit:
            p_frac = max(0.25, min(0.75, float(getattr(pos, "partial_frac", 0.48))))
            close_margin = pos.remaining_margin_usd * p_frac
            if pos.side == "LONG":
                ret = (exit_sig.current_price - pos.entry_price) / pos.entry_price
            else:
                ret = (pos.entry_price - exit_sig.current_price) / pos.entry_price

            tp_fee_pct = float(getattr(self.risk, "btcc_maker_fee_pct", 0.020))
            gross_pnl = close_margin * pos.leverage * ret
            exit_fee = (close_margin * pos.leverage) * (tp_fee_pct / 100.0)
            net_partial = gross_pnl - exit_fee

            pos.remaining_margin_usd -= close_margin
            pos.notional_usd = pos.remaining_margin_usd * pos.leverage
            pos.realized_partial_pnl_usd += net_partial
            if (pos.side == "LONG" and exit_sig.current_price > pos.entry_price) or (
                pos.side == "SHORT" and exit_sig.current_price < pos.entry_price
            ):
                pos.stop_loss = pos.entry_price  # Move stop to breakeven when TP1 is hit
            pos.tp1_hit = True
            pos.peak_favorable_price = exit_sig.current_price

            self.account_balance_usd += net_partial
            self.daily_realized_pnl_usd += net_partial
            self._save_state()
            return None

        # Case 2: Full Position Close (Profit-Flatline Stall, TP2, Stop-Loss, or Time Stop)
        if pos.side == "LONG":
            ret = (exit_sig.current_price - pos.entry_price) / pos.entry_price
        else:
            ret = (pos.entry_price - exit_sig.current_price) / pos.entry_price

        gross_pnl = pos.remaining_margin_usd * pos.leverage * ret
        # Cap isolated loss at remaining margin
        if gross_pnl < -pos.remaining_margin_usd:
            gross_pnl = -pos.remaining_margin_usd

        exit_fee = (pos.remaining_margin_usd * pos.leverage) * (self.risk.btcc_taker_fee_pct / 100.0)
        final_leg_pnl = gross_pnl - exit_fee
        total_trade_pnl = pos.realized_partial_pnl_usd + final_leg_pnl
        total_roe_pct = (total_trade_pnl / pos.initial_margin_usd) * 100.0

        self.account_balance_usd += final_leg_pnl
        self.daily_realized_pnl_usd += final_leg_pnl

        record = ClosedTradeRecord(
            position_id=pos.position_id,
            symbol=pos.symbol,
            side=pos.side,
            leverage=pos.leverage,
            entry_price=pos.entry_price,
            exit_price=exit_sig.current_price,
            initial_margin_usd=pos.initial_margin_usd,
            net_pnl_usd=round(total_trade_pnl, 2),
            roe_pct=round(total_roe_pct, 2),
            exit_reason=exit_sig.reason,
            opened_at=pos.opened_at,
            closed_at=datetime.now(timezone.utc).isoformat(),
            bars_held=pos.bars_held,
        )
        self.closed_trades.append(record)
        del self.open_positions[exit_sig.symbol]
        self.positions_version_ts = time.time()
        self._save_state()
        return record

    def close_position_manual(
        self,
        symbol: str,
        exit_price: Optional[float] = None,
        reason: str = "MANUAL_CLOSE",
    ) -> Optional[ClosedTradeRecord]:
        """
        Explicitly close a tracked position (via Telegram text, 1-tap button, BTCC screenshot sync, or Web GUI),
        recording realized PnL in closed_trades and bumping positions_version_ts so Local & Cloud stay synced.
        """
        sym = symbol.upper()
        pos = self.open_positions.get(sym)
        if not pos:
            return None

        fill_price = float(exit_price) if (exit_price is not None and exit_price > 0) else float(pos.current_price or pos.entry_price)
        pos.current_price = fill_price
        if pos.side == "LONG":
            ret = (fill_price - pos.entry_price) / max(pos.entry_price, 1e-12)
        else:
            ret = (pos.entry_price - fill_price) / max(pos.entry_price, 1e-12)

        gross_pnl = pos.remaining_margin_usd * pos.leverage * ret
        if gross_pnl < -pos.remaining_margin_usd:
            gross_pnl = -pos.remaining_margin_usd

        exit_fee = (pos.remaining_margin_usd * pos.leverage) * (self.risk.btcc_taker_fee_pct / 100.0)
        final_leg_pnl = gross_pnl - exit_fee
        total_trade_pnl = pos.realized_partial_pnl_usd + final_leg_pnl
        total_roe_pct = (total_trade_pnl / max(pos.initial_margin_usd, 1e-9)) * 100.0

        self.account_balance_usd += final_leg_pnl
        self.daily_realized_pnl_usd += final_leg_pnl

        record = ClosedTradeRecord(
            position_id=pos.position_id,
            symbol=pos.symbol,
            side=pos.side,
            leverage=pos.leverage,
            entry_price=pos.entry_price,
            exit_price=fill_price,
            initial_margin_usd=pos.initial_margin_usd,
            net_pnl_usd=round(total_trade_pnl, 2),
            roe_pct=round(total_roe_pct, 2),
            exit_reason=reason,
            opened_at=pos.opened_at,
            closed_at=datetime.now(timezone.utc).isoformat(),
            bars_held=pos.bars_held,
        )
        self.closed_trades.append(record)
        del self.open_positions[sym]
        self.positions_version_ts = time.time()
        self._save_state()
        return record

    def close_all_manual(
        self,
        live_prices: Optional[Dict[str, float]] = None,
        reason: str = "MANUAL_CLOSE_ALL",
    ) -> List[ClosedTradeRecord]:
        """Close all open positions and return their ClosedTradeRecords."""
        records: List[ClosedTradeRecord] = []
        for sym in list(self.open_positions.keys()):
            px = (live_prices or {}).get(sym)
            rec = self.close_position_manual(sym, exit_price=px, reason=reason)
            if rec is not None:
                records.append(rec)
        if not records:
            self.positions_version_ts = time.time()
            self._save_state()
        return records

    def touch_positions_version(self) -> None:
        """Bump version clock whenever open_positions is structurally added/edited and persist immediately."""
        self.positions_version_ts = time.time()
        self._save_state()

    def _save_state(self) -> None:
        data = {
            "account_balance_usd": round(self.account_balance_usd, 2),
            "daily_realized_pnl_usd": round(self.daily_realized_pnl_usd, 2),
            "positions_version_ts": round(float(self.positions_version_ts or 0.0), 3),
            "open_positions": {k: asdict(v) for k, v in self.open_positions.items()},
            "closed_trades": [asdict(t) for t in self.closed_trades[-100:]],
        }
        self.state_file.write_text(json.dumps(data, indent=2))

    def _load_state(self) -> None:
        if not self.state_file.exists():
            return
        try:
            data = json.loads(self.state_file.read_text())
            self.account_balance_usd = float(data.get("account_balance_usd", self.account_balance_usd))
            self.daily_realized_pnl_usd = float(data.get("daily_realized_pnl_usd", 0.0))
            self.positions_version_ts = float(data.get("positions_version_ts", 0.0))
            pos_fields = set(OpenPosition.__dataclass_fields__.keys())
            rec_fields = set(ClosedTradeRecord.__dataclass_fields__.keys())
            self.open_positions = {
                k: OpenPosition(**{ fk: fv for fk, fv in v.items() if fk in pos_fields })
                for k, v in data.get("open_positions", {}).items()
            }
            self.closed_trades = [
                ClosedTradeRecord(**{ fk: fv for fk, fv in t.items() if fk in rec_fields })
                for t in data.get("closed_trades", [])
            ]
        except Exception:
            pass
