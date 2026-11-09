"""
Telegram Alert & Two-Way Command Bot for the BTCC AI Trading Agent System.
Delivers instant phone push notifications for:
1. 'Impulse + Flatline -> Reversal' Entries (validated by Gemini Sentiment Analyst)
2. 'Profit-Flatline Stall' Urgent Exit Alerts (so you never miss taking profits when ETH stalls!)
3. Interactive commands: /status, /positions, /scan, /backtest <SYMBOL> <TF>, /close <SYMBOL>
"""

from __future__ import annotations
import json
from pathlib import Path
import subprocess
from typing import Optional, Callable, Dict, Any
import requests

from config import AppConfig
from detector import ReversalSetupSignal, ExitAlertSignal
from execution_engine import OpenPosition, ClosedTradeRecord
from sentiment_agent import SentimentVerdictSchema


class TelegramNotifier:
    """Handles formatting and sending Telegram alerts + polling interactive commands."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.token = config.telegram_bot_token
        self.chat_id = config.telegram_chat_id
        self.session = requests.Session()
        self.last_update_id: int = 0

    @property
    def is_configured(self) -> bool:
        return bool(self.token and self.chat_id)

    @property
    def enabled(self) -> bool:
        return self.is_configured

    def _api_call(self, method: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Resilient Telegram Bot API call with IPv4 curl fallback if requests hits ConnectionResetError."""
        if not self.token:
            return {"ok": False, "description": "No bot token"}
        url = f"https://api.telegram.org/bot{self.token}/{method}"
        try:
            if payload:
                resp = self.session.post(url, json=payload, timeout=6)
            else:
                resp = self.session.get(url, timeout=6)
            return resp.json()
        except Exception:
            # Fallback to curl -4 (IPv4) which avoids IPv6 connection resets on Linux
            cmd = ["curl", "-4", "-s", "-X", "POST" if payload else "GET", url]
            if payload:
                cmd += ["-H", "Content-Type: application/json", "-d", json.dumps(payload)]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=8)
            if res.stdout:
                try:
                    return json.loads(res.stdout)
                except Exception:
                    pass
            return {"ok": False, "description": res.stderr or "Telegram API unreachable"}

    def _save_chat_id_to_env(self, chat_id: str) -> None:
        self.chat_id = str(chat_id)
        self.config.telegram_chat_id = str(chat_id)
        env_path = Path(__file__).resolve().parent / ".env"
        env_lines: Dict[str, str] = {}
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if "=" in line and not line.strip().startswith("#"):
                    k, v = line.split("=", 1)
                    env_lines[k.strip()] = v.strip()
        env_lines["TELEGRAM_BOT_TOKEN"] = self.token
        env_lines["TELEGRAM_CHAT_ID"] = self.chat_id
        env_path.write_text("\n".join(f"{k}={v}" for k, v in env_lines.items()) + "\n")

    def send_message(self, text: str) -> bool:
        """Send message to Telegram (and print to console for local visibility)."""
        print(f"\n================ TELEGRAM ALERT ================\n{text}\n================================================")
        if not self.is_configured:
            return False
        data = self._api_call(
            "sendMessage",
            {
                "chat_id": self.chat_id,
                "text": text,
                "parse_mode": "Markdown",
                "disable_web_page_preview": True,
            },
        )
        return bool(data.get("ok"))

    def send_entry_setup_alert(
        self,
        signal: ReversalSetupSignal,
        verdict: SentimentVerdictSchema,
        opened_pos: Optional[OpenPosition] = None,
    ) -> None:
        icon = "🟢" if signal.trade_side == "LONG" else "🔴"
        arrow = "📈" if signal.trade_side == "LONG" else "📉"
        conf_pct = int(round(verdict.confidence * 100))

        msg = (
            f"{icon} *{signal.symbol} — {signal.trade_side} {arrow} ({conf_pct}% Confidence)*\n"
            f"• *Safe Entry*: `{signal.entry_price}` _({signal.timeframe.upper()})_\n"
            f"• *Take-Profit*: TP1 `{signal.tp1_price}` | TP2 `{signal.tp2_price}`\n"
            f"• *Stop-Loss (SL)*: `{signal.stop_loss}` _({signal.recommended_leverage}x)_\n"
            f"• *Why*: _{verdict.reasoning}_"
        )
        self.send_message(msg)

    def send_exit_or_stall_alert(
        self,
        exit_sig: ExitAlertSignal,
        closed_record: Optional[ClosedTradeRecord] = None,
    ) -> None:
        if exit_sig.reason == "PROFIT_FLATLINE_STALL":
            header = f"🟠 *POSITION UPDATE — TAKE PROFIT NOW (STALL): {exit_sig.symbol}*"
        elif "TP" in exit_sig.reason:
            header = f"🟠 *POSITION UPDATE — TAKE-PROFIT HIT: {exit_sig.symbol}*"
        elif "STOP" in exit_sig.reason:
            header = f"🟠 *POSITION UPDATE — STOP-LOSS LEVEL CROSSED: {exit_sig.symbol}*"
        else:
            header = f"🟠 *POSITION UPDATE — EXIT/TRAIL ALERT: {exit_sig.symbol}*"

        prec = 8 if exit_sig.current_price < 0.001 else (5 if exit_sig.current_price < 1.0 else (4 if exit_sig.current_price < 100 else 2))
        entry_str = f"{exit_sig.entry_price:.{prec}f}"
        curr_str = f"{exit_sig.current_price:.{prec}f}"

        msg = (
            f"{header}\n"
            f"• *Your Open Position*: `{exit_sig.position_side}` | *Entry*: `{entry_str}` → *Now*: `{curr_str}`\n"
            f"• *Open PnL*: `${exit_sig.unrealized_pnl_usd:+.2f}` (`{exit_sig.unrealized_pnl_pct_on_margin:+.1f}% ROE`)\n"
            f"• *Recommended Update*: _{exit_sig.explanation}_"
        )
        self.send_message(msg)

    def download_file_bytes(self, file_id: str) -> Optional[tuple[bytes, str]]:
        """Download a photo/document sent to the Telegram bot and return (bytes, mime_type)."""
        if not self.token or not file_id:
            return None
        info = self._api_call(f"getFile?file_id={file_id}")
        if not info.get("ok"):
            return None
        file_path = (info.get("result") or {}).get("file_path")
        if not file_path:
            return None
        url = f"https://api.telegram.org/file/bot{self.token}/{file_path}"
        ext = file_path.rsplit(".", 1)[-1].lower() if "." in file_path else "jpg"
        mime_map = {"png": "image/png", "webp": "image/webp", "jpg": "image/jpeg", "jpeg": "image/jpeg"}
        mime_type = mime_map.get(ext, "image/jpeg")
        try:
            resp = self.session.get(url, timeout=10)
            if resp.status_code == 200 and resp.content:
                return resp.content, mime_type
        except Exception:
            pass
        try:
            res = subprocess.run(["curl", "-4", "-s", url], capture_output=True, timeout=12)
            if res.returncode == 0 and res.stdout:
                return res.stdout, mime_type
        except Exception:
            pass
        return None

    def poll_commands(self, command_handler: Callable[[str], str]) -> bool:
        """
        Non-blocking poll for incoming Telegram messages (both /commands, plain-English chat, and screenshots!).
        Returns True if one or more messages were processed.
        """
        if not self.token:
            return False
        processed_any = False
        try:
            data = self._api_call(f"getUpdates?offset={self.last_update_id + 1}&timeout=1")
            if not data.get("ok"):
                return False
            updates = data.get("result", [])
            for upd in updates:
                self.last_update_id = int(upd["update_id"])
                msg = upd.get("message") or upd.get("edited_message") or {}
                cid = msg.get("chat", {}).get("id")
                text = (msg.get("text") or "").strip()
                caption = (msg.get("caption") or "").strip()

                # Check for photo or image document (screenshot)
                photo_file_id: Optional[str] = None
                if msg.get("photo") and isinstance(msg["photo"], list):
                    photo_file_id = msg["photo"][-1].get("file_id")
                elif msg.get("document") and str(msg["document"].get("mime_type", "")).startswith("image/"):
                    photo_file_id = msg["document"].get("file_id")

                # Auto-bind chat_id on first message!
                if cid and not self.chat_id:
                    self._save_chat_id_to_env(str(cid))
                    self.send_message(
                        "✅ *Connected to @kaintraderbot!*\n"
                        f"• *Your Chat ID*: `{self.chat_id}` (saved to `.env`)\n"
                        "• Live BTCC entry recommendations & *Profit-Flatline Stall* exit alerts are now active.\n\n"
                        "You can send a *screenshot of your BTCC positions*, chat in plain English (e.g. _\"I went long on BTC at 74500 10x\"_, _\"set alert threshold to 75%\"_), or use `/status`!"
                    )
                    processed_any = True

                if cid and self.chat_id and str(cid) != str(self.chat_id):
                    continue

                if photo_file_id:
                    processed_any = True
                    dl = self.download_file_bytes(photo_file_id)
                    owner = getattr(command_handler, "__self__", None)
                    if dl and owner and hasattr(owner, "handle_telegram_photo"):
                        img_bytes, mime_type = dl
                        reply = owner.handle_telegram_photo(img_bytes, mime_type=mime_type, caption=caption)
                        if reply:
                            self.send_message(reply)
                    else:
                        self.send_message("⚠️ Could not download screenshot from Telegram. Please try sending it again.")
                    continue

                if text:
                    processed_any = True
                    if text.lower() == "/start":
                        self.send_message(command_handler("/status"))
                    else:
                        reply = command_handler(text)
                        if reply:
                            self.send_message(reply)
        except Exception:
            pass
        return processed_any
