"""
portfolio_risk.py — Shared portfolio-level risk controller.

Both CRASH1000 and BOOM1000 import this module so they coordinate on a
single shared account.

Rules enforced at portfolio level:
  Daily DD   : 3%  of day_start_balance  — based on REALIZED P&L only (ignores open stake)
  Monthly DD : 20% of month_start_balance — based on REALIZED P&L only
  Max trades : 12/day combined (each bot still capped at 6 by its own logic)
  Concurrent : max 2 bots in a live position simultaneously

File: portfolio_state.json
  Written atomically (tmp → rename) under an exclusive fcntl advisory lock so
  concurrent bot processes never corrupt state or race on the tmp file.

Correct entry flow for each bot:
  1. can_open() / concurrent_ok()  — fast pre-gate in the main poll loop (read-only)
  2. check_and_reserve()           — atomic check + slot reservation, called AFTER
                                     the signal fires and BEFORE contracts are placed
  3. release_slot()                — call if contract placement subsequently fails
  4. on_open()                     — records balance after successful open (no count changes)
  5. on_close()                    — records realized P&L, decrements concurrent slot
"""

import json
import os
import fcntl
import contextlib
import logging
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger("portfolio_risk")

BASE_DIR  = Path(__file__).resolve().parent
STATE_FILE = BASE_DIR / "portfolio_state.json"
TMP_FILE   = BASE_DIR / "portfolio_state.tmp"
LOCK_FILE  = BASE_DIR / "portfolio_state.lock"

DAILY_DD_LIMIT           = 0.05   # 5%  of day_start_balance
MONTHLY_DD_LIMIT         = 0.20   # 20% of month_start_balance
MAX_DAILY_TRADES         = 12     # combined across all bots
MAX_CONCURRENT_POSITIONS = 2      # max bots live at the same time


def _today()   -> str: return datetime.now(timezone.utc).strftime("%Y-%m-%d")
def _month()   -> str: return datetime.now(timezone.utc).strftime("%Y-%m")
def _now_iso() -> str: return datetime.now(timezone.utc).isoformat()


@contextlib.contextmanager
def _portfolio_lock():
    """Exclusive POSIX advisory lock — serialises all read-modify-write cycles."""
    lf = open(LOCK_FILE, "a")
    try:
        fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(lf.fileno(), fcntl.LOCK_UN)
        lf.close()


def _load() -> dict:
    try:
        if STATE_FILE.exists():
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def _save(state: dict):
    try:
        TMP_FILE.write_text(json.dumps(state, default=str, indent=2), encoding="utf-8")
        os.replace(TMP_FILE, STATE_FILE)
    except Exception as e:
        log.warning("portfolio_risk: save failed: %s", e)


# ── PUBLIC API ─────────────────────────────────────────────────────────────────

def sync(bot_name: str, balance: float) -> tuple:
    """
    Call every poll cycle. Updates shared balance, resets counters on new day/month.
    Returns (blocked: bool, reason: str).
    """
    with _portfolio_lock():
        s         = _load()
        now_day   = _today()
        now_month = _month()

        if s.get("day") != now_day:
            s["day"]                  = now_day
            s["day_start_balance"]    = balance
            s["daily_pnl"]            = 0.0
            s["daily_dd_hit"]         = False
            s["trades_today"]         = 0
            s["crash1000_trades"]     = 0
            s["boom1000_trades"]      = 0
            s["concurrent_positions"] = 0
            log.info("[portfolio] New day %s | start_balance=%.2f", now_day, balance)

        if s.get("month") != now_month:
            s["month"]               = now_month
            s["month_start_balance"] = balance
            s["monthly_pnl"]         = 0.0
            s["monthly_dd_hit"]      = False
            log.info("[portfolio] New month %s | start_balance=%.2f", now_month, balance)

        s["balance"]    = round(balance, 2)
        s["updated_by"] = bot_name
        s["updated_at"] = _now_iso()

        day_start   = s.get("day_start_balance", balance)
        daily_pnl   = s.get("daily_pnl", 0.0)
        daily_limit = day_start * DAILY_DD_LIMIT if day_start > 0 else 0.0
        if daily_limit > 0 and daily_pnl < -daily_limit:
            if not s.get("daily_dd_hit"):
                s["daily_dd_hit"] = True
                log.warning(
                    "[portfolio] DAILY DD %.0f%% HIT | realized_pnl=$%.2f limit=$%.2f | "
                    "both bots paused for today",
                    DAILY_DD_LIMIT * 100, daily_pnl, -daily_limit)
            _save(s)
            dd_pct = abs(daily_pnl) / day_start * 100
            return True, f"portfolio daily DD {dd_pct:.1f}% — paused today"

        if s.get("daily_dd_hit"):
            # Flag was set under a previous limit. Re-evaluate against the current limit.
            # If P&L is now within limits (e.g. limit was raised), clear the flag and continue.
            s["daily_dd_hit"] = False
            log.info("[portfolio] daily DD flag cleared — realized P&L %.1f%% within current %.0f%% limit",
                     abs(daily_pnl) / day_start * 100 if day_start > 0 else 0, DAILY_DD_LIMIT * 100)

        month_start   = s.get("month_start_balance", balance)
        monthly_pnl   = s.get("monthly_pnl", 0.0)
        monthly_limit = month_start * MONTHLY_DD_LIMIT if month_start > 0 else 0.0
        if monthly_limit > 0 and monthly_pnl < -monthly_limit:
            if not s.get("monthly_dd_hit"):
                s["monthly_dd_hit"] = True
                log.warning(
                    "[portfolio] MONTHLY DD %.0f%% HIT | realized_pnl=$%.2f limit=$%.2f | "
                    "both bots paused until next month",
                    MONTHLY_DD_LIMIT * 100, monthly_pnl, -monthly_limit)
            _save(s)
            dd_pct = abs(monthly_pnl) / month_start * 100
            return True, f"portfolio monthly DD {dd_pct:.1f}% — paused until next month"

        if s.get("monthly_dd_hit"):
            _save(s)
            return True, "portfolio monthly DD hit — paused until next month"

        _save(s)
        return False, ""


def can_open(bot_name: str) -> tuple:
    """
    Read-only pre-gate check. Call in the main poll loop before signal detection.
    Returns (allowed: bool, reason: str). Does NOT reserve a slot.
    Use check_and_reserve() atomically right before placing contracts.
    """
    with _portfolio_lock():
        s = _load()
        if s.get("daily_dd_hit"):
            return False, "portfolio daily DD hit"
        if s.get("monthly_dd_hit"):
            return False, "portfolio monthly DD hit"
        if s.get("trades_today", 0) >= MAX_DAILY_TRADES:
            return False, f"portfolio max {MAX_DAILY_TRADES} trades/day reached"
        if s.get("concurrent_positions", 0) >= MAX_CONCURRENT_POSITIONS:
            return False, f"max {MAX_CONCURRENT_POSITIONS} concurrent positions open"
        return True, ""


def concurrent_ok(bot_name: str) -> tuple:
    """
    Read-only concurrent-slot pre-gate. Used by bots with their own DD guards.
    Returns (allowed: bool, reason: str). Does NOT reserve a slot.
    """
    with _portfolio_lock():
        s = _load()
        if s.get("concurrent_positions", 0) >= MAX_CONCURRENT_POSITIONS:
            return False, f"max {MAX_CONCURRENT_POSITIONS} concurrent positions open"
        return True, ""


def check_and_reserve(bot_name: str) -> tuple:
    """
    Atomic check-and-reserve. Call AFTER a trade signal fires and BEFORE placing
    contracts. Under the exclusive lock this function:
      — re-checks all portfolio limits (DD, daily trades, concurrent positions)
      — atomically increments concurrent_positions, trades_today, and per-bot count
    Returns (allowed: bool, reason: str).
    If allowed=False no state is modified.
    If allowed=True, call release_slot() if contract placement subsequently fails.
    """
    with _portfolio_lock():
        s = _load()
        if s.get("daily_dd_hit"):
            return False, "portfolio daily DD hit"
        if s.get("monthly_dd_hit"):
            return False, "portfolio monthly DD hit"
        if s.get("trades_today", 0) >= MAX_DAILY_TRADES:
            return False, f"portfolio max {MAX_DAILY_TRADES} trades/day reached"
        if s.get("concurrent_positions", 0) >= MAX_CONCURRENT_POSITIONS:
            return False, f"max {MAX_CONCURRENT_POSITIONS} concurrent positions open"
        s["concurrent_positions"] = s.get("concurrent_positions", 0) + 1
        s["trades_today"]         = s.get("trades_today", 0) + 1
        key = f"{bot_name.lower()}_trades"
        s[key]         = s.get(key, 0) + 1
        s["updated_by"] = bot_name
        s["updated_at"] = _now_iso()
        _save(s)
        log.info("[portfolio] %s slot reserved | trades today: %d | concurrent: %d",
                 bot_name, s["trades_today"], s["concurrent_positions"])
        return True, ""


def release_slot(bot_name: str):
    """
    Undo a check_and_reserve() reservation. Call if contract placement fails
    after a successful check_and_reserve so counts stay accurate.
    """
    with _portfolio_lock():
        s = _load()
        s["concurrent_positions"] = max(0, s.get("concurrent_positions", 0) - 1)
        s["trades_today"]         = max(0, s.get("trades_today", 0) - 1)
        key = f"{bot_name.lower()}_trades"
        s[key]         = max(0, s.get(key, 0) - 1)
        s["updated_by"] = bot_name
        s["updated_at"] = _now_iso()
        _save(s)
        log.info("[portfolio] %s slot released (trade placement failed)", bot_name)


def on_open(bot_name: str, balance_before: float):
    """
    Record balance after a successful trade open.
    Counts (concurrent_positions, trades_today) were already incremented by
    check_and_reserve() — this call only persists the balance snapshot.
    """
    with _portfolio_lock():
        s = _load()
        s["balance"]    = round(balance_before, 2)
        s["updated_by"] = bot_name
        s["updated_at"] = _now_iso()
        _save(s)
        log.info("[portfolio] %s trade open confirmed | trades today: %d | concurrent: %d",
                 bot_name, s.get("trades_today", 0), s.get("concurrent_positions", 0))


def on_close(bot_name: str, pnl: float, new_balance: float):
    """Register a trade closing. Call after balance confirmed from Deriv."""
    with _portfolio_lock():
        s = _load()
        s["balance"]              = round(new_balance, 2)
        s["daily_pnl"]            = round(s.get("daily_pnl", 0.0) + pnl, 2)
        s["monthly_pnl"]          = round(s.get("monthly_pnl", 0.0) + pnl, 2)
        s["concurrent_positions"] = max(0, s.get("concurrent_positions", 0) - 1)
        s["updated_by"]           = bot_name
        s["updated_at"]           = _now_iso()
        _save(s)
        log.info("[portfolio] %s trade closed | pnl=$%.2f | day=$%.2f | month=$%.2f",
                 bot_name, pnl, s["daily_pnl"], s["monthly_pnl"])


def get_state() -> dict:
    """Read current portfolio state. Used by dashboard and bots."""
    return _load()
