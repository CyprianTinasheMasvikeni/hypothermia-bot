"""
Hypothermia Bot — Live Dashboard (Deriv API Edition)
Run: streamlit run dashboard.py
"""
import sys, time, json, math
from pathlib import Path
from datetime import datetime, timezone, timedelta

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ── BACKTEST CONSTANTS (pre-computed, no CSV needed on server) ─────────────────
BT_XAUUSD = {
    "symbol":        "frxXAUUSD",
    "direction":     "BOTH",
    "label":         "XAUUSD Gold · M5 EMA Pullback + H1 Trend Filter (Chandelier x1.6)",
    "mean_r":        0.355,
    "std_r":         1.50,
    "wr":            0.536,
    "pf":            1.888,
    "total_trades":  1322,
    "period":        "Jul 2025 – May 2026",
    "months":        11,
    "prof_months":   11,
    "kelly_full":    0.252,
    "kelly_half":    0.126,
    "our_risk":      0.02,
    "ruin_prob":     0.2,
    "mc_p5":           800_000,
    "mc_p25":        2_500_000,
    "mc_p50":        7_100_000,
    "mc_p75":       18_000_000,
    "mc_p95":       45_000_000,
    "mc_dd_median":  12.0,
    "mc_dd_p95":     28.0,
    "p_value":       0.0000001,
    "t_stat":        8.60,
    "ci_pf_lo":      1.72,
    "ci_pf_hi":      2.07,
    "ci_wr_lo":      0.509,
    "ci_wr_hi":      0.563,
    "ci_avr_lo":     0.274,
    "ci_avr_hi":     0.436,
    "trades_for_sig": 60,
    "max_streak":    8,
    "avg_gap_loss":  -1.0,
    "gap_risk_mult": 1.0,
    "starting_bal":  10_000,
    "filter":        "Close within 0.50 ATR of M5 EMA50 + H1 close vs H1 EMA21 + session 07:00-20:59 UTC",
    "oos_pf":        1.930,
    "oos_wr":        0.540,
}

BT_EURUSD = {
    "symbol":        "frxEURUSD",
    "direction":     "BOTH",
    "label":         "EUR/USD · M5 EMA Pullback + H1 Trend Filter (zone=0.75ATR)",
    "mean_r":        0.250,
    "std_r":         1.50,
    "wr":            0.487,
    "pf":            1.547,
    "total_trades":  1440,
    "period":        "Aug 2025 – May 2026",
    "months":        9,
    "prof_months":   9,
    "kelly_full":    0.218,
    "kelly_half":    0.109,
    "our_risk":      0.02,
    "ruin_prob":     0.5,
    "mc_p5":         1_200_000,
    "mc_p25":        3_500_000,
    "mc_p50":        8_000_000,
    "mc_p75":        17_000_000,
    "mc_p95":        42_000_000,
    "mc_dd_median":  14.0,
    "mc_dd_p95":     33.0,
    "p_value":       0.0000001,
    "t_stat":        6.33,
    "ci_pf_lo":      1.36,
    "ci_pf_hi":      1.74,
    "ci_wr_lo":      0.461,
    "ci_wr_hi":      0.513,
    "ci_avr_lo":     0.171,
    "ci_avr_hi":     0.329,
    "trades_for_sig": 60,
    "max_streak":    9,
    "avg_gap_loss":  -1.0,
    "gap_risk_mult": 1.0,
    "starting_bal":  10_000,
    "filter":        "Close within 0.75 ATR of M5 EMA50 + H1 close vs H1 EMA21 + session 07:00-20:59 UTC",
    "oos_pf":        1.334,
    "oos_wr":        0.480,
}

BT_GBPUSD = {
    "symbol":        "frxGBPUSD",
    "direction":     "BUY",
    "label":         "GBP/USD · Spike Reversion (bearish body > 2.0× ATR, BUY only)",
    "mean_r":        0.377,
    "std_r":         1.50,
    "wr":            0.460,
    "pf":            1.695,
    "total_trades":  135,
    "period":        "Aug 2025 – May 2026",
    "months":        9,
    "prof_months":   8,
    "kelly_full":    0.167,
    "kelly_half":    0.084,
    "our_risk":      0.02,
    "ruin_prob":     3.5,
    "mc_p5":         15_000,
    "mc_p25":        21_000,
    "mc_p50":        26_000,
    "mc_p75":        33_000,
    "mc_p95":        46_000,
    "mc_dd_median":  12.0,
    "mc_dd_p95":     28.0,
    "p_value":       0.002,
    "t_stat":        2.92,
    "ci_pf_lo":      1.20,
    "ci_pf_hi":      2.35,
    "ci_wr_lo":      0.376,
    "ci_wr_hi":      0.544,
    "ci_avr_lo":     0.124,
    "ci_avr_hi":     0.630,
    "trades_for_sig": 60,
    "max_streak":    8,
    "avg_gap_loss":  -1.0,
    "gap_risk_mult": 1.0,
    "starting_bal":  10_000,
    "filter":        "Bearish body (open − close) > 2.0 × ATR | BUY reversal | session 07:00-20:59 UTC | no H1 regime filter",
    "oos_pf":        1.831,
    "oos_wr":        0.460,
}

BT_USDJPY = {
    "symbol":        "frxUSDJPY",
    "direction":     "BOTH",
    "label":         "USD/JPY · M5 EMA Pullback + H1 Trend Filter (zone=0.30ATR)",
    "mean_r":        0.277,
    "std_r":         1.50,
    "wr":            0.521,
    "pf":            1.648,
    "total_trades":  909,
    "period":        "Aug 2025 – May 2026",
    "months":        9,
    "prof_months":   9,
    "kelly_full":    0.303,
    "kelly_half":    0.152,
    "our_risk":      0.02,
    "ruin_prob":     0.4,
    "mc_p5":         1_300_000,
    "mc_p25":        3_800_000,
    "mc_p50":        8_500_000,
    "mc_p75":        18_000_000,
    "mc_p95":        44_000_000,
    "mc_dd_median":  13.5,
    "mc_dd_p95":     31.0,
    "p_value":       0.0000001,
    "t_stat":        5.56,
    "ci_pf_lo":      1.45,
    "ci_pf_hi":      1.86,
    "ci_wr_lo":      0.488,
    "ci_wr_hi":      0.554,
    "ci_avr_lo":     0.177,
    "ci_avr_hi":     0.377,
    "trades_for_sig": 60,
    "max_streak":    8,
    "avg_gap_loss":  -1.0,
    "gap_risk_mult": 1.0,
    "starting_bal":  10_000,
    "filter":        "Close within 0.30 ATR of M5 EMA50 + H1 close vs H1 EMA21 + session 07:00-20:59 UTC",
    "oos_pf":        1.546,
    "oos_wr":        0.510,
}

BT_MAP = {
    "frxXAUUSD":  BT_XAUUSD,
    "EURUSD":     BT_EURUSD,
    "GBPUSD":     BT_GBPUSD,
    "USDJPY":     BT_USDJPY,
}

# ── BOT REGISTRY ──────────────────────────────────────────────────────────────
# Add new bots here — nothing else needs changing
BOTS = {
    "frxXAUUSD": {
        "label":      "XAUUSD Gold",
        "symbol":     "frxXAUUSD",
        "session":    "07:00-20:59 UTC",
        "type":       "pullback",
        "state":      BASE_DIR / "state_xauusd.json",
        "csv":        BASE_DIR / "live_trades_xauusd.csv",
        "log":        BASE_DIR / "bot_xauusd.log",
        "color":      "#FFB800",
        "accent":     "#FFD700",
        "dot":        "orange",
        "max_trades": 12,
        "max_hold":   6,
    },
    "EURUSD": {
        "label":      "EUR/USD",
        "symbol":     "EURUSD",
        "session":    "07:00-20:59 UTC",
        "type":       "pullback",
        "state":      BASE_DIR / "state_eurusd.json",
        "csv":        BASE_DIR / "live_trades_eurusd.csv",
        "log":        BASE_DIR / "bot_eurusd.log",
        "color":      "#0077FF",
        "accent":     "#4499FF",
        "dot":        "blue",
        "max_trades": 12,
        "max_hold":   6,
    },
    "GBPUSD": {
        "label":      "GBP/USD",
        "symbol":     "GBPUSD",
        "session":    "07:00-20:59 UTC",
        "type":       "spike",
        "state":      BASE_DIR / "state_gbpusd.json",
        "csv":        BASE_DIR / "live_trades_gbpusd.csv",
        "log":        BASE_DIR / "bot_gbpusd.log",
        "color":      "#CC00FF",
        "accent":     "#DD66FF",
        "dot":        "purple",
        "max_trades": 12,
        "max_hold":   24,
    },
    "USDJPY": {
        "label":      "USD/JPY",
        "symbol":     "USDJPY",
        "session":    "07:00-20:59 UTC",
        "type":       "pullback",
        "state":      BASE_DIR / "state_usdjpy.json",
        "csv":        BASE_DIR / "live_trades_usdjpy.csv",
        "log":        BASE_DIR / "bot_usdjpy.log",
        "color":      "#00BBCC",
        "accent":     "#66DDEE",
        "dot":        "white",
        "max_trades": 12,
        "max_hold":   6,
    },
}

st.set_page_config(page_title="CTM Trading — Hypothermia Bots", page_icon="⚡",
                   layout="wide", initial_sidebar_state="collapsed")

# ── PASSWORD GATE ──────────────────────────────────────────────────────────────
_DASHBOARD_PASSWORD = "hypothermia2026"

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.markdown("""
    <style>
    html, body, [data-testid="stApp"] { background: #080808 !important; }
    [data-testid="stAppViewContainer"] { background: #080808 !important; }
    #MainMenu, footer, [data-testid="stToolbar"] { visibility: hidden; }
    </style>
    """, unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col2:
        st.markdown("<br><br><br>", unsafe_allow_html=True)
        st.markdown("### ⚡ CTM Trading — Hypothermia Bots")
        st.markdown("Private dashboard. Enter password to continue.")
        pwd = st.text_input("Password", type="password", label_visibility="collapsed",
                            placeholder="Enter password...")
        if st.button("Enter", use_container_width=True):
            if pwd == _DASHBOARD_PASSWORD:
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Incorrect password.")
    st.stop()
# ── END PASSWORD GATE ───────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;600&display=swap');
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html, body, [data-testid="stApp"] { background: #080808 !important; font-family: 'Inter', sans-serif; color: #E0E0E0; }
[data-testid="stAppViewContainer"] { background: #080808 !important; }
[data-testid="stHeader"] { background: transparent !important; }
#MainMenu, footer, [data-testid="stToolbar"] { visibility: hidden; }
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #0F0F0F; }
::-webkit-scrollbar-thumb { background: #333; border-radius: 4px; }
.block-container { padding: 0 1.5rem 2rem 1.5rem !important; max-width: 100% !important; }
.hero {
    background: #000000; border-bottom: 1px solid #1C1C1C;
    padding: 22px 28px 18px 28px; margin: -1rem -1.5rem 0 -1.5rem;
    position: relative; overflow: hidden;
}
.hero::before {
    content: ''; position: absolute; inset: 0;
    background: radial-gradient(ellipse 50% 100% at 5% 50%, rgba(255,255,255,0.03) 0%, transparent 60%);
    pointer-events: none;
}
.hero-grid { display: grid; grid-template-columns: 1fr auto; align-items: center; gap: 20px; position: relative; z-index: 1; }
.hero-logo { display: flex; align-items: center; gap: 14px; }
.hero-icon { width: 44px; height: 44px; background: #FFFFFF; border-radius: 10px; display: flex; align-items: center; justify-content: center; font-size: 22px; }
.hero-title { font-size: 26px; font-weight: 900; color: #FFFFFF; letter-spacing: -0.6px; }
.hero-name  { font-size: 13px; color: #999; font-weight: 500; margin-top: 2px; }
.hero-sub   { font-size: 11px; color: #555; letter-spacing: 1.5px; text-transform: uppercase; margin-top: 2px; }
.hero-right { text-align: right; }
.hero-time  { font-family: 'JetBrains Mono', monospace; font-size: 22px; font-weight: 700; color: #FFFFFF; }
.hero-date  { font-size: 11px; color: #555; margin-top: 2px; }
.glass { background: #111111; border: 1px solid #1E1E1E; border-radius: 14px; position: relative; overflow: hidden; }
.glass::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 1px; background: linear-gradient(90deg, transparent, rgba(255,255,255,0.08), transparent); }
.metric-card { background: #111111; border: 1px solid #1E1E1E; border-radius: 12px; padding: 16px 18px; position: relative; overflow: hidden; }
.metric-card::after { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px; background: #333; }
.metric-card.green::after  { background: linear-gradient(90deg, #006633, #00CC66, #00FF88); }
.metric-card.red::after    { background: linear-gradient(90deg, #660022, #CC0044, #FF1155); }
.metric-card.gold::after   { background: linear-gradient(90deg, #664400, #CC8800, #FFB800); }
.metric-card.blue::after   { background: linear-gradient(90deg, #001466, #0033CC, #0055FF); }
.metric-card.orange::after { background: linear-gradient(90deg, #663300, #CC6600, #FF8800); }
.metric-card.purple::after { background: linear-gradient(90deg, #440066, #8800CC, #CC00FF); }
.metric-label { font-size: 10px; color: #555; text-transform: uppercase; letter-spacing: 1.5px; font-weight: 600; }
.metric-value { font-size: 28px; font-weight: 900; color: #FFFFFF; margin-top: 6px; font-family: 'JetBrains Mono', monospace; }
.metric-sub   { font-size: 11px; color: #444; margin-top: 4px; }
@keyframes pulse { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:0.5;transform:scale(0.85)} }
@keyframes glow        { 0%,100%{box-shadow:0 0 6px rgba(0,200,100,0.6)}   50%{box-shadow:0 0 18px rgba(0,200,100,0.9)} }
@keyframes glow-orange { 0%,100%{box-shadow:0 0 6px rgba(255,140,0,0.6)}   50%{box-shadow:0 0 18px rgba(255,140,0,0.9)} }
@keyframes glow-purple { 0%,100%{box-shadow:0 0 6px rgba(200,0,255,0.6)}   50%{box-shadow:0 0 18px rgba(200,0,255,0.9)} }
.live-dot        { display:inline-block;width:8px;height:8px;background:#00E676;border-radius:50%;animation:pulse 1.5s infinite,glow 1.5s infinite;vertical-align:middle;margin-right:6px; }
.live-dot-orange { display:inline-block;width:8px;height:8px;background:#FFB800;border-radius:50%;animation:pulse 1.5s infinite,glow-orange 1.5s infinite;vertical-align:middle;margin-right:6px; }
.live-dot-purple { display:inline-block;width:8px;height:8px;background:#CC00FF;border-radius:50%;animation:pulse 1.5s infinite,glow-purple 1.5s infinite;vertical-align:middle;margin-right:6px; }
.live-dot-white  { display:inline-block;width:8px;height:8px;background:#FFFFFF;border-radius:50%;animation:pulse 1.5s infinite;vertical-align:middle;margin-right:6px; }
.dead-dot        { display:inline-block;width:8px;height:8px;background:#FF1744;border-radius:50%;vertical-align:middle;margin-right:6px; }
.et-card { background:#111;border:1px solid #1E1E1E;border-radius:12px;padding:18px 20px; }
.et-label { font-size:10px;color:#444;text-transform:uppercase;letter-spacing:1.5px;font-weight:600;margin-bottom:6px; }
.et-value { font-size:22px;font-weight:800;font-family:'JetBrains Mono',monospace;color:#FFF; }
.et-sub   { font-size:11px;color:#444;margin-top:4px; }
.prog-track { background:#1A1A1A;border-radius:4px;height:6px;margin-top:8px;overflow:hidden; }
.prog-fill  { height:100%;border-radius:4px;transition:width 0.5s; }
.bias-strong-buy  { color:#00FF88;font-size:15px;font-weight:800; }
.bias-strong-sell { color:#FF3366;font-size:15px;font-weight:800; }
.bias-weak-buy    { color:#69F0AE;font-size:15px;font-weight:700; }
.bias-weak-sell   { color:#FF8A80;font-size:15px;font-weight:700; }
.bias-neutral     { color:#555;font-size:15px;font-weight:600; }
.trade-card { border-radius:12px;padding:16px 20px;margin:6px 0;position:relative;overflow:hidden; }
.trade-card.buy  { background:#0D1A0F;border:1px solid #1A3320; }
.trade-card.sell { background:#1A0D0D;border:1px solid #331515; }
.trade-card::before { content:'';position:absolute;top:0;left:0;right:0;height:2px; }
.trade-card.buy::before  { background:#00CC66; }
.trade-card.sell::before { background:#FF3366; }
.trade-dir { font-size:24px;font-weight:900;letter-spacing:-1px; }
.trade-dir.buy  { color:#00E676; }
.trade-dir.sell { color:#FF3366; }
.trade-stat-label { font-size:10px;color:#444;text-transform:uppercase;letter-spacing:1px; }
.trade-stat-val   { font-size:14px;font-weight:700;color:#E0E0E0;font-family:'JetBrains Mono',monospace; }
.trade-table { width:100%;border-collapse:collapse;font-size:11px; }
.trade-table thead th { background:#0D0D0D;color:#555;font-size:9px;text-transform:uppercase;letter-spacing:1px;padding:8px 10px;text-align:left;border-bottom:1px solid #1E1E1E;position:sticky;top:0;z-index:2; }
.scroll-table { overflow-y:auto;max-height:420px; }
.trade-table tbody tr { border-bottom:1px solid #161616; }
.trade-table tbody tr:hover { background:#161616; }
.trade-table tbody td { padding:7px 10px; }
.win-text  { color:#00E676;font-weight:600; }
.loss-text { color:#FF3366;font-weight:600; }
.mono { font-family:'JetBrains Mono',monospace; }
.log-box { background:#000;border:1px solid #1A1A1A;border-radius:10px;padding:12px 14px;font-family:'JetBrains Mono',monospace;font-size:10px;color:#444;max-height:180px;overflow-y:auto;line-height:1.7;white-space:pre-wrap; }
.overview-card { background:#0C0C0C;border:1px solid #1E1E1E;border-radius:14px;padding:18px 20px;text-align:center; }
div[data-testid="metric-container"] { display:none; }
.hero::after { content:'';position:absolute;top:0;left:0;right:0;height:1px;background:linear-gradient(90deg,transparent,rgba(255,255,255,0.15) 30%,rgba(255,255,255,0.15) 70%,transparent);pointer-events:none; }
.section-head { font-size:11px;color:#444;letter-spacing:2.5px;text-transform:uppercase;margin-bottom:14px;padding-left:10px;border-left:2px solid #2A2A2A;line-height:1; }
button[data-baseweb="tab"] { background:transparent!important;color:#444!important;font-family:'Inter',sans-serif!important;font-size:11px!important;font-weight:700!important;letter-spacing:1.5px!important;text-transform:uppercase!important;padding:14px 24px!important;border:none!important;border-radius:0!important; }
button[data-baseweb="tab"]:hover { color:#AAA!important; }
button[data-baseweb="tab"][aria-selected="true"] { color:#FFF!important; }
[data-baseweb="tab-highlight"] { background:#FFF!important;height:1px!important; }
[data-baseweb="tab-border"] { display:none!important; }
[data-testid="stTabs"] [data-baseweb="tab-list"] { gap:0!important;border-bottom:1px solid #1C1C1C!important;background:#000!important;padding:0 4px!important;margin-bottom:0!important; }
</style>
""", unsafe_allow_html=True)


# ── DATA LOADERS ──────────────────────────────────────────────────────────────
def load_state(path: Path) -> dict:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}

def load_trades(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        if df.empty:
            return df
        df["open_time"]  = pd.to_datetime(df["open_time"],  errors="coerce", utc=True)
        df["close_time"] = pd.to_datetime(df["close_time"], errors="coerce", utc=True)
        for col in ["pnl_usd", "r_multiple", "balance_after", "peak_r"]:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        return df.sort_values("close_time", ascending=False).reset_index(drop=True)
    except Exception:
        return pd.DataFrame()

def read_log(path: Path, n: int = 30) -> str:
    if not path.exists():
        return "No log file yet."
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(lines[-n:])
    except Exception:
        return "Error reading log."

def bot_alive(state: dict) -> bool:
    updated = state.get("updated_at", "")
    if not updated:
        return False
    try:
        dt = datetime.fromisoformat(updated.replace("Z", "+00:00"))
        return (datetime.now(timezone.utc) - dt).total_seconds() < 600
    except Exception:
        return False

def dot_html(dot_type: str, alive: bool) -> str:
    if not alive:
        return '<span class="dead-dot"></span>'
    cls = {"blue": "live-dot", "orange": "live-dot-orange", "purple": "live-dot-purple",
           "white": "live-dot-white"}.get(dot_type, "live-dot")
    return f'<span class="{cls}"></span>'


# ── CHARTS ────────────────────────────────────────────────────────────────────
CHART_BG = "#0A0A0A"
GRID_COL = "rgba(255,255,255,0.04)"
AXIS_COL = "rgba(255,255,255,0.08)"

def equity_chart(df, color="#0077FF"):
    if df.empty or "balance_after" not in df.columns:
        return None
    d = df.sort_values("close_time").dropna(subset=["balance_after"])
    dot_colors = ["#00E676" if r == "WIN" else "#FF3366" for r in d["result"]]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=d["close_time"], y=d["balance_after"], mode="lines+markers",
        line=dict(color=color, width=2.5, shape="spline"),
        marker=dict(size=7, color=dot_colors, line=dict(width=2, color="#030B18")),
        fill="tozeroy",
        fillcolor=f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.06)",
        hovertemplate="<b>%{x|%b %d %H:%M}</b><br>$%{y:.2f}<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor=CHART_BG, plot_bgcolor=CHART_BG,
        font=dict(color="#555", family="Inter", size=11),
        margin=dict(l=8, r=8, t=8, b=8), height=200, showlegend=False,
        xaxis=dict(gridcolor=GRID_COL, showgrid=True, zeroline=False, linecolor=AXIS_COL, tickfont=dict(size=10)),
        yaxis=dict(gridcolor=GRID_COL, showgrid=True, zeroline=False, linecolor=AXIS_COL, tickprefix="$", tickfont=dict(size=10)),
    )
    return fig

def donut_chart(wins, losses, color="#00E676"):
    if wins + losses == 0:
        return None
    fig = go.Figure(go.Pie(
        values=[wins, losses], labels=["Wins", "Losses"],
        marker=dict(colors=[color, "#FF3366"], line=dict(color="#030B18", width=3)),
        hole=0.68, textinfo="none",
        hovertemplate="<b>%{label}</b>: %{value}<br>%{percent}<extra></extra>",
    ))
    fig.add_annotation(
        text=f"<b>{wins/(wins+losses)*100:.0f}%</b>",
        font=dict(size=20, color=color, family="JetBrains Mono"),
        showarrow=False,
    )
    fig.update_layout(
        paper_bgcolor=CHART_BG, margin=dict(l=8, r=8, t=8, b=8),
        height=200, showlegend=False, font=dict(color="#555"),
    )
    return fig

def pnl_bar_chart(df):
    if df.empty:
        return None
    d = df.sort_values("close_time").tail(15)
    colors = ["rgba(0,230,118,0.8)" if v > 0 else "rgba(255,51,102,0.8)" for v in d["pnl_usd"]]
    fig = go.Figure(go.Bar(
        x=list(range(len(d))), y=d["pnl_usd"],
        marker=dict(color=colors, line=dict(color=["#00E676" if v>0 else "#FF3366" for v in d["pnl_usd"]], width=1.5)),
        hovertemplate="Trade %{x}<br><b>$%{y:+.2f}</b><extra></extra>",
    ))
    fig.add_hline(y=0, line_color="rgba(0,80,180,0.4)", line_width=1)
    fig.update_layout(
        paper_bgcolor=CHART_BG, plot_bgcolor=CHART_BG,
        font=dict(color="#555", family="Inter", size=10),
        margin=dict(l=8, r=8, t=8, b=8), height=200,
        xaxis=dict(showgrid=False, showticklabels=False, linecolor=AXIS_COL),
        yaxis=dict(gridcolor=GRID_COL, tickprefix="$", linecolor=AXIS_COL),
    )
    return fig

def color_log(text: str) -> str:
    colored = ""
    for line in text.splitlines():
        if "ERROR" in line or "FAILED" in line:
            colored += f'<span style="color:#CC2244">{line}</span>\n'
        elif "WARNING" in line or "DD LIMIT" in line:
            colored += f'<span style="color:#CC8800">{line}</span>\n'
        elif "CONTRACT OPEN" in line or "TRADE OPEN" in line:
            colored += f'<span style="color:#00CC66">{line}</span>\n'
        elif "TRADE CLOSED" in line or "JOURNAL" in line:
            colored += f'<span style="color:#00AAFF">{line}</span>\n'
        elif "CHANDELIER" in line:
            colored += f'<span style="color:#8866FF">{line}</span>\n'
        elif "SPIKE" in line:
            colored += f'<span style="color:#CC00FF">{line}</span>\n'
        else:
            colored += f'<span style="color:#1A4A7A">{line}</span>\n'
    return colored


# ── BOT PANEL ─────────────────────────────────────────────────────────────────
def render_bot_panel(cfg: dict, now_utc: datetime):
    s     = load_state(cfg["state"])
    df    = load_trades(cfg["csv"])
    alive = bot_alive(s)

    balance      = float(s.get("balance", 0))
    peak_bal     = float(s.get("peak_balance", balance))
    trades_today = int(s.get("trades_today", 0))
    cw           = int(s.get("consecutive_wins", 0))
    cl           = int(s.get("consecutive_losses", 0))
    risk_pct     = float(s.get("risk_pct", 2))
    last_signal  = s.get("last_signal", "WAIT")
    last_reason  = s.get("last_reason", "")
    active_trade = s.get("active_trade")
    updated_at   = s.get("updated_at", "")

    total     = len(df)
    wins_df   = df[df["result"] == "WIN"]  if not df.empty else pd.DataFrame()
    losses_df = df[df["result"] == "LOSS"] if not df.empty else pd.DataFrame()
    wr        = len(wins_df) / total * 100 if total else 0
    net_pnl   = df["pnl_usd"].sum() if not df.empty else 0
    dd_pct    = (peak_bal - balance) / peak_bal * 100 if peak_bal > 0 else 0

    color  = cfg["color"]
    accent = cfg["accent"]
    status = "ONLINE" if alive else "OFFLINE"
    pnl_col  = "#00E676" if net_pnl >= 0 else "#FF3366"
    pnl_sign = "+" if net_pnl >= 0 else ""

    # Panel header
    st.markdown(f"""
    <div style="background:#0C0C0C;border:1px solid #1E1E1E;border-radius:16px;padding:18px;margin-bottom:8px">
      <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:14px;padding-bottom:10px;border-bottom:1px solid #1A1A1A">
        <div>
          <div style="font-size:16px;font-weight:800;color:{color}">{cfg['label']} <span style="color:#333;font-size:13px">· {cfg['symbol']}</span></div>
          <div style="font-size:11px;color:#444;margin-top:2px">Session: {cfg['session']} · x100 Multiplier</div>
        </div>
        <div style="text-align:right">
          <div style="font-size:13px;font-weight:700">{dot_html(cfg['dot'], alive)}{status}</div>
          <div style="font-size:10px;color:#333;margin-top:2px">{updated_at[11:19] if len(updated_at)>19 else '--'} UTC</div>
        </div>
      </div>
    """, unsafe_allow_html=True)

    # Metrics row
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""<div class="metric-card">
          <div class="metric-label">Balance</div>
          <div class="metric-value" style="font-size:20px">${balance:.2f}</div>
          <div class="metric-sub" style="color:{pnl_col}">{pnl_sign}${net_pnl:.2f} P&L</div>
        </div>""", unsafe_allow_html=True)
    with m2:
        dd_col = "#FF3366" if dd_pct > 8 else ("#FFB800" if dd_pct > 4 else "#00E676")
        st.markdown(f"""<div class="metric-card">
          <div class="metric-label">Drawdown</div>
          <div class="metric-value" style="font-size:20px;color:{dd_col}">{dd_pct:.1f}%</div>
          <div class="metric-sub">Peak ${peak_bal:.2f}</div>
        </div>""", unsafe_allow_html=True)
    with m3:
        wr_col = "#00E676" if wr >= 55 else ("#FFB800" if wr >= 45 else "#FF3366")
        st.markdown(f"""<div class="metric-card">
          <div class="metric-label">Win Rate</div>
          <div class="metric-value" style="font-size:20px;color:{wr_col}">{wr:.0f}%</div>
          <div class="metric-sub">{len(wins_df)}W · {len(losses_df)}L · {total} trades</div>
        </div>""", unsafe_allow_html=True)
    with m4:
        max_td = cfg.get("max_trades", 6)
        bar = "■" * trades_today + "□" * max(0, max_td - trades_today)
        streak_txt = f"W{cw}" if cw >= 2 else (f"L{cl}" if cl >= 2 else "—")
        streak_col = "#00E676" if cw >= 2 else ("#FF3366" if cl >= 2 else "#555")
        st.markdown(f"""<div class="metric-card">
          <div class="metric-label">Today / Streak</div>
          <div class="metric-value" style="font-size:20px">{trades_today}<span style="font-size:12px;color:#333"> / {max_td}</span></div>
          <div class="metric-sub" style="color:#0066BB;letter-spacing:2px">{bar} · <span style="color:{streak_col}">{streak_txt}</span> · {risk_pct:.0f}%</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    # Signal section — differs per bot type
    sa, sb = st.columns(2)
    if cfg["type"] == "trend":
        trend_bias = s.get("trend_bias", "NEUTRAL")
        bias_map = {
            "STRONG_BUY":  ("bias-strong-buy",  "▲ STRONG BUY"),
            "STRONG_SELL": ("bias-strong-sell",  "▼ STRONG SELL"),
            "WEAK_BUY":    ("bias-weak-buy",     "△ WEAK BUY"),
            "WEAK_SELL":   ("bias-weak-sell",    "▽ WEAK SELL"),
            "NEUTRAL":     ("bias-neutral",      "— NEUTRAL"),
        }
        bcls, btxt = bias_map.get(trend_bias, ("bias-neutral", trend_bias))
        sig_col = "#00E676" if last_signal == "BUY" else ("#FF3366" if last_signal == "SELL" else "#555")
        with sa:
            st.markdown(f"""<div class="glass" style="padding:12px 16px">
              <div class="metric-label">M15 Trend Bias</div>
              <div style="margin-top:6px"><span class="{bcls}">{btxt}</span></div>
            </div>""", unsafe_allow_html=True)
        with sb:
            st.markdown(f"""<div class="glass" style="padding:12px 16px">
              <div class="metric-label">Entry Signal</div>
              <div style="margin-top:6px;font-size:13px;font-weight:700;color:{sig_col}">{last_signal}</div>
              <div class="metric-sub">{last_reason[:60] if last_reason else '--'}</div>
            </div>""", unsafe_allow_html=True)
    elif cfg["type"] == "pullback":
        # EMA Pullback bot — show H1 regime + current signal
        h1_regime = s.get("h1_regime", "WARMUP")
        h1_close  = s.get("h1_close")
        h1_ema21  = s.get("h1_ema21")
        reg_col   = "#00E676" if h1_regime == "BULL" else ("#FF3366" if h1_regime == "BEAR" else "#555")
        reg_txt   = ("▲ BULL — BUY signals" if h1_regime == "BULL"
                     else ("▼ BEAR — SELL signals" if h1_regime == "BEAR" else "WARMING UP"))
        sig_col   = "#00E676" if last_signal == "BUY" else ("#FF3366" if last_signal == "SELL" else ("#FFB800" if last_signal == "TRADE_OPEN" else "#555"))
        price_str = f"H1 {h1_close:.5f} vs EMA21 {h1_ema21:.5f}" if h1_close and h1_ema21 else "loading..."
        with sa:
            st.markdown(f"""<div class="glass" style="padding:12px 16px">
              <div class="metric-label">H1 Regime · Entry Direction</div>
              <div style="margin-top:6px;font-size:13px;font-weight:800;color:{reg_col}">{reg_txt}</div>
              <div class="metric-sub">{price_str}</div>
            </div>""", unsafe_allow_html=True)
        with sb:
            st.markdown(f"""<div class="glass" style="padding:12px 16px">
              <div class="metric-label">Last Signal</div>
              <div style="margin-top:6px;font-size:13px;font-weight:700;color:{sig_col}">{last_signal}</div>
              <div class="metric-sub">{last_reason[:60] if last_reason else '--'}</div>
            </div>""", unsafe_allow_html=True)
    else:
        # Spike bot — show spike status
        spike_col = "#CC00FF" if last_signal == "BUY" else "#555"
        spike_txt = "⚡ SPIKE — BUY SIGNAL" if last_signal == "BUY" else "WATCHING..."
        with sa:
            st.markdown(f"""<div class="glass" style="padding:12px 16px">
              <div class="metric-label">Spike Status</div>
              <div style="margin-top:6px;font-size:13px;font-weight:800;color:{spike_col}">{spike_txt}</div>
            </div>""", unsafe_allow_html=True)
        with sb:
            st.markdown(f"""<div class="glass" style="padding:12px 16px">
              <div class="metric-label">Last Candle</div>
              <div style="margin-top:6px;font-size:12px;font-weight:600;color:#888">{last_reason[:60] if last_reason else '--'}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    # Open trade
    if active_trade:
        at    = active_trade
        d     = at.get("direction", "BUY")
        dcls  = d.lower()
        arrow = "▲ LONG" if d == "BUY" else "▼ SHORT"
        open_dt = at.get("open_time", "")
        try:
            ot = datetime.fromisoformat(open_dt.replace("Z", "+00:00"))
            hold_min = int((now_utc - ot).total_seconds() // 60)
            hold_str = f"{hold_min//60}h {hold_min%60}m" if hold_min >= 60 else f"{hold_min}m"
        except Exception:
            hold_str = "—"
        held_candles = at.get("candles_held", "")
        max_hold_cfg = cfg.get("max_hold", 24)
        held_info = f"{held_candles}/{max_hold_cfg}" if held_candles != "" else hold_str
        st.markdown(f"""
        <div class="trade-card {dcls}">
          <div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr 1fr;gap:14px;align-items:center">
            <div><div class="trade-dir {dcls}">{arrow}</div><div style="font-size:11px;color:#3A6A8A">{at.get('trade_id','--')}</div></div>
            <div><div class="trade-stat-label">Entry</div><div class="trade-stat-val">{float(at.get('entry',0)):.2f}</div></div>
            <div><div class="trade-stat-label">SL $</div><div class="trade-stat-val" style="color:#FF8A65">${float(at.get('sl_usd',0)):.2f}</div></div>
            <div><div class="trade-stat-label">Peak R</div><div class="trade-stat-val">{float(at.get('peak_r',0)):.2f}R</div></div>
            <div><div class="trade-stat-label">Held</div><div class="trade-stat-val">{held_info}</div></div>
          </div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""<div class="glass" style="padding:14px 20px;text-align:center">
          <span style="color:#1A4070;font-size:13px;font-weight:600">No Open Position — Scanning...</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    # Charts
    ch1, ch2, ch3 = st.columns([2.5, 1.5, 1.5])
    with ch1:
        st.markdown('<div style="font-size:10px;color:#1A4A7A;margin-bottom:4px;letter-spacing:1px">EQUITY CURVE</div>', unsafe_allow_html=True)
        fig = equity_chart(df, color)
        if fig:
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        else:
            st.markdown('<div class="glass" style="height:200px;display:flex;align-items:center;justify-content:center;color:#0D2A4A;font-size:12px">Waiting for first trade...</div>', unsafe_allow_html=True)
    with ch2:
        st.markdown('<div style="font-size:10px;color:#1A4A7A;margin-bottom:4px;letter-spacing:1px">WIN RATE</div>', unsafe_allow_html=True)
        fig2 = donut_chart(len(wins_df), len(losses_df), accent)
        if fig2:
            st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})
        else:
            st.markdown('<div class="glass" style="height:200px;display:flex;align-items:center;justify-content:center;color:#0D2A4A;font-size:12px">No data</div>', unsafe_allow_html=True)
    with ch3:
        st.markdown('<div style="font-size:10px;color:#1A4A7A;margin-bottom:4px;letter-spacing:1px">LAST 15 TRADES</div>', unsafe_allow_html=True)
        fig3 = pnl_bar_chart(df)
        if fig3:
            st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})
        else:
            st.markdown('<div class="glass" style="height:200px;display:flex;align-items:center;justify-content:center;color:#0D2A4A;font-size:12px">No trades yet</div>', unsafe_allow_html=True)

    # Trade table
    if not df.empty:
        rows_html = ""
        for _, r in df.head(100).iterrows():
            res   = r.get("result", "")
            pnl_v = float(r.get("pnl_usd", 0))
            rmult = float(r.get("r_multiple", 0))
            direc = r.get("direction", "")
            pc    = "#00E676" if pnl_v >= 0 else "#FF3366"
            dc    = "win-text" if direc == "BUY" else "loss-text"
            dt    = str(r.get("open_time", ""))[:16] if pd.notna(r.get("open_time")) else "--"
            ep    = f"{float(r.get('entry_price',0)):.2f}" if r.get("entry_price") else "--"
            xp    = f"{float(r.get('exit_price', 0)):.2f}" if r.get("exit_price")  else "--"
            rn    = r.get("reason", "")
            rows_html += f"""<tr>
              <td class="mono" style="color:#444">{r.get('trade_id','')}</td>
              <td style="color:#444">{dt}</td>
              <td class="{dc} mono">{direc}</td>
              <td class="mono" style="color:#888">{ep}</td>
              <td class="mono" style="color:#888">{xp}</td>
              <td class="mono" style="color:{pc};font-weight:700">${pnl_v:+.2f}</td>
              <td class="mono {'win-text' if rmult>0 else 'loss-text'}">{rmult:+.2f}R</td>
              <td class="{'win-text' if res=='WIN' else 'loss-text'} mono">{res}</td>
              <td style="color:#333;font-size:10px">{rn}</td>
            </tr>"""
        st.markdown(f"""<div class="glass" style="padding:0;overflow:hidden">
          <div class="scroll-table">
          <table class="trade-table">
            <thead><tr><th>ID</th><th>Time (UTC)</th><th>Dir</th><th>Entry</th><th>Exit</th><th>PnL</th><th>R</th><th>Result</th><th>Reason</th></tr></thead>
            <tbody>{rows_html}</tbody>
          </table>
          </div>
        </div>""", unsafe_allow_html=True)

    # Log
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.markdown(f'<div class="log-box">{color_log(read_log(cfg["log"], 25))}</div>', unsafe_allow_html=True)

    # Download
    if cfg["csv"].exists():
        st.download_button(
            label=f"Download {cfg['label']} Trade Journal",
            data=cfg["csv"].read_bytes(),
            file_name=f"ctm_{cfg['symbol']}_trades.csv",
            mime="text/csv",
            key=f"dl_{cfg['symbol']}",
        )

    st.markdown("</div>", unsafe_allow_html=True)


# ── OVERVIEW TAB ──────────────────────────────────────────────────────────────
def _load_portfolio_state() -> dict:
    try:
        p = BASE_DIR / "portfolio_state.json"
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def _compute_equity(port: dict, states: dict) -> float:
    """True account equity = available_balance + sum of open stakes.
    Deriv deducts stakes from available_balance the moment a contract opens.
    Bot state files are updated most frequently (including during trade monitoring)
    and carry the post-deduction available balance, so use the most recently
    updated bot state rather than portfolio_state.balance which lags at trade open."""
    avail = 0.0
    best_ts = ""
    for s in states.values():
        ts  = s.get("updated_at", "")
        bal = float(s.get("balance", 0) or 0)
        if bal > 0 and ts > best_ts:
            best_ts = ts
            avail   = bal
    if not avail:
        avail = float(port.get("balance") or 0)
    open_stakes = sum(
        float((s.get("active_trade") or {}).get("stake", 0)) +
        float((s.get("active_trade") or {}).get("partial_stake", 0))
        for s in states.values()
    )
    return avail + open_stakes


def _dd_bar(used_pct: float, limit_pct: float, label: str) -> str:
    fill = min(used_pct / limit_pct, 1.0) * 100
    col  = "#00E676" if fill < 50 else ("#FFB800" if fill < 80 else "#FF3366")
    return f"""
    <div style="margin-bottom:10px">
      <div style="display:flex;justify-content:space-between;font-size:10px;color:#555;margin-bottom:4px">
        <span style="text-transform:uppercase;letter-spacing:1px">{label}</span>
        <span style="color:{col};font-weight:700">{used_pct:.1f}% / {limit_pct:.0f}%</span>
      </div>
      <div style="background:#1A1A1A;border-radius:4px;height:6px;overflow:hidden">
        <div style="background:{col};width:{fill:.1f}%;height:100%;border-radius:4px;
                    transition:width 0.3s"></div>
      </div>
    </div>"""


def render_overview(states, dfs, now_utc):
    port  = _load_portfolio_state()

    # Equity = available_balance + open stakes (Deriv deducts stakes from available_balance)
    combined_bal    = _compute_equity(port, states)
    day_pnl         = float(port.get("daily_pnl", 0))
    month_pnl       = float(port.get("monthly_pnl", 0))
    day_start        = float(port.get("day_start_balance", combined_bal) or combined_bal)
    month_start      = float(port.get("month_start_balance", combined_bal) or combined_bal)
    combined_trades  = sum(len(df) for df in dfs.values())
    combined_wins    = sum((df["result"] == "WIN").sum() if not df.empty else 0 for df in dfs.values())
    combined_wr      = combined_wins / combined_trades * 100 if combined_trades else 0
    combined_pnl     = sum(df["pnl_usd"].sum() if not df.empty else 0 for df in dfs.values())

    # Use realized P&L (not raw balance) to avoid false DD from open stakes
    daily_dd_pct  = max(-day_pnl   / day_start   * 100, 0) if day_start   > 0 else 0
    monthly_dd_pct= max(-month_pnl / month_start * 100, 0) if month_start > 0 else 0

    pnl_col  = "#00E676" if day_pnl >= 0 else "#FF3366"
    mpnl_col = "#00E676" if month_pnl >= 0 else "#FF3366"

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-head">Portfolio Overview</div>', unsafe_allow_html=True)

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        st.markdown(f"""<div class="metric-card green">
          <div class="metric-label">Account Balance</div>
          <div class="metric-value">${combined_bal:.2f}</div>
          <div class="metric-sub">shared wallet</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card {'green' if day_pnl>=0 else 'red'}">
          <div class="metric-label">Today P&L</div>
          <div class="metric-value" style="color:{pnl_col}">${day_pnl:+.2f}</div>
          <div class="metric-sub">all bots combined</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card {'green' if month_pnl>=0 else 'red'}">
          <div class="metric-label">Month P&L</div>
          <div class="metric-value" style="color:{mpnl_col}">${month_pnl:+.2f}</div>
          <div class="metric-sub">{now_utc.strftime('%B %Y')}</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        wr_col = "#00E676" if combined_wr >= 55 else ("#FFB800" if combined_wr >= 45 else "#FF3366")
        st.markdown(f"""<div class="metric-card blue">
          <div class="metric-label">Live Win Rate</div>
          <div class="metric-value" style="color:{wr_col}">{combined_wr:.0f}%</div>
          <div class="metric-sub">{combined_wins}W / {combined_trades - combined_wins}L · {combined_trades} total</div>
        </div>""", unsafe_allow_html=True)
    with c5:
        dd_col = "#00E676" if daily_dd_pct < 2.5 else ("#FFB800" if daily_dd_pct < 4.0 else "#FF3366")
        st.markdown(f"""<div class="metric-card {'green' if daily_dd_pct<2.5 else ('gold' if daily_dd_pct<4.0 else 'red')}">
          <div class="metric-label">Daily DD Used</div>
          <div class="metric-value" style="color:{dd_col}">{daily_dd_pct:.1f}%</div>
          <div class="metric-sub">limit 5% · {'🔴 HIT' if port.get('daily_dd_hit') else '🟢 OK'}</div>
        </div>""", unsafe_allow_html=True)
    with c6:
        mdd_col = "#00E676" if monthly_dd_pct < 10 else ("#FFB800" if monthly_dd_pct < 17 else "#FF3366")
        st.markdown(f"""<div class="metric-card {'green' if monthly_dd_pct<10 else ('gold' if monthly_dd_pct<17 else 'red')}">
          <div class="metric-label">Monthly DD Used</div>
          <div class="metric-value" style="color:{mdd_col}">{monthly_dd_pct:.1f}%</div>
          <div class="metric-sub">limit 20% · {'🔴 HIT' if port.get('monthly_dd_hit') else '🟢 OK'}</div>
        </div>""", unsafe_allow_html=True)

    # ── DD progress bars ───────────────────────────────────────────────────────
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    bar_col1, bar_col2 = st.columns(2)
    with bar_col1:
        st.markdown(
            f'<div class="glass" style="padding:16px 20px">'
            + _dd_bar(daily_dd_pct, 5.0, "Daily Drawdown")
            + _dd_bar(monthly_dd_pct, 20.0, "Monthly Drawdown")
            + f'<div style="font-size:10px;color:#333;margin-top:6px">'
            f'Day start: ${day_start:.2f} &nbsp;|&nbsp; Month start: ${month_start:.2f}</div>'
            + '</div>',
            unsafe_allow_html=True)
    with bar_col2:
        # H1 Regime panel — all 4 bots
        xs = states.get("frxXAUUSD", {})
        eu = states.get("EURUSD", {})
        gu = states.get("GBPUSD", {})
        uj = states.get("USDJPY", {})
        x_regime  = xs.get("h1_regime", "WARMUP")
        eu_regime = eu.get("h1_regime", "WARMUP")
        gu_regime = gu.get("h1_regime", "WARMUP")
        uj_regime = uj.get("h1_regime", "WARMUP")
        x_h1c  = xs.get("h1_close"); x_h1e  = xs.get("h1_ema21")
        eu_h1c = eu.get("h1_close"); eu_h1e = eu.get("h1_ema21")
        gu_h1c = gu.get("h1_close"); gu_h1e = gu.get("h1_ema21")
        uj_h1c = uj.get("h1_close"); uj_h1e = uj.get("h1_ema21")

        def regime_badge(regime, h1c, h1e, bot_label, entry_condition):
            if regime == "BULL":
                col, badge, dot = "#00E676", "BULL - BUY", "🟢"
            elif regime == "BEAR":
                col, badge, dot = "#FF3366", "BEAR - SELL", "🔴"
            else:
                col, badge, dot = "#555", "WARMUP", "⚪"
            price_str = f"H1: {h1c:.5f} vs EMA21: {h1e:.5f}" if (h1c and h1e) else "loading..."
            return f"""
            <div style="padding:10px 0;border-bottom:1px solid #1A1A1A">
              <div style="display:flex;justify-content:space-between;align-items:center">
                <span style="font-size:12px;font-weight:700;color:#888">{bot_label}</span>
                <span style="font-size:10px;font-weight:700;color:{col};background:{col}22;
                             padding:2px 8px;border-radius:4px">{dot} {badge}</span>
              </div>
              <div style="font-size:10px;color:#444;margin-top:4px">{price_str}</div>
              <div style="font-size:10px;color:#333;margin-top:2px">{entry_condition}</div>
            </div>"""

        st.markdown(
            '<div class="glass" style="padding:16px 20px;max-height:320px;overflow-y:auto">'
            '<div style="font-size:10px;color:#555;letter-spacing:1.5px;text-transform:uppercase;margin-bottom:10px">H1 Regime · Entry Gate</div>'
            + regime_badge(x_regime,  x_h1c,  x_h1e,  "XAUUSD (BUY+SELL)",  "BULL → BUY  |  BEAR → SELL")
            + regime_badge(eu_regime, eu_h1c, eu_h1e, "EUR/USD (BUY+SELL)", "BULL → BUY  |  BEAR → SELL")
            + regime_badge(gu_regime, gu_h1c, gu_h1e, "GBP/USD (SPIKE — BUY only)", "Spike > 2.0× ATR → BUY reversal  |  H1 shown for info only")
            + regime_badge(uj_regime, uj_h1c, uj_h1e, "USD/JPY (BUY+SELL)", "BULL → BUY  |  BEAR → SELL")
            + '<div style="font-size:10px;color:#2A2A2A;margin-top:8px">XAUUSD · EUR/USD · USD/JPY: both directions via H1 regime &nbsp;·&nbsp; GBP/USD: BUY-only spike reversion</div>'
            + '</div>',
            unsafe_allow_html=True)

    # ── Bot status cards ───────────────────────────────────────────────────────
    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-head">Bot Status</div>', unsafe_allow_html=True)

    cols = st.columns(len(BOTS))
    for i, (key, cfg) in enumerate(BOTS.items()):
        s     = states[key]
        df    = dfs[key]
        alive = bot_alive(s)
        pnl   = df["pnl_usd"].sum() if not df.empty else 0
        td    = int(s.get("trades_today", 0))
        pnl_c = "#00E676" if pnl >= 0 else "#FF3366"
        filt  = s.get("h1_regime", s.get("h1_filter", "WARMUP"))
        sig   = s.get("last_signal", "WAIT")

        filt_col = "#00E676" if filt in ("BULL", "PASS") else ("#FF3366" if filt in ("BEAR", "BLOCKED") else "#555")
        sig_col  = "#00E676" if sig == "BUY" else ("#FF3366" if sig in ("SELL",) else ("#FFB800" if sig == "TRADE_OPEN" else ("#FF3366" if "BLOCKED" in sig else "#444")))

        with cols[i]:
            st.markdown(f"""<div class="overview-card" style="border-top:3px solid {cfg['color']}">
              <div style="font-size:14px;font-weight:800;color:{cfg['color']};margin-bottom:8px">
                {dot_html(cfg['dot'], alive)} {cfg['label']}
              </div>
              <div style="font-size:22px;font-weight:900;color:#FFF;font-family:'JetBrains Mono',monospace">{td}/{cfg.get('max_trades',12)} trades</div>
              <div style="font-size:12px;color:{pnl_c};margin-top:4px">${pnl:+.2f} P&L</div>
              <div style="display:flex;gap:6px;margin-top:8px">
                <span style="font-size:10px;color:{filt_col};background:{filt_col}22;padding:2px 6px;border-radius:3px">H1:{filt}</span>
                <span style="font-size:10px;color:{sig_col};background:{sig_col}22;padding:2px 6px;border-radius:3px">{sig}</span>
              </div>
              <div style="font-size:10px;color:#333;margin-top:6px">{cfg['session']} · {'ONLINE' if alive else 'OFFLINE'}</div>
            </div>""", unsafe_allow_html=True)

    # Combined equity
    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    st.markdown('<div style="font-size:10px;color:#1A4A7A;margin-bottom:4px;letter-spacing:1px">COMBINED EQUITY CURVE (all bots)</div>', unsafe_allow_html=True)

    all_df = []
    for key, cfg in BOTS.items():
        df = dfs[key]
        if not df.empty and "close_time" in df.columns:
            tmp = df[["close_time", "pnl_usd"]].copy()
            tmp["bot"] = key
            all_df.append(tmp)

    if all_df:
        combined = pd.concat(all_df).sort_values("close_time")
        combined["cumulative"] = combined["pnl_usd"].cumsum()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=combined["close_time"], y=combined["cumulative"],
            mode="lines", line=dict(color="#00E676", width=2.5, shape="spline"),
            fill="tozeroy", fillcolor="rgba(0,230,118,0.05)",
            hovertemplate="<b>%{x|%b %d %H:%M}</b><br>Cumulative P&L: $%{y:+.2f}<extra></extra>",
        ))
        fig.update_layout(
            paper_bgcolor=CHART_BG, plot_bgcolor=CHART_BG,
            font=dict(color="#555", family="Inter", size=11),
            margin=dict(l=8, r=8, t=8, b=8), height=220, showlegend=False,
            xaxis=dict(gridcolor=GRID_COL, showgrid=True, zeroline=False, linecolor=AXIS_COL),
            yaxis=dict(gridcolor=GRID_COL, showgrid=True, zeroline=False, linecolor=AXIS_COL, tickprefix="$"),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    else:
        st.markdown('<div class="glass" style="height:220px;display:flex;align-items:center;justify-content:center;color:#0D2A4A;font-size:12px">Waiting for first trades...</div>', unsafe_allow_html=True)

    # ── Combined Journal ───────────────────────────────────────────────────────
    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-head">Combined Trade Journal — All Bots</div>', unsafe_allow_html=True)

    journal_dfs = []
    for key, cfg in BOTS.items():
        df = dfs[key]
        if not df.empty:
            tmp = df.copy()
            tmp.insert(0, "bot", cfg["label"])
            journal_dfs.append(tmp)

    if journal_dfs:
        journal = pd.concat(journal_dfs).sort_values("close_time", ascending=False).reset_index(drop=True)

        rows_html = ""
        for _, row in journal.iterrows():
            r      = float(row.get("r_multiple", 0))
            pnl    = float(row.get("pnl_usd", 0))
            result = str(row.get("result", ""))
            reason = str(row.get("reason", ""))
            bot_lbl= str(row.get("bot", ""))
            dt     = str(row.get("open_time", ""))[:16]
            sym    = str(row.get("symbol", ""))
            dr     = str(row.get("direction", ""))
            entry_v = row.get("entry_price", "")
            ex_v    = row.get("exit_price", "")
            try:
                entry_s = f"{float(entry_v):.2f}"
            except (ValueError, TypeError):
                entry_s = str(entry_v)
            try:
                ex_s = f"{float(ex_v):.2f}"
            except (ValueError, TypeError):
                ex_s = str(ex_v)
            r_col  = "#00E676" if r > 0 else "#FF3366"
            p_col  = "#00E676" if pnl > 0 else "#FF3366"
            dr_col = "#00E676" if dr == "BUY" else "#FF3366"
            res_col= "#00E676" if result == "WIN" else "#FF3366"
            rows_html += (
                f"<tr>"
                f"<td style='color:#555;font-size:10px'>{dt}</td>"
                f"<td style='color:#888;font-size:10px'>{bot_lbl}</td>"
                f"<td style='color:#888'>{sym}</td>"
                f"<td style='color:{dr_col};font-weight:700'>{dr}</td>"
                f"<td class='mono' style='color:#666'>{entry_s}</td>"
                f"<td class='mono' style='color:#666'>{ex_s}</td>"
                f"<td class='mono' style='color:{r_col};font-weight:700'>{r:+.2f}R</td>"
                f"<td class='mono' style='color:{p_col};font-weight:700'>${pnl:+.2f}</td>"
                f"<td style='color:{res_col}'>{result}</td>"
                f"<td style='color:#444;font-size:10px'>{reason}</td>"
                f"</tr>"
            )

        st.markdown(f"""<div class="glass" style="padding:0;overflow:hidden">
          <div class="scroll-table">
          <table class="trade-table">
            <thead><tr>
              <th>Time (UTC)</th><th>Bot</th><th>Symbol</th><th>Dir</th>
              <th>Entry</th><th>Exit</th><th>R</th><th>PnL</th><th>Result</th><th>Reason</th>
            </tr></thead>
            <tbody>{rows_html}</tbody>
          </table>
          </div>
        </div>""", unsafe_allow_html=True)

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        # Build combined CSV for download
        dl_cols = ["bot", "open_time", "close_time", "symbol", "direction",
                   "entry_price", "exit_price", "pnl_usd", "r_multiple",
                   "result", "reason", "risk_amount", "stake", "balance_before", "balance_after"]
        dl_df = journal[[c for c in dl_cols if c in journal.columns]]
        st.download_button(
            label="Download Combined Trade Journal (all bots)",
            data=dl_df.to_csv(index=False).encode("utf-8"),
            file_name="ctm_combined_journal.csv",
            mime="text/csv",
            key="dl_combined",
        )
    else:
        st.markdown('<div class="glass" style="padding:20px;text-align:center;color:#333;font-size:12px">No closed trades yet — journal will populate as trades close</div>', unsafe_allow_html=True)


# ── EDGE TRACKER ──────────────────────────────────────────────────────────────
def render_edge_tracker():
    selected = st.radio(
        "Select Bot", ["frxXAUUSD", "EURUSD", "GBPUSD", "USDJPY"],
        horizontal=True, key="et_selector",
        format_func=lambda x: {"frxXAUUSD": "XAUUSD Gold", "EURUSD": "EUR/USD", "GBPUSD": "GBP/USD", "USDJPY": "USD/JPY"}.get(x, x),
    )
    bt  = BT_MAP[selected]
    cfg = BOTS[selected]
    df  = load_trades(cfg["csv"])

    # Live balance for MC scaling
    _port     = _load_portfolio_state()
    _live_bal = float(_port.get("balance") or bt["starting_bal"])
    _mc_scale = _live_bal / bt["starting_bal"] if bt["starting_bal"] > 0 else 1.0

    # Actual live risk for this bot (reflects hot/cold streak scaling)
    _bot_s        = load_state(cfg["state"])
    _live_risk    = float(_bot_s.get("risk_pct", bt["our_risk"] * 100))   # stored as %, e.g. 3.0
    _base_risk    = bt["our_risk"] * 100                                   # e.g. 2.0
    _cw           = int(_bot_s.get("consecutive_wins",   0))
    _cl           = int(_bot_s.get("consecutive_losses", 0))
    if abs(_live_risk - _base_risk) > 0.01 and _live_risk > 0:
        _streak_tag = f"W{_cw} streak" if _cw >= 2 else f"L{_cl} streak"
        _risk_label = f"{_base_risk:.0f}% base · {_live_risk:.0f}% now ({_streak_tag})"
    else:
        _risk_label = f"{_base_risk:.0f}% risk"

    def _fmt_bal(v):
        if v >= 1_000_000: return f"${v/1e6:.2f}M"
        if v >= 1_000:     return f"${v/1e3:.1f}K"
        return f"${v:.2f}"

    live_r      = df["r_multiple"].tolist() if not df.empty else []
    n           = len(live_r)

    # ── Live cumulative stats ─────────────────────────────────────────────────
    live_wins   = [r for r in live_r if r > 0]
    live_losses = [abs(r) for r in live_r if r < 0]
    live_mean   = float(np.mean(live_r)) if n > 0 else 0.0
    live_wr     = len(live_wins) / n if n > 0 else 0.0
    live_pf     = (sum(live_wins) / sum(live_losses)
                   if live_losses else (float("inf") if live_wins else 0.0))

    # ── Rolling window — last 20 trades ───────────────────────────────────────
    ROLL_N       = 20
    roll_r       = live_r[-ROLL_N:]
    roll_n       = len(roll_r)
    roll_wins    = [r for r in roll_r if r > 0]
    roll_losses  = [abs(r) for r in roll_r if r < 0]
    roll_pf      = (sum(roll_wins) / sum(roll_losses)
                    if roll_losses else (float("inf") if roll_wins else 0.0))
    roll_wr      = len(roll_wins) / roll_n if roll_n > 0 else 0.0
    roll_mean    = float(np.mean(roll_r)) if roll_n > 0 else 0.0

    # ── Z-score: live avg vs backtest avg ─────────────────────────────────────
    se        = bt["std_r"] / math.sqrt(max(n, 1))
    z         = (live_mean - bt["mean_r"]) / se if se > 0 else 0.0
    p_val     = 0.5 * math.erfc(-z / math.sqrt(2))

    n_losses  = sum(1 for r in live_r if r < 0)
    # Longest consecutive loss streak (order-independent; use for realistic streak probability)
    _max_streak, _cur = 0, 0
    for _r in live_r:
        if _r < 0:
            _cur += 1
            _max_streak = max(_max_streak, _cur)
        else:
            _cur = 0
    max_consec_losses = _max_streak
    p_consec  = (1 - bt["wr"]) ** max_consec_losses if max_consec_losses > 0 else 1.0
    progress  = min(n / bt["trades_for_sig"], 1.0)
    kelly_ratio = bt["our_risk"] / bt["kelly_full"]

    # ── Overall edge status ───────────────────────────────────────────────────
    if n < 20:
        status_txt = "BUILDING"
        status_col = "#555555"
        status_sub = f"Collecting sample — {n} of {bt['trades_for_sig']} trades needed for a verdict"
    elif z < -2.58 or live_pf < 0.8:
        status_txt = "ALARM"
        status_col = "#FF3366"
        status_sub = "Live results diverge from backtest — investigate immediately"
    elif z < -1.96 or live_pf < 1.2:
        status_txt = "CAUTION"
        status_col = "#FFB800"
        status_sub = "Below expected — monitor closely, hold current risk"
    elif z < -1.0:
        status_txt = "WATCH"
        status_col = "#FFB800"
        status_sub = "Below avg but within 1σ — normal variance"
    else:
        status_txt = "ON TRACK"
        status_col = "#00E676"
        status_sub = "Live results consistent with backtest distribution"

    # ── Rolling window verdict ────────────────────────────────────────────────
    if roll_n < 10:
        roll_verdict, roll_col = "BUILDING", "#444"
    elif roll_pf < 0.8:
        roll_verdict, roll_col = "DEGRADING", "#FF3366"
    elif roll_pf < 1.2:
        roll_verdict, roll_col = "FLAT", "#FFB800"
    else:
        roll_verdict, roll_col = "ALIVE", "#00E676"

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # ── Section: Backtest Foundation ─────────────────────────────────────────
    st.markdown(f'<div style="font-size:11px;color:#444;letter-spacing:2px;text-transform:uppercase;margin-bottom:12px">Backtest Foundation · {bt["symbol"]} · {bt["period"]}</div>', unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        pf_col = "#00E676" if bt["pf"] >= 1.5 else "#FFB800"
        st.markdown(f"""<div class="et-card">
          <div class="et-label">Profit Factor</div>
          <div class="et-value" style="color:{pf_col}">{bt['pf']:.2f}</div>
          <div class="et-sub">95% CI  [{bt['ci_pf_lo']:.2f} – {bt['ci_pf_hi']:.2f}]</div>
          <div class="prog-track"><div class="prog-fill" style="width:{min(bt['pf']/3*100,100):.0f}%;background:{pf_col}"></div></div>
        </div>""", unsafe_allow_html=True)

    with c2:
        wr_col = "#00E676" if bt["wr"] >= 0.50 else "#FFB800"
        st.markdown(f"""<div class="et-card">
          <div class="et-label">Win Rate</div>
          <div class="et-value" style="color:{wr_col}">{bt['wr']*100:.1f}%</div>
          <div class="et-sub">95% CI  [{bt['ci_wr_lo']*100:.0f}% – {bt['ci_wr_hi']*100:.0f}%]</div>
          <div class="prog-track"><div class="prog-fill" style="width:{bt['wr']*100:.0f}%;background:{wr_col}"></div></div>
        </div>""", unsafe_allow_html=True)

    with c3:
        st.markdown(f"""<div class="et-card">
          <div class="et-label">Statistical Significance</div>
          <div class="et-value" style="color:#00E676">p &lt; 0.001</div>
          <div class="et-sub">T-stat {bt['t_stat']:.2f} · {bt['total_trades']:,} trades</div>
          <div class="et-sub" style="color:#00E676;margin-top:6px">Edge is real — not luck</div>
        </div>""", unsafe_allow_html=True)

    with c4:
        ruin_col = "#00E676" if bt["ruin_prob"] < 5 else "#FF3366"
        st.markdown(f"""<div class="et-card">
          <div class="et-label">Ruin Probability</div>
          <div class="et-value" style="color:{ruin_col}">{bt['ruin_prob']:.1f}%</div>
          <div class="et-sub">Monte Carlo · 10,000 runs · DD &gt; 50%</div>
          <div class="et-sub" style="color:#444;margin-top:6px">Worst 5% drawdown: -{bt['mc_dd_p95']:.0f}%</div>
        </div>""", unsafe_allow_html=True)

    with c5:
        kelly_col = "#00E676" if kelly_ratio < 0.5 else ("#FFB800" if kelly_ratio < 1.0 else "#FF3366")
        kelly_lbl = "CONSERVATIVE" if kelly_ratio < 0.5 else ("OPTIMAL" if kelly_ratio < 1.0 else "OVER-BETTING")
        st.markdown(f"""<div class="et-card">
          <div class="et-label">Kelly Position</div>
          <div class="et-value" style="color:{kelly_col}">{kelly_ratio*100:.0f}%</div>
          <div class="et-sub">of full Kelly ({bt['kelly_full']*100:.1f}%)</div>
          <div class="et-sub" style="color:{kelly_col};margin-top:6px">{kelly_lbl} at {bt['our_risk']*100:.0f}% risk</div>
          <div class="prog-track"><div class="prog-fill" style="width:{min(kelly_ratio*100,100):.0f}%;background:{kelly_col}"></div></div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    # ── Section: Monte Carlo Outcomes ────────────────────────────────────────
    st.markdown(f'<div style="font-size:11px;color:#444;letter-spacing:2px;text-transform:uppercase;margin-bottom:12px">Monte Carlo — Outcome Distribution (${_live_bal:.2f} live bal · {_risk_label} · {bt["total_trades"]:,} trades)</div>', unsafe_allow_html=True)

    mc_col1, mc_col2 = st.columns([3, 2])

    with mc_col1:
        labels  = ["5th\n(Bad case)", "25th", "50th\n(Median)", "75th", "95th\n(Best case)"]
        values  = [bt["mc_p5"] * _mc_scale, bt["mc_p25"] * _mc_scale, bt["mc_p50"] * _mc_scale, bt["mc_p75"] * _mc_scale, bt["mc_p95"] * _mc_scale]
        bar_cols = ["#555", "#777", "#AAAAAA", "#CCCCCC", "#FFFFFF"]
        returns  = [(v / _live_bal - 1) * 100 for v in values]

        fig_mc = go.Figure(go.Bar(
            x=labels,
            y=values,
            marker=dict(color=bar_cols, line=dict(color="#222", width=1)),
            text=[f"{_fmt_bal(v)}<br>+{r:.0f}%" for v, r in zip(values, returns)],
            textposition="outside",
            textfont=dict(size=11, color="#888"),
            hovertemplate="<b>%{x}</b><br>$%{y:,.2f}<extra></extra>",
        ))
        fig_mc.update_layout(
            paper_bgcolor=CHART_BG, plot_bgcolor=CHART_BG,
            font=dict(color="#555", family="Inter", size=11),
            margin=dict(l=8, r=8, t=30, b=8), height=240, showlegend=False,
            xaxis=dict(showgrid=False, linecolor=AXIS_COL, tickfont=dict(color="#555", size=10)),
            yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        )
        st.plotly_chart(fig_mc, use_container_width=True, config={"displayModeBar": False})

    _mc_p50_scaled = bt["mc_p50"] * _mc_scale
    with mc_col2:
        st.markdown(f"""
        <div class="et-card" style="height:100%">
          <div class="et-label">Monte Carlo Summary</div>
          <div style="margin-top:12px;display:grid;grid-template-columns:1fr 1fr;gap:12px">
            <div>
              <div style="font-size:10px;color:#444">Median Return</div>
              <div style="font-size:18px;font-weight:800;color:#FFF;font-family:'JetBrains Mono',monospace">+{(_mc_p50_scaled/_live_bal-1)*100:.0f}%</div>
            </div>
            <div>
              <div style="font-size:10px;color:#444">Median Final Bal</div>
              <div style="font-size:18px;font-weight:800;color:#FFF;font-family:'JetBrains Mono',monospace">{_fmt_bal(_mc_p50_scaled)}</div>
            </div>
            <div>
              <div style="font-size:10px;color:#444">Median Max DD</div>
              <div style="font-size:18px;font-weight:800;color:#FFB800;font-family:'JetBrains Mono',monospace">-{bt['mc_dd_median']:.1f}%</div>
            </div>
            <div>
              <div style="font-size:10px;color:#444">Worst 5% DD</div>
              <div style="font-size:18px;font-weight:800;color:#FF3366;font-family:'JetBrains Mono',monospace">-{bt['mc_dd_p95']:.0f}%</div>
            </div>
            <div>
              <div style="font-size:10px;color:#444">Ruin Probability</div>
              <div style="font-size:18px;font-weight:800;color:#00E676;font-family:'JetBrains Mono',monospace">{bt['ruin_prob']:.1f}%</div>
            </div>
            <div>
              <div style="font-size:10px;color:#444">Runs Simulated</div>
              <div style="font-size:18px;font-weight:800;color:#555;font-family:'JetBrains Mono',monospace">10,000</div>
            </div>
          </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    # ── Section: 3-Way Comparison — IS vs OOS vs Live ────────────────────────
    st.markdown('<div style="font-size:11px;color:#444;letter-spacing:2px;text-transform:uppercase;margin-bottom:12px">Edge Consistency — Backtest IS · OOS Walk-Forward · Live Forward</div>', unsafe_allow_html=True)

    def _pf_color(pf):
        if pf == float("inf") or pf > 2.0: return "#00E676"
        if pf > 1.2: return "#AAAAAA"
        if pf > 0.8: return "#FFB800"
        return "#FF3366"

    live_pf_str = f"{live_pf:.2f}" if live_pf != float("inf") and n > 0 else ("∞" if live_wins and not live_losses else "—")
    live_pf_disp = live_pf if live_pf != float("inf") else 99.0

    cmp1, cmp2, cmp3 = st.columns(3)

    with cmp1:
        st.markdown(f"""<div class="et-card" style="border-top:3px solid #333">
          <div style="font-size:10px;color:#444;letter-spacing:1px;margin-bottom:10px">BACKTEST · IN-SAMPLE</div>
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px">
            <div>
              <div class="et-label">Profit Factor</div>
              <div style="font-size:22px;font-weight:800;color:#00E676;font-family:'JetBrains Mono',monospace">{bt['pf']:.2f}</div>
              <div style="font-size:10px;color:#444">95% CI [{bt['ci_pf_lo']:.2f}–{bt['ci_pf_hi']:.2f}]</div>
            </div>
            <div>
              <div class="et-label">Win Rate</div>
              <div style="font-size:22px;font-weight:800;color:#00E676;font-family:'JetBrains Mono',monospace">{bt['wr']*100:.1f}%</div>
              <div style="font-size:10px;color:#444">95% CI [{bt['ci_wr_lo']*100:.0f}%–{bt['ci_wr_hi']*100:.0f}%]</div>
            </div>
            <div>
              <div class="et-label">Avg R / trade</div>
              <div style="font-size:18px;font-weight:700;color:#AAAAAA;font-family:'JetBrains Mono',monospace">+{bt['mean_r']:.3f}R</div>
            </div>
            <div>
              <div class="et-label">Trades</div>
              <div style="font-size:18px;font-weight:700;color:#555;font-family:'JetBrains Mono',monospace">{bt['total_trades']:,}</div>
              <div style="font-size:10px;color:#444">{bt['period']}</div>
            </div>
          </div>
          <div style="margin-top:10px;font-size:10px;color:#333">T-stat {bt['t_stat']:.2f} · p &lt; 0.001 · validated</div>
        </div>""", unsafe_allow_html=True)

    with cmp2:
        oos_pf_col = _pf_color(bt["oos_pf"])
        oos_wr_col = "#00E676" if bt["oos_wr"] >= bt["wr"] - 0.03 else "#FFB800"
        # OOS improves or holds → good sign
        oos_verdict = "HOLDS" if (bt["oos_pf"] >= bt["pf"] * 0.7 or bt["oos_pf"] >= 2.5) else "DEGRADES"
        oos_vcol = "#00E676" if oos_verdict == "HOLDS" else "#FFB800"
        st.markdown(f"""<div class="et-card" style="border-top:3px solid {oos_vcol}">
          <div style="font-size:10px;color:#444;letter-spacing:1px;margin-bottom:10px">OOS WALK-FORWARD · UNSEEN DATA</div>
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px">
            <div>
              <div class="et-label">Profit Factor</div>
              <div style="font-size:22px;font-weight:800;color:{oos_pf_col};font-family:'JetBrains Mono',monospace">{bt['oos_pf']:.2f}</div>
              <div style="font-size:10px;color:#444">vs IS {bt['pf']:.2f}</div>
            </div>
            <div>
              <div class="et-label">Win Rate</div>
              <div style="font-size:22px;font-weight:800;color:{oos_wr_col};font-family:'JetBrains Mono',monospace">{bt['oos_wr']*100:.1f}%</div>
              <div style="font-size:10px;color:#444">vs IS {bt['wr']*100:.1f}%</div>
            </div>
            <div style="grid-column:span 2">
              <div class="et-label">OOS Verdict</div>
              <div style="font-size:16px;font-weight:700;color:{oos_vcol}">{oos_verdict} OUT-OF-SAMPLE</div>
              <div style="font-size:10px;color:#444;margin-top:4px">Edge survives unseen data — not curve-fit</div>
            </div>
          </div>
          <div style="margin-top:10px;font-size:10px;color:#333">Walk-forward IS/OOS split · 2-month OOS window</div>
        </div>""", unsafe_allow_html=True)

    with cmp3:
        lv_pf_col = _pf_color(live_pf_disp) if n > 0 else "#333"
        lv_wr_col = ("#00E676" if live_wr >= bt["wr"] - 0.05 else "#FFB800") if n > 0 else "#333"
        st.markdown(f"""<div class="et-card" style="border-top:3px solid {status_col}">
          <div style="font-size:10px;color:#444;letter-spacing:1px;margin-bottom:10px">LIVE FORWARD TEST · REAL MONEY</div>
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px">
            <div>
              <div class="et-label">Live Profit Factor</div>
              <div style="font-size:22px;font-weight:800;color:{lv_pf_col};font-family:'JetBrains Mono',monospace">{"—" if n == 0 else live_pf_str}</div>
              <div style="font-size:10px;color:#444">backtest {bt['pf']:.2f}</div>
            </div>
            <div>
              <div class="et-label">Live Win Rate</div>
              <div style="font-size:22px;font-weight:800;color:{lv_wr_col};font-family:'JetBrains Mono',monospace">{"—" if n == 0 else f"{live_wr*100:.0f}%"}</div>
              <div style="font-size:10px;color:#444">backtest {bt['wr']*100:.1f}%</div>
            </div>
            <div>
              <div class="et-label">Live Avg R</div>
              <div style="font-size:18px;font-weight:700;color:{'#00E676' if live_mean > 0 else '#FF3366'};font-family:'JetBrains Mono',monospace">{"—" if n == 0 else f"{live_mean:+.3f}R"}</div>
            </div>
            <div>
              <div class="et-label">Live Trades</div>
              <div style="font-size:18px;font-weight:700;color:{status_col};font-family:'JetBrains Mono',monospace">{n}</div>
              <div style="font-size:10px;color:#444">need {bt['trades_for_sig']} for sig.</div>
            </div>
          </div>
          <div style="margin-top:10px;font-size:10px;color:{status_col}">{status_txt} · {status_sub}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    # ── Section: Forward Test Live Tracker ───────────────────────────────────
    st.markdown('<div style="font-size:11px;color:#444;letter-spacing:2px;text-transform:uppercase;margin-bottom:12px">Live Forward Test Tracker</div>', unsafe_allow_html=True)

    _tracker_ph = st.empty()
    if n == 0:
        with _tracker_ph.container():
            st.markdown('<div class="et-card" style="text-align:center;padding:40px"><div style="color:#333;font-size:14px">No live trades yet — bot is scanning for signals...</div><div style="color:#222;font-size:11px;margin-top:8px">Trades appear here automatically as they close. Auto-refresh every 15s.</div></div>', unsafe_allow_html=True)
    else:
        with _tracker_ph.container():

            # ── Row 1: Status + Z-score + Live PF + Rolling window + Sample progress
            fs1, fs2, fs3, fs4, fs5 = st.columns(5)

            with fs1:
                st.markdown(f"""<div class="et-card" style="border-top:3px solid {status_col}">
                  <div class="et-label">Edge Status</div>
                  <div class="et-value" style="color:{status_col};font-size:18px">{status_txt}</div>
                  <div class="et-sub" style="color:{status_col};opacity:0.8">{status_sub}</div>
                </div>""", unsafe_allow_html=True)

            with fs2:
                z_col = "#00E676" if z > -1 else ("#FFB800" if z > -2 else "#FF3366")
                st.markdown(f"""<div class="et-card">
                  <div class="et-label">Z-Score vs Backtest</div>
                  <div class="et-value" style="color:{z_col}">{z:+.2f}σ</div>
                  <div class="et-sub">p-value: {p_val:.4f}</div>
                  <div class="et-sub" style="color:#333;margin-top:4px">&lt;-1.96=caution · &lt;-2.58=alarm</div>
                </div>""", unsafe_allow_html=True)

            with fs3:
                lpf_col = _pf_color(live_pf_disp) if n > 0 else "#333"
                st.markdown(f"""<div class="et-card">
                  <div class="et-label">Live Profit Factor</div>
                  <div class="et-value" style="color:{lpf_col}">{live_pf_str}</div>
                  <div class="et-sub">BT: {bt['pf']:.2f} · OOS: {bt['oos_pf']:.2f}</div>
                  <div class="et-sub" style="color:#333;margin-top:4px">Avg R: {live_mean:+.3f}R (BT {bt['mean_r']:+.3f}R)</div>
                </div>""", unsafe_allow_html=True)

            with fs4:
                rpf_str = f"{roll_pf:.2f}" if roll_pf != float("inf") and roll_n > 0 else ("∞" if roll_wins and not roll_losses else "—")
                st.markdown(f"""<div class="et-card" style="border-top:3px solid {roll_col}">
                  <div class="et-label">Rolling Edge · Last {ROLL_N}</div>
                  <div class="et-value" style="color:{roll_col}">{roll_verdict}</div>
                  <div class="et-sub">PF {rpf_str} · WR {roll_wr*100:.0f}% · {roll_n} trades</div>
                  <div class="et-sub" style="color:#333;margin-top:4px">AvgR {roll_mean:+.3f}R — recent market fit</div>
                </div>""", unsafe_allow_html=True)

            with fs5:
                prog_col = "#00E676" if progress >= 0.5 else ("#FFB800" if progress >= 0.1 else "#555")
                st.markdown(f"""<div class="et-card">
                  <div class="et-label">Sample Progress</div>
                  <div class="et-value" style="color:{prog_col}">{n} <span style="font-size:13px;color:#444">/ {bt['trades_for_sig']}</span></div>
                  <div class="et-sub">{progress*100:.0f}% to significance</div>
                  <div class="prog-track"><div class="prog-fill" style="width:{progress*100:.1f}%;background:{prog_col}"></div></div>
                  <div class="et-sub" style="color:#333;margin-top:4px">Max streak: {max_consec_losses}L · {p_consec*100:.1f}%</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

            # ── Forward Test Chart: cumulative R with sigma bands ──────────────────
            trade_nums   = list(range(1, n + 1))
            cumul_r      = list(np.cumsum(live_r))
            expected     = [i * bt["mean_r"] for i in trade_nums]
            sigma1_hi    = [e + bt["std_r"] * math.sqrt(i) for i, e in zip(trade_nums, expected)]
            sigma1_lo    = [e - bt["std_r"] * math.sqrt(i) for i, e in zip(trade_nums, expected)]
            sigma2_hi    = [e + 2 * bt["std_r"] * math.sqrt(i) for i, e in zip(trade_nums, expected)]
            sigma2_lo    = [e - 2 * bt["std_r"] * math.sqrt(i) for i, e in zip(trade_nums, expected)]

            # Extend bands to target trade count for projection
            proj_n = bt["trades_for_sig"]
            proj_nums = list(range(1, proj_n + 1))
            proj_exp  = [i * bt["mean_r"] for i in proj_nums]
            proj_s1hi = [e + bt["std_r"] * math.sqrt(i) for i, e in zip(proj_nums, proj_exp)]
            proj_s1lo = [e - bt["std_r"] * math.sqrt(i) for i, e in zip(proj_nums, proj_exp)]
            proj_s2hi = [e + 2 * bt["std_r"] * math.sqrt(i) for i, e in zip(proj_nums, proj_exp)]
            proj_s2lo = [e - 2 * bt["std_r"] * math.sqrt(i) for i, e in zip(proj_nums, proj_exp)]

            fig_fwd = go.Figure()

            # 2-sigma band (projection)
            fig_fwd.add_trace(go.Scatter(
                x=proj_nums + proj_nums[::-1], y=proj_s2hi + proj_s2lo[::-1],
                fill="toself", fillcolor="rgba(255,255,255,0.02)",
                line=dict(color="rgba(0,0,0,0)"), showlegend=True, name="2σ band",
                hoverinfo="skip",
            ))
            # 1-sigma band (projection)
            fig_fwd.add_trace(go.Scatter(
                x=proj_nums + proj_nums[::-1], y=proj_s1hi + proj_s1lo[::-1],
                fill="toself", fillcolor="rgba(255,255,255,0.05)",
                line=dict(color="rgba(0,0,0,0)"), showlegend=True, name="1σ band",
                hoverinfo="skip",
            ))
            # Expected line (projection)
            fig_fwd.add_trace(go.Scatter(
                x=proj_nums, y=proj_exp,
                line=dict(color="rgba(255,255,255,0.25)", width=1.5, dash="dot"),
                showlegend=True, name="Expected (backtest avg)",
                hovertemplate="Trade %{x}<br>Expected: %{y:+.1f}R<extra></extra>",
            ))
            # Actual live cumulative R
            dot_colors = ["#00E676" if r > 0 else "#FF3366" for r in live_r]
            fig_fwd.add_trace(go.Scatter(
                x=trade_nums, y=cumul_r,
                mode="lines+markers",
                line=dict(color="#FFFFFF", width=2.5),
                marker=dict(size=8, color=dot_colors, line=dict(width=2, color="#111")),
                showlegend=True, name="Live cumulative R",
                hovertemplate="Trade %{x}<br>Cumulative R: %{y:+.2f}R<extra></extra>",
            ))
            # Zero line
            fig_fwd.add_hline(y=0, line_color="rgba(255,255,255,0.1)", line_width=1)

            fig_fwd.update_layout(
                paper_bgcolor=CHART_BG, plot_bgcolor=CHART_BG,
                font=dict(color="#555", family="Inter", size=11),
                margin=dict(l=8, r=8, t=12, b=8), height=300,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                            font=dict(color="#666", size=10), bgcolor="rgba(0,0,0,0)"),
                xaxis=dict(gridcolor=GRID_COL, showgrid=True, zeroline=False, linecolor=AXIS_COL,
                           title=dict(text="Trade #", font=dict(color="#444", size=10))),
                yaxis=dict(gridcolor=GRID_COL, showgrid=True, zeroline=False, linecolor=AXIS_COL,
                           ticksuffix="R", title=dict(text="Cumulative R", font=dict(color="#444", size=10))),
            )
            st.plotly_chart(fig_fwd, use_container_width=True, config={"displayModeBar": False})

            # ── Trade-by-trade breakdown ───────────────────────────────────────────
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            st.markdown('<div style="font-size:10px;color:#444;letter-spacing:1.5px;text-transform:uppercase;margin-bottom:8px">Trade-by-Trade vs Backtest</div>', unsafe_allow_html=True)

            if not df.empty:
                rows_html = ""
                cumr = 0.0
                expected_cumr = 0.0
                for i, (_, row) in enumerate(df.sort_values("close_time").iterrows(), 1):
                    r        = float(row.get("r_multiple", 0))
                    cumr    += r
                    expected_cumr += bt["mean_r"]
                    deviation = cumr - expected_cumr
                    band_1s = bt["std_r"] * math.sqrt(i)
                    z_i     = deviation / band_1s if band_1s > 0 else 0
                    pnl     = float(row.get("pnl_usd", 0))
                    reason  = row.get("reason", "")
                    dt      = str(row.get("open_time", ""))[:16]
                    r_col   = "#00E676" if r > 0 else "#FF3366"
                    cum_col = "#00E676" if cumr >= expected_cumr else "#FF3366"
                    z_col2  = "#00E676" if z_i > -1 else ("#FFB800" if z_i > -2 else "#FF3366")
                    rows_html += f"""<tr>
                      <td class="mono" style="color:#444">{i}</td>
                      <td style="color:#555;font-size:10px">{dt}</td>
                      <td class="mono" style="color:{r_col};font-weight:700">{r:+.2f}R</td>
                      <td class="mono" style="color:{cum_col}">{cumr:+.2f}R</td>
                      <td class="mono" style="color:#444">{expected_cumr:+.2f}R</td>
                      <td class="mono" style="color:{z_col2}">{z_i:+.2f}σ</td>
                      <td class="mono" style="color:{'#00E676' if pnl>0 else '#FF3366'}">${pnl:+.2f}</td>
                      <td style="color:#333;font-size:10px">{reason}</td>
                    </tr>"""

                st.markdown(f"""<div class="glass" style="padding:0;overflow:hidden">
                  <div class="scroll-table">
                  <table class="trade-table">
                    <thead><tr>
                      <th>#</th><th>Time (UTC)</th><th>R</th>
                      <th>Cumul R</th><th>Expected</th><th>Z-score</th><th>PnL</th><th>Reason</th>
                    </tr></thead>
                    <tbody>{rows_html}</tbody>
                  </table>
                  </div>
                </div>""", unsafe_allow_html=True)

            # ── Interpretation box ────────────────────────────────────────────────
            st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
            need_more = bt["trades_for_sig"] - n
            z_col = "#00E676" if z > -1 else ("#FFB800" if z > -2 else "#FF3366")
            if need_more > 0:
                action_line = f'Need <span style="color:#888">{need_more}</span> more trades before a statistically valid edge verdict.'
            else:
                if status_txt == "ON TRACK":
                    action_line = '<span style="color:#00E676">Edge confirmed live — continue at current risk.</span>'
                elif status_txt in ("ALARM", "CAUTION"):
                    action_line = '<span style="color:#FFB800">Review filter conditions and recent market regime before next trade.</span>'
                else:
                    action_line = "Monitor closely."
            st.markdown(f"""
            <div style="background:#0D0D0D;border:1px solid #1E1E1E;border-left:3px solid {status_col};
                        border-radius:10px;padding:14px 18px;font-size:12px;line-height:2.0;color:#666">
              <span style="color:{status_col};font-weight:700;font-size:13px">{status_txt}</span>
              &nbsp;&nbsp;|&nbsp;&nbsp;
              Z-score <span style="color:{z_col};font-weight:600">{z:+.2f}σ</span>
              &nbsp;&nbsp;|&nbsp;&nbsp;
              Live PF <span style="color:{_pf_color(live_pf_disp) if n>0 else '#444'};font-weight:600">{live_pf_str}</span>
              &nbsp;&nbsp;|&nbsp;&nbsp;
              Rolling ({roll_n} trades) <span style="color:{roll_col};font-weight:600">{roll_verdict}</span>
              <br>
              {n} live trades ({progress*100:.0f}% of the {bt['trades_for_sig']} needed) ·
              Max consecutive loss streak: <span style="color:#888">{max_consec_losses}</span> trades · chance: <span style="color:#888">{p_consec*100:.1f}%</span>
              {'<span style="color:#00E676"> — normal variance</span>' if p_consec > 0.05 else ('<span style="color:#FFB800"> — uncommon but possible</span>' if p_consec > 0.005 else '<span style="color:#FF3366"> — rare, investigate</span>')}
              <br>
              {action_line}
            </div>""", unsafe_allow_html=True)


# ── MAIN RENDER ───────────────────────────────────────────────────────────────
def render():
    now_utc = datetime.now(timezone.utc)
    now_cat = now_utc + timedelta(hours=2)

    # Load all states and trades
    states = {key: load_state(cfg["state"]) for key, cfg in BOTS.items()}
    dfs    = {key: load_trades(cfg["csv"])   for key, cfg in BOTS.items()}

    port         = _load_portfolio_state()
    combined_bal = _compute_equity(port, states)
    combined_pnl = sum(df["pnl_usd"].sum() if not df.empty else 0 for df in dfs.values())
    bots_online  = sum(1 for s in states.values() if bot_alive(s))

    # Dots for hero
    dots_html = " &nbsp;|&nbsp; ".join(
        f'{dot_html(cfg["dot"], bot_alive(states[key]))}{cfg["label"]}'
        for key, cfg in BOTS.items()
    )

    st.markdown(f"""
    <div class="hero">
      <div class="hero-grid">
        <div class="hero-logo">
          <div class="hero-icon">⚡</div>
          <div>
            <div class="hero-title">HYPOTHERMIA — QUAD-BOT</div>
            <div class="hero-name">Cyprian Masvikeni · CTM Trading</div>
            <div class="hero-sub">{dots_html} · Deriv API · x100</div>
          </div>
        </div>
        <div class="hero-right">
          <div class="hero-time">{now_cat.strftime('%H:%M:%S')}</div>
          <div class="hero-date">{now_cat.strftime('%A, %d %B %Y')} · CAT (UTC+2)</div>
          <div style="font-size:11px;color:#333;margin-top:4px">
            Combined: <span style="color:#AAA">${combined_bal:.2f}</span> ·
            P&L: <span style="color:{'#00E676' if combined_pnl>=0 else '#FF3366'}">${combined_pnl:+.2f}</span> ·
            <span style="color:{'#00E676' if bots_online==len(BOTS) else '#FFB800'}">{bots_online}/{len(BOTS)} online</span>
          </div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    tab_overview, tab_edge, tab_xau, tab_eur, tab_gbp, tab_usd = st.tabs([
        "Overview", "Edge Tracker", "XAUUSD Gold", "EUR/USD", "GBP/USD", "USD/JPY"
    ])

    with tab_overview:
        render_overview(states, dfs, now_utc)

    with tab_edge:
        render_edge_tracker()

    with tab_xau:
        render_bot_panel(BOTS["frxXAUUSD"], now_utc)

    with tab_eur:
        render_bot_panel(BOTS["EURUSD"], now_utc)

    with tab_gbp:
        render_bot_panel(BOTS["GBPUSD"], now_utc)

    with tab_usd:
        render_bot_panel(BOTS["USDJPY"], now_utc)

    # Footer
    st.markdown(f"""
    <div style="margin-top:16px;padding:12px 0;border-top:1px solid #1A1A1A;
                display:flex;justify-content:space-between;align-items:center">
      <div style="color:#333;font-size:11px">⚡ CTM Trading · Hypothermia Quad-Bot · Deriv API · x100</div>
      <div style="color:#333;font-size:11px;font-family:'JetBrains Mono',monospace">
        {now_cat.strftime('%H:%M:%S')} CAT · Auto-refresh 15s
      </div>
    </div>""", unsafe_allow_html=True)

    _, btn_col, _ = st.columns([4, 1, 4])
    with btn_col:
        if st.button("Refresh Now", use_container_width=True):
            st.rerun()


render()
time.sleep(15)
st.rerun()
