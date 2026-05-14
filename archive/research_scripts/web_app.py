"""
CTM Trading — Live Dashboard Web App
Flask backend serving real-time bot data
"""
import json
import csv
from pathlib import Path
from datetime import datetime, timezone
from flask import Flask, jsonify, render_template_string, send_file

app = Flask(__name__)

BASE_DIR   = Path(__file__).resolve().parent
STATE_JSON = BASE_DIR / "state.json"
LOG_CSV    = BASE_DIR / "data" / "live_trades.csv"
LOG_TXT    = BASE_DIR / "bot.log"


# ── API ENDPOINTS ─────────────────────────────────────────────────────────────
@app.route("/api/state")
def api_state():
    try:
        if STATE_JSON.exists():
            return jsonify(json.loads(STATE_JSON.read_text(encoding="utf-8")))
    except Exception:
        pass
    return jsonify({})


@app.route("/api/trades")
def api_trades():
    trades = []
    try:
        if LOG_CSV.exists():
            with open(LOG_CSV, newline="", encoding="utf-8") as f:
                trades = list(csv.DictReader(f))
    except Exception:
        pass
    return jsonify(trades[-100:])


@app.route("/api/trades/download")
def api_trades_download():
    if LOG_CSV.exists():
        return send_file(str(LOG_CSV), mimetype="text/csv",
                         as_attachment=True, download_name="ctm_trades.csv")
    return jsonify({"error": "No trade file yet"}), 404


@app.route("/api/log")
def api_log():
    try:
        if LOG_TXT.exists():
            lines = LOG_TXT.read_text(encoding="utf-8", errors="replace").splitlines()
            return jsonify({"lines": lines[-50:]})
    except Exception:
        pass
    return jsonify({"lines": []})


# ── HTML DASHBOARD ────────────────────────────────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CTM Trading — Hypothermia Bot</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600;700&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#030812;--bg2:#060f1f;--bg3:#0a1628;
  --border:rgba(30,80,180,0.18);--border2:rgba(0,120,255,0.25);
  --blue:#0066FF;--cyan:#00BFFF;--green:#00E676;--red:#FF3366;--gold:#FFB800;
  --text:#E8F4FF;--muted:#3A6A9A;--dim:#1A3A6A;
}
html,body{background:var(--bg);color:var(--text);font-family:'Space Grotesk',sans-serif;min-height:100vh;overflow-x:hidden}
::-webkit-scrollbar{width:4px}::-webkit-scrollbar-track{background:var(--bg2)}::-webkit-scrollbar-thumb{background:var(--dim);border-radius:4px}

/* ── NAV ── */
nav{
  position:fixed;top:0;left:0;right:0;z-index:100;
  background:rgba(3,8,18,0.85);backdrop-filter:blur(20px);
  border-bottom:1px solid var(--border);
  padding:0 32px;height:64px;
  display:flex;align-items:center;justify-content:space-between;
}
.nav-brand{display:flex;align-items:center;gap:12px}
.nav-logo{
  width:36px;height:36px;border-radius:8px;
  background:linear-gradient(135deg,#0044CC,#00BFFF);
  display:flex;align-items:center;justify-content:center;
  font-weight:800;font-size:14px;letter-spacing:1px;
  box-shadow:0 0 20px rgba(0,100,255,0.3);
}
.nav-name{font-size:16px;font-weight:700;letter-spacing:0.5px}
.nav-sub{font-size:10px;color:var(--muted);letter-spacing:2px;text-transform:uppercase;margin-top:1px}
.nav-right{display:flex;align-items:center;gap:20px}
.nav-time{font-family:'JetBrains Mono',monospace;font-size:13px;color:var(--cyan)}
.status-pill{
  display:flex;align-items:center;gap:6px;
  padding:5px 12px;border-radius:20px;
  font-size:11px;font-weight:600;letter-spacing:1px;text-transform:uppercase;
  border:1px solid rgba(0,230,118,0.3);background:rgba(0,230,118,0.06);color:var(--green);
}
.status-pill.offline{border-color:rgba(255,51,102,0.3);background:rgba(255,51,102,0.06);color:var(--red)}
@keyframes pulse{0%,100%{opacity:1;transform:scale(1)}50%{opacity:0.4;transform:scale(0.7)}}
.dot{width:7px;height:7px;border-radius:50%;background:currentColor;animation:pulse 1.5s infinite}

/* ── LAYOUT ── */
main{padding:88px 32px 48px;max-width:1400px;margin:0 auto}

/* ── HERO ── */
.hero{
  margin-bottom:40px;
  background:linear-gradient(135deg,rgba(0,40,120,0.15),rgba(0,20,60,0.1));
  border:1px solid var(--border);border-radius:20px;
  padding:32px 36px;position:relative;overflow:hidden;
}
.hero::before{
  content:'';position:absolute;top:-60px;right:-60px;
  width:300px;height:300px;border-radius:50%;
  background:radial-gradient(circle,rgba(0,100,255,0.08),transparent 70%);
  pointer-events:none;
}
.hero-grid{display:grid;grid-template-columns:1fr auto;gap:24px;align-items:center}
.hero-title{font-size:36px;font-weight:700;letter-spacing:-1px;line-height:1.1}
.hero-title span{
  background:linear-gradient(135deg,#4488FF,#00BFFF);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
}
.hero-desc{font-size:13px;color:var(--muted);margin-top:8px;letter-spacing:0.5px}
.hero-balance{text-align:right}
.hero-bal-label{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:2px}
.hero-bal-value{font-family:'JetBrains Mono',monospace;font-size:42px;font-weight:700;color:var(--text);line-height:1.1;margin-top:4px}
.hero-bal-sub{font-size:12px;margin-top:4px}

/* ── GRID ── */
.metrics{display:grid;grid-template-columns:repeat(6,1fr);gap:14px;margin-bottom:28px}
@media(max-width:1100px){.metrics{grid-template-columns:repeat(3,1fr)}}
@media(max-width:700px){.metrics{grid-template-columns:repeat(2,1fr)}}

/* ── CARD ── */
.card{
  background:linear-gradient(135deg,rgba(8,20,50,0.9),rgba(5,12,30,0.95));
  border:1px solid var(--border);border-radius:14px;
  padding:18px 20px;position:relative;overflow:hidden;
  transition:border-color 0.3s,transform 0.2s;
}
.card:hover{border-color:var(--border2);transform:translateY(-1px)}
.card::after{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,#0033AA,#0088FF,#00BFFF)}
.card.green::after{background:linear-gradient(90deg,#003322,#00CC66,#00FF88)}
.card.red::after{background:linear-gradient(90deg,#330011,#CC0044,#FF1155)}
.card.gold::after{background:linear-gradient(90deg,#332200,#CC7700,#FFB800)}
.card.cyan::after{background:linear-gradient(90deg,#002233,#0088AA,#00CCFF)}
.card-icon{position:absolute;top:16px;right:18px;font-size:22px;opacity:0.12}
.card-label{font-size:9px;color:var(--muted);text-transform:uppercase;letter-spacing:2px;font-weight:600}
.card-value{font-family:'JetBrains Mono',monospace;font-size:22px;font-weight:700;margin-top:8px;color:var(--text)}
.card-sub{font-size:10px;color:var(--dim);margin-top:5px}

/* ── SECTION HEADER ── */
.section{margin:32px 0 16px}
.section h2{font-size:11px;font-weight:600;color:var(--dim);text-transform:uppercase;letter-spacing:3px;display:flex;align-items:center;gap:10px}
.section h2::after{content:'';flex:1;height:1px;background:linear-gradient(90deg,var(--border),transparent)}

/* ── LIVE STATUS ── */
.status-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-bottom:28px}
@media(max-width:700px){.status-grid{grid-template-columns:1fr}}
.status-card{
  background:rgba(6,15,31,0.8);border:1px solid var(--border);border-radius:14px;padding:18px 20px;
}
.status-label{font-size:9px;color:var(--muted);text-transform:uppercase;letter-spacing:2px;margin-bottom:8px}
.bias-strong-buy{color:#00FF88;font-size:20px;font-weight:800;text-shadow:0 0 15px rgba(0,255,136,0.4)}
.bias-strong-sell{color:#FF3366;font-size:20px;font-weight:800;text-shadow:0 0 15px rgba(255,51,102,0.4)}
.bias-weak-buy{color:#69F0AE;font-size:20px;font-weight:700}
.bias-weak-sell{color:#FF8A80;font-size:20px;font-weight:700}
.bias-neutral{color:var(--muted);font-size:20px;font-weight:600}

/* ── OPEN TRADE ── */
.trade-banner{
  border-radius:16px;padding:24px 28px;margin-bottom:28px;
  position:relative;overflow:hidden;
}
.trade-banner.buy{background:linear-gradient(135deg,rgba(0,50,25,0.8),rgba(0,30,15,0.9));border:1px solid rgba(0,200,100,0.2)}
.trade-banner.sell{background:linear-gradient(135deg,rgba(50,0,20,0.8),rgba(30,0,12,0.9));border:1px solid rgba(255,50,80,0.2)}
.trade-banner.empty{background:rgba(6,15,31,0.6);border:1px solid var(--border);text-align:center;padding:40px}
.trade-banner::before{content:'';position:absolute;top:0;left:0;right:0;height:2px}
.trade-banner.buy::before{background:linear-gradient(90deg,transparent,#00CC66,transparent)}
.trade-banner.sell::before{background:linear-gradient(90deg,transparent,#FF3366,transparent)}
.trade-dir{font-size:40px;font-weight:800;letter-spacing:-2px}
.trade-dir.buy{color:#00E676;text-shadow:0 0 30px rgba(0,230,118,0.3)}
.trade-dir.sell{color:#FF3366;text-shadow:0 0 30px rgba(255,51,102,0.3)}
.trade-grid{display:grid;grid-template-columns:1.5fr repeat(5,1fr);gap:20px;align-items:center}
@media(max-width:900px){.trade-grid{grid-template-columns:1fr 1fr 1fr}}
.t-label{font-size:9px;color:var(--dim);text-transform:uppercase;letter-spacing:1.5px;margin-bottom:4px}
.t-val{font-family:'JetBrains Mono',monospace;font-size:15px;font-weight:600;color:var(--text)}

/* ── CHARTS ── */
.charts-grid{display:grid;grid-template-columns:2fr 1fr;gap:20px;margin-bottom:28px}
@media(max-width:900px){.charts-grid{grid-template-columns:1fr}}
.chart-card{background:rgba(6,15,31,0.8);border:1px solid var(--border);border-radius:14px;padding:20px}
.chart-title{font-size:9px;color:var(--muted);text-transform:uppercase;letter-spacing:2px;margin-bottom:16px;font-weight:600}

/* ── STATS ROW ── */
.stats-row{display:grid;grid-template-columns:repeat(8,1fr);gap:10px;margin-bottom:28px}
@media(max-width:1100px){.stats-row{grid-template-columns:repeat(4,1fr)}}
@media(max-width:700px){.stats-row{grid-template-columns:repeat(2,1fr)}}
.stat-box{background:rgba(5,12,30,0.8);border:1px solid var(--border);border-radius:10px;padding:12px 14px;text-align:center}
.stat-lbl{font-size:8px;color:var(--dim);text-transform:uppercase;letter-spacing:1.5px}
.stat-val{font-family:'JetBrains Mono',monospace;font-size:16px;font-weight:700;color:var(--text);margin-top:4px}

/* ── TABLE ── */
.table-wrap{background:rgba(6,15,31,0.8);border:1px solid var(--border);border-radius:14px;overflow:hidden;margin-bottom:28px}
table{width:100%;border-collapse:collapse;font-size:12px}
thead th{background:rgba(0,25,60,0.6);color:var(--muted);font-size:9px;text-transform:uppercase;letter-spacing:1.5px;padding:12px 16px;text-align:left;border-bottom:1px solid var(--border);font-weight:600}
tbody tr{border-bottom:1px solid rgba(0,40,90,0.15);transition:background 0.15s}
tbody tr:hover{background:rgba(0,50,120,0.08)}
tbody td{padding:10px 16px}
.win{color:var(--green);font-weight:600}
.loss{color:var(--red);font-weight:600}
.mono{font-family:'JetBrains Mono',monospace}

/* ── LOG ── */
.log-box{
  background:#020810;border:1px solid var(--border);border-radius:14px;
  padding:20px;font-family:'JetBrains Mono',monospace;font-size:11px;
  color:var(--dim);max-height:280px;overflow-y:auto;line-height:1.8;
}
.log-line{display:block}
.log-info{color:#1E4A7A}.log-warn{color:#CC8800}.log-error{color:#CC2244}
.log-open{color:#00AA55}.log-close{color:#0088CC}.log-chand{color:#7755CC}

/* ── EMPTY ── */
.empty{color:var(--dim);font-size:13px;padding:40px;text-align:center}

/* ── FOOTER ── */
footer{border-top:1px solid var(--border);padding:24px 32px;max-width:1400px;margin:0 auto;display:flex;justify-content:space-between;align-items:center}
footer span{font-size:11px;color:var(--dim)}
</style>
</head>
<body>

<!-- NAV -->
<nav>
  <div class="nav-brand">
    <div class="nav-logo">CTM</div>
    <div>
      <div class="nav-name">CTM Trading</div>
      <div class="nav-sub">Hypothermia Bot · Step Index</div>
    </div>
  </div>
  <div class="nav-right">
    <div class="nav-time" id="clock">--:--:--</div>
    <div class="status-pill" id="statusPill"><span class="dot"></span><span id="statusText">LOADING</span></div>
  </div>
</nav>

<main>

  <!-- HERO -->
  <div class="hero">
    <div class="hero-grid">
      <div>
        <div class="hero-title">Cyprian Masvikeni<br><span>Portfolio Dashboard</span></div>
        <div class="hero-desc">Algorithmic Trading · Step Index · Deriv Platform · Oracle Cloud · 24/7</div>
      </div>
      <div class="hero-balance">
        <div class="hero-bal-label">Account Balance</div>
        <div class="hero-bal-value" id="heroBalance">$0.00</div>
        <div class="hero-bal-sub" id="heroPnl" style="color:var(--muted)">+$0.00 total P&L</div>
      </div>
    </div>
  </div>

  <!-- METRIC CARDS -->
  <div class="metrics">
    <div class="card">
      <span class="card-icon">💰</span>
      <div class="card-label">Balance</div>
      <div class="card-value" id="balance">$0.00</div>
      <div class="card-sub" id="balSub">—</div>
    </div>
    <div class="card" id="ddCard">
      <span class="card-icon">📉</span>
      <div class="card-label">Drawdown</div>
      <div class="card-value" id="ddVal">0.0%</div>
      <div class="card-sub" id="ddSub">Peak $0.00</div>
    </div>
    <div class="card green">
      <span class="card-icon">🎯</span>
      <div class="card-label">Win Rate</div>
      <div class="card-value" id="winRate">0%</div>
      <div class="card-sub" id="winSub">0W · 0L · 0 total</div>
    </div>
    <div class="card">
      <span class="card-icon">⚖️</span>
      <div class="card-label">Expectancy</div>
      <div class="card-value" id="expVal">+0.00R</div>
      <div class="card-sub" id="expSub">—</div>
    </div>
    <div class="card" id="riskCard">
      <span class="card-icon">🎲</span>
      <div class="card-label">Risk Mode</div>
      <div class="card-value" id="riskVal">BASE 5%</div>
      <div class="card-sub" id="riskSub">neutral</div>
    </div>
    <div class="card cyan">
      <span class="card-icon">📅</span>
      <div class="card-label">Today's Trades</div>
      <div class="card-value" id="todayTrades">0 <span style="font-size:12px;color:var(--dim)">/ 6</span></div>
      <div class="card-sub" id="todayBar">□□□□□□</div>
    </div>
  </div>

  <!-- LIVE STATUS -->
  <div class="section"><h2>📡 Live Status</h2></div>
  <div class="status-grid">
    <div class="status-card">
      <div class="status-label">Trading Session</div>
      <div id="sessionStatus" style="font-size:15px;font-weight:700;margin-top:4px">—</div>
      <div id="sessionMsg" style="font-size:11px;color:var(--dim);margin-top:6px">—</div>
    </div>
    <div class="status-card">
      <div class="status-label">M15 Trend Bias</div>
      <div id="trendBias" class="bias-neutral" style="margin-top:4px">— NEUTRAL</div>
      <div id="lastSignal" style="font-size:11px;color:var(--dim);margin-top:6px">Signal: WAIT</div>
    </div>
    <div class="status-card">
      <div class="status-label">Last Signal Reason</div>
      <div id="lastReason" style="font-size:12px;color:var(--muted);margin-top:6px;line-height:1.5">—</div>
    </div>
  </div>

  <!-- OPEN POSITION -->
  <div class="section"><h2>🔴 Open Position</h2></div>
  <div id="tradeBox" class="trade-banner empty">
    <div style="font-size:36px;opacity:0.2;margin-bottom:10px">⏳</div>
    <div style="color:var(--dim);font-size:14px;font-weight:600">No Open Position</div>
    <div style="color:var(--dim);font-size:12px;margin-top:6px;opacity:0.6">Bot is scanning for the next entry signal...</div>
  </div>

  <!-- CHARTS -->
  <div class="section"><h2>📈 Performance</h2></div>
  <div class="charts-grid">
    <div class="chart-card">
      <div class="chart-title">Equity Curve</div>
      <canvas id="equityChart" height="200"></canvas>
    </div>
    <div class="chart-card">
      <div class="chart-title">P&amp;L Per Trade (Last 20)</div>
      <canvas id="pnlChart" height="200"></canvas>
    </div>
  </div>

  <!-- STATS -->
  <div class="section"><h2>📊 Statistics</h2></div>
  <div class="stats-row" id="statsRow">
    <div class="stat-box"><div class="stat-lbl">Total</div><div class="stat-val" id="sTot">0</div></div>
    <div class="stat-box"><div class="stat-lbl">Winners</div><div class="stat-val win" id="sWin">0</div></div>
    <div class="stat-box"><div class="stat-lbl">Losers</div><div class="stat-val loss" id="sLoss">0</div></div>
    <div class="stat-box"><div class="stat-lbl">Net P&L</div><div class="stat-val" id="sNet">$0.00</div></div>
    <div class="stat-box"><div class="stat-lbl">Avg Win</div><div class="stat-val win" id="sAvgW">0.00R</div></div>
    <div class="stat-box"><div class="stat-lbl">Avg Loss</div><div class="stat-val loss" id="sAvgL">0.00R</div></div>
    <div class="stat-box"><div class="stat-lbl">R:R Ratio</div><div class="stat-val" id="sRR">0.00x</div></div>
    <div class="stat-box"><div class="stat-lbl">Monsters 3R+</div><div class="stat-val" id="sMon">0</div></div>
  </div>

  <!-- TRADE TABLE -->
  <div class="section" style="display:flex;align-items:center;justify-content:space-between">
    <h2>📋 Trade History</h2>
    <a href="/api/trades/download" download="ctm_trades.csv" style="
      font-size:11px;font-weight:600;letter-spacing:1px;text-transform:uppercase;
      color:var(--cyan);border:1px solid rgba(0,191,255,0.3);background:rgba(0,191,255,0.06);
      padding:6px 14px;border-radius:20px;text-decoration:none;transition:all 0.2s;
      font-family:'Space Grotesk',sans-serif;
    " onmouseover="this.style.background='rgba(0,191,255,0.12)'"
       onmouseout="this.style.background='rgba(0,191,255,0.06)'">
      ↓ Download CSV
    </a>
  </div>
  <div class="table-wrap">
    <table>
      <thead><tr>
        <th>ID</th><th>Time</th><th>Direction</th><th>Entry</th>
        <th>Exit</th><th>Stake</th><th>P&L</th><th>R</th><th>Result</th><th>Peak R</th>
      </tr></thead>
      <tbody id="tradeTable"><tr><td colspan="10" class="empty">No trades yet</td></tr></tbody>
    </table>
  </div>

  <!-- LOG -->
  <div class="section"><h2>🖥️ Bot Log</h2></div>
  <div class="log-box" id="logBox">Loading...</div>

</main>

<footer>
  <span>⚡ CTM Trading · Hypothermia Bot · Step Trend Strategy · Deriv API</span>
  <span id="footerTime" style="font-family:'JetBrains Mono',monospace">Auto-refresh: 5s</span>
</footer>

<script>
let equityChart, pnlChart;
let lastTrades = [];

// ── CLOCK ──────────────────────────────────────────────────────────────────
function updateClock() {
  const now = new Date();
  const cat = new Date(now.getTime() + 2*3600*1000);
  document.getElementById('clock').textContent =
    cat.toISOString().substr(11,8) + ' CAT';
}
setInterval(updateClock, 1000);
updateClock();

// ── UTILS ──────────────────────────────────────────────────────────────────
const $ = id => document.getElementById(id);
function fmt(n, decimals=2) { return parseFloat(n||0).toFixed(decimals); }
function fmtUSD(n) { const v=parseFloat(n||0); return (v>=0?'+':'')+'$'+Math.abs(v).toFixed(2); }
function botAlive(updatedAt) {
  if (!updatedAt) return false;
  const diff = (Date.now() - new Date(updatedAt).getTime()) / 1000;
  return diff < 60;
}

function sessionInfo() {
  const h = new Date().getUTCHours();
  const skip = [11, 14];
  if (skip.includes(h)) return ['SKIP', `Skipped hour — ${h}:00 GMT`, '#FFB800'];
  if (h >= 9 && h < 19) return ['LIVE', `Active until 19:00 GMT`, '#00E676'];
  return ['CLOSED', `Opens 09:00 GMT`, '#FF3366'];
}

// ── CHARTS INIT ────────────────────────────────────────────────────────────
function initCharts() {
  const bg = '#030812';
  const grid = 'rgba(0,60,130,0.1)';

  equityChart = new Chart($('equityChart').getContext('2d'), {
    type: 'line',
    data: { labels: [], datasets: [{
      data: [], borderColor: '#0077FF', borderWidth: 2.5,
      fill: true,
      backgroundColor: ctx => {
        const g = ctx.chart.ctx.createLinearGradient(0,0,0,200);
        g.addColorStop(0,'rgba(0,100,255,0.15)');
        g.addColorStop(1,'rgba(0,100,255,0)');
        return g;
      },
      pointBackgroundColor: [], pointRadius: 5, pointBorderWidth: 2,
      pointBorderColor: bg, tension: 0.4,
    }]},
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false }, tooltip: {
        backgroundColor: '#0A1830', borderColor: '#1A3A6E', borderWidth: 1,
        callbacks: { label: ctx => ` $${ctx.parsed.y.toFixed(2)}` }
      }},
      scales: {
        x: { grid: { color: grid }, ticks: { color: '#2A5A8A', maxTicksLimit: 8, font: { size: 10 } } },
        y: { grid: { color: grid }, ticks: { color: '#2A5A8A', callback: v => '$'+v.toFixed(0), font: { size: 10 } } }
      }
    }
  });

  pnlChart = new Chart($('pnlChart').getContext('2d'), {
    type: 'bar',
    data: { labels: [], datasets: [{
      data: [], backgroundColor: [], borderColor: [], borderWidth: 1.5, borderRadius: 4,
    }]},
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false }, tooltip: {
        backgroundColor: '#0A1830', borderColor: '#1A3A6E', borderWidth: 1,
        callbacks: { label: ctx => ` $${ctx.parsed.y >= 0 ? '+' : ''}${ctx.parsed.y.toFixed(2)}` }
      }},
      scales: {
        x: { display: false },
        y: { grid: { color: grid }, ticks: { color: '#2A5A8A', callback: v => '$'+v, font: { size: 10 } } }
      }
    }
  });
}

// ── UPDATE STATE ───────────────────────────────────────────────────────────
async function updateState() {
  try {
    const s = await fetch('/api/state').then(r=>r.json());
    const alive = botAlive(s.updated_at);

    // Status pill
    const pill = $('statusPill');
    pill.className = 'status-pill' + (alive ? '' : ' offline');
    $('statusText').textContent = alive ? 'ONLINE' : 'OFFLINE';

    const bal = parseFloat(s.balance||0);
    const peak = parseFloat(s.peak_balance||bal);
    const dd = peak > 0 ? (peak-bal)/peak*100 : 0;
    const riskPct = parseFloat(s.risk_pct||5);
    const cw = parseInt(s.consecutive_wins||0);
    const cl = parseInt(s.consecutive_losses||0);

    $('heroBalance').textContent = '$' + fmt(bal);
    $('balance').textContent = '$' + fmt(bal);
    $('ddVal').textContent = fmt(dd,1) + '%';
    $('ddSub').textContent = 'Peak $' + fmt(peak);
    $('ddCard').className = 'card' + (dd > 8 ? ' red' : dd > 4 ? ' gold' : '');

    const todayT = parseInt(s.trades_today||0);
    $('todayTrades').innerHTML = `${todayT} <span style="font-size:12px;color:var(--dim)">/ 6</span>`;
    $('todayBar').textContent = '■'.repeat(todayT) + '□'.repeat(Math.max(0,6-todayT));

    // Risk mode
    let riskLbl = 'BASE', riskCls = '';
    if (cw >= 2) { riskLbl = 'HOT'; riskCls = ' gold'; }
    else if (cl >= 2) { riskLbl = 'COLD'; riskCls = ' cyan'; }
    $('riskCard').className = 'card' + riskCls;
    $('riskVal').textContent = `${riskLbl} ${riskPct.toFixed(0)}%`;
    $('riskSub').textContent = cw>=2 ? `🔥 ${cw} wins` : cl>=2 ? `❄️ ${cl} losses` : 'neutral';

    // Session
    const [sess, sessMsg, sessCol] = sessionInfo();
    $('sessionStatus').innerHTML = `<span style="color:${sessCol}">${sess === 'LIVE' ? '● ' : '○ '}${sess}</span>`;
    $('sessionMsg').textContent = sessMsg;

    // Trend bias
    const biasMap = {
      'STRONG_BUY': ['bias-strong-buy','▲ STRONG BUY'],
      'STRONG_SELL': ['bias-strong-sell','▼ STRONG SELL'],
      'WEAK_BUY': ['bias-weak-buy','△ WEAK BUY'],
      'WEAK_SELL': ['bias-weak-sell','▽ WEAK SELL'],
      'NEUTRAL': ['bias-neutral','— NEUTRAL'],
    };
    const [bcls, btxt] = biasMap[s.trend_bias] || ['bias-neutral', s.trend_bias || '— NEUTRAL'];
    $('trendBias').className = bcls;
    $('trendBias').textContent = btxt;
    $('lastSignal').textContent = `Signal: ${s.last_signal || 'WAIT'}`;
    $('lastReason').textContent = s.last_reason || '—';

    // Open trade
    const at = s.active_trade;
    const tradeBox = $('tradeBox');
    if (at) {
      const d = at.direction || 'BUY';
      const dcls = d.toLowerCase();
      const arrow = d === 'BUY' ? '▲ LONG' : '▼ SHORT';
      const pnl = parseFloat(at.current_pnl||0);
      const pnlCol = pnl >= 0 ? 'var(--green)' : 'var(--red)';
      const pnlStr = (pnl >= 0 ? '+' : '') + '$' + Math.abs(pnl).toFixed(2);
      const openDt = at.open_time ? new Date(at.open_time) : null;
      let holdStr = '—';
      if (openDt) {
        const mins = Math.floor((Date.now() - openDt)/60000);
        holdStr = mins >= 60 ? `${Math.floor(mins/60)}h ${mins%60}m` : `${mins}m`;
      }
      tradeBox.className = `trade-banner ${dcls}`;
      tradeBox.innerHTML = `
        <div class="trade-grid">
          <div><div class="trade-dir ${dcls}">${arrow}</div><div style="font-size:12px;color:var(--muted);margin-top:4px">ID: ${at.trade_id||'—'}</div></div>
          <div><div class="t-label">Entry</div><div class="t-val">${parseFloat(at.entry||0).toFixed(1)}</div></div>
          <div><div class="t-label">SL (USD)</div><div class="t-val" style="color:#FF8A65">$${fmt(at.sl_usd)}</div></div>
          <div><div class="t-label">Live P&L</div><div class="t-val" style="color:${pnlCol}">${pnlStr}</div></div>
          <div><div class="t-label">Peak R</div><div class="t-val">${fmt(at.peak_r,2)}R</div></div>
          <div><div class="t-label">Hold</div><div class="t-val">${holdStr}</div></div>
        </div>`;
    } else {
      tradeBox.className = 'trade-banner empty';
      tradeBox.innerHTML = `
        <div style="font-size:36px;opacity:0.2;margin-bottom:10px">⏳</div>
        <div style="color:var(--dim);font-size:14px;font-weight:600">No Open Position</div>
        <div style="color:var(--dim);font-size:12px;margin-top:6px;opacity:0.6">Bot is scanning for the next entry signal...</div>`;
    }

  } catch(e) { console.error('State error:', e); }
}

// ── UPDATE TRADES ──────────────────────────────────────────────────────────
async function updateTrades() {
  try {
    const trades = await fetch('/api/trades').then(r=>r.json());
    if (JSON.stringify(trades) === JSON.stringify(lastTrades)) return;
    lastTrades = trades;

    const wins   = trades.filter(t=>t.result==='WIN');
    const losses = trades.filter(t=>t.result==='LOSS');
    const total  = trades.length;
    const wr     = total ? (wins.length/total*100) : 0;
    const netPnl = trades.reduce((s,t)=>s+parseFloat(t.pnl_usd||0),0);
    const avgW   = wins.length ? wins.reduce((s,t)=>s+parseFloat(t.r_multiple||0),0)/wins.length : 0;
    const avgL   = losses.length ? losses.reduce((s,t)=>s+parseFloat(t.r_multiple||0),0)/losses.length : 0;
    const rr     = avgL ? Math.abs(avgW/avgL) : 0;
    const monsters = wins.filter(t=>parseFloat(t.r_multiple||0)>=3).length;

    $('heroPnl').textContent = (netPnl>=0?'+':'')+'$'+Math.abs(netPnl).toFixed(2)+' total P&L';
    $('heroPnl').style.color = netPnl>=0?'var(--green)':'var(--red)';
    $('balSub').textContent = (netPnl>=0?'+':'')+'$'+Math.abs(netPnl).toFixed(2)+' P&L';
    $('balSub').style.color = netPnl>=0?'var(--green)':'var(--red)';

    $('winRate').textContent = wr.toFixed(0)+'%';
    $('winRate').style.color = wr>=55?'var(--green)':wr>=45?'var(--gold)':'var(--red)';
    $('winSub').textContent = `${wins.length}W · ${losses.length}L · ${total} total`;

    const exp = (wr/100)*avgW + ((1-wr/100)*avgL);
    $('expVal').textContent = (exp>=0?'+':'')+exp.toFixed(2)+'R';
    $('expVal').style.color = exp>=0?'var(--green)':'var(--red)';
    $('expSub').textContent = `+${avgW.toFixed(2)}R win · ${avgL.toFixed(2)}R loss`;

    $('sTot').textContent = total;
    $('sWin').textContent = wins.length;
    $('sLoss').textContent = losses.length;
    $('sNet').textContent = (netPnl>=0?'+$':'−$')+Math.abs(netPnl).toFixed(2);
    $('sNet').style.color = netPnl>=0?'var(--green)':'var(--red)';
    $('sAvgW').textContent = '+'+avgW.toFixed(2)+'R';
    $('sAvgL').textContent = avgL.toFixed(2)+'R';
    $('sRR').textContent = rr.toFixed(2)+'x';
    $('sMon').textContent = monsters;

    // Table
    const sorted = [...trades].reverse().slice(0,30);
    if (sorted.length === 0) {
      $('tradeTable').innerHTML = '<tr><td colspan="10" class="empty">No trades yet — waiting for first signal</td></tr>';
    } else {
      $('tradeTable').innerHTML = sorted.map(t => {
        const pnl = parseFloat(t.pnl_usd||0);
        const r = parseFloat(t.r_multiple||0);
        const win = t.result==='WIN';
        const buy = t.direction==='BUY';
        return `<tr>
          <td class="mono" style="color:var(--muted)">${t.trade_id||''}</td>
          <td style="color:var(--dim)">${(t.open_time||'').substring(0,16)}</td>
          <td class="mono ${buy?'win':'loss'}">${t.direction}</td>
          <td class="mono" style="color:#8AA8C8">${parseFloat(t.entry_price||0).toFixed(1)}</td>
          <td class="mono" style="color:#8AA8C8">${parseFloat(t.exit_price||0).toFixed(1)}</td>
          <td class="mono" style="color:var(--muted)">$${fmt(t.stake)}</td>
          <td class="mono ${win?'win':'loss'}">${pnl>=0?'+':''}$${Math.abs(pnl).toFixed(2)}</td>
          <td class="mono ${r>0?'win':'loss'}">${r>=0?'+':''}${r.toFixed(2)}R</td>
          <td class="mono ${win?'win':'loss'}">${t.result}</td>
          <td class="mono" style="color:var(--dim)">${parseFloat(t.peak_r||0).toFixed(1)}R</td>
        </tr>`;
      }).join('');
    }

    // Equity chart
    const sorted2 = [...trades].sort((a,b)=>new Date(a.close_time)-new Date(b.close_time));
    const labels = sorted2.map(t=>(t.close_time||'').substring(5,16));
    const balances = sorted2.map(t=>parseFloat(t.balance_after||0));
    const ptColors = sorted2.map(t=>t.result==='WIN'?'#00E676':'#FF3366');
    equityChart.data.labels = labels;
    equityChart.data.datasets[0].data = balances;
    equityChart.data.datasets[0].pointBackgroundColor = ptColors;
    equityChart.update('none');

    // PnL chart
    const last20 = sorted2.slice(-20);
    pnlChart.data.labels = last20.map((_,i)=>i+1);
    pnlChart.data.datasets[0].data = last20.map(t=>parseFloat(t.pnl_usd||0));
    pnlChart.data.datasets[0].backgroundColor = last20.map(t=>parseFloat(t.pnl_usd||0)>=0?'rgba(0,230,118,0.7)':'rgba(255,51,102,0.7)');
    pnlChart.data.datasets[0].borderColor = last20.map(t=>parseFloat(t.pnl_usd||0)>=0?'#00E676':'#FF3366');
    pnlChart.update('none');

  } catch(e) { console.error('Trades error:', e); }
}

// ── UPDATE LOG ─────────────────────────────────────────────────────────────
async function updateLog() {
  try {
    const { lines } = await fetch('/api/log').then(r=>r.json());
    const colored = lines.map(l => {
      let cls = 'log-info';
      if (l.includes('ERROR')||l.includes('FAILED')) cls='log-error';
      else if (l.includes('WARNING')||l.includes('DD LIMIT')) cls='log-warn';
      else if (l.includes('CONTRACT OPEN')||l.includes('TRADE OPEN')) cls='log-open';
      else if (l.includes('TRADE CLOSED')||l.includes('JOURNAL')) cls='log-close';
      else if (l.includes('CHANDELIER')) cls='log-chand';
      return `<span class="log-line ${cls}">${l}</span>`;
    }).join('');
    const box = $('logBox');
    box.innerHTML = colored || 'No log yet...';
    box.scrollTop = box.scrollHeight;
  } catch(e) {}
}

// ── REFRESH FOOTER ─────────────────────────────────────────────────────────
function updateFooter() {
  const now = new Date(new Date().getTime() + 2*3600*1000);
  $('footerTime').textContent = now.toISOString().substr(11,8) + ' CAT · Auto-refresh 5s';
}

// ── MAIN LOOP ──────────────────────────────────────────────────────────────
async function refresh() {
  await Promise.all([updateState(), updateTrades(), updateLog()]);
  updateFooter();
}

initCharts();
refresh();
setInterval(refresh, 5000);
</script>
</body>
</html>"""


@app.route("/")
def index():
    return render_template_string(HTML)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8501, debug=False)
