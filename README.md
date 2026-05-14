# Hypothermia Trading Bot

> Algorithmic trading system for XAU/USD (Gold) and synthetic indices (Boom 1000 / Crash 1000) on the Deriv platform — with a live Streamlit monitoring dashboard.

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Plotly](https://img.shields.io/badge/Plotly-Charts-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com)
[![WebSockets](https://img.shields.io/badge/WebSockets-Deriv%20API-010101?style=for-the-badge)](https://api.deriv.com)

---

## About

Hypothermia is a live trading bot that connects to the Deriv API via WebSockets, executes trades based on EMA + ATR strategy rules, and streams real-time performance data to a Streamlit dashboard. It supports three instruments:

- **XAU/USD** — Gold spot (M5 + H1 timeframe strategy)
- **Boom 1000** — Deriv synthetic index
- **Crash 1000** — Deriv synthetic index

---

## Strategy (XAU/USD)

The core strategy uses a multi-timeframe approach:

| Signal | Description |
|--------|-------------|
| Trend filter | H1 EMA-21 determines directional bias |
| Entry trigger | M5 EMA-50 crossover in trend direction |
| ATR zones | Dynamic support/resistance zones (ATR-based) |
| Exit | Chandelier stop tiers (multi-level ATR trailing stop) |
| Session | Configurable trading window (default: London/NY overlap) |

Risk is sized per-trade based on ATR, not fixed pip values.

---

## Features

- Live WebSocket connection to Deriv API
- Real-time trade execution and position management
- Streamlit dashboard with live PnL, trade log, and chart overlays
- Backtesting module for strategy validation
- Parameter sensitivity analysis
- Portfolio risk tools
- Commodity data fetcher (for external price feeds)
- Archive of historical performance results

---

## Project Structure

```
hypothermia-bot/
├── xau_bot.py              # Main XAU/USD trading bot
├── xau_strategy.py         # Signal generation (EMA, ATR, chandelier)
├── xau_config.py           # Strategy parameters
├── deriv_boom1000_bot.py   # Boom 1000 bot
├── deriv_crash1000_bot.py  # Crash 1000 bot
├── dashboard.py            # Streamlit live dashboard
├── commodity_backtest.py   # Backtest engine
├── frequency_backtest.py   # Frequency-based backtest
├── parameter_sensitivity.py # Strategy parameter sweeps
├── portfolio_risk.py       # Portfolio-level risk analysis
├── risk_comparison.py      # Risk model comparison
├── fetch_commodity_data.py # External data fetcher
├── enhancement_test.py     # Strategy enhancement tests
├── data/                   # Historical price data
└── archive/                # Past backtest results
```

---

## Setup

### Prerequisites
- Python 3.11+
- Deriv account with API token

### Install

```bash
git clone https://github.com/CypTynash/hypothermia-bot.git
cd hypothermia-bot
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Configure

Create a `.env` file:

```env
DERIV_API_TOKEN=your_api_token_here
DERIV_APP_ID=your_app_id_here
```

### Run the Bot

```bash
python xau_bot.py
```

### Run the Dashboard

```bash
streamlit run dashboard.py
```

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `pandas` | Data manipulation and OHLC processing |
| `numpy` | Numerical calculations (EMA, ATR) |
| `websockets` | Deriv API real-time connection |
| `streamlit` | Live monitoring dashboard |
| `plotly` | Interactive charts |
| `python-dotenv` | Environment variable management |

---

## Disclaimer

This software is for educational and research purposes. Trading financial instruments carries significant risk. Past performance does not guarantee future results.

---

## License

MIT
