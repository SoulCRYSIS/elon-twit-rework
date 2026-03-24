# Elon Musk Tweet Count Trading Bot for Polymarket

Trading bot for Polymarket's 7-day Elon Musk tweet count bracket markets. Buys YES positions when models detect mispricing, with dry-run simulation and backtest support.

## Setup

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Fetch Data

```bash
python scripts/fetch_data.py --refresh
```

Use `--max-tokens N` to limit price history fetch for faster runs. Omit for full data.

Optional `--include-shorter` on fetch adds 2–5 day tweet events for **exploratory** backtests only.
**Do not use those events for ML training** — `scripts/train_ml_model.py` filters to 7-day+ markets only so distributions match production.

### 2. Train tabular models (XGBoost)

After fetching data, train with temporal split, correlation screening, and capped signal rate:

```bash
python scripts/train_ml_model.py
python scripts/train_ml_model.py --analyze-only   # feature/target + redundancy diagnostics only
```

Writes `data/ml_artifacts/xgb_model.joblib` and `meta.json`. The bot’s daily refresh runs this after `fetch_data.py`.

### 3. Run Backtest

```bash
python scripts/run_backtest.py
python scripts/run_backtest.py --approaches xgboost,ml,rules
```

Use `--max-events N` for quick runs. **XGBoost will not trade** until artifacts exist (run step 2).

**Buy / sell pairing**: The engine passes `position` (weighted avg entry, `buy_time`, shares) into every `get_signal`. **Sells only apply when you are long** that bracket; momentum and other exits no longer “sell flat.” Early exits for `*_exit` approaches use **tiered TP/SL** keyed off `buy_price` (`approaches/exit_policy.py`, optional `data/ml_artifacts/exit_tiers.json`).

**Profit comparison (not accuracy)** — run many approaches and rank by avg $/trade and total PnL:

```bash
python scripts/compare_approach_profits.py
python scripts/compare_approach_profits.py --max-events 20
```

Writes `results/approach_profit_comparison.json`. Names ending in `_exit` wrap the same entry model with coupled exits.

### 4. Run Bot (dry-run default)

```bash
python -m bot.bot
```

Uses **xgboost** by default. `$100` start, `bot/state_dry.json`, max **20** positions, **daily** midnight retrain. **`--live`** + Polymarket env vars for real orders — see `bot/README.md`.

```bash
python -m bot.bot --approach historical
```

### 5. View graphs

```bash
python scripts/plot_bot_performance.py
python scripts/view_graph.py
```

`plot_bot_performance.py` saves `results/bot_equity_dry.png` (and trade P&L by buy time). `view_graph.py` saves `results/pnl_graph.png`.

## Project Structure

- `scripts/fetch_data.py` - Fetch events, markets, price history from Polymarket APIs
- `scripts/train_ml_model.py` - Train XGBoost (7d-only, temporal split, feature correlation, buy-rate cap)
- `scripts/run_backtest.py` - Backtest all approaches on historical data
- `scripts/view_graph.py` - Plot P&L from bot state
- `scripts/plot_bot_performance.py` - Equity + closed trades by buy date
- `approaches/` - Prediction models (historical, negbin, momentum, ml, lstm)
- `bot/bot.py` - 24/7 bot (xgboost, dual state, daily train, live optional)
- `bot/README.md` - systemd, API keys, `--live`
- `data/` - Cached parquet data
- `results/` - Backtest outputs and graphs

## Approaches

| Approach | Description |
|----------|-------------|
| historical | Mean/std of past winning counts, Normal distribution over brackets |
| negbin | Market-implied rate + Negative Binomial (TweetCast-style) |
| momentum | Price momentum over 6h/24h |
| ml | Random Forest on shared tabular features (RSI/EMA/MACD, event time, temporal train split) |
| lstm | LSTM on price sequences (requires torch) |
| ranking | Cross-sectional: only buy top 2 brackets by edge per event |
| price_filter | Historical + price filter (2-15 cents only) |
| ensemble | Weighted combo of historical, ml, momentum |
| xgboost | Loads offline-trained XGBoost from `data/ml_artifacts/` (run `train_ml_model.py`) |
| kelly | Historical + Kelly criterion position sizing |
| regime | Regime detection (calm/volatile), adjust edge thresholds |
| bracket_spread | Adjacent-bracket spread: exploit mispricing between neighbors |
| time_weighted | Time-to-close weighting: stronger signals closer to resolution |
| rules | Interpretable baseline: cheap YES + momentum + time filters (no training) |

## Data limits & bias controls

- **ML training**: closed events with `duration_days >= 6` only (7-day Polymarket weeklies). Shorter horizons from `--include-shorter` are excluded from `train_ml_model.py` so the model is not trained on a different time scale than the bot.
- **Splits**: events ordered by `end_date`; train/val/test are disjoint event sets (no random row shuffle — avoids bracket rows from the same week leaking across splits).
- **Features**: Pearson vs target on train + greedy pruning of pairwise redundant columns (`results/feature_target_correlation_train.csv`).
- **Signal rate**: validation edge threshold chosen so roughly `--max-buy-rate` of rows would signal (default 18%), avoiding “always buy” degeneracy.
- **Backtest range**: ~72 seven-day events in Gamma; `analyze_xgboost.py` includes bootstrap CIs for thin buckets.

## Configuration

- `INITIAL_BALANCE`: 100 (each of dry / live state files)
- `AVG_POSITION_USD`: 1.0
- `POLL_INTERVAL_SEC`: 300 (5 min); override with `python -m bot.bot --poll-sec 120`
- `MAX_OPEN_POSITIONS`: 20
- **Daily retrain** (local midnight): `fetch_data.py --refresh` + `train_ml_model.py` once per calendar day
- `COOLDOWN_HOURS`: 6 – don't re-buy same bracket for N hours
- `MIN_PRICE_CHANGE`: 1.0 (2x = 100%) – require price to double or halve to add to position

Backtest: `--cooldown-hours 6 --min-price-change 1.0`

## Running the Trading Bot

Details: **[bot/README.md](bot/README.md)** (24/7 systemd, API env vars).

### Prerequisites

1. **Fetch data and train XGBoost** (required for default bot):
   ```bash
   python scripts/fetch_data.py --refresh
   python scripts/train_ml_model.py
   ```

2. **Optional: run backtest** to compare approaches:
   ```bash
   python scripts/run_backtest.py
   ```

### Start the Bot (dry-run is default)

```bash
cd /path/to/elon-twit-rework
python -m bot.bot
```

- **Approach**: `xgboost` (default; override `--approach`)
- **Starting balance**: $100 in `bot/state_dry.json`
- **Position size**: ~$1 per trade; **max 20** open positions
- **State**: `bot/state_dry.json` (dry) or `bot/state_live.json` (`--live`) — survives restarts
- **Daily training**: after each local midnight, first loop runs fetch + train

### Plots

```bash
python scripts/plot_bot_performance.py
python scripts/view_graph.py   # legacy cumulative chart from dry state
```

### Live Trading

```bash
export POLYMARKET_PRIVATE_KEY=... POLYMARKET_API_KEY=... POLYMARKET_API_SECRET=... POLYMARKET_API_PASSPHRASE=...
python -m bot.bot --live
```

See `bot/live_executor.py` and `bot/README.md`. Fund your Polymarket/Polygon wallet separately; the bot still tracks a $100-anchored ledger in state for continuity.
