# Elon tweet bot (24/7)

## Quick start (dry-run, recommended)

From repo root, after `fetch_data` + `train_ml_model`:

```bash
python -m bot.bot
# or
python bot/bot.py
```

- **State**: `bot/state_dry.json` — balance, open positions, `trade_history`, retrain date. Survives restarts.
- **Approach**: `xgboost` (default).
- **Training refresh**: once per **local calendar day** (runs `fetch_data.py` — incremental CLOB prices by default — then `train_ml_model.py`).
- **Max positions**: 20.
- **Starting cash**: $100 (dry-run ledger).

## Live trading (`--live`)

Requires a **funded Polymarket / Polygon** wallet and CLOB API credentials.

Set:

| Variable | Description |
|----------|-------------|
| `POLYMARKET_PRIVATE_KEY` | Wallet private key (`0x...`) |
| `POLYMARKET_API_KEY` | CLOB L2 API key |
| `POLYMARKET_API_SECRET` | CLOB L2 secret |
| `POLYMARKET_API_PASSPHRASE` | CLOB L2 passphrase |

Optional: `POLYMARKET_CLOB_HOST` (default `https://clob.polymarket.com`), `POLYMARKET_CHAIN_ID` (default `137`).

Create keys via [Polymarket docs](https://docs.polymarket.com) / `py-clob-client` (`derive_api_key`, builder UI).

```bash
export POLYMARKET_PRIVATE_KEY=...
export POLYMARKET_API_KEY=...
export POLYMARKET_API_SECRET=...
export POLYMARKET_API_PASSPHRASE=...
python -m bot.bot --live
```

- **State**: `bot/state_live.json` (separate from dry-run — two $100 ledgers).
- Orders: limit BUY/SELL via `create_and_post_order`. **Review `bot/live_executor.py` and test with tiny size before production.**

## Run 24/7

**systemd** (user service example):

```ini
[Unit]
Description=Polymarket Elon bot (dry-run)
After=network-online.target

[Service]
Type=simple
WorkingDirectory=/path/to/elon-twit-rework
ExecStart=/usr/bin/python3 -m bot.bot
Restart=always
RestartSec=30

[Install]
WantedBy=default.target
```

**tmux / screen / `nohup`**:

```bash
nohup python -m bot.bot >> bot/bot_dry.log 2>&1 &
```

## Plots

```bash
python scripts/plot_bot_performance.py          # dry state
python scripts/plot_bot_performance.py --live   # live state
```

Writes `results/bot_equity_<mode>.png` (equity path + current balance) and bar chart of closed P&L sorted by buy time.

## Files

| File | Purpose |
|------|---------|
| `state_dry.json` | Simulated wallet + positions |
| `state_live.json` | Live mode state |
| `state.json` | Legacy; migrated to `state_dry.json` on first dry run |

Backup these files if you care about continuity.
