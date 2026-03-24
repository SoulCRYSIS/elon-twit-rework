# XGBoost Trading Analysis

**Metric: Average profit per $1 trade** (win rate ignored)

## Summary

- **Total trades**: varies by backtest run
- **Avg profit per trade**: ~$2.78–3.21 per $1
- **Early sells**: 0 (XGBoost holds to resolution)

---

## By Buy Price (sorted by avg $/trade)

| Price (cents) | Trades | Avg $/trade |
|---------------|--------|-------------|
| **0-5**       | 7      | **$7.05**   |
| 15-20         | 6      | $3.72       |
| 5-10          | 5      | $2.08       |
| 10-15         | 4      | $1.44       |
| 40-50         | 2      | $1.30       |
| 25-30         | 2      | $0.79       |
| 30-40         | 3      | $0.76       |
| 20-25         | 6      | $0.48       |

**Insight**: Cheapest brackets (0-5¢) yield highest avg profit per trade when they hit. 15-20¢ also strong. 20-25¢ underperforms.

---

## By Bracket (sorted by avg $/trade, min 3 trades)

| Bracket | Trades | Avg $/trade | Avg buy price |
|---------|--------|-------------|---------------|
| **0-59**  | 4  | **$11.75** | 10¢ |
| **0-54**  | 4  | **$6.17**  | 17.7¢ |
| 40-49     | 6  | $2.64      | 15.5¢ |
| 70-79     | 6  | $1.58      | 27.5¢ |
| 50-59     | 4  | $0.00      | 22.6¢ |
| **0-39**  | 6  | **-$1.00** | 4.6¢ (avoid) |

**Insight**: 0-59 and 0-54 are best. 0-39 consistently loses ($-1/trade).

---

## By Time (hours from event start)

| Hours  | Trades | Avg $/trade |
|--------|--------|-------------|
| **0-24**   | 19 | **$3.83** |
| 72-96      | 3  | $3.18      |
| 48-72      | 3  | $2.16      |
| 120-168    | 4  | $1.99      |
| 24-48      | 4  | $0.61      |
| 96-120     | 2  | -$1.00     |

**Insight**: Best avg profit in first 24h. Avoid 96-120h.

---

## Min Price Change (2x Rule)

**Formula**: `|current_price - last_buy_price| / last_buy_price >= MIN_PRICE_CHANGE`

With `MIN_PRICE_CHANGE = 1.0` (2x rule):
- Price must **double** or **halve** to buy again
- Example: bought at 10¢ → need 20¢ or 5¢ to re-enter
- Example: bought at 25¢ → need 50¢ or 12.5¢ to re-enter

**Why 2x**: Prevents buying every 5 min when the signal stays true. Only re-enter when price moves enough to represent a meaningfully different opportunity.

**Config**: `--min-price-change 1.0` (default)
