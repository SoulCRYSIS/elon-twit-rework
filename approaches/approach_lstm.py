"""LSTM approach: price sequence -> probability bracket wins."""

import re
from pathlib import Path

import numpy as np
import pandas as pd

from approaches.base import Signal

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
EDGE_THRESHOLD = 0.05
SEQ_LEN = 168  # 7 days hourly


def _parse_bracket(bracket_range: str) -> tuple[float, float]:
    m = re.match(r"(\d+)-(\d+)", bracket_range)
    if m:
        return float(m.group(1)), float(m.group(2))
    m = re.match(r"(\d+)\+", bracket_range)
    if m:
        return float(m.group(1)), float(m.group(1)) + 50
    m = re.match(r"0-(\d+)", bracket_range)
    if m:
        return 0.0, float(m.group(1))
    return 0.0, 0.0


def _build_sequence(price_history: list[tuple[float, float]], current_time: float) -> np.ndarray:
    """Build fixed-length sequence of prices (hourly buckets)."""
    seq = np.zeros(SEQ_LEN)
    if not price_history:
        return seq

    # Sort by time
    sorted_ph = sorted(price_history, key=lambda x: x[0])
    # Hourly buckets: current_time - SEQ_LEN hours to current_time
    start = current_time - SEQ_LEN * 3600
    for i in range(SEQ_LEN):
        t_bucket = start + i * 3600
        t_end = t_bucket + 3600
        in_bucket = [p for t, p in sorted_ph if t_bucket <= t < t_end]
        seq[i] = np.mean(in_bucket) if in_bucket else (seq[i - 1] if i > 0 else 0.05)
    # Forward fill zeros
    last = 0.05
    for i in range(SEQ_LEN):
        if seq[i] < 0.001:
            seq[i] = last
        else:
            last = seq[i]
    return seq.astype(np.float32)


_model_cache = None


def _train_model():
    """Train LSTM on price sequences -> win probability."""
    global _model_cache
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        return None

    events_path = DATA_DIR / "events.parquet"
    markets_path = DATA_DIR / "markets.parquet"
    price_path = DATA_DIR / "price_history.parquet"

    if not all(p.exists() for p in [events_path, markets_path, price_path]):
        return None

    events = pd.read_parquet(events_path)
    markets = pd.read_parquet(markets_path)
    prices = pd.read_parquet(price_path)

    closed = events[events["closed"] == True]
    X_list, y_list = [], []

    for _, ev in closed.iterrows():
        eid = ev["event_id"]
        start = pd.to_datetime(ev["start_date"]).timestamp()
        end = pd.to_datetime(ev["end_date"]).timestamp()
        mkts = markets[markets["event_id"] == eid]

        for _, m in mkts.iterrows():
            tid = m["yes_token_id"]
            resolved = m["resolved_yes"]

            ph = prices[prices["token_id"] == tid]
            if len(ph) < SEQ_LEN:
                continue

            for frac in [0.3, 0.5, 0.7]:
                t = start + frac * (end - start)
                before = ph[ph["timestamp"] <= t]
                if len(before) < SEQ_LEN:
                    continue
                hist = [(r["timestamp"], r["price"]) for _, r in before.tail(SEQ_LEN * 2).iterrows()]
                seq = _build_sequence(hist, t)
                X_list.append(seq)
                y_list.append(resolved)

    if len(X_list) < 30:
        return None

    X = np.stack(X_list)
    y = np.array(y_list, dtype=np.float32)

    class SimpleLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(1, 32, batch_first=True, num_layers=1)
            self.fc = nn.Linear(32, 1)

        def forward(self, x):
            x = x.unsqueeze(-1)
            out, _ = self.lstm(x)
            return torch.sigmoid(self.fc(out[:, -1, :]))

    model = SimpleLSTM()
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    X_t = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
    y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)

    for _ in range(50):
        pred = model(X_t)
        loss = nn.functional.binary_cross_entropy(pred, y_t)
        opt.zero_grad()
        loss.backward()
        opt.step()

    _model_cache = model
    return model


def get_signal(
    event_id: int,
    bracket: str,
    current_time: float,
    price_history: list[tuple[float, float]],
    market_data: dict,
    current_price: float | None = None,
    current_count: float | None = None,
    end_time: float | None = None,
    **kwargs,
) -> Signal:
    """
    LSTM: sequence of prices -> P(win). Buy when model_prob > market_price + threshold.
    """
    global _model_cache
    if _model_cache is None:
        _model_cache = _train_model()

    market_price = current_price if current_price is not None else (price_history[-1][1] if price_history else 0.05)

    if _model_cache is None:
        return Signal(buy=False, sell=False, confidence=0.0, model_prob=market_price, market_price=market_price, edge=0.0)

    seq = _build_sequence(price_history, current_time)
    try:
        import torch

        with torch.no_grad():
            x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
            prob = _model_cache(x).item()
    except Exception:
        prob = market_price

    edge = prob - market_price
    buy = edge >= EDGE_THRESHOLD and market_price < 0.5
    confidence = min(1.0, abs(edge) / 0.2) if edge > 0 else 0.0

    return Signal(
        buy=buy,
        sell=False,
        confidence=confidence,
        model_prob=float(prob),
        market_price=market_price,
        edge=edge,
    )
