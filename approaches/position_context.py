"""Aggregate open lots per bracket for exit decisions (avg entry, oldest time)."""

from __future__ import annotations

from typing import Any


def aggregate_bracket_position(
    positions: list[dict[str, Any]],
    bracket: str,
    event_id: int | str | None = None,
) -> dict[str, float] | None:
    """Weighted average entry for all open lots in this bracket (optionally one event)."""
    sub = [p for p in positions if p.get("bracket") == bracket]
    if event_id is not None:
        sub = [p for p in sub if str(p.get("event_id")) == str(event_id)]
    if not sub:
        return None
    total_usd = sum(float(p["amount_usd"]) for p in sub)
    total_shares = sum(float(p["shares"]) for p in sub)
    if total_shares <= 0:
        return None
    avg_buy = total_usd / total_shares
    buy_time = min(float(p["buy_time"]) for p in sub)
    return {
        "buy_price": avg_buy,
        "buy_time": buy_time,
        "shares": total_shares,
        "amount_usd": total_usd,
        "n_lots": len(sub),
    }
