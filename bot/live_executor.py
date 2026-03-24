"""
Polymarket CLOB live order execution (py-clob-client).

Required environment variables for --live:
  POLYMARKET_PRIVATE_KEY   Wallet private key (0x...)
  POLYMARKET_API_KEY         CLOB L2 API key
  POLYMARKET_API_SECRET      CLOB L2 secret
  POLYMARKET_API_PASSPHRASE  CLOB L2 passphrase

Optional:
  POLYMARKET_CLOB_HOST       Default https://clob.polymarket.com
  POLYMARKET_CHAIN_ID        Default 137 (Polygon)

Create API credentials via Polymarket / py-clob-client docs (derive or builder UI).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

DEFAULT_HOST = "https://clob.polymarket.com"
DEFAULT_CHAIN = 137


@dataclass
class LiveFillResult:
    ok: bool
    message: str
    response: Any = None


def live_configured() -> bool:
    return all(
        os.environ.get(k)
        for k in (
            "POLYMARKET_PRIVATE_KEY",
            "POLYMARKET_API_KEY",
            "POLYMARKET_API_SECRET",
            "POLYMARKET_API_PASSPHRASE",
        )
    )


def missing_live_env() -> list[str]:
    req = [
        "POLYMARKET_PRIVATE_KEY",
        "POLYMARKET_API_KEY",
        "POLYMARKET_API_SECRET",
        "POLYMARKET_API_PASSPHRASE",
    ]
    return [k for k in req if not os.environ.get(k)]


def get_client():
    from py_clob_client.client import ClobClient
    from py_clob_client.clob_types import ApiCreds

    if not live_configured():
        raise RuntimeError(
            "Live trading not configured. Set: " + ", ".join(missing_live_env())
        )
    host = os.environ.get("POLYMARKET_CLOB_HOST", DEFAULT_HOST).rstrip("/")
    chain_id = int(os.environ.get("POLYMARKET_CHAIN_ID", str(DEFAULT_CHAIN)))
    key = os.environ["POLYMARKET_PRIVATE_KEY"]
    creds = ApiCreds(
        api_key=os.environ["POLYMARKET_API_KEY"],
        api_secret=os.environ["POLYMARKET_API_SECRET"],
        api_passphrase=os.environ["POLYMARKET_API_PASSPHRASE"],
    )
    return ClobClient(host, chain_id=chain_id, key=key, creds=creds)


def market_buy_yes(
    token_id: str,
    price: float,
    size_shares: float,
) -> LiveFillResult:
    """Post a limit BUY for YES shares (size = conditional token amount)."""
    try:
        from py_clob_client.clob_types import OrderArgs
        from py_clob_client.order_builder.constants import BUY

        client = get_client()
        order_args = OrderArgs(
            token_id=str(token_id),
            price=float(price),
            size=float(size_shares),
            side=BUY,
        )
        resp = client.create_and_post_order(order_args)
        return LiveFillResult(ok=True, message="order_posted", response=resp)
    except Exception as e:
        return LiveFillResult(ok=False, message=str(e), response=None)


def market_sell_yes(
    token_id: str,
    price: float,
    size_shares: float,
) -> LiveFillResult:
    """Post a limit SELL for YES shares."""
    try:
        from py_clob_client.clob_types import OrderArgs
        from py_clob_client.order_builder.constants import SELL

        client = get_client()
        order_args = OrderArgs(
            token_id=str(token_id),
            price=float(price),
            size=float(size_shares),
            side=SELL,
        )
        resp = client.create_and_post_order(order_args)
        return LiveFillResult(ok=True, message="order_posted", response=resp)
    except Exception as e:
        return LiveFillResult(ok=False, message=str(e), response=None)
