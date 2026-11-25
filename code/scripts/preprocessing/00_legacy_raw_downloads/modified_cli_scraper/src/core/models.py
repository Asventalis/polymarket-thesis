"""Data models for trade monitoring."""

from dataclasses import dataclass


@dataclass
class DecodedOrder:
    """Represents a decoded order from the blockchain transaction."""

    salt: int
    maker: str
    signer: str
    taker: str
    token_id: str
    maker_amount: int
    taker_amount: int
    expiration: int
    nonce: int
    fee_rate_bps: int
    side: int  # 0 = BUY, 1 = SELL
    signature_type: int
    signature: bytes


@dataclass
class DecodedMatch:
    """Represents one decoded matchOrders transaction."""

    taker_order: DecodedOrder
    maker_orders: list[DecodedOrder]
    taker_fill_amount: int
    taker_receive_amount: int
    maker_fill_amounts: list[int]
    taker_fee_amount: int
    maker_fee_amounts: list[int]


@dataclass
class TradeData:
    """One wallet-level filled-order observation."""

    block_number: int
    timestamp: str
    transaction_hash: str
    wallet: str
    token_id: str
    side: int  # 0 = BUY, 1 = SELL

    # Actual filled quantities:
    # maker_amount = asset supplied by this order
    # taker_amount = asset received by this order, before fee
    maker_amount: int
    taker_amount: int

    role: str = "UNKNOWN"  # TAKER / MAKER
    order_index: int = -1  # -1 = taker; 0..n-1 = maker index
    fee_raw: int = 0       # Fee in the asset received by this order