"""Trade formatting utilities."""

from dataclasses import dataclass

from src.core.models import TradeData


@dataclass
class FormattedTrade:
    """Human-readable wallet-level filled-order data."""

    block_number: int
    timestamp: str
    tx_hash: str
    order_index: int
    wallet: str
    role: str
    token_id: str
    side: str
    tokens: float
    price: float
    total_usdc: float
    fee_rate: float


def format_trade(trade: TradeData) -> FormattedTrade:
    """Convert one raw filled order into human-readable values."""
    is_buy = trade.side == 0

    if is_buy:
        # BUY: wallet supplies collateral and receives outcome tokens.
        total_usdc_raw = trade.maker_amount
        tokens_raw = trade.taker_amount
    else:
        # SELL: wallet supplies outcome tokens and receives collateral.
        tokens_raw = trade.maker_amount
        total_usdc_raw = trade.taker_amount

    tokens = tokens_raw / 1_000_000
    total_usdc = total_usdc_raw / 1_000_000
    price = total_usdc / tokens if tokens > 0 else 0.0

    # Fees are charged from the asset received by the order.
    # trade.taker_amount is the gross received quantity for either side.
    fee_rate = (
        trade.fee_raw / trade.taker_amount
        if trade.taker_amount > 0
        else 0.0
    )

    return FormattedTrade(
        block_number=trade.block_number,
        timestamp=trade.timestamp,
        tx_hash=trade.transaction_hash,
        order_index=trade.order_index,
        wallet=trade.wallet,
        role=trade.role,
        token_id=trade.token_id,
        side="BUY" if is_buy else "SELL",
        tokens=tokens,
        price=price,
        total_usdc=total_usdc,
        fee_rate=fee_rate,
    )