"""Transaction decoder for Polymarket orders."""

from typing import Optional

from eth_abi import decode

from src.core.abi import MATCH_ORDERS_ABI_TYPES, MATCH_ORDERS_SELECTOR
from src.core.models import DecodedMatch, DecodedOrder
from src.utils.logging import get_logger

log = get_logger(__name__)


class TransactionDecoder:
    """Decodes Polymarket transactions using ABI decoding."""

    def __init__(self) -> None:
        log.info("Decoder initialized", selector=MATCH_ORDERS_SELECTOR.hex())

    def decode(self, tx_input: str) -> Optional[list[DecodedOrder]]:
        """Preserve the original interface used by the real-time monitor."""
        match = self.decode_match(tx_input)

        if match is None:
            return None

        return [match.taker_order, *match.maker_orders]

    def decode_match(self, tx_input: str) -> Optional[DecodedMatch]:
        """Decode a matchOrders call including actual fills and final fees."""
        try:
            input_bytes = bytes.fromhex(
                tx_input[2:] if tx_input.startswith("0x") else tx_input
            )

            if len(input_bytes) < 4:
                return None

            if input_bytes[:4] != MATCH_ORDERS_SELECTOR:
                return None

            decoded = decode(MATCH_ORDERS_ABI_TYPES, input_bytes[4:])
            return self._extract_match(decoded)

        except Exception:
            return None

    def _extract_match(self, decoded: tuple) -> DecodedMatch:
        """Extract orders, filled quantities, and fee amounts."""
        (
            taker_order,
            maker_orders,
            taker_fill_amount,
            taker_receive_amount,
            maker_fill_amounts,
            taker_fee_amount,
            maker_fee_amounts,
        ) = decoded

        return DecodedMatch(
            taker_order=self._parse_order(taker_order),
            maker_orders=[self._parse_order(order) for order in maker_orders],
            taker_fill_amount=taker_fill_amount,
            taker_receive_amount=taker_receive_amount,
            maker_fill_amounts=list(maker_fill_amounts),
            taker_fee_amount=taker_fee_amount,
            maker_fee_amounts=list(maker_fee_amounts),
        )

    def _parse_order(self, order_tuple: tuple) -> DecodedOrder:
        """Parse an ABI order tuple into a DecodedOrder."""
        return DecodedOrder(
            salt=order_tuple[0],
            maker=order_tuple[1],
            signer=order_tuple[2],
            taker=order_tuple[3],
            token_id=str(order_tuple[4]),
            maker_amount=order_tuple[5],
            taker_amount=order_tuple[6],
            expiration=order_tuple[7],
            nonce=order_tuple[8],
            fee_rate_bps=order_tuple[9],
            side=order_tuple[10],
            signature_type=order_tuple[11],
            signature=order_tuple[12],
        )