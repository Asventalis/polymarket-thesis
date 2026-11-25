"""
lob_layout.py

Single source of truth for the column layout of the precomputed LOB tensors
stored in  data/derived/tensors/year=/month=/day=/hour=/lob_features.parquet
(column ``features``, 52 doubles per row).

The rows are produced by ``OrderbookState.extract_features`` in ``lob_builder.py``:

    full_tensor = concat([bid_prices(10), bid_sizes(10),
                          ask_prices(10), ask_sizes(10),
                          empirical_vector(12)])

so the empirical block starts at index 40.  In particular the mid-price is at
index 40, NOT at index 0: index 0 is ``(best_bid - mid) / mid``, i.e. the
relative deviation of the best bid from the mid (≈ minus half the relative
spread).  Several earlier scripts used index 0 as "mid price"; anything that
needs a price must import the constants below instead of hard-coding 0.
"""
import numpy as np

DEPTH_LEVELS = 10

BID_PRICE_START = 0                       # (p - mid) / mid, 10 levels
BID_SIZE_START = BID_PRICE_START + DEPTH_LEVELS
ASK_PRICE_START = BID_SIZE_START + DEPTH_LEVELS
ASK_SIZE_START = ASK_PRICE_START + DEPTH_LEVELS
EMPIRICAL_START = ASK_SIZE_START + DEPTH_LEVELS   # 40

MID_IDX = EMPIRICAL_START + 0          # 40  mid price in [0, 1]
SPREAD_IDX = EMPIRICAL_START + 1       # 41  best_ask - best_bid
BEST_BID_IDX = EMPIRICAL_START + 2     # 42
BEST_ASK_IDX = EMPIRICAL_START + 3     # 43
OFI_IDX = EMPIRICAL_START + 4          # 44  order-flow imbalance (window accumulator)
VOLUME_IDX = EMPIRICAL_START + 5       # 45
EVENT_COUNT_IDX = EMPIRICAL_START + 6  # 46
BUY_RATIO_IDX = EMPIRICAL_START + 7    # 47
CANCEL_RATIO_IDX = EMPIRICAL_START + 8 # 48
MEAN_BID_DEPTH_IDX = EMPIRICAL_START + 9
MEAN_ASK_DEPTH_IDX = EMPIRICAL_START + 10
DEPTH_SKEW_IDX = EMPIRICAL_START + 11  # 51

FEATURE_DIM = EMPIRICAL_START + 12     # 52

EMPIRICAL_NAMES = [
    "mid_price", "spread", "best_bid", "best_ask", "ofi", "total_event_volume",
    "event_count", "buy_event_ratio", "cancel_velocity", "mean_bid_depth",
    "mean_ask_depth", "depth_skewness",
]

assert FEATURE_DIM == 52
assert len(EMPIRICAL_NAMES) == 12


def feature_name(idx: int) -> str:
    """Human readable name of a tensor column (for logging / sanity checks)."""
    if idx < BID_SIZE_START:
        return f"bid_price_rel_{idx - BID_PRICE_START}"
    if idx < ASK_PRICE_START:
        return f"bid_size_{idx - BID_SIZE_START}"
    if idx < ASK_SIZE_START:
        return f"ask_price_rel_{idx - ASK_PRICE_START}"
    if idx < EMPIRICAL_START:
        return f"ask_size_{idx - ASK_SIZE_START}"
    return EMPIRICAL_NAMES[idx - EMPIRICAL_START]


# ─────────────────────────────────────────────────────────────────────────────
# Scaling
# ─────────────────────────────────────────────────────────────────────────────
# Heavy-tailed columns.  On 2026-03-13 05:00 the median / 99th percentile of
# |x| are e.g. 72 / 2.1e5 for bid_size_0 and 2.0e3 / 2.0e5 for
# total_event_volume, and 37%–99% of the values of these columns exceed the
# ±100 clamp applied in the training loop, i.e. the raw features are mostly
# saturated.  A signed log1p keeps the ordering and brings them to O(10).
# Relative ask prices (p - mid)/mid are included because they explode when the
# mid is close to 0 (99th percentile ≈ 77).
LOG_SCALE_COLUMNS = (
    list(range(BID_SIZE_START, BID_SIZE_START + DEPTH_LEVELS))
    + list(range(ASK_PRICE_START, ASK_PRICE_START + DEPTH_LEVELS))
    + list(range(ASK_SIZE_START, ASK_SIZE_START + DEPTH_LEVELS))
    + [OFI_IDX, VOLUME_IDX, MEAN_BID_DEPTH_IDX, MEAN_ASK_DEPTH_IDX]
)


def signed_log1p_columns(windows: np.ndarray, columns=LOG_SCALE_COLUMNS) -> np.ndarray:
    """
    Returns a copy of ``windows`` (..., >= FEATURE_DIM) with
    sign(x) * log1p(|x|) applied to ``columns`` on the last axis.
    Zero-padded rows stay zero; all other columns are untouched.
    """
    out = np.array(windows, dtype=np.float32, copy=True)
    cols = list(columns)
    x = out[..., cols]
    out[..., cols] = np.sign(x) * np.log1p(np.abs(x))
    return out
