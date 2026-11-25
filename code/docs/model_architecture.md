# Causal Dynamic GNN — Model Architecture Document

This document describes the full architecture of the Causal Dynamic Graph Neural Network for Polymarket price direction prediction. It covers the data pipeline, model components, training loop structure, and design rationale.

---

## 1. Problem Setting

We model the Polymarket prediction market ecosystem as a **dynamic bipartite graph** where:
- **Market nodes** represent individual prediction markets (e.g., "Will X happen?")
- **Wallet nodes** represent trader wallets
- **Edges** represent transactions between wallets and markets, typed by intent (yes\_buy, yes\_sell, no\_buy, no\_sell)

The graph evolves over time as new transactions occur and new orderbook (LOB) data arrives. The goal is to predict the **price direction** (up/down) of each market at the next time step.

### Time Discretization

Time is discretized by **blockchain blocks** (~2 seconds apart). Each block contains a set of atomic transactions. The model processes one block at a time, strictly causally — no future information is ever used.

### Data Scale

| Entity | Count |
|---|---|
| Total markets | 175,292 |
| Total wallets | 646,394 |
| LOB snapshots per hour | ~287,000 |
| Transactions per hour | ~580,000 |
| Blocks per hour | ~1,800 |
| Training data | 504 hours (~21 days) |

---

## 2. Data Pipeline

### 2.1 LOB Tensor Precomputation

Raw orderbook events (bids, asks, trades) are processed by [lob_builder.py](file:///home/vitopampinella/Desktop/Magistrale/Tesi/agy_project/src/polymarket_gnn/data/lob_builder.py) into fixed-size feature vectors. For each market at each timestamp, a **52-dimensional feature vector** is computed:

| Feature Group | Dimensions | Description |
|---|---|---|
| Best bid/ask prices | 2 | Top-of-book prices |
| Bid/ask sizes (5 levels) | 10 | Volume at each of the 5 best price levels |
| Bid/ask price levels (5 levels) | 10 | Price at each of the 5 best levels |
| Spread and imbalance | 4 | Bid-ask spread, mid-price, volume imbalance, depth imbalance |
| Trade statistics | 8 | Recent trade price, volume, buy/sell ratio, VWAP |
| Rolling statistics | 18 | Moving averages, volatility, momentum indicators |

These are stored as Parquet files partitioned by `year/month/day/hour`, totaling ~27 GB for 504 hours.

### 2.2 Market Lifecycle Events

A precomputed file ([market_lifecycle_events.parquet](file:///home/vitopampinella/Desktop/Magistrale/Tesi/agy_project/data/derived/market_lifecycle_events.parquet)) contains 283,318 chronologically sorted events of type `(timestamp, market_id, OPEN|CLOSE)`. This allows O(1) maintenance of the set of currently-active markets as time advances.

### 2.3 Market Metadata Embeddings

Each market has precomputed static features stored in [market_features.pt](file:///home/vitopampinella/Desktop/Magistrale/Tesi/agy_project/data/derived/market_features.pt) — a tensor of shape `(175292, 390)`:
- **384 dimensions**: Sentence embedding of the market's question text (from a frozen `all-MiniLM-L6-v2` encoder)
- **6 dimensions**: Numerical metadata (initial prices, creation date, etc.)

### 2.4 ID Mappings

Two lookup tables map string identifiers to dense integer indices:
- `market_to_idx`: market\_id (hex string) → row index in `h_market` (0 to 175,291)
- `wallet_to_idx`: wallet address → row index in `h_wallet` (0 to 646,393)

---

## 3. In-Memory Data Structures

### 3.1 LOBWindowBuffer

**File**: [lob_window_buffer.py](file:///home/vitopampinella/Desktop/Magistrale/Tesi/agy_project/src/polymarket_gnn/data/lob_window_buffer.py)

An efficient in-memory structure that holds one hour of LOB tensors and supports fast temporal queries.

**Internal state:**

| Field | Type | Description |
|---|---|---|
| `_timestamps` | `Dict[str, np.int64[]]` | Sorted nanosecond timestamps per market (current hour only) |
| `_features` | `Dict[str, np.float32[N,52]]` | LOB feature matrices per market (current hour only) |
| `_open_markets` | `Set[str]` | All currently-open markets (from lifecycle events) |
| `_open_with_data` | `Set[str]` | Intersection: open AND has LOB data in current hour |
| `_lifecycle_df` | `DataFrame` | Full sorted lifecycle events |
| `_lifecycle_ptr` | `int` | Monotonically-advancing pointer into lifecycle events |

**Key operations:**

- **`reload_lob_data(path)`**: Swaps LOB tensors for a new hour without resetting the lifecycle pointer. Recomputes `_open_with_data` as the intersection of `_open_markets` and the new hour's markets. Complexity: O(M) where M = markets in the hour file.

- **`reset_to_time(t_ns)`**: Resets the lifecycle pointer to 0 and fast-forwards through all events up to time `t_ns`, building the correct `_open_markets` set. Called once at the start of training.

- **`_advance_lifecycle(t_end_ns)`**: Processes lifecycle events from the current pointer up to `t_end_ns`. Each event is processed exactly once across the entire training run. Amortized O(1) per step.

- **`load_step(t_prev_ns, t_ns)`**: The main per-block entry point:
  1. Advances lifecycle to `t_ns`
  2. Scans `_open_with_data` for markets with new LOB snapshots in `(t_prev, t]`
  3. For each active market, retrieves the last K=5 snapshots before `t` via binary search
  4. Returns `(active_market_ids, windows)` where `windows` has shape `(N_active, K, 52)`

- **`get_window(market_id, t_ns)`**: Returns the K most recent LOB snapshots strictly before time `t_ns`. If fewer than K snapshots exist (only at the very beginning of a market's life), zero-pads from the left. Complexity: O(log N) via `bisect`.

**Performance**: ~10ms per block step scanning ~9,700 open markets with LOB data.

---

## 4. Model Architecture

**File**: [causal_gnn.py](file:///home/vitopampinella/Desktop/Magistrale/Tesi/agy_project/src/polymarket_gnn/models/causal_gnn.py)

### 4.1 Global Persistent State

Two embedding matrices live on GPU throughout training, **outside** the model:

$$\mathbf{H}^{\text{market}} \in \mathbb{R}^{175292 \times d}, \quad \mathbf{H}^{\text{wallet}} \in \mathbb{R}^{646394 \times d}$$

where $d = 128$ is the hidden dimension.

**Initialization**:
- $\mathbf{H}^{\text{market}}$ is initialized by projecting the 390-dim precomputed metadata embeddings through a learned linear layer: $\mathbf{H}^{\text{market}}_0 = \mathbf{X}^{\text{meta}} \mathbf{W}_{\text{proj}} + \mathbf{b}_{\text{proj}}$, where $\mathbf{X}^{\text{meta}} \in \mathbb{R}^{175292 \times 390}$.
- $\mathbf{H}^{\text{wallet}}$ is initialized to zeros (wallets have no static metadata).

These embeddings persist across all time steps and are updated in-place at each block.

### 4.2 LOBUpdateModule

**Purpose**: Encodes the recent LOB trajectory of each active market and residually updates its node embedding.

**Input**: For each block at time $t$, the buffer identifies $N_{\text{active}}$ markets that received new LOB data since the previous block. For each, it retrieves the last $K=5$ snapshots:

$$\mathbf{W} \in \mathbb{R}^{N_{\text{active}} \times K \times 52}$$

**Architecture**:

```
Input: (N_active, K=5, 52)
   │
   ▼
GRU(input_size=52, hidden_size=128, batch_first=True)
   │  Processes K=5 snapshots sequentially.
   │  Hidden state initialized to zero at every step (no persistent GRU state).
   │  All N_active markets processed in parallel (batch dimension).
   ▼
Last hidden state: (N_active, 128)
   │
   ▼
Linear(128 → 128)
   │  Learned projection of the LOB summary signal.
   ▼
δ: (N_active, 128)
   │
   ▼
Residual Update:
   h_market[active_idx] = LayerNorm(h_market[active_idx] + δ)
```

**Key design decisions**:
- **No persistent GRU state**: The GRU hidden state is initialized to zero at every block. The temporal memory comes from the K-snapshot sliding window and the persistent node embedding — not from carrying GRU state across blocks.
- **Residual update (Option C)**: The GRU output is *added* to the existing embedding rather than replacing it. This preserves information accumulated from previous blocks and transaction-based message passing.
- **Only active markets are touched**: Markets with no new LOB data between blocks retain their embeddings unchanged. This is both computationally efficient and semantically correct — no new information means no update.

**Computational cost**: ~0.4ms per step on GPU (after CUDA warmup).

### 4.3 PersistentHeteroAggregator

**Purpose**: Performs bidirectional message passing between wallet and market nodes using the transaction edges from the current block.

**Input**: The transaction graph for the current block, represented as typed edges:

| Relation | Sign | Interpretation |
|---|---|---|
| `yes_buy` | +1.0 | Buying YES tokens → bullish signal |
| `yes_sell` | -1.0 | Selling YES tokens → bearish signal |
| `no_buy` | -1.0 | Buying NO tokens → bearish signal for YES price |
| `no_sell` | +1.0 | Selling NO tokens → bullish signal for YES price |

Each edge carries a weight (normalized USDC transaction amount).

**Architecture**:

```
── Wallet → Market Messages ──────────────────────────────

h_w_proj = Dropout(0.1)(Linear(h_wallet))           # (N_wallets, 128)

For each relation r ∈ {yes_buy, yes_sell, no_buy, no_sell}:
    msg_i = sign(r) · edge_weight · h_w_proj[wallet_idx]
    msg_to_market = scatter_add(msg_i, market_idx)   # aggregate by target market

h_market_new = LayerNorm(h_market + msg_to_market)   # residual update


── Market → Wallet Messages ──────────────────────────────

h_m_proj = Dropout(0.1)(Linear(h_market))            # (N_markets, 128)

For each relation r:
    msg_j = edge_weight · h_m_proj[market_idx]
    msg_to_wallet = scatter_add(msg_j, wallet_idx)   # aggregate by target wallet

h_wallet_new = LayerNorm(h_wallet + msg_to_wallet)   # residual update
```

**Design notes**:
- The **sign** encodes directional intent: a large YES buy sends a strong positive signal to the market, while a large YES sell sends a negative one.
- Wallet→Market uses signed messages (the market "learns" the aggregate sentiment of its traders). Market→Wallet uses unsigned messages (the wallet absorbs information about market state regardless of trade direction).
- Both directions use residual connections with LayerNorm to maintain training stability.

### 4.4 Predictor Head

**Purpose**: Binary classifier mapping market embeddings to price direction logits.

```
Linear(128 → 64) → SiLU → Dropout(0.1) → Linear(64 → 1)
```

Output: one logit per market, interpreted as $P(\text{price up})$ after sigmoid.

### 4.5 Full Forward Pass (CausalDynamicGNN)

The `forward` method orchestrates one block step:

```
Step 1: LOBUpdateModule
    h_market ← LOBUpdate(windows, h_market, active_idx)

Step 2: PersistentHeteroAggregator
    h_market, h_wallet ← Aggregator(h_market, h_wallet, edges)

Step 3: Predictor
    logits ← Predictor(h_market)

Return: logits, h_market, h_wallet
```

**Parameters**: 86,656 in the LOBUpdateModule + aggregator and predictor parameters.

---

## 5. Training Loop Structure

**File**: [train_new.py](file:///home/vitopampinella/Desktop/Magistrale/Tesi/agy_project/train_new.py)

### 5.1 Initialization Phase (once)

1. Load `market_to_idx` and `wallet_to_idx` mappings
2. Load `market_features.pt` → project to hidden\_dim → initialize $\mathbf{H}^{\text{market}}$ on GPU
3. Initialize $\mathbf{H}^{\text{wallet}}$ to zeros on GPU
4. Load `LOBWindowBuffer` with the first hour's LOB data + lifecycle events
5. Fast-forward lifecycle to the training start time via `reset_to_time()`

### 5.2 Outer Loop: Hours

```
For each (year, month, day, hour) in training_hours:
    1. Load all transactions for this hour (single DuckDB query)
    2. Swap LOB data via buffer.reload_lob_data(new_hour_path)
    3. Group transactions by block_number
    4. Enter inner loop
```

`reload_lob_data` clears the old hour's tensors from memory and loads the new ones, recomputing `_open_with_data` without resetting the lifecycle pointer.

### 5.3 Inner Loop: Blocks

```
For each block in this hour (sorted by block_number):
    t_ns      = exact block timestamp (from first transaction in the block)
    t_prev_ns = timestamp of the previous block

    Step A — LOB Update:
        active_ids, windows = buffer.load_step(t_prev_ns, t_ns)
        Convert market IDs → integer indices via market_to_idx
        Push windows tensor (N_active, 5, 52) to GPU

    Step B — Transaction Edges:                    [TODO]
        Build edge_index_dict and edge_weight_dict from block transactions

    Step C — Forward Pass + Loss:                  [TODO]
        logits, h_market, h_wallet = model.forward(...)
        loss = BCE(logits[predict_mask], targets)
        loss.backward()

    TBPTT — Truncated Backpropagation Through Time:
        Every T steps, call optimizer.step() and detach h_market, h_wallet
        to prevent the computation graph from growing unboundedly.
```

### 5.4 Measured Performance (1 hour, Step A only)

| Metric | Value |
|---|---|
| Blocks processed | 1,800 |
| Total LOB updates | 244,118 |
| Avg step time | ~10ms (data prep) + ~0.4ms (GPU) |
| Wall clock for 1 hour of data | ~23 seconds |
| Transactions loaded per hour | ~580,000 |

---

## 6. Complexity Summary

| Operation | Per-step complexity | Typical size |
|---|---|---|
| Lifecycle advance | O(k) amortized, k = events in interval | ~0–2 events |
| Active market scan | O(\|open\_with\_data\| × log N) | 9,725 × log(22) |
| Window retrieval (per market) | O(K + log N) | 5 + 17 |
| GRU forward (all active markets) | O(N\_active × K × d²) | 140 × 5 × 128² |
| Message passing | O(E × d) per relation | varies per block |
| Predictor | O(N\_markets × d) | 175,292 × 128 |

---

## 7. File Index

| File | Role |
|---|---|
| [causal_gnn.py](file:///home/vitopampinella/Desktop/Magistrale/Tesi/agy_project/src/polymarket_gnn/models/causal_gnn.py) | Model definition (LOBUpdateModule, Aggregator, Predictor, CausalDynamicGNN) |
| [lob_window_buffer.py](file:///home/vitopampinella/Desktop/Magistrale/Tesi/agy_project/src/polymarket_gnn/data/lob_window_buffer.py) | In-memory LOB data structure with lifecycle-aware querying |
| [lob_builder.py](file:///home/vitopampinella/Desktop/Magistrale/Tesi/agy_project/src/polymarket_gnn/data/lob_builder.py) | Offline LOB tensor precomputation from raw orderbook events |
| [train_new.py](file:///home/vitopampinella/Desktop/Magistrale/Tesi/agy_project/train_new.py) | Training loop orchestrating the full pipeline |
| [market_features.pt](file:///home/vitopampinella/Desktop/Magistrale/Tesi/agy_project/data/derived/market_features.pt) | Precomputed market metadata embeddings (175292 × 390) |
| [market_lifecycle_events.parquet](file:///home/vitopampinella/Desktop/Magistrale/Tesi/agy_project/data/derived/market_lifecycle_events.parquet) | Sorted OPEN/CLOSE events for all markets |
