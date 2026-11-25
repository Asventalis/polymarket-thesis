"""The 'time first, then space' LOB window encoder (TimeFirstEncoder, --lob-encoder timefirst)."""
import torch

from polymarket_gnn.data.graph_builder import EDGE_DIM
from polymarket_gnn.data.lob_layout import MID_IDX
from polymarket_gnn.models.causal_gnn import CausalDynamicGNN, N_SERIES_DEFAULT, TimeFirstEncoder
from polymarket_gnn.models.sparse_state import DecayClock, make_lazy_states, sparse_step

T, HID, N_CTX = 16, 24, 7
LOB = N_SERIES_DEFAULT + N_CTX


def windows(n, seed=0, n_pad=0):
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(n, T, LOB, generator=g, dtype=torch.float64)
    w[:, :, MID_IDX] = torch.rand(n, T, generator=g, dtype=torch.float64) * 0.9 + 0.05   # positive mids
    w[:, :, N_SERIES_DEFAULT:] = w[:, -1:, N_SERIES_DEFAULT:]                              # context: constant rows
    if n_pad:
        w[:, :n_pad, :] = 0.0                                                              # zero padding at the start
    return w


def test_shapes_and_padding_are_finite():
    enc = TimeFirstEncoder(LOB, T, HID).double()
    for n_pad in (0, 5, T - 1):
        h = enc(windows(6, n_pad=n_pad))
        assert h.shape == (6, HID) and torch.isfinite(h).all()


def test_padding_rows_do_not_change_the_summary_of_the_real_rows():
    # A window whose padding rows are replaced by garbage must give the same output: the encoder
    # only reads the rows with a positive mid.
    enc = TimeFirstEncoder(LOB, T, HID).double().eval()
    w = windows(4, n_pad=6)
    w2 = w.clone()
    w2[:, :6, :] = torch.randn_like(w2[:, :6, :])
    w2[:, :6, MID_IDX] = 0.0
    assert torch.allclose(enc(w), enc(w2))


def test_time_stage_is_channel_independent_and_shift_invariant():
    # Adding a constant to one channel (in all rows) leaves its normalised series unchanged; only the
    # 'last value' scalar of that channel moves, through scale_proj.
    enc = TimeFirstEncoder(LOB, T, HID).double().eval()
    with torch.no_grad():
        enc.scale_proj.weight.zero_(); enc.scale_proj.bias.zero_()
    w = windows(3)
    w2 = w.clone(); w2[:, :, 3] += 5.0
    assert torch.allclose(enc(w), enc(w2), atol=1e-9)


def test_wrong_window_length_is_rejected():
    enc = TimeFirstEncoder(LOB, T, HID)
    try:
        enc(torch.randn(2, T + 1, LOB)); raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_gru_encoder_is_unchanged_and_sparse_equals_dense_for_timefirst():
    torch.manual_seed(0)
    gru_model = CausalDynamicGNN(lob_dim=LOB, hidden_dim=HID, edge_dim=EDGE_DIM, window=T).double().eval()
    w = windows(5)
    _, h = gru_model.lob_update.gru(w)
    assert torch.equal(gru_model.lob_update.encode(w), h.squeeze(0))

    tf = CausalDynamicGNN(lob_dim=LOB, hidden_dim=HID, edge_dim=EDGE_DIM, lob_encoder="timefirst", window=T).double().eval()
    n_m, n_w = 12, 5
    g = torch.Generator().manual_seed(1)
    h_m = torch.randn(n_m, HID, generator=g, dtype=torch.float64)
    h_w = torch.randn(n_w, HID, generator=g, dtype=torch.float64)
    active = torch.tensor([1, 4, 7, 9]); wins = windows(4, seed=3, n_pad=2)
    ei = torch.empty(2, 0, dtype=torch.long); ef = torch.empty(0, EDGE_DIM, dtype=torch.float64)
    with torch.no_grad():
        dense_logits, hm_d, _ = tf(wins, active, h_m.clone(), h_w.clone(), ei, ef,
                                   torch.tensor(1.5, dtype=torch.float64), torch.tensor(1.5, dtype=torch.float64),
                                   predict_idx=active)
        clock = DecayClock("cpu"); sm, sw = make_lazy_states(tf, h_m.clone(), h_w.clone(), clock)
        sparse_logits = sparse_step(tf, clock, sm, sw, wins, active, ei, ef, 1.5, predict_idx=active)
        hm_s = sm.materialize()
    assert torch.allclose(dense_logits, sparse_logits, atol=1e-9)
    assert torch.allclose(hm_d[active], hm_s[active], atol=1e-9)
