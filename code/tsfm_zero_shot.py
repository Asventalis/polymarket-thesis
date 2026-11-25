"""
tsfm_zero_shot.py — step 1 of the foundation-model study: zero-shot Chronos-2 forecasts of the
mid price at t+H, on a random subsample of the validation samples of a full-length run, turned
into DOWN/CONSTANT/UP decisions and compared with Mean Reversion and with the run's own
predictions on the same samples (hour-block bootstrap).  No training: validation data only.

    PYTHONPATH=/data/v.pampinella/tsfm_pkgs HF_HOME=/data/v.pampinella/hf_cache \
    ./run_dev.sh tsfm_zero_shot.py --horizon 3600 --run-label 1h_lb_hm_sg_lb5_ablation_pred_s0 --step-min 1
    ... --horizon 43200 --run-label 12h_lb_hm_sg_lb5_ablation_pred_s0 --step-min 5

Context: ``--context`` values of the mid sampled every ``--step-min`` minutes, each the last mid
strictly before the grid time (NaN where the market had no snapshot yet), the last one being the mid
the run predicted from (last mid strictly before t).  Chronos-2 forecasts H / step steps ahead; the
quantiles of the last step are kept.  Decision rules (all causal, none fitted on the data):
  median:        UP if q50 > mid, DOWN if q50 < mid, else CONSTANT
  median±tau:    CONSTANT if |q50 - mid| <= tau (tau in ticks of 0.001), else sign
  band(lo,hi):   UP if q_lo > mid, DOWN if q_hi < mid, else CONSTANT
The forecasts are saved (parquet) so other rules can be evaluated later without re-running the model.
"""
import argparse
import os
import sys
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, "/data/v.pampinella/tsfm_pkgs")
os.environ.setdefault("HF_HOME", "/data/v.pampinella/hf_cache")

from polymarket_gnn.data.lob_layout import MID_IDX  # noqa: E402

HOUR_NS = 3_600_000_000_000
QS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def ba_from_cm(cm):
    r = cm.sum(axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        rec = np.diagonal(cm, axis1=-2, axis2=-1) / r
    return np.nanmean(rec, axis=-1)


def hourly_cms(df, col):
    h_idx = pd.factorize(df["t"].values.astype("int64") // HOUR_NS, sort=True)[0]
    flat = np.bincount(h_idx * 9 + df["y_true"].values.astype(int) * 3 + df[col].values.astype(int),
                       minlength=(h_idx.max() + 1) * 9)
    return flat.reshape(-1, 3, 3).astype(float)


def hour_paths(base_dir, k0, k1):
    out = []
    for k in range(k0, k1 + 1):
        ts = pd.Timestamp(k * HOUR_NS)
        p = base_dir / f"tensors/year={ts.year}/month={ts.month}/day={ts.day}/hour={ts.hour}/lob_features.parquet"
        if p.exists():
            out.append(str(p))
    return out


def build_contexts(base_dir, sample, step_ns, context):
    """(n, context) float32 matrix of mids on the step grid ending at the sample time (NaN = unknown)."""
    t = sample["t"].values.astype("int64")
    g_end = (t // step_ns) * step_ns                       # last grid time <= t, per sample
    grid0 = (t.min() // step_ns) * step_ns - (context - 1) * step_ns
    grid1 = g_end.max()
    grid = np.arange(grid0, grid1 + step_ns, step_ns, dtype="int64")
    paths = hour_paths(base_dir, grid0 // HOUR_NS, t.max() // HOUR_NS)
    ids = sample["market_id"].unique()
    con = duckdb.connect()
    con.register("ids", pd.DataFrame({"market_id": ids}))
    files = ", ".join(f"'{p}'" for p in paths)
    rows = con.execute(
        f"SELECT l.market_id, epoch_ns(l.timestamp) AS ts, l.features[{MID_IDX + 1}] AS mid "
        f"FROM read_parquet([{files}]) l JOIN ids USING (market_id) ORDER BY l.market_id, ts"
    ).fetch_df()
    m_codes, m_uniq = pd.factorize(rows["market_id"].values, sort=True)
    ts, mid = rows["ts"].values.astype("int64"), rows["mid"].values.astype("float32")
    starts = np.searchsorted(m_codes, np.arange(len(m_uniq)), side="left")
    ends = np.searchsorted(m_codes, np.arange(len(m_uniq)), side="right")
    M = np.full((len(m_uniq), len(grid)), np.nan, dtype=np.float32)
    for i, (a, b) in enumerate(zip(starts, ends)):
        pos = np.searchsorted(ts[a:b], grid, side="left") - 1     # last snapshot strictly before grid time
        ok = pos >= 0
        M[i, ok] = mid[a:b][pos[ok]]
    row_of = pd.Series(np.arange(len(m_uniq)), index=m_uniq)
    r = row_of.reindex(sample["market_id"].values).values
    c_end = ((g_end - grid0) // step_ns).astype(int)
    offs = np.arange(-(context - 1), 0)
    X = np.full((len(sample), context), np.nan, dtype=np.float32)
    has = ~np.isnan(r)
    ri = r[has].astype(int)
    X[has, :-1] = M[ri[:, None], c_end[has][:, None] + offs[None, :]]
    X[:, -1] = sample["mid"].values.astype(np.float32)
    return X, len(paths), len(m_uniq)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, required=True)
    ap.add_argument("--run-label", required=True)
    ap.add_argument("--step-min", type=int, default=1)
    ap.add_argument("--context", type=int, default=512)
    ap.add_argument("--n-samples", type=int, default=60000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--model", default="amazon/chronos-2")
    ap.add_argument("--base-dir", type=Path, default=Path("/data/v.pampinella/agy_project/data/derived"))
    ap.add_argument("--out-dir", type=Path, default=Path("tsfm_zero_shot"))
    ap.add_argument("--pred-dir", type=Path, default=Path("."), help="directory of val_predictions_<run-label>.parquet")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    H = f"{args.horizon // 3600}h"
    step_ns = args.step_min * 60 * 1_000_000_000
    assert args.horizon % (args.step_min * 60) == 0
    pred_len = args.horizon // (args.step_min * 60)
    rng = np.random.default_rng(args.seed)

    preds = pd.read_parquet(args.pred_dir / f"val_predictions_{args.run_label}.parquet")
    sample = preds.iloc[np.sort(rng.choice(len(preds), size=min(args.n_samples, len(preds)), replace=False))].reset_index(drop=True)
    mapping = pd.read_parquet(args.base_dir / "id_mappings/market_to_idx.parquet")
    idx_to_id = pd.Series(mapping["market_id"].values, index=mapping["market_idx"].values)
    sample["market_id"] = idx_to_id.reindex(sample["market_idx"].values).values
    print(f"[{H}] {len(sample)} of {len(preds)} validation samples, {sample['market_id'].nunique()} markets", flush=True)

    t0 = time.time()
    X, n_files, n_mk = build_contexts(args.base_dir, sample, step_ns, args.context)
    known = (~np.isnan(X)).sum(1)
    print(f"[{H}] contexts built from {n_files} hour files, {n_mk} markets with snapshots, "
          f"{time.time() - t0:.0f}s; known points per context: median {np.median(known):.0f}, "
          f"min {known.min()}, share full {np.mean(known == args.context):.2f}", flush=True)

    import torch
    from chronos import Chronos2Pipeline
    torch.set_num_threads(args.threads)
    pipe = Chronos2Pipeline.from_pretrained(args.model, device_map=args.device, dtype=torch.float32)
    Q = np.zeros((len(sample), len(QS)), dtype=np.float32)
    MEAN = np.zeros(len(sample), dtype=np.float32)
    t0 = time.time()
    for b in range(0, len(sample), args.batch_size):
        ctx = [X[i] for i in range(b, min(b + args.batch_size, len(sample)))]
        q, m = pipe.predict_quantiles(ctx, prediction_length=pred_len, quantile_levels=QS)
        Q[b:b + len(ctx)] = np.stack([qi[0, -1].numpy() for qi in q])
        MEAN[b:b + len(ctx)] = np.array([mi[0, -1].item() for mi in m])
        if (b // args.batch_size) % 20 == 0:
            print(f"[{H}] {b + len(ctx)}/{len(sample)} forecasts, {time.time() - t0:.0f}s", flush=True)
    for j, qv in enumerate(QS):
        sample[f"q{int(qv * 100)}"] = Q[:, j]
    sample["fc_mean"] = MEAN
    sample["known_points"] = known
    sample.to_parquet(args.out_dir / f"forecasts_{H}_{args.run_label}.parquet", index=False)

    # ── decision rules ──────────────────────────────────────────────────────────────────────
    mid = sample["mid"].values.astype(np.float64)
    pdl = sample["past_delta"].values
    sample["MeanRev"] = np.where(np.isnan(pdl) | (pdl == 0), 1, np.where(pdl > 0, 0, 2))
    sample["model"] = sample["y_pred"].values
    rules = ["model", "MeanRev"]

    def sign_rule(d, tau=0.0):
        return np.where(d > tau, 2, np.where(d < -tau, 0, 1)).astype(np.int8)
    d50 = sample["q50"].values - mid
    sample["chronos_median"] = sign_rule(d50); rules.append("chronos_median")
    for tau in (0.0005, 0.001, 0.002, 0.005):
        k = f"chronos_median_tau{tau:g}"; sample[k] = sign_rule(d50, tau); rules.append(k)
    for lo, hi in ((40, 60), (30, 70), (20, 80), (10, 90)):
        k = f"chronos_band{lo}_{hi}"
        sample[k] = np.where(sample[f"q{lo}"].values > mid, 2, np.where(sample[f"q{hi}"].values < mid, 0, 1)).astype(np.int8)
        rules.append(k)
    sample["chronos_mean"] = sign_rule(sample["fc_mean"].values - mid); rules.append("chronos_mean")
    # combination: Mean Reversion where it takes a side, Chronos median elsewhere
    sample["MR_else_chronos"] = np.where(sample["MeanRev"] != 1, sample["MeanRev"], sample["chronos_median"]).astype(np.int8)
    rules.append("MR_else_chronos")

    lines = [f"# Zero-shot Chronos-2 ({args.model}) at {H}: {len(sample):,} validation samples of `{args.run_label}`\n",
             f"Context {args.context} points every {args.step_min} min, forecast {pred_len} steps; contexts fully known: "
             f"{np.mean(known == args.context):.0%} (median known points {np.median(known):.0f}). Balanced accuracy; "
             "Δ = rule − Mean Reversion with 95% hour-block bootstrap interval. The tau/band variants are alternative "
             "rules shown side by side, not fitted; reading the best of them off this table is a validation-set choice.\n"]
    subsets = {"all": sample, "dedup": sample[sample["dedup"]],
               "extreme (mid<0.05 or >0.95)": sample[(mid < 0.05) | (mid > 0.95)],
               "central (mid in [0.05,0.95])": sample[(mid >= 0.05) & (mid <= 0.95)]}
    for name, df in subsets.items():
        cms = {k: hourly_cms(df, k) for k in rules}
        pt = {k: ba_from_cm(cms[k].sum(0)) for k in rules}
        nh = len(cms["MeanRev"])
        boot = rng.integers(0, nh, size=(args.n_boot, nh))
        bs = {k: ba_from_cm(cms[k][boot].sum(1)) for k in rules}
        lines += [f"\n## {name} (n = {len(df):,})\n", "| rule | bal. acc. | Δ vs MR (pt) | 95% CI | pred. share D/C/U |", "|---|---|---|---|---|"]
        print(f"-- {name} n={len(df)}")
        for k in rules:
            dm = bs[k] - bs["MeanRev"]; lo, hi = np.percentile(dm, [2.5, 97.5])
            share = np.bincount(df[k].values.astype(int), minlength=3) / max(len(df), 1)
            s = f"{k:<26} {pt[k]:.4f}  Δ {100 * (pt[k] - pt['MeanRev']):+.2f} [{100 * lo:+.2f}, {100 * hi:+.2f}]  D/C/U {share[0]:.2f}/{share[1]:.2f}/{share[2]:.2f}"
            print(s)
            lines.append(f"| {k} | {pt[k]:.4f} | {100 * (pt[k] - pt['MeanRev']):+.2f} | [{100 * lo:+.2f}, {100 * hi:+.2f}] | {share[0]:.2f}/{share[1]:.2f}/{share[2]:.2f} |")
    # head-to-head where Chronos and Mean Reversion take opposite sides and the price moved
    mr, ch, y = sample["MeanRev"].values, sample["chronos_median"].values, sample["y_true"].values
    opp = (mr != 1) & (ch != 1) & (mr != ch) & (y != 1)
    agree = (mr != 1) & (ch == mr)
    lines += ["\n## Head-to-head\n",
              f"- Mean Reversion takes a side on {np.mean(mr != 1):.1%} of the samples; Chronos (median) agrees with it on "
              f"{agree.sum() / max((mr != 1).sum(), 1):.1%} of those.",
              f"- Opposite sides and the price moved: {opp.sum():,} samples; Chronos right in {np.mean(ch[opp] == y[opp]):.1%}, "
              f"Mean Reversion right in {np.mean(mr[opp] == y[opp]):.1%}.",
              f"- Model vs Chronos (median), opposite sides and price moved: "
              f"{np.mean((sample['model'].values[(sample['model'].values != 1) & (ch != 1) & (sample['model'].values != ch) & (y != 1)]) == y[(sample['model'].values != 1) & (ch != 1) & (sample['model'].values != ch) & (y != 1)]):.1%} model right."]
    (args.out_dir / f"report_{H}_{args.run_label}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[-3:]))


if __name__ == "__main__":
    main()
