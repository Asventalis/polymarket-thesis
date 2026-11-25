"""
analyze_predictions.py

Breakdown of validation performance by market type, from the per-sample
predictions written by ``train_full.py --save-val-predictions``.

    python analyze_predictions.py --pred val_predictions_1h_lb_s0.parquet --name "full" \\
        [--pred2 val_predictions_1h_lb_ablation_s0.parquet --name2 "LOB only"] --out-dir analysis/1h

For every group it reports the number of samples and markets, the class mix,
and balanced accuracy / macro-F1 / accuracy of the model next to the causal
baselines computed on exactly the same samples:
    Mean Reversion H = class of -(mid(t) - mid(t-H)),  Momentum H = class of +(mid(t) - mid(t-H)).

Groupings
    category        exclusive, from the market tags (priority: esports, sports, crypto, weather,
                    politics, economy/tech, culture, other)
    tag             non-exclusive, the most frequent tags
    duration        end_date - start_date of the market
    time_to_end     end_date - t at prediction time
    sports_phase    sports/esports only: before or after game_start_time
    activity_tx     number of transactions of the market in the training days (quartiles)
    activity_usdc   traded USDC in the training days (quartiles)
    price_level     mid price at prediction time
    hour_utc        hour of the day

Balanced accuracy inside a group depends on that group's class mix and is noisy
for small groups: groups below --min-samples or --min-markets are listed separately.
Everything is computed on all samples and on the deduplicated subset.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

DATA = Path("/data/v.pampinella/agy_project/data")

CATEGORY_RULES = [
    ("esports", {"esports"}),
    ("sports", {"sports"}),
    ("crypto", {"crypto", "crypto-prices", "bitcoin", "ethereum", "solana"}),
    ("weather", {"weather"}),
    ("politics", {"politics", "elections", "geopolitics", "global-elections", "world-elections", "world", "trump",
                  "midterms", "primaries", "us-presidential-election"}),
    ("economy_tech", {"finance", "economy", "business", "tech", "big-tech", "ai", "ipos", "macro-indicators",
                      "pre-market", "economic-policy", "fdv"}),
    ("culture", {"pop-culture", "awards", "movies", "music", "celebrities", "oscars"}),
]


def category_of(tags: frozenset) -> str:
    for name, keys in CATEGORY_RULES:
        if tags & keys:
            return name
    return "other" if tags else "no_tags"


def bucket(values, edges, labels):
    idx = np.searchsorted(np.asarray(edges, dtype=np.float64), values, side="right")
    out = np.array(labels, dtype=object)[idx]
    out[np.isnan(values)] = "unknown"
    return out


def direction(delta: np.ndarray) -> np.ndarray:
    out = np.ones(len(delta), dtype=np.int8)
    with np.errstate(invalid="ignore"):
        out[delta > 0] = 2
        out[delta < 0] = 0
    return out


def metrics_by_group(codes: np.ndarray, n_groups: int, y: np.ndarray, pred: np.ndarray) -> pd.DataFrame:
    """Vectorised balanced accuracy / macro-F1 / accuracy per group."""
    cm = np.bincount(codes.astype(np.int64) * 9 + y.astype(np.int64) * 3 + pred.astype(np.int64),
                     minlength=n_groups * 9).reshape(n_groups, 3, 3).astype(np.float64)
    tp = np.einsum("gii->gi", cm)
    support, predicted = cm.sum(axis=2), cm.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        recall = np.where(support > 0, tp / support, np.nan)
        precision = np.where(predicted > 0, tp / predicted, 0.0)
        f1 = np.where(precision + np.nan_to_num(recall) > 0, 2 * precision * np.nan_to_num(recall) / (precision + np.nan_to_num(recall)), 0.0)
        n = cm.sum(axis=(1, 2))
        acc = np.where(n > 0, tp.sum(axis=1) / n, np.nan)
    present = support > 0
    bal = np.nansum(np.where(present, recall, 0.0), axis=1) / np.maximum(present.sum(axis=1), 1)
    mf1 = np.where(present, f1, 0.0).sum(axis=1) / np.maximum(present.sum(axis=1), 1)
    return pd.DataFrame({"bal_acc": bal, "macro_f1": mf1, "acc": acc})


def table_for(df: pd.DataFrame, col: str, models: dict, min_samples: int, min_markets: int) -> pd.DataFrame:
    codes, uniques = pd.factorize(df[col], sort=True)
    g = len(uniques)
    y = df["y_true"].values
    out = pd.DataFrame({col: uniques})
    out["samples"] = np.bincount(codes, minlength=g)
    out["markets"] = df.groupby(codes)["market_idx"].nunique().reindex(range(g)).values
    for c, name in enumerate(["down", "const", "up"]):
        out[f"%{name}"] = 100 * np.bincount(codes, weights=(y == c), minlength=g) / np.maximum(out["samples"], 1)
    for name, pred_col in models.items():
        m = metrics_by_group(codes, g, y, df[pred_col].values)
        out[f"{name} bal_acc"] = m["bal_acc"]
        out[f"{name} acc"] = m["acc"]
    names = list(models)
    out[f"Δ {names[0]}−MeanRev"] = out[f"{names[0]} bal_acc"] - out["MeanRev bal_acc"]
    if len(names) > 3:      # a second model was given
        out[f"Δ {names[0]}−{names[1]}"] = out[f"{names[0]} bal_acc"] - out[f"{names[1]} bal_acc"]
    out["small"] = (out["samples"] < min_samples) | (out["markets"] < min_markets)
    return out.sort_values("samples", ascending=False).reset_index(drop=True)


def fmt(t: pd.DataFrame) -> str:
    t = t.copy()
    for c in t.columns:
        if c.startswith("%"):
            t[c] = t[c].map(lambda v: f"{v:.1f}")
        elif "bal_acc" in c or c.endswith(" acc"):
            t[c] = t[c].map(lambda v: f"{v:.3f}")
        elif c.startswith("Δ"):
            t[c] = t[c].map(lambda v: f"{v:+.3f}")
    t = t.drop(columns=["small"])
    try:
        return t.to_markdown(index=False)
    except ImportError:                      # tabulate not installed
        return "```\n" + t.to_string(index=False) + "\n```"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", type=Path, required=True)
    ap.add_argument("--name", default="model")
    ap.add_argument("--pred2", type=Path, default=None)
    ap.add_argument("--name2", default="model2")
    ap.add_argument("--out-dir", type=Path, default=Path("analysis"))
    ap.add_argument("--train-days", default="6-12", help="days of March 2026 used for the activity statistics")
    ap.add_argument("--min-samples", type=int, default=20000)
    ap.add_argument("--min-markets", type=int, default=20)
    ap.add_argument("--top-tags", type=int, default=40)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.pred).rename(columns={"y_pred": "pred_model"})
    models = {args.name: "pred_model"}
    if args.pred2:
        d2 = pd.read_parquet(args.pred2)[["t", "market_idx", "y_pred"]].rename(columns={"y_pred": "pred_model2"})
        n0 = len(df)
        df = df.merge(d2, on=["t", "market_idx"], how="inner")
        print(f"merged second model: {len(df)} of {n0} samples in common")
        models[args.name2] = "pred_model2"
    df["pred_meanrev"] = direction(-df["past_delta"].values)
    df["pred_momentum"] = direction(df["past_delta"].values)
    models.update({"MeanRev": "pred_meanrev", "Momentum": "pred_momentum"})

    # ── market metadata ──────────────────────────────────────────────────────
    idx = pd.read_parquet(DATA / "derived/id_mappings/market_to_idx.parquet")
    meta = duckdb.query(f"""SELECT market_id, tags, question, start_date, end_date, game_start_time
                            FROM read_parquet('{DATA}/universe_markets.parquet')""").df().drop_duplicates("market_id")
    a, b = [int(x) for x in args.train_days.split("-")]
    files = ", ".join(f"'{DATA}/derived/partitioned/transactions/year=2026/month=3/day={d}/**/*.parquet'" for d in range(a, b + 1))
    act = duckdb.query(f"SELECT market_id, count(*) AS tx_train, sum(total_usdc) AS usdc_train FROM read_parquet([{files}]) GROUP BY 1").df()
    m = idx.merge(meta, on="market_id", how="left").merge(act, on="market_id", how="left")
    m = m[m["market_idx"].isin(df["market_idx"].unique())].copy()
    m["tagset"] = m["tags"].fillna("").map(lambda s: frozenset(t.strip().lower() for t in s.split(",") if t.strip()))
    m["category"] = m["tagset"].map(category_of)
    dur = (m["end_date"] - m["start_date"]).dt.total_seconds().values
    m["duration"] = bucket(dur, [3600, 86400, 7 * 86400, 31 * 86400, 92 * 86400],
                           ["a) <=1h", "b) 1h-1d", "c) 1d-1w", "d) 1w-1m", "e) 1m-3m", "f) >3m"])
    for col, name in (("tx_train", "activity_tx"), ("usdc_train", "activity_usdc")):
        v = m[col].fillna(0).values
        pos = v[v > 0]
        q = np.quantile(pos, [0.25, 0.5, 0.75]) if len(pos) else [1, 2, 3]
        lab = np.array(["Q1 (least)", "Q2", "Q3", "Q4 (most)"], dtype=object)[np.searchsorted(q, v, side="right")]
        lab[v <= 0] = "no trades in training days"
        m[name] = lab
    df = df.merge(m[["market_idx", "market_id", "category", "duration", "activity_tx", "activity_usdc",
                     "end_date", "game_start_time", "tagset"]], on="market_idx", how="left")

    # ── sample-level features ────────────────────────────────────────────────
    t = pd.to_datetime(df["t"])
    tte = (df["end_date"] - t).dt.total_seconds().values
    df["time_to_end"] = bucket(tte, [0, 3600, 6 * 3600, 86400, 7 * 86400, 31 * 86400],
                               ["a) past end_date", "b) <1h", "c) 1h-6h", "d) 6h-1d", "e) 1d-1w", "f) 1w-1m", "g) >1m"])
    is_sport = df["category"].isin(["sports", "esports"]).values
    started = (t >= df["game_start_time"]).values
    phase = np.where(df["game_start_time"].isna().values, "sport, no game time", np.where(started, "sport, game started", "sport, before game"))
    df["sports_phase"] = np.where(is_sport, phase, "not sport")
    df["price_level"] = bucket(df["mid"].values.astype(np.float64), [0.05, 0.2, 0.4, 0.6, 0.8, 0.95],
                               ["a) <0.05", "b) 0.05-0.2", "c) 0.2-0.4", "d) 0.4-0.6", "e) 0.6-0.8", "f) 0.8-0.95", "g) >0.95"])
    df["hour_utc"] = t.dt.hour.map(lambda h: f"{h:02d}")
    df["overall"] = "all markets"

    # ── tables ───────────────────────────────────────────────────────────────
    groupings = ["overall", "category", "duration", "time_to_end", "sports_phase", "activity_tx", "activity_usdc", "price_level", "hour_utc"]
    lines = [f"# Per-market breakdown — {args.pred.name}" + (f" vs {args.pred2.name}" if args.pred2 else ""), "",
             f"Samples: {len(df):,} | markets: {df['market_idx'].nunique():,} | deduplicated samples: {int(df['dedup'].sum()):,}",
             f"Groups with fewer than {args.min_samples:,} samples or {args.min_markets} markets are in the 'small groups' tables.", ""]
    for subset_name, sub in (("all samples", df), ("deduplicated", df[df["dedup"]])):
        ms, mm = (args.min_samples, args.min_markets) if subset_name == "all samples" else (max(args.min_samples // 8, 1), args.min_markets)
        lines += [f"## {subset_name}", ""]
        for col in groupings:
            tab = table_for(sub, col, models, ms, mm)
            tab.to_csv(args.out_dir / f"{col}__{subset_name.replace(' ', '_')}.csv", index=False)
            lines += [f"### by {col}", "", fmt(tab[~tab["small"]]), ""]
            if tab["small"].any():
                lines += ["small groups:", "", fmt(tab[tab["small"]]), ""]
        # non-exclusive tags
        tags = sub[["market_idx", "tagset"]].drop_duplicates("market_idx")
        counts = sub.groupby("market_idx").size()
        tag_n = {}
        for mi, ts in zip(tags["market_idx"].values, tags["tagset"].values):
            for tg in (ts if isinstance(ts, frozenset) else ()):
                tag_n[tg] = tag_n.get(tg, 0) + int(counts.get(mi, 0))
        rows = []
        market_tags = dict(zip(tags["market_idx"].values, tags["tagset"].values))
        for tg in sorted(tag_n, key=tag_n.get, reverse=True)[: args.top_tags]:
            keep_m = np.array([mi for mi, ts in market_tags.items() if isinstance(ts, frozenset) and tg in ts])
            part = sub[sub["market_idx"].isin(keep_m)].assign(tag=tg)
            rows.append(table_for(part, "tag", models, ms, mm))
        if rows:
            tab = pd.concat(rows).sort_values("samples", ascending=False).reset_index(drop=True)
            tab.to_csv(args.out_dir / f"tag__{subset_name.replace(' ', '_')}.csv", index=False)
            lines += [f"### by tag (non-exclusive, top {args.top_tags})", "", fmt(tab[~tab["small"]]), ""]
    report = args.out_dir / "breakdown.md"
    report.write_text("\n".join(lines))
    print(f"report written to {report}")


if __name__ == "__main__":
    main()
