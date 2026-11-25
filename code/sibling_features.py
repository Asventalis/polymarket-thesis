"""sibling_features.py — features of a market's sibling contracts (the same game, or the same question template and
end date: the candidates of one election, the strikes of one ladder, the Yes/No pair of one seat) at every row.

  groups  every market → its group key: sports / esports slugs of the form <league>-<a>-<b>-<date>-... share the game;
          the others share the question with its named entities and numbers replaced by placeholders, plus the end date;
          groups of 2–60 markets are kept  →  <out>/groups.parquet (market_id, group_key, kind, size)
  rows    (market_id, t) rows → 10 causal features from the siblings' last tensor rows at or before t (within 6 h):
          number of siblings and of active ones, mean 15 min and 1 h return, largest |1 h return|, sum of the siblings'
          mids, hours since the latest sibling mid change, net 15 min trade flow, mean sign of the 15 min order-flow
          imbalance, has-siblings  →  --out-rows parquet, or day=... files at the LOB tensor rows (--tensor-days)

    ./run_dev.sh sibling_features.py groups
    ./run_dev.sh sibling_features.py rows --rows a.parquet b.parquet --out-rows x.parquet
    ./run_dev.sh sibling_features.py rows --tensor-days 6-13
"""
from __future__ import annotations

import argparse, json, re, sys, time
from pathlib import Path

import duckdb, numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from news_features import question_terms  # noqa: E402

DATA = Path("/data/v.pampinella/agy_project/data")
HAND = DATA / "derived/hand_features"
OUT_DEFAULT = DATA / "derived/sibling_features"
FEATURES = ["sib_log_n_sib", "sib_log_n_active", "sib_ret_15m_mean", "sib_ret_1h_mean", "sib_absret_1h_max", "sib_mid_sum",
            "sib_log_since_move_h", "sib_flow_15m_asinh", "sib_ofi_15m_sign_mean", "sib_has_sib"]
GAME_RE = re.compile(r"^([a-z0-9]+)-([a-z0-9]+)-([a-z0-9]+)-(\d{4}-\d{2}-\d{2})")
MAX_GROUP = 60
STALE_NS = 6 * 3600 * 10**9

t0 = time.time()
def log(m): print(f"[{time.time()-t0:7.0f}s] {m}", flush=True)

def norm_question(q: str) -> str:
    s = q
    for t in sorted(set(question_terms(q)), key=len, reverse=True):
        s = s.replace(t, "<E>")
    s = re.sub(r"\$?\d[\d,.]*[kKmMbB]?", "<N>", s)
    return re.sub(r"\s+", " ", s).strip().lower()

def stage_groups(a):
    u = pd.read_parquet(DATA / "universe_markets.parquet", columns=["market_id", "question", "slug", "tags", "end_date"])
    sport = u.tags.fillna("").str.contains("sports|esports|games")
    game = u.slug.fillna("").map(lambda s: (lambda m: "game:" + "-".join(m.groups()) if m else None)(GAME_RE.match(s)))
    game = game.where(sport, None)
    tmpl = "tmpl:" + u.question.fillna("").map(norm_question) + "|" + u.end_date.dt.strftime("%Y-%m-%d").fillna("none")
    key = game.where(game.notna(), tmpl)
    kind = np.where(game.notna(), "game", "template")
    g = pd.DataFrame({"market_id": u.market_id, "group_key": key, "kind": kind})
    size = g.groupby("group_key").market_id.transform("size")
    g["size"] = size
    g = g[(size >= 2) & (size <= MAX_GROUP)].reset_index(drop=True)
    a.out.mkdir(parents=True, exist_ok=True); g.to_parquet(a.out / "groups.parquet", index=False)
    log(f"groups: {g.group_key.nunique():,} groups of 2–{MAX_GROUP} markets, {len(g):,} of {len(u):,} markets; "
        f"by kind {g.drop_duplicates('group_key').kind.value_counts().to_dict()}; size quantiles {g.drop_duplicates('group_key')['size'].quantile([.5, .9, .99]).tolist()}")
    ex = g[g.kind == "template"].groupby("group_key").market_id.apply(list)
    qs = u.set_index("market_id").question
    for k in ex.sample(8, random_state=0).index:
        print("   ", k[:60], "->", [qs[m][:45] for m in ex[k][:4]])

def connect():
    con = duckdb.connect(); con.sql("set threads=96; set memory_limit='600GB'; set preserve_insertion_order=false"); return con

def features_for_rows(con, rows: pd.DataFrame, days: list[int], groups_path: Path) -> np.ndarray:
    """rows: market_id, t (int64 ns); days: hand-feature days whose rows may serve as the siblings' last rows."""
    files = ", ".join(f"'{HAND}/day=2026-03-{d:02d}.parquet'" for d in days)
    con.register("rows_df", rows.assign(rid=np.arange(len(rows))))
    con.sql(f"""create or replace temp table feat as
      select market_id, ts, mid, ret_15m, ret_1h, ofi_15m, flow_15m, t_last_move from read_parquet([{files}])""")
    con.sql(f"""create or replace temp table grp as select market_id, group_key, size from read_parquet('{groups_path}')""")
    con.sql("""create or replace temp table pairs as
      select r.rid, r.t, g2.market_id sib, g.size
      from rows_df r join grp g on g.market_id = r.market_id join grp g2 on g2.group_key = g.group_key and g2.market_id <> r.market_id""")
    agg = con.sql(f"""
      with j as (
        select p.rid, p.t, p.size, f.ts, f.mid, f.ret_15m, f.ret_1h, f.ofi_15m, f.flow_15m, f.t_last_move
        from pairs p asof left join feat f on f.market_id = p.sib and f.ts <= p.t)
      select rid, any_value(size) size,
        count(ts) filter (where ts >= t - {STALE_NS}) n_active,
        avg(ret_15m) filter (where ts >= t - {STALE_NS}) ret_15m_mean,
        avg(ret_1h) filter (where ts >= t - {STALE_NS}) ret_1h_mean,
        max(abs(ret_1h)) filter (where ts >= t - {STALE_NS}) absret_1h_max,
        sum(mid) filter (where ts >= t - {STALE_NS}) mid_sum,
        max(t_last_move) filter (where ts >= t - {STALE_NS}) t_last_move_max,
        sum(flow_15m) filter (where ts >= t - {STALE_NS}) flow_15m_sum,
        avg(sign(ofi_15m)) filter (where ts >= t - {STALE_NS}) ofi_sign_mean,
        any_value(t) t
      from j group by rid""").df()
    con.unregister("rows_df")
    f = np.zeros((len(rows), len(FEATURES)), np.float32)
    i = agg.rid.values
    n_sib = agg["size"].values - 1; n_act = agg.n_active.fillna(0).values
    since_h = np.where(agg.t_last_move_max.notna(), (agg.t.values - agg.t_last_move_max.fillna(0).values.astype(np.int64)) / 3.6e12, 24.0)
    f[i, 0] = np.log1p(n_sib); f[i, 1] = np.log1p(n_act)
    f[i, 2] = agg.ret_15m_mean.fillna(0).values; f[i, 3] = agg.ret_1h_mean.fillna(0).values; f[i, 4] = agg.absret_1h_max.fillna(0).values
    f[i, 5] = agg.mid_sum.fillna(0).values; f[i, 6] = np.log1p(np.clip(since_h, 0, 24)); f[i, 7] = np.arcsinh(agg.flow_15m_sum.fillna(0).values / 1000.0)
    f[i, 8] = agg.ofi_sign_mean.fillna(0).values; f[i, 9] = 1.0
    return f

def stage_rows(a):
    con = connect(); groups_path = a.out / "groups.parquet"
    if a.rows:
        parts = []
        for p in a.rows:
            df = pd.read_parquet(p, columns=["market_id", "t"])
            t_ns = pd.to_datetime(df.t).values.astype("datetime64[ns]").astype("int64")
            days = sorted(set(pd.to_datetime(df.t).dt.day.unique()))
            fs = []
            for d in days:
                m = pd.to_datetime(df.t).dt.day.values == d
                f = features_for_rows(con, pd.DataFrame({"market_id": df.market_id.values[m], "t": t_ns[m]}), [x for x in (d - 1, d) if 6 <= x <= 13], groups_path)
                out = np.zeros((len(df), len(FEATURES)), np.float32); out[m] = f; fs.append(out)
                log(f"{Path(p).name} day {d}: {m.sum():,} rows, has_sib {f[:, -1].mean():.3f}, active {np.expm1(f[:, 1]).mean():.2f}")
            f = np.sum(fs, 0)
            parts.append(pd.concat([df.reset_index(drop=True), pd.DataFrame(f, columns=FEATURES)], axis=1))
        pd.concat(parts, ignore_index=True).to_parquet(a.out_rows, index=False); log(f"wrote {a.out_rows}")
    if a.tensor_days:
        lo, hi = (int(x) for x in a.tensor_days.split("-"))
        for d in range(lo, hi + 1):
            df = pd.read_parquet(HAND / f"day=2026-03-{d:02d}.parquet", columns=["market_id", "ts"])
            f = features_for_rows(con, pd.DataFrame({"market_id": df.market_id.values, "t": df.ts.values.astype(np.int64)}), [x for x in (d - 1, d) if 6 <= x <= 13], groups_path)
            pd.concat([df.reset_index(drop=True), pd.DataFrame(f, columns=FEATURES)], axis=1).to_parquet(a.out / f"day=2026-03-{d:02d}.parquet", index=False)
            log(f"day {d}: {len(df):,} rows, has_sib {f[:, -1].mean():.3f}")
        stat_days = [d for d in range(lo, hi + 1) if 6 <= d <= 12]
        X = np.concatenate([pd.read_parquet(a.out / f"day=2026-03-{d:02d}.parquet", columns=FEATURES).values[::10] for d in stat_days])
        stats = {}
        for i, c in enumerate(FEATURES):
            col = X[:, i].astype(np.float64); loc = float(np.median(col)); q75, q25 = np.percentile(col, [75, 25])
            stats[c] = {"loc": loc, "scale": float((q75 - q25) / 1.349) or float(np.std(col)) or 1.0}
        (a.out / "meta.json").write_text(json.dumps({"features": FEATURES, "stale_s": STALE_NS // 10**9, "stat_days": stat_days, "stats": stats}, indent=1))
    log("rows done")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["groups", "rows"])
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--rows", nargs="*"); ap.add_argument("--out-rows", type=Path); ap.add_argument("--tensor-days")
    a = ap.parse_args()
    {"groups": stage_groups, "rows": stage_rows}[a.stage](a)
    log("ALL DONE")
