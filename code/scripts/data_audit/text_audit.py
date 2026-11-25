"""Audit of the market text and of the MiniLM sentence embeddings used to initialise the market state.
CPU only. Uses the 1h validation samples to measure what the text could tell about the dynamics."""
import re, time
from pathlib import Path
import duckdb, numpy as np, pandas as pd, torch

import numpy as _np, pandas as _pd
def _md(self, index=True, floatfmt=".4g", **kw):
    df = self.to_frame() if isinstance(self, _pd.Series) else self
    df = df.reset_index() if index else df
    cols = list(df.columns)
    def fmt(v):
        if isinstance(v, (float, _np.floating)): return "nan" if _np.isnan(v) else format(v, floatfmt)
        return str(v)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for row in df.itertuples(index=False): out.append("| " + " | ".join(fmt(v) for v in row) + " |")
    return "\n".join(out)
_pd.DataFrame.to_markdown = _md
_pd.Series.to_markdown = _md
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

DATA = Path("/data/v.pampinella/agy_project/data"); ORIENT = Path("/data/v.pampinella/agy_project_orient")
OUT = Path("/data/v.pampinella/data_audit"); OUT.mkdir(exist_ok=True)
t0 = time.time(); log = lambda m: print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)
con = duckdb.connect(); con.sql("set threads=64")
L = ["# Market text and sentence-embedding audit (built 2026-10-15)\n"]

mk = con.sql(f"""select m.market_id, i.market_idx, tags, question, description, outcome_1, outcome_2, start_date, end_date, first_event_timestamp
  from '{DATA}/universe_markets.parquet' m left join '{DATA}/derived/id_mappings/market_to_idx.parquet' i using (market_id)""").df()
node = pd.read_csv(DATA / "derived/market_node_mapping.csv")
aligned = (node.sort_values("node_index").market_id.values[:len(mk)] == mk.sort_values("market_idx").market_id.values).mean() if mk.market_idx.notna().all() else float("nan")
L.append(f"Markets in the universe: {len(mk):,}; with a dense index: {mk.market_idx.notna().sum():,}. The row order of `market_features.pt` (built from `market_node_mapping.csv`, markets sorted by id) agrees with `id_mappings/market_to_idx.parquet` on {aligned:.1%} of the rows (checked position by position).\n")
log(f"alignment {aligned}")

CATS = [("esports", {"esports"}), ("sports", {"sports"}), ("crypto", {"crypto", "crypto-prices", "bitcoin", "ethereum", "solana"}),
        ("weather", {"weather"}), ("politics", {"politics", "elections", "geopolitics", "global-elections", "world-elections", "world", "trump", "midterms", "primaries", "us-presidential-election"}),
        ("economy_tech", {"finance", "economy", "business", "tech", "big-tech", "ai", "ipos", "macro-indicators", "pre-market", "economic-policy", "fdv"}),
        ("culture", {"pop-culture", "awards", "movies", "music", "celebrities", "oscars"})]
def category(tags):
    s = frozenset(t.strip().lower() for t in (tags or "").split(",") if t.strip())
    for name, keys in CATS:
        if s & keys: return name
    return "other" if s else "no_tags"
mk["category"] = mk.tags.map(category)
mk["q"] = mk.question.fillna(""); mk["d"] = mk.description.fillna("")
def template(q):
    q = re.sub(r"\d[\d,.:]*", "#", q)
    q = re.sub(r"\b(January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\b", "MONTH", q)
    q = re.sub(r"\b(AM|PM|ET|UTC)\b", "TZ", q)
    q = re.sub(r"\b[A-Z][a-z]+(?: [A-Z][a-z]+)*\b", "NAME", q)   # capitalised names (teams, people, cities)
    return q.strip()
mk["template"] = mk.q.map(template)
mk["desc_len"] = mk.d.str.len(); mk["q_len"] = mk.q.str.len()

# ── validation samples per market ─────────────────────────────────────────────────────────────
pred = ORIENT / "val_predictions_1h_lb_hm_sg_lb5_tf64_ablation_pred_s0.parquet"
v = con.sql(f"""select market_idx, count(*) n, avg((y_true <> 1)::int) p_move, avg((y_true = 2)::int) filter (where y_true <> 1) p_up_given_move,
   avg(((case when past_delta is null or past_delta = 0 then 1 else 0 end) = (case when y_true = 1 then 1 else 0 end))::int) mr_hit_all,
   avg(((case when past_delta > 0 then 0 when past_delta < 0 then 2 else 1 end) = y_true)::int) filter (where y_true <> 1 and past_delta <> 0) mr_hit_moves,
   avg((y_pred = y_true)::int) model_hit, avg((mid < 0.05 or mid > 0.95)::int) p_extreme, avg(mid) mean_mid
   from '{pred}' group by 1""").df()
mk = mk.merge(v, on="market_idx", how="left")
act = mk[mk.n.notna()].copy()
L.append(f"Markets with 1h validation samples on 13 March: {len(act):,} ({act.n.sum():,.0f} samples).\n")

# ── 1. what the text looks like ───────────────────────────────────────────────────────────────
g = mk.groupby("category").agg(markets=("market_id", "size"), active_13_mar=("n", lambda s: s.notna().sum()), val_samples=("n", "sum"),
    desc_empty=("desc_len", lambda s: (s == 0).mean()), q_len_p50=("q_len", "median"), desc_len_p50=("desc_len", "median"),
    templates=("template", "nunique")).sort_values("markets", ascending=False)
g["samples_share"] = g.val_samples / g.val_samples.sum()
L += ["\n## 1. Categories (from the tags) and the shape of the text\n", g.to_markdown(floatfmt=".3f")]
tt = act.groupby("template").agg(markets=("market_id", "size"), samples=("n", "sum"), example=("q", "first")).sort_values("samples", ascending=False)
tt["samples_share"] = tt.samples / tt.samples.sum(); tt["cum_share"] = tt.samples_share.cumsum()
L += [f"\n### Question templates (numbers, dates, times and capitalised names masked)\n\nActive markets on 13 March fall into {len(tt):,} templates; the top 20 cover {tt.samples_share.head(20).sum():.1%} of the validation samples, the top 100 {tt.samples_share.head(100).sum():.1%}.\n",
      tt.head(25)[["markets", "samples", "samples_share", "cum_share", "example"]].to_markdown(floatfmt=".3f")]
log("text stats done")

# ── 2. what the dynamics look like per category / template ────────────────────────────────────
def wavg(df, col): return np.average(df[col].fillna(0), weights=df.n)
rows = []
for c, d in act.groupby("category"):
    rows.append(dict(category=c, markets=len(d), samples=int(d.n.sum()), p_move=wavg(d, "p_move"), p_up_given_move=wavg(d, "p_up_given_move"), p_extreme=wavg(d, "p_extreme"),
                     mr_hit_moves=wavg(d, "mr_hit_moves"), model_hit=wavg(d, "model_hit")))
L += ["\n## 2. Dynamics by category (sample-weighted; `p_move` = share of samples whose mid moved within 1h; `mr_hit_moves` = how often the reversal of the last 1h move gives the direction, on moved samples)\n",
      pd.DataFrame(rows).sort_values("samples", ascending=False).to_markdown(index=False, floatfmt=".3f")]
rows = []
for tpl, d in act.groupby("template"):
    if d.n.sum() < 50_000: continue
    rows.append(dict(template=tpl[:70], markets=len(d), samples=int(d.n.sum()), p_move=wavg(d, "p_move"), p_extreme=wavg(d, "p_extreme"), mr_hit_moves=wavg(d, "mr_hit_moves"), model_hit=wavg(d, "model_hit")))
L += ["\n### The same, by question template (templates with ≥ 50,000 validation samples)\n", pd.DataFrame(rows).sort_values("samples", ascending=False).to_markdown(index=False, floatfmt=".3f")]
# how much of the between-market variance of p_move is explained by category / template (ANOVA-style, markets with >= 200 samples)
big = act[act.n >= 200].copy()
def r2_group(col, target):
    m = big[target].mean(); tot = ((big[target] - m) ** 2).sum()
    within = big.groupby(col)[target].transform("mean"); return 1 - ((big[target] - within) ** 2).sum() / tot
L.append(f"\nAmong the {len(big):,} markets with ≥ 200 samples, the between-market variance of `p_move` explained by the category is {r2_group('category','p_move'):.1%}, by the template {r2_group('template','p_move'):.1%} (in-sample, {big.template.nunique():,} templates); for `mr_hit_moves` {r2_group('category','mr_hit_moves'):.1%} and {r2_group('template','mr_hit_moves'):.1%}.\n")
log("dynamics done")

# ── 3. the embeddings ─────────────────────────────────────────────────────────────────────────
F = torch.load(DATA / "derived/market_features.pt", map_location="cpu")
E = F[:, :384].numpy().astype(np.float32); E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
T = F[:, 384:].numpy()
mk = mk.sort_values("market_idx"); idx = mk.market_idx.values.astype(int)
L += [f"\n## 3. The 390-dimensional static vector\n\n`market_features.pt` is {tuple(F.shape)}: 384 MiniLM dimensions of 'Question | Description | Tags' and 6 sine/cosine encodings of hour, weekday and month of the creation date. "
      f"NaN rows: {np.isnan(F.numpy()).any(1).sum():,}. In the training script the vector is passed through a fixed random linear projection to 128 dimensions and used only as the initial market state, which the recurrent update then overwrites; nothing is trained on it.\n"]
# nearest-neighbour purity
sub = mk[mk.n.notna()].sample(n=min(20000, int(mk.n.notna().sum())), random_state=0)
Es = E[sub.market_idx.values.astype(int)]
nn = NearestNeighbors(n_neighbors=11, metric="cosine").fit(Es); _, nb = nn.kneighbors(Es)
cat_s = sub.category.values; tpl_s = sub.template.values
pc = np.mean([(cat_s[nb[i, 1:]] == cat_s[i]).mean() for i in range(len(sub))]); pt = np.mean([(tpl_s[nb[i, 1:]] == tpl_s[i]).mean() for i in range(len(sub))])
L.append(f"10-nearest-neighbour purity (cosine, {len(sub):,} active markets): same category {pc:.1%}, same question template {pt:.1%} (category base rate {max(pd.Series(cat_s).value_counts(normalize=True)):.1%}, template base rate {max(pd.Series(tpl_s).value_counts(normalize=True)):.1%}).\n")
# linear probes
X = E[idx]; ok = mk.n.notna().values & (mk.n.values >= 200)
Xa, ya = X[ok], mk.category.values[ok]
Xtr, Xte, ytr, yte = train_test_split(Xa, ya, test_size=0.3, random_state=0, stratify=ya)
clf = LogisticRegression(max_iter=2000, C=1.0).fit(Xtr, ytr)
L.append(f"Linear probe embedding → category on markets with ≥ 200 samples ({ok.sum():,}; 70/30 split): accuracy {clf.score(Xte, yte):.1%} (majority class {pd.Series(yte).value_counts(normalize=True).max():.1%}).\n")
def probe(target, name):
    y = mk[target].values[ok]; good = np.isfinite(y)
    Xg, yg, cg, tg = Xa[good], y[good], mk.category.values[ok][good], mk.template.values[ok][good]
    Xtr, Xte, ytr, yte, ctr, cte, ttr, tte = train_test_split(Xg, yg, cg, tg, test_size=0.3, random_state=0)
    r_emb = Ridge(alpha=10.0).fit(Xtr, ytr).score(Xte, yte)
    cm = pd.Series(ytr).groupby(ctr).mean(); pred_c = pd.Series(cte).map(cm).fillna(ytr.mean()).values
    r_cat = 1 - ((yte - pred_c) ** 2).sum() / ((yte - yte.mean()) ** 2).sum()
    tm = pd.Series(ytr).groupby(ttr).mean(); pred_t = pd.Series(tte).map(tm).fillna(ytr.mean()).values
    r_tpl = 1 - ((yte - pred_t) ** 2).sum() / ((yte - yte.mean()) ** 2).sum()
    Xc = np.hstack([Xtr, pd.get_dummies(ctr).reindex(columns=sorted(set(cg)), fill_value=0).values]); Xce = np.hstack([Xte, pd.get_dummies(cte).reindex(columns=sorted(set(cg)), fill_value=0).values])
    r_both = Ridge(alpha=10.0).fit(Xc, ytr).score(Xce, yte)
    return f"| {name} | {len(yg):,} | {r_emb:.3f} | {r_cat:.3f} | {r_tpl:.3f} | {r_both:.3f} |"
L += ["\n### Does the text predict a market's dynamics? Held-out R² of per-market statistics (markets with ≥ 200 validation samples, 70/30 split)\n",
      "| target | markets | ridge on embedding | category means | template means | embedding + category |", "|---|---|---|---|---|---|",
      probe("p_move", "P(move within 1h)"), probe("p_extreme", "share of extreme-price samples"), probe("mr_hit_moves", "mean-reversion hit rate on moves"), probe("model_hit", "time-first model accuracy")]
log("embedding probes done")
# what the embedding encodes: top PCs vs. category
from sklearn.decomposition import PCA
pca = PCA(n_components=20).fit(Es)
L.append(f"\nPCA of the embeddings of the active markets: the first 5 components hold {pca.explained_variance_ratio_[:5].sum():.1%} of the variance, the first 20 {pca.explained_variance_ratio_.sum():.1%}.\n")
(OUT / "text_audit.md").write_text("\n".join(L) + "\n")
mk[["market_id", "market_idx", "category", "template", "n", "p_move", "p_extreme", "mr_hit_moves", "model_hit"]].to_parquet(OUT / "market_text_stats.parquet")
log("ALL DONE")
