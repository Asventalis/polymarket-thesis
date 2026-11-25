"""news_features.py — news-volume features from the GDELT 2.0 Global Knowledge Graph for every market.

Three stages, each re-runnable:

  terms   every market → the entities named in its question and tags (proper-noun phrases of the
          question, entity-like tags), written to <out>/terms.parquet (market_id, term, source)
  counts  every 15-minute GKG file → articles per term (an article counts once per term it names:
          exact match of a multi-word term on the article's names / persons / organisations /
          locations, token match of a single-word term) and the sum of their tone;
          <out>/counts/day=YYYY-MM-DD.parquet (bucket_ns, term, n, tone_sum) and totals.parquet
  rows    (market_id, t) rows → 12 causal features from the per-market 15-minute series (sum of the
          series of its terms): log1p counts over 1h / 6h / 24h, bursts against the previous
          7 days, tone, share of the news flow, time since the last mention, number of terms;
          the lookup uses the GKG files published at or before t − lag

    ./run_dev.sh news_features.py terms
    ./run_dev.sh news_features.py counts [--workers 48]
    ./run_dev.sh news_features.py rows --rows a.parquet [b.parquet ...] --out-rows x.parquet
    ./run_dev.sh news_features.py rows --tensor-days 6-13   # day=... files at the LOB tensor rows
"""
from __future__ import annotations

import argparse, csv, io, json, re, sys, time, zipfile
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np, pandas as pd

DATA = Path("/data/v.pampinella/agy_project/data")
GKG = DATA / "external/gdelt/gkg"
OUT_DEFAULT = DATA / "derived/news_features"
BUCKET_NS = 15 * 60 * 10**9
T0 = pd.Timestamp("2026-02-27 00:00:00").value           # first downloaded file
T1 = pd.Timestamp("2026-03-27 00:00:00").value           # one past the last
N_BUCKETS = (T1 - T0) // BUCKET_NS
FEATURES = ["news_log_n_1h", "news_log_n_6h", "news_log_n_24h", "news_burst_1h", "news_burst_6h", "news_burst_24h",
            "news_tone_24h", "news_dtone_1h", "news_log_share_24h", "news_log_since_h", "news_log_n_terms", "news_has_terms"]

t0 = time.time()
def log(m): print(f"[{time.time()-t0:7.0f}s] {m}", flush=True)

# ── stage 1: terms ────────────────────────────────────────────────────────────────────────────
GENERIC = set("""will yes no up down over under total totals game games map maps match set sets round rounds series
democratic democrat democrats republican republicans party house senate seat seats president presidential vice
election elections primary primaries nominee nomination governor senator senators congress congressional district
balance power odds rate rates meeting announce announced decrease increase change hike cut cuts
us u.s. usa united states america american national federal state states city county
first second third next last new old best worst most least top bottom
january february march april may june july august september october november december
monday tuesday wednesday thursday friday saturday sunday am pm et utc
fdv cap market markets launch token coin price prices dip hit above below between end start day week month year
q1 q2 q3 q4 h1 h2 ou o/u handicap spread moneyline winner loser points kills inhibitors tower towers dragons barons
team teams player players cup league championship playoffs playoff final finals semifinal quarterfinal
bitcoin ethereum solana xrp bnb dogecoin hype crypto
both exact score scores points rebounds assists goals saves hits strikeouts shots corners cards say says said mention mentions
during before after until within race odds live tv show episode season week""".split())
CONNECT = {"of", "de", "da", "del", "della", "di", "la", "le", "the", "and", "von", "van", "al", "bin", "y", "&"}
TICKERS = {"tsla": "Tesla", "aapl": "Apple", "googl": "Google", "amzn": "Amazon", "nvda": "Nvidia", "msft": "Microsoft",
           "nflx": "Netflix", "pltr": "Palantir", "meta": "Meta", "spx": "S&P 500", "sp-500": "S&P 500", "fed": "Federal Reserve",
           "fed-rates": "Federal Reserve", "googl": "Google", "openai": "OpenAI", "spacex": "SpaceX", "mrbeast": "MrBeast",
           "uk": "United Kingdom", "nyc": "New York", "new-york-city": "New York"}
TAG_STOP = set("""sports games hide-from-new recurring crypto crypto-prices up-or-down esports politics weather elections
finance weekly pop-culture daily equities stocks hit-price geopolitics world global-elections today midterms mention-markets
primaries tech neg-risk economy big-tech tweets-markets house-elections nov-4-elections primary-elections monthly world-elections
us-presidential-election pre-market main-election ai business fdv indicies stock-prices commodities macro-indicators
celebrities strike science economic-policy military-strikes movies music awards climate-science reality-tv yearly
finance-updown ipo ipos congress mma internet space open forex trade gold oil gdp cpi inflation president all derivatives
courts earn-4 highest-temperature daily-temperature multi-strikes daily-close token-launch free-agency international-election-props
regional-spillover foreign-policy box-office governor-midterms rewards conference-tournaments conference-championship
iranian-leadership-regime iran-offensive-actions israel-x-iran trump-iran ukraine-map gov-shutdown app-store top-netflix""".split())
TAG_STOP_RE = re.compile(r"(^rewards|^finance-rewards|\d|-primary$|-primaries$|-midterm$|-midterms$|-elections?$|-league$|-liga$|-cup$|-series$|^ufc-|-tournament$)")

def clean_token(tok: str) -> str:
    return tok.strip("“”\"'’‘(),.:;!?[]{}").replace("’", "'")

def question_terms(q: str) -> list[str]:
    toks = [clean_token(t) for t in q.split()]
    phrases, cur = [], []
    def flush():
        while cur and cur[0].lower() in CONNECT: cur.pop(0)
        while cur and cur[-1].lower() in CONNECT: cur.pop()
        if cur: phrases.append(list(cur))
        cur.clear()
    for t in toks:
        if not t: flush(); continue
        if cur and t.lower() in CONNECT: cur.append(t); continue
        if t[0].isupper() and not any(ch.isdigit() for ch in t) and len(t) > 1 and t.lower() not in GENERIC:
            cur.append(t)
        else:
            flush()                                                    # generic capitalised words break a phrase
    flush()
    terms = []
    for ph in phrases:
        content = [t for t in ph if t.lower() not in CONNECT and t.lower() not in GENERIC]
        if not content: continue
        if len(ph) > 1:
            terms.append(" ".join(ph))                                 # the whole phrase, e.g. "Chris Coons", "Middle East"
        if len(ph) == 1:
            terms.append(ph[0])                                        # single-word entity: "Iran", "Trump", "Chelsea"
        elif any(c.lower() in CONNECT for c in ph) and len(content[-1]) >= 5:
            terms.append(content[-1])                                  # "Bank of England" → "England"
    return terms

def tag_terms(tags: str) -> list[str]:
    out = []
    for tag in (tags or "").split(","):
        tag = tag.strip(); low = tag.lower()
        if not tag or low in TAG_STOP or TAG_STOP_RE.search(low): continue
        if low in TICKERS: out.append(TICKERS[low]); continue
        words = [w for w in low.split("-") if w]
        if any(w in GENERIC for w in words) and len(words) > 1: continue
        if len(words) == 1 and len(words[0]) <= 2: continue
        out.append(" ".join(w.capitalize() if len(w) > 3 else w.upper() for w in words))
    return out

def stage_terms(a):
    u = pd.read_parquet(DATA / "universe_markets.parquet", columns=["market_id", "question", "tags"])
    rows = []
    for m, q, tg in zip(u.market_id.values, u.question.values, u.tags.values):
        seen = set()
        for src, terms in (("question", question_terms(q or "")), ("tag", tag_terms(tg or ""))):
            for t in terms:
                k = t.lower()
                if k in seen or k in GENERIC or len(k) < 3: continue
                seen.add(k); rows.append((m, k, src))
    df = pd.DataFrame(rows, columns=["market_id", "term", "source"])
    a.out.mkdir(parents=True, exist_ok=True); df.to_parquet(a.out / "terms.parquet", index=False)
    n_m = df.market_id.nunique()
    log(f"terms: {len(df):,} (market, term) pairs, {df.term.nunique():,} distinct terms, {n_m:,} of {len(u):,} markets with terms")
    log("most common terms: " + ", ".join(f"{t} ({n})" for t, n in df.term.value_counts().head(40).items()))
    for q in u.question.sample(12, random_state=0): print("   ", q[:80], "→", question_terms(q))

# ── stage 2: counts ───────────────────────────────────────────────────────────────────────────
_VOCAB = None
def _init(multi, single):
    global _VOCAB; _VOCAB = (multi, single)

def count_file(path: str):
    multi, single = _VOCAB
    counts: dict[str, list] = defaultdict(lambda: [0, 0.0]); n_art = 0
    try:
        with zipfile.ZipFile(path) as z:
            raw = z.read(z.namelist()[0]).decode("utf-8", errors="replace")
    except Exception as e:
        return path, None, 0, repr(e)
    for line in raw.split("\n"):
        f = line.split("\t")
        if len(f) < 27: continue
        n_art += 1
        try: tone = float(f[15].split(",")[0])
        except ValueError: tone = 0.0
        # multi-word terms: exact match on any name / person / organisation / location of the article;
        # single-word terms: a whole location or organisation name, the surname of a person, or a whole AllNames entry
        # (never a token of a longer generic phrase such as "Department of Justice" or "New York")
        names = {x.rsplit(",", 1)[0].lower() for x in f[23].split(";") if x}
        persons = {x.lower() for x in f[11].split(";") if x}
        orgs = {x.lower() for x in f[13].split(";") if x}
        locs = set()
        for x in f[10].split(";"):
            p = x.split("#")
            if len(p) > 1 and p[1]: locs.add(p[1].lower())
        strings = names | persons | orgs | locs
        hit = {s for s in strings if s in multi}
        for s in locs | orgs | names:
            if s in single: hit.add(s)
        for s in persons:
            tok = s.rsplit(" ", 1)[-1]
            if tok in single: hit.add(tok)
        for term in hit:
            c = counts[term]; c[0] += 1; c[1] += tone
    return path, {k: tuple(v) for k, v in counts.items()}, n_art, ""

def stage_counts(a):
    terms = pd.read_parquet(a.out / "terms.parquet")
    vocab = set(terms.term.unique())
    multi = frozenset(v for v in vocab if " " in v); single = frozenset(v for v in vocab if " " not in v)
    (a.out / "counts").mkdir(parents=True, exist_ok=True)
    days = pd.date_range(pd.Timestamp(T0), pd.Timestamp(T1) - pd.Timedelta(days=1), freq="D")
    tot_path = a.out / "counts/totals.parquet"
    totals = pd.read_parquet(tot_path) if tot_path.exists() else pd.DataFrame(columns=["bucket_ns", "n_articles", "file"])
    with Pool(a.workers, initializer=_init, initargs=(multi, single)) as pool:
        for d in days:
            out = a.out / f"counts/day={d.strftime('%Y-%m-%d')}.parquet"
            files = [GKG / f"{(d + pd.Timedelta(minutes=15 * i)).strftime('%Y%m%d%H%M%S')}.gkg.csv.zip" for i in range(96)]
            present = [f for f in files if f.exists()]
            if out.exists() and not a.redo: continue
            if len(present) < 96 and not a.partial:
                log(f"{d.date()}: {len(present)}/96 files, skipped (use --partial to write anyway)"); continue
            recs, tots, errs = [], [], 0
            for path, counts, n_art, err in pool.imap_unordered(count_file, [str(f) for f in present]):
                ts = pd.Timestamp(Path(path).name[:14]).value
                if counts is None: errs += 1; log(f"  {Path(path).name}: {err}"); continue
                tots.append((ts, n_art, Path(path).name))
                recs.extend((ts, term, n, s) for term, (n, s) in counts.items())
            df = pd.DataFrame(recs, columns=["bucket_ns", "term", "n", "tone_sum"]).astype({"n": np.int32, "tone_sum": np.float32})
            df.sort_values(["bucket_ns", "term"]).to_parquet(out, index=False)
            totals = pd.concat([totals[~totals.file.isin([t[2] for t in tots])], pd.DataFrame(tots, columns=totals.columns)])
            totals.sort_values("bucket_ns").to_parquet(tot_path, index=False)
            log(f"{d.date()}: {len(present)} files ({errs} unreadable), {sum(t[1] for t in tots):,} articles, {len(df):,} (bucket, term) rows")
    log("counts done")

# ── stage 3: rows ─────────────────────────────────────────────────────────────────────────────
def load_series(a):
    """per-market 15-minute series (articles, tone sum) = sum over its terms; dense (n_markets, N_BUCKETS)."""
    terms = pd.read_parquet(a.out / "terms.parquet", columns=["market_id", "term"])
    files = sorted((a.out / "counts").glob("day=*.parquet"))
    cnt = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    tot = pd.read_parquet(a.out / "counts/totals.parquet")
    log(f"counts: {len(files)} days, {len(cnt):,} rows, {len(tot)} files")
    term_idx = {t: i for i, t in enumerate(sorted(cnt.term.unique()))}
    terms = terms[terms.term.isin(term_idx)]
    markets = sorted(terms.market_id.unique()); m_idx = {m: i for i, m in enumerate(markets)}
    b = ((cnt.bucket_ns.values - T0) // BUCKET_NS).astype(np.int64); ok = (b >= 0) & (b < N_BUCKETS)
    C_n = np.zeros((len(term_idx), N_BUCKETS), np.float32); C_t = np.zeros_like(C_n)
    ti = cnt.term.map(term_idx).values
    np.add.at(C_n, (ti[ok], b[ok]), cnt.n.values[ok]); np.add.at(C_t, (ti[ok], b[ok]), cnt.tone_sum.values[ok])
    import scipy.sparse as sp
    M = sp.csr_matrix((np.ones(len(terms), np.float32), (terms.market_id.map(m_idx).values, terms.term.map(term_idx).values)),
                      shape=(len(markets), len(term_idx)))
    S_n = np.asarray(M @ C_n); S_t = np.asarray(M @ C_t)
    n_terms = np.asarray(M.sum(1)).ravel()
    tb = ((tot.bucket_ns.values - T0) // BUCKET_NS).astype(np.int64); total = np.zeros(N_BUCKETS, np.float32)
    okt = (tb >= 0) & (tb < N_BUCKETS); total[tb[okt]] = tot.n_articles.values[okt]
    have = np.zeros(N_BUCKETS, bool); have[tb[okt]] = True
    log(f"series: {len(markets):,} markets with terms, {len(term_idx):,} terms, buckets with a file {have.sum()}/{N_BUCKETS}")
    z = np.zeros((S_n.shape[0], 1), np.float64)
    cum_n = np.concatenate([z, np.cumsum(S_n, 1, dtype=np.float64)], 1)
    cum_t = np.concatenate([z, np.cumsum(S_t, 1, dtype=np.float64)], 1)
    cum_tot = np.concatenate([[0.0], np.cumsum(total, dtype=np.float64)])
    last_nz = np.maximum.accumulate(np.where(S_n > 0, np.arange(N_BUCKETS, dtype=np.int32)[None, :], -1), axis=1)
    return dict(m_idx=m_idx, n_terms=n_terms, cum_n=cum_n, cum_t=cum_t, cum_tot=cum_tot, last_nz=last_nz)

def row_features(t_ns: np.ndarray, mids: np.ndarray, S: dict, lag_ns: int) -> np.ndarray:
    m_idx, n_terms, cum_n, cum_t, cum_tot, last_nz = (S[k] for k in ("m_idx", "n_terms", "cum_n", "cum_t", "cum_tot", "last_nz"))
    # exclusive end of the cumulative sums: the last bucket whose file time is at or before t - lag
    b_end = np.clip(((t_ns - lag_ns - T0) // BUCKET_NS).astype(np.int64) + 1, 0, N_BUCKETS)
    mi = np.array([m_idx.get(m, -1) for m in mids]); has = mi >= 0; mi_safe = np.where(has, mi, 0)
    def win(cum, k):
        return cum[mi_safe, b_end] - cum[mi_safe, np.clip(b_end - k, 0, None)]
    n1, n6, n24, n7d = win(cum_n, 4), win(cum_n, 24), win(cum_n, 96), win(cum_n, 672)
    t1, t24 = win(cum_t, 4), win(cum_t, 96)
    tot24 = cum_tot[b_end] - cum_tot[np.clip(b_end - 96, 0, None)]
    base1, base6, base24 = n7d / 168.0, n7d / 28.0, n7d / 7.0
    tone24 = np.where(n24 > 0, t24 / np.maximum(n24, 1), 0.0); tone1 = np.where(n1 > 0, t1 / np.maximum(n1, 1), tone24)
    be1 = np.clip(b_end - 1, 0, None); last = last_nz[mi_safe, be1]
    since = np.where((last >= 0) & (b_end > 0), np.minimum((be1 - last) * 0.25, 168.0), 168.0)   # hours, capped at 7 days
    f = np.column_stack([np.log1p(n1), np.log1p(n6), np.log1p(n24),
                         np.log((n1 + 1) / (base1 + 1)), np.log((n6 + 1) / (base6 + 1)), np.log((n24 + 1) / (base24 + 1)),
                         tone24, tone1 - tone24, np.log((n24 + 1) / (tot24 + 1)), np.log1p(since),
                         np.log1p(n_terms[mi_safe]), np.ones(len(t_ns))]).astype(np.float32)
    f[~has] = 0.0
    return f

def stage_rows(a):
    series = load_series(a); lag_ns = int(a.lag_min * 60 * 10**9)
    if a.rows:
        parts = []
        for p in a.rows:
            df = pd.read_parquet(p, columns=["market_id", "t"])
            t_ns = pd.to_datetime(df.t).values.astype("datetime64[ns]").astype("int64")
            f = row_features(t_ns, df.market_id.values, series, lag_ns)
            parts.append(pd.concat([df.reset_index(drop=True), pd.DataFrame(f, columns=FEATURES)], axis=1))
            log(f"{p}: {len(df):,} rows, has_terms {f[:, -1].mean():.3f}, mean log n_24h {f[:, 2].mean():.3f}")
        out = pd.concat(parts, ignore_index=True); out.to_parquet(a.out_rows, index=False); log(f"wrote {a.out_rows}")
    if a.tensor_days:
        lo, hi = (int(x) for x in a.tensor_days.split("-"))
        for d in range(lo, hi + 1):
            day = f"2026-03-{d:02d}"
            src = DATA / f"derived/hand_features/day={day}.parquet"
            df = pd.read_parquet(src, columns=["market_id", "ts"])
            f = row_features(df.ts.values.astype(np.int64), df.market_id.values, series, lag_ns)
            out = pd.concat([df.reset_index(drop=True), pd.DataFrame(f, columns=FEATURES)], axis=1)
            out.to_parquet(a.out / f"day={day}.parquet", index=False)
            log(f"day {day}: {len(df):,} rows, has_terms {f[:, -1].mean():.3f}")
        # robust location / scale of every feature on the rows of the training days (6–12 March), for the NewsIndex
        stat_days = [d for d in range(lo, hi + 1) if 6 <= d <= 12]
        X = np.concatenate([pd.read_parquet(a.out / f"day=2026-03-{d:02d}.parquet", columns=FEATURES).values[::10] for d in stat_days])
        stats = {}
        for i, c in enumerate(FEATURES):
            col = X[:, i].astype(np.float64); loc = float(np.median(col)); q75, q25 = np.percentile(col, [75, 25])
            scale = float((q75 - q25) / 1.349) or float(np.std(col)) or 1.0
            stats[c] = {"loc": loc, "scale": scale}
        (a.out / "meta.json").write_text(json.dumps({"features": FEATURES, "lag_min": a.lag_min, "T0": T0, "bucket_ns": BUCKET_NS,
                                                     "stat_days": stat_days, "stats": stats}, indent=1))
    log("rows done")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["terms", "counts", "rows"])
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--workers", type=int, default=48); ap.add_argument("--partial", action="store_true"); ap.add_argument("--redo", action="store_true")
    ap.add_argument("--rows", nargs="*"); ap.add_argument("--out-rows", type=Path); ap.add_argument("--tensor-days")
    ap.add_argument("--lag-min", type=float, default=10.0)
    a = ap.parse_args()
    {"terms": stage_terms, "counts": stage_counts, "rows": stage_rows}[a.stage](a)
    log("ALL DONE")
