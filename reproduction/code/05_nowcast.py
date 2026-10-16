"""Nowcasting decision date counts under reporting delays, evaluated with rolling origins.

For each evaluation day t (every seven days from 4 March 2024 to 30 March 2026) and each platform with at least one
million statements, the target is the number of decisions applied in the 28 days up to t whose statements arrive
within 180 days of the decision. Only statements created up to t may be used. Methods:
  naive        the count known at t;
  cl28, cl91   chain ladder: development factors f_k = N(a, k+1) / N(a, k), estimated from the application days of
               the last 28 or 91 days that have reached delay k+1 by t, applied to the count known at t; a factor is
               set to one when its denominator has fewer than MIN_DEN statements, and completion factors are capped at
               CAP, because batch reporting makes early counts near zero;
  lag91        completeness curve estimated from the 91 application days that ended 180 days before t, which have
               been fully observed, applied multiplicatively with the same cap;
  bf91         Bornhuetter and Ferguson: the count known at t plus the expected share still missing, one minus the
               completeness of lag91, times the mean daily final count of the same 91 fully observed days.
Log scores are log(max(nowcast, 1)/target). The factor-1.1 accuracy band uses the raw nowcast, so a zero
forecast fails for a positive target. Prediction intervals center on max(nowcast, 1) and use the 10th and 90th percentiles of the same
method's past errors for the same series whose targets were known at t (at least 10 past errors).
Run from the package root after code/02_build.py:  python3 code/05_nowcast.py
Outputs: results/nowcast_errors.csv, results/nowcast_summary.csv, results/nowcast.json
"""
import json
import numpy as np
import pandas as pd
import duckdb

D = 180
WIN = 28
SHORT, LONG = 28, 91          # chain ladder windows and window of fully observed days
MIN_PAST = 10                 # past errors needed for a prediction interval
QLO, QHI = 0.1, 0.9           # quantiles of past errors for the 80 percent interval
FACTOR = 1.1                  # accuracy band: nowcast within this factor of the target
MIN_N = 1_000_000
MIN_DEN = 100
CAP = 20.0
START = pd.Timestamp("2023-09-01")
T_END = pd.Timestamp("2026-09-30")
EVAL = pd.date_range("2024-03-04", "2026-03-30", freq="7D")
con = duckdb.connect()
tot = con.sql(f"SELECT p, SUM(n) AS n FROM read_parquet('data/derived/delays.parquet') GROUP BY p HAVING SUM(n) >= {MIN_N}").df()
plats = tot.sort_values("n", ascending=False).p.tolist()
raw = con.sql(f"""SELECT p, a, LEAST(d, {D + 1}) AS d, SUM(n) AS n FROM read_parquet('data/derived/delays.parquet')
                  WHERE a >= DATE '{START.date()}' GROUP BY ALL""").df()
raw["a"] = pd.to_datetime(raw.a)
days = pd.date_range(START, T_END)
idx = {d: i for i, d in enumerate(days)}


def matrix(frame):
    M = np.zeros((len(days), D + 2))
    ix = frame.a.map(idx).values
    np.add.at(M, (ix, frame.d.values.astype(int)), frame.n.values)
    return M


series = {p: matrix(raw[raw.p == p]) for p in plats}
series["All platforms"] = sum(series[p] for p in plats) + matrix(raw[~raw.p.isin(plats)])
series["Without Google Shopping"] = series["All platforms"] - series["Google Shopping"]
cum = {k: np.cumsum(M[:, :D + 1], axis=1) for k, M in series.items()}   # cum[a, k] = statements within k days


def known(C, a, t):
    k = t - a
    return C[a, min(k, D)] if k >= 0 else 0.0


def chain_ladder(C, t, w):
    """Development factors from application days a with a + k + 1 <= t, a in the last w such days."""
    f = np.ones(D)
    for k in range(D):
        hi = t - k - 1
        lo = max(0, hi - w + 1)
        if hi < 0:
            continue
        num = C[lo:hi + 1, k + 1].sum()
        den = C[lo:hi + 1, k].sum()
        if den >= MIN_DEN and num > 0:
            f[k] = num / den
    # completion factor from delay k to D: product of f_j for j >= k
    comp = np.ones(D + 1)
    for k in range(D - 1, -1, -1):
        comp[k] = comp[k + 1] * f[k]
    return np.minimum(comp, CAP)


def lag_curve(C, t, w):
    """Completeness F(k) and mean daily final count from the w fully observed application days before t - D."""
    hi = t - D
    lo = max(0, hi - w + 1)
    if hi < 0:
        return np.ones(D + 1), 0.0
    full = C[lo:hi + 1, D].sum()
    if full <= 0:
        return np.ones(D + 1), 0.0
    return C[lo:hi + 1, :].sum(axis=0) / full, full / (hi - lo + 1)


rows = []
for name, C in cum.items():
    for te in EVAL:
        t = idx[te]
        a_s = range(t - WIN + 1, t + 1)
        target = sum(C[a, D] for a in a_s)
        if target <= 0:
            continue
        obs = {a: known(C, a, t) for a in a_s}
        cl28, cl91 = chain_ladder(C, t, SHORT), chain_ladder(C, t, LONG)
        F, U = lag_curve(C, t, LONG)
        inv = np.minimum(np.where(F > 0, 1 / np.maximum(F, 1e-12), CAP), CAP)
        est = {"naive": sum(obs.values()),
               "cl28": sum(obs[a] * cl28[t - a] for a in a_s),
               "cl91": sum(obs[a] * cl91[t - a] for a in a_s),
               "lag91": sum(obs[a] * inv[t - a] for a in a_s),
               "bf91": sum(obs[a] + (1 - F[t - a]) * U for a in a_s)}
        for mth, e in est.items():
            rows.append(dict(series=name, t=te, method=mth, target=target, nowcast=e,
                             err=np.log(max(e, 1.0) / target)))
E = pd.DataFrame(rows)
E["post_migration"] = E.t >= pd.Timestamp("2025-07-01")
# prediction intervals from past errors whose targets were known at t
lo_, hi_ = [], []
for (s, m), g in E.groupby(["series", "method"]):
    g = g.sort_values("t")
    for i, r in g.iterrows():
        past = g[g.t <= r.t - pd.Timedelta(days=D)].err
        if len(past) >= MIN_PAST:
            lo_.append((i, past.quantile(QLO)))
            hi_.append((i, past.quantile(QHI)))
E["q10"] = pd.Series(dict(lo_))
E["q90"] = pd.Series(dict(hi_))
E["covered"] = np.where(E.q10.notna(), (E.err >= E.q10) & (E.err <= E.q90), np.nan)
E.to_csv("results/nowcast_errors.csv", index=False)


def summ(g):
    a = g.err.abs()
    return pd.Series({"n": len(g), "median_abs": a.median(), "within10": ((g.nowcast >= g.target / FACTOR) & (g.nowcast <= g.target * FACTOR)).mean(), "bias": g.err.median(),
                      "p90_abs": a.quantile(0.9), "coverage": g.covered.dropna().mean() if g.covered.notna().any() else np.nan})


S = E.groupby(["series", "method", "post_migration"]).apply(summ, include_groups=False).reset_index()
S.to_csv("results/nowcast_summary.csv", index=False)
P = E[~E.series.isin(["All platforms", "Without Google Shopping"])]
PS = P.groupby(["method", "post_migration"]).apply(summ, include_groups=False).reset_index()
# platform level: share of platforms where the best method beats naive
best = P.groupby(["series", "method"]).err.apply(lambda e: e.abs().median()).unstack()
out = {"eval_days": int(len(EVAL)), "platforms": int(len(plats)), "window": WIN, "horizon": D, "min_den": MIN_DEN, "cap": CAP,
       "min_n": MIN_N, "short": SHORT, "long": LONG, "min_past": MIN_PAST, "qlo": QLO, "qhi": QHI, "factor": FACTOR, "eval_first": str(EVAL.min().date()), "eval_last": str(EVAL.max().date()),
       "platform_pooled": PS.to_dict(orient="records"),
       "agg": S[S.series.isin(["All platforms", "Without Google Shopping"])].to_dict(orient="records"),
       "cl28_beats_naive_platforms": int((best.cl28 < best.naive).sum()),
       "cl91_beats_naive_platforms": int((best.cl91 < best.naive).sum()),
       "lag91_beats_naive_platforms": int((best.lag91 < best.naive).sum()),
       "bf91_beats_naive_platforms": int((best.bf91 < best.naive).sum())}
worst = P[P.method == "cl28"].assign(a=lambda x: x.err.abs()).sort_values("a", ascending=False).head(10)
out["worst_cl28"] = worst[["series", "t", "err"]].astype(str).to_dict(orient="records")
json.dump(out, open("results/nowcast.json", "w"), indent=1, default=str)
print(PS.round(3).to_string())
print(S[S.series.isin(["All platforms", "Without Google Shopping"])].round(3).to_string())
print({k: v for k, v in out.items() if "beats" in k})
print(worst.to_string())
