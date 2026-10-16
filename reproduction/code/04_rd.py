"""Regression discontinuity in time: did reporting delays change at the 1 July 2025 schema migration?

Unit: platform and application day. Outcomes are the shares of a day's decisions whose statements were created on the
same day (w0), within 7 days (w7) and within 30 days (w30). All application days in the window have been observed for
more than 400 days, so the outcomes are not affected by the end of the data.
Two estimators within 28 days on each side: (1) local linear, y = a + b post + c t + d t post + day of week effects,
ordinary least squares with Newey and West standard errors (7 lags); (2) difference in means between the 28 days after
and the 28 days before, which captures shifts that build up after the cutoff rather than appearing as a step. Inference: placebo cutoffs at every day from 1 February 2024 to 3 June 2025
(windows entirely before the migration), permutation p value with one added to numerator and denominator; a platform
jump is material when p < 0.05 and its size is at least five percentage points. Pooled series: all statements of all
platforms, all statements except Google Shopping, and the unweighted mean over platforms.
Run from the package root after code/02_build.py:  python3 code/04_rd.py
Outputs: results/rd_platform.csv, results/rd_pooled.csv, results/rd_series.csv, results/rd_placebo.csv, results/rd.json
"""
import json
import numpy as np
import pandas as pd
import duckdb
import statsmodels.api as sm

CUT = pd.Timestamp("2025-07-01")
H = 28
MIN_N = 1000
LAGS = 7
MATERIAL = 0.05
OUTS = ["w0", "w7", "w30"]
con = duckdb.connect()
day = con.sql("""SELECT p, a, SUM(n) AS n, SUM(n) FILTER (WHERE d = 0) AS n0, SUM(n) FILTER (WHERE d <= 7) AS n7,
                 SUM(n) FILTER (WHERE d <= 30) AS n30
                 FROM read_parquet('data/derived/delays.parquet') WHERE a >= DATE '2024-01-01' AND a <= DATE '2025-07-28'
                 GROUP BY p, a""").df().fillna(0).sort_values(["p", "a"]).reset_index(drop=True)
day["a"] = pd.to_datetime(day.a)
for o, c in [("w0", "n0"), ("w7", "n7"), ("w30", "n30")]:
    day[o] = day[c] / day.n


def series(frame):
    g = frame.groupby("a")[["n", "n0", "n7", "n30"]].sum().reset_index()
    for o, c in [("w0", "n0"), ("w7", "n7"), ("w30", "n30")]:
        g[o] = g[c] / g.n
    return g


def design(dates, cut):
    t = (dates - cut).dt.days.values.astype(float)
    post = (t >= 0).astype(float)
    dow = np.column_stack([(dates.dt.dayofweek.values == j).astype(float) for j in range(1, 7)])
    return np.column_stack([np.ones_like(t), post, t, t * post, dow])


def dm(y, dates, cut):
    """Difference in means after minus before."""
    ok = np.isfinite(y)
    post = (dates >= cut).values
    if ok.sum() < 20:
        return np.nan
    return float(np.nanmean(y[ok & post]) - np.nanmean(y[ok & ~post]))


def rd(y, dates, cut, se=True):
    X = design(dates, cut)
    ok = np.isfinite(y)
    if ok.sum() < 20:
        return (np.nan, np.nan) if se else np.nan
    if not se:
        return float(np.linalg.lstsq(X[ok], y[ok], rcond=None)[0][1])
    m = sm.OLS(y[ok], X[ok]).fit(cov_type="HAC", cov_kwds={"maxlags": LAGS})
    return float(m.params[1]), float(m.bse[1])


win = day[(day.a >= CUT - pd.Timedelta(days=H)) & (day.a < CUT + pd.Timedelta(days=H))]
keep = []
for p, g in win.groupby("p"):
    pre, post = g[g.a < CUT], g[g.a >= CUT]
    if len(g) >= 2 * H - 6 and pre.n.sum() / H >= MIN_N and post.n.sum() / H >= MIN_N:
        keep.append(p)
pooled = {"All platforms": series(day), "Without Google Shopping": series(day[day.p != "Google Shopping"])}
avg = day[day.p.isin(keep)].groupby("a")[OUTS].mean().reset_index()
pooled["Platform average"] = avg

fakes = pd.date_range("2024-02-01", "2025-06-03")
plc = []
for name, ser in list(pooled.items()) + [(p, day[day.p == p]) for p in keep]:
    ser = ser.sort_values("a")
    for fc in fakes:
        w = ser[(ser.a >= fc - pd.Timedelta(days=H)) & (ser.a < fc + pd.Timedelta(days=H))]
        if len(w) < 2 * H - 6:
            continue
        plc.append({"p": name, "cut": fc, **{o: rd(w[o].values.astype(float), w.a, fc, se=False) for o in OUTS},
                    **{f"{o}_dm": dm(w[o].values.astype(float), w.a, fc) for o in OUTS}})
plc = pd.DataFrame(plc)
plc.round(12).to_csv("results/rd_placebo.csv", index=False)


def perm(name, o, b):
    z = plc[plc.p == name][o].dropna().abs()
    return ((1 + (z >= abs(b)).sum()) / (1 + len(z)) if len(z) else np.nan), len(z)


rows = []
for p in keep:
    g = win[win.p == p].sort_values("a")
    r = {"p": p, "n_pre": float(g[g.a < CUT].n.sum() / H), "n_post": float(g[g.a >= CUT].n.sum() / H)}
    for o in OUTS:
        b, se = rd(g[o].values.astype(float), g.a, CUT)
        pv, npl = perm(p, o, b)
        bd = dm(g[o].values.astype(float), g.a, CUT)
        pvd, _ = perm(p, f"{o}_dm", bd)
        r.update({f"{o}_pre": float(g[g.a < CUT][o].mean()), f"{o}_post": float(g[g.a >= CUT][o].mean()), f"{o}_jump": b,
                  f"{o}_se": se, f"{o}_p": pv, f"{o}_nplc": npl, f"{o}_dm": bd, f"{o}_dm_p": pvd})
    rows.append(r)
res = pd.DataFrame(rows).sort_values("n_pre", ascending=False)
res.round(12).to_csv("results/rd_platform.csv", index=False)
prow = []
for name, ser in pooled.items():
    g = ser[(ser.a >= CUT - pd.Timedelta(days=H)) & (ser.a < CUT + pd.Timedelta(days=H))].sort_values("a")
    for o in OUTS:
        b, se = rd(g[o].values.astype(float), g.a, CUT)
        pv, npl = perm(name, o, b)
        bd = dm(g[o].values.astype(float), g.a, CUT)
        pvd, _ = perm(name, f"{o}_dm", bd)
        prow.append({"series": name, "outcome": o, "pre": float(g[g.a < CUT][o].mean()), "post": float(g[g.a >= CUT][o].mean()),
                     "jump": b, "se": se, "p": pv, "n_placebo": npl, "dm": bd, "dm_p": pvd})
prow = pd.DataFrame(prow)
prow.round(12).to_csv("results/rd_pooled.csv", index=False)
ser_out = []
for name, ser in pooled.items():
    z = ser[(ser.a >= CUT - pd.Timedelta(days=H)) & (ser.a < CUT + pd.Timedelta(days=H))].copy()
    z["p"] = name
    ser_out.append(z)
ser_out.append(win[win.p.isin(keep)])
pd.concat(ser_out).round(12).to_csv("results/rd_series.csv", index=False)
S = {"platforms": int(len(res)), "placebo_cutoffs": int(len(fakes)), "window": H, "min_n": MIN_N, "material": MATERIAL, "lags": LAGS,
     "min_days": 2 * H - 6, "placebo_first": str(fakes.min().date()), "placebo_last": str(fakes.max().date()), "alpha": 0.05}
for o in OUTS:
    mat = (res[f"{o}_p"] < 0.05) & (res[f"{o}_jump"].abs() >= MATERIAL)
    S[o] = {"material_up": int((mat & (res[f"{o}_jump"] > 0)).sum()), "material_down": int((mat & (res[f"{o}_jump"] < 0)).sum()),
            "down_list": res[mat & (res[f"{o}_jump"] < 0)].sort_values(f"{o}_jump").p.tolist(),
            "up_list": res[mat & (res[f"{o}_jump"] > 0)].sort_values(f"{o}_jump", ascending=False).p.tolist()}
    md = (res[f"{o}_dm_p"] < 0.05) & (res[f"{o}_dm"].abs() >= MATERIAL)
    S[o].update({"dm_material_up": int((md & (res[f"{o}_dm"] > 0)).sum()), "dm_material_down": int((md & (res[f"{o}_dm"] < 0)).sum()),
                 "dm_down_list": res[md & (res[f"{o}_dm"] < 0)].sort_values(f"{o}_dm").p.tolist(),
                 "dm_up_list": res[md & (res[f"{o}_dm"] > 0)].sort_values(f"{o}_dm", ascending=False).p.tolist()})
json.dump(S, open("results/rd.json", "w"), indent=1)
print(json.dumps(S, indent=1))
print(prow.round(4).to_string())
print(res[["p", "n_pre", "w7_pre", "w7_post", "w7_jump", "w7_p", "w7_dm", "w7_dm_p", "w30_dm", "w30_dm_p"]].round(3).to_string())
