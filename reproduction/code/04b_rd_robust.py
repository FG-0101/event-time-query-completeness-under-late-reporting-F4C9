"""Robustness of the migration estimates to the window width.

Repeats the difference in means and the local linear estimator of code/04_rd.py with windows of 14, 28 and 42 days on
each side of 1 July 2025, for the three pooled series and for the platforms with a material change under either
estimator for any outcome in the main analysis, plus Amazon and Google Shopping. For each window h, placebo cutoffs run from the later of 1 February 2024 and
1 January 2024 plus h to 1 July 2025 minus h, so that every placebo window lies before the migration and within the
data used, and a window is used when it has data on at least 2h minus round(6h/28) days. With h = 28 this reproduces
the placebo set and the rule of code/04_rd.py.
Run from the package root after code/04_rd.py:  python3 code/04b_rd_robust.py
Outputs: results/rd_robust.csv
"""
import json
import numpy as np
import pandas as pd
import duckdb

CUT = pd.Timestamp("2025-07-01")
WINDOWS = [14, 28, 42]
START = pd.Timestamp("2024-01-01")
RD = json.load(open("results/rd.json"))
con = duckdb.connect()
day = con.sql("""SELECT p, a, SUM(n) AS n, SUM(n) FILTER (WHERE d <= 7) AS n7, SUM(n) FILTER (WHERE d <= 30) AS n30
                 FROM read_parquet('data/derived/delays.parquet') WHERE a >= DATE '2024-01-01' AND a <= DATE '2025-08-11'
                 GROUP BY p, a""").df().fillna(0)
day["a"] = pd.to_datetime(day.a)
for o, c in [("w7", "n7"), ("w30", "n30")]:
    day[o] = day[c] / day.n
keep = pd.read_csv("results/rd_platform.csv").p.tolist()


def pooled(frame):
    g = frame.groupby("a")[["n", "n7", "n30"]].sum().reset_index()
    g["w7"], g["w30"] = g.n7 / g.n, g.n30 / g.n
    return g


series = {"All platforms": pooled(day), "Without Google Shopping": pooled(day[day.p != "Google Shopping"]),
          "Platform average": day[day.p.isin(keep)].groupby("a")[["w7", "w30"]].mean().reset_index()}
named = sorted(set(sum([RD[o][k] for o in ["w0", "w7", "w30"] for k in ["down_list", "up_list", "dm_down_list", "dm_up_list"]], [])
                 + ["Amazon", "Google Shopping"]))
for p in named:
    series[p] = day[day.p == p][["a", "w7", "w30"]]


def est(w, cut, h):
    t = (w.a - cut).dt.days.values.astype(float)
    post = (t >= 0).astype(float)
    dow = np.column_stack([(w.a.dt.dayofweek.values == j).astype(float) for j in range(1, 7)])
    X = np.column_stack([np.ones_like(t), post, t, t * post, dow])
    out = {}
    for o in ["w7", "w30"]:
        y = w[o].values.astype(float)
        out[o + "_dm"] = float(y[post == 1].mean() - y[post == 0].mean())
        out[o + "_ll"] = float(np.linalg.lstsq(X, y, rcond=None)[0][1])
    return out


rows = []
for h in WINDOWS:
    fakes = pd.date_range(max(pd.Timestamp("2024-02-01"), START + pd.Timedelta(days=h)), CUT - pd.Timedelta(days=h))
    need = 2 * h - round(6 * h / 28)
    for name, s in series.items():
        s = s.sort_values("a")
        win = lambda c: s[(s.a >= c - pd.Timedelta(days=h)) & (s.a < c + pd.Timedelta(days=h))]
        w = win(CUT)
        if len(w) < need:
            continue
        actual = est(w, CUT, h)
        plc = [est(win(f), f, h) for f in fakes if len(win(f)) >= need]
        P = pd.DataFrame(plc)
        r = {"series": name, "window": h, "n_placebo": len(P), "placebo_first": str(fakes.min().date()), "placebo_last": str(fakes.max().date())}
        for k, v in actual.items():
            r[k] = v
            r[k + "_p"] = float((1 + (P[k].abs() >= abs(v)).sum()) / (1 + len(P)))
        rows.append(r)
R = pd.DataFrame(rows)
R.round(12).to_csv("results/rd_robust.csv", index=False)   # rounding removes last digit floating point noise between runs
print(R.round(3).to_string())
