"""Delay distributions, vintages of decision date counts, and submission spikes.

1. Delay distribution overall, by platform (platforms with at least one million statements), by source and automation.
2. Vintages. For each application month m from October 2023 (the first full month after the database opened on
   25 September 2023) to March 2026, the number of statements known at the end
   of the month plus k days (k = 0, 7, 30, 90, 180) relative to the number known on 30 September 2026, overall and by
   platform. Months after March 2026 are excluded because fewer than 180 days of reporting have been observed.
3. Submission spikes. Days on which a platform created at least ten times its median daily number of statements (the
   median over days with at least one statement) and at least one million statements, the share of those statements that concern decisions more than 30 days old, and how far back they reach.
Run from the package root after code/02_build.py:  python3 code/03_revisions.py
Outputs: results/delay_overall.csv, results/delay_platform.csv, results/delay_type.csv, results/vintage_total.csv,
results/vintage_platform.csv, results/spikes.csv, results/revisions.json
"""
import json
import numpy as np
import pandas as pd
import duckdb

T = pd.Timestamp("2026-09-30")
KS = [0, 7, 30, 90, 180]
MIN_N = 1_000_000
con = duckdb.connect()
con.sql("CREATE VIEW D AS SELECT * FROM read_parquet('data/derived/delays.parquet')")
con.sql("CREATE VIEW DT AS SELECT * FROM read_parquet('data/derived/delays_by_type.parquet')")
out = {}

# 1. delay distributions
ov = con.sql("SELECT d, SUM(n) AS n FROM D GROUP BY d ORDER BY d").df()
ov["cum"] = ov.n.cumsum() / ov.n.sum()
ov.to_csv("results/delay_overall.csv", index=False)
tot = ov.n.sum()
for k in [0, 1, 7, 30, 90, 365]:
    out[f"share_within_{k}"] = float(ov[ov.d <= k].n.sum() / tot)
gs = con.sql("SELECT d, SUM(n) AS n FROM D WHERE p <> 'Google Shopping' GROUP BY d").df()
for k in [0, 7, 30]:
    out[f"share_within_{k}_exgs"] = float(gs[gs.d <= k].n.sum() / gs.n.sum())
pl = con.sql(f"""SELECT p, SUM(n) AS n, SUM(n) FILTER (WHERE d = 0) / SUM(n) AS d0, SUM(n) FILTER (WHERE d <= 7) / SUM(n) AS d7,
    SUM(n) FILTER (WHERE d <= 30) / SUM(n) AS d30, SUM(n) FILTER (WHERE d > 90) / SUM(n) AS gt90,
    SUM(n * d) / SUM(n) AS mean_d FROM D GROUP BY p HAVING SUM(n) >= {MIN_N} ORDER BY n DESC""").df().fillna(0)
med = []
for p in pl.p:
    z = con.sql(f"SELECT d, SUM(n) AS n FROM D WHERE p = $p GROUP BY d ORDER BY d", params={"p": p}).df()
    c = z.n.cumsum() / z.n.sum()
    med.append(dict(p=p, median_d=int(z.d[c >= 0.5].iloc[0]), p90_d=int(z.d[c >= 0.9].iloc[0]), p99_d=int(z.d[c >= 0.99].iloc[0])))
pl = pl.merge(pd.DataFrame(med), on="p")
pl.to_csv("results/delay_platform.csv", index=False)
out.update({"platforms_big": int(len(pl)), "platforms_d7_below_half": int((pl.d7 < 0.5).sum()),
            "platforms_p90_above_30": int((pl.p90_d > 30).sum()), "platforms_d0_above_90": int((pl.d0 > 0.9).sum())})
ty = con.sql("""SELECT s, ad, SUM(n) AS n, SUM(n) FILTER (WHERE d = 0) / SUM(n) AS d0, SUM(n) FILTER (WHERE d <= 7) / SUM(n) AS d7,
    SUM(n) FILTER (WHERE d <= 30) / SUM(n) AS d30 FROM DT WHERE p <> 'Google Shopping' GROUP BY s, ad ORDER BY n DESC""").df().fillna(0)
ty.to_csv("results/delay_type.csv", index=False)

# 2. vintages by application month
v = con.sql("""SELECT p, date_trunc('month', a) AS m, a, d, n FROM D""").df()
v["m"] = pd.to_datetime(v.m)
v["a"] = pd.to_datetime(v.a)
v["c"] = v.a + pd.to_timedelta(v.d, unit="D")
v["mend"] = v.m + pd.offsets.MonthEnd(0)
v = v[(v.m >= pd.Timestamp("2023-10-01")) & (v.m <= pd.Timestamp("2026-03-01"))]
for k in KS:
    v[f"k{k}"] = np.where(v.c <= v.mend + pd.Timedelta(days=k), v.n, 0.0)


def vint(g):
    f = g.n.sum()
    return pd.Series({"final": f, **{f"r{k}": g[f"k{k}"].sum() / f for k in KS}})


vt = v.groupby("m").apply(vint, include_groups=False).reset_index()
vt.to_csv("results/vintage_total.csv", index=False)
vx = v[v.p != "Google Shopping"].groupby("m").apply(vint, include_groups=False).reset_index()
vx.to_csv("results/vintage_total_exgs.csv", index=False)
big = set(pl.p)
vp = v[v.p.isin(big)].groupby(["p", "m"]).apply(vint, include_groups=False).reset_index()
vp = vp[vp.final >= 10000]
vp.to_csv("results/vintage_platform.csv", index=False)
for k in KS:
    out[f"vint_total_r{k}_min"] = float(vt[f"r{k}"].min())
    out[f"vint_total_r{k}_median"] = float(vt[f"r{k}"].median())
    out[f"vint_exgs_r{k}_min"] = float(vx[f"r{k}"].min())
    out[f"vint_exgs_r{k}_median"] = float(vx[f"r{k}"].median())
    out[f"vint_pm_r{k}_below_90"] = float((vp[f"r{k}"] < 0.9).mean())
    out[f"vint_pm_r{k}_below_50"] = float((vp[f"r{k}"] < 0.5).mean())
out["vint_platform_months"] = int(len(vp))
out["vint_min_final"] = 10000
out["spike_ratio"] = 10
out["spike_min"] = 1_000_000
out["vint_months"] = int(len(vt))
pre = vx[vx.m < pd.Timestamp("2025-07-01")]
post = vx[vx.m >= pd.Timestamp("2025-07-01")]
for k in [0, 30]:
    out[f"vint_exgs_r{k}_pre_median"] = float(pre[f"r{k}"].median())
    out[f"vint_exgs_r{k}_post_median"] = float(post[f"r{k}"].median())
    out[f"vint_exgs_r{k}_post_min"] = float(post[f"r{k}"].min())

# 3. submission spikes
cd = con.sql("""SELECT p, a + CAST(d AS INTEGER) AS c, SUM(n) AS n, SUM(n) FILTER (WHERE d > 30) AS late,
    MIN(a) AS amin FROM D GROUP BY p, a + CAST(d AS INTEGER)""").df()
cd["late"] = cd.late.fillna(0)
medc = cd.groupby("p").n.median().rename("med")
cd = cd.merge(medc, on="p")
sp = cd[(cd.n >= 10 * cd.med) & (cd.n >= 1_000_000)].copy()
sp["late_share"] = sp.late / sp.n
sp["reach_days"] = (pd.to_datetime(sp.c) - pd.to_datetime(sp.amin)).dt.days
sp = sp.sort_values("n", ascending=False)
sp.to_csv("results/spikes.csv", index=False)
out.update({"spike_days": int(len(sp)), "spike_platforms": int(sp.p.nunique()), "spike_statements": float(sp.n.sum()),
            "spike_late_share": float(sp.late.sum() / sp.n.sum()), "spike_max_n": float(sp.n.max()) if len(sp) else 0.0,
            "spike_max_p": str(sp.p.iloc[0]) if len(sp) else "", "spike_max_c": str(pd.Timestamp(sp.c.iloc[0]).date()) if len(sp) else "",
            "spike_max_ratio": float(sp.n.iloc[0] / sp.med.iloc[0]) if len(sp) else 0.0, "spike_max_reach": int(sp.reach_days.iloc[0]) if len(sp) else 0})
json.dump(out, open("results/revisions.json", "w"), indent=1)
print(json.dumps(out, indent=1))
print(pl.head(25).round(3).to_string())
print(vx.round(3).to_string())
print(sp.head(20).to_string())
