"""Tables, figures and macros for the manuscript. Every number in the text comes from here.
Run from the package root after code/02 to code/05:  python3 code/06_report.py
Outputs: tex/macros.tex, tex/tab_*.tex, figures/fig_*.pdf
"""
from paper_style import save_figure
import glob
import json
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

plt.rcParams.update({"font.family": "serif", "font.serif": ["TeX Gyre Termes", "Liberation Serif", "DejaVu Serif"], "font.size": 8,
                     "axes.linewidth": 0.6, "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
J = lambda p: json.load(open(p))
Q, R, RD, NC = J("results/quality.json"), J("results/revisions.json"), J("results/rd.json"), J("results/nowcast.json")
dp = pd.read_csv("results/delay_platform.csv")
dt = pd.read_csv("results/delay_type.csv")
vt = pd.read_csv("results/vintage_total.csv", parse_dates=["m"])
vx = pd.read_csv("results/vintage_total_exgs.csv", parse_dates=["m"])
vp = pd.read_csv("results/vintage_platform.csv", parse_dates=["m"])
sp = pd.read_csv("results/spikes.csv", parse_dates=["c", "amin"])
rpl = pd.read_csv("results/rd_platform.csv")
rpo = pd.read_csv("results/rd_pooled.csv")
rse = pd.read_csv("results/rd_series.csv", parse_dates=["a"])
ne = pd.read_csv("results/nowcast_errors.csv", parse_dates=["t"])
ns = pd.read_csv("results/nowcast_summary.csv")
ov = pd.read_csv("results/delay_overall.csv")
CUT = pd.Timestamp("2025-07-01")
m = {}
c = lambda x: f"{int(round(x)):,}".replace(",", "{,}")
pct = lambda x, d=0: f"{0:.{d}f}" if round(100 * x, d) == 0 else f"{100 * x:.{d}f}"
bn = lambda x: f"{x / 1e9:.1f}"
mn = lambda x: f"{x / 1e6:.1f}"
SHORT = {"willhaben internet service GmbH & Co KG": "willhaben", "Discord Netherlands B.V.": "Discord", "Quora Ireland Limited": "Quora",
         'SIA "JOOM"': "Joom", "Meetic SAS": "Meetic", "Vinted UAB": "Vinted", "www.gutefrage.net": "gutefrage.net",
         "Kaufland Online-Marktplatz": "Kaufland", "Hotel Hideaway": "Hotel Hideaway"}
short = lambda x: SHORT.get(x, x).replace("&", "\\&")
nz = lambda x: 0.0 if (x is None or (isinstance(x, float) and np.isnan(x))) else x

# data and quality
m.update({"nTotal": c(Q["total"]), "nTotalBn": bn(Q["total"]), "nValid": c(Q["valid"]), "nValidBn": bn(Q["valid"]),
          "nNeg": c(nz(Q["created_before_applied"])), "nPreStart": c(nz(Q["applied_before_start"])),
          "nAboveMax": c(nz(Q["delay_above_dmax"])), "nPlatforms": str(int(Q["platforms"])), "dMax": str(Q["dmax"]),
          "nMonths": str(len(glob.glob("data/derived/tri_*.parquet")))})
# delays
for k in [0, 1, 7, 30, 90, 365]:
    m[f"within{['Zero', 'One', 'Seven', 'Thirty', 'Ninety', 'Year'][[0, 1, 7, 30, 90, 365].index(k)]}"] = pct(R[f"share_within_{k}"], 2 if k == 365 else 1)
m.update({"withinZeroExgs": pct(R["share_within_0_exgs"], 1), "withinSevenExgs": pct(R["share_within_7_exgs"], 1),
          "withinThirtyExgs": pct(R["share_within_30_exgs"], 1), "bigPlatforms": str(R["platforms_big"]),
          "bigDsevenHalf": str(R["platforms_d7_below_half"]), "bigPninetyThirty": str(R["platforms_p90_above_30"]),
          "bigSameDayNinety": str(R["platforms_d0_above_90"]), "lateYearN": c(nz(Q["delay_above_dmax"]))})
gsn = dp.set_index("p").loc["Google Shopping", "n"]
m["gsShare"] = pct(gsn / Q["valid"])
own = dt[dt.s == "own initiative"]
note = dt[dt.s == "notice"]
m["ownSeven"] = pct(own.eval("n * d7").sum() / own.n.sum(), 1)
m["noticeSeven"] = pct(note.eval("n * d7").sum() / note.n.sum(), 1)
m["ownZero"] = pct(own.eval("n * d0").sum() / own.n.sum(), 1)
m["noticeZero"] = pct(note.eval("n * d0").sum() / note.n.sum(), 1)
top = dp.head(20)
with open("tex/tab_platforms.tex", "w") as o:
    o.write("\\begin{tabular}{lrrrrrrr}\\toprule\nPlatform & Statements (million) & Same day & Within 7 days & Within 30 days & Median & 90th pct. & 99th pct.\\\\\n")
    o.write(" & & (\\%) & (\\%) & (\\%) & (days) & (days) & (days)\\\\\\midrule\n")
    for _, r in top.iterrows():
        o.write(f"{short(r.p)} & {mn(r.n)} & {pct(r.d0, 1)} & {pct(r.d7, 1)} & {pct(r.d30, 1)} & {int(r.median_d)} & {int(r.p90_d)} & {int(r.p99_d)}\\\\\n")
    o.write("\\bottomrule\\end{tabular}\n")
m["topPlatforms"] = str(len(top))
amz = dp.set_index("p").loc["Amazon"]
m.update({"amzDseven": pct(amz.d7, 1), "amzPninety": str(int(amz.p90_d)), "amzMedian": str(int(amz.median_d))})
# vintages
for k in [0, 7, 30, 90, 180]:
    w = ["Zero", "Seven", "Thirty", "Ninety", "OneEighty"][[0, 7, 30, 90, 180].index(k)]
    m[f"vint{w}Med"] = pct(R[f"vint_total_r{k}_median"], 1)
    m[f"vint{w}MedEx"] = pct(R[f"vint_exgs_r{k}_median"], 1)
    m[f"vint{w}MinEx"] = pct(R[f"vint_exgs_r{k}_min"], 1)
    m[f"vint{w}BelowNinety"] = pct(R[f"vint_pm_r{k}_below_90"])
    m[f"vint{w}BelowHalf"] = pct(R[f"vint_pm_r{k}_below_50"])
m.update({"vintPlatformMonths": c(R["vint_platform_months"]), "vintMonths": str(R["vint_months"]),
          "vintZeroPreEx": pct(R["vint_exgs_r0_pre_median"], 1), "vintZeroPostEx": pct(R["vint_exgs_r0_post_median"], 1),
          "vintZeroPostMinEx": pct(R["vint_exgs_r0_post_min"], 1), "vintThirtyPreEx": pct(R["vint_exgs_r30_pre_median"], 1),
          "vintThirtyPostEx": pct(R["vint_exgs_r30_post_median"], 1), "vintThirtyPostMinEx": pct(R["vint_exgs_r30_post_min"], 1)})
worst_m = vx.sort_values("r0").iloc[0]
m["vintWorstMonth"] = worst_m.m.strftime("%B %Y")
m["vintWorstZero"] = pct(worst_m.r0, 1)
m["vintWorstThirty"] = pct(worst_m.r30, 1)
with open("tex/tab_vintage.tex", "w") as o:
    o.write("\\begin{tabular}{lrrrrr}\\toprule\nShare of the count of 30 September 2026 known & End of month & +7 days & +30 days & +90 days & +180 days\\\\\\midrule\n")
    for lab, src, fn in [("All platforms, median month", "vint_total", "median"), ("Without Google Shopping, median month", "vint_exgs", "median"),
                         ("Without Google Shopping, lowest month in each column", "vint_exgs", "min")]:
        o.write(lab + " & " + " & ".join(pct(R[f"{src}_r{k}_{fn}"], 1) for k in [0, 7, 30, 90, 180]) + "\\\\\n")
    o.write("Platform months below 90\\,\\% & " + " & ".join(pct(R[f"vint_pm_r{k}_below_90"], 1) for k in [0, 7, 30, 90, 180]) + "\\\\\n")
    o.write("Platform months below 50\\,\\% & " + " & ".join(pct(R[f"vint_pm_r{k}_below_50"], 1) for k in [0, 7, 30, 90, 180]) + "\\\\\n")
    o.write("\\bottomrule\\end{tabular}\n")
# spikes
m.update({"spikeDays": str(R["spike_days"]), "spikePlatforms": str(R["spike_platforms"]), "spikeStatementsBn": bn(R["spike_statements"]),
          "spikeLateShare": pct(R["spike_late_share"]), "spikeMaxN": mn(R["spike_max_n"]), "spikeMaxP": short(R["spike_max_p"]),
          "spikeMaxDate": pd.Timestamp(R["spike_max_c"]).strftime("%-d %B %Y").lstrip("0"), "spikeMaxRatio": f"{R['spike_max_ratio']:.0f}"})
amz_sp = sp[sp.p == "Amazon"]
m["amzSpikeDays"] = str(len(amz_sp))
m["amzSpikeMn"] = mn(amz_sp.n.sum())
big_late = amz_sp.sort_values("late", ascending=False).iloc[0]
m["amzBigLateDate"] = f"{big_late.c.day} {big_late.c:%B %Y}"
m["amzBigLateMn"] = mn(big_late.late)
m["amzBigLateFrom"] = f"{big_late.amin.day} {big_late.amin:%B %Y}"
with open("tex/tab_spikes.tex", "w") as o:
    o.write("\\begin{tabular}{llrrrr}\\toprule\nPlatform & Creation day & Statements (million) & Ratio to median day & Older than 30 days (\\%) & Oldest decision (days)\\\\\\midrule\n")
    shown = sp.sort_values("n", ascending=False).groupby("p").head(2).head(12)
    for _, r in shown.iterrows():
        o.write(f"{short(r.p)} & {r.c.day} {r.c:%b %Y} & {mn(r.n)} & {r.n / r.med:.0f} & {pct(r.late_share, 1)} & {int(r.reach_days)}\\\\\n")
    o.write("\\bottomrule\\end{tabular}\n")
# Kaufland
kau = dp.set_index("p").loc["Kaufland Online-Marktplatz"]
m["kauMn"] = mn(kau.n)
# migration: pooled
for srs, tag in [("All platforms", "All"), ("Without Google Shopping", "Ex"), ("Platform average", "Avg")]:
    for o_ in ["w0", "w7", "w30"]:
        r = rpo[(rpo.series == srs) & (rpo.outcome == o_)].iloc[0]
        t2 = {"w0": "Zero", "w7": "Seven", "w30": "Thirty"}[o_]
        m[f"rd{tag}{t2}Pre"] = pct(r.pre, 1)
        m[f"rd{tag}{t2}Post"] = pct(r.post, 1)
        m[f"rd{tag}{t2}Jump"] = pct(r.jump, 1)
        m[f"rd{tag}{t2}P"] = f"{r.p:.3f}"
        m[f"rd{tag}{t2}Dm"] = pct(r.dm, 1)
        m[f"rd{tag}{t2}DmP"] = f"{r.dm_p:.3f}"
m.update({"rdPlatforms": str(RD["platforms"]), "rdPlacebo": str(RD["placebo_cutoffs"]), "rdWindow": str(RD["window"]),
          "rdLags": str(RD["lags"]), "rdMaterial": pct(RD["material"]), "rdPmin": f"{1 / (1 + RD['placebo_cutoffs']):.3f}"})
for o_, t2 in [("w7", "Seven"), ("w30", "Thirty")]:
    m[f"dm{t2}Down"] = str(RD[o_]["dm_material_down"])
    m[f"dm{t2}Up"] = str(RD[o_]["dm_material_up"])
    m[f"dm{t2}DownList"] = ", ".join(short(x) for x in RD[o_]["dm_down_list"])
    m[f"dm{t2}UpList"] = ", ".join(short(x) for x in RD[o_]["dm_up_list"])
    m[f"ll{t2}Down"] = str(RD[o_]["material_down"])
    m[f"ll{t2}Up"] = str(RD[o_]["material_up"])
tt = rpl.set_index("p").loc["TikTok"]
m.update({"ttSevenPre": pct(tt.w7_pre, 1), "ttSevenPost": pct(tt.w7_post, 1), "ttSevenDm": pct(tt.w7_dm, 1), "ttSevenDmP": f"{tt.w7_dm_p:.3f}"})
am = rpl.set_index("p").loc["Amazon"]
m.update({"amzSevenPre": pct(am.w7_pre, 1), "amzSevenPost": pct(am.w7_post, 1), "amzSevenDmP": f"{am.w7_dm_p:.3f}"})
gr = rpl.set_index("p").loc["Google Shopping"]
m.update({"gsSevenPre": pct(gr.w7_pre, 1), "gsSevenPost": pct(gr.w7_post, 1), "gsSevenDmP": f"{gr.w7_dm_p:.3f}"})
# claims in the text that depend on these values
pmin = 1 / (1 + RD["placebo_cutoffs"])
ex7 = rpo[(rpo.series == "Without Google Shopping") & (rpo.outcome == "w7")].iloc[0]
assert abs(ex7.dm_p - pmin) < 1e-9 and abs(tt.w7_dm_p - pmin) < 1e-9          # no placebo change as large
assert (rpo[rpo.series == "All platforms"][["p", "dm_p"]] > 0.05).all().all()      # pooled series with Google Shopping
assert gr.w7_dm_p > 0.05 and am.w7_dm_p > 0.05
dms = rpl.set_index("p").loc[RD["w7"]["dm_down_list"]]
m["dmSevenDownPmax"] = f"{dms.w7_dm_p.max():.3f}"
assert dp.iloc[1].p == "TikTok"                                                   # second largest submitter
m["xSevenPost"] = pct(rpl.set_index("p").loc["X"].w7_post, 1)
m["sheinSevenPost"] = pct(rpl.set_index("p").loc["Shein"].w7_post, 1)
with open("tex/tab_rd.tex", "w") as o:
    o.write("\\begin{tabular}{llrrrrrrr}\\toprule\nSeries & Outcome & Before & After & Local linear jump & Standard error & $p$ & Difference in means & $p$\\\\\\midrule\n")
    lab = {"w0": "Same day", "w7": "Within 7 days", "w30": "Within 30 days"}
    for srs in ["All platforms", "Without Google Shopping", "Platform average"]:
        for o_ in ["w0", "w7", "w30"]:
            r = rpo[(rpo.series == srs) & (rpo.outcome == o_)].iloc[0]
            o.write(f"{srs if o_ == 'w0' else ''} & {lab[o_]} & {r.pre:.3f} & {r.post:.3f} & {r.jump:.3f} & {r.se:.3f} & {r.p:.3f} & {r.dm:.3f} & {r.dm_p:.3f}\\\\\n")
    o.write("\\bottomrule\\end{tabular}\n")
# robustness of the migration estimates to the window width
rb = pd.read_csv("results/rd_robust.csv").set_index(["series", "window"])
f3 = lambda x: "0.000" if round(x, 3) == 0 else re.sub(r"^-", "$-$", f"{x:.3f}")
with open("tex/tab_robust.tex", "w") as o:
    o.write("\\begin{tabular}{llrrr}\\toprule\nSeries & Outcome & 14 days & 28 days & 42 days\\\\\\midrule\n")
    for srs in ["Without Google Shopping", "Platform average", "TikTok", "X", "Shein", "leboncoin", "Pinterest", "Amazon", "SoundCloud"]:
        for o_, lab in [("w7", "Within 7 days"), ("w30", "Within 30 days")]:
            cells = [f"{f3(rb.loc[(srs, h), o_ + '_dm'])} ({rb.loc[(srs, h), o_ + '_dm_p']:.3f})" for h in [14, 28, 42]]
            o.write(f"{srs if o_ == 'w7' else ''} & {lab} & " + " & ".join(cells) + "\\\\\n")
    o.write("\\bottomrule\\end{tabular}\n")
for srs, tg in [("Without Google Shopping", "Ex"), ("Platform average", "Avg"), ("TikTok", "Tt"), ("SoundCloud", "Sc"), ("Amazon", "Amz"), ("leboncoin", "Leb")]:
    for h, hw in [(14, "Short"), (28, "Main"), (42, "Long")]:
        for o_, ow in [("w7", "Seven"), ("w30", "Thirty")]:
            m[f"rob{tg}{ow}{hw}"] = pct(rb.loc[(srs, h), o_ + "_dm"], 1)
            m[f"rob{tg}{ow}{hw}P"] = f"{rb.loc[(srs, h), o_ + '_dm_p']:.3f}"
            m[f"rob{tg}{ow}{hw}LlP"] = f"{rb.loc[(srs, h), o_ + '_ll_p']:.3f}"
            m[f"rob{tg}{ow}{hw}Ll"] = pct(rb.loc[(srs, h), o_ + "_ll"], 1)
ws_ = sorted(set(pd.read_csv("results/rd_robust.csv").window))
assert ws_ == [14, RD["window"], 42]
m["robShortW"], m["robLongW"] = str(ws_[0]), str(ws_[2])
m["rdAlpha"] = f"{RD['alpha']:.2f}"
assert (rb.loc["Pinterest"][["w7_dm", "w30_dm", "w7_ll", "w30_ll"]].abs() < RD["material"]).all().all()
m["robPlcShort"] = str(int(rb.loc[("Without Google Shopping", 14), "n_placebo"]))
m["robPlcLong"] = str(int(rb.loc[("Without Google Shopping", 42), "n_placebo"]))
assert all((rb.loc[(p, h), "w7_dm_p"] < 0.05) for p in ["X", "Shein"] for h in [14, 28, 42])
# migration episode statistics from the delay table
import duckdb
con = duckdb.connect()
ep = con.sql("""SELECT p, SUM(n) FILTER (WHERE d > 30) AS late, SUM(n) AS n FROM read_parquet('data/derived/delays.parquet')
                WHERE a >= DATE '2025-07-01' AND a <= DATE '2026-01-31' GROUP BY p ORDER BY late DESC NULLS LAST LIMIT 5""").df()
pre_ep = con.sql("""SELECT p, SUM(n) FILTER (WHERE d > 30) AS late, SUM(n) AS n FROM read_parquet('data/derived/delays.parquet')
                WHERE a >= DATE '2024-12-01' AND a <= DATE '2025-06-30' GROUP BY p""").df().set_index("p")
for nm, tag in [("TikTok", "Tt"), ("Amazon", "Amz")]:
    r = ep.set_index("p").loc[nm]
    m[f"ep{tag}LateMn"] = mn(r.late)
    m[f"ep{tag}LateShare"] = pct(r.late / r.n)
    m[f"ep{tag}LateSharePre"] = pct(nz(pre_ep.loc[nm, "late"]) / pre_ep.loc[nm, "n"], 1)
wk = con.sql("""SELECT date_trunc('week', a) AS wk, p, SUM(n) AS n, SUM(n * d) / SUM(n) AS mean_d, SUM(n) FILTER (WHERE d <= 7) / SUM(n) AS w7
                FROM read_parquet('data/derived/delays.parquet') WHERE p IN ('TikTok', 'Amazon') AND a >= DATE '2025-01-06' AND a <= DATE '2026-03-29'
                GROUP BY 1, 2 ORDER BY 1""").df()
wk["wk"] = pd.to_datetime(wk.wk)
ttw = wk[wk.p == "TikTok"].set_index("wk")
m["ttMeanDelayMax"] = f"{ttw.mean_d.max():.0f}"
m["ttMeanDelayMaxWeek"] = f"{ttw.mean_d.idxmax().day} {ttw.mean_d.idxmax():%B %Y}"
back = ttw[(ttw.index > pd.Timestamp("2025-12-01")) & (ttw.mean_d < 1)]
m["ttBackToNormal"] = f"{back.index.min().day} {back.index.min():%B %Y}" if len(back) else "not within the data"
assert (ttw[ttw.index >= back.index.min()].mean_d < 1).all()                       # stays below one day to March 2026
pre_w = ttw[ttw.index + pd.Timedelta(days=6) < CUT]
m["ttPreMaxDelay"] = f"{pre_w.mean_d.max():.1f}"
m["ttPreFirstWeek"] = f"{pre_w.index.min().day} {pre_w.index.min():%B %Y}"
epi = ttw[(ttw.index >= pd.Timestamp("2025-06-30")) & (ttw.index < back.index.min())]
m["ttEpiWeeks"] = str(len(epi))
m["ttEpiAboveTen"] = str(int((epi.mean_d > 10).sum()))
m["ttEpiBelowOne"] = str(int((epi.mean_d < 1).sum()))
# Google Shopping delays
gsd = con.sql("""SELECT SUM(n) FILTER (WHERE d = 0) / SUM(n) AS d0, SUM(n) FILTER (WHERE d <= 1) / SUM(n) AS d1
                 FROM read_parquet('data/derived/delays.parquet') WHERE p = 'Google Shopping'""").df().iloc[0]
m.update({"gsZero": pct(gsd.d0, 1), "gsOne": pct(gsd.d1, 1)})
# Kaufland: creation days and application months
kc = con.sql("""SELECT a + CAST(d AS INTEGER) AS c, SUM(n) AS n FROM read_parquet('data/derived/delays.parquet')
                WHERE p = 'Kaufland Online-Marktplatz' GROUP BY 1""").df()
kc["c"] = pd.to_datetime(kc.c)
m["kauLastTwo"] = pct(kc[kc.c >= pd.Timestamp("2026-09-29")].n.sum() / kc.n.sum(), 1)
assert len(kc) == 2 and "Kaufland Online-Marktplatz" not in set(sp.p)                # two creation days, no spike
ka = con.sql("""SELECT MIN(a) AS amin, SUM(n) FILTER (WHERE a >= DATE '2026-09-29') / SUM(n) AS two,
                SUM(n) FILTER (WHERE d = 0) / SUM(n) AS same, SUM(n) FILTER (WHERE a < DATE '2026-09-01') AS old
                FROM read_parquet('data/derived/delays.parquet') WHERE p = 'Kaufland Online-Marktplatz'""").df().iloc[0]
m["kauTwoDays"] = pct(ka.two, 1)
m["kauSameDay"] = pct(ka.same, 1)
m["kauOldN"] = c(ka.old)
m["kauFrom"] = f"{pd.Timestamp(ka.amin).day} {pd.Timestamp(ka.amin):%B %Y}"
# Amazon late statements in 2026
al = con.sql("""SELECT SUM(n) FILTER (WHERE d > 30) / SUM(n) AS s FROM read_parquet('data/derived/delays.parquet')
                WHERE p = 'Amazon' AND a >= DATE '2026-03-01' AND a <= DATE '2026-03-31'""").df().iloc[0]
m["amzLateMarch"] = pct(al.s)
# nowcasting
mt = {"naive": "Count known at $t$", "cl28": "Chain ladder, 28 days", "cl91": "Chain ladder, 91 days", "lag91": "Completed curve, 91 days",
      "bf91": "Bornhuetter and Ferguson, 91 days"}
fz = lambda x: "0.0" if round(x, 1) == 0 else f"{x:.1f}"
with open("tex/tab_nowcast.tex", "w") as o:
    o.write("\\begin{tabular}{lrrrrrr}\\toprule\n & \\multicolumn{3}{c}{Before 1 July 2025} & \\multicolumn{3}{c}{From 1 July 2025}\\\\\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\n")
    o.write("Method & \\shortstack{Median signed\\\\score (\\%)} & Factor 1.1 & Coverage & \\shortstack{Median signed\\\\score (\\%)} & Factor 1.1 & Coverage\\\\\\midrule\n")
    for srs_ in ["Without Google Shopping", "All platforms"]:
        o.write("\\multicolumn{7}{l}{" + srs_ + ", 28 day total}\\\\\n")
        for k_ in ["naive", "cl28", "cl91", "lag91", "bf91"]:
            cells = []
            for post in [False, True]:
                r = ns[(ns.series == srs_) & (ns.method == k_) & (ns.post_migration == post)].iloc[0]
                cells += [fz(100 * (np.exp(r.bias) - 1)), pct(r.within10), pct(r.coverage) if not np.isnan(r.coverage) else ""]
            o.write(f"{mt[k_]} & " + " & ".join(cells) + "\\\\\n")
    o.write("\\multicolumn{7}{l}{Platforms with at least one million statements, pooled platform evaluations}\\\\\n")
    PS = pd.DataFrame(NC["platform_pooled"])
    for k_ in ["naive", "cl28", "cl91", "lag91", "bf91"]:
        cells = []
        for post in [False, True]:
            r = PS[(PS.method == k_) & (PS.post_migration == post)].iloc[0]
            cells += [fz(100 * (np.exp(r.bias) - 1)), pct(r.within10), pct(r.coverage)]
        o.write(f"{mt[k_]} & " + " & ".join(cells) + "\\\\\n")
    o.write("\\bottomrule\\end{tabular}\n")
g = lambda s_, k_, post, col: ns[(ns.series == s_) & (ns.method == k_) & (ns.post_migration == post)][col].iloc[0]
for k_, tag in [("naive", "Naive"), ("cl28", "ClTw"), ("cl91", "ClNi"), ("lag91", "Lag"), ("bf91", "Bf")]:
    for post, pt in [(False, "Pre"), (True, "Post")]:
        m[f"nc{tag}{pt}Bias"] = f"{100 * (np.exp(g('Without Google Shopping', k_, post, 'bias')) - 1):.1f}"
        m[f"nc{tag}{pt}Within"] = pct(g("Without Google Shopping", k_, post, "within10"))
        m[f"nc{tag}{pt}Cov"] = pct(g("Without Google Shopping", k_, post, "coverage"))
m["ncPreN"] = str(int(g("Without Google Shopping", "naive", False, "n")))
m["ncPostN"] = str(int(g("Without Google Shopping", "naive", True, "n")))
wx = ne[ne.series == "Without Google Shopping"]
cl = wx[wx.method == "cl28"].assign(e=lambda x: 100 * (np.exp(x.err) - 1))
for tag, part in [("Pre", cl[~cl.post_migration]), ("Post", cl[cl.post_migration])]:
    top_ = part.sort_values("e", ascending=False).iloc[0]
    m[f"ncCl{tag}Max"] = f"{top_.e:.1f}"
    m[f"ncCl{tag}MaxDate"] = f"{top_.t.day} {top_.t:%B %Y}"
ratio = lambda s_, a_, b_: ne[(ne.series == s_) & (ne.method == "cl28") & (ne.t >= a_) & (ne.t <= b_)].eval("nowcast / target").max()
m["ncAmzClMaxRatio"] = f"{ratio('Amazon', '2024-03-01', '2024-05-31'):.1f}"
m["ncTtClMaxRatio"] = f"{ratio('TikTok', '2026-01-01', '2026-03-31'):.0f}"
assert pd.Timestamp("2024-03-01") <= pd.Timestamp(cl[~cl.post_migration].sort_values("e").t.iloc[-1]) <= pd.Timestamp("2024-05-31")
assert pd.Timestamp("2026-01-01") <= pd.Timestamp(cl[cl.post_migration].sort_values("e").t.iloc[-1]) <= pd.Timestamp("2026-03-31")
m["ncPreCovN"] = str(int(wx[(wx.method == "cl28") & ~wx.post_migration].covered.notna().sum()))
m["ncPostCovN"] = str(int(wx[(wx.method == "cl28") & wx.post_migration].covered.notna().sum()))
m["ncFirstInterval"] = f"{wx[wx.covered.notna()].t.min().day} {wx[wx.covered.notna()].t.min():%B %Y}"
PE = ne[~ne.series.isin(["All platforms", "Without Google Shopping"])]
best = PE.groupby(["series", "method"]).err.apply(lambda e: e.abs().median()).unstack()
m["ncPlatformsEval"] = str(len(best))
m["ncNaiveOnePct"] = str(int((best.naive < np.log(1.01)).sum()))
assert int((best.cl28 < best.naive).sum()) == NC["cl28_beats_naive_platforms"]
ttz = PE[(PE.series == "TikTok") & (PE.method == "naive") & (PE.nowcast <= 0)].sort_values("t")
m["ttZeroEvals"] = str(len(ttz))
m["ttZeroMonths"] = " and ".join(pd.Timestamp(2000, k_, 1).strftime("%B") for k_ in sorted(set(ttz.t.dt.month))) + " " + ttz.t.dt.strftime("%Y").iloc[0]
assert ttz.t.dt.year.nunique() == 1
m.update({"ncEval": str(NC["eval_days"]), "ncPlatforms": str(NC["platforms"]), "ncWindow": str(NC["window"]), "ncHorizon": str(NC["horizon"]),
          "ncClBeats": str(NC["cl28_beats_naive_platforms"]), "ncBfBeats": str(NC["bf91_beats_naive_platforms"])})
# design parameters stated in the text
m.update({"rdMinN": c(RD["min_n"]), "rdMinDays": str(RD["min_days"]), "rdDays": str(2 * RD["window"]),
          "rdPlcFirst": f"{pd.Timestamp(RD['placebo_first']).day} {pd.Timestamp(RD['placebo_first']):%B %Y}",
          "rdPlcLast": f"{pd.Timestamp(RD['placebo_last']).day} {pd.Timestamp(RD['placebo_last']):%B %Y}",
          "vintMinFinal": c(R["vint_min_final"]), "spikeRatio": str(R["spike_ratio"]),
          "ncMinDen": str(NC["min_den"]), "ncCap": f"{NC['cap']:.0f}", "ncLongWin": str(NC["long"]), "ncShortWin": str(NC["short"]),
          "ncMinPast": str(NC["min_past"]), "ncQlo": f"{100 * NC['qlo']:.0f}th", "ncQhi": f"{100 * NC['qhi']:.0f}th",
          "ncLevel": f"{100 * (NC['qhi'] - NC['qlo']):.0f}", "ncFactor": f"{NC['factor']:g}",
          "ncEvalFirst": f"{pd.Timestamp(NC['eval_first']).day} {pd.Timestamp(NC['eval_first']):%B %Y}",
          "ncEvalLast": f"{pd.Timestamp(NC['eval_last']).day} {pd.Timestamp(NC['eval_last']):%B %Y}"})
assert {f"cl{NC['long']}", f"lag{NC['long']}", f"bf{NC['long']}", f"cl{NC['short']}"} == set(ne.method) - {"naive"}
# late statements and spike days
late_all = con.sql("""SELECT p = 'Google Shopping' AS gs, SUM(n) AS n FROM read_parquet('data/derived/delays.parquet') WHERE d > 30 GROUP BY 1""").df().set_index("gs").n
m["lateBn"] = bn(late_all.sum())
m["lateOnSpike"] = pct(sp.late.sum() / late_all.sum())
m["lateOnSpikeEx"] = pct(sp[sp.p != "Google Shopping"].late.sum() / late_all.loc[False])
ttc = con.sql("""SELECT a + CAST(d AS INTEGER) AS c, SUM(n) AS n, SUM(n) FILTER (WHERE d > 30) AS late
                 FROM read_parquet('data/derived/delays.parquet') WHERE p = 'TikTok' GROUP BY 1""").df().fillna(0)
assert "TikTok" not in set(sp.p)
m["ttLateMn"] = mn(ttc.late.sum())
m["ttMaxLateRatio"] = f"{ttc.late.max() / ttc.n.median():.1f}"
m["ttMaxLateDate"] = f"{pd.Timestamp(ttc.sort_values('late').c.iloc[-1]).day} {pd.Timestamp(ttc.sort_values('late').c.iloc[-1]):%B %Y}"
# Amazon on its largest late day: which decisions
bl = con.sql(f"""SELECT date_trunc('month', a) AS mth, SUM(n) AS n FROM read_parquet('data/derived/delays.parquet')
                 WHERE p = 'Amazon' AND d > 30 AND a + CAST(d AS INTEGER) = DATE '{big_late.c.date()}' GROUP BY 1 ORDER BY n DESC""").df()
bl["mth"] = pd.to_datetime(bl.mth)
m["amzBigLateHtwo"] = pct(bl[bl.mth >= CUT].n.sum() / bl.n.sum(), 1)
m["amzBigLateTopMonth"] = bl.mth.iloc[0].strftime("%B %Y")
m["amzBigLateTopMn"] = mn(bl.n.iloc[0])
# notices and own initiative within platforms
ty_p = con.sql("""SELECT p, s, SUM(n) AS n, SUM(n) FILTER (WHERE d <= 7) / SUM(n) AS d7 FROM read_parquet('data/derived/delays_by_type.parquet')
                  WHERE p <> 'Google Shopping' GROUP BY p, s""").df().pivot(index="p", columns="s")
ty_p = ty_p[(ty_p[("n", "notice")] >= 100_000) & (ty_p[("n", "own initiative")] >= 100_000)]
m["typePlatforms"] = str(len(ty_p))
m["typeNoticeFaster"] = str(int((ty_p[("d7", "notice")] > ty_p[("d7", "own initiative")]).sum()))
m["typeNoticeSlower"] = str(int((ty_p[("d7", "notice")] < ty_p[("d7", "own initiative")]).sum()))
# platforms left out of the migration analysis by the rule on days with data
cw = con.sql(f"""SELECT p, a >= DATE '2025-07-01' AS post, COUNT(DISTINCT a) AS days, SUM(n) AS n,
                 SUM(n) FILTER (WHERE d <= 7) / SUM(n) AS w7, SUM(n) FILTER (WHERE d <= 30) / SUM(n) AS w30
                 FROM read_parquet('data/derived/delays.parquet')
                 WHERE a >= DATE '2025-06-03' AND a < DATE '2025-07-29' GROUP BY 1, 2""").df()
pv = cw.pivot(index="p", columns="post")
vol_ok = pv[(pv[("n", False)] / RD["window"] >= RD["min_n"]) & (pv[("n", True)] / RD["window"] >= RD["min_n"])]
dropped = sorted(set(vol_ok.index) - set(rpl.p))
assert dropped == ["Conrad"], dropped
cr = vol_ok.loc["Conrad"]
m.update({"conDays": str(int(cr[("days", False)] + cr[("days", True)])), "conSevenPre": pct(cr[("w7", False)]), "conSevenPost": pct(cr[("w7", True)]),
          "conThirtyPre": pct(cr[("w30", False)]), "conThirtyPost": pct(cr[("w30", True)])})
# placebo cutoffs available per platform
npl = rpl.w7_nplc
m.update({"plcPlatMin": str(int(npl[npl > 0].min())), "plcPlatMedian": f"{npl[npl > 0].median():g}", "plcPlatZero": str(int((npl == 0).sum())),
          "plcPlatZeroName": ", ".join(rpl[npl == 0].p)})
# platform months below 90 percent before and after the migration
m["vintPmPre"] = pct((vp[vp.m < CUT].r0 < 0.9).mean())
m["vintPmPost"] = pct((vp[vp.m >= CUT].r0 < 0.9).mean())
# platform level migration table, both estimators
lists = [RD[o_][k_] for o_ in ["w0", "w7", "w30"] for k_ in ["down_list", "up_list", "dm_down_list", "dm_up_list"]]
shown_p = [p_ for p_ in rpl.p if any(p_ in l_ for l_ in lists)]
with open("tex/tab_rdplat.tex", "w") as o:
    o.write("\\begin{tabular}{llrrrrrr}\\toprule\nPlatform & Outcome & Before & After & Local linear jump & $p$ & Difference in means & $p$\\\\\\midrule\n")
    lab = {"w0": "Same day", "w7": "Within 7 days", "w30": "Within 30 days"}
    for p_ in shown_p:
        r = rpl.set_index("p").loc[p_]
        for o_ in ["w0", "w7", "w30"]:
            o.write(f"{short(p_) if o_ == 'w0' else ''} & {lab[o_]} & {r[o_ + '_pre']:.3f} & {r[o_ + '_post']:.3f} & {f3(r[o_ + '_jump'])} & {r[o_ + '_p']:.3f} & {f3(r[o_ + '_dm'])} & {r[o_ + '_dm_p']:.3f}\\\\\n")
    o.write("\\bottomrule\\end{tabular}\n")
m["rdPlatShown"] = str(len(shown_p))
for o_, t2 in [("w0", "Zero"), ("w7", "Seven"), ("w30", "Thirty")]:
    m[f"ll{t2}DownList"] = ", ".join(short(x) for x in RD[o_]["down_list"]) or "none"
    m[f"ll{t2}UpList"] = ", ".join(short(x) for x in RD[o_]["up_list"]) or "none"
rt = rpl.set_index("p")
m.update({"ttSevenLl": pct(rt.loc["TikTok", "w7_jump"], 1), "ttSevenLlP": f"{rt.loc['TikTok', 'w7_p']:.3f}",
          "ttThirtyPre": pct(rt.loc["TikTok", "w30_pre"], 1), "ttThirtyLl": pct(rt.loc["TikTok", "w30_jump"], 1),
          "amzThirtyPre": pct(rt.loc["Amazon", "w30_pre"], 1), "amzThirtyLl": pct(rt.loc["Amazon", "w30_jump"], 1),
          "lebSevenLl": pct(rt.loc["leboncoin", "w7_jump"], 1), "lebSevenLlP": f"{rt.loc['leboncoin', 'w7_p']:.3f}",
          "xNpre": c(rt.loc["X", "n_pre"]), "sheinNpre": c(rt.loc["Shein", "n_pre"])})
assert rt.loc["TikTok", "w30_pre"] + rt.loc["TikTok", "w30_jump"] > 1.0                 # jump above a level of 100 percent
assert rt.loc["Amazon", "w30_pre"] + rt.loc["Amazon", "w30_jump"] > 1.0
m["typeMin"] = c(100_000)
# nowcast windows: first evaluation day with the full 91 fully observed days (decision days from 1 September 2023)
first_full = pd.Timestamp("2023-09-01") + pd.Timedelta(days=90 + NC["horizon"])
ev = pd.date_range(NC["eval_first"], NC["eval_last"], freq="7D")
m["ncLagFull"] = f"{ev[ev >= first_full][0].day} {ev[ev >= first_full][0]:%B %Y}"
# nowcasting: pooled platform evaluations and the series of all platforms
PSd = pd.DataFrame(NC["platform_pooled"]).set_index(["method", "post_migration"])
for k_, tag in [("naive", "Naive"), ("cl28", "ClTw"), ("cl91", "ClNi"), ("lag91", "Lag"), ("bf91", "Bf")]:
    for post, pt in [(False, "Pre"), (True, "Post")]:
        m[f"ncP{tag}{pt}Within"] = pct(PSd.loc[(k_, post), "within10"])
        m[f"ncP{tag}{pt}Cov"] = pct(PSd.loc[(k_, post), "coverage"])
        r = ns[(ns.series == "All platforms") & (ns.method == k_) & (ns.post_migration == post)].iloc[0]
        m[f"ncA{tag}{pt}Within"] = pct(r.within10)
        m[f"ncA{tag}{pt}Cov"] = pct(r.coverage)
        m[f"ncA{tag}{pt}Bias"] = f"{100 * (np.exp(r.bias) - 1):.1f}"
m["ncClWorse"] = str(int((best.cl28 > best.naive).sum()))
m["ncClTies"] = str(int((best.cl28 == best.naive).sum()))
m["ncBfWorse"] = str(int((best.bf91 > best.naive).sum()))
assert int((best.cl28 < best.naive).sum()) + int(m["ncClWorse"]) + int(m["ncClTies"]) == len(best)
assert all((best[k_] < best.naive).sum() < len(best) / 2 for k_ in ["cl28", "cl91", "lag91", "bf91"])   # no method better for most platforms
# coverage ranges over the five methods, pooled platform evaluations
for post in [False, True]:
    cv = PSd.xs(post, level="post_migration").coverage
    assert cv.min() == PSd.loc[("bf91", post), "coverage"]
    assert cv.max() == PSd.loc[("cl28" if not post else "naive", post), "coverage"]
    assert cv.max() < 0.8
# fitted local linear levels at the cutoff for the share within thirty days
def fitted_at_cut(p_, o_):
    z = rse[rse.p == p_].sort_values("a")
    tt_ = (z.a - CUT).dt.days.values.astype(float)
    post_ = (tt_ >= 0).astype(float)
    dow_ = np.column_stack([(z.a.dt.dayofweek.values == j).astype(float) for j in range(1, 7)])
    X_ = np.column_stack([np.ones_like(tt_), post_, tt_, tt_ * post_, dow_])
    b_ = np.linalg.lstsq(X_, z[o_].values.astype(float), rcond=None)[0]
    dmean = np.r_[0.0, b_[4:]].mean()
    assert abs(b_[1] - rt.loc[p_, o_ + "_jump"]) < 1e-9
    return b_[0] + dmean, b_[0] + b_[1] + dmean
for p_, tg in [("TikTok", "tt"), ("Amazon", "amz")]:
    pre_, post_ = fitted_at_cut(p_, "w30")
    assert post_ > 1.0
    m[f"{tg}ThirtyFitPre"], m[f"{tg}ThirtyFitPost"] = pct(pre_, 1), pct(post_, 1)
# when the share within seven days dropped after the cutoff
def drop_day(frame):
    z = frame.sort_values("a")
    pre_mean = z[z.a < CUT].w7.mean()
    after = z[z.a >= CUT]
    first = after[after.w7 < pre_mean - 0.10].a.min()
    return first, (after[after.a >= first].w7 < pre_mean - 0.10).mean(), after[after.a < first].w7.min()
for nm_, tg in [("TikTok", "tt"), ("Without Google Shopping", "ex")]:
    f_, share_, before_min = drop_day(rse[rse.p == nm_])
    m[f"{tg}DropDate"] = f"{f_.day} {f_:%B %Y}"
    m[f"{tg}DropShare"] = pct(share_)
    m[f"{tg}FirstDaysMin"] = pct(before_min, 1)
    m[f"{tg}LastHigh"] = f"{(f_ - pd.Timedelta(days=1)).day} {(f_ - pd.Timedelta(days=1)):%B %Y}"
ttv = con.sql("""SELECT a + CAST(d AS INTEGER) AS c, SUM(n) AS n FROM read_parquet('data/derived/delays.parquet')
                 WHERE p = 'TikTok' GROUP BY 1""").df()
ttv["c"] = pd.to_datetime(ttv.c)
m["ttCreatedPre"] = mn(ttv[(ttv.c >= CUT - pd.Timedelta(days=RD["window"])) & (ttv.c < CUT)].n.mean())
m["ttCreatedPost"] = mn(ttv[(ttv.c >= CUT) & (ttv.c < CUT + pd.Timedelta(days=RD["window"]))].n.mean())
# platform months below 90 percent for the same platforms in the first half of 2025 and from July 2025
h1 = vp[(vp.m >= pd.Timestamp("2025-01-01")) & (vp.m < CUT)]
h2 = vp[vp.m >= CUT]
both = set(h1.p) & set(h2.p)
m["vintPmBal"] = str(len(both))
m["vintPmHone"] = pct((h1[h1.p.isin(both)].r0 < 0.9).mean(), 1)
m["vintPmHtwo"] = pct((h2[h2.p.isin(both)].r0 < 0.9).mean(), 1)
# follow up after the last decision day of the migration window
m["rdFollowMin"] = str((pd.Timestamp("2026-09-30") - (CUT + pd.Timedelta(days=RD["window"] - 1))).days)
# write macros (proper minus signs)
with open("tex/macros.tex", "w") as o:
    for k_, v in m.items():
        v = re.sub(r"^-(\d)", r"$-$\1", str(v))
        o.write("\\newcommand{\\" + k_ + "}{" + v + "}\n")
for f in glob.glob("tex/tab_*.tex"):
    t = open(f).read()
    open(f, "w").write(re.sub(r"(?<![\w$.])-(\d)", r"$-$\1", t))

# Figure 1: cumulative delay distribution and share within 7 days by application month
fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.5))
x = ov[ov.d <= 120]
ax[0].step(x.d, 100 * x.cum, where="post", color="black", lw=0.9, label="All platforms")
for nm, ls in [("TikTok", "--"), ("Amazon", ":"), ("Google Shopping", "-.")]:
    z = con.sql(f"SELECT d, SUM(n) AS n FROM read_parquet('data/derived/delays.parquet') WHERE p = '{nm}' GROUP BY d ORDER BY d").df()
    z["cum"] = z.n.cumsum() / z.n.sum()
    z = z[z.d <= 120]
    ax[0].step(z.d, 100 * z.cum, where="post", color="black", lw=0.7, ls=ls, label=nm)
ax[0].set_xlabel("Days from decision to statement"); ax[0].set_ylabel("Cumulative share (%)"); ax[0].legend(frameon=False, fontsize=6)
ax[0].set_title("A. Delay distribution", fontsize=7, loc="left")
mo = con.sql("""SELECT date_trunc('month', a) AS mth, SUM(n) FILTER (WHERE d <= 7) / SUM(n) AS w7, SUM(n) FILTER (WHERE d <= 30) / SUM(n) AS w30
                FROM read_parquet('data/derived/delays.parquet') WHERE p <> 'Google Shopping' AND a >= DATE '2023-10-01' AND a <= DATE '2026-03-31'
                GROUP BY 1 ORDER BY 1""").df()
mo["mth"] = pd.to_datetime(mo.mth)
early = mo[mo.mth < pd.Timestamp("2024-02-01")]
stable = mo[(mo.mth >= pd.Timestamp("2024-02-01")) & (mo.mth < CUT)]
late_ = mo[mo.mth >= CUT]
m2 = {"moEarlyMin": pct(early.w30.min(), 1), "moEarlyMinMonth": early.sort_values("w30").mth.iloc[0].strftime("%B %Y"),
      "moStableMin": pct(stable.w30.min(), 1), "moPostBelow": str(int((late_.w30 < 0.9).sum())), "moPostMonths": str(len(late_)),
      "moPostMin": pct(late_.w30.min(), 1), "moPostMinMonth": late_.sort_values("w30").mth.iloc[0].strftime("%B %Y")}
with open("tex/macros.tex", "a") as o:
    for k_, v in m2.items():
        o.write("\\newcommand{\\" + k_ + "}{" + v + "}\n")
m.update(m2)
ax[1].plot(mo.mth, 100 * mo.w7, "o-", ms=2.5, color="black", lw=0.8, mfc="white", label="Within 7 days")
ax[1].plot(mo.mth, 100 * mo.w30, "s-", ms=2.3, color="black", lw=0.8, label="Within 30 days")
ax[1].axvline(CUT, color="black", lw=0.5, ls=":")
ax[1].set_ylabel("Statements, all platforms but Google Shopping (%)", fontsize=6.5); ax[1].legend(frameon=False, fontsize=6)
ax[1].set_title("B. By month of decision", fontsize=7, loc="left")
ax[1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
for l_ in ax[1].get_xticklabels():
    l_.set_rotation(30); l_.set_ha("right")
fig.tight_layout(); save_figure(fig, "figures/fig_delays.pdf", metadata={"Creator": None, "Producer": None, "CreationDate": None, "ModDate": None}, bbox_inches="tight", pad_inches=0.06); plt.close(fig)

# Figure 2: vintages, all platforms except Google Shopping, and platform months below 90 percent
fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.5))
for k, mk in [(0, "o"), (30, "s"), (180, "^")]:
    ax[0].plot(vx.m, 100 * vx[f"r{k}"], mk + "-", ms=2.4, lw=0.7, color="black", mfc="white" if k != 30 else "black",
               label={0: "End of month", 30: "30 days later", 180: "180 days later"}[k])
ax[0].axvline(CUT, color="black", lw=0.5, ls=":")
ax[0].set_ylabel("Share of eventual count known (%)"); ax[0].legend(frameon=False, fontsize=6)
ax[0].set_title("A. Without Google Shopping", fontsize=7, loc="left")
ax[0].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
for l_ in ax[0].get_xticklabels():
    l_.set_rotation(30); l_.set_ha("right")
bel = vp.groupby("m").apply(lambda g: pd.Series({"b0": (g.r0 < 0.9).mean(), "b30": (g.r30 < 0.9).mean()}), include_groups=False).reset_index()
ax[1].plot(bel.m, 100 * bel.b0, "o-", ms=2.4, lw=0.7, color="black", mfc="white", label="End of month")
ax[1].plot(bel.m, 100 * bel.b30, "s-", ms=2.3, lw=0.7, color="black", label="30 days later")
ax[1].axvline(CUT, color="black", lw=0.5, ls=":")
ax[1].set_ylabel("Platforms with less than 90% known (%)", fontsize=6.5); ax[1].legend(frameon=False, fontsize=6)
ax[1].set_title("B. Platform months", fontsize=7, loc="left")
ax[1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
for l_ in ax[1].get_xticklabels():
    l_.set_rotation(30); l_.set_ha("right")
fig.tight_layout(); save_figure(fig, "figures/fig_vintages.pdf", metadata={"Creator": None, "Producer": None, "CreationDate": None, "ModDate": None}, bbox_inches="tight", pad_inches=0.06); plt.close(fig)

# Figure 3: the migration episode, weekly mean delay of TikTok and Amazon, and pooled share within 7 days
fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.5))
for nm, mk in [("TikTok", "o"), ("Amazon", "s")]:
    z = wk[wk.p == nm]
    ax[0].plot(z.wk, z.mean_d, mk + "-", ms=2.2, lw=0.7, color="black", mfc="white" if nm == "TikTok" else "black", label=nm)
ax[0].axvline(CUT, color="black", lw=0.5, ls=":")
ax[0].set_ylabel("Mean delay by week of decision (days)", fontsize=6.5); ax[0].legend(frameon=False, fontsize=6)
ax[0].set_title("A. Two large platforms", fontsize=7, loc="left")
ax[0].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
for l_ in ax[0].get_xticklabels():
    l_.set_rotation(30); l_.set_ha("right")
z = rse[rse.p == "Without Google Shopping"].sort_values("a")
ax[1].plot(z.a, 100 * z.w7, "o", ms=2, mfc="white", mec="black", mew=0.5)
for side in [z.a < CUT, z.a >= CUT]:
    zz = z[side]
    ax[1].hlines(100 * zz.w7.mean(), zz.a.min(), zz.a.max(), color="black", lw=0.8)
ax[1].axvline(CUT, color="black", lw=0.5, ls=":")
ax[1].set_ylabel("Within 7 days, without Google Shopping (%)", fontsize=6.5)
ax[1].set_title("B. Around 1 July 2025", fontsize=7, loc="left")
ax[1].xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
for l_ in ax[1].get_xticklabels():
    l_.set_rotation(30); l_.set_ha("right")
fig.tight_layout(); save_figure(fig, "figures/fig_migration.pdf", metadata={"Creator": None, "Producer": None, "CreationDate": None, "ModDate": None}, bbox_inches="tight", pad_inches=0.06); plt.close(fig)

# Figure 4: nowcast errors over evaluation days, without Google Shopping
fig, ax = plt.subplots(figsize=(6.6, 2.4))
z = ne[ne.series == "Without Google Shopping"]
for k_, mk in [("naive", "o"), ("cl28", "s"), ("bf91", "^")]:
    zz = z[z.method == k_].sort_values("t")
    ax.plot(zz.t, 100 * (np.exp(zz.err) - 1), mk + "-", ms=2, lw=0.6, color="black", mfc="white" if k_ != "cl28" else "black",
            label={"naive": "Count known at t", "cl28": "Chain ladder, 28 days", "bf91": "Bornhuetter and Ferguson"}[k_])
ax.axhline(0, color="black", lw=0.5)
ax.axvline(CUT, color="black", lw=0.5, ls=":")
ax.set_ylabel("Nowcast error (%)"); ax.legend(frameon=False, fontsize=6, ncol=3)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
fig.tight_layout(); save_figure(fig, "figures/fig_nowcast.pdf", metadata={"Creator": None, "Producer": None, "CreationDate": None, "ModDate": None}, bbox_inches="tight", pad_inches=0.06); plt.close(fig)
print(json.dumps(m, indent=1))

from layout_tables import format_generated_tables
format_generated_tables('P10')
