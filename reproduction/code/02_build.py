"""Delay table and data quality counts from the reporting triangle.

The delay of a statement is the number of days between the application of the decision and the creation of the
statement in the database. Statements with a creation day before the application day, or with an application day
before 25 August 2023 (four months after the designation of the first very large online platforms on 25 April
2023), are counted and set aside. The delay is stored uncapped. The table by source and automation keeps delays above DMAX days in one top bin.
Run from the package root after code/01_triangle.py:  python3 code/02_build.py
Outputs: data/derived/delays.parquet (p, a, d, n), data/derived/delays_by_type.parquet (p, a, d, s, ad, n),
results/quality.json
"""
import json
import duckdb

DMAX = 400
START = "2023-08-25"
T = "2026-09-30"
con = duckdb.connect()
con.sql("CREATE VIEW t AS SELECT * FROM read_parquet('data/derived/tri_*.parquet')")
q = con.sql(f"""SELECT SUM(n) AS total,
    SUM(n) FILTER (WHERE a IS NULL OR c IS NULL) AS missing_date,
    SUM(n) FILTER (WHERE c < a) AS created_before_applied,
    SUM(n) FILTER (WHERE a < DATE '{START}' AND c >= a) AS applied_before_start,
    SUM(n) FILTER (WHERE c - a > {DMAX} AND a >= DATE '{START}') AS delay_above_dmax,
    COUNT(DISTINCT p) AS platforms, MIN(c) AS first_created, MAX(c) AS last_created FROM t""").df().iloc[0]
Q = {k: (str(v) if k in ("first_created", "last_created") else float(v)) for k, v in q.items()}
Q["dmax"] = DMAX
Q["start"] = START
Q["t_end"] = T
base = f"FROM t WHERE a IS NOT NULL AND c IS NOT NULL AND c >= a AND a >= DATE '{START}' AND c <= DATE '{T}'"
con.sql(f"""COPY (SELECT p, a, CAST(c - a AS INTEGER) AS d, SUM(n) AS n {base} GROUP BY ALL ORDER BY p, a, d)
            TO 'data/derived/delays.parquet' (FORMAT parquet)""")
con.sql(f"""COPY (SELECT p, a, LEAST(c - a, {DMAX}) AS d,
            CASE WHEN s = 'SOURCE_VOLUNTARY' THEN 'own initiative' ELSE 'notice' END AS s, ad, SUM(n) AS n
            {base} GROUP BY ALL ORDER BY ALL) TO 'data/derived/delays_by_type.parquet' (FORMAT parquet)""")
Q["valid"] = float(con.sql(f"SELECT SUM(n) {base}").fetchone()[0])
json.dump(Q, open("results/quality.json", "w"), indent=1)
print(json.dumps(Q, indent=1))
