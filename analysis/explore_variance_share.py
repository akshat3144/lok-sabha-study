"""Share of the variance in question scores explained by ministry and by Lok Sabha (exploratory, RQ4).

Like for like comparison behind the claim that completeness varies more by ministry than over time: the design
weighted between group sum of squares of y_m2 over the total sum of squares (eta squared), for the grouping by
ministry and by Lok Sabha, on the analysis sample of labelled questions. Writes results/explore/variance_share.md.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FRAME = ROOT / "data/measures/analysis_frame_real.parquet"
OUT = ROOT / "results/explore/variance_share.md"


def eta_squared(df, group, y="y_m2", w="m2_weight"):
    mu = np.average(df[y], weights=df[w])
    tss = float(np.sum(df[w] * (df[y] - mu) ** 2))
    g = df.groupby(group)
    sw = g[w].sum()
    gm = g.apply(lambda d: np.average(d[y], weights=d[w]), include_groups=False)
    return float(np.sum(sw * (gm - mu) ** 2) / tss), int(len(sw))


def main():
    f = pd.read_parquet(FRAME)
    a = f[f["analysis_sample"] & f["y_m2"].notna() & (f["m2_weight"] > 0)]
    rows = []
    for label, col in (("Ministry", "ministry_canonical"), ("Lok Sabha", "ls"), ("Ministry group", "ministry_group")):
        e, k = eta_squared(a, col)
        rows.append((label, k, e))
    lines = ["# Variance in question scores explained by ministry and by Lok Sabha", "",
             f"Analysis sample, labelled questions: {len(a):,}. Outcome y_m2 (0 to 1), design weights m2_weight.", "",
             "| Grouping | Groups | Share of weighted variance (eta squared) |", "|---|---:|---:|"]
    lines += [f"| {r[0]} | {r[1]} | {r[2]:.4f} |" for r in rows]
    ratio = rows[0][2] / rows[1][2]
    lines += ["", f"Ministry / Lok Sabha ratio: {ratio:.2f}. With {len(a):,} questions the share expected by chance "
              f"from {rows[0][1]} groups is about {(rows[0][1] - 1) / len(a):.4f}."]
    OUT.write_text("\n".join(lines) + "\n", encoding="utf8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
