"""Multiple comparison adjustment for the paper's headline hypothesis tests (exploratory; no new estimation of the models).

    python -I -X utf8 analysis/multiple_testing.py

Family = the headline tests quoted in paper/draft/FACTS.md for RQ2, RQ3 and RQ4. Exact p-values are read from the result files in
results/explore/ where they exist; otherwise the p-value is recovered from the reported 95% CI by the normal approximation
(se = (hi - lo) / (2 * 1.96), z = estimate / se, two sided) and the row says so. Holm (step down, FWER) and Benjamini-Hochberg
(FDR) are applied to the whole family and within each RQ. Writes results/explore/multiple_testing.md and multiple_testing.csv.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

import common as C  # noqa: E402
import models as M  # noqa: E402

EX = os.path.join(C.RESULTS, "explore")
NORMAL_NOTE = "recovered from the 95% CI by the normal approximation"


def p_from_ci(est, lo, hi):
    se = (hi - lo) / (2 * 1.959964)
    return float(2 * stats.norm.sf(abs(est / se)))


def bh(p):
    p = np.asarray(p, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    run = 1.0
    for rank in range(m - 1, -1, -1):
        i = order[rank]
        run = min(run, p[i] * m / (rank + 1))
        adj[i] = run
    return adj


def parse_ci(s):
    """'-0.080 [-0.099, -0.062]' -> (est, lo, hi)"""
    nums = [float(x) for x in re.findall(r"-?\d+\.?\d*(?:e-?\d+)?", s)]
    return nums[0], nums[1], nums[2]


def row(rq, test, est, lo, hi, p, psrc, src, d=3):
    return {"RQ": rq, "test": test, "estimate": est, "ci_lo": lo, "ci_hi": hi, "raw_p": p, "p_basis": psrc, "source": src}


def build():
    R = []
    # ---------------- RQ2
    pc = pd.read_csv(os.path.join(EX, "rq2_t1_ppi_contrasts.csv"))
    r = pc[(pc["method"] == "PPI++ (lambda tuned)") & (pc["contrast"] == "LS18 - LS13")].iloc[0]
    e, lo, hi = parse_ci(r["diff"])
    R.append(row("RQ2", "PPI++ LS18 minus LS13", e, lo, hi, float(2 * stats.norm.sf(abs(r["z"]))), f"exact: from reported z = {r['z']:.3f} (normal)",
                 "results/explore/rq2_t1_ppi_contrasts.csv"))
    sl = pd.read_csv(os.path.join(EX, "rq2_t1_ppi_slope.csv"))
    e, lo, hi = parse_ci(sl[sl["method"] == "PPI++ (lambda tuned)"].iloc[0, 1])
    R.append(row("RQ2", "PPI++ trend per decade", e, lo, hi, p_from_ci(e, lo, hi), NORMAL_NOTE, "results/explore/rq2_t1_ppi_slope.csv"))
    rl = pd.read_csv(os.path.join(EX, "rq2_t2_reg_ls.csv"))
    x = rl[(rl["y"] == "y_m2") & rl["spec"].str.startswith("S5 ")].iloc[0]
    R.append(row("RQ2", "Controlled LS18 minus LS13 (S5)", x["_b18"], x["_lo18"], x["_hi18"], p_from_ci(x["_b18"], x["_lo18"], x["_hi18"]),
                 NORMAL_NOTE + " (the model CI is t with 77 df; the p-value differs negligibly)", "results/explore/rq2_t2_reg_ls.csv"))
    rd = pd.read_csv(os.path.join(EX, "rq2_t2_reg_dec.csv"))
    x = rd[(rd["y"] == "y_m2") & rd["spec"].str.startswith("S5 ")].iloc[0]
    R.append(row("RQ2", "Controlled trend per decade (S5)", x["_b"], x["_lo"], x["_hi"], p_from_ci(x["_b"], x["_lo"], x["_hi"]),
                 NORMAL_NOTE + " (model CI is t with 77 df)", "results/explore/rq2_t2_reg_dec.csv"))
    sg = pd.read_csv(os.path.join(EX, "rq2_t4_segmented.csv"))
    sg = sg[(sg["series"] == "unadjusted") & (sg["type"] == "government change")]
    for _, x in sg.iterrows():
        R.append(row("RQ2", f"Jump at change of government {x['break'][:4]}", x["_jump"], x["_jlo"], x["_jhi"], float(x["p(jump)"]),
                     "exact: p(jump) from file", "results/explore/rq2_t4_segmented.csv (unadjusted series)"))
    ct = pd.read_csv(os.path.join(EX, "rq2_t3_contributions.csv"))
    for typ, lab in (("INCOMPLETE", "INCOMPLETE rise LS18 minus LS13 (pp of parts)"), ("DEFERRAL", "DEFERRAL fall LS18 minus LS13 (pp of parts)")):
        s = ct[(ct["contrast"] == "LS18 - LS13") & (ct["type"] == typ)].iloc[0]["change (pp of parts)"]
        e, lo, hi = parse_ci(s)
        R.append(row("RQ2", lab, e, lo, hi, p_from_ci(e, lo, hi), NORMAL_NOTE + " (CI is a session cluster bootstrap percentile CI)",
                     "results/explore/rq2_t3_contributions.csv"))
    # ---------------- RQ3
    am = pd.read_csv(os.path.join(EX, "rq3_all_models.csv"))
    for ls in range(13, 19):
        x = am[(am["term"] == "opp") & (am["spec"] == "two-way (MP,session)") & (am["group"] == f"LS{ls}")].iloc[0]
        R.append(row("RQ3", f"Opposition minus ruling gap, LS{ls}", x["estimate"], x["ci_lo"], x["ci_hi"], x["p"], "exact: t(sessions-1) p from file",
                     "results/explore/rq3_all_models.csv"))
    x = am[(am["term"] == "c_gov_partner") & (am["group"] == "LS18") & (am["spec"] == "roles by LS (ref gov_main)")].iloc[0]
    R.append(row("RQ3", "LS18 coalition partner minus BJP gap", x["estimate"], x["ci_lo"], x["ci_hi"], x["p"], "exact: t(sessions-1) p from file",
                 "results/explore/rq3_all_models.csv"))
    x = am[(am["term"] == "opp") & (am["group"] == "gold LS17-18") & (am["spec"] == "y_gold, 1/pi weights, LS FE + core ctrl")].iloc[0]
    R.append(row("RQ3", "Gold LS17+LS18 opposition gap (n = 180)", x["estimate"], x["ci_lo"], x["ci_hi"], x["p"], "exact: p from file",
                 "results/explore/rq3_all_models.csv"))
    # ---------------- RQ4
    mt = pd.read_csv(os.path.join(EX, "a1_ministry_table_full.csv"))
    w = 1 / mt["score_se"] ** 2
    xbar = (w * mt["score"]).sum() / w.sum()
    Q = float((w * (mt["score"] - xbar) ** 2).sum())
    df_ = len(mt) - 1
    pQ = float(stats.chi2.sf(Q, df_))
    I2 = max(0.0, (Q - df_) / Q)
    R.append(dict(row("RQ4", f"Ministry heterogeneity (Cochran Q across {len(mt)} ministries, df {df_}, I2 {I2:.3f})", Q, np.nan, np.nan, pQ,
                      "computed here from the table (score, score_se): Q = sum w (x - xbar)^2, chi2(54); no CI",
                      "results/explore/a1_ministry_table_full.csv"), estimate_label=f"Q = {Q:.1f}"))
    # jurisdiction rise, LS13 0.71 [0.67, 0.76] -> LS18 1.83 [1.73, 1.93] (% of all parts), from results/explore/rq4_rq1.md A2 (2 decimals)
    a, alo, ahi = 0.71, 0.67, 0.76
    b, blo, bhi = 1.83, 1.73, 1.93
    sa, sb = (ahi - alo) / 3.919928, (bhi - blo) / 3.919928
    dlt = b - a
    sed = float(np.hypot(sa, sb))
    R.append(row("RQ4", "JURISDICTION share rise LS13 to LS18 (pp of parts)", dlt, dlt - 1.959964 * sed, dlt + 1.959964 * sed,
                 float(2 * stats.norm.sf(abs(dlt / sed))),
                 "recovered from the two reported CIs (independent, normal approximation; inputs rounded to 2 decimals)",
                 "results/explore/rq4_rq1.md A2 (by LS table)"))
    return pd.DataFrame(R)


def fmt_p(p):
    if p < 1e-300:
        return "<1e-300"
    return f"{p:.2e}" if p < 1e-3 else f"{p:.4f}"


def main():
    D = build()
    D["holm_p_all"] = M.holm(D["raw_p"].values)
    D["bh_q_all"] = bh(D["raw_p"].values)
    D["survives_holm_all"] = D["holm_p_all"] < 0.05
    D["holm_p_within_RQ"] = np.nan
    D["bh_q_within_RQ"] = np.nan
    for rq, g in D.groupby("RQ"):
        D.loc[g.index, "holm_p_within_RQ"] = M.holm(g["raw_p"].values)
        D.loc[g.index, "bh_q_within_RQ"] = bh(g["raw_p"].values)
    D["survives_holm_within_RQ"] = D["holm_p_within_RQ"] < 0.05
    D["survives_bh_all_q05"] = D["bh_q_all"] < 0.05
    D.to_csv(os.path.join(EX, "multiple_testing.csv"), index=False)

    def est_s(r):
        return r["estimate_label"] if isinstance(r.get("estimate_label"), str) else f"{r['estimate']:.3f}"

    def ci_s(r):
        return "n/a" if pd.isna(r["ci_lo"]) else f"[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]"

    def table(d, within):
        hc, bc, sc = ("holm_p_within_RQ", "bh_q_within_RQ", "survives_holm_within_RQ") if within else ("holm_p_all", "bh_q_all", "survives_holm_all")
        lines = ["| RQ | test | estimate | 95% CI | raw p | p basis | Holm p | BH q | survives at 0.05 (Holm) | source |", "|---|---|---|---|---|---|---|---|---|---|"]
        for _, r in d.iterrows():
            lines.append(f"| {r['RQ']} | {r['test']} | {est_s(r)} | {ci_s(r)} | {fmt_p(r['raw_p'])} | {r['p_basis']} | {fmt_p(r[hc])} | {fmt_p(r[bc])} | "
                         f"{'yes' if r[sc] else 'NO'} | {r['source']} |")
        return "\n".join(lines) + "\n"

    m = len(D)
    fail_all = D[~D["survives_holm_all"]]["test"].tolist()
    fail_rq = D[~D["survives_holm_within_RQ"]]["test"].tolist()
    fail_bh = D[~D["survives_bh_all_q05"]]["test"].tolist()
    L = ["# Multiple comparison adjustment for the headline tests (exploratory)\n",
         f"Family: {m} headline tests quoted in FACTS.md (RQ2: {int((D.RQ=='RQ2').sum())}, RQ3: {int((D.RQ=='RQ3').sum())}, RQ4: {int((D.RQ=='RQ4').sum())}). "
         "Raw p-values are two sided. Exact p-values are taken from the result files where they exist; otherwise they are recovered from the 95% CI by the "
         "normal approximation (column 'p basis' says which, row by row). Holm controls the family wise error rate; Benjamini and Hochberg (BH q) controls the false discovery rate. "
         "Each is applied to the whole family and, separately, within each RQ.\n",
         "Notes. (1) The three government change jumps and the DEFERRAL and INCOMPLETE rows are separate tests of separate claims, but the jump tests are claims of NO effect, "
         "so for them a large adjusted p-value is the expected, consistent outcome (they were never expected to be significant); adjusting them mostly lengthens the family and "
         "makes the Holm threshold for the other tests slightly stricter. (2) The PPI++ and controlled LS18 minus LS13 and trend tests are four views of one decline and are strongly "
         "dependent; Holm is conservative under dependence. (3) The ministry heterogeneity row is a Cochran Q test computed here from the reported ministry scores and design based "
         "standard errors (it ignores any covariance between ministries, which are disjoint sets of parts, so this is reasonable). (4) The jurisdiction row uses rounded CI inputs.\n",
         "## Whole family (Holm and BH over all tests)\n", table(D, False),
         "## Within each RQ (Holm and BH over the tests of that RQ)\n", table(D, True),
         "## Summary\n",
         f"- Tests that do NOT survive Holm at 0.05 over the whole family ({len(fail_all)} of {m}): " + ("; ".join(fail_all) if fail_all else "none") + ".",
         f"- Tests that do NOT survive Holm within their RQ ({len(fail_rq)} of {m}): " + ("; ".join(fail_rq) if fail_rq else "none") + ".",
         f"- Tests with BH q at or above 0.05 over the whole family ({len(fail_bh)} of {m}): " + ("; ".join(fail_bh) if fail_bh else "none") + ".\n",
         "Full table: results/explore/multiple_testing.csv.\n"]
    with open(os.path.join(EX, "multiple_testing.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(D[["RQ", "test", "estimate", "raw_p", "holm_p_all", "bh_q_all", "survives_holm_all", "holm_p_within_RQ", "survives_holm_within_RQ"]].to_string())


if __name__ == "__main__":
    main()
