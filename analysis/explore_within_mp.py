"""Within-MP test of the decline: is the fall in y_m2 a change in WHO asks or in how ministers answer?

    python -I -X utf8 analysis/explore_within_mp.py

Writes results/explore/within_mp.md and results/explore/within_mp.csv. Reuses models.fit_model with exactly the conventions of
explore_rq2.py task 2: sample = analysis_sample & y_m2 notna & m2_weight > 0, design weights m2_weight, WLS, two-way clustered
SEs (first-asker MP = frame column mp_id, and session_id), t(min G - 1) CIs, LS dummies with LS13 as reference and a linear
trend per decade ((year - 2000) / 10). The controlled specification is S5 of explore_rq2.py: n_parts, log_qlen, starred, the
request-type shares rt_reason / rt_action / rt_opinion, plus ministry_canonical and topic fixed effects.
Multi-asker questions are assigned to the FIRST asker (mp_id), as everywhere else in the project.
MP fixed effects are dummies for mp_id (sparse design). Does not modify any data file or script, no network.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import common as C  # noqa: E402
import models as M  # noqa: E402

OUT = os.path.join(C.RESULTS, "explore")
os.makedirs(OUT, exist_ok=True)
CORE = ["n_parts", "log_qlen", "starred"]
RT = ["rt_reason", "rt_action", "rt_opinion"]
CTRL_NUM = CORE + RT
CTRL_CAT = ["ministry_canonical", "topic"]
EARLY, LATE = (13, 14, 15, 16), (17, 18)


def f3(e, lo, hi):
    return f"{e:.3f} [{lo:.3f}, {hi:.3f}]"


def load():
    df = pd.read_parquet(C.frame_path("real"))
    df["dec"] = (df["year"] - 2000.0) / 10.0
    est = df[df["analysis_sample"] & df["y_m2"].notna() & (df["m2_weight"] > 0)].copy()
    est["resolved"] = est["mp_id"].astype(str).str.startswith("mp")
    return df, est


def run(d, label, fe, ctrl, rows):
    """LS dummies model and decade trend model; append result rows."""
    num = CTRL_NUM if ctrl else []
    cat = (CTRL_CAT if ctrl else []) + (["mp_id"] if fe else [])
    fl = M.fit_model(d, "y_m2", num, cat + ["ls"], ref={"ls": 13}, weight="m2_weight")
    fd = M.fit_model(d, "y_m2", num + ["dec"], cat, weight="m2_weight")
    spec = ("MP FE" if fe else "no MP FE") + (" + controls" if ctrl else " raw")
    base = {"sample": label, "spec": spec, "n_questions": fl.n, "n_mps": int(d["mp_id"].nunique()),
            "g_mp": fl.g_sizes[0], "g_session": fl.g_sizes[1]}
    for ls in range(14, 19):
        c = fl.coef(f"ls[{ls}]")
        rows.append({**base, "contrast": f"LS{ls} minus LS13", "estimate": c["estimate"], "se": c["se"], "ci_lo": c["ci_lo"],
                     "ci_hi": c["ci_hi"], "p": c["p"]})
    c = fd.coef("dec")
    rows.append({**base, "contrast": "trend per decade", "estimate": c["estimate"], "se": c["se"], "ci_lo": c["ci_lo"],
                 "ci_hi": c["ci_hi"], "p": c["p"]})
    print(label, spec, fl.n, flush=True)


def main():
    df, est = load()
    res = est[est["resolved"]].copy()
    ls_by_mp = res.groupby("mp_id")["ls"].agg(lambda s: set(s))
    n_ls = ls_by_mp.map(len)
    multi = set(n_ls[n_ls >= 2].index)
    both = set(ls_by_mp[ls_by_mp.map(lambda s: bool(s & set(EARLY)) and bool(s & set(LATE)))].index)
    samples = [
        ("S_all: all labelled questions (incl. unresolved MP id)", est, False),
        ("S_res: resolved mpsno (baseline)", res, True),
        ("S_multi: MPs asking in 2+ Lok Sabhas", res[res["mp_id"].isin(multi)], True),
        ("S_both: MPs asking in both early (LS13 to 16) and late (LS17 or 18)", res[res["mp_id"].isin(both)], True),
    ]
    rows = []
    for label, d, fe_ok in samples:
        for ctrl in (False, True):
            run(d, label, False, ctrl, rows)
            if fe_ok:
                run(d, label, True, ctrl, rows)
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(OUT, "within_mp.csv"), index=False)

    # ---------- descriptives
    a = pd.read_parquet(C.ASKERS, columns=["qid", "ls", "mpsno"])
    ml = a.dropna(subset=["mpsno"]).groupby("mpsno")["ls"].nunique()
    n_multi_all = int((ml >= 2).sum())
    n_mp_all = int(len(ml))
    allq = df[df["analysis_sample"]].copy()
    allq["resolved"] = allq["mp_id"].astype(str).str.startswith("mp")
    ra = allq[allq["resolved"]]
    first_ls = ra.groupby("mp_id")["ls"].min()
    last_ls = ra.groupby("mp_id")["ls"].max()
    comp = []
    for ls in range(13, 19):
        q = allq[allq["ls"] == ls]
        r = q[q["resolved"]]
        comp.append({"LS": ls, "questions": len(q), "% resolved MP id": 100 * len(r) / len(q), "MPs asking": r["mp_id"].nunique(),
                     "% questions by MPs who asked in an earlier LS": 100 * (r["mp_id"].map(first_ls) < ls).mean(),
                     "% questions by MPs who ask again in a later LS": 100 * (r["mp_id"].map(last_ls) > ls).mean()})
    comp = pd.DataFrame(comp)
    l18 = df[df["ls"] == 18]
    cov = l18.groupby("session").agg(first=("date", "min"), last=("date", "max"), questions=("qid", "count")).reset_index()
    cov["labelled (M2)"] = cov["session"].map(est[est["ls"] == 18].groupby("session").size())
    cov["first"] = cov["first"].dt.strftime("%Y-%m-%d")
    cov["last"] = cov["last"].dt.strftime("%Y-%m-%d")

    # ---------- markdown
    def tab(sample_prefix, contrasts):
        sub = R[R["sample"].str.startswith(sample_prefix)]
        out = []
        for spec in ["no MP FE raw", "MP FE raw", "no MP FE + controls", "MP FE + controls"]:
            s = sub[sub["spec"] == spec].set_index("contrast")
            if s.empty:
                continue
            r = {"spec": spec, "N questions": int(s["n_questions"].iloc[0]), "N MPs": int(s["n_mps"].iloc[0])}
            for c in contrasts:
                r[c] = f3(s.loc[c, "estimate"], s.loc[c, "ci_lo"], s.loc[c, "ci_hi"])
            out.append(r)
        return pd.DataFrame(out)

    def share(sample_prefix, c):
        s = R[R["sample"].str.startswith(sample_prefix)].set_index(["spec", "contrast"])["estimate"]
        return {"raw": s[("MP FE raw", c)] / s[("no MP FE raw", c)],
                "controlled": s[("MP FE + controls", c)] / s[("no MP FE + controls", c)],
                "controlled_vs_noFE_raw": s[("MP FE + controls", c)] / s[("no MP FE raw", c)]}

    key = ["LS18 minus LS13", "trend per decade"]
    allc = [f"LS{k} minus LS13" for k in range(14, 19)] + ["trend per decade"]
    L = ["# Within-MP test of the decline (exploratory)\n",
         "Question: does the fall in the LLM question score reflect a change in WHO asks (composition of askers) rather than in how ministers answer?\n",
         "Method: outcome y_m2 (0 to 1), design weights m2_weight, WLS, two-way clustered SEs (first-asker MP x session), t(min G - 1) CIs, "
         "exactly as explore_rq2.py (task 2). LS contrasts use LS dummies (reference LS13); the trend uses (year - 2000) / 10. Controls = the S5 "
         "specification of explore_rq2.py (n_parts, log question length, starred, request-type shares, ministry and topic fixed effects). "
         "MP fixed effects = dummies for mpsno of the first asker (multi-asker questions are assigned to the first asker, the project convention). "
         "Estimates are in score units, 95% CIs in brackets.\n",
         f"Sample: {len(est):,} labelled questions; {len(res):,} ({100*len(res)/len(est):.1f}%) have a resolved mpsno for the first asker "
         f"({res['mp_id'].nunique():,} MPs). Across the whole corpus (all askers) {n_multi_all} of {n_mp_all:,} MPs with a resolved mpsno appear in two or more Lok Sabhas. "
         f"In the labelled sample, {len(multi):,} MPs (first asker) ask in 2+ Lok Sabhas and {len(both):,} ask in both an early (LS13 to LS16) and a late (LS17 or LS18) Lok Sabha.\n",
         "Identification note: with MP fixed effects the Lok Sabha contrasts and the trend are identified only from MPs observed in several Lok Sabhas "
         "(an MP seen in one Lok Sabha only absorbs his or her own level and contributes nothing to the contrasts). In the raw model the point "
         "estimates on S_res and S_multi are therefore identical in principle; the controlled model differs slightly because the control coefficients also use "
         "within MP variation of single Lok Sabha MPs. The informative comparison is always the no FE estimate on the SAME sample (rows 'no MP FE').\n",
         "## 1 to 3. Baseline and MP fixed effects on the resolved mpsno sample\n",
         "FACTS reference (all labelled questions): raw LS18 minus LS13 -0.080 [-0.097, -0.063], trend -0.026 [-0.034, -0.018]; controlled -0.051 [-0.063, -0.038], trend -0.013 [-0.020, -0.006].\n",
         "### S_all (all labelled questions, no MP FE; replication of FACTS)\n", C.df_to_markdown(tab("S_all", key)),
         "### S_res (resolved mpsno), key contrasts\n", C.df_to_markdown(tab("S_res", key)),
         "### S_res, every Lok Sabha against LS13 and trend\n", C.df_to_markdown(tab("S_res", allc)),
         "## 4. Restricted to MPs observed in several Lok Sabhas\n",
         "### S_multi (MPs asking in 2+ Lok Sabhas)\n", C.df_to_markdown(tab("S_multi", allc)),
         "### S_both (MPs asking in both early LS13 to LS16 and late LS17 or LS18)\n", C.df_to_markdown(tab("S_both", allc)),
         "## Share of the decline that remains within MP (ratio of point estimates)\n"]
    srows = []
    for pref in ("S_res", "S_multi", "S_both"):
        for c in key:
            s = share(pref, c)
            srows.append({"sample": pref, "contrast": c, "MP FE raw / no FE raw": s["raw"],
                          "MP FE controlled / no FE controlled": s["controlled"],
                          "MP FE controlled / no FE raw": s["controlled_vs_noFE_raw"]})
    L += [C.df_to_markdown(pd.DataFrame(srows), "{:.3f}"),
          "The first ratio is the share of the raw decline that survives holding the identity of the asker fixed. The second is the share of the "
          "controlled decline that survives; the third compares the fully adjusted within MP estimate with the raw no FE estimate (all on the same sample).\n",
          "## Composition of askers by Lok Sabha (all analysed questions, resolved first asker)\n", C.df_to_markdown(comp, "{:.1f}"),
          "## LS18 session coverage\n",
          f"LS18 sessions present in the data: {', '.join(str(int(s)) for s in cov['session'])} ({len(cov)} sessions with questions; "
          f"session 1 in June 2024 was the oath and Speaker election session without a question hour). "
          f"Date range of questions: {cov['first'].min()} to {cov['last'].max()}. {len(l18):,} questions in the corpus, {int((est['ls']==18).sum()):,} labelled.\n",
          C.df_to_markdown(cov), "Full estimate table: results/explore/within_mp.csv.\n"]
    with open(os.path.join(OUT, "within_mp.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))
    print("done")


if __name__ == "__main__":
    main()
