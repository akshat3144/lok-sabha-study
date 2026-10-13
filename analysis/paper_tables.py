"""Publication tables T1-T4 (+ appendix T3A) from the REAL data -> results/tables/paper_*.{csv,md,tex}.

    python -I -X utf8 analysis/paper_tables.py [--boot 1000]

paper_T1_corpus            corpus by Lok Sabha: questions/parts (starred, unstarred), M2-labelled share, gold n
paper_T2_validity          human reliability (kappa, alpha), M2 vs gold (accuracy, kappa, macro-F1, CIs), M1 vs gold (AUC);
                           recomputed with validate.py functions (same inputs, seed and exclusions as validation_report.md)
paper_T3_main_ols          H1-H4: M2 estimate, SE, 95% CI, one-sided p, Holm p, PPI++ estimate and CI (OLS)
paper_T3A_main_fracit      the same for the fractional logit (appendix)
paper_T4_descriptive       design-weighted mean M2 score and evasion-type distribution, by Lok Sabha and by ministry group

CSV holds full-precision numbers; the Markdown/LaTeX versions are rounded display copies. No estimate is recomputed
here: T3/T3A read results/tables/{main,bias_corrected}_hypotheses_y_m2_{ols,fracit}.csv. Design weights (m2_weight) are
applied in T4 exactly as in models.design_weight: weight-0 gold-only extras are dropped, parts inherit their question's
weight. No answer text is read or printed anywhere.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import common as C  # noqa: E402

# evasion types (codebook EVASION) -> display groups; rare types are pooled into "Other"
EVA_GROUPS = [("GENERIC", "Generic"), ("INCOMPLETE", "Incomplete"), ("DIFFERENT_QUESTION", "Different question"),
              ("NOT_MAINTAINED", "Not maintained"), ("JURISDICTION", "Jurisdiction"), ("DEFERRAL", "Deferral"),
              ("OTHER", "Other")]
EVA_POOL = {"OTHER": ["OTHER", "WITHHELD", "DOES_NOT_ARISE", "ANNEXURE_ONLY"]}
TAB = C.outdir("tables")


# ------------------------------------------------------------------------------------------------ writers
def _tex_escape(x):
    return str(x).replace("\\", r"\textbackslash{}").replace("&", r"\&").replace("%", r"\%").replace("_", r"\_").replace("#", r"\#")


def to_tex(disp, caption, note=""):
    cols = list(disp.columns)
    L = [r"\begin{table}[ht]", r"\centering", r"\small", rf"\caption{{{_tex_escape(caption)}}}",
         r"\begin{tabular}{l" + "r" * (len(cols) - 1) + "}", r"\toprule",
         " & ".join(_tex_escape(c) for c in cols) + r" \\", r"\midrule"]
    for row in disp.itertuples(index=False):
        L.append(" & ".join(_tex_escape(x).replace("[", "{[}") if isinstance(x, str) else str(x) for x in row) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    if note:
        L += [r"\par\footnotesize " + _tex_escape(note)]
    L += [r"\end{table}", ""]
    return "\n".join(L)


def emit(stem, title, num, disp, note=""):
    num.to_csv(os.path.join(TAB, stem + ".csv"), index=False)
    with open(os.path.join(TAB, stem + ".md"), "w", encoding="utf-8") as fh:
        fh.write(f"### {title}\n\n{C.df_to_markdown(disp)}\n")
        if note:
            fh.write(f"_{note}_\n")
    with open(os.path.join(TAB, stem + ".tex"), "w", encoding="utf-8") as fh:
        fh.write(to_tex(disp, title, note))
    print("wrote", os.path.join(TAB, stem + ".{csv,md,tex}"))


def f3(x, d=3, sign=False):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    return f"{x:+.{d}f}" if sign else f"{x:.{d}f}"


def fci(lo, hi, d=3, sign=True):
    if lo is None or hi is None or np.isnan(lo) or np.isnan(hi):
        return ""
    sg = "+" if sign else ""
    return f"[{lo:{sg}.{d}f}, {hi:{sg}.{d}f}]"


def fp(p):
    if p is None or np.isnan(p):
        return ""
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def pretty(s):
    return str(s).replace("_", " ")


# ------------------------------------------------------------------------------------------------ shared data
def load_real_frame():
    from build_frame import load_frame
    return load_frame("real")


def weighted_parts(df):
    """M2-labelled parts of the questions in the weighted estimation sample (model-plan population), with the
    question's design weight. Used by T4 and fig_evasion_time."""
    import models as M
    d = M.estimation_sample(df, "y_m2", weight="m2_weight")[["qid", "ls", "session_id", "date", "ministry_group", "m2_weight"]]
    m = pd.read_parquet(os.path.join(C.MEASURES, "m2_labels.parquet"), columns=["part_id", "qid", "label", "evasion_type", "status"])
    m = m[m["status"] == "ok"]
    p = m.merge(d, on="qid", how="inner")
    ev = p["evasion_type"].copy()
    for grp, members in EVA_POOL.items():
        ev = ev.where(~ev.isin(members), grp)
    p["eva_group"] = ev
    return p


# ------------------------------------------------------------------------------------------------ T1
def table_t1(df):
    q = pd.read_parquet(C.QUESTIONS, columns=["qid", "ls", "session", "qtype"])
    parts = pd.read_parquet(C.PARTS, columns=["qid"]).groupby("qid").size().rename("np")
    q = q.join(parts, on="qid")
    q["np"] = q["np"].fillna(1)
    m2 = pd.read_parquet(os.path.join(C.MEASURES, "m2_labels.parquet"), columns=["qid"])
    q["m2_parts"] = q["qid"].map(m2.groupby("qid").size()).fillna(0)
    q["m2_q"] = q["m2_parts"] > 0
    gold = pd.read_csv(os.path.join(C.CODING, "gold_labels.csv"), keep_default_na=False)
    gold = _gold_estimate_set(gold)
    q["gold_parts"] = q["qid"].map(gold.groupby("qid").size()).fillna(0)
    rows = []
    for ls, g in list(q.groupby("ls")) + [("Total", q)]:
        st = g["qtype"] == "starred"
        rows.append({
            "Lok Sabha": f"{ls}th" if ls != "Total" else "Total",
            "Sessions": int(g.groupby(["ls", "session"]).ngroups),
            "Questions": len(g), "Starred questions": int(st.sum()), "Unstarred questions": int((~st).sum()),
            "Parts": int(g["np"].sum()), "Starred parts": int(g.loc[st, "np"].sum()), "Unstarred parts": int(g.loc[~st, "np"].sum()),
            "M2-labelled questions": int(g["m2_q"].sum()), "% questions M2-labelled": 100 * g["m2_q"].mean(),
            "M2-labelled parts": int(g["m2_parts"].sum()), "% parts M2-labelled": 100 * g["m2_parts"].sum() / g["np"].sum(),
            "Gold parts": int(g["gold_parts"].sum()), "Gold questions": int((g["gold_parts"] > 0).sum())})
    num = pd.DataFrame(rows)
    disp = num.copy()
    for c in ("% questions M2-labelled", "% parts M2-labelled"):
        disp[c] = disp[c].map(lambda x: f"{x:.1f}")
    for c in disp.columns:
        if c not in ("Lok Sabha", "% questions M2-labelled", "% parts M2-labelled"):
            disp[c] = disp[c].map(lambda x: f"{int(x):,}")
    emit("paper_T1_corpus", "Table 1. Corpus description by Lok Sabha", num, disp,
         "Parts = parsed question parts (a question without parsed parts counts as one part). M2-labelled = Gemini M2 labels exist "
         "(LS17-18 census, 1-in-6 stratified sample of LS13-16, plus gold-only extras). Gold = human gold-label parts in the "
         "estimation strata (sanity stratum without answer text excluded), as used in the validation report.")
    return num


def _gold_estimate_set(gold):
    sample = pd.read_csv(os.path.join(C.CODING, "main_sample.csv"), keep_default_na=False)
    sample["in_estimate"] = sample["in_estimate"].astype(str).str.lower().eq("true")
    est = set(sample.loc[sample["in_estimate"], "qid"])
    g = gold[gold["label"].astype(str).str.len() > 0]
    return g[g["qid"].isin(est)]


# ------------------------------------------------------------------------------------------------ T2
def table_t2(boot=1000):
    import validate as V
    inp = V.load_inputs(C.ROOT, "main")
    human, sample, adj = inp["human"], inp["sample"], inp["adj"]
    est_q = set(sample.loc[sample["in_estimate"], "qid"])
    h_est = human[human["qid"].isin(est_q)]
    r = V.reliability(h_est, None, B=boot)
    rows = []

    def add(panel, comp, n, metric, est, lo=np.nan, hi=np.nan, note=""):
        rows.append({"Panel": panel, "Comparison": comp, "n parts": n, "Metric": metric, "Estimate": est,
                     "CI lo": lo, "CI hi": hi, "Note": note})

    A = "A. Human inter-coder reliability (double-coded estimate parts)"
    for fld, nm in (("label", "Responsiveness label (FULL/PARTIAL/NONE)"), ("evasion_type", "Evasion type"),
                    ("request_type", "Request type")):
        v = r["fields"][fld]
        add(A, nm, v["n_parts"], "% agreement", 100 * v["pct_agree"])
        add(A, nm, v["n_parts"], "Cohen's kappa", v["kappa"])
        if "kappa_lw" in v and not np.isnan(v["kappa_lw"]):
            add(A, nm, v["n_parts"], "Linear-weighted kappa", v["kappa_lw"])
        add(A, nm, v["n_parts"], "Krippendorff alpha (nominal)", v["alpha_nominal"], *v["alpha_nominal_ci"])
        if "alpha_ordinal" in v and not np.isnan(v.get("alpha_ordinal", np.nan)):
            add(A, nm, v["n_parts"], "Krippendorff alpha (ordinal)", v["alpha_ordinal"], *v["alpha_ordinal_ci"])
    if "evasion_both_nonfull" in r:
        v = r["evasion_both_nonfull"]
        add(A, "Evasion type (both coders non-FULL)", v["n_parts"], "Cohen's kappa", v["kappa"])
        add(A, "Evasion type (both coders non-FULL)", v["n_parts"], "Krippendorff alpha (nominal)", v["alpha_nominal"], *v["alpha_nominal_ci"])
    # gold (same construction and exclusions as validation_report.md)
    gold, _ = V.build_gold(human, adj, sample=sample)
    gold = gold.merge(sample.drop_duplicates("qid")[["qid", "in_estimate", "ls", "ministry_group"]], on="qid", how="left")
    gold = gold[gold["in_estimate"].fillna(True) == True]  # noqa: E712
    B = "B. M2 (Gemini) vs gold labels"
    tab, confs = V.evaluate_m2(gold, inp["m2"], sample, B=min(boot, 500))
    t = tab.iloc[0]
    C_ = confs[t["model_id"]].values
    C_nf = int(C_[:2].sum())     # gold NONE + PARTIAL rows (labels ordered NONE, PARTIAL, FULL)
    add(B, "M2 label vs gold label", int(t["n_parts"]), "Accuracy", t["accuracy"], t["accuracy_ci_lo"], t["accuracy_ci_hi"])
    add(B, "M2 label vs gold label", int(t["n_parts"]), "Cohen's kappa", t["kappa"], t["kappa_ci_lo"], t["kappa_ci_hi"])
    add(B, "M2 label vs gold label", int(t["n_parts"]), "Linear-weighted kappa", t["kappa_lw"])
    add(B, "M2 label vs gold label", int(t["n_parts"]), "Macro-F1", t["macro_f1"])
    add(B, "M2 label vs gold label", int(t["n_parts"]), "Design-weighted accuracy", t["accuracy_weighted"])
    add(B, "M2 label vs gold label", int(t["n_parts"]), "Design-weighted kappa", t["kappa_weighted"])
    add(B, "M2 evasion type vs gold (gold non-FULL parts)", int(C_nf), "Accuracy", t["evasion_acc_nonfull"],
        note="share of gold non-FULL parts with the same evasion type")
    for i, lab in enumerate(["NONE", "PARTIAL", "FULL"]):
        add(B, f"M2 vs gold, gold = {lab}", int(C_[i].sum()), "Share labelled correctly by M2", C_[i, i] / C_[i].sum(),
            note="per-class recall")
    Cc = "C. M1 (embedding similarity) vs gold labels"
    e = V.evaluate_m1(gold, inp["m1"], sample, B=min(boot, 500))
    add(Cc, "M1 cosine, gold FULL vs NONE", e["n_full"] + e["n_none"], "AUC", e["auc_full_vs_none"], *e["auc_full_vs_none_ci"])
    add(Cc, "M1 cosine, gold FULL vs rest", e["n_parts"], "AUC", e["auc_full_vs_rest"])
    add(Cc, "M1 cosine vs gold label score", e["n_parts"], "Spearman rho", e["spearman_label_score"], *e["spearman_ci"])
    num = pd.DataFrame(rows)
    disp = num.drop(columns=["Note"]).copy()
    disp["n parts"] = disp["n parts"].map(lambda x: f"{int(x):,}")
    disp["95% CI"] = [fci(a, b, sign=False) for a, b in zip(num["CI lo"], num["CI hi"])]
    disp["Estimate"] = [f"{x:.1f}" if m == "% agreement" else f"{x:.3f}" for x, m in zip(num["Estimate"], num["Metric"])]
    disp = disp.drop(columns=["CI lo", "CI hi"])
    emit("paper_T2_validity", "Table 2. Measurement validity", num, disp,
         f"Recomputed with validate.py (bootstrap B={boot} for reliability, {min(boot, 500)} for M2/M1; cluster = question). "
         "Reliability: parts coded by both coders in the estimation strata. Gold: adjudicated/agreed/single-coder labels, sanity "
         "stratum (no answer text) excluded. Evasion-type agreement is not weighted. Class recall uses gold-label rows of the "
         "M2 confusion matrix.")
    return num, t


# ------------------------------------------------------------------------------------------------ T3
HYP_LABEL = {"H1": "Opposition vs government askers", "H2": "Coalition-dependent vs majority (LS18 vs LS17)",
             "H3": "Security vs other ministries", "H4": "Opposition x coalition-dependent (LS18 vs LS17)"}
HYP_EXPECT = {"H1": "-", "H2": "+", "H3": "-", "H4": "+"}


def table_t3(family):
    main = pd.read_csv(os.path.join(TAB, f"main_hypotheses_y_m2_{family}.csv"))
    bias = pd.read_csv(os.path.join(TAB, f"bias_corrected_hypotheses_y_m2_{family}.csv"))
    rows, disp = [], []
    for h in ("H1", "H2", "H3", "H4"):
        m = main[main["hypothesis"] == h].iloc[0]
        nv = bias[(bias["hypothesis"] == h) & bias["method"].str.startswith("naive")].iloc[0]
        pp = bias[(bias["hypothesis"] == h) & bias["method"].str.startswith("PPI")].iloc[0]
        scale = "marginal effect (score units)" if (family == "ols" or "AME" in str(m["term"])) else "logit coefficient"
        sample = "LS13-18, design-weighted" if m["weights"] != "none" else "LS17-18 census, unweighted"
        rows.append({"hypothesis": h, "contrast": HYP_LABEL[h], "expected_sign": HYP_EXPECT[h], "m2_scale": scale, "sample": sample,
                     "n": m["n"], "m2_estimate": m["estimate"], "m2_se": m["se"], "m2_ci_lo": m["ci_lo"], "m2_ci_hi": m["ci_hi"],
                     "m2_p_two_sided": m["p"], "m2_p_one_sided": m["p_one_sided"], "m2_holm_p_one_sided": m["p_holm_one_sided"],
                     "m2_holm_p_two_sided": m["p_holm"], "direction_as_expected": bool(m["direction_ok"]),
                     "gold_n": int(pp["n_gold"]), "ppi_n": pp["n"], "ppi_lambda": pp["lambda"],
                     "naive_gold_universe_estimate": nv["estimate"], "naive_gold_universe_se": nv["se"],
                     "ppi_estimate": pp["estimate"], "ppi_se": pp["se"], "ppi_ci_lo": pp["ci_lo"], "ppi_ci_hi": pp["ci_hi"],
                     "ppi_p_two_sided": pp["p"], "ppi_holm_p_two_sided": pp["p_holm"], "ppi_scale": "logit coefficient" if family == "fracit" else "marginal effect (score units)"})
        r = rows[-1]
        if family == "ols":
            disp.append({"Hypothesis": f"{h} {HYP_LABEL[h]}", "Exp. sign": HYP_EXPECT[h], "N": f"{int(m['n']):,}",
                         "M2 estimate": f3(r["m2_estimate"], 3, True), "SE": f3(r["m2_se"]), "95% CI": fci(r["m2_ci_lo"], r["m2_ci_hi"]),
                         "p (1-sided)": fp(r["m2_p_one_sided"]), "Holm p (1-sided)": fp(r["m2_holm_p_one_sided"]),
                         "Gold n": r["gold_n"], "PPI++ estimate": f3(r["ppi_estimate"], 3, True),
                         "PPI++ 95% CI": fci(r["ppi_ci_lo"], r["ppi_ci_hi"]), "PPI++ p (2-sided, Holm)": fp(r["ppi_holm_p_two_sided"])})
        else:
            disp.append({"Hypothesis": f"{h} {HYP_LABEL[h]}", "Exp. sign": HYP_EXPECT[h], "N": f"{int(m['n']):,}",
                         "M2 estimate": f3(r["m2_estimate"], 3, True), "Scale": "AME" if scale.startswith("marginal") else "logit",
                         "SE": f3(r["m2_se"]), "95% CI": fci(r["m2_ci_lo"], r["m2_ci_hi"]),
                         "p (1-sided)": fp(r["m2_p_one_sided"]), "Holm p (1-sided)": fp(r["m2_holm_p_one_sided"]),
                         "Gold n": r["gold_n"], "M2 on gold universe (logit)": f3(r["naive_gold_universe_estimate"], 3, True),
                         "PPI++ (logit)": f3(r["ppi_estimate"], 3, True),
                         "PPI++ 95% CI": fci(r["ppi_ci_lo"], r["ppi_ci_hi"]), "PPI++ p (2-sided, Holm)": fp(r["ppi_holm_p_two_sided"])})
    num, disp = pd.DataFrame(rows), pd.DataFrame(disp)
    if family == "ols":
        emit("paper_T3_main_ols", "Table 3. Main results, H1-H4 (OLS, outcome = mean M2 part score, 0-1)", num, disp,
             "Estimates are differences in mean part score (FULL=1, PARTIAL=0.5, NONE=0), two-way clustered by MP and session, 95% CI "
             "two-sided. One-sided p tests the pre-specified direction (see Exp. sign); Holm over H1-H4 (one-sided). H1 and H3 pool "
             "LS13-18 with session fixed effects and design weights; H2 and H4 are the LS17 vs LS18 contrasts (census, unweighted). "
             "PPI++ = bias-corrected estimate using the gold sample (lambda tuned, two-way clustered); its Holm p is two-sided within method. "
             "Gold n = gold questions behind PPI++ in that specification.")
    else:
        emit("paper_T3A_main_fracit", "Table A1. Main results, H1-H4 (fractional logit, appendix)", num, disp,
             "Fractional logit with the same specifications and clustering as Table 3. M2 estimates for H1 and H2 are average marginal "
             "effects (score units); H3 and H4 are logit-scale coefficients (H4 is the interaction). PPI++ is estimated on the logit scale "
             "throughout; the logit-scale M2 estimate on the gold universe is shown for comparison. One-sided p and Holm as in Table 3.")
    return num


# ------------------------------------------------------------------------------------------------ T4
def _wmean_se(y, w):
    y, w = np.asarray(y, float), np.asarray(w, float)
    m = (w * y).sum() / w.sum()
    n = len(y)
    se = np.sqrt(((w * (y - m)) ** 2).sum() * n / max(n - 1, 1)) / w.sum()
    return m, se


def table_t4(df):
    import models as M
    d = M.estimation_sample(df, "y_m2", weight="m2_weight")
    p = weighted_parts(df)
    rows = []

    def one(panel, name, dq, dp):
        m, se = _wmean_se(dq["y_m2"], dq["m2_weight"])
        w = dp["m2_weight"]
        lab = dp.groupby("label")["m2_weight"].sum() / w.sum()
        nf = dp[dp["label"] != "FULL"]
        ev = nf.groupby("eva_group")["m2_weight"].sum() / nf["m2_weight"].sum()
        r = {"Panel": panel, "Group": name, "Questions (n)": len(dq), "Parts (n)": len(dp), "Mean M2 score": m, "SE": se,
             "% parts FULL": 100 * lab.get("FULL", 0), "% parts PARTIAL": 100 * lab.get("PARTIAL", 0), "% parts NONE": 100 * lab.get("NONE", 0)}
        for k, nm in EVA_GROUPS:
            r[f"% of PARTIAL/NONE: {nm}"] = 100 * ev.get(k, 0.0)
        rows.append(r)

    one("All", "LS13-18", d, p)
    for ls in sorted(d["ls"].unique()):
        one("Lok Sabha", f"{ls}th", d[d["ls"] == ls], p[p["ls"] == ls])
    order = d.groupby("ministry_group")["m2_weight"].sum().sort_values(ascending=False).index
    for g in order:
        one("Ministry group", pretty(g), d[d["ministry_group"] == g], p[p["ministry_group"] == g])
    num = pd.DataFrame(rows)
    disp = num.copy()
    disp["Questions (n)"] = disp["Questions (n)"].map(lambda x: f"{x:,}")
    disp["Parts (n)"] = disp["Parts (n)"].map(lambda x: f"{x:,}")
    disp["Mean M2 score"] = disp["Mean M2 score"].map(lambda x: f"{x:.3f}")
    disp["SE"] = disp["SE"].map(lambda x: f"{x:.3f}")
    for c in disp.columns:
        if c.startswith("%"):
            disp[c] = disp[c].map(lambda x: f"{x:.1f}")
    emit("paper_T4_descriptive", "Table 4. Descriptive responsiveness (design-weighted, M2 labels)", num, disp,
         "Design-weighted (m2_weight) over the estimation sample: mean M2 question score (mean of part scores FULL=1, PARTIAL=0.5, NONE=0) "
         "with a linearised, unclustered SE (descriptive; inference uses Table 3). Part shares are weighted by the question's design weight. "
         "Evasion columns: distribution of the M2 evasion type over PARTIAL and NONE parts (each row sums to 100); Other pools "
         "OTHER, WITHHELD, DOES_NOT_ARISE and ANNEXURE_ONLY. LS columns include the 1-in-6 sample of LS13-16 weighted up.")
    return num


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--only", default="t1,t2,t3,t4")
    a = ap.parse_args()
    want = set(a.only.split(","))
    df = load_real_frame() if want & {"t1", "t4"} else None
    if "t1" in want:
        table_t1(df)
    if "t2" in want:
        table_t2(a.boot)
    if "t3" in want:
        table_t3("ols")
        table_t3("fracit")
    if "t4" in want:
        table_t4(df)


if __name__ == "__main__":
    main()
