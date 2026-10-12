"""Pre-specified robustness checks (study plan section 6) + placebo contrasts (section 7).

    python -I -X utf8 analysis/robustness.py --outcome synthetic [--y y_m2] [--family ols] [--skip-ministries]

Every variant re-estimates the four primary estimates (H1, H2 key LS17-vs-LS18, H3, H4 key) with models.run_hypotheses:
    baseline | fractional logit | outcome = share FULL | outcome = M1 cosine | M1 part coverage | second embedding model
    | exclude grouped answers | exclude annexure-only (M2 label-based) | exclude every answer with an annexure
    | exclude statement-laid answers | starred only | unstarred only | strict party coding (asker_bloc_strict)
    | drop questions inside split-party windows | pure blocs only (government vs opposition askers)
    | single-asker questions only (no multi-asker clustering issue) | no topic FE | full ministry FE (H3 not identified)
    | + starred and minister rank | drop-one-ministry (each canonical ministry with >= 1,000 sample questions)
    | [14] unweighted M2 models (H1/H3) | [15] M1 outcome for H1/H3 on (a) the full corpus vs (b) the M2 sample with
    its design weights (plan §4.6 items 14-15; only when the outcome has M2 design weights, §4.3a)
    | [16] exclude answers transcribed by OCR from scanned images (answer_ocr)
    | [12] exclude questions whose answer refers to an annexure with no annexure text (has_annexure & annexure_len == 0;
      text-based flag, not outcome-based; plan §4.6 item 12). The frame is question-level and an annexure belongs to the
      whole answer, so excluding its parts = excluding the question.
[13] Mis-split parts (plan §4.6 item 13) can only be applied where human notes exist = the GOLD sample, so it is a PPI++
variant (ppi_variants): PPI++ re-estimated after dropping every gold question with any coder / adjudicator note matching
'split' or 'garbled' (case-insensitive substring; data/coding/human_labels.csv + adjudication.csv), reported next to the
PPI++ baseline run on the same code path. The M2-only models cannot use it (no notes outside the gold sample).
M2 outcomes use the M2 design weights (models.design_weight) in every variant unless stated otherwise.
Placebo: consecutive Lok Sabhas with the SAME majority type (13-14, 14-15 coalition-dependent; 16-17 single-party majority)
plus the within-LS17 / within-LS18 time trend. A large placebo contrast (or trend) means "time trends" can mimic H2.
Estimates are also reported in outcome-SD units (est_sd) so variants with different outcome scales are comparable.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import common as C  # noqa: E402
import models as M  # noqa: E402

STRICT = ("opp_strict", "mixed_strict", "outside_strict")
NOTE_FLAG_RE = r"split|garbled"          # plan §4.6 item 13 (case-insensitive substring)


def _run(df, y, family, name, **kw):
    """kw go to models.run_hypotheses; weight defaults to the pre-registered design weight for y; hyps keeps only
    those hypotheses."""
    kw = dict(kw)
    hyps = kw.pop("hyps", None)
    kw.setdefault("weight", M.design_weight(df, y))
    try:
        prim, _, _ = M.run_hypotheses(df, y, family, secondary=False, **kw)
    except Exception as e:  # keep the loop going; record failure
        return pd.DataFrame([{"variant": name, "hypothesis": "ERR", "term": str(e)[:80]}])
    s = M.estimation_sample(df, y, weight=kw["weight"])
    if kw["weight"] is None:
        sd = s[y].std()
    else:
        sw = s[kw["weight"]]
        sd = np.sqrt(np.average((s[y] - np.average(s[y], weights=sw)) ** 2, weights=sw))
    prim = prim[["hypothesis", "term", "estimate", "se", "ci_lo", "ci_hi", "p", "n", "weights"]].copy()
    if hyps:
        prim = prim[prim["hypothesis"].isin(hyps)]
    prim.insert(0, "variant", name)
    prim["est_sd"] = prim["estimate"] / sd
    prim["outcome"] = y
    return prim


def unseen_annexure(df):
    """Plan §4.6 item 12: answer refers to an annexure (`has_annexure`) but no annexure text is available (annexure_len == 0),
    so neither coders nor Gemini could see it. Text-based flag, independent of any label."""
    return df["has_annexure"].fillna(False).astype(bool) & (df["annexure_len"].fillna(0) <= 0)


def gold_note_flagged_qids(coding_dir=None, pattern=NOTE_FLAG_RE):
    """Plan §4.6 item 13: qids with ANY human note mentioning 'split' or 'garbled' (case-insensitive substring) in
    human_labels.csv (coder notes) or adjudication.csv (coder1/coder2/adjudicator notes). Returns (set of qids, counts)."""
    coding_dir = coding_dir or C.CODING
    qids, n_notes = set(), 0
    for fn, cols in (("human_labels.csv", ["note"]),
                     ("adjudication.csv", ["coder1_note", "coder2_note", "adjudicator_note"])):
        p = os.path.join(coding_dir, fn)
        if not os.path.exists(p):
            continue
        t = pd.read_csv(p)
        if "qid" not in t:
            t["qid"] = t["part_id"].str.rsplit("#", n=1).str[0]
        for c in (c for c in cols if c in t):
            hit = t[c].fillna("").astype(str).str.contains(pattern, case=False, regex=True)
            qids |= set(t.loc[hit, "qid"])
            n_notes += int(hit.sum())
    return qids, {"n_notes": n_notes, "n_questions": len(qids)}


def ppi_variants(df, y="y_m2", family="ols", flagged=None):
    """Plan §4.6 item 13 (gold sample only): PPI++ for H1-H4 on the full gold sample vs after dropping gold questions whose
    human notes mention 'split'/'garbled'. Flagged questions are removed from the PPI frame entirely (their M2 prediction
    and their rectifier term): the flag exists only for gold questions, so it cannot be applied to the population term.
    Returns a long DataFrame (variant x method x hypothesis); naive and gold-only rows are on the same rows."""
    import bias_correct as B
    if y not in M.M2_OUTCOMES:
        return pd.DataFrame()
    if flagged is None:
        flagged, _ = gold_note_flagged_qids()
    out = []
    for name, d in (("PPI++ baseline (all gold questions)", df),
                    ("[13] PPI++ excluding gold questions with 'split'/'garbled' notes", df[~df["qid"].isin(flagged)])):
        r = B.run_ppi_hypotheses(d, family, f_col=y)
        r.insert(0, "variant", name)
        r["n_flagged_gold_dropped"] = (int((df["qid"].isin(flagged) & df["gold_complete"] & df["gold_universe"]
                                            & df["analysis_sample"]).sum()) if name.startswith("[13]") else 0)
        out.append(r)
    return pd.concat(out, ignore_index=True)


PLAN_ITEMS = {   # plan §4.6 item -> substrings that must appear in a variant / contrast name
    1: ("outcome: M1 cosine", "outcome: M1 part coverage"), 2: ("outcome: second embedding model",),
    3: ("exclude grouped answers",), 4: ("starred only", "unstarred only"), 5: ("drop ministry:",),
    6: ("single-asker",), 7: ("strict party coding",), 8: ("drop split-party windows",),
    9: ("LS15 vs LS14", "LS17 vs LS16"), 10: ("within-LS17 trend", "within-LS18 trend"),
    11: ("fractional logit",), 12: ("[12]",), 13: ("[13]",), 14: ("[14]",), 15: ("[15a]", "[15b]"), 16: ("[16]",)}


def plan_coverage(names):
    """{item: missing substrings} for every plan §4.6 item ({} = all present). names: all variant, ministry, placebo and
    PPI variant labels of a run."""
    names = [str(n) for n in names]
    miss = {i: [t for t in toks if not any(t in n for n in names)] for i, toks in PLAN_ITEMS.items()}
    return {i: m for i, m in miss.items() if m}


def summarise_variants(res):
    """Per hypothesis over all variants (baseline excluded): number keeping the baseline's sign, min / max estimate,
    variants where the sign flips, number with two-sided p < .05. M1 / FULL-share variants have other outcome scales
    (see est_sd); they are included in the sign counts."""
    rows = []
    for h, x in res[res["hypothesis"].isin(["H1", "H2", "H3", "H4"])].groupby("hypothesis"):
        b = x[x["variant"] == "baseline"].iloc[0]
        v = x[x["variant"] != "baseline"]
        same = np.sign(v["estimate"]) == np.sign(b["estimate"])
        lo, hi = v.loc[v["estimate"].idxmin()], v.loc[v["estimate"].idxmax()]
        rows.append({"hypothesis": h, "baseline": b["estimate"], "baseline_p": b["p"], "n_variants": len(v),
                     "n_same_sign": int(same.sum()), "min": lo["estimate"], "min_variant": lo["variant"],
                     "max": hi["estimate"], "max_variant": hi["variant"],
                     "sign_flips": "; ".join(v.loc[~same, "variant"]) or "none", "n_p_lt_05": int((v["p"] < 0.05).sum())})
    return pd.DataFrame(rows)


def standard_variants(df, y, family):
    ann_m2 = df["annexure_only_m2"] if "annexure_only_m2" in df else pd.Series(False, index=df.index)
    V = [("baseline", df, y, family, {}),
         ("fractional logit (AME for H1,H2; logit scale H3,H4)", df, y, "fracit", {}),
         ("exclude grouped answers", df[~df["grouped_answer"]], y, family, {}),
         ("exclude annexure-only (M2 label-based)", df[~ann_m2.fillna(False).astype(bool)], y, family, {}),
         ("exclude all answers with an annexure", df[~df["has_annexure"]], y, family, {}),
         ("exclude 'statement laid' answers", df[~df["statement_laid"]], y, family, {}),
         ("[12] exclude unseen annexures (has_annexure, no annexure text)", df[~unseen_annexure(df)], y, family, {}),
         ("starred only", df[df["starred"] == 1], y, family, {}),
         ("unstarred only", df[df["starred"] == 0], y, family, {}),
         ("strict party coding (asker_bloc_strict)", df, y, family, {"bloc": STRICT}),
         ("drop split-party windows", df[~df["any_split_party_window"]], y, family, {}),
         ("pure blocs only (gov vs opp askers)", df[df["asker_bloc"].isin(["government", "opposition"])], y, family,
          {"bloc": ("opp",)}),
         ("single-asker questions only", df[df["n_askers"] == 1], y, family, {}),
         ("no topic FE", df, y, family, {"topic_fe": False}),
         ("full ministry FE (H3 not identified)", df, y, family,
          {"extra_cat": ("ministry_canonical",), "drop_cat": ("ministry_group",)}),
         ("[16] exclude OCR-transcribed answers", df[~df["answer_ocr"]] if "answer_ocr" in df else df, y, family, {}),
         ("+ starred, minister rank", df.assign(minister_rank=df["minister_rank"].fillna("unknown")), y, family,
          {"extra_num": ("starred",), "extra_cat": ("minister_rank",)})]
    for col, nm in (("y_full", "outcome: share of FULL parts"), ("y_m1", "outcome: M1 cosine"),
                    ("y_m1_cov", "outcome: M1 part coverage"), ("y_m1b", "outcome: second embedding model")):
        if col != y and col in df and df[col].notna().any():
            V.append((nm, df, col, family if col == "y_full" else "ols", {}))
    wt = M.design_weight(df, y)
    if wt is not None:      # plan §4.6 items 14-15 (design weights of the M2 sample, §4.3a)
        V.append(("[14] M2 unweighted (ignores design weights; H1/H3)", df[df[wt] > 0], y, family,
                  {"weight": None, "hyps": ("H1", "H3")}))
        if "y_m1" in df and df["y_m1"].notna().any():
            V += [("[15a] M1 outcome, full corpus (unweighted; H1/H3)", df, "y_m1", "ols",
                   {"weight": None, "hyps": ("H1", "H3")}),
                  ("[15b] M1 outcome, M2 sample with design weights (H1/H3)", df, "y_m1", "ols",
                   {"weight": wt, "hyps": ("H1", "H3")})]
    return V


def drop_one_ministry(df, y, family, min_n=1000):
    d = M.estimation_sample(df, y, weight=M.design_weight(df, y))
    counts = d["ministry_canonical"].value_counts()
    out = []
    for m_ in counts[counts >= min_n].index:
        r = _run(df[df["ministry_canonical"] != m_], y, family, f"drop ministry: {m_}")
        out.append(r)
    return pd.concat(out, ignore_index=True)


def placebo_contrasts(df, y="y_m2", pairs=((13, 14), (14, 15), (16, 17), (15, 16), (17, 18)), cluster=("mp_id", "session_id"),
                      weight="design"):
    """Coefficient on 'later LS' between consecutive Lok Sabhas (no session FE), same controls as the H2 model.
    First three pairs keep the same majority type (placebos); (15,16) and (17,18) are the actual switches.
    weight: 'design' = models.design_weight(df, y) (pairs touching LS13-16 mix sampled and census questions)."""
    weight = M.design_weight(df, y) if weight == "design" else weight
    d = M.estimation_sample(df, y, need=("topic",), weight=weight)
    ctrl = M.controls(d)
    rows = []
    mt = d.groupby("ls")["majority_type"].first()
    for a, b in pairs:
        s = d[d["ls"].isin((a, b))].assign(late=lambda x: (x["ls"] == b).astype(int))
        f = M.fit_model(s, y, ["late"] + list(M.BLOC) + ctrl, ["ministry_group", "topic"], "ols", cluster, weight=weight)
        r = f.coef("late")
        r.update({"contrast": f"LS{b} vs LS{a}", "majority_a": mt[a], "majority_b": mt[b],
                  "same_majority_type": mt[a] == mt[b], "n": f.n, "g_session": f.g_sizes[1]})
        rows.append(r)
    for ls in (17, 18):    # within-LS time trend (per year since the LS began)
        s = d[d["ls"] == ls].assign(yrs=lambda x: x["year"] - x["year"].min())
        f = M.fit_model(s, y, ["yrs"] + list(M.BLOC) + ctrl, ["ministry_group", "topic"], "ols", cluster, weight=weight)
        r = f.coef("yrs")
        r.update({"contrast": f"within-LS{ls} trend per year", "majority_a": mt[ls], "majority_b": mt[ls],
                  "same_majority_type": True, "n": f.n, "g_session": f.g_sizes[1]})
        rows.append(r)
    return pd.DataFrame(rows)[["contrast", "majority_a", "majority_b", "same_majority_type", "term", "estimate", "se",
                               "ci_lo", "ci_hi", "p", "n", "g_session"]]


def run_all(df, y="y_m2", family="ols", ministries=True):
    parts = [_run(dd, col, fam, nm, **kw) for nm, dd, col, fam, kw in standard_variants(df, y, family)]
    res = pd.concat(parts, ignore_index=True)
    dom = drop_one_ministry(df, y, family) if ministries else pd.DataFrame()
    plc = placebo_contrasts(df, y)
    return res, dom, plc


def summarise_dom(dom):
    if dom.empty:
        return dom
    g = dom[dom["hypothesis"].isin(["H1", "H2", "H3", "H4"])]
    rows = []
    for h, x in g.groupby("hypothesis"):
        lo, hi = x.loc[x["estimate"].idxmin()], x.loc[x["estimate"].idxmax()]
        rows.append({"hypothesis": h, "n_drops": len(x), "min_estimate": lo["estimate"], "min_when": lo["variant"],
                     "max_estimate": hi["estimate"], "max_when": hi["variant"],
                     "share_same_sign_as_median": float((np.sign(x["estimate"]) == np.sign(x["estimate"].median())).mean()),
                     "share_p_lt_0.05": float((x["p"] < 0.05).mean())})
    return pd.DataFrame(rows)


def main():
    from build_frame import load_frame
    ap = argparse.ArgumentParser()
    ap.add_argument("--outcome", choices=C.OUTCOMES, default="synthetic")
    ap.add_argument("--y", default="y_m2")
    ap.add_argument("--family", choices=("ols", "fracit"), default="ols")
    ap.add_argument("--skip-ministries", action="store_true")
    a = ap.parse_args()
    df = load_frame(a.outcome)
    res, dom, plc = run_all(df, a.y, a.family, ministries=not a.skip_ministries)
    out = C.outdir("tables", a.outcome)
    tag = f"{a.y}_{a.family}"
    res.to_csv(os.path.join(out, f"robustness_{tag}.csv"), index=False)
    dom.to_csv(os.path.join(out, f"robustness_drop_one_ministry_{tag}.csv"), index=False)
    plc.to_csv(os.path.join(out, f"placebo_contrasts_{tag}.csv"), index=False)
    summ = summarise_dom(dom)
    ppi = ppi_variants(df, a.y, a.family) if "gold_complete" in df else pd.DataFrame()
    vsum = summarise_variants(res)
    miss = plan_coverage(list(res["variant"].unique()) + (list(dom["variant"].unique()) if len(dom) else [])
                         + list(plc["contrast"]) + (list(ppi["variant"].unique()) if len(ppi) else []))
    if not ppi.empty:
        ppi.to_csv(os.path.join(out, f"robustness_ppi_{tag}.csv"), index=False)
    vsum.to_csv(os.path.join(out, f"robustness_summary_{tag}.csv"), index=False)
    with open(os.path.join(out, f"robustness_{tag}.md"), "w", encoding="utf-8") as fh:
        fh.write(("**SYNTHETIC OUTCOME - NOT A RESULT**\n\n" if a.outcome == "synthetic" else "")
                 + "### Robustness (primary estimates H1-H4 under each variant)\n\n")
        w = res.pivot_table(index="variant", columns="hypothesis", values="estimate", sort=False)
        fh.write(C.df_to_markdown(w.reset_index(), floatfmt="{:.4f}"))
        fh.write("\n### Drop-one-ministry summary\n\n" + C.df_to_markdown(summ, floatfmt="{:.4f}"))
        fh.write("\n### Placebo contrasts and within-LS trends\n\n" + C.df_to_markdown(plc, floatfmt="{:.4f}"))
        fh.write("\n### Summary over variants (baseline excluded; drop-one-ministry reported separately)\n\n"
                 + C.df_to_markdown(vsum, floatfmt="{:.4f}"))
        if not ppi.empty:
            pp = ppi[ppi["method"].str.startswith("PPI")][["variant", "hypothesis", "estimate", "se", "ci_lo", "ci_hi", "p",
                                                           "lambda", "n", "n_gold", "n_flagged_gold_dropped"]]
            fh.write("\n### PPI++ variants (item 13; gold sample only)\n\n" + C.df_to_markdown(pp, floatfmt="{:.4f}"))
        fh.write("\n### Plan 4.6 coverage\n\n" + ("all items 1-16 present" if not miss else f"MISSING: {miss}") + "\n")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(res.pivot_table(index="variant", columns="hypothesis", values="estimate", sort=False).round(4).to_string())
    print(summ.round(4).to_string(index=False))
    print(plc.round(4).to_string(index=False))
    print(vsum.round(4).to_string(index=False))
    if not ppi.empty:
        print(ppi[ppi["method"].str.startswith("PPI")][["variant", "hypothesis", "estimate", "se", "p", "n", "n_gold",
                                                        "n_flagged_gold_dropped"]].round(4).to_string(index=False))
    print("plan items missing:", miss or "none")


if __name__ == "__main__":
    main()
