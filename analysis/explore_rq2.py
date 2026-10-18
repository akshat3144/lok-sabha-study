"""Exploratory RQ2: is the decline in responsiveness (share of parts not fully answered, y_m2) real?

    python -I -X utf8 analysis/explore_rq2.py [--tasks 1,2,3,4] [--boot 2000]

Writes results/explore/rq2_trend.md (+ csv tables, 3 figures). Statistics only (no answer text is quoted; the optional
deferral-marker check in task 3 only computes regex flags on answer text and stores counts).
Does NOT modify any pipeline file / data file / frame. Reuses models.fit_model and bias_correct.ppi_fit.

Task 1  human gold vs M2 by LS / period: design-weighted gold means, M2 accuracy / kappa / bias, PPI / PPI++ means.
        Gold statistics use the 400 'in_estimate' gold questions with HT weights 1/gold_pi (Hajek means) and a Rao-Wu
        rescaled bootstrap (strata from main_sample.csv; singleton strata collapsed within LS), so all gold CIs are
        design-based percentile CIs.
Task 2  complexity: regressions of y_m2 on LS dummies / decade with controls (fit_model, weights m2_weight, MP x session
        two-way clustered SEs) + descriptive trends in n_parts / question length / request types.
Task 3  composition of evasion types (design-weighted M2 parts, session-cluster bootstrap) + gold-based shares by period
        + an independent text-marker check of the DEFERRAL formula.
Task 4  within-government trends: session means, segmented regressions at 2004 / 2014 / 2024 (+ placebo 2009 / 2019).
"""
import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

import bias_correct as B  # noqa: E402
import common as C  # noqa: E402
import models as M  # noqa: E402

OUT = os.path.join(C.RESULTS, "explore")
os.makedirs(OUT, exist_ok=True)
SEED = 20261018
LAB = ["FULL", "PARTIAL", "NONE"]
SCORE = {"FULL": 1.0, "PARTIAL": 0.5, "NONE": 0.0}
PERIOD = {13: "early (LS13-14, 1999-2009)", 14: "early (LS13-14, 1999-2009)", 15: "middle (LS15-16, 2009-2019)",
          16: "middle (LS15-16, 2009-2019)", 17: "late (LS17-18, 2019-2026)", 18: "late (LS17-18, 2019-2026)"}
PERIODS = ["early (LS13-14, 1999-2009)", "middle (LS15-16, 2009-2019)", "late (LS17-18, 2019-2026)"]
EVA = ["GENERIC", "INCOMPLETE", "DIFFERENT_QUESTION", "NOT_MAINTAINED", "JURISDICTION", "DEFERRAL", "OTHER"]
EVA_POOL = {"OTHER": ["OTHER", "WITHHELD", "DOES_NOT_ARISE", "ANNEXURE_ONLY"]}
GOV_CHANGES = {"2004 (NDA-V3 -> UPA-1)": pd.Timestamp("2004-07-05"), "2014 (UPA-2 -> NDA-M1)": pd.Timestamp("2014-07-07"),
               "2024 (NDA-M2 -> NDA-M3)": pd.Timestamp("2024-07-22")}
PLACEBO = {"2009 (UPA-1 -> UPA-2, same govt)": pd.Timestamp("2009-07-02"),
           "2019 (NDA-M1 -> NDA-M2, same govt)": pd.Timestamp("2019-06-21")}
TAB = {}          # name -> DataFrame (also written to csv)
KEY = {}          # headline numbers for the final report


def fmt(e, lo, hi, d=3):
    return f"{e:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]"


def save(name, df):
    TAB[name] = df
    df.to_csv(os.path.join(OUT, f"rq2_{name}.csv"), index=False)
    return df


def md(df, floatfmt="{:,.3f}"):
    return C.df_to_markdown(df, floatfmt=floatfmt)


def dec_year(ts):
    ts = pd.to_datetime(ts)
    return ts.dt.year + (ts.dt.dayofyear - 1) / 365.25


# =========================================================================================== data
MK_A = re.compile(r"being\s+(?:collected|compiled|gathered|collated)", re.I)
MK_B = re.compile(r"will\s+be\s+laid\s+(?:on|upon)\s+the\s+table", re.I)


def text_features():
    """Per-question text statistics (no text stored): question word count, whitespace-normalised length, deferral-formula regex flags,
    extraction source. Cached in results/explore/rq2_textfeat.parquet (derived; the data files are untouched)."""
    cp = os.path.join(OUT, "rq2_textfeat.parquet")
    if os.path.exists(cp):
        return pd.read_parquet(cp)
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(C.QUESTIONS)
    recs = []
    for b in pf.iter_batches(batch_size=20000, columns=["qid", "question_text", "answer_text", "source_q", "question_has_hindi", "answer_has_hindi"]):
        d = b.to_pandas()
        q = d["question_text"].fillna("")
        a = d["answer_text"].fillna("")
        recs.append(pd.DataFrame({"qid": d["qid"], "q_words": q.str.split().str.len(), "a_words": a.str.split().str.len(),
                                  "mkA": a.str.contains(MK_A), "mkB": a.str.contains(MK_B), "source_q": d["source_q"],
                                  "q_hindi": d["question_has_hindi"].astype(bool), "a_hindi": d["answer_has_hindi"].astype(bool)}))
    tf = pd.concat(recs, ignore_index=True)
    tf.to_parquet(cp)
    return tf


def load():
    df = pd.read_parquet(C.frame_path("real"))
    df = df.merge(text_features(), on="qid", how="left")
    df["log_words"] = np.log(df["q_words"].clip(lower=1))
    df["log_awords"] = np.log1p(df["a_words"])
    df["qlen_dec"] = pd.qcut(df["question_len"], 10, labels=False, duplicates="drop").astype(float)
    df["nparts_c"] = df["n_parts"].clip(upper=8)
    df["dec"] = (df["year"] - 2000.0) / 10.0
    est = df[df["analysis_sample"] & df["y_m2"].notna() & (df["m2_weight"] > 0)].copy()
    ms = pd.read_csv(os.path.join(C.CODING, "main_sample.csv"), keep_default_na=False)
    g = df[df["gold_complete"] & (df["gold_pi"] > 0)].merge(ms[["qid", "stratum"]], on="qid", how="left")
    g = g.sort_values("qid").reset_index(drop=True)
    g["w"] = 1.0 / g["gold_pi"]
    g["period"] = g["ls"].map(PERIOD)
    # strata: collapse singletons within LS, then merge a lone collapsed stratum into the biggest stratum of that LS
    st = g["stratum"].astype(str)
    cnt = st.map(st.value_counts())
    st = pd.Series(np.where(cnt == 1, "coll_" + g["ls"].astype(str), st), index=g.index)
    for ls in g["ls"].unique():
        k = g["ls"] == ls
        c = st[k].value_counts()
        for s, n in c.items():
            if n == 1:
                st[st == s] = c.index[0] if c.index[0] != s else c.index[1]
    g["strat"] = st
    assert (g["strat"].value_counts() >= 2).all()
    return df, est, g


def rep_weights(w, strata, B_, rng):
    """Rao-Wu rescaled bootstrap weights (B x n): per stratum draw n_h - 1 units with replacement."""
    n = len(w)
    R = np.zeros((B_, n))
    for h in np.unique(strata):
        idx = np.where(strata == h)[0]
        nh = len(idx)
        R[:, idx] = rng.multinomial(nh - 1, np.ones(nh) / nh, size=B_) * nh / (nh - 1)
    return R * w


def boot_summary(fn, w, Wrep, names):
    """fn(W: m x n) -> (m x k).  Returns DataFrame est, se, lo, hi (percentile) per output k."""
    est = fn(w[None, :])[0]
    reps = fn(Wrep)
    se = np.nanstd(reps, axis=0, ddof=1)
    lo, hi = np.nanpercentile(reps, [2.5, 97.5], axis=0)
    return pd.DataFrame({"stat": names, "est": est, "se": se, "lo": lo, "hi": hi}), reps


def wmean(W, y):
    return (W * y).sum(1) / W.sum(1)


def kappa3(P, linear=False):
    P = P / P.sum()
    pr, pc = P.sum(1), P.sum(0)
    if linear:
        i, j = np.indices((3, 3))
        Wt = 1 - np.abs(i - j) / 2.0
        po, pe = (Wt * P).sum(), (Wt * np.outer(pr, pc)).sum()
    else:
        po, pe = np.trace(P), pr @ pc
    return (po - pe) / (1 - pe) if pe < 1 else np.nan


CONF_NAMES = ["accuracy", "kappa", "kappa_linear", "bias_part (M2-gold)", "gold_nonFULL_rate", "m2_nonFULL_rate",
              "sens_nonFULL", "spec_FULL"]


def conf_stats(c9):
    """c9: 9 weighted counts, index gold*3+m2, order FULL, PARTIAL, NONE."""
    P = c9.reshape(3, 3)
    tot = P.sum()
    score = np.array([1.0, 0.5, 0.0])
    gold_s = (P.sum(1) * score).sum() / tot
    m2_s = (P.sum(0) * score).sum() / tot
    gold_nf = P[1:].sum() / tot
    m2_nf = P[:, 1:].sum() / tot
    sens = P[1:, 1:].sum() / P[1:].sum()
    spec = P[0, 0] / P[0].sum()
    return np.array([np.trace(P) / tot, kappa3(P), kappa3(P, True), m2_s - gold_s, gold_nf, m2_nf, sens, spec])


# =========================================================================================== task 1
def task1(df, est, g, B_):
    rng = np.random.default_rng(SEED)
    out = ["## 1. Real or LLM drift?\n"]
    n = len(g)
    w = g["w"].to_numpy()
    Wrep = rep_weights(w, g["strat"].to_numpy(), B_, rng)
    y, f = g["y_gold"].to_numpy(), g["y_m2"].to_numpy()
    D = f - y
    # ---- part-level confusion per question
    gl = pd.read_csv(os.path.join(C.CODING, "gold_labels.csv"), keep_default_na=False)
    gl = gl[gl["qid"].isin(g["qid"]) & (gl["label"] != "")]
    ml = pd.read_parquet(os.path.join(C.MEASURES, "m2_labels.parquet"),
                         columns=["part_id", "label", "evasion_type", "status", "request_type"])
    ml = ml[ml["status"] == "ok"].rename(columns={"label": "m2_label", "evasion_type": "m2_eva", "request_type": "m2_rt"})
    pp = gl.merge(ml, on="part_id", how="left")
    miss = int(pp["m2_label"].isna().sum())
    pp = pp[pp["m2_label"].notna()].copy()
    qidx = pd.Series(np.arange(n), index=g["qid"])
    pp["qi"] = qidx.reindex(pp["qid"]).to_numpy()
    pp["gi"] = pp["label"].map({k: i for i, k in enumerate(LAB)})
    pp["mi"] = pp["m2_label"].map({k: i for i, k in enumerate(LAB)})
    Q = np.zeros((n, 9))
    np.add.at(Q, (pp["qi"].to_numpy(), (pp["gi"] * 3 + pp["mi"]).to_numpy()), 1)
    KEY["gold_parts_used"] = len(pp)
    out.append(f"Gold sample: {n} questions ({len(pp):,} parts) from the 400-question `in_estimate` set; {miss} gold parts lack an M2 "
               f"label (dropped). Weights 1/gold_pi (Hajek means). CIs: Rao-Wu rescaled bootstrap, B={B_}, strata = main_sample "
               f"strata (singletons collapsed within LS), 95% percentile. Estimates refer to the M3 universe (answered, 1-6 parts).\n")
    groups = {f"LS{ls}": (g["ls"] == ls).to_numpy() for ls in range(13, 19)}
    groups.update({p: (g["period"] == p).to_numpy() for p in PERIODS})
    groups["All LS (pop.-weighted)"] = np.ones(n, bool)

    def stat_fn(mask):
        def fn(W):
            Wm = W * mask
            conf = Wm @ Q
            cs = np.array([conf_stats(c) for c in conf])
            return np.column_stack([wmean(Wm, y), wmean(Wm, f), wmean(Wm, D), cs])
        return fn

    names = ["gold_mean", "m2_mean_on_gold_q", "bias_q (M2-gold)"] + CONF_NAMES
    rows, reps_all = [], {}
    for gname, mask in groups.items():
        sm, reps = boot_summary(stat_fn(mask), w, Wrep, names)
        reps_all[gname] = reps
        sm.insert(0, "group", gname)
        sm["n_q"] = int(mask.sum())
        sm["n_parts"] = int(pp["qi"].isin(np.where(mask)[0]).sum())
        rows.append(sm)
    S = pd.concat(rows, ignore_index=True)
    save("t1_gold_vs_m2_long", S)
    P_ = lambda gname, s: S[(S["group"] == gname) & (S["stat"] == s)].iloc[0]   # noqa: E731

    def cell(gname, s, d=3):
        r = P_(gname, s)
        return fmt(r["est"], r["lo"], r["hi"], d)

    # ---- table 1a: means
    t = []
    for gname in groups:
        r = {"group": gname, "n_q": P_(gname, "gold_mean")["n_q"], "n_parts": P_(gname, "gold_mean")["n_parts"]}
        for s in ("gold_mean", "m2_mean_on_gold_q", "bias_q (M2-gold)"):
            r[s] = cell(gname, s)
        t.append(r)
    t1a = pd.DataFrame(t)
    out.append("### 1a. Design-weighted human gold score vs M2 on the same questions (question score, mean over parts)\n")
    out.append(md(t1a) + "\n")
    # ---- table 1b: part-level agreement
    t = []
    for gname in groups:
        r = {"group": gname, "n_parts": P_(gname, "gold_mean")["n_parts"]}
        for s in CONF_NAMES:
            r[s] = cell(gname, s, 3)
        # unweighted
        mk = np.where(groups[gname])[0]
        c9 = Q[mk].sum(0)
        us = conf_stats(c9)
        r["accuracy_unweighted"] = f"{us[0]:.3f}"
        r["kappa_unweighted"] = f"{us[1]:.3f}"
        t.append(r)
    t1b = pd.DataFrame(t)
    out.append("### 1b. M2 (gemini-3.8-flash) vs human gold at the part level (design-weighted; unweighted point estimates in last two columns)\n")
    out.append(md(t1b) + "\n")
    out.append("bias_part = mean M2 part score minus mean gold part score (positive = M2 more lenient than humans). "
               "sens_nonFULL = P(M2 says not FULL | gold not FULL); spec_FULL = P(M2 says FULL | gold FULL).\n")
    save("t1_agreement", t1b)
    # ---- time-variation of bias: homogeneity tests and slopes
    ls_names = [f"LS{ls}" for ls in range(13, 19)]
    bq = np.column_stack([reps_all[k][:, names.index("bias_q (M2-gold)")] for k in ls_names])
    bp = np.column_stack([reps_all[k][:, names.index("bias_part (M2-gold)")] for k in ls_names])
    est_bq = np.array([P_(k, "bias_q (M2-gold)")["est"] for k in ls_names])
    est_bp = np.array([P_(k, "bias_part (M2-gold)")["est"] for k in ls_names])
    Cm = np.hstack([-np.ones((5, 1)), np.eye(5)])

    def wald(est_, reps_):
        d = Cm @ est_
        V = np.cov((reps_ @ Cm.T).T)
        stat = float(d @ np.linalg.solve(V, d))
        return stat, float(stats.chi2.sf(stat, 5))

    wq, pq = wald(est_bq, bq)
    wp, pp_ = wald(est_bp, bp)
    # slope per decade of bias, gold, M2 across all gold questions (weighted OLS on year)
    x = (g["year"].to_numpy() - 2000.0) / 10.0

    def slopes(W):
        xb = (W * x).sum(1, keepdims=True) / W.sum(1, keepdims=True)
        sxx = (W * (x - xb) ** 2).sum(1)
        return np.column_stack([(W * (x - xb) * (v - wmean(W, v)[:, None])).sum(1) / sxx for v in (y, f, D)])

    sl, _ = boot_summary(slopes, w, Wrep, ["gold slope /decade", "M2 slope /decade (gold q)", "bias slope /decade"])
    save("t1_slopes", sl)
    # early vs late contrasts
    cont = []
    for nm, (a, b) in {"LS18 - LS13": ("LS18", "LS13"), "LS17 - LS13": ("LS17", "LS13"),
                       "late - early": (PERIODS[2], PERIODS[0])}.items():
        for s in ("gold_mean", "m2_mean_on_gold_q", "bias_q (M2-gold)", "bias_part (M2-gold)", "accuracy", "kappa"):
            k = names.index(s)
            d_ = reps_all[a][:, k] - reps_all[b][:, k]
            e_ = P_(a, s)["est"] - P_(b, s)["est"]
            cont.append({"contrast": nm, "stat": s, "diff": fmt(e_, *np.percentile(d_, [2.5, 97.5]))})
    cont = pd.DataFrame(cont)
    save("t1_contrasts", cont)
    out.append("### 1c. Is Gemini's bias time-varying?\n")
    out.append(f"- Homogeneity of question-level bias across LS13-18 (bootstrap Wald, 5 df): chi2={wq:.2f}, p={pq:.3f}; "
               f"part-level bias: chi2={wp:.2f}, p={pp_:.3f}.\n"
               f"- Slopes per decade on the gold questions (weighted OLS on year):\n")
    sl2 = sl.copy()
    sl2["95% CI"] = [fmt(r.est, r.lo, r.hi) for r in sl.itertuples()]
    out.append(md(sl2[["stat", "95% CI"]]) + "\n")
    out.append("Contrasts (95% bootstrap CI):\n")
    out.append(md(cont) + "\n")
    KEY.update(bias_q_homog_p=pq, bias_part_homog_p=pp_)
    for s in sl.itertuples():
        KEY[s.stat] = fmt(s.est, s.lo, s.hi)

    # ---- PPI means
    dppi = B.prepare_ppi_frame(df, "y_m2", "y_gold", "m2_weight")
    out.append("### 1d. PPI / PPI++ corrected means (M3 universe)\n")
    out.append(f"`bias_correct.ppi_fit`, intercept-only per group, design weights m2_weight for the prediction term, 1/gold_pi for "
               f"the rectifier; two-way (MP x session) cluster SEs, t(sessions-1) CIs. n rows used={len(dppi):,}, gold={int(dppi['R'].sum())}; "
               f"sampled-but-unlabelled={dppi.attrs['n_sampled_unlabelled']}, gold-without-M2={dppi.attrs['n_gold_without_f']}. "
               f"'simple PPI' = design-weighted M2 mean over the universe + Hajek mean of (gold - M2) on gold (lambda=1), "
               f"SE = sqrt(session-cluster SE^2 + bootstrap SE^2).\n")
    rows = []
    gsets = {f"LS{ls}": dppi["ls"] == ls for ls in range(13, 19)}
    gsets.update({p: dppi["ls"].map(PERIOD) == p for p in PERIODS})
    gsets["All LS (pop.-weighted)"] = pd.Series(True, index=dppi.index)
    est_u = est[est["gold_universe"]]
    simple = {}
    for gname, mk in gsets.items():
        d = dppi[mk.to_numpy()]
        r1 = B.ppi_fit(d, "y_m2", "y_gold", "R", "pi", [], [], {"mean": {"const": 1.0}}, "ols", lam="opt", w_col="m2_weight")
        r2 = B.ppi_fit(d, "y_m2", "y_gold", "R", "pi", [], [], {"mean": {"const": 1.0}}, "ols", lam=1.0, w_col="m2_weight")
        rr = pd.concat([r1, r2[r2["method"].str.startswith("PPI")]], ignore_index=True)
        for _, r in rr.iterrows():
            rows.append({"group": gname, "method": r["method"], "lambda": r["lambda"], "est": r["estimate"], "se": r["se"],
                         "lo": r["ci_lo"], "hi": r["ci_hi"], "n_gold": r["n_gold"]})
        # simple PPI
        if gname == "All LS (pop.-weighted)":
            eu = est_u
        elif gname in PERIODS:
            eu = est_u[est_u["ls"].map(PERIOD) == gname]
        else:
            eu = est_u[est_u["ls"] == int(gname[2:])]
        fit = M.fit_model(eu, "y_m2", [], [], weight="m2_weight")
        fm, fse = fit.beta[0], fit.se[0]
        mk_g = groups[gname]
        mean_m2_pop = fm
        reps = reps_all[gname][:, names.index("bias_q (M2-gold)")]
        Dh = P_(gname, "bias_q (M2-gold)")["est"]
        e_simple = mean_m2_pop - Dh
        se_simple = np.sqrt(fse ** 2 + np.nanstd(reps, ddof=1) ** 2)
        q = stats.t.ppf(0.975, fit.df_t)
        rows.append({"group": gname, "method": "simple PPI (lambda=1, bootstrap rectifier)", "lambda": 1.0, "est": e_simple, "se": se_simple,
                     "lo": e_simple - q * se_simple, "hi": e_simple + q * se_simple, "n_gold": int(mk_g.sum())})
        gm_ = P_(gname, "gold_mean")
        rows.append({"group": gname, "method": "gold only (Hajek, bootstrap)", "lambda": 0.0, "est": gm_["est"], "se": gm_["se"],
                     "lo": gm_["lo"], "hi": gm_["hi"], "n_gold": int(mk_g.sum())})
        simple[gname] = (mean_m2_pop, fse, fit.df_t)
    PP = pd.DataFrame(rows)
    save("t1_ppi_means", PP)
    pv = PP.copy()
    pv["est [95% CI]"] = [fmt(r.est, r.lo, r.hi) for r in pv.itertuples()]
    pv["lambda"] = pv["lambda"].map(lambda v: "" if pd.isna(v) else f"{v:.2f}")
    piv = pv.pivot(index="group", columns="method", values="est [95% CI]").reindex(list(gsets))
    piv = piv[[c for c in ["naive (on predictions)", "gold only (Hajek, bootstrap)", "gold only (HT, lambda=0)", "PPI (lambda=1.0)", "PPI++ (lambda tuned)",
                           "simple PPI (lambda=1, bootstrap rectifier)"] if c in piv.columns]]
    out.append("gold only (HT, lambda=0) divides the HT total by the M2-sample estimate of N and is very noisy; the Hajek gold mean is the usable gold-only benchmark.\n")
    out.append(md(piv.reset_index()) + "\n")
    # trend contrasts for PPI++ (independent LS: SE from CI half widths via se)
    cc = []
    for method in ("naive (on predictions)", "gold only (Hajek, bootstrap)", "PPI++ (lambda tuned)", "simple PPI (lambda=1, bootstrap rectifier)"):
        for nm, (a, b) in {"LS18 - LS13": ("LS18", "LS13"), "LS17 - LS13": ("LS17", "LS13"),
                           "late - early": (PERIODS[2], PERIODS[0])}.items():
            ra = PP[(PP["group"] == a) & (PP["method"] == method)].iloc[0]
            rb = PP[(PP["group"] == b) & (PP["method"] == method)].iloc[0]
            d_ = ra.est - rb.est
            se = np.sqrt(ra.se ** 2 + rb.se ** 2)
            cc.append({"method": method, "contrast": nm, "diff": fmt(d_, d_ - 1.96 * se, d_ + 1.96 * se),
                       "z": d_ / se})
    cc = pd.DataFrame(cc)
    save("t1_ppi_contrasts", cc)
    out.append("Differences between groups (independent groups; SE = sqrt(se_a^2 + se_b^2); normal 95% CI):\n")
    out.append(md(cc) + "\n")
    # PPI slope
    dppi2 = dppi.assign(dec=(dppi["year"] - 2000.0) / 10.0)
    rs = []
    for lam in ("opt", 1.0):
        r = B.ppi_fit(dppi2, "y_m2", "y_gold", "R", "pi", ["dec"], [], {"slope": {"dec": 1.0}}, "ols", lam=lam, w_col="m2_weight")
        rs.append(r)
    rs = pd.concat(rs, ignore_index=True)
    sl_ppi = pd.DataFrame({"method": rs["method"], "slope per decade [95% CI]": [fmt(r.estimate, r.ci_lo, r.ci_hi, 4) for r in rs.itertuples()]})
    save("t1_ppi_slope", sl_ppi)
    out.append("Linear time trend in the mean (per decade), design-based:\n")
    out.append(md(sl_ppi.drop_duplicates()) + "\n")
    for r in rs.itertuples():
        if r.method.startswith("PPI++") or r.method.startswith("naive") or r.method.startswith("gold"):
            KEY["ppi_slope " + r.method] = fmt(r.estimate, r.ci_lo, r.ci_hi, 4)
    # plain M2 population means (all analysis sample) with CIs, for the headline numbers
    rows = []
    for ls in range(13, 19):
        for lab, src in (("all analysed questions", est), ("M3 universe (1-6 parts)", est_u)):
            e = src[src["ls"] == ls]
            fit = M.fit_model(e, "y_m2", [], [], weight="m2_weight")
            c_ = fit.coef("const")
            rows.append({"ls": ls, "population": lab, "n": fit.n, "mean_y_m2 [95% CI]": fmt(c_["estimate"], c_["ci_lo"], c_["ci_hi"])})
    m2m = pd.DataFrame(rows)
    save("t1_m2_means", m2m)
    out.append("Design-weighted M2 means by LS (no correction), two-way clustered CIs:\n")
    out.append(md(m2m) + "\n")
    # figure 1
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    cols = {"gold only (Hajek, bootstrap)": "#009E73", "naive (on predictions)": "#D55E00", "PPI++ (lambda tuned)": "#0072B2"}
    lbl = {"gold only (Hajek, bootstrap)": "human gold (design-weighted)", "naive (on predictions)": "M2 uncorrected", "PPI++ (lambda tuned)": "PPI++ corrected"}
    for ax, labs, title in ((axs[0], [f"LS{ls}" for ls in range(13, 19)], "By Lok Sabha"), (axs[1], PERIODS, "By period")):
        for j, m in enumerate(cols):
            sub = PP[(PP["method"] == m) & PP["group"].isin(labs)].set_index("group").reindex(labs)
            xs = np.arange(len(labs)) + (j - 1) * 0.22
            ax.errorbar(xs, sub["est"], yerr=[sub["est"] - sub["lo"], sub["hi"] - sub["est"]], fmt="o", ms=4, capsize=2,
                        color=cols[m], label=lbl[m])
        ax.set_xticks(range(len(labs)))
        ax.set_xticklabels([l.split(" (")[0] if "(" in l else l for l in labs])
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=.3)
    axs[0].set_ylabel("Mean question score (M3 universe)")
    axs[1].legend(fontsize=8, loc="upper right")
    fig.suptitle("Human gold vs M2 vs bias-corrected means (95% CIs); gold CIs are wide because n per LS is 51-90 questions", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "rq2_fig1_gold_vs_m2.png"), dpi=170)
    plt.close(fig)
    out.append("Figure: `rq2_fig1_gold_vs_m2.png`\n")
    return "\n".join(out), (pp, g, Q)


# =========================================================================================== task 2
def task2(df, est, g, B_=1000):
    out = ["## 2. Is it question complexity?\n"]
    # --- descriptive trends
    rows = []
    allq = df[df["analysis_sample"]]
    for ls in range(13, 19):
        a = allq[allq["ls"] == ls]
        e = est[est["ls"] == ls]
        r = {"LS": ls, "n questions (all)": len(a), "mean n_parts": a["n_parts"].mean(), "% >=5 parts": 100 * (a["n_parts"] >= 5).mean(),
             "median question chars": a["question_len"].median(), "mean log_qlen": a["log_qlen"].mean(),
             "% starred": 100 * a["starred"].mean(), "% security": 100 * (a["ministry_group"] == "security").mean(),
             "% >0 opposition askers": 100 * (a["asker_bloc"] == "opposition").mean()}
        for k in ("rt_information", "rt_reason", "rt_action", "rt_opinion"):
            r["% " + k] = 100 * np.average(e[k].dropna(), weights=e.loc[e[k].notna(), "m2_weight"])
        r["median question words"] = a["q_words"].median()
        r["median words per part"] = (a["q_words"] / a["n_parts"]).median()
        r["% questions flagged Hindi (text stripped)"] = 100 * a["q_hindi"].mean()
        r["mean answer chars"] = a["answer_len"].mean()
        r["median answer words"] = a["a_words"].median()
        r["% has annexure"] = 100 * a["has_annexure"].mean()
        r["% OCR answers"] = 100 * a["answer_ocr"].mean()
        rows.append(r)
    desc = pd.DataFrame(rows)
    save("t2_complexity_by_ls", desc)
    out.append("### 2a. Did questions get longer / more multi-part? (all analysed questions; request-type shares are design-weighted M2 labels)\n")
    out.append(md(desc, "{:,.2f}") + "\n")
    # trend per decade, clustered
    tr = []
    for v, wt, d_ in (("n_parts", None, allq), ("log_qlen", None, allq), ("starred", None, allq), ("rt_action", "m2_weight", est),
                      ("rt_information", "m2_weight", est), ("rt_reason", "m2_weight", est), ("rt_opinion", "m2_weight", est)):
        d2 = d_[d_[v].notna()]
        fit = M.fit_model(d2, v, ["dec"], [], weight=wt)
        c_ = fit.coef("dec")
        tr.append({"variable": v, "slope per decade [95% CI]": fmt(c_["estimate"], c_["ci_lo"], c_["ci_hi"], 4), "mean": d2[v].mean()})
    tr = pd.DataFrame(tr)
    save("t2_complexity_trends", tr)
    out.append("Linear trend per decade (two-way clustered CI):\n")
    out.append(md(tr) + "\n")
    # --- regressions
    core = ["n_parts", "log_qlen", "starred"]
    rt = ["rt_reason", "rt_action", "rt_opinion"]
    est = est.copy()
    est["ls_c"] = est["ls"]
    base = dict(df=est, weight="m2_weight")
    specs = [
        ("S0 none", [], []),
        ("S1a n_parts", ["n_parts"], []), ("S1b log_qlen", ["log_qlen"], []), ("S1c starred", ["starred"], []),
        ("S1 n_parts+log_qlen+starred", core, []),
        ("S2 S1 + request-type shares", core + rt, []),
        ("S3 S1 + ministry_group FE", core, ["ministry_group"]),
        ("S3b S1 + ministry_canonical FE (77)", core, ["ministry_canonical"]),
        ("S4 S1 + topic FE (K=50)", core, ["topic"]),
        ("S5 S2 + ministry_canonical + topic FE (full)", core + rt, ["ministry_canonical", "topic"]),
        ("S6 S5 + coasker + asker bloc", core + rt + ["coasker", "opp", "mixed", "outside"], ["ministry_canonical", "topic"]),
        ("S7 S6 + answer-side (log answer len, annexure; mediators)", core + rt + ["coasker", "opp", "mixed", "outside", "log_alen", "has_annex"],
         ["ministry_canonical", "topic"]),
        ("S8 S5 with flexible length: question-length decile FE + n_parts FE (1..8+)", rt + ["starred"], ["qlen_dec", "nparts_c", "ministry_canonical", "topic"]),
        ("S9 S5 with log words per part instead of log_qlen", ["n_parts", "log_wpp", "starred"] + rt, ["ministry_canonical", "topic"]),
    ]
    est["log_alen"] = np.log1p(est["answer_len"])
    est["log_wpp"] = np.log(est["q_words"].clip(lower=1) / est["n_parts"])
    est["has_annex"] = est["has_annexure"].astype(float)
    res_ls, res_dec = [], []
    fits = {}
    for y in ("y_m2", "y_full"):
        for name, num, cat in specs:
            if y == "y_full" and not (name.startswith("S0") or name.startswith("S1 ") or name.startswith("S5") or name.startswith("S8")):
                continue
            fl = M.fit_model(est, y, num, cat + ["ls"], ref={"ls": 13}, weight="m2_weight", label=name)
            fd = M.fit_model(est, y, num + ["dec"], cat, weight="m2_weight", label=name)
            fits[(y, name)] = (fl, fd)
            r = {"y": y, "spec": name, "n": fl.n}
            for ls in range(14, 19):
                c_ = fl.coef(f"ls[{ls}]")
                r[f"LS{ls} vs LS13"] = fmt(c_["estimate"], c_["ci_lo"], c_["ci_hi"], 3)
                r[f"_b{ls}"] = c_["estimate"]
                r[f"_lo{ls}"] = c_["ci_lo"]
                r[f"_hi{ls}"] = c_["ci_hi"]
            res_ls.append(r)
            c_ = fd.coef("dec")
            res_dec.append({"y": y, "spec": name, "n": fd.n, "slope per decade": fmt(c_["estimate"], c_["ci_lo"], c_["ci_hi"], 4),
                            "_b": c_["estimate"], "_lo": c_["ci_lo"], "_hi": c_["ci_hi"], "g_mp": fd.g_sizes[0], "g_session": fd.g_sizes[1]})
    R1, R2 = pd.DataFrame(res_ls), pd.DataFrame(res_dec)
    for y in ("y_m2", "y_full"):
        b0 = R1[(R1["y"] == y) & R1["spec"].str.startswith("S0")].iloc[0]
        d0 = R2[(R2["y"] == y) & R2["spec"].str.startswith("S0")].iloc[0]
        R1.loc[R1["y"] == y, "% of LS18 gap surviving"] = 100 * R1.loc[R1["y"] == y, "_b18"] / b0["_b18"]
        R1.loc[R1["y"] == y, "% of LS17 gap surviving"] = 100 * R1.loc[R1["y"] == y, "_b17"] / b0["_b17"]
        R2.loc[R2["y"] == y, "% of slope surviving"] = 100 * R2.loc[R2["y"] == y, "_b"] / d0["_b"]
    save("t2_reg_ls", R1)
    save("t2_reg_dec", R2)
    show1 = R1[[c for c in R1.columns if not c.startswith("_")]]
    show2 = R2[[c for c in R2.columns if not c.startswith("_")]]
    out.append("### 2b. Regressions of the question score (y_m2: FULL=1, PARTIAL=.5, NONE=0; y_full: share of parts FULL) on LS dummies (ref LS13)\n")
    out.append("WLS with design weights m2_weight; two-way clustered SEs (first-asker MP x session); t(min G - 1) CIs. Coefficients are in score units "
               "(1.0 = full scale). S2 onward uses request-type shares that are themselves M2 labels (so they may absorb part of the measured "
               "decline if M2 labels differ over time); S7 adds answer-side variables that are possible mediators and is shown only as a sensitivity.\n")
    out.append(md(show1, "{:,.1f}") + "\n")
    out.append("### 2c. Same, linear trend per decade (decade = (year - 2000) / 10)\n")
    out.append(md(show2, "{:,.1f}") + "\n")
    # full coefficients of S5
    fl, fd = fits[("y_m2", "S5 S2 + request-type shares + ministry_canonical + topic FE (full)".replace("request-type shares + ", "request-type shares + "))] \
        if False else fits[("y_m2", "S5 S2 + ministry_canonical + topic FE (full)")]
    cf = fl.table(core + rt + [f"ls[{ls}]" for ls in range(14, 19)])
    cf["95% CI"] = [fmt(r.estimate, r.ci_lo, r.ci_hi, 3) for r in cf.itertuples()]
    out.append("Coefficients on controls in S5 (y_m2, LS dummies; per unit of each control):\n")
    out.append(md(cf[["term", "95% CI", "p"]], "{:,.4f}") + "\n")
    save("t2_s5_coefs", cf)
    # ---- data-source diagnostics (LS18 is built from docx/pdf, LS13-17 from the API)
    rows = []
    for (ls, src), d in est.groupby(["ls", "source_q"]):
        rows.append({"LS": ls, "source_q": src, "n (M2 sample)": len(d), "weighted share of LS %": 100 * d["m2_weight"].sum() / est.loc[est["ls"] == ls, "m2_weight"].sum(),
                     "mean y_m2": np.average(d["y_m2"], weights=d["m2_weight"]), "mean log_qlen": np.average(d["log_qlen"], weights=d["m2_weight"]),
                     "median question words": d["q_words"].median(),
                     "% question has Hindi": 100 * np.average(d["q_hindi"], weights=d["m2_weight"])})
    sd_ = pd.DataFrame(rows)
    save("t2_source_diag", sd_)
    out.append("### 2d. Data-source diagnostics (extraction source differs by LS; is the LS18 length increase an extraction artefact?)\n")
    out.append("LS13-17 questions come from the API ('api', a few fallback docx), LS18 from docx / pdf / ocr documents. Word counts (whitespace-split) "
               "rise in every LS18 source, so the length increase is not a whitespace / encoding artefact of one extractor. Devanagari text is already "
               "stripped from question_text in the data (the Hindi flag only records its presence).\n")
    out.append(md(sd_, "{:,.3f}") + "\n")
    e18 = est[est["ls"] == 18].copy()
    e18["q_hindi_f"] = e18["q_hindi"].astype(float)
    f18 = M.fit_model(e18, "y_m2", ["n_parts", "log_qlen", "starred"] + rt + ["q_hindi_f"], ["ministry_canonical", "topic", "source_q"],
                      weight="m2_weight", label="LS18 only")
    tt = f18.table(["n_parts", "log_qlen", "starred", "q_hindi_f"] + [n for n in f18.names if n.startswith("source_q[")])
    tt["95% CI"] = [fmt(r.estimate, r.ci_lo, r.ci_hi, 3) for r in tt.itertuples()]
    out.append(f"Within LS18 only (n={f18.n:,}; S5 controls + Hindi flag + source_q dummies, reference = most frequent source): does the extraction source predict y_m2?\n")
    out.append(md(tt[["term", "95% CI", "p"]], "{:,.4f}") + "\n")
    save("t2_ls18_within", tt)
    # does the length gradient exist in the human gold? (gold sample, HT weights, Rao-Wu bootstrap)
    gg = g.copy()
    xs = ["log_qlen", "n_parts", "starred"]
    Xg = np.column_stack([np.ones(len(gg))] + [gg[c].to_numpy(float) for c in xs] + [(gg["ls"] == k).to_numpy(float) for k in range(14, 19)])
    wg_ = gg["w"].to_numpy()
    Wr = rep_weights(wg_, gg["strat"].to_numpy(), B_, np.random.default_rng(SEED + 2))

    def wls(W, yv):
        out_ = np.empty((len(W), 3))
        for i, ww in enumerate(W):
            sw = np.sqrt(ww)
            b = np.linalg.lstsq(Xg * sw[:, None], yv * sw, rcond=None)[0]
            out_[i] = b[1:4]
        return out_
    rows = []
    res_ = {}
    for nm_, col in (("human gold", "y_gold"), ("M2 (same questions)", "y_m2")):
        yv = gg[col].to_numpy(float)
        pt = wls(wg_[None, :], yv)[0]
        rp = wls(Wr, yv)
        res_[nm_] = (pt, rp)
    pt_d = res_["M2 (same questions)"][0] - res_["human gold"][0]
    rp_d = res_["M2 (same questions)"][1] - res_["human gold"][1]
    for j, c in enumerate(xs):
        for nm_ in res_:
            pt, rp = res_[nm_]
            rows.append({"variable": c, "outcome": nm_, "coef [95% CI]": fmt(pt[j], *np.percentile(rp[:, j], [2.5, 97.5]))})
        rows.append({"variable": c, "outcome": "M2 minus gold", "coef [95% CI]": fmt(pt_d[j], *np.percentile(rp_d[:, j], [2.5, 97.5]))})
    gt_ = pd.DataFrame(rows)
    save("t2_gold_length_gradient", gt_)
    out.append("Does the question-length gradient show up in the human gold? Gold sample (400 questions), HT-weighted WLS of the question score on "
               "log question length, n_parts, starred and LS dummies; Rao-Wu bootstrap 95% CI. 'M2 minus gold' tests whether M2 over-penalises long questions.\n")
    out.append(md(gt_) + "\n")
    KEY["gold_length_gradient"] = {f"{a_}|{b_}": v for (a_, b_), v in gt_.set_index(["variable", "outcome"])["coef [95% CI]"].to_dict().items()}
    # Year-by-year means of raw y and composition
    yr = est.assign(yr=np.floor(est["year"]).astype(int))
    yt = yr.groupby("yr").apply(lambda d: pd.Series({"n": len(d), "y_m2": np.average(d["y_m2"], weights=d["m2_weight"]),
                                                    "n_parts": np.average(d["n_parts"], weights=d["m2_weight"]),
                                                    "log_qlen": np.average(d["log_qlen"], weights=d["m2_weight"])}), include_groups=False).reset_index()
    save("t2_year_means", yt)
    KEY["reg_ls18_S0"] = R1[(R1["y"] == "y_m2") & R1["spec"].str.startswith("S0")]["LS18 vs LS13"].iloc[0]
    KEY["reg_ls18_S5"] = R1[(R1["y"] == "y_m2") & R1["spec"].str.startswith("S5")]["LS18 vs LS13"].iloc[0]
    KEY["reg_dec_S0"] = R2[(R2["y"] == "y_m2") & R2["spec"].str.startswith("S0")]["slope per decade"].iloc[0]
    KEY["reg_dec_S5"] = R2[(R2["y"] == "y_m2") & R2["spec"].str.startswith("S5")]["slope per decade"].iloc[0]
    KEY["pct_surv_S5_LS18"] = float(R1[(R1["y"] == "y_m2") & R1["spec"].str.startswith("S5")]["% of LS18 gap surviving"].iloc[0])
    KEY["pct_surv_S5_dec"] = float(R2[(R2["y"] == "y_m2") & R2["spec"].str.startswith("S5")]["% of slope surviving"].iloc[0])
    return "\n".join(out)


# =========================================================================================== task 3
def eva_group(s):
    s = s.copy()
    for grp, mem in EVA_POOL.items():
        s = s.where(~s.isin(mem), grp)
    return s


def sess_boot(S, tot, sess_ls, B_, rng, group_map):
    """Session-cluster bootstrap. S: sessions x K weighted counts; tot: sessions weighted part total; group_map: name -> list of LS.
    Returns dict name -> (point (K+1,), reps (B, K+1)) where last column is the nonFULL total rate. Rates are x100 per part."""
    res = {}
    for gname, lss in group_map.items():
        idx = np.where(np.isin(sess_ls, lss))[0]
        # resample sessions within each LS of the group (stratified by LS)
        cnt = np.zeros((B_, len(idx)))
        for ls in lss:
            loc = np.where(sess_ls[idx] == ls)[0]
            cnt[:, loc] = rng.multinomial(len(loc), np.ones(len(loc)) / len(loc), size=B_)
        Sg, tg = S[idx], tot[idx]

        def rates(cn):
            num = cn @ Sg
            den = cn @ tg
            return 100 * num / den[:, None]

        pt = rates(np.ones((1, len(idx))))[0]
        res[gname] = (pt, rates(cnt))
    return res


def task3(df, est, gold_pack):
    rng = np.random.default_rng(SEED + 3)
    out = ["## 3. Composition of evasion\n"]
    ml = pd.read_parquet(os.path.join(C.MEASURES, "m2_labels.parquet"), columns=["part_id", "qid", "label", "evasion_type", "request_type", "status"])
    ml = ml[ml["status"] == "ok"]
    q = est[["qid", "ls", "session_id", "m2_weight", "n_parts", "year", "mp_id"]]
    p = ml.merge(q, on="qid", how="inner")
    p["eva"] = eva_group(p["evasion_type"])
    p["nonfull"] = p["label"] != "FULL"
    sessions = sorted(p["session_id"].unique())
    sidx = {s: i for i, s in enumerate(sessions)}
    sess_ls = np.array([int(s.split("-")[0]) for s in sessions])
    p["si"] = p["session_id"].map(sidx)
    K = EVA + ["PARTIAL_all", "NONE_all"]
    S = np.zeros((len(sessions), len(K) + 1))
    np.add.at(S, p.loc[p["nonfull"], "si"].to_numpy(), 0)  # placeholder
    wv = p["m2_weight"].to_numpy()
    for j, k in enumerate(EVA):
        m = (p["eva"] == k).to_numpy()
        np.add.at(S[:, j], p["si"].to_numpy()[m], wv[m])
    np.add.at(S[:, len(EVA)], p["si"].to_numpy()[(p["label"] == "PARTIAL").to_numpy()], wv[(p["label"] == "PARTIAL").to_numpy()])
    np.add.at(S[:, len(EVA) + 1], p["si"].to_numpy()[(p["label"] == "NONE").to_numpy()], wv[(p["label"] == "NONE").to_numpy()])
    np.add.at(S[:, len(EVA) + 2], p["si"].to_numpy()[p["nonfull"].to_numpy()], wv[p["nonfull"].to_numpy()])
    tot = np.zeros(len(sessions))
    np.add.at(tot, p["si"].to_numpy(), wv)
    cols = EVA + ["PARTIAL", "NONE", "ALL non-FULL"]
    gm = {f"LS{ls}": [ls] for ls in range(13, 19)}
    gm.update({"early (LS13-14)": [13, 14], "middle (LS15-16)": [15, 16], "late (LS17-18)": [17, 18]})
    # NB: pooling LS in a period: weighting by m2_weight already makes each LS count by population size
    res = sess_boot(S, tot, sess_ls, 2000, rng, gm)
    rows = []
    for g_, (pt, reps) in res.items():
        lo, hi = np.percentile(reps, [2.5, 97.5], axis=0)
        r = {"group": g_}
        for j, c in enumerate(cols):
            r[c] = fmt(pt[j], lo[j], hi[j], 2)
        rows.append(r)
    t3a = pd.DataFrame(rows)
    save("t3_rate_per_100_parts", t3a)
    out.append("### 3a. Rate per 100 labelled parts (design-weighted M2; session-cluster bootstrap 95% CI, B=2000, stratified by LS)\n")
    out.append("The evasion-type columns add up to 'ALL non-FULL' (= PARTIAL + NONE). OTHER pools OTHER, WITHHELD, DOES_NOT_ARISE, ANNEXURE_ONLY.\n")
    out.append(md(t3a) + "\n")
    # shares among non-FULL
    rows = []
    for g_, (pt, reps) in res.items():
        sh = reps[:, :len(EVA)] / reps[:, [len(EVA) + 2]] * 100
        pt_sh = pt[:len(EVA)] / pt[len(EVA) + 2] * 100
        lo, hi = np.percentile(sh, [2.5, 97.5], axis=0)
        r = {"group": g_}
        for j, c in enumerate(EVA):
            r[c] = fmt(pt_sh[j], lo[j], hi[j], 1)
        rows.append(r)
    t3b = pd.DataFrame(rows)
    save("t3_share_among_nonfull_m2", t3b)
    out.append("### 3b. Evasion-type share (%) among non-FULL parts, design-weighted M2\n")
    out.append(md(t3b) + "\n")
    # contributions to the change
    rows = []
    for a, b in (("LS18", "LS13"), ("LS17", "LS13"), ("late (LS17-18)", "early (LS13-14)"), ("LS18", "LS17")):
        da = res[a][1] - res[b][1]
        dp = res[a][0] - res[b][0]
        tot_ch = dp[len(EVA) + 2]
        for j, c in enumerate(cols):
            lo, hi = np.percentile(da[:, j], [2.5, 97.5])
            rows.append({"contrast": f"{a} - {b}", "type": c, "change (pp of parts)": fmt(dp[j], lo, hi, 2),
                         "% of total rise": 100 * dp[j] / tot_ch if c in EVA else np.nan,
                         "ratio late/early": np.nan})
        # ratio
    t3c = pd.DataFrame(rows)
    save("t3_contributions", t3c)
    out.append("### 3c. Which types drive the rise? Change in rate per 100 parts (percentage points), with CI\n")
    out.append(md(t3c.drop(columns=["ratio late/early"]), "{:,.1f}") + "\n")
    KEY["nonfull_rate"] = {g_: fmt(res[g_][0][len(EVA) + 2], *np.percentile(res[g_][1][:, len(EVA) + 2], [2.5, 97.5]), 1) for g_ in res}
    KEY["contrib_LS18_LS13"] = t3c[t3c["contrast"] == "LS18 - LS13"].set_index("type")["change (pp of parts)"].to_dict()
    KEY["contrib_LS17_LS13"] = t3c[t3c["contrast"] == "LS17 - LS13"].set_index("type")["change (pp of parts)"].to_dict()
    KEY["contrib_late_early"] = t3c[t3c["contrast"] == "late (LS17-18) - early (LS13-14)"].set_index("type")["change (pp of parts)"].to_dict()
    KEY["share_nonfull"] = t3b.set_index("group").to_dict("index")

    # ---- request type composition vs within-type
    out.append("### 3d. Composition vs within-type change at the part level (request type, M2-labelled)\n")
    rtab = []
    for rt in ["INFORMATION", "ACTION", "REASON", "OPINION"]:
        r = {"request_type": rt}
        for per, lss in (("early", [13, 14]), ("late", [17, 18])):
            d = p[p["ls"].isin(lss)]
            r[f"share {per} %"] = 100 * d.loc[d["request_type"] == rt, "m2_weight"].sum() / d["m2_weight"].sum()
            dd = d[d["request_type"] == rt]
            r[f"nonFULL rate {per} %"] = 100 * (dd["m2_weight"] * dd["nonfull"]).sum() / dd["m2_weight"].sum()
        rtab.append(r)
    rtab = pd.DataFrame(rtab)
    tot_e = (rtab["share early %"] * rtab["nonFULL rate early %"] / 100).sum()
    tot_l = (rtab["share late %"] * rtab["nonFULL rate late %"] / 100).sum()
    comp = ((rtab["share late %"] - rtab["share early %"]) / 100 * (rtab["nonFULL rate early %"] + rtab["nonFULL rate late %"]) / 2).sum()
    within = ((rtab["nonFULL rate late %"] - rtab["nonFULL rate early %"]) / 100 * (rtab["share early %"] + rtab["share late %"]) / 2).sum()
    save("t3_request_type", rtab)
    out.append(md(rtab, "{:,.1f}") + "\n")
    out.append(f"Kitagawa decomposition of the early -> late change in the non-FULL part rate ({tot_e:.1f}% -> {tot_l:.1f}%, "
               f"+{tot_l - tot_e:.1f} pp): composition across request types {comp:+.2f} pp, within-type {within:+.2f} pp.\n")
    KEY["kitagawa"] = dict(early=tot_e, late=tot_l, comp=comp, within=within)

    # ---- by year deferral / generic / incomplete series for the figure (and text marker check)
    # ---- gold-based shares by period
    pp, g, Q = gold_pack
    gl = pd.read_csv(os.path.join(C.CODING, "gold_labels.csv"), keep_default_na=False)
    gl = gl[gl["qid"].isin(g["qid"]) & (gl["label"] != "")].copy()
    gl["eva"] = eva_group(gl["evasion_type"])
    gl["nf"] = gl["label"] != "FULL"
    mlf = ml[["part_id", "label", "evasion_type"]].rename(columns={"label": "m2l", "evasion_type": "m2e"})
    gl = gl.merge(mlf, on="part_id", how="left")
    gl["m2eva"] = eva_group(gl["m2e"].fillna("NONE_APPLICABLE"))
    gl["m2nf"] = gl["m2l"].fillna("FULL") != "FULL"
    nq = len(g)
    qi = pd.Series(np.arange(nq), index=g["qid"])
    gl["qi"] = qi.reindex(gl["qid"]).to_numpy()
    QG = np.zeros((nq, len(EVA)))
    QM = np.zeros((nq, len(EVA)))
    QN = np.zeros(nq)
    QGnf, QMnf = np.zeros(nq), np.zeros(nq)
    for j, k in enumerate(EVA):
        sel = gl[(gl["eva"] == k) & gl["nf"]]
        np.add.at(QG[:, j], sel["qi"].to_numpy(), 1)
        sel = gl[(gl["m2eva"] == k) & gl["m2nf"]]
        np.add.at(QM[:, j], sel["qi"].to_numpy(), 1)
    np.add.at(QN, gl["qi"].to_numpy(), 1)
    np.add.at(QGnf, gl.loc[gl["nf"], "qi"].to_numpy(), 1)
    np.add.at(QMnf, gl.loc[gl["m2nf"], "qi"].to_numpy(), 1)
    w = g["w"].to_numpy()
    Wrep = rep_weights(w, g["strat"].to_numpy(), 2000, np.random.default_rng(SEED + 1))
    per_mask = {pn: (g["period"] == pn).to_numpy() for pn in PERIODS}
    per_mask["all"] = np.ones(nq, bool)

    def gfn(mask):
        def fn(W):
            Wm = W * mask
            den = Wm @ QN
            return np.column_stack([100 * (Wm @ QG) / den[:, None], 100 * (Wm @ QGnf) / den,
                                    100 * (Wm @ QM) / den[:, None], 100 * (Wm @ QMnf) / den])
        return fn

    nm = [f"gold_{k}" for k in EVA] + ["gold_nonFULL"] + [f"m2_{k}" for k in EVA] + ["m2_nonFULL"]
    grows, gshare_rows = [], []
    gres = {}
    for pn, mk in per_mask.items():
        sm, reps = boot_summary(gfn(mk), w, Wrep, nm)
        gres[pn] = (sm, reps)
        r = {"period": pn, "gold parts": int(gl["qi"].isin(np.where(mk)[0]).sum())}
        sh = {}
        for typ in ("gold", "m2"):
            tot_i = nm.index(f"{typ}_nonFULL")
            for j, k in enumerate(EVA):
                i = nm.index(f"{typ}_{k}")
                share = 100 * reps[:, i] / reps[:, tot_i]
                pt = 100 * sm.loc[i, "est"] / sm.loc[tot_i, "est"]
                lo, hi = np.percentile(share, [2.5, 97.5])
                sh[(typ, k)] = fmt(pt, lo, hi, 1)
        for typ in ("gold", "m2"):
            for k in EVA:
                r[f"{typ} {k}"] = sh[(typ, k)]
            tot_i = nm.index(f"{typ}_nonFULL")
            r[f"{typ} nonFULL rate %"] = fmt(sm.loc[tot_i, "est"], sm.loc[tot_i, "lo"], sm.loc[tot_i, "hi"], 1)
        grows.append(r)
    gt = pd.DataFrame(grows)
    save("t3_gold_shares_wide", gt)
    show = []
    for pn in list(per_mask):
        r = gt[gt["period"] == pn].iloc[0]
        for typ, lab in (("gold", "human gold"), ("m2", "M2 on the same parts")):
            show.append({"period": pn, "source": lab, "nonFULL rate %": r[f"{typ} nonFULL rate %"], **{k: r[f"{typ} {k}"] for k in EVA}})
    show = pd.DataFrame(show)
    out.append("### 3e. Gold-based evasion-type shares (% of non-FULL parts) by period, with M2 labels on the same gold parts for comparison\n")
    out.append("HT-weighted (1/gold_pi), Rao-Wu bootstrap 95% CI. 'nonFULL rate' = % of parts that are not FULL. Small n: treat as indicative.\n")
    out.append(md(show) + "\n")
    # gold rates per 100 parts and changes late-early
    rows = []
    for typ, lab in (("gold", "human gold"), ("m2", "M2 on same parts")):
        for k in EVA + ["nonFULL"]:
            i = nm.index(f"{typ}_{k}")
            d = gres[PERIODS[2]][1][:, i] - gres[PERIODS[0]][1][:, i]
            e = gres[PERIODS[2]][0].loc[i, "est"] - gres[PERIODS[0]][0].loc[i, "est"]
            rows.append({"source": lab, "type": k, "rate early": gres[PERIODS[0]][0].loc[i, "est"], "rate late": gres[PERIODS[2]][0].loc[i, "est"],
                         "change late - early (pp of parts) [95% CI]": fmt(e, *np.percentile(d, [2.5, 97.5]), 1)})
    gch = pd.DataFrame(rows)
    save("t3_gold_changes", gch)
    out.append("Gold vs M2 on the same parts: rate per 100 parts, early (LS13-14) vs late (LS17-18):\n")
    out.append(md(gch, "{:,.1f}") + "\n")
    KEY["gold_change"] = {f"{a_}|{b_}": v for (a_, b_), v in gch.set_index(["source", "type"])["change late - early (pp of parts) [95% CI]"].to_dict().items()}
    # evasion type agreement among parts both nonFULL
    both = gl[gl["nf"] & gl["m2nf"]]
    ag = both.groupby("period_g" if False else both["qid"].map(g.set_index("qid")["period"]))["eva"].count()
    rows = []
    for pn in PERIODS + ["all"]:
        d = both if pn == "all" else both[both["qid"].map(g.set_index("qid")["period"]) == pn]
        rows.append({"period": pn, "parts non-FULL in both": len(d),
                     "evasion-type agreement (unweighted)": float((d["eva"] == d["m2eva"]).mean()) if len(d) else np.nan,
                     "gold DEFERRAL parts": int((gl.loc[gl["qid"].map(g.set_index("qid")["period"]).eq(pn) if pn != "all" else slice(None), "eva"] == "DEFERRAL").sum()),
                     "of which M2 DEFERRAL": int(((d["eva"] == "DEFERRAL") & (d["m2eva"] == "DEFERRAL")).sum())})
    agdf = pd.DataFrame(rows)
    save("t3_type_agreement", agdf)
    out.append("Evasion-type agreement among parts that are non-FULL in both (types as defined above):\n")
    out.append(md(agdf) + "\n")

    # ---- independent text-marker check for DEFERRAL
    out.append("### 3f. Is the decline of DEFERRAL real? Independent text-marker check\n")
    out.append("Regex flags computed on the answer text of every analysed question (no text is stored or quoted). "
               "Marker A: wording of information 'being collected / compiled / gathered / collated'. Marker B: wording that the information "
               "'will be laid on the Table'. 'Both' is the classic deferral formula. Rates are % of questions (census frame, unweighted; "
               "LS13-16 are the full population here, not the 1/6 sample).\n")
    mk = df[["qid", "mkA", "mkB"]].copy()
    mk["both"] = mk["mkA"] & mk["mkB"]
    f2 = df[df["analysis_sample"]][["qid", "ls", "year", "in_m2_sample", "m2_weight", "answer_len", "statement_laid"]].merge(mk, on="qid")
    # M2 deferral flag per question (any part DEFERRAL)
    dq = ml.assign(dfr=ml["evasion_type"] == "DEFERRAL").groupby("qid")["dfr"].agg(["any", "mean"]).rename(columns={"any": "m2_any_defer", "mean": "m2_defer_share"})
    f2 = f2.merge(dq, on="qid", how="left")
    rows = []
    for ls in range(13, 19):
        d = f2[f2["ls"] == ls]
        dm = d[d["m2_any_defer"].notna() & (d["m2_weight"] > 0)]
        rows.append({"LS": ls, "n questions": len(d), "% marker A": 100 * d["mkA"].mean(), "% marker B": 100 * d["mkB"].mean(),
                     "% both (formula)": 100 * d["both"].mean(),
                     "M2 sample: % questions with any DEFERRAL part (weighted)": 100 * np.average(dm["m2_any_defer"].astype(float), weights=dm["m2_weight"]),
                     "M2 sample: % formula questions (weighted)": 100 * np.average(dm["both"], weights=dm["m2_weight"]),
                     "among formula Qs: % with any M2 DEFERRAL (weighted)": 100 * np.average(dm.loc[dm["both"], "m2_any_defer"].astype(float), weights=dm.loc[dm["both"], "m2_weight"]),
                     "among formula Qs: % with any M2 DEFERRAL or non-FULL?": np.nan})
    tm = pd.DataFrame(rows).drop(columns=["among formula Qs: % with any M2 DEFERRAL or non-FULL?"])
    save("t3_deferral_marker", tm)
    out.append(md(tm, "{:,.2f}") + "\n")
    # y_m2 for formula questions vs others
    dm = f2[f2["m2_any_defer"].notna() & (f2["m2_weight"] > 0)].merge(est[["qid", "y_m2"]], on="qid")
    rows = []
    for ls in range(13, 19):
        d = dm[dm["ls"] == ls]
        for flag, nme in ((True, "formula questions"), (False, "other questions")):
            e = d[d["both"] == flag]
            rows.append({"LS": ls, "group": nme, "n": len(e), "mean y_m2 (weighted)": np.average(e["y_m2"], weights=e["m2_weight"])})
    ym = pd.DataFrame(rows).pivot(index="LS", columns="group", values="mean y_m2 (weighted)").reset_index()
    out.append("Mean y_m2 of questions carrying the formula vs the rest:\n")
    out.append(md(ym) + "\n")
    save("t3_deferral_ym2", ym)
    # confusion marker vs M2 DEFERRAL (M2 sample, weighted)
    d = dm
    w_ = d["m2_weight"]
    sens = np.average(d.loc[d["m2_any_defer"] > 0, "both"], weights=w_[d["m2_any_defer"] > 0])
    prec = np.average(d.loc[d["both"], "m2_any_defer"].astype(float), weights=w_[d["both"]])
    out.append(f"Agreement between the text marker and M2 DEFERRAL (design-weighted, all LS): among questions with any M2 DEFERRAL part, "
               f"{100 * sens:.1f}% carry the formula; among formula questions, {100 * prec:.1f}% have an M2 DEFERRAL part.\n")
    KEY["deferral_marker"] = tm.set_index("LS").to_dict("index")
    KEY["deferral_agree"] = dict(share_defer_with_formula=sens, share_formula_with_defer=prec)
    # decline excluding the deferral-formula questions (they score ~0.1 and have almost vanished)
    ee = est.merge(mk[["qid", "both"]], on="qid", how="left")
    ee["both"] = ee["both"].fillna(False)
    core_ = ["n_parts", "log_qlen", "starred", "rt_reason", "rt_action", "rt_opinion"]
    rows = []
    for lab, sub in (("all questions", ee), ("excluding deferral-formula questions", ee[~ee["both"]])):
        for spec, num, cat in (("S0 none", [], []), ("S5 full controls", core_, ["ministry_canonical", "topic"])):
            fl = M.fit_model(sub, "y_m2", num, cat + ["ls"], ref={"ls": 13}, weight="m2_weight")
            fd = M.fit_model(sub, "y_m2", num + ["dec"], cat, weight="m2_weight")
            c18, c17, cd = fl.coef("ls[18]"), fl.coef("ls[17]"), fd.coef("dec")
            rows.append({"sample": lab, "spec": spec, "n": fl.n, "LS17 vs LS13": fmt(c17["estimate"], c17["ci_lo"], c17["ci_hi"]),
                         "LS18 vs LS13": fmt(c18["estimate"], c18["ci_lo"], c18["ci_hi"]), "slope per decade": fmt(cd["estimate"], cd["ci_lo"], cd["ci_hi"], 4)})
    xd = pd.DataFrame(rows)
    save("t3_excl_deferral", xd)
    out.append("### 3g. How much does the vanishing deferral formula mask? y_m2 regressions with and without formula questions\n")
    out.append(md(xd) + "\n")
    KEY["excl_deferral"] = xd.to_dict("records")
    # yearly series for figure
    p["yr"] = np.floor(p["year"]).astype(int)
    ys = p.groupby("yr").apply(lambda d: pd.Series({**{k: 100 * (d["m2_weight"] * (d["eva"] == k)).sum() / d["m2_weight"].sum() for k in EVA},
                                                    "nonFULL": 100 * (d["m2_weight"] * d["nonfull"]).sum() / d["m2_weight"].sum()}),
                               include_groups=False).reset_index()
    save("t3_year_rates", ys)
    mkyr = f2.assign(yr=np.floor(f2["year"]).astype(int)).groupby("yr")[["mkA", "mkB", "both"]].mean().mul(100).reset_index()
    save("t3_year_marker", mkyr)
    # figure 2
    cmap = {"GENERIC": "#0072B2", "INCOMPLETE": "#56B4E9", "DIFFERENT_QUESTION": "#E69F00", "NOT_MAINTAINED": "#009E73",
            "JURISDICTION": "#D55E00", "DEFERRAL": "#CC79A7", "OTHER": "#999999"}
    fig, axs = plt.subplots(1, 3, figsize=(13.5, 4.2), gridspec_kw={"width_ratios": [1.5, 1, 1]})
    ax = axs[0]
    ax.stackplot(ys["yr"], [ys[k] for k in EVA], colors=[cmap[k] for k in EVA], labels=[k.replace("_", " ").title() for k in EVA], lw=.3, edgecolor="white")
    ax.set_xlabel("Year")
    ax.set_ylabel("% of parts, by evasion type (stacked)")
    ax.set_title("M2, design-weighted, by calendar year", fontsize=9)
    ax.legend(fontsize=6.5, ncol=2, loc="upper left")
    ax = axs[1]
    labs = [f"LS{ls}" for ls in range(13, 19)]
    bot = np.zeros(len(labs))
    for k in EVA:
        j = EVA.index(k)
        v = np.array([res[l][0][j] for l in labs])
        ax.bar(labs, v, bottom=bot, color=cmap[k], width=.75)
        bot += v
    ax.set_title("M2 by Lok Sabha (rate per 100 parts)", fontsize=9)
    ax = axs[2]
    pl = ["early", "middle", "late"]
    for typ, off, hatch in (("gold", -0.2, None), ("m2", 0.2, "//")):
        bot = np.zeros(3)
        for k in EVA:
            v = np.array([gres[pn][0].loc[nm.index(f"{typ}_{k}"), "est"] for pn in PERIODS])
            ax.bar(np.arange(3) + off, v, bottom=bot, color=cmap[k], width=.38, hatch=hatch, edgecolor="white", lw=.3)
            bot += v
    ax.set_xticks(range(3))
    ax.set_xticklabels(pl)
    ax.set_title("Gold (solid) vs M2 on same parts (hatched)", fontsize=9)
    fig.suptitle("Composition of non-FULL parts by evasion type", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "rq2_fig2_evasion_composition.png"), dpi=170)
    plt.close(fig)
    out.append("Figure: `rq2_fig2_evasion_composition.png`\n")
    return "\n".join(out)


# =========================================================================================== task 4
def ols_hc3(X, y, w=None):
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    XtWX = X.T @ (X * w[:, None])
    Ainv = np.linalg.inv(XtWX)
    beta = Ainv @ (X.T @ (w * y))
    e = y - X @ beta
    h = np.einsum("ij,jk,ik->i", X, Ainv, X) * w
    u = w * e / (1 - h)
    V = Ainv @ (X.T @ (X * (u ** 2)[:, None])) @ Ainv
    return beta, V, e, h


def seg_fit(s, b, W=None, weights=None, label=""):
    """level jump and slope change at date b (decimal years) within +-W years. y = a + c1*(t-b) + g*post + d*post*(t-b)."""
    d = s if W is None else s[(s["t"] - b).abs() <= W]
    d = d.reset_index(drop=True)
    tc = d["t"] - b
    post = (tc >= 0).astype(float)
    X = np.column_stack([np.ones(len(d)), tc, post, post * tc])
    w = None if weights is None else d[weights].to_numpy()
    beta, V, e, h = ols_hc3(X, d["y"].to_numpy(), w)
    df_ = len(d) - 4
    q = stats.t.ppf(0.975, df_)
    se = np.sqrt(np.diag(V))
    return {"n_sessions": len(d), "n_pre": int((post == 0).sum()), "n_post": int((post == 1).sum()),
            "pre-slope/yr": (beta[1], beta[1] - q * se[1], beta[1] + q * se[1]),
            "jump at break": (beta[2], beta[2] - q * se[2], beta[2] + q * se[2]),
            "slope change/yr": (beta[3], beta[3] - q * se[3], beta[3] + q * se[3]),
            "p_jump": 2 * stats.t.sf(abs(beta[2] / se[2]), df_), "p_slope": 2 * stats.t.sf(abs(beta[3] / se[3]), df_)}


def extrap_gap(s, b, W=5.0):
    """Pre-break linear trend (last W years before b) extrapolated into the post period: mean (actual - predicted) gap."""
    pre = s[(s["t"] < b) & (s["t"] >= b - W)]
    post = s[(s["t"] >= b) & (s["t"] <= b + W)]
    X = np.column_stack([np.ones(len(pre)), pre["t"] - b])
    beta, res, *_ = np.linalg.lstsq(X, pre["y"].to_numpy(), rcond=None)
    e = pre["y"].to_numpy() - X @ beta
    s2 = (e ** 2).sum() / (len(pre) - 2)
    Xp = np.column_stack([np.ones(len(post)), post["t"] - b])
    pred = Xp @ beta
    gap = float((post["y"].to_numpy() - pred).mean())
    xb = Xp.mean(0)
    var = s2 * (1 / len(post) + xb @ np.linalg.inv(X.T @ X) @ xb)
    q = stats.t.ppf(0.975, len(pre) - 2)
    return gap, gap - q * np.sqrt(var), gap + q * np.sqrt(var), len(pre), len(post)


def design_mat_breaks(t, breaks):
    cols = [np.ones(len(t)), t - 2000]
    for b in breaks:
        post = (t >= b).astype(float)
        cols += [post, post * (t - b)]
    return np.column_stack(cols)


def task4(df, est):
    rng = np.random.default_rng(SEED + 4)
    out = ["## 4. Within-government trends\n"]
    e = est.copy()
    e["t"] = dec_year(e["date"])
    # unadjusted session means
    def sess_stats(d, col):
        w = d["m2_weight"].to_numpy()
        yv = d[col].to_numpy()
        m = (w * yv).sum() / w.sum()
        se = np.sqrt(((w * (yv - m)) ** 2).sum()) / w.sum()
        return pd.Series({"y": m, "se": se, "n": len(d), "t": np.average(d["t"], weights=w), "ls": d["ls"].iloc[0],
                          "govt_period": d["govt_period"].iloc[0], "y_full": np.average(d["y_full"], weights=w)})
    sd = e.groupby("session_id").apply(lambda d: sess_stats(d, "y_m2"), include_groups=False).reset_index().sort_values("t").reset_index(drop=True)
    # adjusted session means (controls S5-like, session FE)
    core = ["n_parts", "log_qlen", "starred", "rt_reason", "rt_action", "rt_opinion"]
    fit = M.fit_model(e, "y_m2", core, ["ministry_canonical", "topic", "session_id"], weight="m2_weight")
    names = fit.names
    X = fit.X
    wv = fit.w
    xbar = np.asarray(X.T @ wv).ravel() / wv.sum()
    isess = np.array([nm.startswith("session_id[") for nm in names])
    base = float(xbar[~isess] @ fit.beta[~isess])
    adj = {}
    for nm, b in zip(names, fit.beta):
        if nm.startswith("session_id["):
            adj[nm[len("session_id["):-1]] = base + b
    ref = [s for s in sd["session_id"] if s not in adj][0]
    adj[ref] = base
    sd["y_adj"] = sd["session_id"].map(adj)
    save("t4_session_means", sd)
    out.append(f"{len(sd)} sessions (LS13: {int((sd['ls'] == 13).sum())}, LS14: {int((sd['ls'] == 14).sum())}, LS15: {int((sd['ls'] == 15).sum())}, "
               f"LS16: {int((sd['ls'] == 16).sum())}, LS17: {int((sd['ls'] == 17).sum())}, LS18: {int((sd['ls'] == 18).sum())}); "
               f"design-weighted mean y_m2 per session, SE (linearised, iid within session) median {sd['se'].median():.4f}, range "
               f"{sd['se'].min():.4f}-{sd['se'].max():.4f}. 'Adjusted' = session fixed effects from a WLS with n_parts, log_qlen, starred, request-type shares, "
               f"ministry_canonical and topic FE (S5 controls) evaluated at the overall design-weighted covariate mean. Break dates = first sitting of the "
               f"new Lok Sabha (the government changes at the election in between, there are no sittings between February and July).\n")
    # within-government slopes
    rows = []
    for gp, d in sd.groupby("govt_period", sort=False):
        d = d.sort_values("t")
        X = np.column_stack([np.ones(len(d)), d["t"] - d["t"].mean()])
        for col in ("y", "y_adj"):
            beta, V, *_ = ols_hc3(X, d[col].to_numpy())
            se = np.sqrt(np.diag(V))
            q = stats.t.ppf(0.975, len(d) - 2)
            rows.append({"govt_period": gp, "series": "adjusted" if col == "y_adj" else "unadjusted", "LS": int(d["ls"].iloc[0]),
                         "sessions": len(d), "first sitting": f"{d['t'].min():.1f}", "last": f"{d['t'].max():.1f}",
                         "mean level": d[col].mean(), "slope/yr [95% CI, HC3]": fmt(beta[1], beta[1] - q * se[1], beta[1] + q * se[1], 4)})
    wg = pd.DataFrame(rows)
    save("t4_within_gov_slopes", wg)
    out.append("### 4a. Linear trend within each government period (session-level, equal-weighted OLS, HC3 CIs)\n")
    out.append(md(wg, "{:,.3f}") + "\n")
    # segmented per break
    allb = {**GOV_CHANGES, **PLACEBO}
    rows = []
    for ser in ("y", "y_adj"):
        s_ = sd.rename(columns={ser: "yy"}).assign(y=lambda d: d["yy"])
        s_ = sd[["t", ser, "se", "ls"]].rename(columns={ser: "y"})
        for nm, bd in allb.items():
            b = float(dec_year(pd.Series([bd]))[0])
            r = seg_fit(s_, b, W=5.0)
            gap = extrap_gap(s_, b, 5.0)
            rows.append({"series": "adjusted" if ser == "y_adj" else "unadjusted", "break": nm, "type": "government change" if nm in GOV_CHANGES else "placebo (no change of govt)",
                         "sessions (pre/post)": f"{r['n_pre']}/{r['n_post']}",
                         "pre-slope/yr": fmt(*r["pre-slope/yr"], d=4), "jump [95% CI]": fmt(*r["jump at break"], d=4), "p(jump)": r["p_jump"],
                         "slope change/yr [95% CI]": fmt(*r["slope change/yr"], d=4), "p(slope chg)": r["p_slope"],
                         "pre-trend extrapolation gap [95% CI]": fmt(gap[0], gap[1], gap[2], 4),
                         "_jump": r["jump at break"][0], "_jlo": r["jump at break"][1], "_jhi": r["jump at break"][2]})
    seg = pd.DataFrame(rows)
    save("t4_segmented", seg)
    out.append("### 4b. Segmented check at each election: y = a + b1 (t-t0) + g POST + d POST (t-t0), sessions within +-5 years, HC3, t(n-4)\n")
    out.append("g = level jump at t0 (extrapolating each side's line); d = change in slope per year. The extrapolation gap is the mean of "
               "(actual - pre-trend prediction) over the post sessions, pre-trend fitted on the preceding 5 years only.\n")
    out.append(md(seg[[c for c in seg.columns if not c.startswith("_")]], "{:,.3f}") + "\n")
    # precision weighted variant
    rows = []
    for nm, bd in allb.items():
        b = float(dec_year(pd.Series([bd]))[0])
        s_ = sd[["t", "y", "se", "ls"]].copy()
        s_["pw"] = 1 / (s_["se"] ** 2 + 0.02 ** 2)
        r = seg_fit(s_, b, W=5.0, weights="pw")
        rows.append({"break": nm, "jump [95% CI] (precision-weighted)": fmt(*r["jump at break"], d=4), "slope change/yr": fmt(*r["slope change/yr"], d=4)})
    out.append("Sensitivity: precision-weighted (weights 1/(se^2 + 0.02^2)), unadjusted series:\n")
    out.append(md(pd.DataFrame(rows)) + "\n")
    save("t4_segmented_pw", pd.DataFrame(rows))
    # global model comparison
    t = sd["t"].to_numpy()
    y = sd["y"].to_numpy()
    bg = [float(dec_year(pd.Series([d_]))[0]) for d_ in GOV_CHANGES.values()]
    bp = [float(dec_year(pd.Series([d_]))[0]) for d_ in PLACEBO.values()]
    models = {"linear": design_mat_breaks(t, []),
              "quadratic": np.column_stack([np.ones(len(t)), t - 2000, (t - 2000) ** 2]),
              "cubic": np.column_stack([np.ones(len(t)), t - 2000, (t - 2000) ** 2, (t - 2000) ** 3]),
              "linear + jumps & slope changes at 3 govt changes": design_mat_breaks(t, bg),
              "linear + jumps only at 3 govt changes": np.column_stack([np.ones(len(t)), t - 2000] + [(t >= b).astype(float) for b in bg]),
              "linear + jumps & slope changes at 2 placebo dates": design_mat_breaks(t, bp),
              "linear + jumps only at 2 placebo dates": np.column_stack([np.ones(len(t)), t - 2000] + [(t >= b).astype(float) for b in bp]),
              "linear + jumps at all 5 LS boundaries (jumps only)": np.column_stack([np.ones(len(t)), t - 2000] + [(t >= b).astype(float) for b in bg + bp])}
    rows = []
    n = len(t)
    for nm, X in models.items():
        beta, V, e, h = ols_hc3(X, y)
        rss = float((e ** 2).sum())
        k = X.shape[1]
        loo = e / (1 - h)
        rows.append({"model": nm, "k": k, "R2": 1 - rss / ((y - y.mean()) ** 2).sum(), "AIC": n * np.log(rss / n) + 2 * k,
                     "LOO-CV RMSE": float(np.sqrt((loo ** 2).mean())), "resid SD": np.sqrt(rss / (n - k))})
    mc = pd.DataFrame(rows)
    save("t4_model_comparison", mc)
    out.append("### 4c. Does a smooth trend fit as well as a trend with breaks? (78 session means, unadjusted)\n")
    out.append(md(mc, "{:,.4f}") + "\n")
    # joint Wald tests of the break terms ADDED to a smooth cubic trend (the relevant null: smooth nonlinear trend, no breaks at elections)
    def wald_joint(X, idxs):
        beta, V, e, h = ols_hc3(X, y)
        bb = beta[idxs]
        Vs = V[np.ix_(idxs, idxs)]
        stat = float(bb @ np.linalg.solve(Vs, bb))
        F = stat / len(idxs)
        return stat, len(idxs), float(stats.f.sf(F, len(idxs), n - X.shape[1]))
    cub = models["cubic"]
    def add_breaks(bk, slopes):
        cols = []
        for b_ in bk:
            post = (t >= b_).astype(float)
            cols.append(post)
            if slopes:
                cols.append(post * (t - b_))
        return np.column_stack([cub] + cols)
    wj = [("3 govt changes: jumps only (on top of cubic)", add_breaks(bg, False), [4, 5, 6]),
          ("3 govt changes: jumps + slope changes (on top of cubic)", add_breaks(bg, True), [4, 5, 6, 7, 8, 9]),
          ("2 placebo dates: jumps only (on top of cubic)", add_breaks(bp, False), [4, 5]),
          ("2 placebo dates: jumps + slope changes (on top of cubic)", add_breaks(bp, True), [4, 5, 6, 7])]
    rows = [{"joint test (all break terms = 0)": nm, "Wald chi2": w[0], "df": w[1], "p (F approx)": w[2]} for nm, X, ix in wj for w in [wald_joint(X, ix)]]
    wjt = pd.DataFrame(rows)
    save("t4_joint_tests", wjt)
    out.append("Joint HC3 Wald tests of break terms added to a smooth cubic time trend (null = smooth nonlinear trend):\n")
    out.append(md(wjt, "{:,.3f}") + "\n")
    # LOO for cubic + breaks
    rows = []
    for nm, X, ix in wj:
        beta, V, e, h = ols_hc3(X, y)
        loo = e / (1 - h)
        rows.append({"model": "cubic + " + nm.replace(" (on top of cubic)", ""), "k": X.shape[1], "LOO-CV RMSE": float(np.sqrt((loo ** 2).mean()))})
    out.append("Leave-one-session-out RMSE of cubic + break terms (cubic alone: " + f"{mc.loc[mc['model'] == 'cubic', 'LOO-CV RMSE'].iloc[0]:.4f}" + "):\n")
    out.append(md(pd.DataFrame(rows), "{:,.4f}") + "\n")
    save("t4_cubic_plus_breaks", pd.DataFrame(rows))
    # hinge (flat-then-decline) model with estimated kink
    def hinge_fit(yv):
        best = None
        for kap in np.arange(2008.0, 2025.01, 0.1):
            X = np.column_stack([np.ones(n), np.maximum(t - kap, 0)])
            bb = np.linalg.lstsq(X, yv, rcond=None)[0]
            rss = ((yv - X @ bb) ** 2).sum()
            if best is None or rss < best[0]:
                best = (rss, kap, bb)
        return best
    rss_h, kap_h, bb_h = hinge_fit(y)
    Xh = np.column_stack([np.ones(n), np.maximum(t - kap_h, 0)])
    eh = y - Xh @ bb_h
    kb = []
    for _ in range(500):
        yb = Xh @ bb_h + rng.choice(eh, size=n, replace=True)
        kb.append(hinge_fit(yb)[1])
    # compare with a global linear trend and the break models on LOO
    hh = np.einsum("ij,jk,ik->i", Xh, np.linalg.inv(Xh.T @ Xh), Xh)
    loo_h = float(np.sqrt(((eh / (1 - hh)) ** 2).mean()))
    X2 = np.column_stack([np.ones(n), t - 2000, np.maximum(t - kap_h, 0)])
    b2 = np.linalg.lstsq(X2, y, rcond=None)[0]
    e2 = y - X2 @ b2
    h2 = np.einsum("ij,jk,ik->i", X2, np.linalg.inv(X2.T @ X2), X2)
    loo2 = float(np.sqrt(((e2 / (1 - h2)) ** 2).mean()))
    out.append("### 4c2. A single estimated kink (hinge) instead of election breaks\n")
    out.append(f"y = a + c * max(t - kappa, 0) (flat before kappa), kappa searched on 2008-2025: kappa_hat = {kap_h:.1f} "
               f"(residual-bootstrap 95% interval {np.percentile(kb, 2.5):.1f} to {np.percentile(kb, 97.5):.1f}, B=500); slope after kappa = {bb_h[1]:+.4f} per year; "
               f"R2 = {1 - ((y - Xh @ bb_h) ** 2).sum() / ((y - y.mean()) ** 2).sum():.3f}; LOO RMSE = {loo_h:.4f} "
               f"(vs cubic {mc.loc[mc['model'] == 'cubic', 'LOO-CV RMSE'].iloc[0]:.4f}, 3 govt-change breaks "
               f"{mc.loc[mc['model'] == 'linear + jumps & slope changes at 3 govt changes', 'LOO-CV RMSE'].iloc[0]:.4f}). "
               f"Linear trend + kink: pre-kink slope {b2[1]:+.4f}/yr, extra slope after kappa {b2[2]:+.4f}/yr, LOO RMSE {loo2:.4f}.\n")
    KEY["hinge"] = dict(kappa=float(kap_h), kappa_ci=[float(np.percentile(kb, 2.5)), float(np.percentile(kb, 97.5))], slope_after=float(bb_h[1]), loo=loo_h)
    # sup-F scan
    cand = sd["t"].to_numpy()
    lo_i, hi_i = int(0.15 * n), int(0.85 * n)
    def supF(yv):
        best = (-1, None)
        Fs = []
        for i in range(lo_i, hi_i):
            b = (cand[i - 1] + cand[i]) / 2
            post = (t >= b).astype(float)
            X1 = np.column_stack([np.ones(n), t - b, post, post * (t - b)])
            X0 = np.column_stack([np.ones(n), t - b])
            r1 = ((yv - X1 @ np.linalg.lstsq(X1, yv, rcond=None)[0]) ** 2).sum()
            r0 = ((yv - X0 @ np.linalg.lstsq(X0, yv, rcond=None)[0]) ** 2).sum()
            F = ((r0 - r1) / 2) / (r1 / (n - 4))
            Fs.append((F, b))
        return max(Fs), Fs
    (Fmax, bmax), Fs = supF(y)
    X0 = np.column_stack([np.ones(n), t - 2000])
    b0 = np.linalg.lstsq(X0, y, rcond=None)[0]
    res0 = y - X0 @ b0
    sims = []
    for _ in range(500):
        yb = X0 @ b0 + rng.choice(res0, size=n, replace=True)
        sims.append(supF(yb)[0][0])
    pval = float((np.array(sims) >= Fmax).mean())
    top = sorted(Fs, reverse=True)[:5]
    out.append("### 4d. Unknown-break scan (sup-F for a level + slope break in a linear trend)\n")
    out.append(f"sup-F={Fmax:.2f} at t={bmax:.1f} (candidate breaks between sessions, 15% trimming); residual-bootstrap p={pval:.3f} (B=500, iid residuals, "
               f"sessions treated as exchangeable so p is approximate). Top-5 candidate dates: " + ", ".join(f"{b:.1f} (F={F:.1f})" for F, b in top) +
               ". Government changes in decimal years: " + ", ".join(f"{k.split()[0]}={float(dec_year(pd.Series([v]))[0]):.1f}" for k, v in GOV_CHANGES.items()) + ".\n")
    KEY["supF"] = dict(F=Fmax, t=bmax, p=pval)
    KEY["seg"] = seg[["series", "break", "jump [95% CI]", "slope change/yr [95% CI]", "pre-trend extrapolation gap [95% CI]"]].to_dict("records")
    KEY["joint"] = wjt.to_dict("records")
    KEY["modelcmp"] = mc.to_dict("records")
    # figure 3
    fig, axs = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    colp = {"NDA-V3": "#cccccc", "UPA-1": "#aad3ee", "UPA-2": "#cfe8f6", "NDA-M1": "#f6d2b7", "NDA-M2": "#f4b98a", "NDA-M3": "#eba66b"}
    for ax, col, ttl in ((axs[0], "y", "Unadjusted design-weighted session mean of y_m2"), (axs[1], "y_adj", "Adjusted for complexity controls (S5)")):
        for gp, d in sd.groupby("govt_period"):
            ax.axvspan(d["t"].min() - 0.2, d["t"].max() + 0.2, color=colp.get(gp, "#eee"), alpha=.5, lw=0)
            ax.text((d["t"].min() + d["t"].max()) / 2, 0.985, gp, ha="center", va="top", fontsize=7, transform=ax.get_xaxis_transform())
        ax.errorbar(sd["t"], sd[col], yerr=1.96 * sd["se"], fmt="o", ms=3, color="#222", lw=.7, capsize=1.5)
        # per-LS linear fits
        for ls, d in sd.groupby("ls"):
            Xd = np.column_stack([np.ones(len(d)), d["t"]])
            bb = np.linalg.lstsq(Xd, d[col], rcond=None)[0]
            ax.plot(d["t"], Xd @ bb, color="#D55E00", lw=2)
        Xg_ = np.column_stack([np.ones(n), t])
        bb = np.linalg.lstsq(Xg_, sd[col], rcond=None)[0]
        ax.plot(t, Xg_ @ bb, color="#0072B2", lw=1.3, ls="--", label="single linear trend")
        for nm, bd in GOV_CHANGES.items():
            ax.axvline(float(dec_year(pd.Series([bd]))[0]), color="red", lw=1, ls=":")
        for nm, bd in PLACEBO.items():
            ax.axvline(float(dec_year(pd.Series([bd]))[0]), color="grey", lw=1, ls=":")
        ax.set_ylabel("mean question score")
        ax.set_title(ttl, fontsize=9)
        ax.grid(alpha=.25)
    axs[0].plot([], [], color="#D55E00", lw=2, label="linear fit within each LS")
    axs[0].legend(fontsize=7.5, loc="lower left")
    axs[1].set_xlabel("Session date (red dotted: change of government; grey dotted: re-election of same government)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "rq2_fig3_sessions_breaks.png"), dpi=170)
    plt.close(fig)
    out.append("Figure: `rq2_fig3_sessions_breaks.png`\n")
    return "\n".join(out)


SUMMARY = """## What holds and what does not (all CIs 95%)

**Holds**
1. The decline is not an LLM drift artefact. Human gold (design weighted, n=400 questions, 1,636 parts) falls in step with M2: late minus early gold -0.048 [-0.121, 0.025] vs M2 on the same questions -0.044 [-0.109, 0.025]; LS18 minus LS13 gold -0.115 [-0.212, -0.007] vs M2 -0.103 [-0.204, 0.006]. Gold alone is underpowered (LS level CIs about +-0.05 to +-0.09), so the evidence is "consistent and not contradicted", not gold-only proof.
2. Gemini's bias is stable over time. M2 minus gold is +0.024 [0.009, 0.041] overall (M2 lenient), homogeneity across LS13-18 p=0.96 (question level) and p=0.91 (part level), bias slope 0.000 [-0.014, 0.015] per decade. Part level accuracy 0.89 (early) / 0.86 (middle) / 0.87 (late), kappa 0.71 / 0.69 / 0.73; sensitivity for non-FULL about 0.80 in every period. M2 under-detects non-FULL parts if anything (gold non-FULL 22.5% -> 33.3%, M2 on same parts 20.7% -> 29.9%).
3. Bias-corrected (PPI++) means: LS13 0.832 [0.803, 0.860], LS17 0.797 [0.766, 0.829], LS18 0.738 [0.697, 0.780]; LS18 minus LS13 -0.093 [-0.135, -0.051] (naive -0.080 [-0.099, -0.062]); late minus early -0.060 [-0.096, -0.023]; trend -0.027 [-0.044, -0.009] per decade (naive -0.027 [-0.035, -0.019]). Correction leaves the decline intact (slightly larger).
4. Complexity does not explain all of it. LS18 minus LS13 is -0.080 [-0.097, -0.063] raw and -0.051 [-0.063, -0.038] with full controls (63% survives); trend -0.026 [-0.034, -0.018] per decade raw, -0.013 [-0.020, -0.006] controlled (51% survives; 47 to 57% across specifications; same for the share of FULL parts, 64%). Robust to flexible length (decile FE, parts FE), words per part, and asker bloc.
5. Deferral's decline is real. The formula marker (information being collected / will be laid on the Table) is in 3.9% of LS13 answers and 0.02% in LS18, M2 labels DEFERRAL for 99.8 to 100% of formula questions in every LS, and M2 DEFERRAL parts fall from 4.7 to 0.03 per 100 parts. Gold agrees (4.6 -> 0.7 per 100 parts, same in M2 on those parts; only 25 gold deferral parts, CI wide).
6. Excluding formula questions (which score about 0.11) the decline is larger: LS18 minus LS13 -0.112 [-0.129, -0.095] raw, -0.087 [-0.099, -0.075] controlled. The vanishing of deferral partly masks the rise of other shortfalls.
7. Rise of GENERIC and INCOMPLETE (M2): per 100 parts GENERIC 6.6 -> 15.2 (LS13 to LS18, +8.6 [7.8, 9.4]), INCOMPLETE 4.8 -> 13.3 (+8.5 [7.2, 9.9]); together about 111% of the +15.3 [13.1, 17.7] pp rise in non-FULL parts, offset by DEFERRAL -4.7 [-5.2, -4.3]. Within request-type change is +11.1 pp of +10.7 (composition -0.4 pp). INCOMPLETE rise is confirmed in gold (+6.6 [3.6, 9.8] early to late).
8. No break at changes of government. Jumps at first sitting of the new LS (+-5y windows, HC3): 2004 +0.021 [-0.007, 0.048], 2014 +0.005 [-0.010, 0.019], 2024 +0.007 [-0.032, 0.045]; placebo re-elections 2009 -0.015 [-0.036, 0.005], 2019 +0.006 [-0.013, 0.024]. Break models at placebo dates fit as well as at government changes (LOO RMSE 0.0134 vs 0.0137; cubic 0.0138). No level jump at the LS17/LS18 boundary, the extraction source change (API to docx) therefore shows no discontinuity.
9. The trend is flat then accelerating, not government specific. Within LS13-16 slopes are -0.004 to +0.001 per year (CIs include 0 except LS16, -0.003 [-0.006, -0.0001]); the decline is concentrated after about 2018: estimated kink 2018.3 [2016.9, 2019.7], slope -0.011 per year afterwards; within LS17 -0.009 [-0.019, 0.001]; within LS18 -0.026 [-0.041, -0.012] per year (7 sessions).

**Does not hold / caveats**
- The raw tail (LS18 0.771) is not all "worse behaviour": about 37% of the LS18 gap and half the trend is associated with longer questions (median 356 -> 644 characters, 59 -> 101 words, same pattern in every extraction source) and a shift to ACTION requests; the length gradient is also present in human gold (-0.160 [-0.268, -0.058] per log unit) and M2 does not over penalise long questions (M2 minus gold -0.007 [-0.048, 0.037]). Whether longer questions are a cause or just a marker (e.g. bundled asks) is not identified. Request type shares are M2 labels (S2 onward), so treat those controls with care.
- Controlling for answer side variables (length, annexure; mediators, S7) makes the decline larger (120%), so answers are longer and more complete in form but not in content: a sensitivity only.
- GENERIC's rise is not confirmed by gold (+0.3 [-4.9, 5.4] early to late vs M2 +5.4 [0.5, 10.2] on the same parts): humans put more late shortfalls into DIFFERENT_QUESTION, JURISDICTION and OTHER. Type agreement among non-FULL in both is 82% (92% early, 79 to 81% later), so type shares from M2 are less reliable than the non-FULL rate itself.
- The sup-F scan prefers a break near 2020.9 (inside LS17), the hinge near 2018.3: the exact timing is imprecise and neither is a government change. Joint HC3 Wald tests add little over a cubic trend (3 government dates p=0.02; placebo dates p=0.003 to 0.013), session means are autocorrelated and n=78, so treat as descriptive. Power at 2024 is low (7 post sessions, jump CI +-0.04); 2024 pre-trend extrapolation gap -0.010 [-0.040, 0.021].
- Design caveats: LS13-16 are a 1/6 sample (design weighted), LS17-18 a census, LS18 comes from different extraction (docx / pdf vs API) and is still incomplete (to Aug 2026); gold CIs use a Rao-Wu bootstrap on 400 questions with collapsed singleton strata; PPI++ CIs use two-way clusters with few sessions per LS (t with sessions-1 df).
"""


# =========================================================================================== main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="1,2,3,4")
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    want = set(a.tasks.split(","))
    df, est, g = load()
    parts = ["# RQ2 exploratory: is the decline in responsiveness real?\n",
             "Exploratory; everything run is reported. Outcome y_m2 = mean over parts of FULL=1 / PARTIAL=.5 / NONE=0 (Gemini 3.8 flash). "
             "Design weights m2_weight (LS13-16 1/6 sample, LS17-18 census). Statistics only; no answer text is quoted. "
             "Script: `analysis/explore_rq2.py`.\n"]
    gold_pack = None
    if "1" in want or "3" in want:
        txt, gold_pack = task1(df, est, g, a.boot)
        parts.append(txt)
        print("task1 done", flush=True)
    if "2" in want:
        parts.append(task2(df, est, g, a.boot))
        print("task2 done", flush=True)
    if "3" in want:
        parts.append(task3(df, est, gold_pack))
        print("task3 done", flush=True)
    if "4" in want:
        parts.append(task4(df, est))
        print("task4 done", flush=True)
    with open(os.path.join(OUT, "rq2_trend_body.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))
    if want >= {"1", "2", "3", "4"}:
        with open(os.path.join(OUT, "rq2_trend.md"), "w", encoding="utf-8") as fh:
            fh.write(parts[0] + "\n" + SUMMARY + "\n" + "\n".join(parts[1:]))
    import json
    with open(os.path.join(OUT, "rq2_key.json"), "w", encoding="utf-8") as fh:
        json.dump(KEY, fh, indent=1, default=str)
    print(json.dumps(KEY, indent=1, default=str)[:6000])


if __name__ == "__main__":
    main()
