"""Exploratory RQ4 (where ministers evade) and RQ1 (measurement) analyses. NEW FILE ONLY; reads existing data, writes
results/explore/*. No pipeline file, data file or the frame is modified.

    python -I -X utf8 analysis/explore_label_alt.py      # (Gemini, ~$0.44) -> data/measures/m2_gold_alt_3.1fl.parquet
    python -I -X utf8 analysis/explore_rq4_rq1.py        # everything else -> results/explore/_md_body.md, CSVs, PNGs

Estimation (task A): M2 labels are a stratified probability sample of questions (m2_sample.csv: strata ls|session|qtype,
census in LS17/18, ~1/6 random in LS13-16, design weight N_h/n_h).  Domain means are ratio estimators
R = sum(w*y)/sum(w*m) over PARTS (m = parts per question) restricted to analysis_sample questions; variance by Taylor
linearisation with question as the cluster, strata as in the design and the finite-population correction
(census strata contribute 0); domain estimation keeps every sampled question in the variance (u=0 outside the domain).
Score: FULL=1, PARTIAL=.5, NONE=0.  95% CI = R +/- 1.96 se (clipped to [0,1]).
"""
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = C.ROOT
MEAS = C.MEASURES
OUT = os.path.join(C.RESULTS, "explore")
os.makedirs(OUT, exist_ok=True)
SCORE = {"FULL": 1.0, "PARTIAL": 0.5, "NONE": 0.0}
TYPES = ["GENERIC", "INCOMPLETE", "DIFFERENT_QUESTION", "NOT_MAINTAINED", "JURISDICTION", "DEFERRAL",
         "DOES_NOT_ARISE", "WITHHELD", "OTHER", "ANNEXURE_ONLY"]
SECURITY = ["Home Affairs", "Defence", "External Affairs"]
MD = []          # markdown chunks
NUMS = {}        # key numbers kept for the report


def md(s=""):
    MD.append(s)


def tbl(df, fmt="{:,.3f}"):
    md(C.df_to_markdown(df, floatfmt=fmt))


# --------------------------------------------------------------------------------------------- design estimation
class Design:
    """All sampled questions (census + random roles) with per-question sums; ratio estimator with strata/FPC."""

    def __init__(self):
        fr = pd.read_parquet(os.path.join(MEAS, "analysis_frame_real.parquet"),
                             columns=["qid", "ls", "ministry", "ministry_canonical", "ministry_group", "analysis_sample",
                                      "m2_weight", "m2_role", "topic"])
        s = pd.read_csv(os.path.join(MEAS, "m2_sample.csv"), usecols=["qid", "stratum", "N_stratum", "n_stratum"])
        q = fr[fr["m2_role"].isin(["census", "random"])].merge(s, on="qid", how="left")
        assert q["stratum"].notna().all()
        q["n_h"] = q.groupby("stratum")["qid"].transform("size")
        chk = (q["n_h"] == q["n_stratum"]).mean()
        print(f"stratum n check (sampled count == n_stratum): {chk:.4f}")
        q["fpc"] = 1.0 - q["n_h"] / q["N_stratum"]
        lab = pd.read_parquet(os.path.join(MEAS, "m2_labels.parquet"),
                              columns=["part_id", "qid", "label", "evasion_type"])
        lab["score"] = lab["label"].map(SCORE)
        lab["loss"] = 1.0 - lab["score"]
        self.parts = lab.merge(q[["qid", "ls", "ministry_canonical", "ministry_group", "analysis_sample", "m2_weight",
                                  "topic"]], on="qid", how="inner")
        self.parts = self.parts[self.parts["m2_weight"] > 0]
        g = lab.groupby("qid")
        agg = pd.DataFrame({"m": g.size(), "score": g["score"].sum(),
                            "full": (lab["label"] == "FULL").groupby(lab["qid"]).sum()})
        for t in TYPES:
            ind = (lab["evasion_type"] == t)
            agg["n_" + t] = ind.groupby(lab["qid"]).sum()
            agg["loss_" + t] = (lab["loss"] * ind).groupby(lab["qid"]).sum()
        q = q.merge(agg, left_on="qid", right_index=True, how="left")
        for c in agg.columns:
            q[c] = q[c].fillna(0.0)
        q["nonfull"] = q["m"] - q["full"]
        q["loss_all"] = q["m"] - q["score"]
        self.q = q.reset_index(drop=True)
        codes, uniq = pd.factorize(self.q["stratum"])
        self.sid = codes
        self.nstr = len(uniq)
        h = self.q.groupby("stratum").agg(n=("n_h", "first"), f=("fpc", "first")).reindex(uniq)
        self.n_h = h["n"].to_numpy(float)
        self.f_h = h["f"].to_numpy(float)
        self.w = self.q["m2_weight"].to_numpy(float)
        self.in_sample = self.q["analysis_sample"].fillna(False).to_numpy(bool)
        print(f"design: {len(self.q):,} sampled questions in {self.nstr} strata; "
              f"{int(self.in_sample.sum()):,} in analysis sample; parts used {len(self.parts):,}", flush=True)

    def _var_total(self, u):
        su = np.bincount(self.sid, u, self.nstr)
        su2 = np.bincount(self.sid, u * u, self.nstr)
        with np.errstate(divide="ignore", invalid="ignore"):
            s2 = np.where(self.n_h > 1, (su2 - su ** 2 / self.n_h) / (self.n_h - 1), 0.0)
        return (self.n_h * (1 - self.f_h) * s2).sum()   # u already contains the weight w_i

    def ratio(self, num, den, domain=None):
        """ratio estimator sum(w num)/sum(w den) over the domain and its linearised SE."""
        d = self.in_sample if domain is None else (domain & self.in_sample)
        num = np.asarray(num, float) * d
        den = np.asarray(den, float) * d
        sw_den = (self.w * den).sum()
        if sw_den <= 0:
            return np.nan, np.nan
        R = (self.w * num).sum() / sw_den
        u = self.w * (num - R * den)
        return R, np.sqrt(max(self._var_total(u), 0)) / sw_den

    def est(self, num_col, domain=None, den_col="m"):
        return self.ratio(self.q[num_col], self.q[den_col], domain)

    def diff(self, col, dom_a, dom_b):
        """difference of two domain ratios (a - b) of col/m with joint linearised SE."""
        num = self.q[col].to_numpy(float)
        den = self.q["m"].to_numpy(float)
        da, db = dom_a & self.in_sample, dom_b & self.in_sample
        Ra = self.ratio(num, den, dom_a)[0]
        Rb = self.ratio(num, den, dom_b)[0]
        sa = (self.w * den * da).sum()
        sb = (self.w * den * db).sum()
        u = self.w * (np.where(da, num - Ra * den, 0) / sa - np.where(db, num - Rb * den, 0) / sb)
        return Ra - Rb, np.sqrt(max(self._var_total(u), 0))

    def parts_in(self, mask=None):
        p = self.parts[self.parts["analysis_sample"].fillna(False)]
        return p if mask is None else p[mask(p)]


def ci(R, se):
    return max(R - 1.96 * se, 0.0), min(R + 1.96 * se, 1.0)


def fmt_ci(R, se, pct=False, d=3):
    lo, hi = ci(R, se)
    if pct:
        return f"{100*R:.{d}f} [{100*lo:.{d}f}, {100*hi:.{d}f}]"
    return f"{R:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]"


# --------------------------------------------------------------------------------------------- mapping
CORE5 = {
    "Health and Family Welfare": "health", "AYUSH": "health",
    "Agriculture and Farmers Welfare": "agriculture", "Fisheries, Animal Husbandry and Dairying": "agriculture",
    "Home Affairs": "police/home",
    "Jal Shakti": "water", "Water Resources, River Development and Ganga Rejuvenation": "water",
    "Drinking Water and Sanitation": "water",
    "Education": "education",
}
SECOND = ["Rural Development", "Panchayati Raj", "Cooperation", "Housing and Urban Affairs", "Urban Development",
          "Urban Development and Poverty Alleviation", "Housing and Urban Poverty Alleviation",
          "Urban Employment and Poverty Alleviation", "Consumer Affairs, Food and Public Distribution",
          "Social Justice and Empowerment", "Women and Child Development", "Tribal Affairs", "Minority Affairs",
          "Labour and Employment", "Environment, Forest and Climate Change", "Law and Justice",
          "Law, Justice and Company Affairs", "Youth Affairs and Sports", "Skill Development and Entrepreneurship",
          "Skill Development, Entrepreneurship, Youth Affairs and Sports"]
TA = "A core State/Concurrent"
TB = "B other State/Concurrent-heavy"
TC = "C Union / other"


def tier_of(m):
    return TA if m in CORE5 else (TB if m in SECOND else TC)


# --------------------------------------------------------------------------------------------- A1
def a1_ministry(D):
    md("## A. RQ4: where ministers evade (M2 labels by gemini-3.8-flash, design-weighted)")
    md("Units are question PARTS (n = sampled parts in the analysis sample, i.e. after the pre-specified exclusions). "
       "Weights, strata and CIs: ratio estimator over parts, question clusters, design strata with finite-population "
       "correction (see script header). Score: FULL=1, PARTIAL=.5, NONE=0. Evasion shares are shares of ALL parts, so "
       "FULL% plus the type shares sum to 100%.\n")
    md("### A1. Responsiveness and evasion-type shares by canonical ministry (>= 2,000 sampled parts)")
    p = D.parts_in()
    cnt = p.groupby("ministry_canonical").size()
    keep = cnt[cnt >= 2000].index.tolist()
    ls_R = {l: D.est("score", (D.q["ls"] == l).to_numpy())[0] for l in sorted(p["ls"].unique())}
    Rall, seall = D.est("score")
    NUMS["overall_score"] = (Rall, seall)
    rows = []
    for m in keep:
        dom = (D.q["ministry_canonical"] == m).to_numpy()
        R, se = D.est("score", dom)
        full, sf = D.est("full", dom)
        r = {"ministry": m, "group": D.q.loc[dom, "ministry_group"].iloc[0], "security": m in SECURITY,
             "tier": tier_of(m), "n_parts": int(cnt[m]), "score": R, "score_se": se, "score_lo": ci(R, se)[0],
             "score_hi": ci(R, se)[1], "full": full, "full_lo": ci(full, sf)[0], "full_hi": ci(full, sf)[1]}
        for t in TYPES:
            v, s = D.est("n_" + t, dom)
            r[t] = v
            r[t + "_lo"] = ci(v, s)[0]
            r[t + "_hi"] = ci(v, s)[1]
        pm = p[p["ministry_canonical"] == m]
        sh = pm.groupby("ls")["m2_weight"].sum() / pm["m2_weight"].sum()
        exp = sum(sh[l] * ls_R[l] for l in sh.index)
        r["score_ls_adj"] = R - exp + Rall      # indirect standardisation to the overall LS mix
        r["ls_first"] = int(pm["ls"].min())
        r["ls_last"] = int(pm["ls"].max())
        rows.append(r)
    A = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    A["rank"] = np.arange(1, len(A) + 1)
    A["rank_ls_adj"] = A["score_ls_adj"].rank(ascending=False).astype(int)
    A.to_csv(os.path.join(OUT, "a1_ministry_table_full.csv"), index=False)
    NUMS["a1"] = A
    md(f"Overall design-weighted score (all ministries, analysis sample): **{Rall:.3f}** (se {seall:.4f}). "
       f"{len(A)} of {cnt.size} canonical ministries pass the 2,000-part threshold. "
       f"Security group marked with *. Full table with CIs for every type: `a1_ministry_table_full.csv`. "
       f"'rank LS-adj' = rank after indirect standardisation to the overall LS mix (parts of a ministry compared with the overall score in the same LS).\n")
    show = pd.DataFrame({
        "rank": A["rank"], "ministry": [("*" if s else "") + m for m, s in zip(A["ministry"], A["security"])],
        "parts": A["n_parts"],
        "score [95% CI]": [fmt_ci(r.score, r.score_se) for r in A.itertuples()],
        "FULL %": [f"{100*v:.1f}" for v in A["full"]],
        "GENERIC": [f"{100*v:.1f}" for v in A["GENERIC"]],
        "INCOMPL": [f"{100*v:.1f}" for v in A["INCOMPLETE"]],
        "DIFF_Q": [f"{100*v:.1f}" for v in A["DIFFERENT_QUESTION"]],
        "NOT_MAINT": [f"{100*v:.1f}" for v in A["NOT_MAINTAINED"]],
        "JURISD [CI]": [f"{100*r.JURISDICTION:.1f} [{100*r.JURISDICTION_lo:.1f}, {100*r.JURISDICTION_hi:.1f}]" for r in A.itertuples()],
        "DEFERRAL": [f"{100*v:.1f}" for v in A["DEFERRAL"]],
        "NOT_ARISE": [f"{100*v:.1f}" for v in A["DOES_NOT_ARISE"]],
        "WITHHELD [CI]": [f"{100*r.WITHHELD:.2f} [{100*r.WITHHELD_lo:.2f}, {100*r.WITHHELD_hi:.2f}]" for r in A.itertuples()],
        "OTHER+ANNEX": [f"{100*(r.OTHER + r.ANNEXURE_ONLY):.1f}" for r in A.itertuples()],
        "rank LS-adj": A["rank_ls_adj"]})
    tbl(show)
    from scipy.stats import spearmanr
    rho = spearmanr(A["score"], A["score_ls_adj"])[0]
    nsig = int(((A.score_lo > Rall) | (A.score_hi < Rall)).sum())
    NUMS["a1_rho"] = rho
    NUMS["a1_nsig"] = nsig
    md(f"Rank correlation between the raw and LS-adjusted ministry ranking: Spearman {rho:.3f}. "
       f"Range of ministry scores: {A['score'].min():.3f} ({A.iloc[-1]['ministry']}) to {A['score'].max():.3f} ({A.iloc[0]['ministry']}). "
       f"Ministries whose CI excludes the overall mean: {nsig} of {len(A)}.\n")
    sec = A[A["security"]]
    md("Security group ranks (1 = most responsive): " + "; ".join(
        f"{r.ministry} #{r.rank} of {len(A)} ({r.score:.3f})" for r in sec.itertuples()) + ".\n")
    fig, ax = plt.subplots(figsize=(8.5, 11))
    y = np.arange(len(A))[::-1]
    cols = ["#c0392b" if s else "#34495e" for s in A["security"]]
    ax.hlines(y, A["score_lo"], A["score_hi"], color=cols, lw=1.6)
    ax.scatter(A["score"], y, c=cols, s=28, zorder=3)
    ax.scatter(A["score_ls_adj"], y, facecolors="none", edgecolors="#7f8c8d", s=22, zorder=2, label="LS-composition adjusted")
    ax.axvline(Rall, color="#888", ls="--", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([m[:44] for m in A["ministry"]], fontsize=7.5)
    for t, s in zip(ax.get_yticklabels(), A["security"]):
        if s:
            t.set_color("#c0392b")
            t.set_fontweight("bold")
    ax.set_xlabel("Design-weighted responsiveness (mean part score; FULL=1, PARTIAL=.5, NONE=0), 95% CI")
    ax.set_title("Responsiveness by ministry (>= 2,000 sampled parts); security in red")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig1_ministry_responsiveness.png"), dpi=160)
    plt.close(fig)
    return A


# --------------------------------------------------------------------------------------------- A2
def a2_jurisdiction(D, A):
    md("### A2. Jurisdiction deflection (JURISDICTION, i.e. 'State subject')")
    md("**Mapping used (approximate, by ministry name only, from the constitutional List II / List III subject matter):** "
       "Tier A = health (Health and Family Welfare, AYUSH), agriculture (Agriculture and Farmers Welfare, Fisheries/Animal "
       "Husbandry/Dairying), police/home (Home Affairs), water (Jal Shakti, Water Resources, Drinking Water and Sanitation), "
       "education (Education). Tier B = other State/Concurrent-heavy ministries: Rural Development, Panchayati Raj, "
       "Cooperation, Housing/Urban (all variants), Consumer Affairs/Food/PDS, Social Justice, Women and Child "
       "Development, Tribal and Minority Affairs, Labour and Employment, Environment/Forest, Law and Justice, Youth Affairs "
       "and Sports, Skill Development. Tier C = everything else (Union List or unclear: Defence, External Affairs, "
       "Railways, Finance, Civil Aviation, Communications, Petroleum, Power, Commerce, Mines, Coal, Steel, Space, ...). "
       "This is a ministry-level proxy only: a Home Affairs question can concern a Union matter and a Tier C ministry can "
       "be asked about a State subject, so it is approximate; a topic-cluster check follows.\n")
    q = D.q
    tier = q["ministry_canonical"].map(tier_of)
    rows = []
    tiers = [TA, TB, TC]
    for t in tiers:
        dom = (tier == t).to_numpy()
        j, sj = D.est("n_JURISDICTION", dom)
        jn, sjn = D.ratio(q["n_JURISDICTION"], q["nonfull"], dom)
        sc, ssc = D.est("score", dom)
        rows.append({"tier": t, "parts": int(q.loc[dom & D.in_sample, "m"].sum()),
                     "JURISD % of all parts [CI]": fmt_ci(j, sj, True, 2),
                     "JURISD % of non-FULL parts [CI]": fmt_ci(jn, sjn, True, 1), "score": sc})
        NUMS["tier_" + t[0]] = (j, sj, jn, sjn)
    tbl(pd.DataFrame(rows))
    pj = D.parts_in(lambda p: p["evasion_type"] == "JURISDICTION")
    shj = pj.groupby(pj["ministry_canonical"].map(tier_of))["m2_weight"].sum()
    shj = shj / shj.sum()
    pa = D.parts_in()
    sha = pa.groupby(pa["ministry_canonical"].map(tier_of))["m2_weight"].sum()
    sha = sha / sha.sum()
    NUMS["tier_share_J"] = shj
    md("Where JURISDICTION evasions fall (population-weighted): " + "; ".join(
        f"Tier {k[0]}: {100*shj[k]:.1f}% of all JURISDICTION parts (vs {100*sha[k]:.1f}% of all parts)" for k in shj.index) + ".\n")
    ltab = D.parts_in(lambda p: p["evasion_type"] == "JURISDICTION")
    Jt = A[["ministry", "n_parts", "JURISDICTION", "JURISDICTION_lo", "JURISDICTION_hi", "tier"]].sort_values("JURISDICTION", ascending=False)
    top = Jt.head(12).copy()
    top["JURISD % [CI]"] = [f"{100*r.JURISDICTION:.2f} [{100*r.JURISDICTION_lo:.2f}, {100*r.JURISDICTION_hi:.2f}]" for r in top.itertuples()]
    md("Top 12 ministries (>= 2,000 parts) by JURISDICTION share of all parts:\n")
    tbl(top[["ministry", "n_parts", "JURISD % [CI]", "tier"]])
    from scipy.stats import mannwhitneyu
    A2 = A.copy()
    A2["isAB"] = ~A2["tier"].str.startswith("C")
    mw = mannwhitneyu(A2[A2.isAB].JURISDICTION, A2[~A2.isAB].JURISDICTION)[1]
    NUMS["a2_mw"] = mw
    md(f"Among the {len(A2)} ministries, median JURISDICTION share: Tier A {100*A2[A2.tier.str.startswith('A')].JURISDICTION.median():.2f}%, "
       f"Tier B {100*A2[A2.tier.str.startswith('B')].JURISDICTION.median():.2f}%, Tier C {100*A2[~A2.isAB].JURISDICTION.median():.2f}%. "
       f"Of the 10 highest ministries, {int(Jt.head(10)['tier'].str[0].isin(['A', 'B']).sum())} are Tier A/B "
       f"({int(Jt.head(10)['tier'].str.startswith('A').sum())} Tier A). Mann-Whitney (ministry level, A+B vs C) p = {mw:.4f}.\n")
    lss = sorted(q["ls"].unique())
    years = {13: "1999-2004", 14: "2004-09", 15: "2009-14", 16: "2014-19", 17: "2019-24", 18: "2024-"}
    md("**Over time (Lok Sabha): JURISDICTION evasions as % of all parts, by tier [95% CI]:**\n")
    rows = []
    ts = {}
    for l in lss:
        r = {"LS": l, "years": years[l]}
        dl = (q["ls"] == l).to_numpy()
        j, sj = D.est("n_JURISDICTION", dl)
        r["all"] = fmt_ci(j, sj, True, 2)
        ts[("all", l)] = (j, sj)
        for t in tiers:
            j, sj = D.est("n_JURISDICTION", dl & (tier == t).to_numpy())
            r["Tier " + t[0]] = fmt_ci(j, sj, True, 2)
            ts[(t[0], l)] = (j, sj)
        rows.append(r)
    tbl(pd.DataFrame(rows))
    NUMS["jur_ls"] = ts
    rows = []
    for l in lss:
        dl = (q["ls"] == l).to_numpy()
        r = {"LS": l, "score": f"{D.est('score', dl)[0]:.3f}", "non-FULL % of parts": 100 * D.ratio(q["nonfull"], q["m"], dl)[0]}
        for t in ["JURISDICTION", "NOT_MAINTAINED", "DEFERRAL", "GENERIC", "INCOMPLETE", "DIFFERENT_QUESTION", "WITHHELD", "DOES_NOT_ARISE"]:
            r[t] = 100 * D.ratio(q["n_" + t], q["nonfull"], dl)[0]
        rows.append(r)
    md("Evasion-type mix among non-FULL parts (% of non-FULL parts), by LS, all ministries:\n")
    tbl(pd.DataFrame(rows), "{:,.1f}")
    NUMS["mix_ls"] = pd.DataFrame(rows)
    md("**JURISDICTION share (% of all parts) in selected ministry groups by LS** (blank = fewer than 100 sampled parts in that cell; point estimates):\n")
    groups = {"Health (Health&FW + AYUSH)": [m for m, g in CORE5.items() if g == "health"],
              "Agriculture (+Fisheries/AH)": [m for m, g in CORE5.items() if g == "agriculture"],
              "Home Affairs": ["Home Affairs"], "Water (Jal Shakti etc.)": [m for m, g in CORE5.items() if g == "water"],
              "Education": ["Education"], "Rural Development": ["Rural Development"],
              "Housing/Urban": [m for m in SECOND if m.startswith(("Housing", "Urban"))],
              "Defence": ["Defence"], "External Affairs": ["External Affairs"], "Railways": ["Railways"], "Finance": ["Finance"]}
    heat = pd.DataFrame(index=list(groups), columns=lss, dtype=float)
    for g, ms in groups.items():
        for l in lss:
            dom = ((q["ministry_canonical"].isin(ms)) & (q["ls"] == l)).to_numpy()
            if q.loc[dom & D.in_sample, "m"].sum() >= 100:
                heat.loc[g, l] = D.est("n_JURISDICTION", dom)[0]
    hs = heat.copy()
    hs.columns = [f"LS{c}" for c in hs.columns]
    hshow = hs.map(lambda v: "" if pd.isna(v) else f"{100*v:.1f}").reset_index().rename(columns={"index": "group"})
    tbl(hshow)
    NUMS["heat"] = heat
    tt = pd.read_csv(os.path.join(MEAS, "topic_terms.csv"))
    pa = D.parts_in()
    g = pa.groupby("topic").apply(lambda d: pd.Series({
        "parts": len(d), "JURISD_share": np.average(d["evasion_type"] == "JURISDICTION", weights=d["m2_weight"])}))
    g = g.reset_index().merge(tt[["topic", "top_terms"]], on="topic")
    g["top_ministry"] = [pa[pa.topic == t].groupby("ministry_canonical")["m2_weight"].sum().idxmax() for t in g["topic"]]
    g = g.sort_values("JURISD_share", ascending=False)
    md("Topic-cluster cross-check (K=50 TF-IDF topics from question text only, `topic` in the frame): the 8 topics with the highest "
       "JURISDICTION share of parts (weighted):\n")
    t8 = g.head(8).copy()
    t8["JURISD %"] = (100 * t8["JURISD_share"]).round(1)
    t8["parts"] = t8["parts"].astype(int)
    t8["top_terms"] = t8["top_terms"].str.split().str[:7].str.join(" ")
    tbl(t8[["topic", "parts", "JURISD %", "top_ministry", "top_terms"]], "{:,.1f}")
    g["modal_tier"] = g["top_ministry"].map(tier_of)
    md("Mean topic JURISDICTION share by tier of the topic's modal ministry (unweighted over topics): " + "; ".join(
        f"Tier {k[0]}: {100*v:.2f}% (n={int((g.modal_tier == k).sum())} topics)" for k, v in g.groupby("modal_tier")["JURISD_share"].mean().items()) + ".\n")
    g.to_csv(os.path.join(OUT, "a2_topic_jurisdiction.csv"), index=False)
    NUMS["topics"] = g


# --------------------------------------------------------------------------------------------- A3
def a3_security(D, A):
    md("### A3. Security ministries (Home Affairs, Defence, External Affairs): which evasion types drive the gap?")
    q = D.q
    sec = q["ministry_canonical"].isin(SECURITY).to_numpy()
    oth = ~sec
    Rs, ses = D.est("score", sec)
    Ro, seo = D.est("score", oth)
    fs = D.est("full", sec)[0]
    fo = D.est("full", oth)[0]
    dif, sd = D.diff("loss_all", sec, oth)
    NUMS["gap"] = (dif, sd, Rs, ses, Ro, seo)
    md(f"Security score **{fmt_ci(Rs, ses)}** vs other ministries **{fmt_ci(Ro, seo)}**; FULL share {100*fs:.1f}% vs {100*fo:.1f}%. "
       f"Gap in mean score loss (security minus other) = {dif:+.3f}, 95% CI [{dif-1.96*sd:+.3f}, {dif+1.96*sd:+.3f}] (joint linearised SE). "
       "Score loss (1 - score) decomposes exactly by evasion type: a PARTIAL part costs .5, a NONE part costs 1, summed over "
       "parts of that type and divided by all parts.\n")
    rows = []
    gap_total = dif
    dec = {}
    for t in TYPES:
        ls_, _ = D.est("loss_" + t, sec)
        lo_, _ = D.est("loss_" + t, oth)
        ns = D.est("n_" + t, sec)[0]
        no = D.est("n_" + t, oth)[0]
        d_, s_ = D.diff("loss_" + t, sec, oth)
        dec[t] = (ls_, lo_)
        rows.append({"type": t, "share of parts, security %": 100 * ns, "other %": 100 * no,
                     "ratio sec/other": ns / no if no > 0 else np.nan,
                     "loss security": ls_, "loss other": lo_, "diff (sec - other)": d_,
                     "95% CI of diff": f"[{d_-1.96*s_:+.4f}, {d_+1.96*s_:+.4f}]", "% of gap": 100 * d_ / gap_total})
    T = pd.DataFrame(rows)
    NUMS["a3_dec"] = T
    tbl(T, "{:,.3f}")
    rows = []
    for name, dom in (("Security", sec), ("Other ministries", oth), ("Home Affairs", (q.ministry_canonical == "Home Affairs").to_numpy()),
                      ("Defence", (q.ministry_canonical == "Defence").to_numpy()),
                      ("External Affairs", (q.ministry_canonical == "External Affairs").to_numpy())):
        r = {"group": name, "parts": int(q.loc[dom & D.in_sample, "m"].sum()), "score": f"{D.est('score', dom)[0]:.3f}",
             "non-FULL %": 100 * D.ratio(q["nonfull"], q["m"], dom)[0]}
        for t in TYPES[:9]:
            r[t[:9]] = 100 * D.ratio(q["n_" + t], q["nonfull"], dom)[0]
        rows.append(r)
    md("Evasion-type mix among non-FULL parts (% of non-FULL parts):\n")
    tbl(pd.DataFrame(rows), "{:,.1f}")
    NUMS["a3_mix"] = pd.DataFrame(rows)
    rows = []
    for m in SECURITY:
        dom = (q.ministry_canonical == m).to_numpy()
        Rm = D.est("score", dom)[0]
        r = {"ministry": m, "score": f"{Rm:.3f}", "loss minus other": (1 - Rm) - (1 - Ro)}
        for t in TYPES[:9]:
            r[t[:9]] = D.est("loss_" + t, dom)[0] - dec[t][1]
        rows.append(r)
    md("Per-ministry decomposition: score loss attributable to each type minus the same quantity for all other ministries (columns sum to 'loss minus other', up to ANNEXURE_ONLY):\n")
    tbl(pd.DataFrame(rows), "{:+.4f}")
    NUMS["a3_perm"] = pd.DataFrame(rows)
    lss = sorted(q.ls.unique())
    ps = D.parts_in(lambda p: p["ministry_canonical"].isin(SECURITY))
    wls = ps.groupby("ls")["m2_weight"].sum()
    wls = wls / wls.sum()
    tot = {}
    for t in ["ALL"] + TYPES:
        col = "loss_all" if t == "ALL" else "loss_" + t
        a_s = a_o = 0.0
        for l in lss:
            dl = (q.ls == l).to_numpy()
            a_s += wls.get(l, 0) * D.est(col, sec & dl)[0]
            a_o += wls.get(l, 0) * D.est(col, oth & dl)[0]
        tot[t] = (a_s, a_o)
    md(f"LS-standardised check (cell means within each LS, averaged with security's own LS distribution): security loss {tot['ALL'][0]:.3f} "
       f"vs other {tot['ALL'][1]:.3f} (gap {tot['ALL'][0]-tot['ALL'][1]:+.3f}); share of that gap by type: " +
       "; ".join(f"{t} {100*(tot[t][0]-tot[t][1])/(tot['ALL'][0]-tot['ALL'][1]):.0f}%" for t in TYPES[:9]) + ".\n")
    NUMS["a3_std"] = tot
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.8))
    ts = NUMS["jur_ls"]
    col = {"all": "#000", "A": "#c0392b", "B": "#e67e22", "C": "#2c7fb8"}
    lab = {"all": "all ministries", "A": "Tier A core State/Concurrent", "B": "Tier B other State/Concurrent", "C": "Tier C Union/other"}
    for k in ["A", "B", "C", "all"]:
        v = np.array([ts[(k, l)][0] for l in lss]) * 100
        s = np.array([ts[(k, l)][1] for l in lss]) * 100
        axs[0].errorbar(lss, v, yerr=1.96 * s, color=col[k], marker="o", capsize=2, label=lab[k], lw=2 if k == "all" else 1.3)
    axs[0].set_xlabel("Lok Sabha")
    axs[0].set_ylabel("JURISDICTION evasions, % of all parts")
    axs[0].set_title("State-subject deflection over time, by ministry tier")
    axs[0].legend(fontsize=8)
    T2 = T[T["type"].isin(TYPES[:9])].copy()
    x = np.arange(len(T2))
    wdt = 0.38
    axs[1].bar(x - wdt / 2, T2["loss security"], wdt, color="#c0392b", label="security")
    axs[1].bar(x + wdt / 2, T2["loss other"], wdt, color="#34495e", label="other ministries")
    axs[1].set_xticks(x)
    axs[1].set_xticklabels([t.replace("_", "\n") for t in T2["type"]], fontsize=7)
    axs[1].set_ylabel("score loss attributable to type (per part)")
    axs[1].set_title(f"Security vs other: loss by evasion type (total gap {dif:+.3f})")
    axs[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig2_jurisdiction_security.png"), dpi=160)
    plt.close(fig)


# --------------------------------------------------------------------------------------------- B4
def auc(pos, neg):
    """P(pos > neg) + .5 P(=)  (Mann-Whitney)."""
    from scipy.stats import rankdata
    r = rankdata(np.concatenate([pos, neg]))
    n1, n2 = len(pos), len(neg)
    return (r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n2)


def wauc(pos, neg, wp, wn):
    o = np.argsort(neg)
    s = neg[o]
    cw = np.cumsum(wn[o])
    lo = np.searchsorted(s, pos, "left")
    hi = np.searchsorted(s, pos, "right")
    lt = np.where(lo > 0, cw[np.maximum(lo - 1, 0)], 0.0)
    le = np.where(hi > 0, cw[np.maximum(hi - 1, 0)], 0.0)
    return float((wp * (lt + 0.5 * (le - lt))).sum() / (wp.sum() * cw[-1]))


def gold_frame():
    g = pd.read_csv(os.path.join(C.CODING, "gold_labels.csv"))
    ms = pd.read_csv(os.path.join(C.CODING, "main_sample.csv"))
    ms = ms[ms["in_estimate"].astype(bool)][["qid", "ls", "weight", "stratum", "ministry_group"]]
    g = g.merge(ms, on="qid", how="inner")
    g["score"] = g["label"].map(SCORE)
    return g


def b4_embeddings():
    md("## B. RQ1: measurement")
    md("### B4. Why embeddings fail: M1 cosine on human-gold parts")
    g = gold_frame()
    m1 = pd.read_parquet(os.path.join(MEAS, "m1_scores.parquet"), columns=["part_id", "m1_cos", "m1_cos_untuned", "m1_valid", "split"])
    d = g.merge(m1, on="part_id", how="left")
    n0 = len(d)
    d = d[d["m1_valid"].fillna(False)]
    n1 = len(d)
    dh = d[d["split"] == "heldout"].copy()
    md(f"Gold parts of the 400 in-estimate questions: {n0}; with a valid M1 score {n1}; heldout split (not used in M1 fine-tuning) "
       f"{len(dh)}. Everything below uses the **heldout** parts (n = {len(dh)}), unweighted; design-weighted AUCs follow the table. "
       "AUC(FULL vs group) = P(cosine of a FULL part > cosine of a part in the group); 0.5 = no discrimination, "
       "below 0.5 = the non-answers have HIGHER cosine than full answers. 95% CI: question-cluster bootstrap (B = 1000). "
       "'tuned' = the fine-tuned bi-encoder used in the study; 'untuned' = the base model.\n")
    dh["type"] = np.where(dh["label"] == "FULL", "FULL", dh["evasion_type"])
    qids = list(dh["qid"].unique())
    idx_by_q = {q: np.where(dh["qid"].values == q)[0] for q in qids}
    rng = np.random.default_rng(20261018)
    cos = dh["m1_cos"].to_numpy(float)
    cosu = dh["m1_cos_untuned"].to_numpy(float)
    typ = dh["type"].to_numpy()
    lab = dh["label"].to_numpy()
    w = dh["weight"].to_numpy(float)
    masks = {"any non-FULL": lab != "FULL", "label NONE": lab == "NONE", "label PARTIAL": lab == "PARTIAL"}
    for t in ["GENERIC", "INCOMPLETE", "OTHER", "DIFFERENT_QUESTION", "NOT_MAINTAINED", "JURISDICTION", "DEFERRAL", "DOES_NOT_ARISE", "WITHHELD"]:
        masks[t] = typ == t
        masks[t + " (label NONE only)"] = (typ == t) & (lab == "NONE")
    isfull = typ == "FULL"
    B = 1000
    boots = {k: [] for k in masks}
    bootsu = {k: [] for k in masks}
    for b in range(B):
        pick = rng.integers(0, len(qids), len(qids))
        ii = np.concatenate([idx_by_q[qids[k]] for k in pick])
        f_i = ii[isfull[ii]]
        for name, mk in masks.items():
            n_i = ii[mk[ii]]
            if len(n_i) >= 3 and len(f_i) >= 3:
                boots[name].append(auc(cos[f_i], cos[n_i]))
                bootsu[name].append(auc(cosu[f_i], cosu[n_i]))
    rows = []
    qf = np.quantile(cos[isfull], [0.25, 0.5, 0.75])
    for name, mk in masks.items():
        n = int(mk.sum())
        if n < 5:
            continue
        lo, hi = np.percentile(boots[name], [2.5, 97.5]) if boots[name] else (np.nan, np.nan)
        lou, hiu = np.percentile(bootsu[name], [2.5, 97.5]) if bootsu[name] else (np.nan, np.nan)
        rows.append({"gold group": name, "n parts": n, "mean cos": cos[mk].mean(), "median cos": np.median(cos[mk]),
                     "% above FULL median": f"{100 * (cos[mk] > qf[1]).mean():.1f}",
                     "AUC vs FULL (tuned)": auc(cos[isfull], cos[mk]), "95% CI": f"[{lo:.3f}, {hi:.3f}]",
                     "AUC (untuned)": auc(cosu[isfull], cosu[mk]), "CI untuned": f"[{lou:.3f}, {hiu:.3f}]"})
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(OUT, "b4_auc_by_type.csv"), index=False)
    NUMS["b4"] = R
    md(f"FULL parts (n = {int(isfull.sum())}): mean cosine {cos[isfull].mean():.3f}, median {qf[1]:.3f}, IQR [{qf[0]:.3f}, {qf[2]:.3f}].\n")
    tbl(R)
    wrows = []
    for name in ["any non-FULL", "label NONE", "label PARTIAL", "NOT_MAINTAINED", "JURISDICTION", "DEFERRAL", "GENERIC", "INCOMPLETE", "DIFFERENT_QUESTION"]:
        mk = masks[name]
        wrows.append({"group": name, "AUC design-weighted": wauc(cos[isfull], cos[mk], w[isfull], w[mk])})
    pd.DataFrame(wrows).to_csv(os.path.join(OUT, "b4_auc_weighted.csv"), index=False)
    NUMS["b4w"] = wrows
    md("Design-weighted AUCs (question weights from main_sample.csv): " +
       "; ".join(f"{r['group']} {r['AUC design-weighted']:.3f}" for r in wrows) + ".\n")
    d["type"] = np.where(d["label"] == "FULL", "FULL", d["evasion_type"])
    a_all = auc(d.loc[d.type == "FULL", "m1_cos"].to_numpy(float), d.loc[d.type != "FULL", "m1_cos"].to_numpy(float))
    md(f"Sensitivity: FULL vs any non-FULL AUC using all {len(d)} valid gold parts (heldout + train/val) = {a_all:.3f}.\n")
    return dh, R


# --------------------------------------------------------------------------------------------- B5
def kappa_w(x, y, w=None, cats=("NONE", "PARTIAL", "FULL")):
    idx = {c: i for i, c in enumerate(cats)}
    M = np.zeros((len(cats), len(cats)))
    w = np.ones(len(x)) if w is None else w
    for a, b, ww in zip(x, y, w):
        M[idx[a], idx[b]] += ww
    n = M.sum()
    po = np.trace(M) / n
    pe = (M.sum(1) * M.sum(0)).sum() / n ** 2
    return (po - pe) / (1 - pe)


def kappa_nominal(x, y):
    cats = tuple(sorted(set(x) | set(y)))
    return kappa_w(np.asarray(x), np.asarray(y), None, cats)


def b5_alt():
    g = gold_frame()
    alt_p = os.path.join(MEAS, "m2_gold_alt_3.1fl.parquet")
    if not os.path.exists(alt_p):
        md("### B5. Second-model robustness: NOT RUN (m2_gold_alt_3.1fl.parquet missing)")
        return None
    alt = pd.read_parquet(alt_p)
    prod = pd.read_parquet(os.path.join(MEAS, "m2_labels.parquet"))
    cost = json.load(open(os.path.join(MEAS, "m2_gold_alt_3.1fl_cost.json")))
    NUMS["cost"] = cost
    md("### B5. Second-model robustness (gemini-3.1-flash-lite vs gemini-3.8-flash vs human gold)")
    md(f"gemini-3.1-flash-lite run on the 400 in-estimate gold questions with the production configuration (no rationale, "
       f"default thinking = MINIMAL for lite, temperature auto = 1.0, seed 0, prompt hash {alt['prompt_hash'].iloc[0]}, "
       f"synchronous client, 8 workers). Parts returned: {len(alt)} ({int((alt.status != 'ok').sum())} invalid). "
       f"Cost **${cost['usd']:.3f}** (sync = 2 x batch list price; tokens in {cost['input_tokens']:,.0f}, out {cost['output_tokens']:,.0f}; cap $3). "
       "3.8-flash labels are the production labels in `m2_labels.parquet`. Comparison set = gold parts labelled ok by both models.\n")
    a = alt[alt["status"] == "ok"][["part_id", "label", "evasion_type"]].rename(columns={"label": "lab_alt", "evasion_type": "ev_alt"})
    p = prod[["part_id", "label", "evasion_type"]].rename(columns={"label": "lab_38", "evasion_type": "ev_38"})
    d = g.merge(p, on="part_id", how="inner").merge(a, on="part_id", how="inner").reset_index(drop=True)
    d["s_gold"] = d["label"].map(SCORE)
    d["s_38"] = d["lab_38"].map(SCORE)
    d["s_alt"] = d["lab_alt"].map(SCORE)
    n = len(d)
    qids = list(d["qid"].unique())
    q_idx = {q: np.where(d["qid"].values == q)[0] for q in qids}
    rng = np.random.default_rng(20261019)
    B = 2000

    def kap_triple(ii):
        return (kappa_w(d["label"].values[ii], d["lab_38"].values[ii]), kappa_w(d["label"].values[ii], d["lab_alt"].values[ii]),
                kappa_w(d["lab_38"].values[ii], d["lab_alt"].values[ii]))
    allidx = np.arange(n)
    k38, kalt, k38alt = kap_triple(allidx)
    bk = []
    for b in range(B):
        pick = rng.integers(0, len(qids), len(qids))
        ii = np.concatenate([q_idx[qids[k]] for k in pick])
        kt = kap_triple(ii)
        bk.append(kt + (kt[1] - kt[0],))
    bk = np.array(bk)
    cis = np.percentile(bk, [2.5, 97.5], axis=0)
    rows = []
    agree = {0: d.label == d.lab_38, 1: d.label == d.lab_alt, 2: d.lab_38 == d.lab_alt}
    for name, v, j in (("3.8-flash vs gold", k38, 0), ("3.1-flash-lite vs gold", kalt, 1), ("3.8-flash vs 3.1-flash-lite", k38alt, 2)):
        rows.append({"comparison": name, "n parts": n, "label agreement %": 100 * agree[j].mean(),
                     "kappa (3-class label)": v, "95% CI": f"[{cis[0][j]:.3f}, {cis[1][j]:.3f}]"})
    dk = kalt - k38
    md(f"Label kappa vs human gold on {n} parts (unweighted Cohen kappa, 3 classes; question-cluster bootstrap B = {B}). "
       f"Reference from the validation report for 3.8-flash on all 1,636 gold parts: 0.711. "
       f"kappa(3.1-lite) minus kappa(3.8) = {dk:+.3f}, bootstrap 95% CI [{cis[0][3]:+.3f}, {cis[1][3]:+.3f}].\n")
    tbl(pd.DataFrame(rows))
    NUMS["b5_kappa"] = (k38, kalt, k38alt, cis, dk)
    wk = [kappa_w(d.label.values, d.lab_38.values, d.weight.values), kappa_w(d.label.values, d.lab_alt.values, d.weight.values)]
    NUMS["b5_wk"] = wk
    md(f"Design-weighted kappa (question weights): 3.8-flash {wk[0]:.3f}; 3.1-flash-lite {wk[1]:.3f}. Confusion matrices (rows = gold, columns = model):\n")
    for nm, col in (("3.8-flash", "lab_38"), ("3.1-flash-lite", "lab_alt")):
        cm = pd.crosstab(d["label"], d[col]).reindex(index=["NONE", "PARTIAL", "FULL"], columns=["NONE", "PARTIAL", "FULL"], fill_value=0)
        cm.index.name = f"gold / {nm}"
        tbl(cm.reset_index())
    ev = []
    for nm, lc, ec in (("3.8-flash", "lab_38", "ev_38"), ("3.1-flash-lite", "lab_alt", "ev_alt")):
        s = d[(d.label != "FULL") & (d[lc] != "FULL")]
        ev.append({"model": nm, "n (both non-FULL)": len(s), "evasion-type agreement %": 100 * (s.evasion_type == s[ec]).mean(),
                   "evasion-type kappa": kappa_nominal(s.evasion_type.values, s[ec].values)})
    md("Evasion-type agreement with gold where gold and the model both say non-FULL:\n")
    tbl(pd.DataFrame(ev))
    NUMS["b5_ev"] = ev
    w = d.weight.values
    S = {"gold": d.s_gold.values, "38": d.s_38.values, "alt": d.s_alt.values}
    qs = d.groupby("qid").agg(w=("weight", "first"), g=("s_gold", "mean"), a38=("s_38", "mean"), aalt=("s_alt", "mean"), ls=("ls", "first"))
    qpos = {q: i for i, q in enumerate(qs.index)}

    def stats(ii_p, ii_q):
        pm = [(S[k][ii_p] * w[ii_p]).sum() / w[ii_p].sum() for k in ("gold", "38", "alt")]
        qm = [(qs[c].values[ii_q] * qs["w"].values[ii_q]).sum() / qs["w"].values[ii_q].sum() for c in ("g", "a38", "aalt")]
        return pm + qm
    base = np.array(stats(allidx, np.arange(len(qs))))
    bs = []
    for b in range(B):
        pick = rng.integers(0, len(qids), len(qids))
        ii = np.concatenate([q_idx[qids[k]] for k in pick])
        iq = np.array([qpos[qids[k]] for k in pick])
        bs.append(stats(ii, iq))
    bs = np.array(bs)
    rows = []
    for lv, o in (("part level (design-weighted)", 0), ("question level (design-weighted)", 3)):
        for nm, j in (("human gold", 0), ("3.8-flash", 1), ("3.1-flash-lite", 2)):
            r = {"level": lv, "rater": nm, "weighted mean score": base[o + j],
                 "95% CI": f"[{np.percentile(bs[:, o+j], 2.5):.3f}, {np.percentile(bs[:, o+j], 97.5):.3f}]"}
            if j > 0:
                diff = bs[:, o + j] - bs[:, o]
                r["bias vs gold"] = base[o + j] - base[o]
                r["bias 95% CI"] = f"[{np.percentile(diff, 2.5):+.3f}, {np.percentile(diff, 97.5):+.3f}]"
            else:
                r["bias vs gold"] = np.nan
                r["bias 95% CI"] = ""
            rows.append(r)
    md("Leniency: design-weighted mean score (FULL=1, PARTIAL=.5, NONE=0) on the same parts; bias = model minus gold (paired, question-cluster bootstrap):\n")
    tbl(pd.DataFrame(rows))
    NUMS["b5_len"] = pd.DataFrame(rows)
    md(f"Unweighted part means: gold {d.s_gold.mean():.3f}, 3.8-flash {d.s_38.mean():.3f}, 3.1-flash-lite {d.s_alt.mean():.3f}. "
       f"Label shares (FULL/PARTIAL/NONE, unweighted): gold {(d.label=='FULL').mean():.3f}/{(d.label=='PARTIAL').mean():.3f}/{(d.label=='NONE').mean():.3f}; "
       f"3.8 {(d.lab_38=='FULL').mean():.3f}/{(d.lab_38=='PARTIAL').mean():.3f}/{(d.lab_38=='NONE').mean():.3f}; "
       f"3.1-lite {(d.lab_alt=='FULL').mean():.3f}/{(d.lab_alt=='PARTIAL').mean():.3f}/{(d.lab_alt=='NONE').mean():.3f}.\n")
    lss = sorted(qs["ls"].unique())
    mat = {}
    for l in lss:
        s = qs[qs.ls == l]
        mat[l] = [(s[c] * s["w"]).sum() / s["w"].sum() for c in ("g", "a38", "aalt")]
    M = pd.DataFrame(mat, index=["gold", "3.8-flash", "3.1-flash-lite"]).T
    nq = qs.groupby("ls").size()
    from scipy.stats import kendalltau, spearmanr
    rk = M.rank(ascending=False).astype(int)
    show = pd.DataFrame({"LS": M.index, "questions": nq.values})
    for c in M.columns:
        show[c] = [f"{v:.3f} (#{r})" for v, r in zip(M[c], rk[c])]
    md("LS-level design-weighted mean question score on the 400 gold questions (rank 1 = most responsive):\n")
    tbl(show)
    tau_gm = [kendalltau(M["gold"], M[c])[0] for c in ("3.8-flash", "3.1-flash-lite")]
    tau_ma = kendalltau(M["3.8-flash"], M["3.1-flash-lite"])[0]
    rho_ma = spearmanr(M["3.8-flash"], M["3.1-flash-lite"])[0]
    qls = qs["ls"].values
    qw = qs["w"].values
    bt = []
    cols3 = ("g", "a38", "aalt")
    for b in range(B):
        pick = rng.integers(0, len(qids), len(qids))
        iq = np.array([qpos[qids[k]] for k in pick])
        mm = np.array([[(qs[c].values[iq][qls[iq] == l] * qw[iq][qls[iq] == l]).sum() / qw[iq][qls[iq] == l].sum() for c in cols3] for l in lss])
        bt.append([kendalltau(mm[:, 1], mm[:, 2])[0], kendalltau(mm[:, 0], mm[:, 1])[0], kendalltau(mm[:, 0], mm[:, 2])[0]])
    bt = np.array(bt)
    same = bool((rk["3.8-flash"] == rk["3.1-flash-lite"]).all())
    md(f"Kendall tau of the six LS means: 3.8 vs 3.1-lite {tau_ma:.2f} (Spearman {rho_ma:.2f}; bootstrap 95% CI [{np.percentile(bt[:,0],2.5):.2f}, {np.percentile(bt[:,0],97.5):.2f}]); "
       f"gold vs 3.8 {tau_gm[0]:.2f} (CI [{np.percentile(bt[:,1],2.5):.2f}, {np.percentile(bt[:,1],97.5):.2f}]); "
       f"gold vs 3.1-lite {tau_gm[1]:.2f} (CI [{np.percentile(bt[:,2],2.5):.2f}, {np.percentile(bt[:,2],97.5):.2f}]). "
       f"Identical full LS ordering under the two models: {'yes' if same else 'no'}. With about 65 questions per LS, adjacent-LS "
       "differences are mostly within sampling noise, so rank agreement is a coarse check.\n")
    NUMS["b5_ls"] = (M, tau_ma, tau_gm, bt, same, rk)
    return M


def fig3(dh, M):
    fig, axs = plt.subplots(1, 2, figsize=(13.5, 5.4))
    order = ["FULL", "GENERIC", "INCOMPLETE", "OTHER", "DIFFERENT_QUESTION", "DEFERRAL", "NOT_MAINTAINED", "JURISDICTION", "DOES_NOT_ARISE", "WITHHELD"]
    data, labs = [], []
    for t in order:
        v = dh.loc[dh["type"] == t, "m1_cos"].to_numpy(float)
        if len(v) >= 5:
            data.append(v)
            labs.append(f"{t.replace('_', chr(10))}\n(n={len(v)})")
    bp = axs[0].boxplot(data, tick_labels=labs, showfliers=False, patch_artist=True, widths=0.6)
    for i, b in enumerate(bp["boxes"]):
        b.set_facecolor("#a6cee3" if i == 0 else "#fdbf6f")
        b.set_alpha(0.8)
    rng = np.random.default_rng(0)
    for i, v in enumerate(data, 1):
        axs[0].scatter(i + rng.uniform(-0.18, 0.18, len(v)), v, s=4, color="#555", alpha=0.35, zorder=3)
    axs[0].axhline(np.median(data[0]), color="#1f78b4", ls="--", lw=0.8)
    axs[0].set_ylabel("M1 cosine (fine-tuned bi-encoder), heldout gold parts")
    axs[0].set_title("M1 cosine by gold evasion type (dashed = FULL median)")
    axs[0].tick_params(axis="x", labelsize=6.5)
    if M is not None:
        x = np.arange(len(M))
        for c, mk, colr in (("gold", "s-", "#000"), ("3.8-flash", "o-", "#c0392b"), ("3.1-flash-lite", "^--", "#2c7fb8")):
            axs[1].plot(x, M[c], mk, color=colr, label=c)
        axs[1].set_xticks(x)
        axs[1].set_xticklabels([f"LS{l}" for l in M.index])
        axs[1].set_ylabel("design-weighted mean question score (gold questions)")
        axs[1].set_title("LS ordering on the 400 gold questions: gold vs two models")
        axs[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig3_embeddings_and_second_model.png"), dpi=160)
    plt.close(fig)


def write_report():
    N = NUMS
    A = N["a1"]
    Rall = N["overall_score"][0]
    dif, sd, Rs, ses, Ro, seo = N["gap"]
    T = N["a3_dec"].set_index("type")
    perm = N["a3_perm"].set_index("ministry")
    b4 = N["b4"].set_index("gold group")
    k38, kalt, k38alt, cis, dk = N["b5_kappa"]
    cost = N["cost"]
    ts = N["jur_ls"]
    M, tau_ma, tau_gm, bt, same, rk = N["b5_ls"]
    pl = N["b5_len"]
    pl = pl[pl["level"].str.startswith("part")].set_index("rater")
    sec = A[A.security].set_index("ministry")
    ev = {e["model"]: e for e in N["b5_ev"]}
    sj = N["tier_share_J"]
    H = ["## What holds (exploratory; every analysis run is reported above, none dropped)", "", "**RQ4**"]
    H.append(
        f"- Responsiveness differs strongly by ministry: scores run from {A.score.min():.3f} ({A.iloc[-1].ministry}) to "
        f"{A.score.max():.3f} ({A.iloc[0].ministry}) around an overall {Rall:.3f}; {N['a1_nsig']} of {len(A)} ministry CIs exclude "
        f"the overall mean. The ranking is stable to LS composition (Spearman {N['a1_rho']:.2f}, raw vs LS-adjusted). The top is "
        "scientific/technical ministries (Atomic Energy, Earth Sciences, Science and Technology, New and Renewable Energy, Power); "
        "the bottom is Law and Justice, Labour and Employment, Minority Affairs, Corporate Affairs, Personnel.")
    H.append(
        f"- Security is not uniformly low: External Affairs #{sec.loc['External Affairs', 'rank']} of {len(A)}, Defence "
        f"#{sec.loc['Defence', 'rank']}, Home Affairs #{sec.loc['Home Affairs', 'rank']}. Pooled, security scores {Rs:.3f} vs {Ro:.3f} "
        f"for other ministries (loss gap {dif:+.3f}, CI [{dif-1.96*sd:+.3f}, {dif+1.96*sd:+.3f}]); the gap is carried by Home Affairs "
        f"(loss {perm.loc['Home Affairs', 'loss minus other']:+.3f} vs other ministries) and Defence "
        f"({perm.loc['Defence', 'loss minus other']:+.3f}); External Affairs is {perm.loc['External Affairs', 'loss minus other']:+.3f} (better than average).")
    H.append(
        f"- JURISDICTION is concentrated in State/Concurrent-list ministries: Tier A (health, agriculture, home, water, education) "
        f"{100*N['tier_A'][0]:.2f}% of parts, Tier B {100*N['tier_B'][0]:.2f}%, Tier C (Union/other) {100*N['tier_C'][0]:.2f}%; Tier A "
        f"holds {100*sj.iloc[0]:.0f}% of all JURISDICTION parts with 24% of parts. It is uneven within the tier: highest in AYUSH, "
        "Housing and Urban Affairs, Panchayati Raj, Water/Jal Shakti, Home Affairs; low in Agriculture (about 1%) and near zero in "
        "Defence and External Affairs. The mapping is by ministry name only and approximate; the topic-cluster check points the same "
        "way (health, water, education and courts topics highest).")
    H.append(
        f"- JURISDICTION deflection rises over time: all ministries {100*ts[('all', 13)][0]:.2f}% of parts (LS13) to "
        f"{100*ts[('all', 18)][0]:.2f}% (LS18); Tier A {100*ts[('A', 13)][0]:.2f}% to {100*ts[('A', 18)][0]:.2f}%, Tier B "
        f"{100*ts[('B', 13)][0]:.2f}% to {100*ts[('B', 18)][0]:.2f}%, Tier C {100*ts[('C', 13)][0]:.2f}% to {100*ts[('C', 18)][0]:.2f}%. "
        "Among non-FULL parts its share rises from 3.5% (LS13) to 5-6% (LS16-18). Home Affairs peaks in LS17 (5.9% of parts); Water "
        "is highest in LS13 (4.7%) and dips to 2.5% in LS15.")
    H.append(
        f"- The security gap is NOT mainly WITHHELD. WITHHELD is where security is most distinctive ({T.loc['WITHHELD', 'ratio sec/other']:.1f}x "
        f"the rate of other ministries; {T.loc['WITHHELD', '% of gap']:.0f}% of the loss gap) but it is under 1% of parts. GENERIC "
        f"({T.loc['GENERIC', '% of gap']:.0f}%), NOT_MAINTAINED ({T.loc['NOT_MAINTAINED', '% of gap']:.0f}%), DEFERRAL "
        f"({T.loc['DEFERRAL', '% of gap']:.0f}%) and JURISDICTION ({T.loc['JURISDICTION', '% of gap']:.0f}%) carry most of it; security "
        f"answers less often give a different question's answer ({T.loc['DIFFERENT_QUESTION', '% of gap']:.0f}%, an offset). By ministry: "
        "Home Affairs = NOT_MAINTAINED + JURISDICTION (crime and State data); Defence = WITHHELD + DEFERRAL + GENERIC; External Affairs = "
        "GENERIC only, offset elsewhere.")
    H += ["", "**RQ1**"]
    H.append(
        f"- Embedding cosine barely separates answers from non-answers: AUC(FULL vs any non-FULL) = "
        f"{b4.loc['any non-FULL', 'AUC vs FULL (tuned)']:.3f} {b4.loc['any non-FULL', '95% CI']} (design-weighted 0.530). It separates "
        f"NONE-labelled parts moderately ({b4.loc['label NONE', 'AUC vs FULL (tuned)']:.3f}) but PARTIAL parts score no lower than FULL "
        f"({b4.loc['label PARTIAL', 'AUC vs FULL (tuned)']:.3f}).")
    H.append(
        f"- The failure is type-specific. NOT_MAINTAINED non-answers have HIGHER cosine than FULL answers (AUC "
        f"{b4.loc['NOT_MAINTAINED', 'AUC vs FULL (tuned)']:.3f} {b4.loc['NOT_MAINTAINED', '95% CI']}; median "
        f"{b4.loc['NOT_MAINTAINED', 'median cos']:.3f} vs 0.523 for FULL); JURISDICTION is a coin flip "
        f"({b4.loc['JURISDICTION', 'AUC vs FULL (tuned)']:.3f} {b4.loc['JURISDICTION', '95% CI']}); INCOMPLETE "
        f"({b4.loc['INCOMPLETE', 'AUC vs FULL (tuned)']:.3f}) and DIFFERENT_QUESTION ({b4.loc['DIFFERENT_QUESTION', 'AUC vs FULL (tuned)']:.3f}) "
        f"are weak. Only DEFERRAL ({b4.loc['DEFERRAL', 'AUC vs FULL (tuned)']:.3f}), DOES_NOT_ARISE "
        f"({b4.loc['DOES_NOT_ARISE', 'AUC vs FULL (tuned)']:.3f}), OTHER and WITHHELD (n = 6, unreliable) are separable. The untuned base "
        "model shows the same pattern.")
    H.append(
        f"- Second model: gemini-3.1-flash-lite agrees materially less with the human gold: kappa {kalt:.3f} [{cis[0][1]:.3f}, "
        f"{cis[1][1]:.3f}] vs {k38:.3f} [{cis[0][0]:.3f}, {cis[1][0]:.3f}] for 3.8-flash (difference {dk:+.3f}, CI "
        f"[{cis[0][3]:+.3f}, {cis[1][3]:+.3f}]); it falls below the 0.6 adequacy threshold used for model choice. Evasion-type kappa "
        f"{ev['3.1-flash-lite']['evasion-type kappa']:.3f} vs {ev['3.8-flash']['evasion-type kappa']:.3f}. Cost ${cost['usd']:.2f}.")
    H.append(
        f"- Leniency: design-weighted part-level mean score gold {pl.loc['human gold', 'weighted mean score']:.3f}; 3.8-flash "
        f"{pl.loc['3.8-flash', 'weighted mean score']:.3f} (bias {pl.loc['3.8-flash', 'bias vs gold']:+.3f}, CI "
        f"{pl.loc['3.8-flash', 'bias 95% CI']}: slightly lenient); 3.1-flash-lite {pl.loc['3.1-flash-lite', 'weighted mean score']:.3f} "
        f"(bias {pl.loc['3.1-flash-lite', 'bias vs gold']:+.3f}, CI {pl.loc['3.1-flash-lite', 'bias 95% CI']}: not distinguishable from "
        "zero). The lite model's near-zero net bias hides larger two-way noise (142 gold FULL parts downgraded to PARTIAL vs 47 for 3.8, "
        "plus upgrades of PARTIAL to FULL).")
    H.append(
        f"- LS ordering on the gold questions is similar but not identical: LS18 is lowest under gold and both models and LS13 or LS14 "
        f"highest; the full six-LS ordering is {'identical' if same else 'not identical'} between the two models (Kendall tau "
        f"{tau_ma:.2f}, bootstrap CI [{np.percentile(bt[:, 0], 2.5):.2f}, {np.percentile(bt[:, 0], 97.5):.2f}]; gold vs 3.8 "
        f"{tau_gm[0]:.2f}; gold vs lite {tau_gm[1]:.2f}). With about 65 questions per LS only the broad pattern (LS18 lowest) is "
        "resolvable.")
    H += ["", "**Caveats**: the ministry-level jurisdiction mapping is approximate; ministry gaps are descriptive (no controls for "
          "topic, asker bloc or LS beyond the indirect LS standardisation); WITHHELD has few cases per ministry (zero-count cells give "
          "degenerate CIs of [0, 0]); gold evasion-type cells are small (WITHHELD n = 6, DOES_NOT_ARISE n = 13, DEFERRAL n = 20); B5 "
          "uses 400 questions so LS-level comparisons are noisy; LS18 is a partial parliament."]
    hdr = ["# RQ4 (where ministers evade) and RQ1 (measurement): exploratory results", "",
           "Generated by `analysis/explore_rq4_rq1.py` (labelling run: `analysis/explore_label_alt.py`). M2 = gemini-3.8-flash "
           "production labels; gold = human gold labels. Statistics only.",
           f"Gemini cost of the second-model run: **${cost['usd']:.3f}** (gemini-3.1-flash-lite, 1,636 gold parts, 400 questions; cap $3).", "",
           "Figures: `fig1_ministry_responsiveness.png`, `fig2_jurisdiction_security.png`, `fig3_embeddings_and_second_model.png`. "
           "CSVs: `a1_ministry_table_full.csv`, `a2_topic_jurisdiction.csv`, `b4_auc_by_type.csv`, `b4_auc_weighted.csv`.", ""]
    with open(os.path.join(OUT, "rq4_rq1.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(hdr + MD + [""] + H) + "\n")


def main():
    D = Design()
    A = a1_ministry(D)
    a2_jurisdiction(D, A)
    a3_security(D, A)
    dh, R = b4_embeddings()
    M = b5_alt()
    fig3(dh, M)
    write_report()
    print("report written")


if __name__ == "__main__":
    main()
