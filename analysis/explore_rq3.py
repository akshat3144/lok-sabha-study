"""Exploratory RQ3: does the opposition-vs-government gap in answer responsiveness depend on WHO asks and on the question?

    python -I -X utf8 analysis/explore_rq3.py

Reads (never modifies): data/measures/analysis_frame_real.parquet, data/final/question_askers.parquet,
data/ref/coalitions.csv, data/measures/m2_labels.parquet.   Writes: results/explore/rq3_askers.md, rq3_*.csv, 2 PNGs.
(results/explore/rq3_what_holds.txt, if present, is a hand-written closing section appended verbatim.)

EXPLORATORY: every model run is written to results/explore/rq3_all_models.csv (nothing is dropped); p-values are raw.

Specification (as models.py H1/H3 spec): OLS, outcome y_m2, controls n_parts, log_qlen, coasker, request-type shares
(rt_reason, rt_action, rt_opinion), FE for ministry_group, topic (K=50) and session_id; two-way clustered SEs (mp_id x
session_id, t(min G - 1)); design weights m2_weight (1 in LS17/18).  Within a single Lok Sabha the session FE are
within-LS session FE, and the number of session clusters is small (LS18: 7), so an MP-only clustered CI is reported next
to the two-way CI for LS-level models.  Reference group = government askers unless stated.

Asker roles (task 2): the roles in question_askers.parquet (role: government / opposition / outside_support, from
coalitions.csv by party and question date) are split as follows, all derived from data, nothing hard-coded by party name
except the two alliance anchors read from coalitions.csv itself (the rows whose notes say "PM's party": BJP in NDA, INC in UPA):
    main ruling party   = the anchor party of the alliance in office on the question date (INC under UPA, BJP under NDA)
    coalition partner   = role government and party != main ruling party  (the partner list is whatever coalitions.csv has
                          as government-role rows for that date, incl. faction-level overrides in coalition_lookup)
    main opposition     = the anchor party of the OTHER alliance (BJP under UPA, INC under NDA)  [largest-opposition-party proxy]
    other opposition    = role opposition, any other party
    outside support     = role outside_support (kept separate)
Question-level code: all askers in one category -> that category; all in one bloc but different sub-categories -> gov_both /
opp_both; askers from different blocs (incl. any outside_support with others) -> mixed.
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

import common as C  # noqa: E402
import models as M  # noqa: E402
from codebook import LABEL_SCORE  # noqa: E402

warnings.filterwarnings("ignore")
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = os.path.join(C.RESULTS, "explore")
os.makedirs(OUT, exist_ok=True)
FRAME = C.frame_path("real")
COAL = os.path.join(C.REF, "coalitions.csv")
CTRL = ["n_parts", "log_qlen", "coasker", "rt_reason", "rt_action", "rt_opinion"]
CTRL_NORT = ["n_parts", "log_qlen", "coasker"]
CAT = ["ministry_group", "topic", "session_id"]
LSS = [13, 14, 15, 16, 17, 18]
BLOC = ["opp", "mixed", "outside"]
LOG = []        # every model run: dict rows -> rq3_all_models.csv


# ---------------------------------------------------------------------------------------------- helpers
def fit(d, y, num, cat, weight=None, cluster=("mp_id", "session_id")):
    need = [y, *num, *cat, *cluster] + ([weight] if weight else [])
    d = d.dropna(subset=need)
    if weight:
        d = d[d[weight] > 0]
    return M.fit_model(d, y, num, cat, "ols", cluster, weight=weight)


def term_row(f, term, group, spec, extra=None):
    if term not in f.idx:
        r = {"term": term, "estimate": np.nan, "se": np.nan, "ci_lo": np.nan, "ci_hi": np.nan, "t": np.nan, "p": np.nan}
    else:
        r = f.coef(term)
    r.update({"group": group, "spec": spec, "n": f.n, "g_mp": f.g_sizes[0], "g_sess": f.g_sizes[-1] if len(f.g_sizes) > 1 else np.nan})
    if extra:
        r.update(extra)
    LOG.append(dict(r))
    return r


def contrast_row(f, weights, name, group, spec):
    try:
        r = f.contrast(weights, name)
    except KeyError:
        return {"term": name, "estimate": np.nan, "se": np.nan, "ci_lo": np.nan, "ci_hi": np.nan, "p": np.nan,
                "group": group, "spec": spec, "n": f.n}
    r.update({"group": group, "spec": spec, "n": f.n, "g_mp": f.g_sizes[0], "g_sess": f.g_sizes[-1] if len(f.g_sizes) > 1 else np.nan})
    LOG.append(dict(r))
    return r


def md(df, cols, ff="{:.3f}"):
    df = df[cols].copy()
    for c in ("n_opp", "n_party", "size"):
        if c in df:
            df[c] = [("" if pd.isna(v) else f"{int(v):,}") for v in df[c]]
    return C.df_to_markdown(df, floatfmt=ff)


def fmt_ci(r, p=True):
    if np.isnan(r["estimate"]):
        return "n/a"
    s = f"{r['estimate']:+.4f} [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]"
    return s + (f" p={r['p']:.3g}" if p else "")


def wald(f, terms_a, q=None):
    """Joint Wald test that coefficients terms_a are all equal to each other (successive differences = 0)."""
    idx = [f.idx[t] for t in terms_a if t in f.idx]
    k = len(idx)
    R = np.zeros((k - 1, len(f.beta)))
    for i in range(k - 1):
        R[i, idx[i]], R[i, idx[i + 1]] = 1, -1
    d = R @ f.beta
    V = R @ f.V @ R.T
    W = float(d @ np.linalg.pinv(V) @ d)
    return W, k - 1, float(stats.chi2.sf(W, k - 1)), float(stats.f.sf(W / (k - 1), k - 1, f.df_t))


# ---------------------------------------------------------------------------------------------- data
def load():
    f = pd.read_parquet(FRAME)
    a = pd.read_parquet(C.ASKERS)
    co = pd.read_csv(COAL)
    # --- anchors (derived from coalitions.csv)
    anc = co[co["notes"].fillna("").str.contains("PM's party")]
    anchors = dict(zip(anc["alliance"], anc["party_short"]))                       # {NDA: BJP, UPA: INC}
    ar = co[(co["role"] == "government") & co["party_short"].isin(anchors.values())
            & (co["party_short"] == co["alliance"].map(anchors))].copy()
    ar["s"] = pd.to_datetime(ar["start_date"])
    ar["e"] = pd.to_datetime(ar["end_date"]).fillna(pd.Timestamp("2100-01-01"))
    dts = f[["qid", "date"]].drop_duplicates("qid").set_index("qid")["date"]

    def main_party(dt):
        r = ar[(ar["s"] <= dt) & (ar["e"] >= dt)]
        return (r["party_short"].iloc[0], r["alliance"].iloc[0]) if len(r) else (None, None)

    ud = pd.Series(sorted(dts.unique()))
    mp = {d: main_party(d) for d in ud}
    other = {al: [p for k, p in anchors.items() if k != al][0] for al in anchors}
    a = a.merge(dts.rename("qdate"), left_on="qid", right_index=True, how="inner")
    a["main_gov"] = a["qdate"].map(lambda d: mp[d][0])
    a["alliance"] = a["qdate"].map(lambda d: mp[d][1])
    a["main_opp"] = a["alliance"].map(other)

    def cat(r):
        if r["role"] == "government":
            return "gov_main" if r["party_short"] == r["main_gov"] else "gov_partner"
        if r["role"] == "opposition":
            return "opp_main" if r["party_short"] == r["main_opp"] else "opp_other"
        if r["role"] == "outside_support":
            return "outside"
        return None

    a["acat"] = [cat(r) for r in a[["role", "party_short", "main_gov", "main_opp"]].to_dict("records")]
    a = a.dropna(subset=["acat"])
    g = a.groupby("qid")["acat"].agg(lambda s: tuple(sorted(set(s))))

    def qcat(t):
        if len(t) == 1:
            return t[0]
        bl = {x.split("_")[0] for x in t}
        if len(bl) == 1:
            return {"gov": "gov_both", "opp": "opp_both"}[bl.pop()]
        return "mixed"

    f["qcat"] = f["qid"].map(g.map(qcat))
    # which party for single-asker questions
    first = a.sort_values("qid").drop_duplicates("qid")[["qid", "party_short"]].set_index("qid")["party_short"]
    f["asker_party"] = f["qid"].map(first)
    f["qdate_alliance"] = f["qid"].map(a.drop_duplicates("qid").set_index("qid")["alliance"])
    for c in ["gov_partner", "gov_both", "opp_main", "opp_other", "opp_both", "outside", "mixed"]:
        f["c_" + c] = (f["qcat"] == c).astype(int)
    f["post"] = (f["ls"] >= 17).astype(int)
    f["opp_post"] = f["opp"] * f["post"]
    f["starred_x_opp"] = f["starred"] * f["opp"]
    f["sec_x_opp"] = f["security"] * f["opp"]
    f["single"] = f["n_askers"] == 1
    f["ls_idx"] = f["ls"] - 13
    f["opp_trend"] = f["opp"] * f["ls_idx"]
    for k in LSS:
        f[f"opp_ls{k}"] = f["opp"] * (f["ls"] == k)
    return f, a, anchors


def sample_m2(f):
    return f[f["analysis_sample"] & f["y_m2"].notna() & (f["m2_weight"] > 0)].copy()


# ---------------------------------------------------------------------------------------------- task 1
def task1(d, out):
    rows, rows_mp = [], []
    for k in LSS:
        dk = d[d["ls"] == k]
        f2 = fit(dk, "y_m2", BLOC + CTRL, CAT, "m2_weight")
        r = term_row(f2, "opp", f"LS{k}", "two-way (MP,session)", {"mean_y": np.average(dk["y_m2"], weights=dk["m2_weight"])})
        rows.append(r)
        f1 = fit(dk, "y_m2", BLOC + CTRL, CAT, "m2_weight", cluster=("mp_id",))
        rows_mp.append(term_row(f1, "opp", f"LS{k}", "MP-cluster only"))
        # without topic FE and without ministry FE (lean) as sensitivity
        fl = fit(dk, "y_m2", BLOC + CTRL, ["session_id"], "m2_weight")
        term_row(fl, "opp", f"LS{k}", "no topic/ministry FE")
    t = pd.DataFrame(rows)
    tm = pd.DataFrame(rows_mp)
    t["ci_mp_lo"], t["ci_mp_hi"], t["p_mp"] = tm["ci_lo"].values, tm["ci_hi"].values, tm["p"].values
    t["n_opp"] = [int(d[(d["ls"] == k)]["opp"].sum()) for k in LSS]
    t["share_opp_w"] = [np.average(d[d["ls"] == k]["opp"], weights=d[d["ls"] == k]["m2_weight"]) for k in LSS]
    out["t1_by_ls"] = t
    # pooled interactions: LS-specific gaps in one model (session FE), joint tests
    fp = fit(d, "y_m2", [f"opp_ls{k}" for k in LSS] + ["mixed", "outside"] + CTRL, CAT, "m2_weight")
    pl = pd.DataFrame([term_row(fp, f"opp_ls{k}", f"LS{k}", "pooled LS-specific gaps") for k in LSS])
    out["t1_pooled_ls"] = pl
    W, df_, pchi, pf = wald(fp, [f"opp_ls{k}" for k in LSS])
    out["t1_equal_test"] = dict(W=W, df=df_, p_chi2=pchi, p_F=pf)
    ft = fit(d, "y_m2", ["opp", "opp_trend", "mixed", "outside"] + CTRL, CAT, "m2_weight")
    out["t1_trend"] = [term_row(ft, "opp", "pooled", "opp + opp x (LS-13) trend: gap at LS13"),
                       term_row(ft, "opp_trend", "pooled", "opp x (LS-13): change in gap per Lok Sabha")]
    fpp = fit(d, "y_m2", ["opp", "opp_post", "mixed", "outside"] + CTRL, CAT, "m2_weight")
    out["t1_prepost"] = [term_row(fpp, "opp", "LS13-16", "pre/post: gap LS13-16"),
                         term_row(fpp, "opp_post", "LS17-18 minus LS13-16", "pre/post: change in gap")]
    out["t1_prepost"].append(contrast_row(fpp, {"opp": 1, "opp_post": 1}, "opp + opp_post (gap LS17-18)", "LS17-18", "pre/post"))
    # pooled H1 replicate (weighted, session FE) and LS17-18 only
    h1 = fit(d, "y_m2", BLOC + CTRL, CAT, "m2_weight")
    h1u = fit(d, "y_m2", BLOC + CTRL, CAT, None)
    d1718 = d[d["ls"] >= 17]
    h1718 = fit(d1718, "y_m2", BLOC + CTRL, CAT, None)
    h1718_k = fit(d1718, "y_m2", BLOC + CTRL, [c for c in CAT if c != "session_id"], None)
    out["t1_ref"] = [term_row(h1, "opp", "pooled LS13-18", "H1 (weighted, session FE)"),
                     term_row(h1u, "opp", "pooled LS13-18", "H1 unweighted"),
                     term_row(h1718, "opp", "LS17-18", "session FE"),
                     term_row(h1718_k, "opp", "LS17-18", "no session FE (H4key-style)")]
    # LS-by-LS gaps with ministry FE in place of ministry_group (finer): sensitivity
    rows = []
    for k in LSS:
        dk = d[d["ls"] == k]
        fm = fit(dk, "y_m2", BLOC + CTRL, ["ministry_canonical", "topic", "session_id"], "m2_weight")
        rows.append(term_row(fm, "opp", f"LS{k}", "ministry_canonical FE"))
    out["t1_minfe"] = pd.DataFrame(rows)
    # raw weighted means by LS x bloc
    raw = (d.assign(wy=d["y_m2"] * d["m2_weight"]).groupby(["ls", "asker_bloc"])[["wy", "m2_weight"]].sum())
    raw = (raw["wy"] / raw["m2_weight"]).unstack().round(3)
    raw["all"] = [np.average(d[d["ls"] == k]["y_m2"], weights=d[d["ls"] == k]["m2_weight"]) for k in raw.index]
    raw["raw_opp_minus_gov"] = raw["opposition"] - raw["government"]
    out["t1_raw"] = raw.reset_index()


# ---------------------------------------------------------------------------------------------- task 2
CATS = ["gov_partner", "gov_both", "opp_main", "opp_other", "opp_both", "outside", "mixed"]


def task2(d, out, f_all):
    dd = d.dropna(subset=["qcat"])
    out["t2_counts"] = (dd.groupby(["ls", "qcat"]).size().unstack(fill_value=0).reset_index())
    num = ["c_" + c for c in CATS] + CTRL
    rows = []
    for k in LSS:
        dk = dd[dd["ls"] == k]
        fk = fit(dk, "y_m2", num, CAT, "m2_weight")
        for c in CATS:
            rows.append(term_row(fk, "c_" + c, f"LS{k}", "roles by LS (ref gov_main)"))
        rows.append(contrast_row(fk, {"c_opp_other": 1, "c_opp_main": -1}, "opp_other - opp_main", f"LS{k}", "roles by LS"))
        rows.append(contrast_row(fk, {"c_opp_main": 1, "c_gov_partner": -1}, "opp_main - gov_partner", f"LS{k}", "roles by LS"))
    out["t2_by_ls"] = pd.DataFrame(rows)
    rows = []
    for name, sub, wt in (("pooled LS13-18 (weighted)", dd, "m2_weight"), ("LS17-18", dd[dd["ls"] >= 17], None),
                          ("LS18", dd[dd["ls"] == 18], None)):
        fk = fit(sub, "y_m2", num, CAT, wt)
        for c in CATS:
            rows.append(term_row(fk, "c_" + c, name, "roles (ref gov_main)"))
        rows.append(contrast_row(fk, {"c_opp_other": 1, "c_opp_main": -1}, "opp_other - opp_main", name, "roles"))
    out["t2_pooled"] = pd.DataFrame(rows)
    # single-asker questions only
    ds = dd[dd["single"]]
    rows = []
    for name, sub, wt in (("pooled LS13-18 (weighted)", ds, "m2_weight"), ("LS17-18", ds[ds["ls"] >= 17], None),
                          ("LS18", ds[ds["ls"] == 18], None), ("LS17", ds[ds["ls"] == 17], None)):
        cs = [c for c in CATS if c not in ("gov_both", "opp_both", "mixed")]
        fk = fit(sub, "y_m2", ["c_" + c for c in cs] + CTRL, CAT, wt)
        for c in cs:
            rows.append(term_row(fk, "c_" + c, name, "roles, single-asker only (ref gov_main)"))
    out["t2_single"] = pd.DataFrame(rows)
    # LS18 / LS17 coalition partners with ministry_canonical FE (do partners ask different ministries?)
    rows = []
    for k in (17, 18):
        sub = dd[(dd["ls"] == k) & dd["single"]]
        cs = ["gov_partner", "opp_main", "opp_other"] + (["outside"] if k == 18 else [])
        for lab, cats in (("ministry_group FE", CAT), ("ministry_canonical FE", ["ministry_canonical", "topic", "session_id"])):
            fk = fit(sub, "y_m2", ["c_" + c for c in cs] + CTRL, cats, None)
            for c in cs:
                rows.append(term_row(fk, "c_" + c, f"LS{k}", f"single-asker, {lab}"))
        fk = fit(sub, "y_m2", ["c_" + c for c in cs] + CTRL, CAT, None, cluster=("mp_id",))
        for c in cs:
            rows.append(term_row(fk, "c_" + c, f"LS{k}", "single-asker, ministry_group FE, MP-cluster only"))
    out["t2_minfe"] = pd.DataFrame(rows)
    # party level within government single-asker questions, LS18 and LS17 (ref BJP)
    rows = []
    for k in (18, 17):
        sub = dd[(dd["ls"] == k) & dd["single"] & dd["qcat"].isin(["gov_main", "gov_partner"])].copy()
        vc = sub[sub["qcat"] == "gov_partner"]["asker_party"].value_counts()
        keep = [p for p, n in vc.items() if n >= 150]
        sub["p_other_partner"] = ((sub["qcat"] == "gov_partner") & ~sub["asker_party"].isin(keep)).astype(int)
        cols = []
        for p in keep:
            sub["pp_" + p] = (sub["asker_party"] == p).astype(int)
            cols.append("pp_" + p)
        fk = fit(sub, "y_m2", cols + ["p_other_partner"] + CTRL, CAT, None)
        fkm = fit(sub, "y_m2", cols + ["p_other_partner"] + CTRL, CAT, None, cluster=("mp_id",))
        for c in cols + ["p_other_partner"]:
            r = term_row(fk, c, f"LS{k}", "gov single-asker, party vs BJP",
                         {"n_party": int((sub[c] == 1).sum()), "mean_y_party": sub.loc[sub[c] == 1, "y_m2"].mean()})
            rm = term_row(fkm, c, f"LS{k}", "gov single-asker, party vs BJP, MP-cluster only")
            r["p_mp"], r["ci_mp_lo"], r["ci_mp_hi"] = rm["p"], rm["ci_lo"], rm["ci_hi"]
            rows.append(r)
        rows.append({"term": "BJP (reference)", "estimate": 0, "se": np.nan, "ci_lo": np.nan, "ci_hi": np.nan, "p": np.nan,
                     "group": f"LS{k}", "n_party": int((sub["qcat"] == "gov_main").sum()),
                     "mean_y_party": sub.loc[sub["qcat"] == "gov_main", "y_m2"].mean()})
    out["t2_party"] = pd.DataFrame(rows)
    # raw means by qcat in LS18
    rr = dd[dd["ls"] >= 17].groupby(["ls", "qcat"])["y_m2"].agg(["mean", "size"]).reset_index()
    out["t2_raw1718"] = rr
    # who are the partners? (list derived, LS-level)
    pt = dd[dd["qcat"] == "gov_partner"].groupby(["ls", "asker_party"]).size().reset_index(name="n")
    pt = pt.sort_values(["ls", "n"], ascending=[True, False]).groupby("ls").head(6)
    out["t2_partner_list"] = pt


# ---------------------------------------------------------------------------------------------- task 3
def gap_by_split(d, split_col, labels, y, wt, num_extra, cat, group_prefix, spec, ctrl=CTRL):
    rows = []
    for val, lab in labels:
        sub = d[d[split_col] == val]
        fk = fit(sub, y, BLOC + ctrl + num_extra, cat, wt)
        rows.append(term_row(fk, "opp", f"{group_prefix}{lab}", spec,
                             {"mean_y": np.average(sub[y], weights=None if wt is None else sub[wt]),
                              "n_opp": int(sub["opp"].sum())}))
    return rows


def task3(d, out, parts):
    res = {}
    # ---- starred vs unstarred
    for name, sub, wt in (("pooled LS13-18 (weighted)", d, "m2_weight"), ("LS17-18", d[d["ls"] >= 17], None)):
        rows = gap_by_split(sub, "starred", [(0, "unstarred"), (1, "starred")], "y_m2", wt, [], CAT, f"{name}: ", "split sample")
        fi = fit(sub, "y_m2", BLOC + ["starred", "starred_x_opp"] + CTRL, CAT, wt)
        rows.append(term_row(fi, "starred_x_opp", f"{name}: interaction", "opp x starred (starred minus unstarred gap)"))
        rows.append(contrast_row(fi, {"opp": 1, "starred_x_opp": 1}, "gap among starred (interaction model)", name, "interaction"))
        res.setdefault("starred", []).extend(rows)
    out["t3_starred"] = pd.DataFrame(res["starred"])
    # ---- security vs other
    res = []
    for name, sub, wt in (("pooled LS13-18 (weighted)", d, "m2_weight"), ("LS17-18", d[d["ls"] >= 17], None)):
        rows = gap_by_split(sub, "security", [(0, "other ministries"), (1, "security")], "y_m2", wt, [], CAT, f"{name}: ", "split sample")
        fi = fit(sub, "y_m2", BLOC + ["sec_x_opp"] + CTRL, CAT, wt)
        rows.append(term_row(fi, "sec_x_opp", f"{name}: interaction", "opp x security (security minus other gap)"))
        res.extend(rows)
    out["t3_security"] = pd.DataFrame(res)
    # ---- by ministry group (LS17-18 and pooled)
    res = []
    for name, sub, wt in (("pooled LS13-18 (weighted)", d, "m2_weight"), ("LS17-18", d[d["ls"] >= 17], None)):
        for g in sorted(d["ministry_group"].unique()):
            s2 = sub[sub["ministry_group"] == g]
            fk = fit(s2, "y_m2", BLOC + CTRL, [c for c in CAT if c != "ministry_group"], wt)
            res.append(term_row(fk, "opp", f"{name}: {g}", "by ministry group", {"n_opp": int(s2["opp"].sum())}))
    out["t3_mgroup"] = pd.DataFrame(res)
    # ---- by request type (part level)
    p = parts[parts["qid"].isin(d["qid"])].merge(
        d[["qid", "ls", "session_id", "mp_id", "opp", "mixed", "outside", "topic", "ministry_group", "n_parts", "log_qlen",
           "coasker", "m2_weight"]], on="qid")
    p["ps"] = p["label"].map(LABEL_SCORE)
    out["t3_rt_counts"] = p.groupby("request_type").agg(n_parts=("ps", "size"), mean_score=("ps", "mean")).reset_index()
    res = []
    for name, sub, wt in (("pooled LS13-18 (weighted)", p, "m2_weight"), ("LS17-18", p[p["ls"] >= 17], None)):
        for rt in ("INFORMATION", "REASON", "ACTION", "OPINION"):
            s2 = sub[sub["request_type"] == rt]
            fk = fit(s2, "ps", BLOC + CTRL_NORT, CAT, wt)
            res.append(term_row(fk, "opp", f"{name}: {rt}", "part-level, by request type",
                                {"n_opp": int(s2["opp"].sum()),
                                 "mean_y": np.average(s2["ps"], weights=None if wt is None else s2[wt])}))
        fa = fit(sub, "ps", BLOC + CTRL_NORT, CAT, wt)
        res.append(term_row(fa, "opp", f"{name}: all parts", "part-level, all parts"))
        # interaction model: request-type dummies and opp x rt (INFORMATION = reference)
        for rt in ("REASON", "ACTION", "OPINION"):
            sub[f"rt_{rt}"] = (sub["request_type"] == rt).astype(int)
            sub[f"opp_x_{rt}"] = sub["opp"] * sub[f"rt_{rt}"]
        fi = fit(sub, "ps", BLOC + [f"rt_{r}" for r in ("REASON", "ACTION", "OPINION")]
                 + [f"opp_x_{r}" for r in ("REASON", "ACTION", "OPINION")] + CTRL_NORT, CAT, wt)
        for rt in ("REASON", "ACTION", "OPINION"):
            res.append(term_row(fi, f"opp_x_{rt}", f"{name}: opp x {rt}", "part-level interaction (minus INFORMATION gap)"))
        W, df_, pc, pf = wald(fi, [f"opp_x_{r}" for r in ("REASON", "ACTION", "OPINION")] + ["opp"])
        res.append({"term": "joint test: 4 gaps equal (chi2)", "estimate": W, "se": np.nan, "ci_lo": np.nan, "ci_hi": np.nan,
                    "p": pc, "group": name, "spec": "Wald, df=%d" % df_, "n": fi.n})
    out["t3_rt"] = pd.DataFrame(res)


# ---------------------------------------------------------------------------------------------- task 4
def task4(d, out, f_all):
    # (a) single-asker only
    ds = d[d["single"]]
    rows = []
    for k in LSS:
        fk = fit(ds[ds["ls"] == k], "y_m2", BLOC + CTRL_NORT + ["rt_reason", "rt_action", "rt_opinion"], CAT, "m2_weight")
        rows.append(term_row(fk, "opp", f"LS{k}", "single-asker only"))
    for name, sub, wt in (("pooled LS13-18 (weighted)", ds, "m2_weight"), ("LS17-18", ds[ds["ls"] >= 17], None)):
        fk = fit(sub, "y_m2", BLOC + CTRL, CAT, wt)
        rows.append(term_row(fk, "opp", name, "single-asker only"))
    out["t4_single"] = pd.DataFrame(rows)
    # (b) strict coding
    dst = d.dropna(subset=["asker_bloc_strict"])
    sb = ["opp_strict", "mixed_strict", "outside_strict"]
    rows = []
    for k in LSS:
        fk = fit(dst[dst["ls"] == k], "y_m2", sb + CTRL, CAT, "m2_weight")
        rows.append(term_row(fk, "opp_strict", f"LS{k}", "strict bloc coding"))
    for name, sub, wt in (("pooled LS13-18 (weighted)", dst, "m2_weight"), ("LS17-18", dst[dst["ls"] >= 17], None)):
        fk = fit(sub, "y_m2", sb + CTRL, CAT, wt)
        rows.append(term_row(fk, "opp_strict", name, "strict bloc coding"))
    out["t4_strict"] = pd.DataFrame(rows)
    # (c) gap size vs overall decline
    w = d["m2_weight"]
    mean_ls = {k: np.average(d[d["ls"] == k]["y_m2"], weights=d[d["ls"] == k]["m2_weight"]) for k in LSS}
    pre = d[d["ls"] <= 16]
    post = d[d["ls"] >= 17]
    m_pre, m_post = np.average(pre["y_m2"], weights=pre["m2_weight"]), post["y_m2"].mean()
    sh_post = post["opp"].mean()
    sh_18 = d[d["ls"] == 18]["opp"].mean()
    dec = {"mean_LS13": mean_ls[13], "mean_LS16": mean_ls[16], "mean_LS17": mean_ls[17], "mean_LS18": mean_ls[18],
           "mean_LS13-16 (weighted)": m_pre, "mean_LS17-18": m_post,
           "decline LS13-16 -> LS17-18": m_post - m_pre, "decline LS13 -> LS18": mean_ls[18] - mean_ls[13],
           "decline LS16 -> LS18": mean_ls[18] - mean_ls[16], "opp share LS17-18": sh_post, "opp share LS18": sh_18}
    g1718 = [r for r in LOG if r["group"] == "LS17-18" and r["spec"] == "session FE" and r["term"] == "opp"][0]
    g18 = [r for r in LOG if r["group"] == "LS18" and r["spec"] == "two-way (MP,session)" and r["term"] == "opp"][0]
    dec["gap LS17-18 (beta)"] = g1718["estimate"]
    dec["mean gap LS17-18 = beta x opp share"] = g1718["estimate"] * sh_post
    dec["mean-gap / decline (LS13-16 -> LS17-18)"] = g1718["estimate"] * sh_post / (m_post - m_pre)
    pp_ = [r for r in LOG if r["spec"] == "pre/post: change in gap"][0]
    pre_ = [r for r in LOG if r["spec"] == "pre/post: gap LS13-16"][0]
    dec["gap LS13-16 (beta, pre/post model)"] = pre_["estimate"]
    dec["opp share LS13-16 (weighted)"] = np.average(pre["opp"], weights=pre["m2_weight"])
    dec["change in mean gap = post gap x share_post - pre gap x share_pre"] = (
        (pre_["estimate"] + pp_["estimate"]) * sh_post - pre_["estimate"] * np.average(pre["opp"], weights=pre["m2_weight"]))
    dec["  as share of decline LS13-16 -> LS17-18"] = dec["change in mean gap = post gap x share_post - pre gap x share_pre"] / (m_post - m_pre)
    dec["gap LS18 (beta)"] = g18["estimate"]
    dec["mean gap LS18 = beta x opp share"] = g18["estimate"] * sh_18
    dec["mean-gap LS18 / decline (LS16 -> LS18)"] = g18["estimate"] * sh_18 / (mean_ls[18] - mean_ls[16])
    dec["mean-gap LS18 / decline (LS13 -> LS18)"] = g18["estimate"] * sh_18 / (mean_ls[18] - mean_ls[13])
    # decline among government askers vs opposition askers (bloc-specific change)
    for b in ("government", "opposition"):
        a_pre = pre[pre["asker_bloc"] == b]
        a_post = post[post["asker_bloc"] == b]
        dec[f"{b}: LS13-16 -> LS17-18"] = a_post["y_m2"].mean() - np.average(a_pre["y_m2"], weights=a_pre["m2_weight"])
    out["t4_decline"] = dec
    # decline net of controls: LS18 vs LS17 and LS17-18 vs LS13-16 post dummy (controls+topic+ministry FE, no session FE)
    fpost = fit(d, "y_m2", ["post"] + BLOC + CTRL, ["ministry_group", "topic"], "m2_weight")
    out["t4_post"] = term_row(fpost, "post", "LS17-18 vs LS13-16", "conditional decline (no session FE)")
    # (d) gold-only
    g = f_all[f_all["analysis_sample"] & f_all["gold_complete"] & f_all["y_gold"].notna() & f_all["gold_pi"].notna()].copy()
    g["gw"] = 1.0 / g["gold_pi"]
    gl = ["n_parts", "log_qlen", "coasker"]
    rows = []
    gcat = []
    for name, sub in (("gold pooled LS13-18", g), ("gold LS17-18", g[g["ls"] >= 17]), ("gold LS13-16", g[g["ls"] <= 16])):
        for wlab, wt in (("1/pi weights", "gw"), ("unweighted", None)):
            for y in ("y_gold",):
                sub2 = sub.assign(ls_s=sub["ls"].astype(str))
                try:
                    fk = fit(sub2, y, BLOC + gl, ["ls_s"], wt)
                    rows.append(term_row(fk, "opp", name, f"y_gold, {wlab}, LS FE + core ctrl",
                                         {"n_opp": int(sub["opp"].sum()), "n_ls": sub["ls"].nunique()}))
                except Exception as e:                                       # pragma: no cover
                    rows.append({"term": "opp", "group": name, "spec": f"y_gold, {wlab}: {e}", "estimate": np.nan})
        # same lean spec on y_m2 for the SAME questions and for the full M2 sample (apples-to-apples)
    gm = g[g["y_m2"].notna()]
    for name, sub in (("gold pooled LS13-18", gm), ("gold LS17-18", gm[gm["ls"] >= 17])):
        sub2 = sub.assign(ls_s=sub["ls"].astype(str))
        fk = fit(sub2, "y_m2", BLOC + gl, ["ls_s"], None)
        rows.append(term_row(fk, "opp", name, "y_m2 on the gold questions, unweighted, LS FE + core ctrl", {"n_opp": int(sub["opp"].sum())}))
    for name, sub, wt in (("M2 full pooled LS13-18", d, "m2_weight"), ("M2 full LS17-18", d[d["ls"] >= 17], None)):
        sub2 = sub.assign(ls_s=sub["ls"].astype(str))
        fk = fit(sub2, "y_m2", BLOC + gl, ["ls_s"], wt)
        rows.append(term_row(fk, "opp", name, "y_m2 full sample, LS FE + core ctrl (lean spec)", {"n_opp": int(sub["opp"].sum())}))
    out["t4_gold"] = pd.DataFrame(rows)
    out["t4_gold_n"] = g.groupby("ls").agg(n=("qid", "size"), n_opp=("opp", "sum"), mean_gold=("y_gold", "mean")).reset_index()
    # raw gold difference with per-stratum CIs (bootstrap over questions, 1/pi weights), LS17-18
    rng = np.random.default_rng(20261018)
    rawrows = []
    for name, sub in (("gold LS13-16", g[g["ls"] <= 16]), ("gold LS17-18", g[g["ls"] >= 17])):
        s1 = sub[sub["asker_bloc"].isin(["government", "opposition"])]
        def diff(x):
            a_, b_ = x[x["asker_bloc"] == "opposition"], x[x["asker_bloc"] == "government"]
            return np.average(a_["y_gold"], weights=a_["gw"]) - np.average(b_["y_gold"], weights=b_["gw"])
        est = diff(s1)
        bs = [diff(s1.iloc[rng.integers(0, len(s1), len(s1))]) for _ in range(1000)]
        rawrows.append({"group": name, "n": len(s1), "raw opp - gov (1/pi weighted)": est,
                        "boot_lo": np.percentile(bs, 2.5), "boot_hi": np.percentile(bs, 97.5)})
    out["t4_gold_raw"] = pd.DataFrame(rawrows)


# ---------------------------------------------------------------------------------------------- figures
def figures(out):
    t = out["t1_by_ls"]
    x = np.arange(len(LSS))
    CB = {"blue": "#0072B2", "orange": "#D55E00", "green": "#009E73", "grey": "#666666", "purple": "#CC79A7"}
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.15, 1]})
    a = ax[0]
    a.axhline(0, color="#999", lw=0.8)
    a.errorbar(x - 0.12, t["estimate"], yerr=[t["estimate"] - t["ci_lo"], t["ci_hi"] - t["estimate"]], fmt="o", color=CB["blue"],
               capsize=3, label="opposition vs government, 95% CI two-way (MP, session)")
    a.errorbar(x + 0.12, t["estimate"], yerr=[t["estimate"] - t["ci_mp_lo"], t["ci_mp_hi"] - t["estimate"]], fmt="none",
               ecolor=CB["orange"], capsize=3, label="95% CI, MP-clustered only")
    s = out["t4_strict"].iloc[:6]
    a.plot(x + 0.28, s["estimate"], "s", color=CB["green"], ms=4, label="strict bloc coding (point)")
    a.set_xticks(x)
    a.set_xticklabels([f"LS{k}" for k in LSS])
    a.set_ylabel("coefficient on opposition askers (y_m2 units)")
    a.set_title("Opposition gap within each Lok Sabha")
    a.legend(fontsize=7.5, frameon=False, loc="lower left")
    b = ax[1]
    raw = out["t1_raw"]
    b.plot(x, raw["government"], "o-", color=CB["blue"], label="government askers")
    b.plot(x, raw["opposition"], "o-", color=CB["orange"], label="opposition askers")
    b.plot(x, raw["all"], "k--", lw=1, label="all (weighted)")
    b.set_xticks(x)
    b.set_xticklabels([f"LS{k}" for k in LSS])
    b.set_ylabel("mean y_m2 (design weighted)")
    b.set_title("Raw means: the decline hits both blocs")
    b.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "rq3_fig1_gap_by_ls.png"), dpi=160)
    plt.close(fig)

    # fig 2: roles by LS
    r = out["t2_by_ls"]
    r = r[r["spec"] == "roles by LS (ref gov_main)"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    series = [("c_gov_partner", "coalition partner", CB["green"], -0.15), ("c_opp_main", "main opposition", CB["orange"], 0.0),
              ("c_opp_other", "other opposition", CB["purple"], 0.15)]
    a = ax[0]
    a.axhline(0, color="#999", lw=0.8)
    for term, lab, col, off in series:
        q = r[r["term"] == term].set_index("group").reindex([f"LS{k}" for k in LSS])
        a.errorbar(x + off, q["estimate"], yerr=[q["estimate"] - q["ci_lo"], q["ci_hi"] - q["estimate"]], fmt="o", color=col, capsize=2.5, label=lab)
    a.set_xticks(x)
    a.set_xticklabels([f"LS{k}" for k in LSS])
    a.set_ylabel("difference vs main-ruling-party askers (y_m2 units)")
    a.set_title("Roles by Lok Sabha (ref: INC under UPA, BJP under NDA)")
    a.legend(fontsize=8, frameon=False)
    p = out["t2_party"]
    p = p[(p["group"] == "LS18") & (p["term"] != "BJP (reference)")].copy()
    p["lab"] = p["term"].str.replace("pp_", "").str.replace("p_other_partner", "other partners") + " (n=" + p["n_party"].astype(int).astype(str) + ")"
    b = ax[1]
    yy = np.arange(len(p))[::-1]
    b.axvline(0, color="#999", lw=0.8)
    b.errorbar(p["estimate"], yy, xerr=[p["estimate"] - p["ci_lo"], p["ci_hi"] - p["estimate"]], fmt="o", color=CB["green"], capsize=3)
    b.set_yticks(yy)
    b.set_yticklabels(p["lab"])
    b.set_xlabel("difference vs BJP askers (LS18, gov single-asker)")
    b.set_title("LS18 coalition partners by party")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "rq3_fig2_roles.png"), dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------------------------- report
def write_md(out, f_all, d):
    L = []
    w = L.append
    ci = ["term", "group", "estimate", "ci_lo", "ci_hi", "p", "n"]
    w("# RQ3 (exploratory): who asks, and does the opposition gap depend on the question?\n")
    w("Statistics only. Outcome y_m2 (0-1) unless stated; OLS with the models.py H1/H3 controls (n_parts, log_qlen, co-askers, "
      "request-type shares), ministry_group + topic (K=50) + session FE; design weights m2_weight (pooled LS13-18; LS17/18 are a census, weight 1); "
      "SEs two-way clustered by first-asker MP and session, t(min G-1) CIs. Reference = government askers. "
      "Every model run is in `rq3_all_models.csv` (%d coefficient rows logged); p-values are raw and unadjusted for the many comparisons here.\n" % len(LOG))
    w("Sample: %d questions with a design weight (LS13 %d, LS14 %d, LS15 %d, LS16 %d, LS17 %d, LS18 %d); weighted shares in the tables below.\n"
      % (len(d), *[int((d["ls"] == k).sum()) for k in LSS]))

    # --- task 1
    w("\n## 1. Opposition gap by Lok Sabha\n")
    t = out["t1_by_ls"].copy()
    t["CI two-way"] = [f"[{a:+.4f}, {b:+.4f}]" for a, b in zip(t["ci_lo"], t["ci_hi"])]
    t["CI MP-only"] = [f"[{a:+.4f}, {b:+.4f}]" for a, b in zip(t["ci_mp_lo"], t["ci_mp_hi"])]
    t["sessions"] = t["g_sess"].astype(int)
    w(md(t.assign(LS=t["group"]), ["LS", "estimate", "CI two-way", "p", "CI MP-only", "p_mp", "n", "n_opp", "share_opp_w", "sessions", "mean_y"], "{:.4f}"))
    w("\n(`estimate` = coefficient on opposition askers, controls + ministry_group, topic and within-LS session FE; `p` two-way t(sessions-1), `p_mp` MP-clustered; "
      "`share_opp_w` weighted share of all-opposition questions; `mean_y` weighted LS mean.)\n")
    w("Raw weighted means of y_m2 by LS and asker bloc (no controls):\n")
    w(md(out["t1_raw"], ["ls", "government", "opposition", "mixed", "outside_support", "all", "raw_opp_minus_gov"], "{:.3f}"))
    w("\nSensitivity of the LS-level opposition coefficient to the fixed effects:\n")
    s = pd.DataFrame([r for r in LOG if r["spec"] in ("no topic/ministry FE", "ministry_canonical FE") and r["term"] == "opp"])
    sp = s.pivot(index="group", columns="spec", values="estimate")
    sp2 = s.pivot(index="group", columns="spec", values="p")
    sp.columns = [c + " (est)" for c in sp.columns]
    sp2.columns = [c + " (p)" for c in sp2.columns]
    w(C.df_to_markdown(pd.concat([sp, sp2], axis=1).reset_index().rename(columns={"group": "LS"}), floatfmt="{:.4f}"))
    w("\nPooled model with LS-specific opposition coefficients (one model, session FE, weights):\n")
    w(md(out["t1_pooled_ls"], ["group", "estimate", "ci_lo", "ci_hi", "p", "n"], "{:.4f}"))
    et = out["t1_equal_test"]
    w(f"\nJoint Wald test that the six LS gaps are equal: chi2({et['df']}) = {et['W']:.2f}, p = {et['p_chi2']:.3g} (p = {et['p_F']:.3g} with F, t-df).\n")
    w("Trend and pre/post tests:\n")
    tt = pd.DataFrame(out["t1_trend"] + out["t1_prepost"])
    w(md(tt, ["spec", "term", "estimate", "ci_lo", "ci_hi", "p", "n"], "{:.4f}"))
    w("\nReference estimates (replication of the known numbers):\n")
    w(md(pd.DataFrame(out["t1_ref"]), ["group", "spec", "estimate", "ci_lo", "ci_hi", "p", "n"], "{:.4f}"))

    # --- task 2
    w("\n## 2. Finer asker roles\n")
    w("Categories (derived from coalitions.csv, see script docstring): gov_main = INC (UPA periods) / BJP (NDA periods); gov_partner = other "
      "government-role parties; opp_main = BJP under UPA / INC under NDA; opp_other = all other opposition; outside = outside_support; "
      "`*_both` = multi-asker questions mixing sub-categories inside one bloc; mixed = askers from different blocs.\n")
    w("Questions per category (M2 sample, unweighted counts):\n")
    w(C.df_to_markdown(out["t2_counts"], floatfmt="{:,.0f}"))
    w("\nMost frequent coalition-partner askers (single and multi-asker questions in the M2 sample, by asker_party = first asker of single-asker rows; top 6 per LS):\n")
    w(C.df_to_markdown(out["t2_partner_list"].rename(columns={"asker_party": "party"}), floatfmt="{:,.0f}"))
    w("\nCoefficients relative to main-ruling-party askers (gov_main), by Lok Sabha (full spec, weights, within-LS session FE):\n")
    r = out["t2_by_ls"]
    piv = r[r["spec"] == "roles by LS (ref gov_main)"]
    tab = piv.assign(val=[fmt_ci(x, p=False) + f" (p={x['p']:.2g})" if not np.isnan(x["estimate"]) else "n/a" for x in piv.to_dict("records")]
                     ).pivot(index="term", columns="group", values="val").reindex(["c_" + c for c in CATS])[[f"LS{k}" for k in LSS]]
    w(C.df_to_markdown(tab.reset_index(), floatfmt="{:.3f}"))
    cc = r[r["spec"] == "roles by LS"]
    w("\nContrasts (same models):\n")
    w(md(cc, ["term", "group", "estimate", "ci_lo", "ci_hi", "p"], "{:.4f}"))
    w("\nPooled / LS17-18 / LS18 (ref gov_main):\n")
    pp = out["t2_pooled"]
    w(md(pp, ["group", "term", "estimate", "ci_lo", "ci_hi", "p", "n"], "{:.4f}"))
    w("\nSingle-asker questions only (clean coding; ref gov_main; same controls and FE):\n")
    w(md(out["t2_single"], ["group", "term", "estimate", "ci_lo", "ci_hi", "p", "n"], "{:.4f}"))
    w("\nLS17 and LS18 single-asker questions: ministry_group FE vs ministry_canonical FE (do partners simply ask different ministries?):\n")
    w(md(out["t2_minfe"], ["group", "spec", "term", "estimate", "ci_lo", "ci_hi", "p", "n"], "{:.4f}"))
    w("\nWithin government single-asker questions, party vs BJP (parties with at least 150 questions; the rest pooled as `other partners`):\n")
    pt = out["t2_party"]
    w(md(pt, ["group", "term", "n_party", "mean_y_party", "estimate", "ci_lo", "ci_hi", "p", "p_mp"], "{:.4f}"))
    w("\nRaw mean y_m2 by question category, LS17-18 (unweighted census):\n")
    w(md(out["t2_raw1718"], ["ls", "qcat", "mean", "size"], "{:.3f}"))

    # --- task 3
    w("\n## 3. Does the gap depend on the question?\n")
    w("### Starred vs unstarred\n")
    w(md(out["t3_starred"], ["group", "term", "spec", "estimate", "ci_lo", "ci_hi", "p", "n", "n_opp", "mean_y"], "{:.4f}"))
    w("\n### Security vs other ministries\n")
    w(md(out["t3_security"], ["group", "spec", "estimate", "ci_lo", "ci_hi", "p", "n", "n_opp", "mean_y"], "{:.4f}"))
    w("\nBy ministry group (opposition coefficient within each group; topic and session FE):\n")
    w(md(out["t3_mgroup"], ["group", "estimate", "ci_lo", "ci_hi", "p", "n", "n_opp"], "{:.4f}"))
    w("\n### By request type (part level)\n")
    w("Each row is a part-level regression (score of each question part: FULL=1, PARTIAL=.5, NONE=0) on the question's asker bloc; "
      "parts of a question share the question's regressors, so inference is clustered by MP and session; part-level means weight questions by their number of parts, "
      "so the 'all parts' row is not identical to the question-level estimate. Controls: n_parts, log_qlen, coasker, ministry_group, topic, session FE.\n")
    w(C.df_to_markdown(out["t3_rt_counts"], floatfmt="{:,.3f}"))
    w("")
    w(md(out["t3_rt"], ["group", "spec", "estimate", "ci_lo", "ci_hi", "p", "n", "n_opp", "mean_y"], "{:.4f}"))

    # --- task 4
    w("\n## 4. Robustness\n")
    w("### Single-asker questions only\n")
    w(md(out["t4_single"], ["group", "spec", "estimate", "ci_lo", "ci_hi", "p", "n"], "{:.4f}"))
    w("\n### Strict bloc coding (asker_bloc_strict, opp_strict vs strict government)\n")
    w(md(out["t4_strict"], ["group", "estimate", "ci_lo", "ci_hi", "p", "n"], "{:.4f}"))
    w("\n### Size of the mean gap relative to the overall decline\n")
    dec = out["t4_decline"]
    w(C.df_to_markdown(pd.DataFrame({"quantity": list(dec), "value": list(dec.values())}), floatfmt="{:.4f}"))
    pr = out["t4_post"]
    w(f"\nConditional decline (controls, ministry_group + topic FE, no session FE): LS17-18 vs LS13-16 = {pr['estimate']:+.4f} [{pr['ci_lo']:+.4f}, {pr['ci_hi']:+.4f}], p={pr['p']:.3g}.\n")
    w("### Gold (human-coded) questions, y_gold\n")
    w("Gold-complete questions (all parts human-coded), weights 1/gold_pi (inclusion probability from the main-sample draw); lean spec "
      "(n_parts, log_qlen, coasker, LS FE; no topic or ministry FE because n is small). Compare with the same lean spec on y_m2.\n")
    w(C.df_to_markdown(out["t4_gold_n"], floatfmt="{:,.3f}"))
    w("")
    w(md(out["t4_gold"], ["group", "spec", "estimate", "ci_lo", "ci_hi", "p", "n", "n_opp"], "{:.4f}"))
    w("\nRaw weighted opposition-minus-government difference in y_gold (bootstrap over questions, 1000 draws, percentile CI):\n")
    w(md(out["t4_gold_raw"], ["group", "n", "raw opp - gov (1/pi weighted)", "boot_lo", "boot_hi"], "{:.4f}"))

    w("\n## Figures\n\n- `rq3_fig1_gap_by_ls.png`: opposition coefficient by Lok Sabha (two-way and MP-only CIs, strict-coding point) and raw means by bloc.\n"
      "- `rq3_fig2_roles.png`: coalition-partner / main-opposition / other-opposition coefficients by Lok Sabha; LS18 coalition partners by party vs BJP.\n")
    wh = os.path.join(OUT, "rq3_what_holds.txt")
    if os.path.exists(wh):
        w("\n" + open(wh, encoding="utf-8").read())
    open(os.path.join(OUT, "rq3_askers.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")


def main():
    f_all, a, anchors = load()
    print("anchors:", anchors)
    d = sample_m2(f_all)
    # consistency check of derived categories with the frame's bloc coding
    chk = pd.crosstab(f_all["asker_bloc"], f_all["qcat"].fillna("NA"))
    print(chk.to_string())
    out = {}
    task1(d, out)
    print(out["t1_by_ls"][["group", "estimate", "ci_lo", "ci_hi", "p", "ci_mp_lo", "ci_mp_hi", "p_mp"]].round(4).to_string())
    task2(d, out, f_all)
    task3(d, out, pd.read_parquet(os.path.join(C.MEASURES, "m2_labels.parquet"), columns=["part_id", "qid", "label", "request_type"]))
    task4(d, out, f_all)
    figures(out)
    pd.DataFrame(LOG).to_csv(os.path.join(OUT, "rq3_all_models.csv"), index=False)
    write_md(out, f_all, d)
    print("done; models logged:", len(LOG))


if __name__ == "__main__":
    main()
