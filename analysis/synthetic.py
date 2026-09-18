"""SYNTHETIC outcomes with known planted effects, generated on the REAL covariate frame.

    *** Everything written here is synthetic and lives ONLY under data/measures/synthetic/ . None of it is a result. ***

Purpose: validate that the estimators in models.py / bias_correct.py recover the planted H1-H4 effects before any real
outcome exists, and let the full pipeline (frame -> models -> robustness -> figures) run end to end.

Data-generating process (question q with k_q parts, asker bloc, ministry group, period, ...):

    mu_q = BASE + PLANTED effects + controls + topic/session/MP random effects         (E[question score | X] = mu_q)
    each part: FULL / PARTIAL / NONE with P(FULL)=mu-0.5*p_part, P(PARTIAL)=p_part=min(.3,2mu,2(1-mu)), rest NONE
    -> question true score = mean part score (LABEL_SCORE), exactly the real outcome definition (E = mu_q).

    "M2" labels = true labels with measurement error that is DIFFERENTIAL (more upward errors for government
    askers, more downward errors for security ministries) -> naive analyses of M2 are biased, bias-corrected
    ones (PPI++) should not be.  "Human gold" = true labels on a stratified random sample of 600 questions with
    known inclusion probabilities (written in the format of data/coding/main_sample.csv and gold_labels.csv).
    M1 = noisy monotone transform of the true score (plus a second embedding model).

Planted effects (in score units, scale 0-1) are in PLANTED / written to planted_effects.json:
    H1  opposition gap under coalition dependence ......... opp                 = -0.030
        opposition gap under single-party majority ......... opp - opp_x_coal    = -0.060   (so H4 interaction on
    H2  coalition dependence (vs majority), non-opp askers  = +0.030                          coalition_dep is +0.030)
    H3  security ministries ................................ = -0.060
    H4  opp x coalition_dep ................................ = +0.030
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import common as C  # noqa: E402

PLANTED = {
    "base": 0.55,
    "opp": -0.060,              # opposition effect when coalition_dep == 0 (single-party majority)
    "opp_x_coalition": 0.030,   # H4 interaction -> opposition effect under coalition dependence = -0.030
    "coalition_dep": 0.030,     # H2
    "mixed": -0.020,
    "outside": 0.010,
    "security": -0.060,         # H3
    "starred": 0.020,
    "n_parts": -0.010,          # per part
    "log_qlen": -0.020,         # per log-unit of question length (centred)
    "coasker": 0.004,           # per co-asker (capped at 5)
    "ministry_group_sd": 0.03,
    "topic_sd": 0.04,
    "session_sd": 0.015,
    "mp_sd": 0.03,
}
GOLD_N = 600
SEED = C.SEED_SYNTHETIC


def planted_truth():
    """Planted coefficients in models.py's parametrisation. For H2 (no interaction in the model) the estimand is the
    average effect beta_c + beta_int * share_opp; the exact probability limit of every spec is obtained by fitting the
    same spec to the noise-free mu_q stored in truth_scores.parquet ('oracle')."""
    return {"H1_opp_under_coalition": PLANTED["opp"] + PLANTED["opp_x_coalition"],
            "H1_opp_under_majority": PLANTED["opp"],
            "H2_coalition_dep_nonopp": PLANTED["coalition_dep"],
            "H3_security": PLANTED["security"],
            "H4_opp_x_coalition_dep": PLANTED["opp_x_coalition"]}


def _paths():
    d = C.SYN_MEASURES
    return {k: os.path.join(d, v) for k, v in dict(
        m2="m2_labels.parquet", m1="m1_question.parquet", gold="gold_labels.csv", sample="main_sample.csv",
        truth="truth_scores.parquet", meta="planted_effects.json").items()}


def simulate_mu(q, rng, P=PLANTED):
    """Noise-free question mean mu_q = E[question score | covariates, random effects] (see module docstring)."""
    n = len(q)
    topic = q["topic"].fillna(-1).astype(int)
    topic_eff = pd.Series(rng.normal(0, P["topic_sd"], 51), index=range(-1, 50)).reindex(topic).values
    topic_eff = np.where(topic.values < 0, 0.0, topic_eff)
    groups = sorted(q["ministry_group"].unique())
    g_eff = dict(zip(groups, rng.normal(0, P["ministry_group_sd"], len(groups))))
    g_eff[C.SECURITY_GROUP] = 0.0     # the planted security effect is the whole security shift
    sz = q["ministry_group"].value_counts()
    oth = [g for g in groups if g != C.SECURITY_GROUP]
    cen = sum(g_eff[g] * sz[g] for g in oth) / sum(sz[g] for g in oth)
    for g in oth:                      # other groups average to zero (question-weighted) -> H3 contrast = security effect
        g_eff[g] -= cen
    sess = q["session_id"].unique()
    sess_eff = dict(zip(sess, rng.normal(0, P["session_sd"], len(sess))))
    mps = q["mp_id"].unique()
    mp_eff = dict(zip(mps, rng.normal(0, P["mp_sd"], len(mps))))
    qlen = q["log_qlen"] - q["log_qlen"].mean()
    mu = (P["base"] + P["opp"] * q["opp"] + P["opp_x_coalition"] * q["opp"] * q["coalition_dep"]
          + P["coalition_dep"] * q["coalition_dep"] + P["mixed"] * q["mixed"] + P["outside"] * q["outside"]
          + P["security"] * q["security"] + P["starred"] * q["starred"]
          + P["n_parts"] * (q["n_parts"] - 4) + P["log_qlen"] * qlen + P["coasker"] * q["coasker"].clip(upper=5)
          + topic_eff + q["ministry_group"].map(g_eff).values + q["session_id"].map(sess_eff).values
          + q["mp_id"].map(mp_eff).values)
    return mu.clip(0.05, 0.95).values


def draw_part_scores(mu, n_parts, rng):
    """true question score = mean over n_parts parts of FULL(1)/PARTIAL(.5)/NONE(0) draws with E = mu."""
    mu = np.asarray(mu)
    k = np.asarray(n_parts).astype(int)
    rep = np.repeat(np.arange(len(mu)), k)
    m = mu[rep]
    pp = np.minimum(0.30, np.minimum(2 * m, 2 * (1 - m)))
    u = rng.random(len(rep))
    sc = np.where(u < m - 0.5 * pp, 1.0, np.where(u < m + 0.5 * pp, 0.5, 0.0))
    return np.bincount(rep, weights=sc, minlength=len(mu)) / k


def ensure(q, force=False):
    p = _paths()
    if not force and all(os.path.exists(v) for v in p.values()):
        with open(p["meta"], encoding="utf-8") as fh:
            meta = json.load(fh)
        if meta.get("n_questions") == len(q) and meta.get("planted") == PLANTED:
            return
    generate(q)


def _draw_gold(q, rng):
    """600-question stratified sample (LS x starred, sqrt allocation) from the answered, <=6-part universe."""
    u = q[(~q["answer_missing"]) & (q["n_question_parts"] >= 1) & (q["n_parts"] <= 6)
          & q["govt_period"].notna() & (q["asker_bloc"] != "unresolved")].copy()
    u["stratum"] = u["ls"].astype(str) + "|" + u["qtype"]
    sizes = u["stratum"].value_counts()
    w = np.sqrt(sizes)
    n_h = np.maximum(2, np.round(GOLD_N * w / w.sum())).astype(int).clip(upper=sizes)
    rows = []
    for s, n in n_h.items():
        idx = rng.choice(u.index[u["stratum"] == s].values, size=int(n), replace=False)
        rows.append(pd.DataFrame({"qid": u.loc[idx, "qid"].values, "stratum": s, "N_stratum": int(sizes[s]),
                                  "n_stratum": int(n)}))
    s = pd.concat(rows, ignore_index=True)
    s["sample"] = "main"
    s["in_estimate"] = True
    s["weight"] = s["N_stratum"] / s["n_stratum"]
    return s


def generate(q):
    rng = np.random.default_rng(SEED)
    d = C.SYN_MEASURES
    os.makedirs(d, exist_ok=True)
    P = PLANTED

    mu = simulate_mu(q, rng)
    n = len(q)

    # ---- part-level labels
    parts = pd.read_parquet(C.PARTS, columns=["part_id", "qid", "part_label"])
    qi = pd.Series(np.arange(n), index=q["qid"].values)
    pi_ = qi.reindex(parts["qid"]).values
    ok = ~np.isnan(pi_) & ~q["answer_missing"].values[np.nan_to_num(pi_).astype(int)]
    parts = parts[ok].reset_index(drop=True)
    pi_ = pi_[ok].astype(int)
    m = mu[pi_]
    pp = np.minimum(0.30, np.minimum(2 * m, 2 * (1 - m)))
    pf = m - 0.5 * pp
    u = rng.random(len(parts))
    true_lab = np.where(u < pf, 2, np.where(u < pf + pp, 1, 0))      # 2=FULL 1=PARTIAL 0=NONE
    # measurement error of "M2": differential
    gov = (q["asker_bloc"] == "government").values[pi_]
    sec = q["security"].values[pi_].astype(bool)
    p_up = 0.10 + 0.05 * gov
    p_dn = 0.06 + 0.06 * sec
    v = rng.random(len(parts))
    m2_lab = true_lab.copy()
    m2_lab = np.where((v < p_up) & (true_lab < 2), true_lab + 1, m2_lab)
    m2_lab = np.where((v >= p_up) & (v < p_up + p_dn) & (true_lab > 0), true_lab - 1, m2_lab)
    names = np.array(["NONE", "PARTIAL", "FULL"])
    evas = np.where(m2_lab == 2, "NONE_APPLICABLE", np.where(rng.random(len(parts)) < 0.04, "ANNEXURE_ONLY",
                    rng.choice(["DEFERRAL", "NOT_MAINTAINED", "JURISDICTION", "GENERIC", "DIFFERENT_QUESTION"],
                               len(parts))))
    out = pd.DataFrame({"part_id": parts["part_id"], "qid": parts["qid"], "part_label": parts["part_label"],
                        "label": names[m2_lab], "evasion_type": evas, "model_id": "SYNTHETIC-DO-NOT-USE"})
    out.to_parquet(os.path.join(d, "m2_labels.parquet"), index=False)

    score = pd.Series(true_lab / 2.0)
    truth = score.groupby(parts["qid"].values).mean().rename("y_true").reset_index()
    truth.columns = ["qid", "y_true"]
    truth["mu"] = pd.Series(mu, index=q["qid"].values).reindex(truth["qid"]).values
    truth.to_parquet(os.path.join(d, "truth_scores.parquet"), index=False)

    # ---- M1 (question level): noisy transform of truth, second model with independent noise
    t = truth.set_index("qid")["y_true"].reindex(q["qid"]).values
    m1 = pd.DataFrame({"qid": q["qid"].values,
                       "m1_cos": 0.35 + 0.30 * t + 0.015 * (q["log_qlen"].values - q["log_qlen"].mean()) + rng.normal(0, 0.08, n),
                       "part_coverage": np.clip(0.2 + 0.7 * t + rng.normal(0, 0.15, n), 0, 1),
                       "m1_cos_alt": 0.30 + 0.35 * t + rng.normal(0, 0.09, n)})
    m1.loc[np.isnan(t), ["m1_cos", "part_coverage", "m1_cos_alt"]] = np.nan
    m1.to_parquet(os.path.join(d, "m1_question.parquet"), index=False)

    # ---- gold sample (true labels on the sampled questions)
    s = _draw_gold(q, rng)
    s.to_csv(os.path.join(d, "main_sample.csv"), index=False)
    tl = pd.DataFrame({"part_id": parts["part_id"], "qid": parts["qid"], "label": names[true_lab]})
    gold = tl[tl["qid"].isin(set(s["qid"]))][["part_id", "label"]]
    gold.to_csv(os.path.join(d, "gold_labels.csv"), index=False)

    meta = {"SYNTHETIC": True, "note": "synthetic outcome; not a result", "planted": PLANTED,
            "n_questions": n, "n_parts_labelled": int(len(parts)), "gold_questions": int(len(s)), "seed": SEED}
    with open(os.path.join(d, "planted_effects.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    print(f"synthetic outcome written to {d}: {len(parts):,} part labels, {len(s)} gold questions")


if __name__ == "__main__":
    import build_frame
    ensure(build_frame.load_covariates(), force=True)
