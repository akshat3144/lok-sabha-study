"""Bias correction with the human gold subset: prediction-powered inference (PPI / PPI++) and DSL-style regression.

    python -I -X utf8 analysis/bias_correct.py --outcome synthetic [--family ols|fracit]

Ingredients
    f_i   cheap prediction for EVERY question in the population (here the M2 question score; M1 works too)
    Y_i   human gold score, observed only on the sampled questions (R_i = 1) with KNOWN inclusion probabilities pi_i
          (data/coding/main_sample.csv: pi = 1/weight; stratified design -> design-based, not SRS, estimation)

Estimator (Angelopoulos et al. 2023 PPI / PPI++ power tuning; Egami et al. 2023 DSL for the design-based form).
For any estimating equation that is LINEAR in the outcome (OLS normal equations, fractional-logit quasi-score) the
bias-corrected estimator is obtained by running the ordinary estimator on the pseudo-outcome

        Ytilde_i(lambda) = lambda * f_i + (R_i / pi_i) * (Y_i - lambda * f_i),         E[Ytilde_i | X_i, f_i] = E[Y_i | X_i]

for ANY lambda (E[R/pi] = 1).  lambda = 1 is PPI/DSL, lambda = 0 is the gold-only (Horvitz-Thompson) estimator and
lambda in [0,1] tuned to minimise the variance of the target is PPI++.  Because Ytilde(lambda) = Ytilde(0) + lambda*D with
D_i = f_i (1 - R_i/pi_i), the influence function of a target (a coefficient or linear contrast) is linear in lambda, so the
variance is a quadratic in lambda and the optimum lambda* = -Cov(IF0, IFD) / Var(IFD) has a closed form; we estimate those
(co)variances with the same two-way (MP x session) cluster-robust meat used in models.py and clip lambda* to [0,1].
Standard errors are then the cluster-robust sandwich of the fit on Ytilde(lambda*) (lambda* treated as fixed, as in PPI++).

M2 design weights (analysis plan §4.3a, paper/analysis_plan.md).  M2 predictions f exist only on the M2 probability sample s (weights w_i =
N_h/n_h, census 1), not on every population row, so the population term lambda * sum_pop x_i f_i (and the Gram matrix
sum_pop x_i x_i') is estimated with those weights, while the rectifier sum_gold (1/pi_i) x_i (Y_i - lambda f_i) is the
unchanged Horvitz-Thompson sum over the gold sample with its own main_sample inclusion probabilities.  The estimating
equation (OLS: m = identity; fractional logit: m = logistic) is
        sum_i x_i [ c_i(lambda) - w_i m(x_i'b) ] = 0,     c_i(lambda) = lambda w_i f_i + r_i (Y_i - lambda f_i),  r = R/pi
over the rows with w_i > 0 OR R_i = 1 (models.fit_ee).  With w = 1 everywhere this is exactly the estimator above
(c = Ytilde).  Gold questions not drawn into the M2 sample ('gold' role, w = 0) were M2-labelled deliberately: they need
f_i for their rectifier term, but must not enter the population term or the Gram matrix, which w_i = 0 achieves.
c(lambda) = c(0) + lambda * D with D_i = (w_i - r_i) f_i, so the lambda* formula is unchanged with IF0 = v (r Y - w mu) and
IFD = v D.  The sandwich treats each row's total score x_i (c_i - w_i mu_i) as the unit contribution (two-way clustered,
no finite-population correction; ignoring the M2 strata in the variance is typically conservative for the population
term).

Also provided: ppi_mean_srs() = textbook PPI++ mean for SRS labelled / unlabelled samples (used in the coverage simulation).

Population note: the gold sample is drawn from the M3 universe (answered questions with 1-6 parts, pilot excluded;
frame column gold_universe); all bias-corrected estimates refer to that universe, so each is reported next to the
naive estimate on the SAME rows.  Sampled-but-not-(completely)-labelled questions would break R/pi unbiasedness; their
number is reported and must be 0 for the final analysis.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import sparse  # noqa: E402
from scipy import stats  # noqa: E402

import common as C  # noqa: E402
import models as M  # noqa: E402


# ------------------------------------------------------------------------------------ textbook PPI++ mean (SRS)
def ppi_mean_srs(y_l, f_l, f_u, alpha=0.05, lam="opt"):
    """PPI++ CI for E[Y] with n labelled (Y, f) pairs and N unlabelled predictions (SRS). Returns dict."""
    y_l, f_l, f_u = map(np.asarray, (y_l, f_l, f_u))
    n, N = len(y_l), len(f_u)
    if lam == "opt":
        v = np.var(f_l, ddof=1)
        lam = 0.0 if v == 0 else float(np.clip(np.cov(y_l, f_l)[0, 1] / ((1 + n / N) * v), 0, 1))
    est = lam * f_u.mean() + (y_l - lam * f_l).mean()
    se = np.sqrt(lam ** 2 * np.var(f_u, ddof=1) / N + np.var(y_l - lam * f_l, ddof=1) / n)
    z = stats.norm.ppf(1 - alpha / 2)
    return {"estimate": est, "se": se, "ci_lo": est - z * se, "ci_hi": est + z * se, "lambda": lam}


def classical_mean(y_l, alpha=0.05):
    y_l = np.asarray(y_l)
    est, se = y_l.mean(), y_l.std(ddof=1) / np.sqrt(len(y_l))
    z = stats.norm.ppf(1 - alpha / 2)
    return {"estimate": est, "se": se, "ci_lo": est - z * se, "ci_hi": est + z * se}


# ------------------------------------------------------------------------------------ design-based regression PPI++
def _meat2(cols, groups):
    """two-way cluster 'meat' (no small-sample factor) of an n x m matrix of per-unit influence contributions."""
    n = cols.shape[0]
    Xs = sparse.csr_matrix(cols)
    return M.cluster_cov(Xs, np.ones(n), np.eye(cols.shape[1]), groups, correction=False, psd_fix=False)


def ppi_fit(d, f_col, y_col, r_col, pi_col, num, cat, targets, family="ols", cluster=("mp_id", "session_id"),
            lam="opt", ref=None, label="", w_col=None):
    """Bias-corrected regression targets. d: estimation sample (all rows need f and pi>0; gold Y on rows with r_col).
    w_col: M2 design weights of the prediction sample (None = f on every population row, w = 1); every row needs
    w > 0 or R (see module docstring). targets: {name: {coef_name: weight}} (a coefficient is {name: 1.0}).
    Returns DataFrame of naive (f as outcome), gold-only (lambda=0) and PPI (tuned or fixed lambda) estimates per target."""
    X, names = M.build_X(d, list(num), list(cat), ref)
    idx = {nm: i for i, nm in enumerate(names)}
    n = len(d)
    R = d[r_col].to_numpy(bool)
    pi = d[pi_col].to_numpy(float)
    assert np.all(pi[R] > 0), "labelled rows need inclusion probability > 0"
    w = np.ones(n) if w_col is None else d[w_col].to_numpy(float)
    assert np.all(w >= 0) and np.all((w > 0) | R), "rows need a design weight > 0 or a gold label"
    r = np.where(R, 1.0 / np.where(pi > 0, pi, 1.0), 0.0)
    f = d[f_col].to_numpy(float)
    Y = np.where(R, d[y_col].fillna(0).to_numpy(float), 0.0)
    Y0, D = r * Y, f * (w - r)                                   # c(lambda) = Y0 + lambda * D
    groups = [pd.factorize(d[c])[0] for c in cluster]
    gs = [g.max() + 1 for g in groups]

    def make_fit(c):
        beta, s, bread, mu = M.fit_ee(X, c, w, family)
        V = M.cluster_cov(X, s, bread, groups)                   # scores x_i (c_i - w_i mu_i)
        return M.Fit(names, beta, V, n, gs, family, X=X, y=c, mu=mu, w=w), bread

    def vec(t):
        v = np.zeros(len(names))
        for k, x in t.items():
            v[idx[k]] = x
        return v

    f_naive, _ = make_fit(w * f)
    f_pre, bread_pre = make_fit(Y0 + D)                          # lambda = 1 (preliminary mu for the influence functions)
    f_gold, _ = make_fit(Y0)
    rows = []
    for tname, tw in targets.items():
        wv = vec(tw)
        if lam == "opt":
            v = X @ (bread_pre @ wv)
            mu = f_pre.mu
            cols = np.column_stack([v * (Y0 - w * mu), v * D])
            m2 = _meat2(cols, groups)
            lam_t = float(np.clip(-m2[0, 1] / m2[1, 1], 0, 1)) if m2[1, 1] > 0 else 0.0
        else:
            lam_t = float(lam)
        fit_l, _ = make_fit(Y0 + lam_t * D)
        for method, fit, lv in (("naive (on predictions)", f_naive, np.nan), ("gold only (HT, lambda=0)", f_gold, 0.0),
                                ("PPI++ (lambda tuned)" if lam == "opt" else f"PPI (lambda={lam})", fit_l, lam_t)):
            row = fit.contrast(tw, tname)
            row.update({"method": method, "lambda": lv, "n": n, "n_gold": int(R.sum()), "spec": label, "family": family})
            rows.append(row)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------ pipeline on the frame
def prepare_ppi_frame(df, f_col="y_m2", y_col="y_gold", weight=None):
    """rows usable for design-based correction: analysis sample AND gold universe with a prediction;
    R = gold sampled & completely labelled. weight: M2 design-weight column -> keep rows with weight > 0 OR R (gold-only
    extras stay for their rectifier term). attrs: sampled-but-unlabelled gold, complete gold without a prediction
    (both must be 0 for the final analysis: dropping them would break R/pi unbiasedness)."""
    u = df[df["analysis_sample"] & df["gold_universe"]]
    no_f = int((u["gold_complete"] & (u["gold_pi"].fillna(0) > 0) & u[f_col].isna()).sum())
    d = u[u[f_col].notna()].copy()
    d["pi"] = d["gold_pi"].fillna(0.0)
    d["R"] = d["gold_complete"] & (d["pi"] > 0)
    d["sampled_unlabelled"] = d["gold_sampled"] & ~d["R"]
    n_unl = int(d["sampled_unlabelled"].sum())
    if weight is not None:
        d = d[(d[weight] > 0) | d["R"]]
    d.attrs.update(n_sampled_unlabelled=n_unl, n_gold_without_f=no_f)
    return d


def run_ppi_hypotheses(df, family="ols", f_col="y_m2", y_col="y_gold", lam="opt", weight="design"):
    """Bias-corrected primary estimates for H1-H4 using the SAME specifications as models.run_hypotheses, restricted
    to the gold universe. Returns a long DataFrame (method x hypothesis) with Holm over H1-H4 within each method.
    weight: 'design' = models.design_weight(df, f_col) (M2 design weights for M2 predictions), a column name, or None.
    Specs are built WITHOUT dropping weight-0 rows (gold-only extras carry the rectifier); every spec, including the
    LS17-vs-LS18 census ones (where w = 1, so this is a no-op), passes the weights to ppi_fit. The H3 contrast uses
    design-weighted group shares, as in models.run_hypotheses."""
    weight = M.design_weight(df, f_col) if weight == "design" else weight
    d = prepare_ppi_frame(df, f_col, y_col, weight)
    att = dict(d.attrs, weight=weight)
    d = d.assign(**{"_y": d[f_col]})
    specs, opp = M.hypothesis_specs(d, "_y")
    out = []
    plan = [("H1", "H1H3", {opp: 1.0}), ("H2", "H2key", {"coalition_dep": 1.0}),
            ("H3", "H1H3", M.h3_weights(specs["H1H3"]["d"], weight=weight)), ("H4", "H4key", {"opp_x_coal": 1.0})]
    for h, key, w in plan:
        sp = specs[key]
        res = ppi_fit(sp["d"], f_col, y_col, "R", "pi", sp["num"], sp["cat"], {h: w}, family, lam=lam, label=sp["label"],
                      w_col=weight)
        res.insert(0, "hypothesis", h)
        res["weights"] = weight or "none"
        out.append(res)
    res = pd.concat(out, ignore_index=True)
    res["p_holm"] = np.nan
    for m_ in res["method"].unique():
        k = res["method"] == m_
        res.loc[k, "p_holm"] = M.holm(res.loc[k, "p"].values)
    res.attrs.update(att)
    return res


def main():
    from build_frame import load_frame
    ap = argparse.ArgumentParser()
    ap.add_argument("--outcome", choices=C.OUTCOMES, default="synthetic")
    ap.add_argument("--family", choices=("ols", "fracit"), default="ols")
    ap.add_argument("--f", default="y_m2")
    a = ap.parse_args()
    df = load_frame(a.outcome)
    res = run_ppi_hypotheses(df, a.family, f_col=a.f)
    n_unl, n_nof, wt = (res.attrs.get(k) for k in ("n_sampled_unlabelled", "n_gold_without_f", "weight"))
    if a.outcome == "synthetic":      # oracle on the SAME rows (probability limit of each spec), for apples-to-apples checks
        d = prepare_ppi_frame(df, a.f, weight=wt)
        orc, _, _ = M.run_hypotheses(d, "mu", a.family, secondary=False, weight=wt)
        orc = orc[["hypothesis", "estimate"]].assign(method="ORACLE (noise-free mu, same rows)")
        res = pd.concat([res, orc], ignore_index=True)
        res["ci_covers_oracle"] = pd.Series([None] * len(res), dtype=object)
        k = res["method"].str.startswith("PPI")
        o = orc.set_index("hypothesis")["estimate"]
        res.loc[k, "ci_covers_oracle"] = [bool(r.ci_lo <= o[r.hypothesis] <= r.ci_hi) for r in res[k].itertuples()]
        k = res["method"].str.startswith("naive")
        res.loc[k, "ci_covers_oracle"] = [bool(r.ci_lo <= o[r.hypothesis] <= r.ci_hi) for r in res[k].itertuples()]
    out = C.outdir("tables", a.outcome)
    res.to_csv(os.path.join(out, f"bias_corrected_hypotheses_{a.f}_{a.family}.csv"), index=False)
    with open(os.path.join(out, f"bias_corrected_hypotheses_{a.f}_{a.family}.md"), "w", encoding="utf-8") as fh:
        fh.write(("**SYNTHETIC OUTCOME - NOT A RESULT**\n\n" if a.outcome == "synthetic" else "")
                 + "Bias-corrected (PPI++/DSL) estimates; gold universe only; two-way clustered; Holm within method"
                 + (f"; prediction term design-weighted ({wt})" if wt else "") + "\n\n")
        fh.write(C.df_to_markdown(res, floatfmt="{:.4f}"))
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(res.drop(columns=[c for c in ("term", "t", "p", "spec", "family") if c in res]).round(4).to_string(index=False))
    print("sampled but unlabelled:", n_unl, "| complete gold without a prediction:", n_nof)


if __name__ == "__main__":
    main()
