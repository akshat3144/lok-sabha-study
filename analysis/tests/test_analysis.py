"""Tests for the analysis code.

    python -I -X utf8 -m pytest analysis/tests -q          (slow tests: simulations, ~1-2 min)

Covers: two-way cluster-robust covariance vs statsmodels and vs a brute-force loop; OLS / fractional-logit estimators vs
statsmodels; Holm; PPI coverage simulations (textbook SRS mean and design-based regression); planted H1-H4 effect
recovery on a synthetic frame; table builder smoke test on real data (skipped if data absent).
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
import statsmodels.api as sm  # noqa: E402
from scipy import sparse  # noqa: E402

import bias_correct as B  # noqa: E402
import common as C  # noqa: E402
import models as M  # noqa: E402
import synthetic as S  # noqa: E402


# ------------------------------------------------------------------------------------------ helpers
def _toy(n=700, seed=1, k=3, g1=40, g2=12):
    rng = np.random.default_rng(seed)
    X = np.column_stack([np.ones(n), rng.normal(size=(n, k - 1))])
    c1, c2 = rng.integers(0, g1, n), rng.integers(0, g2, n)
    # cluster-correlated errors in both dimensions
    e = rng.normal(size=n) + rng.normal(size=g1)[c1] + 0.7 * rng.normal(size=g2)[c2]
    y = X @ np.array([1.0, 0.5, -0.3][:k]) + e
    return X, y, c1, c2


def _bruteforce_twoway(X, u, c1, c2, bread):
    """independent loop implementation of CGM two-way cluster meat with the Stata/statsmodels small-sample factors."""
    n, k = X.shape

    def meat(g):
        G = len(np.unique(g))
        m = np.zeros((k, k))
        for v in np.unique(g):
            s = (X[g == v] * u[g == v, None]).sum(axis=0)
            m += np.outer(s, s)
        return m * G / (G - 1) * (n - 1) / (n - k)

    c12 = pd.factorize(pd.Series(c1.astype(str)) + "_" + pd.Series(c2.astype(str)))[0]
    return bread @ (meat(c1) + meat(c2) - meat(c12)) @ bread


# ------------------------------------------------------------------------------------------ clustering / estimators
def test_twoway_cluster_matches_bruteforce_and_statsmodels():
    X, y, c1, c2 = _toy()
    Xs = sparse.csr_matrix(X)
    beta, u, bread, _ = M.fit_ols(Xs, y)
    V = M.cluster_cov(Xs, u, bread, [c1, c2], psd_fix=False)
    ref = _bruteforce_twoway(X, u, c1, c2, bread)
    assert np.allclose(V, ref, rtol=1e-9, atol=1e-12)
    sm_fit = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": np.column_stack([c1, c2])})
    assert np.allclose(beta, sm_fit.params, atol=1e-10)
    assert np.allclose(V, sm_fit.cov_params(), rtol=1e-6, atol=1e-10)


def test_oneway_cluster_matches_statsmodels():
    X, y, c1, _ = _toy(seed=5)
    Xs = sparse.csr_matrix(X)
    beta, u, bread, _ = M.fit_ols(Xs, y)
    V = M.cluster_cov(Xs, u, bread, [c1])
    sm_fit = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": c1})
    assert np.allclose(V, sm_fit.cov_params(), rtol=1e-6)


def test_fractional_logit_matches_statsmodels_glm():
    X, _, c1, c2 = _toy(seed=3)
    rng = np.random.default_rng(3)
    mu = 1 / (1 + np.exp(-(X @ np.array([0.2, 0.4, -0.3]))))
    y = np.clip(mu + rng.normal(0, 0.15, len(mu)), 0.01, 0.99)
    Xs = sparse.csr_matrix(X)
    beta, u, bread, _ = M.fit_fracit(Xs, y)
    glm = sm.GLM(y, X, family=sm.families.Binomial()).fit(cov_type="cluster", cov_kwds={"groups": c1})
    assert np.allclose(beta, glm.params, atol=1e-6)
    V = M.cluster_cov(Xs, u, bread, [c1])
    assert np.allclose(V, glm.cov_params(), rtol=1e-3)


def test_build_X_drops_reference_and_empty_levels():
    d = pd.DataFrame({"a": ["x", "y", "y", "z", "y"], "n": [1.0, 2, 3, 4, 5]})
    X, names = M.build_X(d, ["n"], ["a"])
    assert names == ["const", "n", "a[x]", "a[z]"]            # 'y' (most frequent) is the reference
    assert X.shape == (5, 4)


def test_fit_matches_statsmodels_with_fe_dummies():
    rng = np.random.default_rng(0)
    n = 1500
    d = pd.DataFrame({"x": rng.normal(size=n), "g": rng.choice(list("abcde"), n), "mp": rng.integers(0, 80, n),
                      "ss": rng.integers(0, 9, n)})
    d["y"] = 0.3 * d["x"] + d["g"].map(dict(zip("abcde", [0, .2, -.1, .3, 0]))) + rng.normal(size=n)
    f = M.fit_model(d, "y", ["x"], ["g"], "ols", ("mp", "ss"))
    Xr = pd.concat([d[["x"]], pd.get_dummies(d["g"], drop_first=True, dtype=float)], axis=1)
    ref = sm.OLS(d["y"], sm.add_constant(Xr)).fit()
    assert np.isclose(f.coef("x")["estimate"], ref.params["x"], atol=1e-10)


def test_holm():
    p = np.array([0.01, 0.04, 0.03, 0.005])
    assert np.allclose(M.holm(p), [0.03, 0.06, 0.06, 0.02])
    # statsmodels reference
    from statsmodels.stats.multitest import multipletests
    rng = np.random.default_rng(2)
    q = rng.uniform(size=7) ** 2
    assert np.allclose(M.holm(q), multipletests(q, method="holm")[1])


# ------------------------------------------------------------------------------------------ PPI
def test_ppi_mean_coverage_srs():
    """PPI++ mean CI covers the truth ~95% of the time when predictions are biased and noisy."""
    rng = np.random.default_rng(10)
    N, n, reps = 4000, 250, 1000
    hits = hits_classic = 0
    widths = []
    cw = []
    truth = 0.55
    for _ in range(reps):
        y = rng.normal(truth, 0.25, N + n)                                # superpopulation with E[Y] = truth
        f = 0.8 * y + 0.12 + rng.normal(0, 0.08, N + n)                   # biased + noisy proxy
        lab = rng.permutation(N + n)[:n]
        unl = np.setdiff1d(np.arange(N + n), lab)
        r = B.ppi_mean_srs(y[lab], f[lab], f[unl])
        hits += r["ci_lo"] <= truth <= r["ci_hi"]
        widths.append(r["ci_hi"] - r["ci_lo"])
        c = B.classical_mean(y[lab])
        hits_classic += c["ci_lo"] <= truth <= c["ci_hi"]
        cw.append(c["ci_hi"] - c["ci_lo"])
    cov = hits / reps
    print(f"[PPI mean] coverage {cov:.3f} (classical {hits_classic / reps:.3f}); mean CI width {np.mean(widths):.4f} vs {np.mean(cw):.4f}")
    assert 0.93 <= cov <= 0.97, cov
    assert 0.92 <= hits_classic / reps <= 0.98
    assert np.mean(widths) < 0.6 * np.mean(cw)      # the proxy buys a real efficiency gain


def test_ppi_regression_coverage_design_based():
    """Design-based PPI++ regression (stratified Poisson sampling, known pi): coverage of the finite-population OLS
    coefficients near nominal; the naive regression on predictions is biased and does not cover."""
    rng = np.random.default_rng(11)
    N = 6000
    opp = rng.binomial(1, 0.4, N)
    z = rng.normal(size=N)
    y = 0.5 - 0.06 * opp + 0.05 * z + rng.normal(0, 0.2, N)
    f = 0.85 * y + 0.08 - 0.05 * (1 - opp) + rng.normal(0, 0.07, N)    # differential bias by opp
    pi = np.where(opp == 1, 0.07, 0.04)
    d0 = pd.DataFrame({"opp": opp, "z": z, "y": y, "f": f, "pi": pi, "uid": np.arange(N)})
    Xp = np.column_stack([np.ones(N), opp, z])
    truth = np.linalg.lstsq(Xp, y, rcond=None)[0][1]
    reps, hit_p, hit_n, hit_g = 400, 0, 0, 0
    for _ in range(reps):
        R = rng.random(N) < pi
        d = d0.assign(R=R, y_gold=np.where(R, d0["y"], np.nan))
        out = B.ppi_fit(d, "f", "y_gold", "R", "pi", ["opp", "z"], [], {"opp": {"opp": 1.0}}, "ols", cluster=("uid", "uid"))
        p = out[out["method"].str.startswith("PPI")].iloc[0]
        nv = out[out["method"].str.startswith("naive")].iloc[0]
        g = out[out["method"].str.startswith("gold")].iloc[0]
        hit_p += p["ci_lo"] <= truth <= p["ci_hi"]
        hit_n += nv["ci_lo"] <= truth <= nv["ci_hi"]
        hit_g += g["ci_lo"] <= truth <= g["ci_hi"]
    print(f"[PPI regression] coverage PPI++ {hit_p / reps:.3f}, gold-only {hit_g / reps:.3f}, naive {hit_n / reps:.3f}")
    assert 0.92 <= hit_p / reps <= 0.995, hit_p / reps
    assert hit_g / reps >= 0.90
    assert hit_n / reps < 0.5                                         # naive analysis of f is badly biased here


def test_ppi_fracit_pseudo_outcome_unbiased():
    """fractional-logit estimating equation is linear in y -> same pseudo-outcome device works; check bias near 0."""
    rng = np.random.default_rng(12)
    N = 5000
    opp = rng.binomial(1, 0.5, N)
    mu = 1 / (1 + np.exp(-(0.2 - 0.3 * opp)))
    y = np.clip(mu + rng.normal(0, 0.15, N), 0.001, 0.999)
    f = np.clip(0.9 * y + 0.05 + rng.normal(0, 0.05, N), 0, 1)
    d0 = pd.DataFrame({"opp": opp, "y": y, "f": f, "pi": 0.08, "uid": np.arange(N)})
    X, _ = M.build_X(d0, ["opp"], [])
    truth = M.fit_fracit(X, y)[0][1]
    ests = []
    for _ in range(150):
        R = rng.random(N) < 0.08
        d = d0.assign(R=R, y_gold=np.where(R, y, np.nan))
        out = B.ppi_fit(d, "f", "y_gold", "R", "pi", ["opp"], [], {"opp": {"opp": 1.0}}, "fracit", cluster=("uid", "uid"))
        ests.append(out[out["method"].str.startswith("PPI")].iloc[0]["estimate"])
    se = np.std(ests) / np.sqrt(len(ests))
    assert abs(np.mean(ests) - truth) < 4 * se + 0.01


# ------------------------------------------------------------------------------------------ synthetic recovery
def _fake_frame(n=60000, seed=0):
    rng = np.random.default_rng(seed)
    ls = rng.choice([13, 14, 15, 16, 17, 18], n, p=[.19, .17, .2, .2, .15, .09])
    sess = ls * 100 + rng.integers(1, 8, n)
    d = pd.DataFrame({"ls": ls, "session_id": [f"{s // 100}-{s % 100:02d}" for s in sess]})
    d["coalition_dep"] = (~np.isin(ls, [16, 17])).astype(int)
    bloc = rng.choice(["government", "opposition", "outside_support", "mixed"], n, p=[.38, .42, .06, .14])
    d["opp"] = (bloc == "opposition").astype(int)
    d["mixed"] = (bloc == "mixed").astype(int)
    d["outside"] = (bloc == "outside_support").astype(int)
    d["ministry_group"] = rng.choice(list("ABCDEFG") + [C.SECURITY_GROUP], n, p=[.15] * 6 + [.04, .06])
    d["ministry_group"] = d["ministry_group"].replace({"G": "G"})
    d["security"] = (d["ministry_group"] == C.SECURITY_GROUP).astype(int)
    d["topic"] = rng.integers(0, 50, n)
    d["mp_id"] = [f"mp{i}" for i in rng.integers(0, 1500, n)]
    d["starred"] = rng.binomial(1, 0.08, n)
    d["n_parts"] = np.clip(rng.poisson(3, n) + 1, 1, 8)
    d["log_qlen"] = rng.normal(6, 0.5, n)
    d["coasker"] = rng.poisson(0.6, n)
    d["analysis_sample"] = True
    mu = S.simulate_mu(d, rng)
    d["mu"] = mu
    d["y_true"] = S.draw_part_scores(mu, d["n_parts"].values, rng)
    d["year"] = 2000 + rng.uniform(0, 25, n)
    d["any_split_party_window"] = False
    return d


def test_h1_h4_planted_effects_recovered():
    d = _fake_frame()
    prim, sec, fits = M.run_hypotheses(d, "y_true")
    orc, _, _ = M.run_hypotheses(d, "mu", secondary=False)
    P = S.PLANTED
    p = prim.set_index("hypothesis")
    o = orc.set_index("hypothesis")
    for h in ("H1", "H2", "H3", "H4"):
        assert p.loc[h, "ci_lo"] <= o.loc[h, "estimate"] <= p.loc[h, "ci_hi"], (h, p.loc[h].to_dict(), o.loc[h, "estimate"])
        assert p.loc[h, "direction_ok"]
    # parametric planted values where the specification is correct
    assert p.loc["H3", "ci_lo"] <= P["security"] <= p.loc["H3", "ci_hi"]
    assert p.loc["H4", "ci_lo"] <= P["opp_x_coalition"] <= p.loc["H4", "ci_hi"]
    assert abs(o.loc["H3", "estimate"] - P["security"]) < 0.01


def test_twoway_ci_coverage_on_synthetic_replications():
    """Coverage of the planted H3 (security) and H4 (interaction) values across independent synthetic draws."""
    reps = 30
    hit = {"H3": 0, "H4": 0}
    P = S.PLANTED
    for r in range(reps):
        d = _fake_frame(n=12000, seed=100 + r)
        prim, _, _ = M.run_hypotheses(d, "y_true", secondary=False)
        p = prim.set_index("hypothesis")
        hit["H3"] += p.loc["H3", "ci_lo"] <= P["security"] <= p.loc["H3", "ci_hi"]
        hit["H4"] += p.loc["H4", "ci_lo"] <= P["opp_x_coalition"] <= p.loc["H4", "ci_hi"]
    print(f"[two-way CI coverage, {reps} synthetic draws] H3 {hit['H3'] / reps:.2f}, H4 {hit['H4'] / reps:.2f}")
    assert hit["H3"] / reps >= 0.83, hit
    assert hit["H4"] / reps >= 0.83, hit


# ------------------------------------------------------------------------------------------ M2 design weights (§4.3a)
def test_wls_integer_weights_equal_ols_on_duplicated_rows():
    X, y, c1, c2 = _toy(seed=7)
    w = np.random.default_rng(7).integers(1, 5, len(y)).astype(float)
    rep = np.repeat(np.arange(len(y)), w.astype(int))
    Xs, Xd = sparse.csr_matrix(X), sparse.csr_matrix(X[rep])
    b_w, u_w, br_w, _ = M.fit_ols(Xs, y, w)
    b_d, u_d, br_d, _ = M.fit_ols(Xd, y[rep])
    assert np.allclose(b_w, b_d, atol=1e-10)
    assert np.allclose(b_w, sm.WLS(y, X, weights=w).fit().params, atol=1e-10)
    # same clusters -> same cluster score sums: sandwiches agree up to the small-sample factor (N differs)
    V_w = M.cluster_cov(Xs, u_w, br_w, [c1, c2], correction=False, psd_fix=False, w=w)
    V_d = M.cluster_cov(Xd, u_d, br_d, [c1[rep], c2[rep]], correction=False, psd_fix=False)
    assert np.allclose(V_w, V_d, rtol=1e-8, atol=1e-14)
    yf = 1 / (1 + np.exp(-(X @ np.array([0.2, 0.4, -0.3])))) * 0.9 + 0.05
    assert np.allclose(M.fit_fracit(Xs, yf, w)[0], M.fit_fracit(Xd, yf[rep])[0], atol=1e-8)


def test_weighted_cluster_se_reduce_to_unweighted_when_w_is_one():
    X, y, c1, c2 = _toy(seed=8)
    Xs, one = sparse.csr_matrix(X), np.ones(len(y))
    b0, u0, br0, _ = M.fit_ols(Xs, y)
    b1, u1, br1, _ = M.fit_ols(Xs, y, one)
    assert np.allclose(b0, b1, atol=1e-12)
    assert np.allclose(M.cluster_cov(Xs, u0, br0, [c1, c2]), M.cluster_cov(Xs, u1, br1, [c1, c2], w=one), atol=1e-14)
    yf = np.clip(0.5 + 0.1 * y, 0.01, 0.99)
    b0, u0, br0, _ = M.fit_fracit(Xs, yf)
    b1, u1, br1, _ = M.fit_fracit(Xs, yf, one)
    assert np.allclose(b0, b1, atol=1e-10)
    assert np.allclose(M.cluster_cov(Xs, u0, br0, [c1]), M.cluster_cov(Xs, u1, br1, [c1], w=one), atol=1e-14)
    d = pd.DataFrame({"x": X[:, 1], "g": c2 % 4, "mp": c1, "ss": c2, "y": y, "w": 1.0})
    f0 = M.fit_model(d, "y", ["x"], ["g"], cluster=("mp", "ss"))
    f1 = M.fit_model(d, "y", ["x"], ["g"], cluster=("mp", "ss"), weight="w")
    assert np.allclose(f0.beta, f1.beta) and np.allclose(f0.V, f1.V) and f0.g_sizes == f1.g_sizes


def test_weight_zero_rows_are_excluded():
    rng = np.random.default_rng(9)
    n = 3000
    d = pd.DataFrame({"x": rng.normal(size=n), "g": rng.choice(list("abc"), n), "mp": rng.integers(0, 200, n),
                      "ss": rng.integers(0, 12, n)})
    d["y"] = 0.3 * d["x"] + rng.normal(size=n)
    d["w"] = rng.integers(1, 7, n).astype(float)
    z = rng.random(n) < 0.15
    d.loc[z, "w"] = np.where(rng.random(z.sum()) < 0.5, 0.0, np.nan)    # gold-only extras / outside the design
    d.loc[z, "y"] += 5.0                                                  # would move beta if they leaked in
    d.loc[z, "mp"] = 1000 + np.arange(z.sum())                           # clusters that exist only in excluded rows
    fw = M.fit_model(d, "y", ["x"], ["g"], cluster=("mp", "ss"), weight="w")
    fs = M.fit_model(d[~z], "y", ["x"], ["g"], cluster=("mp", "ss"), weight="w")
    assert fw.n == fs.n == (~z).sum() and fw.g_sizes == fs.g_sizes
    assert np.allclose(fw.beta, fs.beta) and np.allclose(fw.V, fs.V)
    d["analysis_sample"] = True
    assert len(M.estimation_sample(d, "y", weight="w")) == (~z).sum()


def _weighted_fake(n=12000, seed=3):
    """fake frame + M2-like design weights: LS17/18 census (1), LS13-16 integer weights, some gold-only extras (0)."""
    d = _fake_frame(n, seed)
    rng = np.random.default_rng(seed + 1)
    d["m2_weight"] = np.where(d["ls"] >= 17, 1.0, rng.integers(1, 7, n).astype(float))
    d.loc[(d["ls"] < 17) & (rng.random(n) < 0.03), "m2_weight"] = 0.0
    return d


def test_h2_h4_key_spec_ignores_weights():
    d = _weighted_fake()
    pw, _, fw = M.run_hypotheses(d, "y_true", weight="m2_weight")
    pu, _, _ = M.run_hypotheses(d[d["m2_weight"] > 0], "y_true")
    pw, pu = pw.set_index("hypothesis"), pu.set_index("hypothesis")
    for h in ("H2", "H4"):
        assert np.isclose(pw.loc[h, "estimate"], pu.loc[h, "estimate"], rtol=0, atol=1e-12)
        assert np.isclose(pw.loc[h, "se"], pu.loc[h, "se"], rtol=0, atol=1e-12)
        assert pw.loc[h, "weights"] == "none"
    assert fw["H2key"].w is None and fw["H4key"].w is None and fw["H1H3"].w is not None
    assert pw.loc["H1", "weights"] == "m2_weight"
    assert not np.isclose(pw.loc["H1", "estimate"], pu.loc["H1", "estimate"], rtol=0, atol=1e-9)   # pooled: weighted
    bad = d.assign(m2_weight=np.where(d["ls"] == 18, 2.0, d["m2_weight"]))
    with pytest.raises(AssertionError):
        M.run_hypotheses(bad, "y_true", weight="m2_weight", secondary=False)


def test_weighted_hypotheses_equal_row_duplicated_frame():
    """integer design weights == duplicating rows, for every primary estimate (incl. the weighted H3 group shares)."""
    d = _weighted_fake(n=9000, seed=5)
    pw, _, _ = M.run_hypotheses(d, "y_true", weight="m2_weight", secondary=False)
    dup = d.loc[d.index.repeat(d["m2_weight"].astype(int))].reset_index(drop=True)
    pd_, _, _ = M.run_hypotheses(dup, "y_true", secondary=False)
    assert np.allclose(pw["estimate"].values, pd_["estimate"].values, rtol=0, atol=1e-10)
    s = d[d["m2_weight"] > 0]
    ref = s["ministry_group"].value_counts().index[0]
    a, b = M.h3_weights(s, ref, weight="m2_weight"), M.h3_weights(dup, ref)
    assert a.keys() == b.keys() and all(np.isclose(a[k], b[k]) for k in a)


def test_design_weight_choice():
    d = pd.DataFrame({"y_m2": [0.5], "y_m1": [0.4], "m2_weight": [6.0], "outcome_source": ["real"]})
    assert M.design_weight(d, "y_m2") == "m2_weight" and M.design_weight(d, "y_full") == "m2_weight"
    assert M.design_weight(d, "y_m1") is None and M.design_weight(d, "mu") is None
    with pytest.raises(ValueError):
        M.design_weight(d.drop(columns="m2_weight"), "y_m2")
    assert M.design_weight(d.drop(columns="m2_weight").assign(outcome_source="synthetic"), "y_m2") is None


def test_attach_m2_design(tmp_path):
    import build_frame as BF
    q = pd.DataFrame({"qid": list("abcde"), "answer_missing": False, "y_m2": [.5, np.nan, 1, .2, np.nan]})
    p = tmp_path / "m2_sample.csv"
    pd.DataFrame({"qid": ["a", "b", "c"], "ls": 17, "role": ["census", "random", "gold"],
                  "m2_weight": [1.0, 6.0, 0.0]}).to_csv(p, index=False)
    out, note = BF.attach_m2_design(q, "real", path=str(p))
    assert list(out["m2_role"]) == ["census", "random", "gold", "out", "out"]
    assert list(out["m2_weight"]) == [1.0, 6.0, 0.0, 0.0, 0.0]
    assert list(out["in_m2_sample"]) == [True, True, True, False, False]
    assert "weight 0: 2" in note and "unlabelled: 1" in note
    syn, _ = BF.attach_m2_design(q, "synthetic", path=str(tmp_path / "absent.csv"))
    assert (syn["m2_weight"] == 1).all() and syn["in_m2_sample"].all()


def test_ppi_unchanged_when_all_weights_one():
    rng = np.random.default_rng(14)
    N = 3000
    opp, z = rng.binomial(1, 0.4, N), rng.normal(size=N)
    y = 0.5 - 0.06 * opp + 0.05 * z + rng.normal(0, 0.2, N)
    d = pd.DataFrame({"opp": opp, "z": z, "f": 0.85 * y + 0.08 + rng.normal(0, 0.07, N), "pi": 0.06, "uid": np.arange(N),
                      "one": 1.0})
    d["R"] = rng.random(N) < 0.06
    d["y_gold"] = np.where(d["R"], y, np.nan)
    for fam in ("ols", "fracit"):
        a = B.ppi_fit(d, "f", "y_gold", "R", "pi", ["opp", "z"], [], {"opp": {"opp": 1.0}}, fam, cluster=("uid", "uid"))
        b = B.ppi_fit(d, "f", "y_gold", "R", "pi", ["opp", "z"], [], {"opp": {"opp": 1.0}}, fam, cluster=("uid", "uid"),
                      w_col="one")
        assert np.allclose(a[["estimate", "se", "lambda"]].fillna(-9).values, b[["estimate", "se", "lambda"]].fillna(-9).values)


def test_ppi_with_m2_design_weights_and_gold_only_extras():
    """M2 predictions only on a stratified sample (census stratum w = 1, other stratum 1/4 SRS, w = 4); gold drawn
    independently with known pi; gold questions not in the M2 sample enter with w = 0 (rectifier only). PPI++ must be
    unbiased for the finite-population coefficient and cover it; the naive weighted analysis of f must not."""
    rng = np.random.default_rng(15)
    N = 6000
    opp, z = rng.binomial(1, 0.4, N), rng.normal(size=N)
    y = 0.5 - 0.06 * opp + 0.05 * z + rng.normal(0, 0.2, N)
    f = 0.85 * y + 0.08 - 0.05 * (1 - opp) + rng.normal(0, 0.07, N)
    pi = np.where(opp == 1, 0.07, 0.04)
    census = rng.random(N) < 0.4
    d0 = pd.DataFrame({"opp": opp, "z": z, "f": f, "pi": pi, "uid": np.arange(N)})
    truth = np.linalg.lstsq(np.column_stack([np.ones(N), opp, z]), y, rcond=None)[0][1]
    s0 = np.where(~census)[0]
    n0 = len(s0) // 4
    reps, est, hit_p, hit_n = 300, [], 0, 0
    for _ in range(reps):
        w = census.astype(float)
        w[rng.choice(s0, n0, replace=False)] = len(s0) / n0
        R = rng.random(N) < pi
        keep = (w > 0) | R
        d = d0.assign(R=R, w=w, y_gold=np.where(R, y, np.nan))[keep]
        out = B.ppi_fit(d, "f", "y_gold", "R", "pi", ["opp", "z"], [], {"opp": {"opp": 1.0}}, "ols",
                        cluster=("uid", "uid"), w_col="w")
        p = out[out["method"].str.startswith("PPI")].iloc[0]
        nv = out[out["method"].str.startswith("naive")].iloc[0]
        est.append(p["estimate"])
        hit_p += p["ci_lo"] <= truth <= p["ci_hi"]
        hit_n += nv["ci_lo"] <= truth <= nv["ci_hi"]
    print(f"[PPI + M2 design] mean bias {np.mean(est) - truth:+.4f}, coverage PPI++ {hit_p / reps:.3f}, naive {hit_n / reps:.3f}")
    assert abs(np.mean(est) - truth) < 4 * np.std(est) / np.sqrt(reps)
    assert 0.92 <= hit_p / reps <= 1.0
    assert hit_n / reps < 0.5


# ------------------------------------------------------------------------------------------ real-data smoke test
@pytest.mark.skipif(not os.path.exists(C.QUESTIONS), reason="real data not present")
def test_descriptive_tables_real_data():
    import tables as T
    q = T.load_questions()
    t = T.by_ls(q, None, T.mp_counts())
    assert list(t["Lok Sabha"][:6]) == [f"{i}th" for i in range(13, 19)]
    assert t.loc[t["Lok Sabha"] == "Total", "Questions"].iloc[0] == len(q)
    assert t["Questions"][:6].sum() == len(q)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
