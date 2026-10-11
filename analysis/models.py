"""Main specifications for H1-H4: OLS / fractional logit, two-way cluster-robust SEs (MP x session), Holm.

    python -I -X utf8 analysis/models.py --outcome synthetic [--y y_m2] [--family ols|fracit]

Estimators are implemented on scipy.sparse design matrices (FE dummies stay sparse) and validated against
statsmodels (OLS, GLM-Binomial, two-way cluster covariance) in analysis/tests/test_analysis.py.

Specifications (all with controls: n_parts, log question length, co-askers, request-type shares if available,
topic FE (K=50 TF-IDF/KMeans clusters), and ministry-group indicators):
    H1/H3  all Lok Sabhas, SESSION FE (absorbs time shocks).  H1 = coef(opp) [vs government askers];
           H3 = security minus question-weighted mean of the other ministry groups (linear contrast).
    H2     key contrast: LS17 vs LS18 only, no session FE (they would absorb the contrast); coef(coalition_dep).
           Secondary H2: all LS, coalition-dependent vs single-party-majority periods.
    H4     opp x coalition_dep interaction (key contrast sample; all-LS and with-session-FE variants reported too).
Expected signs: H1 < 0, H2 > 0, H3 < 0, H4 > 0 (coalition_dep coding: an opposition gap that is LARGER under single-party
majority <=> positive interaction on coalition_dep).
Inference: two-way clustered by first-asker MP and session (Cameron-Gelbach-Miller), t(min(G_mp, G_sess) - 1)
reference distribution; two-sided p-values with Holm over H1-H4 (also one-sided directional p-values).

Design weights (analysis plan §4.3a, paper/analysis_plan.md): M2 labels exist only for a probability sample (census of LS17/LS18, stratified
1/6 sample of LS13-16, weight N_h/n_h; gold-only extras weight 0). For M2 outcomes (design_weight) the pooled all-LS
models (H1/H3, H2all, H4all*) are WLS / weighted quasi-likelihood on m2_weight > 0 rows:
    beta = (X'WX)^-1 X'Wy;  V = B M B with B = (X'WX)^-1, M = two-way cluster meat of scores w_i x_i u_i
(same small-sample factors and t(min G - 1) inference). The LS17-vs-LS18 key models are a census: unweighted.
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

BLOC = ("opp", "mixed", "outside")
CORE_CTRL = ("n_parts", "log_qlen", "coasker")
RT_CTRL = ("rt_reason", "rt_action", "rt_opinion")      # INFORMATION share is the omitted category
M2_OUTCOMES = ("y_m2", "y_full")                         # observed only on the M2 probability sample (§4.3a)
HYPOTHESES = {"H1": ("opposition vs government askers", -1), "H2": ("coalition-dependent vs majority (LS17 vs LS18)", +1),
              "H3": ("security vs other ministries", -1), "H4": ("opposition x coalition-dependent", +1)}


# ------------------------------------------------------------------------------------ design
def build_X(df, num, cat, ref=None):
    """Sparse design: intercept, numeric columns, dummy blocks for each categorical (reference level dropped:
    the most frequent one unless ref[c] given). Empty / collinear-empty columns are dropped."""
    n = len(df)
    ref = ref or {}
    blocks = [sparse.csr_matrix(np.ones((n, 1)))]
    names = ["const"]
    for c in num:
        blocks.append(sparse.csr_matrix(df[c].to_numpy(dtype=float).reshape(-1, 1)))
        names.append(c)
    for c in cat:
        s = df[c]
        cs = pd.Categorical(s)
        levels = list(cs.categories)
        r = ref.get(c, s.value_counts().index[0])
        keep = [i for i, lv in enumerate(levels) if lv != r]
        remap = -np.ones(len(levels), dtype=int)
        remap[keep] = np.arange(len(keep))
        codes = cs.codes
        col = np.where(codes >= 0, remap[np.where(codes >= 0, codes, 0)], -1)
        rows = np.where(col >= 0)[0]
        blocks.append(sparse.csr_matrix((np.ones(len(rows)), (rows, col[rows])), shape=(n, len(keep))))
        names += [f"{c}[{levels[i]}]" for i in keep]
    X = sparse.hstack(blocks, format="csr")
    nz = np.asarray(abs(X).sum(axis=0)).ravel() > 0
    if not nz.all():
        X = X[:, np.where(nz)[0]]
        names = [nm for nm, k in zip(names, nz) if k]
    return X, names


# ------------------------------------------------------------------------------------ estimators
def _solve(A, b):
    return np.linalg.lstsq(A, b, rcond=1e-12)[0]


def _logistic(eta):
    return 1 / (1 + np.exp(-np.clip(eta, -30, 30)))


def fit_ee(X, c, w, family="ols", maxit=100, tol=1e-9):
    """Solve the estimating equation  sum_i x_i (c_i - w_i m(x_i'b)) = 0,  m = identity (OLS) or logistic (fractional
    logit; IRLS = Newton for the canonical link).  With c = w*y this is WLS / weighted quasi-likelihood; a row may have
    w_i = 0 and c_i != 0 (used by bias_correct for gold-only rows).  Returns beta, score residual s = c - w*mu (the
    per-row score is x_i s_i), bread = (X' diag(w m') X)^+, mu."""
    if family == "ols":
        A = (X.T @ X.multiply(w[:, None])).toarray()
        beta = _solve(A, X.T @ c)
        mu = X @ beta
        return beta, c - w * mu, np.linalg.pinv(A, hermitian=True), mu
    ybar = np.clip(c.sum() / w.sum(), 1e-3, 1 - 1e-3)
    beta = np.zeros(X.shape[1])
    beta[0] = np.log(ybar / (1 - ybar))
    for _ in range(maxit):
        eta = X @ beta
        mu = _logistic(eta)
        W = w * np.clip(mu * (1 - mu), 1e-9, None)
        A = (X.T @ X.multiply(W[:, None])).toarray()
        new = _solve(A, X.T @ (W * eta + c - w * mu))        # = X'W z, working response z = eta + (y - mu)/v
        if np.max(np.abs(new - beta)) < tol:
            beta = new
            break
        beta = new
    mu = _logistic(X @ beta)
    W = w * np.clip(mu * (1 - mu), 1e-9, None)
    A = (X.T @ X.multiply(W[:, None])).toarray()
    return beta, c - w * mu, np.linalg.pinv(A, hermitian=True), mu


def _w(w, n):
    return np.ones(n) if w is None else np.asarray(w, dtype=float)


def fit_ols(X, y, w=None):
    """(W)LS. Returns beta, residual u = y - Xb (unweighted; cluster_cov applies w), bread (X'WX)^+, fitted values."""
    w = _w(w, X.shape[0])
    beta, _, bread, mu = fit_ee(X, w * y, w, "ols")
    return beta, y - mu, bread, mu


def fit_fracit(X, y, w=None, maxit=100, tol=1e-9):
    """Fractional logit (Papke-Wooldridge): (weighted) Bernoulli quasi-likelihood with logit link, IRLS."""
    w = _w(w, X.shape[0])
    beta, _, bread, mu = fit_ee(X, w * y, w, "fracit", maxit, tol)
    return beta, y - mu, bread, mu


def _group_codes(*cols):
    s = cols[0].astype(str)
    for c in cols[1:]:
        s = s + "||" + c.astype(str)
    return pd.factorize(s)[0]


def cluster_meat(S, g):
    """sum over clusters of (sum of scores in cluster)(...)' for a sparse n x K score matrix S."""
    n = S.shape[0]
    G = sparse.csr_matrix((np.ones(n), (np.arange(n), g)), shape=(n, g.max() + 1))
    M = (G.T @ S).tocsr()
    return (M.T @ M).toarray()


def cluster_cov(X, u, bread, groups, correction=True, psd_fix=True, w=None):
    """Cluster-robust covariance sandwich: bread * meat * bread, meat from scores w_i * x_i * u_i (w = None: 1).
    For WLS / weighted quasi-likelihood the bread is (X'WX)^+ (as returned by fit_ols / fit_fracit with w).
    groups: list of 1 or 2 arrays (one-way / Cameron-Gelbach-Miller two-way: V = V_1 + V_2 - V_12).
    Small-sample factor per term: G/(G-1) * (N-1)/(N-K) (Stata/statsmodels convention; N = rows passed)."""
    n, k = X.shape
    S = X.multiply((u if w is None else w * u)[:, None]).tocsr()

    def term(g):
        G = g.max() + 1
        c = G / (G - 1) * (n - 1) / (n - k) if correction else 1.0
        return c * cluster_meat(S, g)

    if len(groups) == 1:
        meat = term(groups[0])
    else:
        g1, g2 = groups
        meat = term(g1) + term(g2) - term(pd.factorize(pd.Series(g1.astype(np.int64) * (g2.max() + 1) + g2))[0])
    V = bread @ meat @ bread
    V = (V + V.T) / 2
    if psd_fix:
        w, Q = np.linalg.eigh(V)
        if w.min() < 0:
            V = (Q * np.clip(w, 0, None)) @ Q.T
    return V


# ------------------------------------------------------------------------------------ fit object
class Fit:
    def __init__(self, names, beta, V, n, g_sizes, family, X=None, y=None, mu=None, label="", w=None):
        self.names, self.beta, self.V, self.n, self.g_sizes, self.family = names, beta, V, n, g_sizes, family
        self.X, self.y, self.mu, self.label, self.w = X, y, mu, label, w
        self.idx = {nm: i for i, nm in enumerate(names)}
        self.df_t = max(min(g_sizes) - 1, 1)
        self.se = np.sqrt(np.clip(np.diag(V), 0, None))

    def _row(self, name, est, se):
        t = est / se if se > 0 else np.nan
        p = 2 * stats.t.sf(abs(t), self.df_t) if se > 0 else np.nan
        q = stats.t.ppf(0.975, self.df_t)
        return {"term": name, "estimate": est, "se": se, "ci_lo": est - q * se, "ci_hi": est + q * se, "t": t, "p": p}

    def coef(self, name):
        i = self.idx[name]
        return self._row(name, self.beta[i], self.se[i])

    def contrast(self, weights, name="contrast"):
        """linear combination sum_j w_j beta_j; weights: dict term -> weight."""
        w = np.zeros(len(self.beta))
        for k, v in weights.items():
            w[self.idx[k]] = v
        return self._row(name, float(w @ self.beta), float(np.sqrt(max(w @ self.V @ w, 0))))

    def ame(self, name):
        """Average marginal effect of a binary regressor (fractional logit) with delta-method SE (design-weighted
        average when the fit is weighted)."""
        if self.family == "ols":
            return self.coef(name)
        j = self.idx[name]
        xj = np.asarray(self.X[:, j].todense()).ravel()
        eta0 = self.X @ self.beta - xj * self.beta[j]
        eta1 = eta0 + self.beta[j]
        m0, m1 = 1 / (1 + np.exp(-eta0)), 1 / (1 + np.exp(-eta1))
        a0, a1 = m0 * (1 - m0), m1 * (1 - m1)
        p = _w(self.w, len(xj))
        p = p / p.sum()
        g = self.X.T @ (p * (a1 - a0))
        g[j] += p @ (a1 * (1 - xj) + a0 * xj)
        est = float(p @ (m1 - m0))
        return self._row(name + " (AME)", est, float(np.sqrt(max(g @ self.V @ g, 0))))

    def table(self, terms=None):
        terms = terms or self.names
        return pd.DataFrame([self.coef(t) for t in terms])


def fit_model(df, y, num, cat, family="ols", cluster=("mp_id", "session_id"), ref=None, label="", weight=None):
    """df must already be restricted to the estimation sample (no NaN in y/num/cat/cluster).
    weight: column of design weights (WLS / weighted quasi-likelihood); rows with weight <= 0 or NaN are dropped
    before the design is built, so they count neither in n nor in the clusters."""
    if weight is not None:
        df = df[df[weight] > 0]
    X, names = build_X(df, list(num), list(cat), ref)
    yv = df[y].to_numpy(dtype=float)
    wv = None if weight is None else df[weight].to_numpy(dtype=float)
    beta, u, bread, mu = fit_ols(X, yv, wv) if family == "ols" else fit_fracit(X, yv, wv)
    groups = [pd.factorize(df[c])[0] for c in cluster]
    V = cluster_cov(X, u, bread, groups, w=wv)
    return Fit(names, beta, V, len(df), [g.max() + 1 for g in groups], family, X=X, y=yv, mu=mu, label=label, w=wv)


# ------------------------------------------------------------------------------------ multiple testing
def holm(p):
    """Holm-Bonferroni step-down adjusted p-values (monotone, capped at 1)."""
    p = np.asarray(p, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i])
        adj[i] = min(run, 1.0)
    return adj


# ------------------------------------------------------------------------------------ hypotheses
def design_weight(df, y):
    """Weight column for outcome y (analysis plan §4.3a, paper/analysis_plan.md): M2-derived outcomes are observed only on the M2 probability
    sample, so they use the design weights m2_weight; every other outcome (M1, gold, synthetic oracle) is unweighted.
    On the synthetic frame m2_weight is 1 everywhere unless a synthetic m2_sample.csv exists."""
    if y not in M2_OUTCOMES:
        return None
    if "m2_weight" not in df:
        if "outcome_source" in df and (df["outcome_source"] == "real").any():
            raise ValueError("real frame has no m2_weight column: rebuild it (build_frame.py --outcome real)")
        return None
    return "m2_weight"


def estimation_sample(df, y, need=(), weight=None):
    d = df[df["analysis_sample"] & df[y].notna()]
    if weight is not None:
        d = d[d[weight] > 0]            # weight 0 = gold-only extras / outside the M2 design: never in population estimates
    for c in need:
        d = d[d[c].notna()]
    return d


def controls(d, rt=True):
    ctrl = list(CORE_CTRL)
    if rt and all(c in d and d[c].notna().all() for c in RT_CTRL):
        ctrl += list(RT_CTRL)
    return ctrl


def hypothesis_specs(df, y, bloc=BLOC, extra_num=(), extra_cat=(), drop_cat=(), topic_fe=True, key_pair=(17, 18),
                     weight=None):
    """Estimation samples and regressor lists for every H1-H4 model (shared by OLS/fracit, robustness and PPI).
    weight: design-weight column (§4.3a). Weight-0 rows are dropped; the pooled all-LS specs use the weights (spec
    key 'w'); the key LS17-vs-LS18 specs are a census and stay unweighted (asserted: all weights 1 there)."""
    opp = bloc[0]
    d_all = estimation_sample(df, y, need=("topic",) if topic_fe else (), weight=weight)
    d_all = d_all.assign(opp_x_coal=d_all[opp] * d_all["coalition_dep"])
    ctrl = controls(d_all) + list(extra_num)
    cat_base = [c for c in ["ministry_group"] + (["topic"] if topic_fe else []) + list(extra_cat) if c not in drop_cat]
    dk = d_all[d_all["ls"].isin(key_pair)]
    if weight is not None:
        assert np.allclose(dk[weight], 1.0), f"key-contrast sample LS{key_pair} must be a census (all {weight} == 1)"
    h4 = ["coalition_dep", opp, "opp_x_coal"] + list(bloc[1:]) + ctrl
    w = weight
    specs = {
        "H1H3": dict(d=d_all, num=list(bloc) + ctrl, cat=cat_base + ["session_id"], label="H1/H3 (session FE)", w=w),
        "H2key": dict(d=dk, num=["coalition_dep"] + list(bloc) + ctrl, cat=cat_base, label="H2 key (LS17 vs LS18)", w=None),
        "H2all": dict(d=d_all, num=["coalition_dep"] + list(bloc) + ctrl, cat=cat_base, label="H2 secondary (all LS)", w=w),
        "H4key": dict(d=dk, num=h4, cat=cat_base, label="H4 key (LS17 vs LS18)", w=None),
        "H4all": dict(d=d_all, num=h4, cat=cat_base, label="H4 (all LS)", w=w),
        "H4all_sessFE": dict(d=d_all, num=[opp, "opp_x_coal"] + list(bloc[1:]) + ctrl, cat=cat_base + ["session_id"],
                             label="H4 (all LS, session FE)", w=w),
    }
    return specs, opp


def h3_weights(d, ref_group=None, weight=None):
    """weights over ministry_group[...] coefficients giving: security minus question-weighted mean of the other groups
    (reference group coefficient is 0 by construction). weight: design weights -> estimated POPULATION question shares.
    The default reference is the unweighted modal group, as in build_X."""
    o = d[d["ministry_group"] != C.SECURITY_GROUP]
    shares = (o["ministry_group"].value_counts(normalize=True) if weight is None
              else o.groupby("ministry_group")[weight].sum() / o[weight].sum())
    ref_group = ref_group or d["ministry_group"].value_counts().index[0]
    w = {f"ministry_group[{C.SECURITY_GROUP}]": 1.0}
    for g, sh in shares.items():
        if g != ref_group:
            w[f"ministry_group[{g}]"] = -sh
    return w


def run_hypotheses(df, y="y_m2", family="ols", bloc=BLOC, extra_num=(), extra_cat=(), drop_cat=(),
                   cluster=("mp_id", "session_id"), topic_fe=True, key_pair=(17, 18), secondary=True, weight=None):
    """Fit the main H1-H4 models; returns (primary table with Holm, secondary table, dict of Fit objects).
    secondary=False fits only the three models behind the primary estimates (faster; used by robustness loops).
    weight: design-weight column (None = unweighted; design_weight(df, y) gives the pre-registered choice)."""
    specs, opp = hypothesis_specs(df, y, bloc, extra_num, extra_cat, drop_cat, topic_fe, key_pair, weight)
    if not secondary:
        specs = {k: v for k, v in specs.items() if k in ("H1H3", "H2key", "H4key")}
    fits = {k: fit_model(v["d"], y, v["num"], v["cat"], family, cluster, label=v["label"], weight=v["w"])
            for k, v in specs.items()}
    f13 = fits["H1H3"]
    w = h3_weights(specs["H1H3"]["d"], weight=specs["H1H3"]["w"])
    rows = []

    def add(h, fit, row, spec):
        r = dict(row)
        r.update({"hypothesis": h, "spec": spec, "n": fit.n, "g_mp": fit.g_sizes[0], "g_session": fit.g_sizes[1],
                  "family": family, "weights": "none" if fit.w is None else weight})
        rows.append(r)

    def est_row(fit, term):
        return fit.ame(term) if family != "ols" else fit.coef(term)

    add("H1", f13, est_row(f13, opp), "H1/H3 (session FE)")
    add("H2", fits["H2key"], est_row(fits["H2key"], "coalition_dep"), "H2 key LS17 vs LS18")
    if f"ministry_group[{C.SECURITY_GROUP}]" in f13.idx:
        add("H3", f13, f13.contrast(w, "security_vs_other"),
            "H1/H3 (session FE)" + ("" if family == "ols" else "; logit scale"))
    add("H4", fits["H4key"], fits["H4key"].coef("opp_x_coal"), "H4 key LS17 vs LS18")
    prim = pd.DataFrame(rows)
    prim["expected_sign"] = prim["hypothesis"].map(lambda h: HYPOTHESES[h][1])
    prim["direction_ok"] = np.sign(prim["estimate"]) == prim["expected_sign"]
    prim["p_one_sided"] = np.where(prim["direction_ok"], prim["p"] / 2, 1 - prim["p"] / 2)
    prim["p_holm"] = holm(prim["p"].values)
    prim["p_holm_one_sided"] = holm(prim["p_one_sided"].values)
    if not secondary:
        return prim, pd.DataFrame(), fits
    sec = []
    for h, k, term, spec in (("H2", "H2all", "coalition_dep", "H2 secondary: all LS"),
                             ("H4", "H4all", "opp_x_coal", "H4: all LS"),
                             ("H4", "H4all_sessFE", "opp_x_coal", "H4: all LS + session FE"),
                             ("H1", "H4key", opp, "H1 within key contrast (opp gap | majority, LS17)")):
        fit = fits[k]
        r = dict(fit.coef(term))
        r.update({"hypothesis": h, "spec": spec, "n": fit.n, "g_mp": fit.g_sizes[0], "g_session": fit.g_sizes[1], "family": family,
                  "weights": "none" if fit.w is None else weight})
        sec.append(r)
    sec = pd.DataFrame(sec)
    sec["expected_sign"] = sec["hypothesis"].map(lambda h: HYPOTHESES[h][1])
    sec["direction_ok"] = np.sign(sec["estimate"]) == sec["expected_sign"]
    return prim, sec, fits


def recovery_check(df, family="ols"):
    """Synthetic validation: for each primary estimate, compare (i) the estimate on the noise-free ORACLE mu_q (the
    probability limit of the spec), (ii) the estimate on the realised true score y_true with its 95% CI, and
    (iii) the naive estimate on the error-prone M2 score y_m2 (should be biased: differential measurement error).
    Also reports whether the y_true CI covers the oracle and the parametric planted value where one exists."""
    import synthetic
    pl = synthetic.planted_truth()
    planted = {"H1": np.nan, "H2": np.nan, "H3": pl["H3_security"], "H4": pl["H4_opp_x_coalition_dep"]}
    out = {}
    wt = design_weight(df, "y_m2")         # same rows and design weights for all three
    for name, y in (("oracle_mu", "mu"), ("true_score", "y_true"), ("naive_m2", "y_m2")):
        prim, _, _ = run_hypotheses(df, y, family, weight=wt)
        out[name] = prim.set_index("hypothesis")
    rows = []
    for h in ("H1", "H2", "H3", "H4"):
        o, t, m = out["oracle_mu"].loc[h], out["true_score"].loc[h], out["naive_m2"].loc[h]
        rows.append({"hypothesis": h, "planted_param": planted[h], "oracle": o["estimate"], "est_true_score": t["estimate"],
                     "ci_lo": t["ci_lo"], "ci_hi": t["ci_hi"], "ci_covers_oracle": bool(t["ci_lo"] <= o["estimate"] <= t["ci_hi"]),
                     "ci_covers_planted": (bool(t["ci_lo"] <= planted[h] <= t["ci_hi"]) if not np.isnan(planted[h]) else np.nan),
                     "naive_m2": m["estimate"], "naive_bias": m["estimate"] - o["estimate"],
                     "naive_ci_covers_oracle": bool(m["ci_lo"] <= o["estimate"] <= m["ci_hi"])})
    return pd.DataFrame(rows)


COLS = ["hypothesis", "spec", "term", "estimate", "se", "ci_lo", "ci_hi", "p", "p_holm", "p_one_sided",
        "p_holm_one_sided", "direction_ok", "n", "g_mp", "g_session", "family", "weights"]


def main():
    from build_frame import load_frame
    ap = argparse.ArgumentParser()
    ap.add_argument("--outcome", choices=C.OUTCOMES, default="synthetic")
    ap.add_argument("--y", default=None)
    ap.add_argument("--family", choices=("ols", "fracit"), default="ols")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--recovery", action="store_true", help="synthetic only: planted-effect recovery table")
    a = ap.parse_args()
    df = load_frame(a.outcome, rebuild=a.rebuild)
    if a.recovery:
        assert a.outcome == "synthetic"
        rec = recovery_check(df, a.family)
        rec.to_csv(os.path.join(C.outdir("tables", "synthetic"), f"recovery_{a.family}.csv"), index=False)
        pd.set_option("display.width", 250)
        pd.set_option("display.max_columns", 30)
        print(rec.round(4).to_string(index=False))
        return
    y = a.y or "y_m2"
    wt = design_weight(df, y)
    prim, sec, fits = run_hypotheses(df, y, a.family, weight=wt)
    out = C.outdir("tables", a.outcome)
    tag = f"{y}_{a.family}"
    prim[COLS[:1] + [c for c in COLS[1:] if c in prim]].to_csv(os.path.join(out, f"main_hypotheses_{tag}.csv"), index=False)
    sec.to_csv(os.path.join(out, f"secondary_hypotheses_{tag}.csv"), index=False)
    full = pd.concat([f.table().assign(model=k, weights="none" if f.w is None else wt) for k, f in fits.items()])
    full.to_csv(os.path.join(out, f"all_coefficients_{tag}.csv"), index=False)
    with open(os.path.join(out, f"main_hypotheses_{tag}.md"), "w", encoding="utf-8") as fh:
        fh.write(("**SYNTHETIC OUTCOME - NOT A RESULT**\n\n" if a.outcome == "synthetic" else "")
                 + f"Outcome {y}, {a.family}; two-way clustered (MP, session); Holm over H1-H4"
                 + (f"; pooled all-LS models design-weighted ({wt}, §4.3a), LS17-vs-LS18 key models unweighted (census)"
                    if wt else "; unweighted") + "\n\n")
        fh.write(C.df_to_markdown(prim[[c for c in COLS if c in prim]], floatfmt="{:.4f}"))
        fh.write("\nSecondary / variant specifications (no multiplicity adjustment)\n\n")
        fh.write(C.df_to_markdown(sec, floatfmt="{:.4f}"))
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(prim[[c for c in COLS if c in prim]].to_string(index=False))
    print(sec.to_string(index=False))


if __name__ == "__main__":
    main()
