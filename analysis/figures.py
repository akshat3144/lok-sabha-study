"""Publication figures (PNG 300 dpi + PDF) -> results/figures/ (real) or results/synthetic/figures/ (synthetic outcome).

    python -I -X utf8 analysis/figures.py --outcome synthetic [--y y_m2] [--only coverage,time,h2,ministry,coef,evasion]
    python -I -X utf8 analysis/figures.py --outcome real

fig_data_coverage   corpus coverage by session (does NOT need any outcome; always written to results/figures/)
fig_time_by_bloc    mean responsiveness by session, government vs opposition askers (+ outside support, mixed)
fig_h2_contrast     LS17 vs LS18 session means, and H2 / placebo contrasts (forest plot)
fig_ministry_groups ministry-group differences (adjusted deviations from the average group), security highlighted
fig_coef_h1_h4      H1-H4 estimates: naive (M2) vs bias-corrected (PPI++), 95% CI, Holm-adjusted p
fig_evasion_time    (real only) how PARTIAL/NONE parts fall short, by evasion type, by session (design-weighted parts)
M2 outcomes are pooled over LS13-18 with the design weights (models.design_weight): session means and shares in these
figures are weighted too (weight 0 = gold-only extras are dropped).
Synthetic-outcome figures carry a diagonal 'SYNTHETIC OUTCOME - NOT A RESULT' watermark.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import common as C  # noqa: E402
import models as M  # noqa: E402

# Okabe-Ito colour-blind-safe palette
COL = {"government": "#0072B2", "opposition": "#D55E00", "outside_support": "#009E73", "mixed": "#999999",
       "ink": "#222222", "grey": "#8c8c8c", "light": "#e6e6e6", "sec": "#CC79A7"}
PERIOD_LABEL = {"NDA-V3": "NDA-V3\n(coalition)", "UPA-1": "UPA-1\n(coalition)", "UPA-2": "UPA-2\n(coalition)",
                "NDA-M1": "NDA-M1\n(BJP majority)", "NDA-M2": "NDA-M2\n(BJP majority)", "NDA-M3": "NDA-M3\n(coalition)"}
Y_LABEL = {"y_m2": "Responsiveness (mean part score, 0-1)", "y_full": "Share of parts fully answered",
           "y_m1": "M1 embedding relevance", "y_true": "True score (synthetic)"}


def _style():
    plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 9.5, "axes.labelsize": 8.5, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.edgecolor": COL["ink"], "axes.linewidth": 0.7,
                         "xtick.color": COL["ink"], "ytick.color": COL["ink"], "legend.frameon": False,
                         "figure.dpi": 110, "savefig.dpi": 300, "pdf.fonttype": 42, "font.family": "DejaVu Sans"})


def _save(fig, name, outcome, synthetic_mark=True):
    d = C.outdir("figures", outcome)
    if outcome == "synthetic" and synthetic_mark:
        fig.text(0.5, 0.5, "SYNTHETIC OUTCOME - NOT A RESULT", ha="center", va="center", rotation=25, fontsize=22,
                 color="#b00000", alpha=0.18, weight="bold", transform=fig.transFigure)
    paths = []
    for ext in ("png", "pdf"):
        p = os.path.join(d, f"{name}.{ext}")
        fig.savefig(p, bbox_inches="tight")
        paths.append(p)
    plt.close(fig)
    return paths


# --------------------------------------------------------------------------- 1. data coverage (outcome-free)
def fig_data_coverage():
    cols = ["qid", "ls", "session", "date", "source_a", "answer_missing", "asker_bloc", "n_question_parts", "question_missing"]
    q = pd.read_parquet(C.QUESTIONS, columns=cols)
    q["sid"] = q["ls"].astype(str) + "-" + q["session"].astype(str).str.zfill(2)
    q["date"] = pd.to_datetime(q["date"], errors="coerce")
    g = q.groupby("sid")
    s = pd.DataFrame({"n": g.size(), "ls": g["ls"].first(), "date": g["date"].median()})
    for src in ("api", "docx", "pdf"):
        s[src] = g.apply(lambda x, k=src: (x["source_a"] == k).sum(), include_groups=False)
    s["noans"] = g["answer_missing"].mean() * 100
    s["unres"] = g["asker_bloc"].apply(lambda x: (x == "unresolved").mean() * 100)
    s["noparts"] = g["n_question_parts"].apply(lambda x: (x == 0).mean() * 100)
    s = s.sort_values("date").reset_index(drop=True)
    fig, ax = plt.subplots(2, 1, figsize=(6.8, 5.2), sharex=True, gridspec_kw={"height_ratios": [1.5, 1]})
    x = np.arange(len(s))
    ax[0].bar(x, s["api"] / 1000, color=COL["government"], width=0.8, label="answer text from API")
    ax[0].bar(x, s["docx"] / 1000, bottom=s["api"] / 1000, color=COL["opposition"], width=0.8, label="answer from .docx/.pdf files")
    ax[0].set_ylabel("Questions per session (thousands)")
    ax[0].set_ylim(0, s["n"].max() / 1000 * 1.22)
    ax[0].legend(loc="upper right", ncol=1, fontsize=7.5)
    ax[0].set_title(f"Corpus coverage by session: {int(s['n'].sum()):,} questions, 13th-18th Lok Sabha", pad=22)
    ax[1].plot(x, s["noans"], color=COL["ink"], lw=1.4, marker="o", ms=2.5, label="no answer text (%)")
    ax[1].plot(x, s["noparts"], color=COL["sec"], lw=1.4, marker="s", ms=2.5, label="no parsed question parts (%)")
    ax[1].plot(x, s["unres"], color=COL["outside_support"], lw=1.4, marker="^", ms=2.5, label="asker party unresolved (%)")
    ax[1].set_ylabel("% of questions")
    ax[1].set_ylim(0, max(s[["noans", "noparts", "unres"]].max()) * 1.35)
    ax[1].legend(loc="upper left", ncol=3, fontsize=7)
    bounds = s.groupby("ls").apply(lambda z: (z.index.min() - 0.5, z.index.max() + 0.5), include_groups=False)
    for k, (lo, hi) in bounds.items():
        for a in ax:
            a.axvline(lo, color=COL["light"], lw=0.8, zorder=0)
        ax[0].text((lo + hi) / 2, 1.0, f"{k}th LS", transform=ax[0].get_xaxis_transform(), ha="center", va="bottom",
                   fontsize=8, color=COL["ink"])
    mid = [int(round((lo + hi) / 2)) for lo, hi in bounds.values]
    ax[1].set_xticks([int(round(lo + 0.5)) for lo, _ in bounds.values])
    ax[1].set_xticklabels([str(s["date"].iloc[int(round(lo + 0.5))].year) for lo, _ in bounds.values])
    ax[1].set_xlabel("Session (median question date)")
    return _save(fig, "fig_data_coverage", "real", synthetic_mark=False)


# --------------------------------------------------------------------------- helpers
def _session_means(d, y, by, w=None):
    """(Design-weighted) mean of y per session x `by` cell with a linearised SE (unclustered, descriptive)."""
    def f(g):
        yv = g[y].to_numpy(float)
        wv = np.ones(len(yv)) if w is None else g[w].to_numpy(float)
        m = (wv * yv).sum() / wv.sum()
        n = len(yv)
        se = np.sqrt(((wv * (yv - m)) ** 2).sum() * n / max(n - 1, 1)) / wv.sum()
        return pd.Series({"mean": m, "se": se, "count": n})
    t = d.groupby(["session_id"] + by).apply(f, include_groups=False).reset_index()
    t["date"] = d.groupby("session_id")["date"].median().reindex(t["session_id"]).values
    t["ls"] = t["session_id"].str.split("-").str[0].astype(int)
    return t.sort_values("date")


def _shade_periods(ax, d, labels=True):
    per = d.groupby("govt_period")["date"].agg(["min", "max"]).sort_values("min")
    mt = d.groupby("govt_period")["majority_type"].first()
    for i, (p, r) in enumerate(per.iterrows()):
        if mt[p] == "single-party majority":
            ax.axvspan(r["min"], r["max"], color=COL["light"], alpha=0.65, lw=0, zorder=0)
        if labels:
            ax.text(r["min"] + (r["max"] - r["min"]) / 2, 1.0, PERIOD_LABEL.get(p, p), transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=7, color=COL["ink"])


# --------------------------------------------------------------------------- 2. time by bloc
def fig_time_by_bloc(df, y, outcome):
    wt = M.design_weight(df, y)
    d = M.estimation_sample(df, y, weight=wt)     # drops weight-0 (gold-only) M2 extras
    fig, ax = plt.subplots(figsize=(6.8, 3.9))
    t = _session_means(d, y, ["asker_bloc"], wt)
    t = t[t["count"] >= 100]       # suppress cells with < 100 questions (noise)
    for bloc, lw, a in (("government", 1.8, 1), ("opposition", 1.8, 1), ("outside_support", 1.0, 0.8), ("mixed", 1.0, 0.8)):
        s = t[t["asker_bloc"] == bloc]
        ax.errorbar(s["date"], s["mean"], yerr=1.96 * s["se"], color=COL[bloc], lw=lw, alpha=a, marker="o", ms=2.8,
                    capsize=0, elinewidth=0.7, label=bloc.replace("_", " "))
    _shade_periods(ax, d)
    ax.set_ylabel(Y_LABEL.get(y, y))
    ax.set_xlabel("Session date")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=4, fontsize=7.5)
    ax.set_title("Responsiveness by session and asker bloc (grey = single-party majority)", pad=22)
    ax.text(0.0, -0.31, ("Design-weighted session means with 95% CI (unclustered, descriptive); cells with fewer than 100 questions "
                         "suppressed." if wt else "Session means with 95% CI (unclustered, descriptive); cells with fewer than 100 questions suppressed."),
            transform=ax.transAxes, ha="left", va="top", fontsize=6.5, color=COL["grey"])
    return _save(fig, "fig_time_by_bloc", outcome)


# --------------------------------------------------------------------------- 3. H2 contrast
def fig_h2_contrast(df, y, outcome):
    import robustness as R
    wt = M.design_weight(df, y)
    d = M.estimation_sample(df, y, weight=wt)
    prim, sec, fits = M.run_hypotheses(df, y, "ols", weight=wt)
    plc = R.placebo_contrasts(df, y)
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.4), gridspec_kw={"width_ratios": [1.15, 1]})
    # left: session means LS17 vs LS18, by session index within LS
    t = _session_means(d[d["ls"].isin([17, 18])], y, [], wt)
    for ls, c, nm in ((17, COL["government"], "17th LS: BJP majority (2019-24)"), (18, COL["opposition"], "18th LS: coalition-dependent (2024-)")):
        s = t[t["ls"] == ls].reset_index(drop=True)
        ax[0].errorbar(s["date"], s["mean"], yerr=1.96 * s["se"], color=c, lw=1.6, marker="o", ms=3, capsize=0,
                       elinewidth=0.7, label=nm)
        dl = d[d["ls"] == ls]
        m = np.average(dl[y], weights=dl[wt]) if wt else dl[y].mean()
        ax[0].hlines(m, s["date"].min(), s["date"].max(), color=c, lw=0.8, ls="--")
    ax[0].axvline(pd.Timestamp("2024-06-09"), color=COL["grey"], lw=0.8, ls=":")
    ax[0].set_ylabel(Y_LABEL.get(y, y))
    ax[0].set_title("Session means (dashed: LS mean)")
    ax[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1, fontsize=7)
    ax[0].tick_params(axis="x", rotation=30)
    # right: forest of contrasts
    rows = [("H2 key: LS18 vs LS17", fits["H2key"].coef("coalition_dep"), True),
            ("H2 secondary: coalition vs majority\n(all LS, no session FE)", fits["H2all"].coef("coalition_dep"), True)]
    for r in plc.itertuples():
        if r.contrast == "LS18 vs LS17":
            continue
        kind = "actual switch" if not r.same_majority_type else "placebo"
        lab = f"{r.contrast} ({kind})" if "trend" not in r.contrast else r.contrast
        rows.append((lab, {"estimate": r.estimate, "ci_lo": r.ci_lo, "ci_hi": r.ci_hi}, not r.same_majority_type or "trend" in r.contrast and False))
    ypos = np.arange(len(rows))[::-1]
    for yy, (lab, r, real) in zip(ypos, rows):
        isph = "placebo" in lab or "trend" in lab
        c = COL["grey"] if isph else COL["opposition"]
        ax[1].plot([r["ci_lo"], r["ci_hi"]], [yy, yy], color=c, lw=1.5)
        ax[1].plot(r["estimate"], yy, "o", color=c, ms=4.5)
    ax[1].axvline(0, color=COL["ink"], lw=0.7)
    ax[1].set_yticks(ypos)
    ax[1].set_yticklabels([r[0] for r in rows], fontsize=7)
    ax[1].set_xlabel("Later minus earlier period (score units)")
    ax[1].set_title("H2 contrast vs placebo contrasts")
    fig.tight_layout()
    return _save(fig, "fig_h2_contrast", outcome)


# --------------------------------------------------------------------------- 4. ministry groups
def fig_ministry_groups(df, y, outcome):
    wt = M.design_weight(df, y)
    prim, sec, fits = M.run_hypotheses(df, y, "ols", secondary=False, weight=wt)
    f = fits["H1H3"]
    d = M.estimation_sample(df, y, need=("topic",), weight=wt)
    shares = (d["ministry_group"].value_counts(normalize=True) if wt is None
              else d.groupby("ministry_group")[wt].sum().sort_values(ascending=False) / d[wt].sum())
    ref = d["ministry_group"].value_counts().index[0]
    rows = []
    for g in shares.index:
        w = {f"ministry_group[{h}]": (1.0 if h == g else 0.0) - s for h, s in shares.items() if h != ref}
        if g == ref:
            w = {f"ministry_group[{h}]": -s for h, s in shares.items() if h != ref}
        r = f.contrast({k: v for k, v in w.items() if k in f.idx}, g)
        r["n"] = int((d["ministry_group"] == g).sum())
        rows.append(r)
    t = pd.DataFrame(rows).sort_values("estimate")
    fig, ax = plt.subplots(figsize=(5.8, 3.4))
    yy = np.arange(len(t))
    for i, r in zip(yy, t.itertuples()):
        c = COL["sec"] if r.term == C.SECURITY_GROUP else COL["government"]
        ax.plot([r.ci_lo, r.ci_hi], [i, i], color=c, lw=1.7)
        ax.plot(r.estimate, i, "o", color=c, ms=5)
    ax.axvline(0, color=COL["ink"], lw=0.7)
    ax.set_yticks(yy)
    ax.set_yticklabels([f"{r.term.replace('_', ' ')} (n={r.n:,})" for r in t.itertuples()], fontsize=7.5)
    ax.set_xlabel("Adjusted difference from average group (score units)")
    h3 = prim[prim["hypothesis"] == "H3"].iloc[0]
    ax.set_title("Ministry-group differences (security highlighted)\n"
                 f"H3, security vs other: {h3['estimate']:+.3f} [{h3['ci_lo']:+.3f}, {h3['ci_hi']:+.3f}]", fontsize=9)
    ax.text(0.0, -0.2, "n = M2-labelled questions in the estimation sample (unweighted counts); estimates are design-weighted.",
            transform=ax.transAxes, ha="left", va="top", fontsize=6.5, color=COL["grey"])
    fig.tight_layout()
    return _save(fig, "fig_ministry_groups", outcome)


# --------------------------------------------------------------------------- 5. coefficient plot H1-H4
def fig_coef(df, y, outcome):
    import bias_correct as B
    prim, _, _ = M.run_hypotheses(df, y, "ols", secondary=False, weight=M.design_weight(df, y))
    ppi = None
    if df["gold_complete"].sum() > 0:
        ppi = B.run_ppi_hypotheses(df, "ols", f_col=y)
        ppi = ppi[ppi["method"].str.startswith("PPI")]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    labels = {"H1": "H1 opposition vs government askers\n(expected -)", "H2": "H2 coalition-dependent vs majority,\nLS18 vs LS17 (expected +)",
              "H3": "H3 security vs other ministries\n(expected -)", "H4": "H4 opposition x coalition-dependent\n(expected +)"}
    hs = ["H1", "H2", "H3", "H4"]
    for i, h in enumerate(hs[::-1]):
        r = prim[prim["hypothesis"] == h].iloc[0]
        off = 0.14 if ppi is not None else 0
        ax.plot([r["ci_lo"], r["ci_hi"]], [i + off] * 2, color=COL["government"], lw=1.8)
        ax.plot(r["estimate"], i + off, "o", color=COL["government"], ms=5,
                label="Naive (M2 labels)" if i == 0 else None)
        if ppi is not None:
            p = ppi[ppi["hypothesis"] == h].iloc[0]
            ax.plot([p["ci_lo"], p["ci_hi"]], [i - off] * 2, color=COL["opposition"], lw=1.8)
            ax.plot(p["estimate"], i - off, "s", color=COL["opposition"], ms=5, label="Bias-corrected (PPI++)" if i == 0 else None)
    ax.axvline(0, color=COL["ink"], lw=0.7)
    ax.set_yticks(range(4))
    ax.set_yticklabels([labels[h] for h in hs[::-1]], fontsize=7.5)
    ax.set_xlabel("Estimated difference in responsiveness (score units, 0-1)")
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.05))
    ax.set_title("H1-H4: association estimates with 95% CI (two-way clustered)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.35, -0.2), ncol=2, fontsize=7.5)
    pt = ", ".join(f"{h} {prim.loc[prim.hypothesis == h, 'p_holm'].iloc[0]:.2g}" for h in hs)
    note = "Holm-adjusted two-sided p, naive: " + pt
    if ppi is not None:
        note += "\nHolm-adjusted two-sided p, PPI++: " + ", ".join(
            f"{h} {ppi.loc[ppi.hypothesis == h, 'p_holm'].iloc[0]:.2g}" for h in hs)
    ax.text(-0.62, -0.38, note, transform=ax.transAxes, ha="left", va="top", fontsize=6.5, color=COL["grey"])
    fig.tight_layout()
    return _save(fig, "fig_coef_h1_h4", outcome)


# --------------------------------------------------------------------------- 6. evasion types over time
EVA_COL = {"GENERIC": "#0072B2", "INCOMPLETE": "#56B4E9", "DIFFERENT_QUESTION": "#E69F00", "NOT_MAINTAINED": "#009E73",
           "JURISDICTION": "#D55E00", "DEFERRAL": "#CC79A7", "OTHER": "#999999"}


def fig_evasion_time(df, y, outcome):
    """Share of all parts that are PARTIAL/NONE, stacked by M2 evasion type, per session (design-weighted parts)."""
    import paper_tables as PT
    p = PT.weighted_parts(df)
    p["date"] = pd.to_datetime(p["date"])
    sess = p.groupby("session_id")["date"].median().sort_values()
    tot = p.groupby("session_id")["m2_weight"].sum()
    nf = p[p["label"] != "FULL"]
    sh = (nf.groupby(["session_id", "eva_group"])["m2_weight"].sum().unstack(fill_value=0).reindex(sess.index, fill_value=0)
          .div(tot.reindex(sess.index), axis=0) * 100)
    order = [k for k, _ in PT.EVA_GROUPS if k in sh.columns]
    names = dict(PT.EVA_GROUPS)
    fig, ax = plt.subplots(figsize=(6.8, 3.9))
    ax.stackplot(sess.values, [sh[k].values for k in order], colors=[EVA_COL[k] for k in order],
                 labels=[names[k] for k in order], lw=0.3, edgecolor="white")
    _shade_periods(ax, M.estimation_sample(df, y, weight=M.design_weight(df, y)), labels=True)
    ax.set_ylabel("% of parts partly or not answered")
    ax.set_xlabel("Session date")
    ax.set_ylim(0, (sh.sum(axis=1).max() * 1.05))
    h, l = ax.get_legend_handles_labels()
    ax.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=4, fontsize=7.5)
    ax.set_title("How parts fall short of a full answer, by evasion type (grey = single-party majority)", pad=22, fontsize=9)
    ax.text(0.0, -0.34, "Design-weighted M2 parts (PARTIAL or NONE), stacked; height = share of all labelled parts in the session.\n"
            "Other pools OTHER, WITHHELD, DOES_NOT_ARISE, ANNEXURE_ONLY.", transform=ax.transAxes, ha="left", va="top",
            fontsize=6.3, color=COL["grey"])
    return _save(fig, "fig_evasion_time", outcome)


def main():
    from build_frame import load_frame
    ap = argparse.ArgumentParser()
    ap.add_argument("--outcome", choices=C.OUTCOMES, default="synthetic")
    ap.add_argument("--y", default="y_m2")
    ap.add_argument("--only", default="coverage,time,h2,ministry,coef,evasion")
    a = ap.parse_args()
    _style()
    want = set(a.only.split(","))
    if "coverage" in want:
        print(fig_data_coverage())
    need_outcome = want - {"coverage"}
    if need_outcome:
        df = load_frame(a.outcome)
        if df[a.y].notna().sum() == 0:
            print(f"outcome {a.y} not available for outcome={a.outcome}; skipping outcome figures")
            return
        if "time" in want:
            print(fig_time_by_bloc(df, a.y, a.outcome))
        if "h2" in want:
            print(fig_h2_contrast(df, a.y, a.outcome))
        if "ministry" in want:
            print(fig_ministry_groups(df, a.y, a.outcome))
        if "coef" in want:
            print(fig_coef(df, a.y, a.outcome))
        if "evasion" in want:
            if a.outcome == "real" and a.y == "y_m2":
                print(fig_evasion_time(df, a.y, a.outcome))
            else:
                print("evasion figure needs the real M2 labels (--outcome real --y y_m2); skipped")


if __name__ == "__main__":
    main()
