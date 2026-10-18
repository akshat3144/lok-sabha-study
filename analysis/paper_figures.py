"""FINAL paper figures (1 to 5) and tables (1 to 3) -> paper/draft/figures, paper/draft/tables.

    python -I -X utf8 analysis/paper_figures.py            # from lok_sabha/

Nothing is re-estimated differently: every plotted estimate is read from the exploration CSVs / markdown tables in
results/explore and results/tables (rq2_*, rq3_all_models.csv, a1_ministry_table_full.csv, b4_auc_by_type.csv, ...).
The only fresh computations are descriptive: per part cosine values for the Figure 1 box plot (same heldout gold parts as
explore_rq4_rq1.b4_embeddings), the hinge fit line for Figure 2 (same OLS as explore_rq2 at the stored kink), and the
agreement statistics of Table 2 (validate.py functions; bootstrap CIs for human vs human kappa are new, all other CIs are
read from the existing reports).

STRICT TEXT RULE: no hyphen, en dash or em dash in any figure text, table or caption (true minus U+2212 is allowed).
The script scans every Text artist of every figure, and every table / caption file, and fails if it finds one.
"""
import json
import os
import re
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C  # noqa: E402
import validate as V  # noqa: E402
import explore_rq4_rq1 as E4  # noqa: E402  (only gold_frame; import has no side effects besides makedirs)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.text import Text  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

LS_ROOT = C.ROOT
PROJECT = os.path.dirname(LS_ROOT)
EXP = os.path.join(C.RESULTS, "explore")
TAB_SRC = os.path.join(C.RESULTS, "tables")
FIG_DIR = os.path.join(PROJECT, "draft", "figures")
TAB_DIR = os.path.join(PROJECT, "draft", "tables")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)

MINUS = "−"
# Okabe Ito
BLACK, ORANGE, SKY, GREEN, YELLOW, BLUE, VERM, PURPLE = (
    "#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")
GREY = "#7a7a7a"
LIGHT = "#cfcfcf"

FONT = "Arial"
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": [FONT, "DejaVu Sans"],
    "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "legend.fontsize": 6.2, "axes.unicode_minus": True, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5, "lines.linewidth": 1.0,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300, "figure.dpi": 150,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})
W1, W2 = 3.5, 7.0

FIG_TEXT = {}          # figure name -> list of strings drawn (for the dash check)
FACT = {}              # numbers reused in captions


def num_ci(s):
    """'0.123 [0.100, 0.150]' or with ASCII minus -> (est, lo, hi)."""
    s = str(s).replace(MINUS, "-")
    m = re.match(r"\s*([-+]?\d*\.?\d+)\s*\[\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*\]", s)
    if not m:
        raise ValueError(f"cannot parse '{s}'")
    return tuple(float(x) for x in m.groups())


def fmt(x, d=3, sign=False):
    s = f"{x:+.{d}f}" if sign else f"{x:.{d}f}"
    return s.replace("-", MINUS)


def fmt_ci(e, lo, hi, d=3):
    return f"{fmt(e, d)} [{fmt(lo, d)}, {fmt(hi, d)}]"


def save_fig(fig, name):
    fig.canvas.draw()
    strings = [t.get_text() for t in fig.findobj(Text) if t.get_text().strip()]
    FIG_TEXT[name] = strings
    fig.savefig(os.path.join(FIG_DIR, name + ".png"), dpi=300)
    fig.savefig(os.path.join(FIG_DIR, name + ".pdf"))
    plt.close(fig)
    print("saved", name, flush=True)


def panel(ax, letter):
    ax.text(-0.02, 1.03, letter, transform=ax.transAxes, fontsize=8, fontweight="bold", ha="right", va="bottom")


LS_LAB = [f"LS{k}" for k in range(13, 19)]
NICE = {"GENERIC": "Generic", "INCOMPLETE": "Incomplete", "DIFFERENT_QUESTION": "Different question",
        "NOT_MAINTAINED": "Not maintained", "JURISDICTION": "State subject", "DEFERRAL": "Deferral",
        "OTHER": "Other", "DOES_NOT_ARISE": "Does not arise", "WITHHELD": "Withheld"}


# ------------------------------------------------------------------------------------------------ Figure 1
def figure1():
    g = E4.gold_frame()
    m1 = pd.read_parquet(os.path.join(C.MEASURES, "m1_scores.parquet"),
                         columns=["part_id", "m1_cos", "m1_valid", "split"])
    d = g.merge(m1, on="part_id", how="left")
    d = d[d["m1_valid"].fillna(False)]
    dh = d[d["split"] == "heldout"].copy()
    dh["type"] = np.where(dh["label"] == "FULL", "FULL", dh["evasion_type"])
    auc_tab = pd.read_csv(os.path.join(EXP, "b4_auc_by_type.csv")).set_index("gold group")
    # reproduce the published b4 table sizes as a guard
    assert int((dh["type"] == "FULL").sum()) == 1038 and int((dh["type"] == "NOT_MAINTAINED").sum()) == int(auc_tab.loc["NOT_MAINTAINED", "n parts"])
    full = dh.loc[dh["type"] == "FULL", "m1_cos"].to_numpy(float)
    full_med = float(np.median(full))
    types = [t for t in NICE if (dh["type"] == t).sum() >= 10]
    types.sort(key=lambda t: -np.median(dh.loc[dh["type"] == t, "m1_cos"]))
    order = ["FULL"] + types
    fig, ax = plt.subplots(figsize=(W1, 3.45), constrained_layout=True)
    data, labels = [], []
    for t in order:
        v = dh.loc[dh["type"] == t, "m1_cos"].to_numpy(float)
        data.append(v)
        nm = "Full answers" if t == "FULL" else NICE[t]
        labels.append(f"{nm} (n={len(v):,})")
    pos = np.arange(len(order))[::-1]
    bp = ax.boxplot(data, positions=pos, vert=False, widths=0.62, showfliers=False, patch_artist=True,
                    medianprops=dict(color=BLACK, lw=1.0), whiskerprops=dict(lw=0.7, color=GREY),
                    capprops=dict(lw=0.7, color=GREY), boxprops=dict(lw=0.6))
    cols = []
    for t in order:
        cols.append(SKY if t == "FULL" else (VERM if t == "NOT_MAINTAINED" else ORANGE))
    rng = np.random.default_rng(1)
    for b, c, p, v in zip(bp["boxes"], cols, pos, data):
        b.set_facecolor(c)
        b.set_alpha(0.55)
        ax.scatter(v, p + rng.uniform(-0.2, 0.2, len(v)), s=1.8, color=BLACK, alpha=0.25, lw=0, zorder=3)
    ax.axvline(full_med, color=BLUE, ls="--", lw=0.9, zorder=1)
    ax.text(full_med + 0.02, pos[0] + 0.62, f"Median of full answers {full_med:.2f}", color=BLUE, fontsize=6, ha="left", va="bottom")
    ax.set_yticks(pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Embedding cosine of question part and answer part")
    ax.set_ylim(-0.7, pos[0] + 1.05)
    nmv = auc_tab.loc["NOT_MAINTAINED"]
    FACT["fig1"] = dict(n=len(dh), n_full=len(full), full_med=full_med,
                        nm_med=float(nmv["median cos"]), nm_n=int(nmv["n parts"]), nm_auc=float(nmv["AUC vs FULL (tuned)"]),
                        nm_ci=nmv["95% CI"], omitted=int((dh["type"] == "WITHHELD").sum()) + int((dh["type"] == "ANNEXURE_ONLY").sum()))
    save_fig(fig, "fig1_embedding_by_evasion_type")


# ------------------------------------------------------------------------------------------------ Figure 2
def figure2():
    sm = pd.read_csv(os.path.join(EXP, "rq2_t4_session_means.csv")).sort_values("t").reset_index(drop=True)
    key = json.load(open(os.path.join(EXP, "rq2_key.json")))
    kap = key["hinge"]["kappa"]
    klo, khi = key["hinge"]["kappa_ci"]
    ppi = pd.read_csv(os.path.join(EXP, "rq2_t1_ppi_means.csv"))
    ppi = ppi[ppi["method"] == "PPI++ (lambda tuned)"].set_index("group")
    gold = pd.read_csv(os.path.join(EXP, "rq2_t1_gold_vs_m2_long.csv"))
    gold = gold[(gold["stat"] == "gold_mean") & gold["group"].isin(LS_LAB)].set_index("group")
    t, y, se = sm["t"].to_numpy(), sm["y"].to_numpy(), sm["se"].to_numpy()
    # hinge fit at the stored kink (same OLS as explore_rq2.hinge_fit)
    X = np.column_stack([np.ones(len(t)), np.maximum(t - kap, 0)])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    r2 = 1 - ((y - X @ beta) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    assert abs(beta[1] - key["hinge"]["slope_after"]) < 1e-6, (beta, key["hinge"])
    fig, ax = plt.subplots(figsize=(W2, 3.5), constrained_layout=True)
    # government periods
    sm["bloc"] = np.where(sm["govt_period"].str.startswith("UPA"), "UPA", "NDA")
    per = sm.groupby("govt_period").agg(t0=("t", "min"), t1=("t", "max"), ls=("ls", "first"), bloc=("bloc", "first")).sort_values("t0")
    edges = [per["t0"].iloc[0] - 0.15]
    for i in range(1, len(per)):
        edges.append((per["t1"].iloc[i - 1] + per["t0"].iloc[i]) / 2)
    edges.append(per["t1"].iloc[-1] + 0.15)
    ymin, ymax = 0.665, 0.935
    blocs = list(per["bloc"])
    i = 0
    while i < len(blocs):
        j = i
        while j + 1 < len(blocs) and blocs[j + 1] == blocs[i]:
            j += 1
        col = SKY if blocs[i] == "UPA" else ORANGE
        ax.axvspan(edges[i], edges[j + 1], color=col, alpha=0.13 if blocs[i] == "UPA" else 0.10, lw=0, zorder=0)
        ax.text((edges[i] + edges[j + 1]) / 2, ymax - 0.004, "UPA government" if blocs[i] == "UPA" else "NDA government",
                ha="center", va="top", fontsize=6.3, color=GREY)
        i = j + 1
    # kink
    ax.axvspan(klo, khi, color=VERM, alpha=0.14, lw=0, zorder=0.5)
    ax.axvline(kap, color=VERM, ls=":", lw=1.1, zorder=1)
    tt = np.linspace(t.min(), t.max(), 400)
    ax.plot(tt, beta[0] + beta[1] * np.maximum(tt - kap, 0), color=BLACK, lw=0.9, ls="--", zorder=2)
    # sessions
    ax.errorbar(t, y, yerr=1.96 * se, fmt="none", ecolor=GREY, elinewidth=0.5, alpha=0.8, zorder=2)
    ax.plot(t, y, color=GREY, lw=0.45, alpha=0.7, zorder=2)
    ax.scatter(t, y, s=7, color=GREY, zorder=3, lw=0)
    # PPI++ and gold per LS
    for k, name in zip(range(13, 19), LS_LAB):
        sel = sm[sm["ls"] == k]
        a, b = sel["t"].min(), sel["t"].max()
        mid = (a + b) / 2
        r = ppi.loc[name]
        ax.hlines(r["est"], a, b, color=BLUE, lw=2.0, zorder=4)
        ax.fill_between([a, b], r["lo"], r["hi"], color=BLUE, alpha=0.18, lw=0, zorder=3.5)
        ax.errorbar([mid], [r["est"]], yerr=[[r["est"] - r["lo"]], [r["hi"] - r["est"]]], fmt="none", ecolor=BLUE,
                    elinewidth=1.0, capsize=2, zorder=4)
        ax.scatter([mid + 0.55], [gold.loc[name, "est"]], marker="D", s=15, color=VERM, edgecolor="white", lw=0.4, zorder=5)
        ax.text(mid, ymin + 0.004, name, ha="center", va="bottom", fontsize=6.3, color=GREY)
    ax.set_xlim(edges[0], edges[-1])
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("Year of session")
    ax.set_ylabel("Mean answer score (0 to 1)")
    ax.set_xticks(range(2000, 2027, 4))
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.05))
    ax.annotate(f"Estimated kink {kap:.1f}\n(95% interval {klo:.1f} to {khi:.1f})", xy=(kap, 0.716), xytext=(2021.3, 0.700),
                fontsize=6.3, color=VERM, ha="center", va="center",
                arrowprops=dict(arrowstyle="->", color=VERM, lw=0.7, shrinkA=0, shrinkB=1))
    handles = [
        Line2D([], [], color=GREY, marker="o", ms=2.8, lw=0.5, label="Session mean from LLM labels, design weighted (95% CI)"),
        Line2D([], [], color=BLUE, lw=2.0, label="Bias corrected (PPI++) mean by Lok Sabha (95% CI band)"),
        Line2D([], [], color=VERM, marker="D", ms=3.5, lw=0, markeredgecolor="white", label="Human gold mean by Lok Sabha"),
        Line2D([], [], color=BLACK, ls="--", lw=0.9, label="Flat then declining fit at the kink"),
    ]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 0.07), fontsize=6.2, handlelength=1.8)
    FACT["fig2"] = dict(n_sessions=len(sm), n_q=int(sm["n"].sum()), kap=kap, klo=klo, khi=khi, slope=float(beta[1]), r2=float(r2),
                        n_ppi=139654)
    save_fig(fig, "fig2_trend_by_session")


# ------------------------------------------------------------------------------------------------ Figure 3
def figure3():
    t = pd.read_csv(os.path.join(EXP, "rq2_t3_rate_per_100_parts.csv")).set_index("group").loc[LS_LAB]
    P = {c: t[c].map(num_ci) for c in t.columns}
    est = lambda c: np.array([P[c].iloc[i][0] for i in range(6)])  # noqa: E731
    lo = lambda c: np.array([P[c].iloc[i][1] for i in range(6)])  # noqa: E731
    hi = lambda c: np.array([P[c].iloc[i][2] for i in range(6)])  # noqa: E731
    other = est("GENERIC") + est("DIFFERENT_QUESTION") + est("OTHER")
    assert np.allclose(other + est("DEFERRAL") + est("INCOMPLETE") + est("JURISDICTION") + est("NOT_MAINTAINED"),
                       est("ALL non-FULL"), atol=0.05)
    fig, ax = plt.subplots(figsize=(W1, 3.45), constrained_layout=True)
    x = np.arange(6)
    series = [("Deferral", "DEFERRAL", BLUE, "o", "-"), ("Incomplete", "INCOMPLETE", VERM, "s", "-"),
              ("State subject", "JURISDICTION", GREEN, "^", "-"), ("Not maintained", "NOT_MAINTAINED", PURPLE, "v", "-")]
    ax.fill_between(x, lo("ALL non-FULL"), hi("ALL non-FULL"), color=GREY, alpha=0.18, lw=0)
    ax.plot(x, est("ALL non-FULL"), color=GREY, lw=1.6, marker="o", ms=3, label="All shortfalls")
    ax.plot(x, other, color=ORANGE, lw=1.3, ls="--", marker="D", ms=3, label="Other types")
    for lab, c, col, mk, ls in series:
        ax.fill_between(x, lo(c), hi(c), color=col, alpha=0.20, lw=0)
        ax.plot(x, est(c), color=col, lw=1.2, marker=mk, ms=3, ls=ls, label=lab)
    ax.set_xticks(x)
    ax.set_xticklabels(LS_LAB)
    ax.set_ylabel("Parts not fully answered, per 100 parts")
    ax.set_ylim(0, 39)
    ax.legend(loc="upper left", ncol=2, fontsize=6.2, columnspacing=1.0, handlelength=1.8)
    FACT["fig3"] = dict(all=[float(v) for v in est("ALL non-FULL")], other=[float(v) for v in other])
    save_fig(fig, "fig3_composition_of_shortfalls")


# ------------------------------------------------------------------------------------------------ Figure 4
def figure4():
    d = pd.read_csv(os.path.join(EXP, "rq3_all_models.csv"))
    a = d[(d["spec"] == "two-way (MP,session)") & (d["term"] == "opp")].set_index("group").loc[LS_LAB]
    p = d[(d["spec"] == "roles by LS (ref gov_main)") & (d["term"] == "c_gov_partner") & (d["group"] == "LS18")].iloc[0]
    fig, ax = plt.subplots(figsize=(W1, 3.2), constrained_layout=True)
    x = np.arange(6)
    ax.axhline(0, color=BLACK, lw=0.7)
    ax.errorbar(x, a["estimate"], yerr=[a["estimate"] - a["ci_lo"], a["ci_hi"] - a["estimate"]], fmt="o", color=BLUE, ms=4,
                elinewidth=1.1, capsize=2.5, zorder=3, label="Opposition minus ruling party askers")
    ax.errorbar([5.22], [p["estimate"]], yerr=[[p["estimate"] - p["ci_lo"]], [p["ci_hi"] - p["estimate"]]], fmt="D", color=VERM,
                ms=4, elinewidth=1.1, capsize=2.5, zorder=3, label="Coalition partner MPs minus BJP MPs (LS18)")
    ax.set_xticks(x)
    ax.set_xticklabels(LS_LAB)
    ax.set_xlim(-0.5, 5.75)
    ax.set_ylabel("Difference in answer score (0 to 1 scale)")
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.02))
    ax.set_ylim(-0.062, 0.047)
    ax.legend(loc="lower left", fontsize=6.2, handletextpad=0.4)
    FACT["fig4"] = dict(opp=[(float(r.estimate), float(r.ci_lo), float(r.ci_hi)) for r in a.itertuples()],
                        partner=(float(p["estimate"]), float(p["ci_lo"]), float(p["ci_hi"])), n=[int(v) for v in a["n"]])
    save_fig(fig, "fig4_asker_gap_by_ls")


# ------------------------------------------------------------------------------------------------ Figure 5
def parse_tier_table():
    txt = open(os.path.join(EXP, "rq4_rq1.md"), encoding="utf-8").read()
    i = txt.index("Over time (Lok Sabha): JURISDICTION evasions as % of all parts, by tier")
    block = txt[i:].split("\n\n")[1]
    rows = [r for r in block.splitlines() if re.match(r"\|\s*1[3-8]\s*\|", r)]
    assert len(rows) == 6
    out = {"A": [], "C": [], "all": []}
    for r in rows:
        cells = [c.strip() for c in r.strip("|").split("|")]
        out["all"].append(num_ci(cells[2]))
        out["A"].append(num_ci(cells[3]))
        out["C"].append(num_ci(cells[5]))
    return out


SHORT = {"Personnel, Public Grievances and Pensions": "Personnel and Pensions",
         "Science and Technology": "Science and Technology"}


def figure5():
    A = pd.read_csv(os.path.join(EXP, "a1_ministry_table_full.csv")).sort_values("rank").reset_index(drop=True)
    assert len(A) == 55
    overall = 0.836
    fig, axs = plt.subplots(1, 2, figsize=(W2, 3.6), gridspec_kw={"width_ratios": [1.9, 1.0]}, constrained_layout=True)
    ax = axs[0]
    r = A["rank"].to_numpy()
    sec = A["security"].astype(bool).to_numpy()
    ax.axhline(overall, color=GREY, lw=0.7, ls="--", zorder=1)
    ax.text(55.6, overall + 0.003, "Mean 0.836", ha="right", va="bottom", fontsize=6, color=GREY)
    for m, c, z in ((~sec, BLUE, 2), (sec, VERM, 4)):
        ax.errorbar(r[m], A["score"][m], yerr=[A["score"][m] - A["score_lo"][m], A["score_hi"][m] - A["score"][m]], fmt="o",
                    ms=3.2 if c == BLUE else 4.6, color=c, elinewidth=0.6, capsize=0, zorder=z, alpha=0.95 if c == VERM else 0.8,
                    markeredgecolor="white" if c == VERM else c, markeredgewidth=0.4)
    top = A.head(5)
    for k, row in enumerate(top.itertuples()):
        ty = 0.949 - 0.0125 * k
        nm = SHORT.get(row.ministry, row.ministry)
        ax.plot([row.rank + 0.4, 8.6], [row.score, ty], color=LIGHT, lw=0.5, zorder=1)
        ax.text(9.0, ty, f"{row.rank}. {nm}", fontsize=6, ha="left", va="center")
    bot = A.tail(5).iloc[::-1].reset_index(drop=True)
    for k, row in enumerate(bot.itertuples()):
        ty = 0.752 + 0.011 * k
        nm = SHORT.get(row.ministry, row.ministry)
        ax.plot([row.rank - 0.4, 43.9], [row.score, ty], color=LIGHT, lw=0.5, zorder=1)
        ax.text(43.5, ty, f"{nm} ({row.rank})", fontsize=6, ha="right", va="center")
    ax.set_xlabel("Ministry rank (1 = most fully answered)")
    ax.set_ylabel("Mean answer score (0 to 1)")
    ax.set_xlim(0, 56)
    ax.set_ylim(0.695, 0.96)
    ax.set_xticks([1, 10, 20, 30, 40, 50, 55])
    ax.legend(handles=[Line2D([], [], color=BLUE, marker="o", ms=3, lw=0.6, label="Ministry (95% CI)"),
                       Line2D([], [], color=VERM, marker="o", ms=4.5, lw=0.6, markeredgecolor="white",
                              label="Security ministries (Home Affairs, Defence, External Affairs)")],
              loc="lower left", bbox_to_anchor=(0.0, 0.0), fontsize=6.0)
    panel(ax, "a")
    # panel b
    bx = axs[1]
    T = parse_tier_table()
    x = np.arange(6)
    for key, lab, col, mk, off in (("A", "State or Concurrent List ministries", VERM, "o", -0.07), ("C", "Union ministries", BLUE, "s", 0.07)):
        e = np.array([v[0] for v in T[key]])
        lo = np.array([v[1] for v in T[key]])
        hi = np.array([v[2] for v in T[key]])
        bx.errorbar(x + off, e, yerr=[e - lo, hi - e], fmt=mk + "-", color=col, ms=3.3, elinewidth=0.8, capsize=1.8, lw=1.0, label=lab)
    bx.set_xticks(x)
    bx.set_xticklabels(LS_LAB, fontsize=6.2)
    bx.set_ylabel("Parts answered with a State subject deflection (%)")
    bx.set_ylim(0, 3.7)
    bx.legend(loc="upper left", fontsize=6.0, handlelength=1.6)
    panel(bx, "b")
    FACT["fig5"] = dict(n_min=len(A), nsec=int(sec.sum()), A=T["A"], C=T["C"], top=list(top["ministry"]), bot=list(bot["ministry"]))
    save_fig(fig, "fig5_ministries_and_jurisdiction")


# ------------------------------------------------------------------------------------------------ tables
def esc_tex(s):
    s = str(s).replace("%", r"\%").replace("&", r"\&")
    s = s.replace("κ", r"$\kappa$").replace("α", r"$\alpha$")
    return s


def write_table(name, number, title, header, rows, note, align=None, first_left=True):
    ncol = len(header)
    align = align or ("l" + "r" * (ncol - 1))
    # markdown
    md = [f"**Table {number}. {title}**", "", "| " + " | ".join(header) + " |",
          "|" + "|".join([":---"] + ["---:"] * (ncol - 1)) + "|"]
    for r in rows:
        md.append("| " + " | ".join(str(c) for c in r) + " |")
    md += ["", f"_{note}_", ""]
    open(os.path.join(TAB_DIR, name + ".md"), "w", encoding="utf-8").write("\n".join(md))
    # latex
    tex = ["\\begin{table}[ht]", "\\centering", "\\small", f"\\caption{{{esc_tex(title)}}}", f"\\label{{tab:{name}}}",
           f"\\begin{{tabular}}{{{align}}}", "\\toprule", " & ".join(esc_tex(h) for h in header) + " \\\\", "\\midrule"]
    for r in rows:
        tex.append(" & ".join(esc_tex(c) for c in r) + " \\\\")
    tex += ["\\bottomrule", "\\end{tabular}", f"\\par\\footnotesize {esc_tex(note)}", "\\end{table}", ""]
    open(os.path.join(TAB_DIR, name + ".tex"), "w", encoding="utf-8").write("\n".join(tex))
    print("saved", name, flush=True)


def table1():
    t = pd.read_csv(os.path.join(TAB_SRC, "paper_T1_corpus.csv"))
    years = {"13th": "1999 to 2004", "14th": "2004 to 2009", "15th": "2009 to 2014", "16th": "2014 to 2019",
             "17th": "2019 to 2024", "18th": "2024 to 2026", "Total": "1999 to 2026"}
    header = ["Lok Sabha", "Years", "Sessions", "Questions", "Starred", "Unstarred", "Parts", "Labelled questions",
              "Labelled (%)", "Labelled parts", "Gold questions", "Gold parts"]
    rows = []
    for r in t.itertuples(index=False):
        lsn = r[0]
        rows.append([lsn, years[lsn], f"{int(r[1])}", f"{int(r[2]):,}", f"{int(r[3]):,}", f"{int(r[4]):,}", f"{int(r[5]):,}",
                     f"{int(r[8]):,}", f"{r[9]:.1f}", f"{int(r[10]):,}", f"{int(r[13]):,}", f"{int(r[12]):,}"])
    note = ("Questions are written and oral questions with a ministerial answer. Parts are parsed question parts; a question "
            "without parsed parts counts as one part. Labelled means labelled by Gemini 3.8 Flash: all of LS17 and LS18, and a "
            "stratified random sample of about one in six questions in LS13 to LS16 (the total includes extra gold questions). "
            "Gold means parts coded by humans in the estimation strata, with the sanity stratum excluded. LS18 covers seven "
            "sessions and is still sitting.")
    write_table("table1_corpus", 1, "Corpus by Lok Sabha", header, rows, note, align="llrrrrrrrrrr")
    FACT["t1_total"] = rows[-1]


def _parse_ci_from(path, pat):
    txt = open(path, encoding="utf-8").read()
    m = re.search(pat, txt)
    assert m, pat
    return float(m.group(1)), float(m.group(2))


def table2():
    # ---------- human vs human
    human = pd.read_csv(os.path.join(C.CODING, "human_labels.csv"), keep_default_na=False)
    human = human[human["stage"] == "main"]
    h = V.double_coded_only(human)
    ms = pd.read_csv(os.path.join(C.CODING, "main_sample.csv"))
    est_q = set(ms.loc[ms["in_estimate"].astype(bool), "qid"])      # estimation strata only (sanity stratum excluded)
    w = V.human_wide(h, None, "label").dropna(subset=["coder1", "coder2"])
    w = w[w.index.str.rsplit("#", n=1).str[0].isin(est_q)]
    qid = w.index.str.rsplit("#", n=1).str[0]
    cats = V.ORD_LABELS
    Cm = V.confusion(w["coder1"], w["coder2"], cats)
    met = V.metrics_from_confusion(Cm)
    K = 3
    qc, G = V._cluster_codes(qid)
    per = np.zeros((G, K * K))
    li = {c: i for i, c in enumerate(cats)}
    for q, a, b in zip(qc, w["coder1"], w["coder2"]):
        per[q, li[a] * K + li[b]] += 1
    klo, khi = V.bootstrap_ci(per, lambda v: V.kappa_from_confusion(v.reshape(K, K)), B=1000)
    arr = np.array([list(w["coder1"]), list(w["coder2"])], dtype=object)
    a_h, alo, ahi = V.alpha_with_ci(arr, qid, cats, "ordinal", B=1000)
    assert abs(a_h - 0.752) < 0.001 and abs(alo - 0.653) < 0.002 and abs(ahi - 0.839) < 0.002, (a_h, alo, ahi)
    assert int(Cm.sum()) == 413 and abs(met["kappa"] - 0.708) < 0.001
    rows = {}
    rows["hh"] = ("Human coder 1 vs coder 2 (double coded)", int(Cm.sum()), 100 * met["accuracy"], met["kappa"], (klo, khi),
                  (a_h, alo, ahi), met["macro_f1"], None)

    # ---------- models vs gold
    g = E4.gold_frame()
    prod = pd.read_parquet(os.path.join(C.MEASURES, "m2_labels.parquet"))[["part_id", "label"]].rename(columns={"label": "l38"})
    alt = pd.read_parquet(os.path.join(C.MEASURES, "m2_gold_alt_3.1fl.parquet"))
    alt = alt[alt["status"] == "ok"][["part_id", "label"]].rename(columns={"label": "lalt"})
    d = g.merge(prod, on="part_id", how="inner").merge(alt, on="part_id", how="inner")
    assert len(d) == 1636
    kci38 = _parse_ci_from(os.path.join(C.RESULTS, "validation_report.md"), r"gemini-3\.8-flash \|[^\n]*\| \[(\d\.\d+), (\d\.\d+)\] \|")
    kcial = _parse_ci_from(os.path.join(EXP, "rq4_rq1.md"), r"\| 3\.1-flash-lite vs gold \| 1,636 \| [\d.]+ \| [\d.]+ \| \[(\d\.\d+), (\d\.\d+)\] \|")
    for key, col, lab, kci in (("m38", "l38", "Gemini 3.8 Flash vs human gold", kci38),
                               ("malt", "lalt", "Gemini 3.1 Flash Lite vs human gold", kcial)):
        Cx = V.confusion(d["label"], d[col], cats)
        mt = V.metrics_from_confusion(Cx)
        arr = np.array([list(d["label"]), list(d[col])], dtype=object)
        ax_, axlo, axhi = V.alpha_with_ci(arr, d["qid"].to_numpy(), cats, "ordinal", B=1000)
        rows[key] = (lab, int(Cx.sum()), 100 * mt["accuracy"], mt["kappa"], kci, (ax_, axlo, axhi), mt["macro_f1"], None)
    assert abs(rows["m38"][3] - 0.711) < 0.001 and abs(rows["malt"][3] - 0.570) < 0.001

    # ---------- source parity (LS17, 274 files with a text layer)
    pp = pd.read_csv(os.path.join(C.RESULTS, "parity_pairs.csv"))
    ok_q = set(pp.loc[~pp["file_answer_empty"].astype(bool), "qid"])
    fl = pd.read_parquet(os.path.join(C.MEASURES, "m2_parity_file.parquet"))[["part_id", "qid", "label"]].rename(columns={"label": "lfile"})
    pr = pd.read_parquet(os.path.join(C.MEASURES, "m2_labels.parquet"))[["part_id", "label"]].rename(columns={"label": "lapi"})
    pd_ = fl.merge(pr, on="part_id", how="inner")
    pd_ = pd_[pd_["qid"].isin(ok_q)]
    Cp = V.confusion(pd_["lapi"], pd_["lfile"], cats)
    mp = V.metrics_from_confusion(Cp)
    pj = json.load(open(os.path.join(C.RESULTS, "parity_m2.json")))["nonempty274"]
    assert int(Cp.sum()) == pj["n_parts_matched"] == 1110 and abs(mp["kappa"] - pj["label_kappa"]) < 1e-6, (Cp.sum(), mp["kappa"])
    arr = np.array([list(pd_["lapi"]), list(pd_["lfile"])], dtype=object)
    ap, aplo, aphi = V.alpha_with_ci(arr, pd_["qid"].to_numpy(), cats, "ordinal", B=1000)
    rows["par"] = ("Gemini 3.8 Flash, API text vs official file text (LS17, 274 files)", int(Cp.sum()), 100 * mp["accuracy"],
                   mp["kappa"], tuple(pj["label_kappa_ci"]), (ap, aplo, aphi), mp["macro_f1"], None)

    # ---------- embedding AUC
    t2 = pd.read_csv(os.path.join(TAB_SRC, "paper_T2_validity.csv"))
    r_none = t2[(t2["Metric"] == "AUC") & t2["Comparison"].str.contains("FULL vs NONE")].iloc[0]
    b4 = pd.read_csv(os.path.join(EXP, "b4_auc_by_type.csv")).set_index("gold group")
    n_full = 1038

    def b4row(g_):
        r_ = b4.loc[g_]
        lo_, hi_ = [float(v) for v in re.findall(r"[\d.]+", r_["95% CI"])]
        return int(r_["n parts"]) + n_full, (float(r_["AUC vs FULL (tuned)"]), lo_, hi_)
    emb = [("Embedding cosine (M1), full vs NONE parts", int(r_none["n parts"]), (r_none["Estimate"], r_none["CI lo"], r_none["CI hi"])),
           ("Embedding cosine (M1), full vs any partial or NONE part",) + b4row("any non-FULL"),
           ("Embedding cosine (M1), full vs not maintained parts",) + b4row("NOT_MAINTAINED")]
    # write
    def cell_ci(v):
        return fmt_ci(*v) if v else "n/a"
    header = ["Comparison", "n parts", "Agreement (%)", "κ [95% CI]", "α ordinal [95% CI]", "Macro F1", "AUC [95% CI]"]
    out = []
    for key in ("hh", "m38", "malt", "par"):
        lab, n, agr, kap, kci, a3, f1, _ = rows[key]
        out.append([lab, f"{n:,}", f"{agr:.1f}", f"{fmt(kap)} [{fmt(kci[0])}, {fmt(kci[1])}]", fmt_ci(*a3), fmt(f1), "n/a"])
    for lab, n, auc3 in emb:
        out.append([lab, f"{n:,}", "n/a", "n/a", "n/a", "n/a", fmt_ci(*auc3)])
    note = ("Rows compare labels of question parts (FULL, PARTIAL, NONE). The gold standard is the adjudicated human label on the "
            "main sample of 400 questions. Agreement is the share of identical labels; κ is Cohen's kappa; α is Krippendorff's alpha "
            "on the ordered scale; macro F1 averages the F1 of the three classes. Intervals are cluster bootstrap intervals over "
            "questions. Embedding rows give the AUC that a full answer has a higher cosine than a non answer: the first uses "
            "all valid gold parts, the last two use the held out parts that were not used to fine tune M1; 0.5 means no "
            "discrimination and values below 0.5 mean non answers score higher than full answers.")
    write_table("table2_validity", 2, "Measurement validity of the answer completeness scores", header, out, note,
                align="lrrrrrr")
    FACT["t2"] = out


def table3():
    reg = pd.read_csv(os.path.join(EXP, "rq2_t2_reg_ls.csv"))
    dec = pd.read_csv(os.path.join(EXP, "rq2_t2_reg_dec.csv"))
    y = lambda df: df[df["y"] == "y_m2"]  # noqa: E731
    s0 = y(reg)[y(reg)["spec"] == "S0 none"].iloc[0]
    s5 = y(reg)[y(reg)["spec"].str.startswith("S5 ")].iloc[0]
    d0 = y(dec)[y(dec)["spec"] == "S0 none"].iloc[0]
    d5 = y(dec)[y(dec)["spec"].str.startswith("S5 ")].iloc[0]
    pc = pd.read_csv(os.path.join(EXP, "rq2_t1_ppi_contrasts.csv"))
    ps = pd.read_csv(os.path.join(EXP, "rq2_t1_ppi_slope.csv"))
    ct = pd.read_csv(os.path.join(EXP, "rq2_t1_contrasts.csv"))
    sl = pd.read_csv(os.path.join(EXP, "rq2_t1_slopes.csv")).set_index("stat")
    ppi_c = num_ci(pc[(pc["method"] == "PPI++ (lambda tuned)") & (pc["contrast"] == "LS18 - LS13")].iloc[0]["diff"])
    ppi_s = num_ci(ps[ps["method"] == "PPI++ (lambda tuned)"].iloc[0]["slope per decade [95% CI]"])
    gold_c = num_ci(ct[(ct["contrast"] == "LS18 - LS13") & (ct["stat"] == "gold_mean")].iloc[0]["diff"])
    gs = sl.loc["gold slope /decade"]
    gold_s = (gs["est"], gs["lo"], gs["hi"])
    rows = [
        ["Raw LLM score", fmt_ci(s0["_b18"], s0["_lo18"], s0["_hi18"]), fmt_ci(d0["_b"], d0["_lo"], d0["_hi"]), "144,349 questions"],
        ["LLM score with controls", fmt_ci(s5["_b18"], s5["_lo18"], s5["_hi18"]), fmt_ci(d5["_b"], d5["_lo"], d5["_hi"]), "144,349 questions"],
        ["Bias corrected (PPI++)", fmt_ci(*ppi_c), fmt_ci(*ppi_s), "139,654 questions and 400 gold"],
        ["Human gold only", fmt_ci(*gold_c), fmt_ci(*gold_s), "400 gold questions"],
    ]
    header = ["Estimate", "LS18 minus LS13 [95% CI]", "Trend per decade [95% CI]", "Data"]
    note = ("Differences are in answer score units on the 0 to 1 scale; negative values mean lower completeness. The raw row uses "
            "design weighted LLM scores. The controls are question length, number of parts, starred status, request types, and "
            "ministry and topic fixed effects; intervals are clustered by MP and session. Bias corrected estimates combine all LLM "
            "labels with the human gold sample (prediction powered inference, PPI++). Gold only uses the 400 human labelled "
            "questions with design weights and a Rao Wu bootstrap; its trend interval includes zero.")
    write_table("table3_decline_estimates", 3, "Decline in answer completeness under four estimates", header, rows, note,
                align="lrrl")
    FACT["t3"] = rows


# ------------------------------------------------------------------------------------------------ captions
def captions():
    f1, f2, f3, f4, f5 = (FACT[k] for k in ("fig1", "fig2", "fig3", "fig4", "fig5"))
    nm_ci = f1["nm_ci"]
    p = f4["partner"]
    opp = f4["opp"]
    a_ = f3["all"]
    txt = f"""# Figure captions

**Figure 1.** Embedding cosine between question part and answer part (M1), by human gold evasion type, compared with full answers. Non answers marked "not maintained" score higher than full answers (median {f1['nm_med']:.2f} against {f1['full_med']:.2f}; AUC {f1['nm_auc']:.3f} {nm_ci}), so similarity cannot separate answers from refusals. Source: {f1['n']:,} held out human gold parts from 400 questions; boxes show quartiles, dots are parts, the dashed line is the median of full answers (n={f1['n_full']:,}), and groups with fewer than 10 parts are omitted.

**Figure 2.** Mean answer score by session, 1999 to 2026, with design weighted means and 95% confidence intervals from LLM labels of {f2['n_q']:,} questions in {f2['n_sessions']} sessions. Blue bands are bias corrected (PPI++) means by Lok Sabha, using {f2['n_ppi']:,} labelled questions and 400 human gold questions; red diamonds are the human gold means. Shading marks NDA and UPA governments, the dotted line marks the estimated kink at {f2['kap']:.1f} (95% interval {f2['klo']:.1f} to {f2['khi']:.1f}), and the dashed line is the flat then declining fit.

**Figure 3.** Parts not fully answered per 100 parts, by evasion type and Lok Sabha, from LLM labels of {f2['n_q']:,} questions with design weights; bands are 95% intervals. Deferral falls, while incomplete answers, State subject replies and not maintained replies rise, from {a_[0]+1e-9:.1f} to {a_[5]+1e-9:.1f} shortfalls per 100 parts overall. Types whose rise is not confirmed by the human labels, including generic answers, are folded into other types, shown without an interval.

**Figure 4.** Difference in answer score between opposition and ruling party askers within each Lok Sabha (coefficient and 95% interval, session, topic and ministry group fixed effects, standard errors clustered by MP and session), from {sum(f4['n']):,} labelled questions. The gap is near zero until LS16, then reaches {fmt(opp[4][0])} in LS17 and {fmt(opp[5][0])} in LS18. The red diamond shows MPs of coalition partners against BJP MPs in LS18 ({fmt(p[0])}, 95% interval {fmt(p[1])} to {fmt(p[2])}).

**Figure 5.** (a) Mean answer score for the {f5['n_min']} ministries with at least 2,000 labelled parts, ranked, with 95% intervals; the three security ministries are in red and the five highest and lowest ministries are labelled. (b) Share of parts answered with a State subject deflection by Lok Sabha, for ministries whose subjects lie mainly on the State or Concurrent List (health, agriculture, home, water, education) against Union ministries, with 95% intervals. Source: design weighted LLM labels, with ministry groups assigned by ministry name.
"""
    open(os.path.join(FIG_DIR, "CAPTIONS.md"), "w", encoding="utf-8").write(txt)


# ------------------------------------------------------------------------------------------------ dash check
BAD = ["-", "–", "—", "‐", "‑", "‒", "―"]


def md_visible_lines(path):
    out = []
    for ln in open(path, encoding="utf-8").read().splitlines():
        if re.fullmatch(r"\|[\s:|\-]+\|", ln.strip()):   # markdown table separator row (syntax, not text)
            continue
        out.append(ln)
    return out


def check():
    problems = []
    n_str = 0
    for nm, strs in FIG_TEXT.items():
        for s in strs:
            n_str += 1
            if any(b in s for b in BAD):
                problems.append(f"figure {nm}: {s!r}")
    n_files = 0
    for folder in (TAB_DIR, FIG_DIR):
        for f in sorted(os.listdir(folder)):
            if f.endswith((".md", ".tex")):
                n_files += 1
                for ln in md_visible_lines(os.path.join(folder, f)):
                    if any(b in ln for b in BAD):
                        problems.append(f"{f}: {ln[:140]!r}")
    print(f"DASH CHECK: scanned {n_str} figure strings over {len(FIG_TEXT)} figures and {n_files} table or caption files "
          f"(markdown separator rows excluded): {'PASS' if not problems else 'FAIL'}")
    for p in problems:
        print("  ", p)
    return problems


def main():
    figure1()
    figure2()
    figure3()
    figure4()
    figure5()
    table1()
    table2()
    table3()
    captions()
    probs = check()
    if probs:
        sys.exit(1)


if __name__ == "__main__":
    main()
