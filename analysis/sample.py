"""M3 sampling: draw the PILOT (50 questions) and MAIN (400 questions + 20-question sanity stratum) samples.

    python -I -X utf8 analysis/sample.py                 -> data/coding/pilot_sample.csv, main_sample.csv, sample_meta.json
    python -I -X utf8 analysis/sample.py --force         -> overwrite existing samples (re-draw)
    python -I -X utf8 analysis/sample.py --main-only --force
                                                         -> keep the existing pilot_sample.csv untouched (its qids are
                                                            excluded from the main universe), re-draw main only
    python -I -X utf8 analysis/sample.py --reweight      -> keep the drawn qids, recompute weights against the
                                                            current corpus (use after the corpus is completed)

Design (all draws are at the QUESTION level; coders code every part of a sampled question):
  * Universe  = questions with a question text, an answer text and <= MAX_PARTS parts (counts of what is dropped
                are reported and stored in sample_meta.json).
  * Main strata = Lok Sabha (13-18) x qtype (starred/unstarred) x ministry group (security/other) x risk, where
                risk = boilerplate-flagged answer (oversampled to BOILER_SHARE of the sample) or, for the
                non-boilerplate part, the quintile of the question's mean M1 score (only when m1_scores.parquet
                exists; otherwise a single 'uniform' bin).
  * LS oversampling (--ls-share "17,18:0.45", default): the listed Lok Sabhas get `share` of the MAIN estimate
                questions combined, split evenly across the listed LS (180 = 90 + 90 of 400); the remaining
                questions (220) go to the other LS (13-16). The boilerplate oversample (25%) is applied within each
                budget group (listed LS separately, others together). Pass "none" for the old single-pool design.
                Weights stay exact because N_stratum/n_stratum is computed per LS-containing stratum.
  * M1 strata: the non-boilerplate part of each cell is stratified by the quintile of the QUESTION-level M1 score
                (m1_part_mean = mean cosine over the question's valid parts, data/measures/m1_question.parquet; falls
                back to the mean of valid part scores in m1_scores.parquet). Quintiles are computed within
                ls x qtype x ministry_group among non-boilerplate questions of the main universe (cells < 50
                questions: one bin). If a cell's allocation k is smaller than the number of quintile bins, adjacent
                quintiles are merged into k bins (stratum label |q123 etc.), so every bin is sampled and weights stay
                exact. Questions with no valid M1 score form their own `m1_missing|<ls>` stratum per budget group.
  * Allocation: n_h proportional to N_h**ALLOC_POWER (0.5 = square-root allocation, so rare cells such as
                starred questions are not starved), floor MIN_PER_CELL, capped at N_h.
  * Pilot is drawn first (strata Lok Sabha x boilerplate flag, proportional-to-sqrt allocation); the main sample
    is drawn from the universe minus the pilot, so the two are disjoint. Main weights are conditional on the pilot:
    weight = (N_h - pilot_h) / n_h.
  * weight = inverse inclusion probability = (stratum population size) / (stratum sample size). Part-level
    estimates: give every part its question's weight and use a ratio estimator
    (sum w_q * sum_parts y) / (sum w_q * n_parts).
  * Sanity stratum: SANITY_N questions whose answer text is missing (still <= MAX_PARTS parts); flagged
    in_estimate=False and excluded from all estimates.
  * Coder assignment (main): DOUBLE_N of the MAIN estimate questions are double-coded (coder1;coder2; a stratified
                random subsample, strata = LS group {17,18} vs {13-16} x boilerplate flag), the rest are single-coded,
                split evenly between coder1 and coder2 within strata. Recorded in `coders` and `double_coded`
                (the sanity questions are coded by both: coders = "coder1;coder2", double_coded = True, in_estimate
                False). Pilot: both coders code everything.
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

SEED = 20261017
PILOT_N = 50
MAIN_N = 400
DOUBLE_N = 100
SANITY_N = 20
MAX_PARTS = 6
BOILER_SHARE = 0.25
ALLOC_POWER = 0.5
MIN_PER_CELL = 2
CODERS = ("coder1", "coder2")
LS_SHARE = "17,18:0.45"
SECURITY_PATTERNS = ("HOME AFFAIRS", "DEFENCE", "EXTERNAL AFFAIRS")

QFILE = os.path.join("data", "final", "questions_analysis.parquet")
PFILE = os.path.join("data", "measures", "parts.parquet")
M1FILE = os.path.join("data", "measures", "m1_scores.parquet")
M1QFILE = os.path.join("data", "measures", "m1_question.parquet")
OUT_COLS = ["qid", "sample", "stratum", "ls", "qtype", "ministry", "ministry_group", "boilerplate_flag",
            "m1_quintile", "n_parts", "in_estimate", "N_stratum", "n_stratum", "weight", "coders", "seed"]
MAIN_COLS = OUT_COLS + ["double_coded"]  # the pilot csv keeps the original OUT_COLS


# ------------------------------------------------------------------------------------------------ universe
def provisional_ministry_group(ministry):
    m = str(ministry).upper()
    return "security" if any(p in m for p in SECURITY_PATTERNS) else "other"


def load_universe(root=ROOT, qfile=QFILE, pfile=PFILE, m1file=M1FILE, m1qfile=M1QFILE):
    """One row per question: qid, ls, qtype, ministry, ministry_group, boilerplate_flag, n_parts, no_answer,
    no_question, m1_mean = question-level M1 score (NaN when unavailable for the question)."""
    qpath = os.path.join(root, qfile)
    cols = ["qid", "ls", "qtype", "ministry", "boilerplate", "answer_missing", "question_missing",
            "qa_mismatch", "answer_garbled"]
    schema = pq.read_schema(qpath).names
    if "ministry_group" in schema:
        cols.append("ministry_group")
    q = pd.read_parquet(qpath, columns=cols)
    if "ministry_group" in q.columns:  # 8-group column in the data -> sampling uses security vs other
        q["ministry_group_raw"] = q["ministry_group"].astype(str)
        q["ministry_group"] = np.where(q["ministry_group_raw"] == "security", "security", "other")
    else:
        q["ministry_group"] = q["ministry"].map(provisional_ministry_group)
    q["boilerplate_flag"] = q["boilerplate"].map(lambda b: b is not None and len(b) > 0)
    q = q.drop(columns="boilerplate")
    q["no_answer"] = q["answer_missing"].fillna(True).astype(bool)
    q["no_question"] = q["question_missing"].fillna(False).astype(bool)
    q["bad_quality"] = q["qa_mismatch"].fillna(False).astype(bool) | q["answer_garbled"].fillna(False).astype(bool)
    q = q.drop(columns=["answer_missing", "question_missing", "qa_mismatch", "answer_garbled"])
    p = pd.read_parquet(os.path.join(root, pfile), columns=["qid", "n_parts"])
    npart = p.groupby("qid")["n_parts"].first().rename("n_parts")
    q = q.merge(npart, left_on="qid", right_index=True, how="left")
    q["n_parts"] = q["n_parts"].fillna(0).astype(int)
    q["m1_mean"] = np.nan
    m1qp, m1p = os.path.join(root, m1qfile), os.path.join(root, m1file)
    m1_source = "none"
    if os.path.exists(m1qp):  # question-level score: m1_part_mean (mean cosine over valid parts)
        m1q = pd.read_parquet(m1qp, columns=["qid", "m1_part_mean"])
        q = q.drop(columns="m1_mean").merge(m1q.set_index("qid")["m1_part_mean"].rename("m1_mean"),
                                           left_on="qid", right_index=True, how="left")
        m1_source = "m1_question.parquet:m1_part_mean"
    elif os.path.exists(m1p):
        cols = ["part_id", "m1_cos"] + (["m1_valid"] if "m1_valid" in pq.read_schema(m1p).names else [])
        m1 = pd.read_parquet(m1p, columns=cols)
        if "m1_valid" in m1.columns:
            m1 = m1[m1["m1_valid"].fillna(False).astype(bool)]
        m1["qid"] = m1["part_id"].str.rsplit("#", n=1).str[0]
        q = q.drop(columns="m1_mean").merge(m1.groupby("qid")["m1_cos"].mean().rename("m1_mean"),
                                           left_on="qid", right_index=True, how="left")
        m1_source = "m1_scores.parquet:mean of valid part m1_cos"
    q = q.sort_values("qid").reset_index(drop=True)
    q.attrs["m1_source"] = m1_source
    q.attrs["ministry_group_source"] = ("ministry_group column" if "ministry_group" in schema else
                                        "provisional (HOME AFFAIRS/DEFENCE/EXTERNAL AFFAIRS = security)")
    return q


def eligibility(u, max_parts=MAX_PARTS):
    """Masks + exclusion report. Estimate universe: question and answer present, 1..max_parts parts."""
    base = (~u["no_question"]) & (u["n_parts"] >= 1)
    bad = u["bad_quality"] if "bad_quality" in u else pd.Series(False, index=u.index)
    est = base & (~u["no_answer"]) & (u["n_parts"] <= max_parts) & (~bad)
    san = base & u["no_answer"] & (u["n_parts"] <= max_parts)
    ans = base & (~u["no_answer"])
    rep = {
        "n_questions_total": int(len(u)),
        "n_no_question_or_no_parts": int((~base).sum()),
        "n_no_answer": int((base & u["no_answer"]).sum()),
        "n_answered": int(ans.sum()),
        "n_excluded_qa_mismatch_or_garbled": int((ans & bad).sum()),
        "n_excluded_qa_mismatch_or_garbled_with_le_max_parts": int((ans & bad & (u["n_parts"] <= max_parts)).sum()),
        "n_answered_gt_max_parts": int((ans & (u["n_parts"] > max_parts)).sum()),
        "share_answered_excluded_gt_max_parts": float((ans & (u["n_parts"] > max_parts)).sum() / max(ans.sum(), 1)),
        "share_answered_parts_excluded_gt_max_parts": float(
            u.loc[ans & (u["n_parts"] > max_parts), "n_parts"].sum() / max(u.loc[ans, "n_parts"].sum(), 1)),
        "n_estimate_universe": int(est.sum()),
        "n_sanity_universe": int(san.sum()),
        "max_parts": max_parts,
    }
    return est, san, rep


# ------------------------------------------------------------------------------------------------ allocation
def allocate(sizes, n_total, power=ALLOC_POWER, min_per=MIN_PER_CELL):
    """Integer allocation n_h ~ N_h**power summing to min(n_total, sum N_h); n_h in [min(min_per, N_h), N_h].
    Water-filling for the caps, largest-remainder rounding. Deterministic (ties broken by index order)."""
    sizes = pd.Series(sizes, dtype=float)
    sizes = sizes[sizes > 0]
    if sizes.empty or n_total <= 0:
        return pd.Series(dtype=int)
    n_total = int(min(n_total, sizes.sum()))
    lo = np.minimum(min_per, sizes.values)
    if lo.sum() > n_total:  # too many cells for the budget: drop the floor, keep proportionality
        lo = np.zeros(len(sizes))
    hi = sizes.values
    w = sizes.values ** power
    alloc = np.zeros(len(sizes))
    fixed = np.zeros(len(sizes), dtype=bool)
    for _ in range(len(sizes) + 2):
        free = ~fixed
        remaining = n_total - alloc[fixed].sum()
        a = remaining * w[free] / w[free].sum()
        full = np.zeros(len(sizes))
        full[free] = a
        viol_hi = free & (full > hi)
        viol_lo = free & (full < lo)
        if not viol_hi.any() and not viol_lo.any():
            alloc[free] = a
            break
        # fix the worst-violating side first (caps before floors) and iterate
        if viol_hi.any():
            alloc[viol_hi] = hi[viol_hi]
            fixed |= viol_hi
        else:
            alloc[viol_lo] = lo[viol_lo]
            fixed |= viol_lo
    out = np.floor(alloc).astype(int)
    out = np.maximum(out, lo.astype(int))
    out = np.minimum(out, hi.astype(int))
    short = n_total - out.sum()
    if short > 0:
        frac = alloc - np.floor(alloc)
        room = out < hi
        order = [i for i in np.argsort(-frac, kind="stable") if room[i]]
        k = 0
        while short > 0 and order:
            i = order[k % len(order)]
            if out[i] < hi[i]:
                out[i] += 1
                short -= 1
            k += 1
            if k > 10 * len(order) + short:
                break
    elif short < 0:  # floors pushed the total over budget: trim the largest cells above their floor
        for i in np.argsort(-out, kind="stable"):
            while short < 0 and out[i] > lo[i]:
                out[i] -= 1
                short += 1
    return pd.Series(out, index=sizes.index)


def _draw(rng, qids, n):
    qids = np.asarray(sorted(qids))
    n = int(min(n, len(qids)))
    return list(rng.choice(qids, size=n, replace=False)) if n > 0 else []


def m1_quintiles(u, mask):
    """Quintile (1..5) of question mean-M1 within ls x qtype x ministry_group among non-boilerplate questions in
    `mask`; cells with < 50 questions or missing M1 get 0 (= single bin). Returns an int Series aligned to u."""
    out = pd.Series(0, index=u.index, dtype=int)
    if u["m1_mean"].notna().sum() == 0:
        return out
    sub = u[mask & (~u["boilerplate_flag"]) & u["m1_mean"].notna()]
    for _, g in sub.groupby(["ls", "qtype", "ministry_group"]):
        if len(g) < 50:
            continue
        ranks = g["m1_mean"].rank(method="first")
        out.loc[g.index] = pd.qcut(ranks, 5, labels=False).astype(int) + 1
    return out


def parse_ls_share(spec):
    """'17,18:0.45' -> ([17, 18], 0.45); None / '' / 'none' -> ([], 0.0)."""
    if spec is None or str(spec).strip().lower() in ("", "none", "off"):
        return [], 0.0
    if isinstance(spec, (tuple, list)):
        return [int(x) for x in spec[0]], float(spec[1])
    try:
        lss, share = str(spec).split(":")
        ls = [int(x) for x in lss.split(",") if x.strip()]
        share = float(share)
    except ValueError:
        raise ValueError(f"bad --ls-share {spec!r}; expected e.g. '17,18:0.45'")
    if not ls or not 0 < share < 1:
        raise ValueError(f"bad --ls-share {spec!r}: need >=1 Lok Sabha and 0 < share < 1")
    return ls, share


def _split(total, weights):
    """Largest-remainder split of an integer total proportional to weights (deterministic)."""
    w = np.asarray(weights, dtype=float)
    raw = total * w / w.sum()
    out = np.floor(raw).astype(int)
    for i in np.argsort(-(raw - out), kind="stable")[: int(total - out.sum())]:
        out[i] += 1
    return out


# ------------------------------------------------------------------------------------------------ main logic
def draw_pilot(u_est, rng, n=PILOT_N, boiler_share=BOILER_SHARE):
    """Pilot: strata ls x boilerplate_flag; ~boiler_share of the pilot is boilerplate-flagged."""
    n_b = int(round(n * boiler_share))
    rows = []
    for flag, n_flag in ((True, n_b), (False, n - n_b)):
        sub = u_est[u_est["boilerplate_flag"] == flag]
        sizes = sub.groupby("ls").size()
        al = allocate(sizes, n_flag, min_per=1)
        for ls, k in al.items():
            pool = sub.loc[sub["ls"] == ls, "qid"]
            for qid in _draw(rng, pool, k):
                rows.append((qid, f"ls{ls}|{'boiler' if flag else 'other'}", int(sizes[ls]), int(k)))
    d = pd.DataFrame(rows, columns=["qid", "stratum", "N_stratum", "n_stratum"])
    return d


def m1_missing_mask(u):
    """Non-boilerplate questions without a valid M1 score (only when M1 scores exist for the universe at all)."""
    if u["m1_mean"].notna().sum() == 0:
        return pd.Series(False, index=u.index)
    return u["m1_mean"].isna() & (~u["boilerplate_flag"])


def draw_main(u_est, rng, n=MAIN_N, boiler_share=BOILER_SHARE, power=ALLOC_POWER, min_per=MIN_PER_CELL,
              ls_share=LS_SHARE):
    """Main: budget groups (each listed oversampled LS on its own, all other LS pooled) x
    (ls, qtype, ministry_group) cells x {boilerplate | non-boilerplate x M1 quintile}, plus one `m1_missing`
    stratum per budget group for non-boilerplate questions without a valid M1 score."""
    u = u_est.copy()
    u["m1_miss"] = m1_missing_mask(u)
    u["m1_q"] = m1_quintiles(u, ~u["m1_miss"])
    over, share = parse_ls_share(ls_share)
    over = [x for x in over if (u["ls"] == x).any()]
    all_ls = sorted(u["ls"].unique())
    rest_ls = [x for x in all_ls if x not in over]
    if over and rest_ls:
        n_over = int(round(n * share))
        parts = _split(n_over, [1] * len(over))
        groups = [([ls], int(k)) for ls, k in zip(over, parts)]
        groups.append((rest_ls, n - n_over))
        # a listed LS cannot give more than it has: move the shortfall to the pooled group
        for i, (lsg, k) in enumerate(groups[:-1]):
            cap = int((u["ls"].isin(lsg)).sum())
            if k > cap:
                groups[-1] = (rest_ls, groups[-1][1] + k - cap)
                groups[i] = (lsg, cap)
    else:
        groups = [(all_ls, n)]
    n_b_total = int(round(n * boiler_share))
    b_split = _split(n_b_total, [k for _, k in groups])
    cells = ["ls", "qtype", "ministry_group"]
    rows = []
    for (lsg, n_g), n_b in zip(groups, b_split):
        n_b = min(int(n_b), n_g)
        ug = u[u["ls"].isin(lsg)]
        miss = ug[ug["m1_miss"]]
        k_miss = 1 if len(miss) > 0 and n_g - n_b > 1 else 0
        if k_miss:
            for qid in _draw(rng, miss["qid"], 1):
                rows.append((qid, "m1_missing|ls" + ",".join(str(x) for x in lsg), int(len(miss)), 1, 0))
        for flag, n_flag in ((True, n_b), (False, n_g - n_b - k_miss)):
            sub = ug[(ug["boilerplate_flag"] == flag) & (~ug["m1_miss"])]
            sizes = sub.groupby(cells).size()
            al = allocate(sizes, n_flag, power=power, min_per=min_per)
            for key, k in al.items():
                ls, qt, mg = key
                cell = sub[(sub["ls"] == ls) & (sub["qtype"] == qt) & (sub["ministry_group"] == mg)]
                qof = dict(zip(cell["qid"], cell["m1_q"]))
                base = f"ls{ls}|{qt}|{mg}|{'boiler' if flag else 'other'}"
                qvals = sorted(cell["m1_q"].unique())
                k = int(k)
                if flag or qvals == [0] or k < 1:
                    bins = [("", cell, k)]
                else:
                    nb = len(qvals)
                    if k >= nb:  # one bin per quintile, every bin sampled
                        kk = allocate(cell.groupby("m1_q").size(), k, power=1.0, min_per=1)
                        grp = [([int(qv)], int(kk[qv])) for qv in qvals]
                    else:  # fewer draws than quintiles: merge adjacent quintiles into k bins of one draw each
                        grp = [([int(x) for x in g], 1) for g in np.array_split(np.array(qvals), k)]
                    bins = [("|q" + "".join(str(x) for x in g) if len(g) < nb else "",
                             cell[cell["m1_q"].isin(g)], kb) for g, kb in grp]
                for suf, bc, kb in bins:
                    for qid in _draw(rng, bc["qid"], kb):
                        rows.append((qid, base + suf, int(len(bc)), int(kb), int(qof[qid])))
    return pd.DataFrame(rows, columns=["qid", "stratum", "N_stratum", "n_stratum", "m1_quintile"])


def assign_coders(df, seed=SEED, double_n=DOUBLE_N, coders=CODERS):
    """Set `coders` and `double_coded` on a main-sample table (estimate rows get the double/single assignment,
    sanity rows are coded by both). Double subset = stratified random subsample of the estimate questions
    (strata: LS group {17,18} vs {13-16} x boilerplate flag; proportional allocation, largest remainder);
    the single-coded remainder is split evenly between the two coders within strata."""
    if len(coders) != 2:
        raise ValueError("assign_coders needs exactly two coders")
    df = df.copy()
    df["coders"] = ";".join(coders)
    df["double_coded"] = True
    est = df[df["in_estimate"]]
    rng = np.random.default_rng([seed, 4])
    grp = np.where(est["ls"].isin([17, 18]), "ls17-18", "ls13-16") + "|" + np.where(est["boilerplate_flag"], "boiler", "other")
    strata = {k: sorted(est.index[grp == k]) for k in sorted(set(grp))}
    sizes = np.array([len(v) for v in strata.values()])
    double_n = int(min(double_n, sizes.sum()))
    nd = np.minimum(_split(double_n, sizes), sizes) if double_n > 0 else np.zeros(len(sizes), dtype=int)
    short = double_n - int(nd.sum())  # only if a cap bound (tiny strata)
    for i in np.argsort(-(sizes - nd), kind="stable"):
        add = int(min(short, sizes[i] - nd[i]))
        nd[i] += add
        short -= add
    toggle = 0
    for (k, idx), n_d in zip(strata.items(), nd):
        idx = [int(x) for x in rng.permutation(np.array(idx))]
        sgl = idx[int(n_d):]
        df.loc[sgl, "double_coded"] = False
        c1_n = len(sgl) // 2 + ((len(sgl) % 2) if toggle == 0 else 0)
        if len(sgl) % 2:
            toggle ^= 1
        df.loc[sgl[:c1_n], "coders"] = coders[0]
        df.loc[sgl[c1_n:], "coders"] = coders[1]
    return df


def decorate(d, u, sample, in_estimate, seed, coders=CODERS):
    meta = u.set_index("qid")[["ls", "qtype", "ministry", "ministry_group", "boilerplate_flag", "n_parts"]]
    d = d.merge(meta, left_on="qid", right_index=True, how="left")
    d["sample"] = sample
    d["in_estimate"] = in_estimate
    d["weight"] = d["N_stratum"] / d["n_stratum"]
    d["coders"] = ";".join(coders)
    d["seed"] = seed
    if "m1_quintile" not in d.columns:
        d["m1_quintile"] = 0
    d["m1_quintile"] = d["m1_quintile"].fillna(0).astype(int)
    return d[OUT_COLS].sort_values(["ls", "qtype", "stratum", "qid"]).reset_index(drop=True)


def run(root=ROOT, seed=SEED, pilot_n=PILOT_N, main_n=MAIN_N, sanity_n=SANITY_N, max_parts=MAX_PARTS,
        boiler_share=BOILER_SHARE, force=False, out_dir=None, universe=None, ls_share=LS_SHARE, double_n=DOUBLE_N,
        keep_pilot=False):
    """Draw pilot + main. keep_pilot=True: do not draw or write the pilot; read the existing pilot_sample.csv and
    draw main from the universe minus those qids (main_sample.csv and sample_meta.json are rewritten)."""
    out_dir = out_dir or os.path.join(root, "data", "coding")
    os.makedirs(out_dir, exist_ok=True)
    pp, mp, meta_p = (os.path.join(out_dir, f) for f in ("pilot_sample.csv", "main_sample.csv", "sample_meta.json"))
    if keep_pilot:
        if not os.path.exists(pp):
            raise SystemExit(f"--main-only needs an existing {pp}")
        if os.path.exists(mp) and not force:
            raise SystemExit(f"{mp} already exists; coding may have started. Use --force to re-draw main.")
    elif (os.path.exists(pp) or os.path.exists(mp)) and not force:
        raise SystemExit(f"{pp} / {mp} already exist; coding may have started. Use --force to re-draw "
                         f"or --reweight to only refresh the weights.")
    u = universe if universe is not None else load_universe(root)
    est, san, rep = eligibility(u, max_parts)
    u_est, u_san = u[est], u[san]

    if keep_pilot:
        pilot_df = pd.read_csv(pp)
        pilot = pilot_df[["qid"]]
        gone = ~pilot["qid"].isin(set(u_est["qid"]))
        if gone.any():
            print(f"warning: {int(gone.sum())} pilot qids are not in the eligible estimate universe")
    else:
        rng_pilot = np.random.default_rng([seed, 1])
        pilot = draw_pilot(u_est, rng_pilot, pilot_n, boiler_share)
        pilot_df = decorate(pilot, u, "pilot", True, seed)

    rest = u_est[~u_est["qid"].isin(pilot["qid"])]
    rng_main = np.random.default_rng([seed, 2])
    main = draw_main(rest, rng_main, main_n, boiler_share, ls_share=ls_share)
    main_df = decorate(main, u, "main", True, seed)

    rng_san = np.random.default_rng([seed, 3])
    sq = _draw(rng_san, u_san["qid"], sanity_n)
    san_df = pd.DataFrame({"qid": sq, "stratum": "sanity_no_answer", "N_stratum": len(u_san),
                           "n_stratum": len(sq)})
    san_df = decorate(san_df, u, "main", False, seed)
    main_df = pd.concat([main_df, san_df], ignore_index=True)
    main_df = assign_coders(main_df, seed, double_n)[MAIN_COLS]

    assert not set(pilot_df["qid"]) & set(main_df["qid"]), "pilot and main overlap"
    assert pilot_df["qid"].is_unique and main_df["qid"].is_unique
    if not keep_pilot:
        pilot_df.to_csv(pp, index=False)
    main_df.to_csv(mp, index=False)
    me = main_df[main_df["in_estimate"]]
    meta = {
        "seed": seed, "pilot_n": len(pilot_df), "main_estimate_n": int(main_df["in_estimate"].sum()),
        "main_sanity_n": int((~main_df["in_estimate"]).sum()), "boiler_share_target": boiler_share,
        "pilot_kept_from_existing_file": bool(keep_pilot),
        "ls_share": ls_share, "ls_share_parsed": {"ls": parse_ls_share(ls_share)[0], "share": parse_ls_share(ls_share)[1]},
        "main_estimate_n_by_ls": {int(k): int(v) for k, v in me["ls"].value_counts().sort_index().items()},
        "double_n_target": int(double_n),
        "coder_assignment_estimate": {
            "double_coded": int(me["double_coded"].sum()),
            "single_coder1": int((me["coders"] == CODERS[0]).sum()),
            "single_coder2": int((me["coders"] == CODERS[1]).sum()),
            "sanity_coded_by_both": int((~main_df["in_estimate"]).sum())},
        "alloc_power": ALLOC_POWER, "min_per_cell": MIN_PER_CELL, "coders": list(CODERS),
        "m1_available": bool(u["m1_mean"].notna().any()),
        "m1_score_source": u.attrs.get("m1_source", "universe supplied by caller"),
        "n_m1_missing_nonboiler_universe": int(m1_missing_mask(u_est).sum()),
        "ministry_group_source": u.attrs.get("ministry_group_source", "universe supplied by caller"),
        "universe_qid_hash": hashlib.sha256("\n".join(u_est["qid"]).encode()).hexdigest()[:16],
        "exclusions": rep,
        "note": "Weights are w.r.t. the corpus snapshot identified by universe_qid_hash; run --reweight after the "
                "corpus changes.",
    }
    with open(meta_p, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return pilot_df, main_df, meta


def reweight(root=ROOT, out_dir=None, max_parts=MAX_PARTS, universe=None):
    """Recompute N_stratum and weights for already-drawn qids against the CURRENT universe, keeping the
    sample. Strata are the stored `stratum` labels (ls|qtype|group|risk[|q<digits>] and m1_missing|ls<list>);
    M1-quintile strata need the quintile bins recomputed from the current M1 scores, so quintile boundaries may
    shift slightly."""
    out_dir = out_dir or os.path.join(root, "data", "coding")
    u = universe if universe is not None else load_universe(root)
    est, san, rep = eligibility(u, max_parts)
    ue = u[est].copy()
    pilot = pd.read_csv(os.path.join(out_dir, "pilot_sample.csv"))
    main = pd.read_csv(os.path.join(out_dir, "main_sample.csv"))
    pilot_ids = set(pilot["qid"])

    def stratum_key(df):
        return ("ls" + df["ls"].astype(str) + "|" + df["qtype"] + "|" + df["ministry_group"] + "|" +
                np.where(df["boilerplate_flag"], "boiler", "other"))

    ue["_skey"] = stratum_key(ue)
    ue["_pkey"] = "ls" + ue["ls"].astype(str) + "|" + np.where(ue["boilerplate_flag"], "boiler", "other")
    # pilot
    Np = ue.groupby("_pkey").size()
    pilot["N_stratum"] = pilot["stratum"].map(Np)
    pilot["n_stratum"] = pilot.groupby("stratum")["qid"].transform("size")
    pilot["weight"] = pilot["N_stratum"] / pilot["n_stratum"]
    # main: strata exclude the pilot
    rest = ue[~ue["qid"].isin(pilot_ids)].copy()
    rest["m1_miss"] = m1_missing_mask(rest)
    rest["m1_q"] = m1_quintiles(rest, ~rest["m1_miss"])  # same bins as draw_main
    m = main["in_estimate"]
    est_rows = main[m].copy()
    est_rows["_digs"] = est_rows["stratum"].str.extract(r"\|q(\d+)$")[0]
    est_rows["_base"] = est_rows["stratum"].str.replace(r"\|q\d+$", "", regex=True)
    rm = rest[~rest["m1_miss"]]
    N_cell = rm.groupby("_skey").size()
    N_q = rm.groupby(["_skey", "m1_q"]).size()

    def n_of(stratum, base, digs):
        if stratum.startswith("m1_missing|ls"):
            lss = [int(x) for x in stratum.split("|ls")[1].split(",")]
            return int((rest["m1_miss"] & rest["ls"].isin(lss)).sum())
        if isinstance(digs, str):
            return sum(N_q.get((base, int(d)), 0) for d in digs)
        return N_cell.get(base, np.nan)

    est_rows["N_stratum"] = [n_of(s_, b, d) for s_, b, d in
                             zip(est_rows["stratum"], est_rows["_base"], est_rows["_digs"])]
    est_rows["n_stratum"] = est_rows.groupby("stratum")["qid"].transform("size")
    est_rows["weight"] = est_rows["N_stratum"] / est_rows["n_stratum"]
    san_rows = main[~m].copy()
    san_rows["N_stratum"] = int(san.sum())
    san_rows["n_stratum"] = len(san_rows)
    san_rows["weight"] = san_rows["N_stratum"] / san_rows["n_stratum"].clip(lower=1)
    cols = MAIN_COLS if "double_coded" in main.columns else OUT_COLS
    main2 = pd.concat([est_rows.drop(columns=["_digs", "_base"]), san_rows], ignore_index=True)[cols]
    still = ~main2["qid"].isin(set(ue["qid"]) | set(u[san]["qid"]))
    if still.any():
        print(f"warning: {int(still.sum())} sampled qids are no longer in the eligible universe")
    pilot[OUT_COLS].to_csv(os.path.join(out_dir, "pilot_sample.csv"), index=False)
    main2.to_csv(os.path.join(out_dir, "main_sample.csv"), index=False)
    return pilot, main2, rep


def composition_report(df):
    est = df[df["in_estimate"]]
    lines = [f"rows={len(df)}  in_estimate={len(est)}  sanity={len(df) - len(est)}",
             f"boilerplate share (estimate rows) = {est['boilerplate_flag'].mean():.3f}",
             f"sum of weights (estimate rows) = {est['weight'].sum():,.0f}",
             f"weight min/median/max = {est['weight'].min():.1f} / {est['weight'].median():.1f} / {est['weight'].max():.1f}",
             f"parts to code (estimate rows) = {int(est['n_parts'].sum())}; with sanity = {int(df['n_parts'].sum())}"]
    if "double_coded" in df:
        lines.append("coding (estimate rows): double=%d, coder1 only=%d, coder2 only=%d" % (
            int(est["double_coded"].sum()), int((est["coders"] == "coder1").sum()),
            int((est["coders"] == "coder2").sum())))
    for c in ("ls", "qtype", "ministry_group", "boilerplate_flag", "m1_quintile"):
        lines.append(f"{c}: " + ", ".join(f"{k}={v}" for k, v in df[c].value_counts().sort_index().items()))
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--reweight", action="store_true")
    ap.add_argument("--ls-share", default=LS_SHARE,
                    help="Lok Sabhas to oversample in MAIN and their combined share, e.g. '17,18:0.45' "
                         "(split evenly across the listed LS); 'none' = single pool")
    ap.add_argument("--max-parts", type=int, default=MAX_PARTS)
    ap.add_argument("--main-n", type=int, default=MAIN_N, help="MAIN estimate questions (excl. sanity)")
    ap.add_argument("--double-n", type=int, default=DOUBLE_N, help="MAIN estimate questions coded by both coders")
    ap.add_argument("--main-only", action="store_true",
                    help="keep the existing pilot_sample.csv untouched; (re-)draw main only (needs --force if main exists)")
    a = ap.parse_args(argv)
    if a.reweight:
        pilot, main_df, rep = reweight(a.root, max_parts=a.max_parts)
    else:
        pilot, main_df, meta = run(a.root, seed=a.seed, max_parts=a.max_parts, force=a.force,
                                ls_share=a.ls_share, main_n=a.main_n, double_n=a.double_n,
                                   keep_pilot=a.main_only)
        rep = meta["exclusions"]
    print(json.dumps(rep, indent=2))
    print("PILOT\n" + composition_report(pilot))
    print("MAIN\n" + composition_report(main_df))
    return 0


if __name__ == "__main__":
    sys.exit(main())
