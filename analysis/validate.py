"""M3 validation: inter-coder reliability, adjudication/gold labels, M2 and M1 validation against gold.

    python -I -X utf8 analysis/validate.py adjudication --stage main   # list disagreements -> adjudication.csv
    python -I -X utf8 analysis/validate.py gold --stage main           # gold_labels.csv (agreed / adjudicated / single)
    python -I -X utf8 analysis/validate.py report --stage main --threshold 0.6 \
        --cost gemini-flash-lite=1.2 --cost gemini-flash=6.5           # results/validation_report.md

Design (main stage): 100 of the 400 estimate questions are double-coded, the other 300 single-coded (150 per
coder), the 20 sanity questions are coded by both. Reliability uses ONLY parts coded by both coders (primary:
the double-coded estimate questions; the sanity questions are reported separately). Gold label per part:
agreed (double-coded, coders agree), adjudicated (double-coded, they disagree; value from adjudication.csv), or
single (the one coder's label); gold_labels.csv records this in `gold_source`. A double-coded part (per the sample's
`coders` column) for which one coder's label has not been imported yet is counted as pending, not as single.

Inputs (any may be missing; the report then says so):
  data/coding/human_labels.csv     (coding_sheet.py import)         part_id, qid, coder, label, evasion_type, request_type ...
  data/coding/adjudication.csv     (this script; fill the adjudicated_* columns by hand; pilot: adjudication_pilot.csv)
  data/coding/{stage}_sample.csv   (sample.py; gives ls, ministry_group, weight, in_estimate)
  data/measures/m2_labels.parquet  part_id, label, evasion_type, model_id
  data/measures/m1_scores.parquet  part_id, m1_cos
Cost units for --cost are arbitrary but must be consistent (e.g. projected USD for the full corpus). Model rule:
among models with Cohen's kappa >= threshold, pick the highest kappa per unit cost.

Krippendorff's alpha is implemented here (no dependency) and unit-tested against Krippendorff's (2011)
worked example: nominal 0.743, ordinal 0.815, interval 0.849.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import codebook as cb  # noqa: E402

NONE_APPLICABLE = "NONE_APPLICABLE"
EVASION_CHOICES = cb.EVASION_CHOICES
ORD_LABELS = ["NONE", "PARTIAL", "FULL"]  # ascending responsiveness
BOOT_SEED = 20261017
MIN_GROUP_N = 20


# =============================================================================================== agreement stats
def _delta2(n_c, level, values=None):
    K = len(n_c)
    if level == "nominal":
        return 1.0 - np.eye(K)
    if level == "interval":
        v = np.arange(K, dtype=float) if values is None else np.asarray(values, dtype=float)
        return (v[:, None] - v[None, :]) ** 2
    if level == "ordinal":
        d = np.zeros((K, K))
        for c in range(K):
            for k in range(c + 1, K):
                s = n_c[c:k + 1].sum() - (n_c[c] + n_c[k]) / 2.0
                d[c, k] = d[k, c] = s ** 2
        return d
    raise ValueError(level)


def unit_coincidences(counts):
    """counts: U x K matrix (how many coders gave category k to unit u). Returns U x K x K coincidence matrices
    (units with < 2 ratings contribute zeros)."""
    counts = np.asarray(counts, dtype=float)
    m = counts.sum(1)
    ok = m >= 2
    O = np.einsum("uc,uk->uck", counts, counts)
    idx = np.arange(counts.shape[1])
    O[:, idx, idx] -= counts
    denom = np.where(ok, m - 1, 1.0)
    O = O / denom[:, None, None]
    O[~ok] = 0.0
    return O


def alpha_from_coincidence(O, level="nominal", values=None):
    """Krippendorff's alpha from a K x K coincidence matrix O (sum over units)."""
    O = np.asarray(O, dtype=float)
    n_c = O.sum(1)
    n = n_c.sum()
    if n <= 1:
        return float("nan")
    d2 = _delta2(n_c, level, values)
    num = (O * d2).sum()
    den = (np.outer(n_c, n_c) * d2).sum()
    if den == 0:
        return float("nan")
    return float(1.0 - (n - 1) * num / den)


def counts_matrix(data, categories):
    """data: coders x units array/list-of-lists with None/NaN = missing; returns units x K count matrix."""
    arr = np.asarray(data, dtype=object)
    idx = {c: i for i, c in enumerate(categories)}
    out = np.zeros((arr.shape[1], len(categories)))
    for r in range(arr.shape[0]):
        for u in range(arr.shape[1]):
            v = arr[r, u]
            if v is None or (isinstance(v, float) and np.isnan(v)):
                continue
            out[u, idx[v]] += 1
    return out


def krippendorff_alpha(data, level="nominal", categories=None, values=None):
    """Krippendorff's alpha. data: coders x units (missing = None/NaN). categories: ordered category list
    (required order for 'ordinal'; default = sorted unique values). values: numeric values per category for
    'interval' (default: the categories themselves if numeric, else 0..K-1)."""
    arr = np.asarray(data, dtype=object)
    if categories is None:
        vals = {v for v in arr.ravel() if not (v is None or (isinstance(v, float) and np.isnan(v)))}
        categories = sorted(vals)
    if values is None and level == "interval":
        try:
            values = [float(c) for c in categories]
        except (TypeError, ValueError):
            values = None
    O = unit_coincidences(counts_matrix(arr, categories)).sum(0)
    return alpha_from_coincidence(O, level, values)


def confusion(y_true, y_pred, labels, w=None):
    idx = {c: i for i, c in enumerate(labels)}
    C = np.zeros((len(labels), len(labels)))
    w = np.ones(len(y_true)) if w is None else np.asarray(w, dtype=float)
    for a, b, wi in zip(y_true, y_pred, w):
        if a in idx and b in idx:
            C[idx[a], idx[b]] += wi
    return C


def kappa_from_confusion(C, weights=None):
    """Cohen's kappa from a confusion matrix; weights None | 'linear' | 'quadratic' (ordered categories)."""
    C = np.asarray(C, dtype=float)
    n = C.sum()
    if n == 0:
        return float("nan")
    K = C.shape[0]
    i, j = np.indices((K, K))
    if weights is None:
        W = (i != j).astype(float)
    elif weights == "linear":
        W = np.abs(i - j) / max(K - 1, 1)
    elif weights == "quadratic":
        W = ((i - j) / max(K - 1, 1)) ** 2
    else:
        raise ValueError(weights)
    E = np.outer(C.sum(1), C.sum(0)) / n
    den = (W * E).sum()
    if den == 0:
        return float("nan")
    return float(1.0 - (W * C).sum() / den)


def metrics_from_confusion(C):
    """accuracy, macro-F1 (over classes present in truth or prediction), kappa, linear-weighted kappa."""
    C = np.asarray(C, dtype=float)
    n = C.sum()
    tp = np.diag(C)
    prec_d, rec_d = C.sum(0), C.sum(1)
    present = (prec_d + rec_d) > 0
    f1 = np.where(prec_d + rec_d > 0, 2 * tp / np.where(prec_d + rec_d == 0, 1, prec_d + rec_d), 0.0)
    return {"n": float(n), "accuracy": float(tp.sum() / n) if n else float("nan"),
            "macro_f1": float(f1[present].mean()) if present.any() else float("nan"),
            "kappa": kappa_from_confusion(C), "kappa_lw": kappa_from_confusion(C, "linear")}


def _cluster_codes(clusters):
    codes, uniq = pd.factorize(pd.Series(clusters))
    return codes, len(uniq)


def bootstrap_ci(per_cluster, stat, B=1000, seed=BOOT_SEED, alpha=0.05):
    """per_cluster: array G x ... of additive per-cluster summaries; stat(sum over resampled clusters) -> float.
    Cluster bootstrap percentile CI."""
    per_cluster = np.asarray(per_cluster, dtype=float)
    G = per_cluster.shape[0]
    if G < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(B):
        cnt = np.bincount(rng.integers(0, G, G), minlength=G).astype(float)
        v = stat(np.tensordot(cnt, per_cluster, axes=(0, 0)))
        if not np.isnan(v):
            vals.append(v)
    if len(vals) < max(10, B // 10):
        return (float("nan"), float("nan"))
    return (float(np.percentile(vals, 100 * alpha / 2)), float(np.percentile(vals, 100 * (1 - alpha / 2))))


def alpha_with_ci(codes_by_coder, clusters, categories, level="nominal", values=None, B=1000):
    """codes_by_coder: coders x units object array (missing=None). Returns (alpha, lo, hi)."""
    arr = np.asarray(codes_by_coder, dtype=object)
    Ou = unit_coincidences(counts_matrix(arr, categories))
    point = alpha_from_coincidence(Ou.sum(0), level, values)
    if B <= 0:
        return point, float("nan"), float("nan")
    g, G = _cluster_codes(clusters)
    per = np.zeros((G,) + Ou.shape[1:])
    np.add.at(per, g, Ou)
    lo, hi = bootstrap_ci(per, lambda O: alpha_from_coincidence(O, level, values), B=B)
    return point, lo, hi


# =============================================================================================== reliability
def human_wide(human, stage=None, field="label"):
    h = human if stage is None else human[human["stage"] == stage]
    w = h.pivot(index="part_id", columns="coder", values=field)
    return w.dropna(how="all")


def _pair_stats(a, b, cats, ordered):
    C = confusion(a, b, cats)
    out = {"n": int(C.sum()), "pct_agree": float(np.trace(C) / C.sum()) if C.sum() else float("nan"),
           "kappa": kappa_from_confusion(C)}
    if ordered:
        out["kappa_lw"] = kappa_from_confusion(C, "linear")
    return out, C


def double_coded_only(human):
    """Rows of the parts that have labels from at least two coders (single-coded parts carry no reliability
    information and are dropped)."""
    if len(human) == 0:
        return human
    return human[human.groupby("part_id")["coder"].transform("nunique") >= 2]


def reliability(human, stage=None, B=1000, groups=None):
    """Reliability of the human labels. human: long table (part_id, qid, coder, label, evasion_type, request_type).
    groups: optional Series indexed by part_id (or qid-level dict) to break results down; see reliability_by."""
    out = {"coders": [], "fields": {}}
    h = human if stage is None else human[human["stage"] == stage]
    h = double_coded_only(h)
    coders = sorted(h["coder"].unique())
    out["coders"] = coders
    if len(coders) < 2:
        return out
    spec = {"label": (ORD_LABELS, True), "evasion_type": (EVASION_CHOICES, False),
            "request_type": (cb.REQUEST_TYPES, False)}
    for field, (cats, ordered) in spec.items():
        w = human_wide(h, None, field)
        w = w.dropna(subset=coders[:2]) if len(coders) == 2 else w
        if len(w) == 0:
            continue
        qid = w.index.str.rsplit("#", n=1).str[0]
        arr = np.array([[None if pd.isna(v) else v for v in w[c]] for c in coders], dtype=object)
        a, lo, hi = alpha_with_ci(arr, qid, cats, "nominal", B=B)
        res = {"n_parts": int(len(w)), "alpha_nominal": a, "alpha_nominal_ci": (lo, hi)}
        if ordered:
            ao, olo, ohi = alpha_with_ci(arr, qid, cats, "ordinal", B=B)
            res.update(alpha_ordinal=ao, alpha_ordinal_ci=(olo, ohi))
        pairs = []
        confs = None
        for i in range(len(coders)):
            for j in range(i + 1, len(coders)):
                ww = w.dropna(subset=[coders[i], coders[j]])
                st, C = _pair_stats(ww[coders[i]], ww[coders[j]], cats, ordered)
                pairs.append(st)
                if confs is None:
                    confs = (coders[i], coders[j], pd.DataFrame(C.astype(int), index=cats, columns=cats))
        res["pct_agree"] = float(np.mean([p["pct_agree"] for p in pairs]))
        res["kappa"] = float(np.mean([p["kappa"] for p in pairs]))
        if ordered:
            res["kappa_lw"] = float(np.mean([p["kappa_lw"] for p in pairs]))
        res["confusion"] = confs
        out["fields"][field] = res
    # evasion type conditional on both coders saying PARTIAL/NONE
    wl = human_wide(h, None, "label")
    we = human_wide(h, None, "evasion_type")
    both = wl.index[(wl[coders[:2]].notna().all(axis=1)) & (wl[coders[:2]] != "FULL").all(axis=1)]
    if len(both) >= 2 and len(coders) >= 2:
        sub = we.loc[both, coders[:2]]
        qid = sub.index.str.rsplit("#", n=1).str[0]
        arr = np.array([[None if pd.isna(v) else v for v in sub[c]] for c in coders[:2]], dtype=object)
        a, lo, hi = alpha_with_ci(arr, qid, cb.EVASION_TYPES + [NONE_APPLICABLE], "nominal", B=B)
        C = confusion(sub[coders[0]], sub[coders[1]], EVASION_CHOICES)
        out["evasion_both_nonfull"] = {"n_parts": int(len(both)), "alpha_nominal": a, "alpha_nominal_ci": (lo, hi),
                                       "pct_agree": float(np.trace(C) / C.sum()),
                                       "kappa": kappa_from_confusion(C)}
    return out


def reliability_by(human, meta, by, stage=None, B=0):
    """Label reliability per level of `by` ('ls' or 'ministry_group'); meta: sample table with qid + by."""
    h = human if stage is None else human[human["stage"] == stage]
    h = h.merge(meta[["qid", by]].drop_duplicates("qid"), on="qid", how="left")
    rows = []
    for k, g in h.groupby(by):
        r = reliability(g, None, B=B)
        f = r["fields"].get("label")
        if not f:
            continue
        rows.append({by: k, "n_parts": f["n_parts"], "pct_agree": f["pct_agree"], "kappa": f["kappa"],
                     "alpha_nominal": f["alpha_nominal"], "alpha_ordinal": f["alpha_ordinal"],
                     "small_n": f["n_parts"] < MIN_GROUP_N})
    return pd.DataFrame(rows)


# =============================================================================================== adjudication / gold
ADJ_FIELDS = ["label", "evasion_type", "request_type"]


def disagreements(human, stage=None):
    """Wide table of parts coded by >= 2 coders where any field differs (first two coders)."""
    h = human if stage is None else human[human["stage"] == stage]
    coders = sorted(h["coder"].unique())
    if len(coders) < 2:
        return pd.DataFrame(), coders
    c1, c2 = coders[:2]
    w = h.pivot(index="part_id", columns="coder", values=ADJ_FIELDS + ["note"])
    w.columns = [f"{f}__{c}" for f, c in w.columns]
    w = w.dropna(subset=[f"label__{c1}", f"label__{c2}"])
    dis = pd.DataFrame(index=w.index)
    dis["label"] = w[f"label__{c1}"] != w[f"label__{c2}"]
    both_nonfull = (w[f"label__{c1}"] != "FULL") & (w[f"label__{c2}"] != "FULL")
    dis["evasion_type"] = (w[f"evasion_type__{c1}"] != w[f"evasion_type__{c2}"]) & both_nonfull
    dis["request_type"] = w[f"request_type__{c1}"] != w[f"request_type__{c2}"]
    any_d = dis.any(axis=1)
    out = pd.DataFrame({"part_id": w.index[any_d]})
    out["qid"] = out["part_id"].str.rsplit("#", n=1).str[0]
    out["disagree_on"] = [";".join(f for f in ADJ_FIELDS if dis.loc[p, f]) for p in out["part_id"]]
    for c in (c1, c2):
        for f in ADJ_FIELDS + ["note"]:
            out[f"{c}_{f}"] = w.loc[out["part_id"], f"{f}__{c}"].values
    return out.reset_index(drop=True), (c1, c2)


def build_adjudication(human, stage=None, parts_text=None, existing=None, max_chars=1500):
    """Disagreement list with empty adjudicated_* columns; keeps values already entered in `existing`."""
    d, coders = disagreements(human, stage)
    if d.empty:
        return d
    if parts_text is not None:
        t = parts_text.set_index("part_id")[["question_part", "answer_part"]]
        for col in ("question_part", "answer_part"):
            d[col] = d["part_id"].map(t[col]).fillna("").astype(str).str.slice(0, max_chars)
    for c in ("adjudicated_label", "adjudicated_evasion_type", "adjudicated_request_type", "adjudicator_note"):
        d[c] = ""
    if existing is not None and len(existing):
        ex = existing.set_index("part_id")
        for c in ("adjudicated_label", "adjudicated_evasion_type", "adjudicated_request_type", "adjudicator_note"):
            if c in ex.columns:
                d[c] = d["part_id"].map(ex[c]).fillna("").astype(str)
    return d


GOLD_COLS = ["part_id", "qid", "label", "evasion_type", "request_type", "gold_source"]


def build_gold(human, adjudication=None, stage=None, sample=None):
    """Gold labels. Parts coded by both coders: the agreed value where they agree, else the adjudicated value
    (NaN while still open); gold_source = 'agreed' if nothing needed adjudication, else 'adjudicated'. Parts coded
    by one coder only: that coder's label, gold_source = 'single'. If `sample` has a `coders` column, a part of a
    question assigned to both coders that only one has coded so far is pending (not emitted).
    Returns (gold DataFrame, status dict)."""
    h = human if stage is None else human[human["stage"] == stage]
    coders = sorted(h["coder"].unique())
    status0 = {"n_coded_by_both": 0, "n_single": 0, "n_pending": 0, "n_gold": 0, "n_open": 0, "n_adjudicated": 0,
               "n_coders": len(coders)}
    if len(coders) == 0:
        return pd.DataFrame(columns=GOLD_COLS), status0
    c1 = coders[0]
    c2 = coders[1] if len(coders) >= 2 else None
    w = h.pivot(index="part_id", columns="coder", values=ADJ_FIELDS)
    w.columns = [f"{f}__{c}" for f, c in w.columns]
    for f in ADJ_FIELDS:
        for c in (c1, c2):
            if c is not None and f"{f}__{c}" not in w.columns:
                w[f"{f}__{c}"] = np.nan
    expected_both = None
    if sample is not None and "coders" in sample.columns:
        sq = sample.drop_duplicates("qid").set_index("qid")["coders"].astype(str)
        expected_both = set(sq.index[sq.map(lambda v: len({x.strip() for x in v.split(";") if x.strip()}) >= 2)])
    adj = adjudication.set_index("part_id") if adjudication is not None and len(adjudication) else None

    def adjval(pid, col, allowed):
        if adj is None or pid not in adj.index or col not in adj.columns:
            return None
        v = str(adj.at[pid, col]).strip().upper().replace(" ", "_")
        v = cb.EVASION_RENAMES.get(v, v)  # codes renamed between codebook versions (PROCEDURAL -> WITHHELD)
        return v if v in allowed else None

    rows, pending = [], 0
    for pid, r in w.iterrows():
        qid = pid.rsplit("#", 1)[0]
        rec = {"part_id": pid, "qid": qid}
        has1 = pd.notna(r[f"label__{c1}"])
        has2 = c2 is not None and pd.notna(r[f"label__{c2}"])
        if has1 and has2:
            src = "agreed"
            if r[f"label__{c1}"] == r[f"label__{c2}"]:
                rec["label"] = r[f"label__{c1}"]
            else:
                rec["label"] = adjval(pid, "adjudicated_label", cb.LABELS)
                src = "adjudicated"
            if r[f"request_type__{c1}"] == r[f"request_type__{c2}"]:
                rec["request_type"] = r[f"request_type__{c1}"]
            else:
                rec["request_type"] = adjval(pid, "adjudicated_request_type", cb.REQUEST_TYPES)
                src = "adjudicated"
            # an explicit adjudicated value always overrides, even for a field the coders agreed on
            # (e.g. both chose OTHER but the adjudicator re-classified it under a newer type such as WITHHELD)
            for f, allowed in (("label", cb.LABELS), ("request_type", cb.REQUEST_TYPES)):
                v = adjval(pid, f"adjudicated_{f}", allowed)
                if v is not None and v != rec[f]:
                    rec[f] = v
                    src = "adjudicated"
            adj_ev = adjval(pid, "adjudicated_evasion_type", cb.EVASION_TYPES + [NONE_APPLICABLE])
            if rec["label"] == "FULL":
                rec["evasion_type"] = NONE_APPLICABLE
            elif rec["label"] is None:
                rec["evasion_type"] = None
            elif adj_ev is not None and adj_ev != NONE_APPLICABLE:
                rec["evasion_type"] = adj_ev
                if adj_ev != r[f"evasion_type__{c1}"] or adj_ev != r[f"evasion_type__{c2}"]:
                    src = "adjudicated"
            elif r[f"evasion_type__{c1}"] == r[f"evasion_type__{c2}"] and r[f"evasion_type__{c1}"] != NONE_APPLICABLE:
                rec["evasion_type"] = r[f"evasion_type__{c1}"]
            else:
                rec["evasion_type"] = None  # open: needs adjudication
                src = "adjudicated"
            rec["gold_source"] = src
        elif has1 or has2:
            if expected_both is not None and qid in expected_both:
                pending += 1
                continue
            c = c1 if has1 else c2
            rec["label"] = r[f"label__{c}"]
            rec["request_type"] = r[f"request_type__{c}"]
            rec["evasion_type"] = NONE_APPLICABLE if rec["label"] == "FULL" else r[f"evasion_type__{c}"]
            rec["gold_source"] = "single"
        else:
            continue
        rows.append(rec)
    g = pd.DataFrame(rows, columns=GOLD_COLS)
    status = {"n_coded_by_both": int(g["gold_source"].isin(["agreed", "adjudicated"]).sum()),
              "n_single": int((g["gold_source"] == "single").sum()), "n_pending": int(pending),
              "n_gold": int(g["label"].notna().sum()), "n_open": int(g["label"].isna().sum()),
              "n_adjudicated": int((g["gold_source"] == "adjudicated").sum()), "n_coders": len(coders)}
    return g, status


# =============================================================================================== M2 / M1 vs gold
def _part_weights(gold, sample):
    if sample is None or "weight" not in sample:
        return pd.Series(1.0, index=gold.index)
    wmap = sample.drop_duplicates("qid").set_index("qid")["weight"]
    return gold["qid"].map(wmap).fillna(0.0)


def evaluate_m2(gold, m2, sample=None, B=500, labels=ORD_LABELS):
    """Per model_id: coverage, accuracy, macro-F1, kappa (+CIs), design-weighted accuracy/kappa, confusion matrix,
    evasion-type accuracy on gold non-FULL parts. gold needs part_id, qid, label, evasion_type."""
    g = gold.dropna(subset=["label"]).copy()
    g["w"] = _part_weights(g, sample).values
    rows, confs = [], {}
    for model, mm in m2.groupby("model_id"):
        j = g.merge(mm[["part_id", "label", "evasion_type"]].rename(columns={"label": "m2_label",
                                                                           "evasion_type": "m2_ev"}),
                    on="part_id", how="inner")
        if j.empty:
            continue
        C = confusion(j["label"], j["m2_label"], labels)
        met = metrics_from_confusion(C)
        qc, G = _cluster_codes(j["qid"])
        K = len(labels)
        per = np.zeros((G, K * K))
        li = {c: i for i, c in enumerate(labels)}
        for q, a, b in zip(qc, j["label"], j["m2_label"]):
            if a in li and b in li:
                per[q, li[a] * K + li[b]] += 1
        kap = lambda v: kappa_from_confusion(v.reshape(K, K))  # noqa: E731
        acc = lambda v: float(np.trace(v.reshape(K, K)) / v.sum()) if v.sum() else float("nan")  # noqa: E731
        klo, khi = bootstrap_ci(per, kap, B=B)
        alo, ahi = bootstrap_ci(per, acc, B=B)
        Cw = confusion(j["label"], j["m2_label"], labels, j["w"])
        mw = metrics_from_confusion(Cw) if Cw.sum() > 0 else {"accuracy": np.nan, "kappa": np.nan}
        nf = j[j["label"] != "FULL"]
        ev_acc = float((nf["evasion_type"] == nf["m2_ev"]).mean()) if len(nf) else float("nan")
        rows.append({"model_id": model, "n_parts": len(j), "coverage": len(j) / len(g), "accuracy": met["accuracy"],
                     "accuracy_ci_lo": alo, "accuracy_ci_hi": ahi, "macro_f1": met["macro_f1"],
                     "kappa": met["kappa"], "kappa_ci_lo": klo, "kappa_ci_hi": khi, "kappa_lw": met["kappa_lw"],
                     "accuracy_weighted": mw["accuracy"], "kappa_weighted": mw["kappa"],
                     "evasion_acc_nonfull": ev_acc})
        confs[model] = pd.DataFrame(C.astype(int), index=[f"gold:{l}" for l in labels],
                                    columns=[f"m2:{l}" for l in labels])
    return pd.DataFrame(rows), confs


def select_model(table, costs=None, threshold=0.6):
    """Cheapest-adequate rule: among models with kappa >= threshold choose the highest kappa per unit cost.
    Returns (table with kappa_per_cost/passes columns, chosen model_id or None, message)."""
    t = table.copy()
    costs = costs or {}
    t["cost"] = t["model_id"].map(costs)
    t["passes"] = t["kappa"] >= threshold
    if "kappa_ci_lo" in t:
        t["ci_lo_clears"] = t["kappa_ci_lo"] >= threshold
    t["kappa_per_cost"] = np.where(t["cost"] > 0, t["kappa"] / t["cost"], np.nan)
    ok = t[t["passes"] & t["kappa_per_cost"].notna()]
    if ok.empty:
        if t["passes"].any():
            return t, None, "models pass the kappa threshold but none has a cost given (use --cost id=value)"
        best = t.sort_values("kappa", ascending=False).head(1)
        msg = (f"no model reaches kappa >= {threshold}; best is {best['model_id'].iloc[0]} "
               f"(kappa={best['kappa'].iloc[0]:.3f}) - revise the prompt/codebook or use a stronger model"
               if len(best) else "no models evaluated")
        return t, None, msg
    pick = ok.sort_values(["kappa_per_cost", "kappa"], ascending=False).iloc[0]
    ci = ""
    if "ci_lo_clears" in t:
        ci = (f"; lower 95% CI bound {pick['kappa_ci_lo']:.3f} "
              f"{'clears' if pick['ci_lo_clears'] else 'does NOT clear'} the threshold")
    return t, pick["model_id"], (f"{pick['model_id']}: kappa={pick['kappa']:.3f} >= {threshold}, "
                                 f"highest kappa per cost ({pick['kappa_per_cost']:.3f}){ci}")


def auc_rank(score, pos):
    """AUC (Mann-Whitney) with tie handling; pos is boolean. NaN if one class is empty."""
    score = np.asarray(score, dtype=float)
    pos = np.asarray(pos, dtype=bool)
    n1, n0 = pos.sum(), (~pos).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = pd.Series(score).rank(method="average").values
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def youden_threshold(score, pos):
    """Threshold t (predict positive if score >= t) maximising TPR - FPR. Returns (t, tpr, fpr, J)."""
    score = np.asarray(score, dtype=float)
    pos = np.asarray(pos, dtype=bool)
    order = np.argsort(-score, kind="stable")
    s, p = score[order], pos[order]
    tp, fp = np.cumsum(p), np.cumsum(~p)
    last = np.r_[s[1:] != s[:-1], True]  # cut only after the last of tied scores
    tpr, fpr = tp[last] / max(p.sum(), 1), fp[last] / max((~p).sum(), 1)
    J = tpr - fpr
    k = int(np.argmax(J))
    return float(s[last][k]), float(tpr[k]), float(fpr[k]), float(J[k])


def spearman(x, y):
    x, y = pd.Series(x).rank(method="average").values, pd.Series(y).rank(method="average").values
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def evaluate_m1(gold, m1, sample=None, B=500):
    """AUC FULL-vs-NONE (and FULL vs not-FULL), Spearman vs LABEL_SCORE, Youden threshold. Cluster-bootstrap CIs."""
    j = gold.dropna(subset=["label"]).merge(m1[["part_id", "m1_cos"]].dropna(), on="part_id", how="inner")
    if j.empty:
        return None
    out = {"n_parts": len(j), "coverage": len(j) / gold["label"].notna().sum()}
    fn = j[j["label"].isin(["FULL", "NONE"])]
    pos = (fn["label"] == "FULL").values
    out["n_full"], out["n_none"] = int(pos.sum()), int((~pos).sum())
    out["auc_full_vs_none"] = auc_rank(fn["m1_cos"], pos)
    out["auc_full_vs_rest"] = auc_rank(j["m1_cos"], (j["label"] == "FULL").values)
    out["spearman_label_score"] = spearman(j["m1_cos"], j["label"].map(cb.LABEL_SCORE))
    if out["n_full"] and out["n_none"]:
        t, tpr, fpr, J = youden_threshold(fn["m1_cos"], pos)
        out.update(youden_threshold=t, tpr_at_t=tpr, fpr_at_t=fpr, youden_j=J)
    rng = np.random.default_rng(BOOT_SEED)
    qids = j["qid"].unique()
    groups = {q: d for q, d in j.groupby("qid")}
    aucs, rhos = [], []
    for _ in range(B):
        pick = rng.choice(qids, len(qids))
        d = pd.concat([groups[q] for q in pick])
        dd = d[d["label"].isin(["FULL", "NONE"])]
        aucs.append(auc_rank(dd["m1_cos"], (dd["label"] == "FULL").values))
        rhos.append(spearman(d["m1_cos"], d["label"].map(cb.LABEL_SCORE)))
    q = lambda v: tuple(np.nanpercentile(v, [2.5, 97.5])) if np.isfinite(v).sum() > 10 else (np.nan, np.nan)  # noqa: E731
    out["auc_full_vs_none_ci"], out["spearman_ci"] = q(aucs), q(rhos)
    return out


# =============================================================================================== IO + report
def _read_csv(path, **kw):
    return pd.read_csv(path, keep_default_na=False, **kw) if os.path.exists(path) else None


def coding_file(root, name, stage):
    """data/coding/<name>.csv for the main stage (what build_frame/bias_correct read), <name>_pilot.csv for the pilot,
    so building one stage never overwrites the other's adjudication or gold file."""
    return os.path.join(root, "data", "coding", f"{name}{'_pilot' if stage == 'pilot' else ''}.csv")


def load_inputs(root, stage):
    d = os.path.join(root, "data", "coding")
    human = _read_csv(os.path.join(d, "human_labels.csv"))
    if human is not None and "stage" in human and stage:
        human = human[human["stage"] == stage]
    sample = _read_csv(os.path.join(d, f"{stage}_sample.csv"))
    if sample is not None:
        sample["in_estimate"] = sample["in_estimate"].astype(str).str.lower().eq("true")
        sample["weight"] = pd.to_numeric(sample["weight"])
    adj = _read_csv(coding_file(root, "adjudication", stage))
    m2p = os.path.join(root, "data", "measures", "m2_labels.parquet")
    m1p = os.path.join(root, "data", "measures", "m1_scores.parquet")
    return {"human": human if human is not None and len(human) else None, "sample": sample, "adj": adj,
            "m2": pd.read_parquet(m2p) if os.path.exists(m2p) else None,
            "m1": pd.read_parquet(m1p) if os.path.exists(m1p) else None}


def parts_text(root, part_ids):
    p = os.path.join(root, "data", "measures", "parts.parquet")
    if not os.path.exists(p):
        return None
    qids = sorted({x.rsplit("#", 1)[0] for x in part_ids})
    return pd.read_parquet(p, filters=[("qid", "in", qids)], columns=["part_id", "question_part", "answer_part"])


def md_table(df, fmt="{:.3f}"):
    if df is None or len(df) == 0:
        return "_(empty)_\n"
    cols = list(df.columns)
    out = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, (float, np.floating)):
                cells.append("" if np.isnan(v) else fmt.format(v))
            elif isinstance(v, tuple):
                cells.append("[" + ", ".join("nan" if np.isnan(x) else fmt.format(x) for x in v) + "]")
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


def _ci(t):
    return f"[{t[0]:.3f}, {t[1]:.3f}]" if t and not np.isnan(t[0]) else "n/a"


def write_report(root=ROOT, stage="main", threshold=0.6, costs=None, B=1000, out_path=None, exclude_sanity=True):
    inp = load_inputs(root, stage)
    human, sample, adj = inp["human"], inp["sample"], inp["adj"]
    L = [f"# M3 validation report (stage: {stage})", "",
         f"Codebook {cb.CODEBOOK_VERSION}; kappa threshold for M2 model choice = {threshold}; bootstrap B = {B} "
         f"(cluster = question).", ""]
    gold = None
    L += ["## 1. Inter-coder reliability (human labels)", ""]
    if human is None:
        L += ["_human_labels.csv not found or empty for this stage - nothing to report yet._", ""]
    else:
        if sample is not None:
            est_q = set(sample.loc[sample["in_estimate"], "qid"])
            h_est, h_san = human[human["qid"].isin(est_q)], human[~human["qid"].isin(est_q)]
        else:
            h_est, h_san = human, human.iloc[0:0]
        dbl = double_coded_only(h_est)
        r = reliability(h_est, None, B=B)
        if len(r["coders"]) < 2:
            L += [f"_Only {len(r['coders'])} coder(s) present; reliability needs 2._", ""]
        elif len(dbl) == 0:
            L += ["_No part has been coded by both coders yet._", ""]
        else:
            L += [f"Coders: {', '.join(r['coders'])}. Only parts coded by both coders are used (primary set = "
                  f"double-coded estimate questions: {dbl['qid'].nunique()} questions, "
                  f"{dbl['part_id'].nunique()} parts; single-coded parts are excluded).", ""]
            rows = []
            for f, v in r["fields"].items():
                rows.append({"field": f, "n_parts": v["n_parts"], "% agree": 100 * v["pct_agree"],
                             "kappa": v["kappa"], "kappa_lw": v.get("kappa_lw", np.nan),
                             "alpha_nominal": v["alpha_nominal"], "alpha_nom_95CI": v["alpha_nominal_ci"],
                             "alpha_ordinal": v.get("alpha_ordinal", np.nan),
                             "alpha_ord_95CI": v.get("alpha_ordinal_ci", (np.nan, np.nan))})
            if "evasion_both_nonfull" in r:
                v = r["evasion_both_nonfull"]
                rows.append({"field": "evasion_type (both non-FULL)", "n_parts": v["n_parts"],
                             "% agree": 100 * v["pct_agree"], "kappa": v["kappa"], "kappa_lw": np.nan,
                             "alpha_nominal": v["alpha_nominal"], "alpha_nom_95CI": v["alpha_nominal_ci"],
                             "alpha_ordinal": np.nan, "alpha_ord_95CI": (np.nan, np.nan)})
            L += [md_table(pd.DataFrame(rows)), ""]
            if "label" in r["fields"]:
                c1, c2, C = r["fields"]["label"]["confusion"]
                L += [f"Label confusion matrix (rows: {c1}, columns: {c2}):", "",
                      md_table(C.reset_index().rename(columns={"index": ""}), "{:.0f}"), ""]
            if sample is not None:
                for by in ("ls", "ministry_group"):
                    t = reliability_by(h_est, sample, by)
                    L += [f"Label reliability by {by} (small_n = fewer than {MIN_GROUP_N} parts):", "", md_table(t), ""]
        sdbl = double_coded_only(h_san)
        if len(sdbl):
            rs = reliability(h_san, None, B=0)
            f = rs["fields"].get("label")
            if f:
                L += [f"Sanity stratum (no answer text), reported separately: {f['n_parts']} parts coded by both, "
                      f"label agreement {100 * f['pct_agree']:.1f}% (kappa {f['kappa']:.3f}; undefined when "
                      f"both coders use one category only).", ""]
    # adjudication / gold
    L += ["## 2. Adjudication and gold labels", ""]
    if human is not None and human["coder"].nunique() >= 1:
        gold, st = build_gold(human, adj, sample=sample)
        d, _ = disagreements(human)
        src = gold["gold_source"].value_counts().to_dict()
        L += [f"Parts coded by both coders: {st['n_coded_by_both']}; single-coded parts: {st['n_single']}; "
              f"pending (second coder not imported yet): {st['n_pending']}; disagreements on any field: {len(d)}; "
              f"gold labels available: {st['n_gold']} (by source: "
              f"{', '.join(f'{k}={v}' for k, v in sorted(src.items())) or 'none'}); "
              f"adjudicated: {st['n_adjudicated']}; open: {st['n_open']}.",
              "(Run `validate.py adjudication`, fill the adjudicated_* columns, then re-run.)", ""]
        gpath = coding_file(root, "gold_labels", stage)
        gold.to_csv(gpath, index=False)
        if sample is not None:
            gold = gold.merge(sample.drop_duplicates("qid")[["qid", "in_estimate", "ls", "ministry_group"]],
                              on="qid", how="left")
            san = gold[(gold["in_estimate"] == False)]  # noqa: E712
            if len(san):
                L += [f"Sanity stratum (no answer text): {len(san)} parts, "
                      f"{(san['label'] == 'NONE').mean():.0%} gold NONE (expect ~100%).", ""]
            if exclude_sanity:
                gold = gold[gold["in_estimate"].fillna(True) == True]  # noqa: E712
    else:
        L += ["_No human labels yet._", ""]
    # M2
    L += ["## 3. M2 (LLM labels) vs gold", ""]
    if gold is None or gold["label"].notna().sum() == 0:
        L += ["_No gold labels yet._", ""]
    elif inp["m2"] is None:
        L += ["_data/measures/m2_labels.parquet not found yet._", ""]
    else:
        tab, confs = evaluate_m2(gold, inp["m2"], sample, B=min(B, 500))
        if tab.empty:
            L += ["_No overlap between M2 labels and gold parts._", ""]
        else:
            tab2, pick, msg = select_model(tab, costs, threshold)
            show = tab2[["model_id", "n_parts", "coverage", "accuracy", "macro_f1", "kappa", "kappa_lw",
                         "accuracy_weighted", "kappa_weighted", "evasion_acc_nonfull", "cost", "kappa_per_cost",
                         "passes", "ci_lo_clears"]].copy()
            show["kappa_95CI"] = [(a, b) for a, b in zip(tab2["kappa_ci_lo"], tab2["kappa_ci_hi"])]
            L += [md_table(show), f"**Model rule** (kappa >= {threshold}, max kappa per cost): {msg}", ""]
            for m, C in confs.items():
                L += [f"Confusion matrix for `{m}`:", "", md_table(C.reset_index().rename(columns={"index": ""}),
                                                                   "{:.0f}"), ""]
    # M1
    L += ["## 4. M1 (embedding similarity) vs gold", ""]
    if gold is None or gold["label"].notna().sum() == 0:
        L += ["_No gold labels yet._", ""]
    elif inp["m1"] is None:
        L += ["_data/measures/m1_scores.parquet not found yet._", ""]
    else:
        e = evaluate_m1(gold, inp["m1"], sample, B=min(B, 500))
        if e is None:
            L += ["_No overlap between M1 scores and gold parts._", ""]
        else:
            L += [f"Parts: {e['n_parts']} (FULL {e['n_full']}, NONE {e['n_none']}).", "",
                  f"- AUC FULL vs NONE: {e['auc_full_vs_none']:.3f} {_ci(e['auc_full_vs_none_ci'])}",
                  f"- AUC FULL vs rest: {e['auc_full_vs_rest']:.3f}",
                  f"- Spearman(M1, LABEL_SCORE): {e['spearman_label_score']:.3f} {_ci(e['spearman_ci'])}"]
            if "youden_threshold" in e:
                L += [f"- Youden-optimal threshold: {e['youden_threshold']:.3f} (TPR {e['tpr_at_t']:.2f}, "
                      f"FPR {e['fpr_at_t']:.2f}, J {e['youden_j']:.2f})"]
            L += [""]
    out_path = out_path or os.path.join(root, "results", "validation_report.md")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    return out_path, "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")
    for name in ("report", "adjudication", "gold"):
        p = sub.add_parser(name)
        p.add_argument("--stage", default="main", choices=["pilot", "main"])
        p.add_argument("--root", default=ROOT)
        if name == "report":
            p.add_argument("--threshold", type=float, default=0.6)
            p.add_argument("--cost", action="append", default=[], metavar="MODEL=COST")
            p.add_argument("--boot", type=int, default=1000)
            p.add_argument("--include-sanity", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd is None:
        a = ap.parse_args(["report"] + (argv or []))
    inp = load_inputs(a.root, a.stage)
    if a.cmd in ("adjudication", "gold") and inp["human"] is None:
        print("no human labels for this stage")
        return 1
    if a.cmd == "adjudication":
        pt = parts_text(a.root, inp["human"]["part_id"])
        d = build_adjudication(inp["human"], None, pt, inp["adj"])
        path = coding_file(a.root, "adjudication", a.stage)
        d.to_csv(path, index=False)
        print(f"{len(d)} disagreeing parts -> {path} (fill adjudicated_label / _evasion_type / _request_type)")
    elif a.cmd == "gold":
        g, st = build_gold(inp["human"], inp["adj"], sample=inp["sample"])
        g.to_csv(coding_file(a.root, "gold_labels", a.stage), index=False)
        print(json.dumps(st))
    else:
        costs = {k: float(v) for k, v in (c.split("=", 1) for c in a.cost)}
        path, text = write_report(a.root, a.stage, a.threshold, costs, a.boot, exclude_sanity=not a.include_sanity)
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
