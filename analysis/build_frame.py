"""Assemble the analysis frame: one row per question = covariates + outcomes (+ gold subset).

    python -I -X utf8 analysis/build_frame.py --topics            # (re)build topic clusters (slow, ~10 min), once
    python -I -X utf8 analysis/build_frame.py --outcome synthetic # frame with the SYNTHETIC outcome (validation)
    python -I -X utf8 analysis/build_frame.py --outcome real      # frame with the real M1/M2 outcomes (when available)

Outputs data/measures/analysis_frame_real.parquet or data/measures/synthetic/analysis_frame_synthetic.parquet.

Topic fixed effects (pre-specifiable, fixed before outcome models are run):
    TF-IDF (unigrams, English + parliamentary stop words, min_df=50, max_df=0.30, max_features=30000, sublinear tf)
    on `subject + question text[:1500]` -> TruncatedSVD(100) -> L2 normalise -> KMeans(K=50, n_init=3,
    random_state=20261017). Written to data/measures/topics.parquet (+ topics_meta.json, topic_terms.csv).
    Topic clusters use ONLY question text (never answers/outcomes), so they can be frozen before outcomes exist.

Outcome conventions (columns in the frame):
    y_m2     mean over parts of LABEL_SCORE (FULL=1, PARTIAL=.5, NONE=0) from M2 labels   [primary]
    y_full   share of parts labelled FULL
    y_m1     mean part cosine (M1);   y_m1_cov  M1 part-coverage;   y_m1b  second embedding model
    y_gold   human gold question score (mean LABEL_SCORE over the human-coded parts of the question)
    gold_complete  all parts of the question are human-coded; gold_pi = 1/weight from data/coding/main_sample.csv
    gold_universe  question belongs to the M3 sampling universe (answer present, 1..6 parts, not in the pilot):
                   the population to which the bias-corrected (PPI/DSL) estimates refer.
M2 sampling design (analysis plan §4.3a, paper/analysis_plan.md; data/measures/m2_sample.csv from analysis/m2_subsample.py):
    in_m2_sample   question is in the M2 sample (any role)
    m2_role        census (LS17/18) | random (stratified 1/6 of LS13-16) | gold (gold-only extra) | out (not drawn)
    m2_weight      design weight N_h/n_h (census 1); 0 for gold-only extras and for questions not drawn, so weighted
                   models (models.design_weight) never use them.  Synthetic outcome: data/measures/synthetic/
                   m2_sample.csv if present, else every question is 'census' with weight 1.
Covariates are documented in COVARIATE_DOC below.
"""
import argparse
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import common as C  # noqa: E402
from codebook import LABEL_SCORE  # noqa: E402

TOPICS = os.path.join(C.MEASURES, "topics.parquet")
TOPICS_META = os.path.join(C.MEASURES, "topics_meta.json")
TOPIC_TERMS = os.path.join(C.MEASURES, "topic_terms.csv")
N_TOPICS = 50
MAX_GOLD_PARTS = 6      # M3 sampling universe (analysis/sample.py MAX_PARTS)

EXTRA_STOP = ("whether government will state states india indian minister ministry details thereof therefor steps "
              "taken reasons pleased aware proposal proposed said regard various last years year wise number "
              "specific lok sabha sabha so if any has have been being such made provide period during since "
              "along also per under").split()

COVARIATE_DOC = {
    "opp": "asker_bloc == 'opposition' (all askers opposition)",
    "mixed": "asker_bloc == 'mixed' (askers from more than one bloc)",
    "outside": "asker_bloc == 'outside_support' (parties giving outside support to the government)",
    "opp_strict": "same as opp but from asker_bloc_strict (split-party windows coded conservatively)",
    "coalition_dep": "majority_type == 'coalition-dependent' (H2 expects +)",
    "ls18": "ls == 18 (NDA-M3; the key H2 contrast is ls18 vs ls17)",
    "security": "ministry_group == 'security' (Home Affairs, Defence, External Affairs)",
    "mp_id": "mpsno of the FIRST asker (cluster id; fallback to name key when mpsno missing)",
    "n_parts": "max(n_question_parts, 1)", "log_qlen": "log(question_len)",
    "coasker": "n_askers - 1", "topic": "TF-IDF+KMeans question topic (K=50)",
    "annexure_only": "text-based proxy (no outcome information): has_annexure & annexure_len == 0 & answer_len < 300; "
                     "baseline exclusion (plan §4). annexure_only_m2 (M2 ANNEXURE_ONLY on every part) is outcome-derived "
                     "and used only as a robustness exclusion.",
    "m2_weight": "M2 design weight N_h/n_h from m2_sample.csv (census LS17/18: 1; gold-only extras and undrawn: 0); "
                 "used for M2 outcomes in the pooled all-LS models (plan §4.3a)",
    "m2_role": "census | random | gold (labelled only for PPI++, weight 0) | out (not in the M2 sample)",
    "in_m2_sample": "m2_role != 'out' (question is sent to M2)",
}


# ----------------------------------------------------------------------------------- topics
def build_topics(seed=C.SEED_TOPICS, k=N_TOPICS):
    from sklearn.cluster import KMeans
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
    from sklearn.preprocessing import Normalizer
    import sklearn

    q = pd.read_parquet(C.QUESTIONS, columns=["qid", "subject", "question_text"])
    text = (q["subject"].fillna("") + " . " + q["question_text"].fillna("").str.slice(0, 1500)).str.lower()
    stop = sorted(set(ENGLISH_STOP_WORDS) | set(EXTRA_STOP))
    vec = TfidfVectorizer(stop_words=stop, min_df=50, max_df=0.30, max_features=30000, sublinear_tf=True,
                          token_pattern=r"(?u)\b[a-z][a-z]{2,}\b", dtype=np.float32)
    X = vec.fit_transform(text)
    print(f"tfidf {X.shape}", flush=True)
    svd = TruncatedSVD(n_components=100, random_state=seed, n_iter=5)
    Z = Normalizer(copy=False).fit_transform(svd.fit_transform(X))
    print("svd done", flush=True)
    km = KMeans(n_clusters=k, n_init=3, random_state=seed).fit(Z)
    lab = km.labels_
    out = pd.DataFrame({"qid": q["qid"].values, "topic": lab.astype(np.int16)})
    os.makedirs(C.MEASURES, exist_ok=True)
    out.to_parquet(TOPICS, index=False)
    terms = np.array(vec.get_feature_names_out())
    rows = []
    for t in range(k):
        m = np.asarray(X[lab == t].mean(axis=0)).ravel()
        rows.append({"topic": t, "n": int((lab == t).sum()), "top_terms": " ".join(terms[np.argsort(-m)[:12]])})
    pd.DataFrame(rows).to_csv(TOPIC_TERMS, index=False)
    meta = {"k": k, "seed": seed, "tfidf": {"min_df": 50, "max_df": 0.30, "max_features": 30000, "sublinear_tf": True},
            "svd_components": 100, "kmeans_n_init": 3, "text": "subject + question_text[:1500], lowercased",
            "sklearn": sklearn.__version__, "n_questions": len(q),
            "qid_hash": hashlib.sha256("\n".join(sorted(q["qid"])).encode()).hexdigest()[:16],
            "inertia": float(km.inertia_)}
    with open(TOPICS_META, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    print(pd.DataFrame(rows).to_string())
    return out


# ----------------------------------------------------------------------------------- covariates
def load_covariates():
    cols = ["qid", "ls", "session", "date", "qtype", "ministry", "n_askers", "n_members", "n_gov", "n_opp", "n_support",
            "asker_bloc", "asker_bloc_strict", "any_opposition", "any_split_party_window", "any_needs_review",
            "govt_period", "majority_type", "boilerplate", "statement_laid", "grouped_answer", "n_question_parts",
            "question_len", "answer_len", "answer_missing", "question_missing", "has_annexure", "annexure_len",
            "source_a", "minister_rank", "qa_mismatch", "answer_garbled"]
    import pyarrow.parquet as pq
    have = set(pq.read_schema(C.QUESTIONS).names)
    cols += [c for c in ("ministry_group", "answer_ocr") if c in have]
    q = pd.read_parquet(C.QUESTIONS, columns=cols)
    q["answer_ocr"] = q["answer_ocr"].fillna(False).astype(bool) if "answer_ocr" in q else False  # pipeline/apply_ocr.py
    q["date"] = pd.to_datetime(q["date"], errors="coerce")
    q["year"] = q["date"].dt.year + (q["date"].dt.dayofyear - 1) / 365.25
    q["session_id"] = q["ls"].astype(str) + "-" + q["session"].astype(str).str.zfill(2)

    mm = pd.read_csv(C.MINISTRY_MAP).drop_duplicates("ministry").set_index("ministry")
    q["ministry_canonical"] = q["ministry"].map(mm["ministry_canonical"])
    if "ministry_group" not in q:
        q["ministry_group"] = q["ministry"].map(mm["ministry_group"])
    q["ministry_group"] = q["ministry_group"].fillna("unmapped")
    q["ministry_canonical"] = q["ministry_canonical"].fillna(q["ministry"])
    q["security"] = (q["ministry_group"] == C.SECURITY_GROUP).astype(int)

    a = pd.read_parquet(C.ASKERS, columns=["qid", "asker", "mpsno"])
    first = a.groupby("qid", sort=False).first()
    key = first["mpsno"].map(lambda v: f"mp{int(v)}" if pd.notna(v) else None)
    key = key.fillna("nm:" + first["asker"].fillna("unknown").str.upper())
    q["mp_id"] = q["qid"].map(key).fillna("unknown")

    q["n_askers"] = q["n_askers"].fillna(q["n_members"]).fillna(1).clip(lower=1)
    q["coasker"] = q["n_askers"] - 1
    q["n_parts"] = q["n_question_parts"].clip(lower=1)
    q["log_qlen"] = np.log(q["question_len"].clip(lower=1))
    q["starred"] = (q["qtype"] == "starred").astype(int)
    for b, name in (("opposition", "opp"), ("mixed", "mixed"), ("outside_support", "outside")):
        q[name] = (q["asker_bloc"] == b).astype(int)
    q["opp_strict"] = (q["asker_bloc_strict"] == "opposition").astype(int)
    q["mixed_strict"] = (q["asker_bloc_strict"] == "mixed").astype(int)
    q["outside_strict"] = (q["asker_bloc_strict"] == "outside_support").astype(int)
    q["coalition_dep"] = (q["majority_type"] == "coalition-dependent").astype(int)
    q["ls18"] = (q["ls"] == 18).astype(int)
    q["annexure_proxy"] = (q["has_annexure"].astype(bool) & (q["annexure_len"] == 0) & (q["answer_len"] < 300))
    q["annexure_only"] = q["annexure_proxy"]
    for c in ("answer_missing", "question_missing", "statement_laid", "grouped_answer", "has_annexure",
              "any_split_party_window", "qa_mismatch", "answer_garbled"):
        q[c] = q[c].fillna(False).astype(bool)
    q["boilerplate_any"] = q["boilerplate"].map(lambda b: b is not None and len(b) > 0)
    if os.path.exists(TOPICS):
        q = q.merge(pd.read_parquet(TOPICS), on="qid", how="left")
    else:
        print("WARNING: topics.parquet missing -> topic FE unavailable (run with --topics)", file=sys.stderr)
        q["topic"] = np.nan
    return q.drop(columns=["boilerplate"])


# ----------------------------------------------------------------------------------- outcomes
def question_scores_from_m2(m2):
    m2 = m2.copy()
    m2["score"] = m2["label"].map(LABEL_SCORE)
    m2["is_full"] = (m2["label"] == "FULL").astype(float)
    g = m2.groupby("qid")
    out = pd.DataFrame({"y_m2": g["score"].mean(), "y_full": g["is_full"].mean(), "n_parts_labelled": g.size()})
    if "evasion_type" in m2:
        out["m2_annexure_only"] = (m2["evasion_type"] == "ANNEXURE_ONLY").groupby(m2["qid"]).all()
    if "request_type" in m2:
        shares = pd.crosstab(m2["qid"], m2["request_type"], normalize="index")
        for rt in ("INFORMATION", "REASON", "ACTION", "OPINION"):
            out["rt_" + rt.lower()] = shares[rt] if rt in shares else np.nan
    return out


def load_gold(coding_dir):
    """Question-level human gold. Returns DataFrame indexed by qid (or None):
    y_gold, gold_n_parts (labelled parts), gold_pi (main-sample inclusion prob = 1/weight), gold_sampled."""
    gp = os.path.join(coding_dir, "gold_labels.csv")
    if not os.path.exists(gp):
        return None
    g = pd.read_csv(gp)
    g["qid"] = g["part_id"].str.rsplit("#", n=1).str[0]
    g["score"] = g["label"].map(LABEL_SCORE)
    out = g.groupby("qid").agg(y_gold=("score", "mean"), gold_n_parts=("score", "size"))
    sp = os.path.join(coding_dir, "main_sample.csv")
    out["gold_pi"] = np.nan
    out["gold_sampled"] = False
    if os.path.exists(sp):
        s = pd.read_csv(sp)
        if "in_estimate" in s:
            s = s[s["in_estimate"].astype(bool)]
        if "sample" in s:
            s = s[s["sample"] == "main"]
        s = s.drop_duplicates("qid").set_index("qid")
        pi = s["pi"] if "pi" in s else 1.0 / s["weight"]
        out = out.join(pi.rename("gold_pi_s"), how="outer")
        out["gold_pi"] = out["gold_pi_s"]
        out["gold_sampled"] = out["gold_pi_s"].notna()
        out = out.drop(columns="gold_pi_s")
    else:
        print("WARNING: main_sample.csv missing -> gold_pi unknown", file=sys.stderr)
    return out


def pilot_qids(coding_dir):
    p = os.path.join(coding_dir, "pilot_sample.csv")
    return set(pd.read_csv(p, usecols=["qid"])["qid"]) if os.path.exists(p) else set()


def attach_outcomes(q, outcome):
    mdir = C.measures_dir(outcome)
    coding = C.CODING if outcome == "real" else C.SYN_MEASURES
    notes = []
    p = os.path.join(mdir, "m2_labels.parquet")
    if os.path.exists(p):
        m2 = pd.read_parquet(p)
        sc = question_scores_from_m2(m2)
        q = q.merge(sc, left_on="qid", right_index=True, how="left")
        if "m2_annexure_only" in q:
            # outcome-derived flag: used ONLY in the robustness check, never in the baseline exclusion
            q["annexure_only_m2"] = q["m2_annexure_only"].fillna(False).astype(bool)
        notes.append(f"m2 labels: {len(m2):,} parts -> {q['y_m2'].notna().sum():,} questions")
    else:
        for c in ("y_m2", "y_full"):
            q[c] = np.nan
        notes.append("m2_labels.parquet missing")
    p = os.path.join(mdir, "m1_question.parquet")
    ren = {"m1_cos": "y_m1", "part_coverage": "y_m1_cov", "m1_cos_alt": "y_m1b"}
    if os.path.exists(p):
        m1 = pd.read_parquet(p)
        if "m1_part_mean" in m1 and "m1_cos" not in m1:
            # REAL m1_embed.py output schema: question score = mean fine-tuned part cosine (m1_part_mean);
            # second encoder = the same column from m1_question__bge-base-en-v1.5.parquet (no coverage file yet)
            m1 = m1.rename(columns={"m1_part_mean": "y_m1"})
            pb = os.path.join(mdir, "m1_question__bge-base-en-v1.5.parquet")
            if os.path.exists(pb):
                b = pd.read_parquet(pb, columns=["qid", "m1_part_mean"]).rename(columns={"m1_part_mean": "y_m1b"})
                m1 = m1.merge(b, on="qid", how="left")
        m1 = m1.rename(columns=ren)
        q = q.merge(m1[["qid"] + [v for v in ren.values() if v in m1]], on="qid", how="left")
        notes.append(f"m1 question: {q['y_m1'].notna().sum():,}")
    elif os.path.exists(os.path.join(mdir, "m1_scores.parquet")):
        s = pd.read_parquet(os.path.join(mdir, "m1_scores.parquet"))
        if "qid" not in s:
            s["qid"] = s["part_id"].str.rsplit("#", n=1).str[0]
        q = q.merge(s.groupby("qid")["m1_cos"].mean().rename("y_m1"), left_on="qid", right_index=True, how="left")
        notes.append("m1 from part scores")
    pc = os.path.join(mdir, "m1_coverage.parquet")      # written by m1_embed.py --stage calibrate (Youden threshold)
    if os.path.exists(pc) and ("y_m1_cov" not in q or q["y_m1_cov"].isna().all()):
        cov = pd.read_parquet(pc).rename(columns={"m1_part_coverage": "y_m1_cov"})
        q = q.drop(columns=["y_m1_cov"], errors="ignore").merge(cov[["qid", "y_m1_cov"]], on="qid", how="left")
        notes.append(f"m1 coverage: {q['y_m1_cov'].notna().sum():,}")
    for c in ("y_m1", "y_m1_cov", "y_m1b"):
        if c not in q:
            q[c] = np.nan
    gold = load_gold(coding)
    # n_parts = max(n_question_parts, 1) equals parts.parquet n_parts (the sampler's part count, analysis/sample.py), so
    # questions with 0 parsed parts are one-part questions IN the sampling universe (2 of them were drawn into the gold).
    est_universe = ((~q["question_missing"]) & (~q["answer_missing"]) & (q["n_parts"] <= MAX_GOLD_PARTS)
                    & ~q["qid"].isin(pilot_qids(coding)))
    q["gold_universe"] = est_universe
    if gold is not None:
        q = q.merge(gold, left_on="qid", right_index=True, how="left")
        q["gold_sampled"] = q["gold_sampled"].fillna(False).astype(bool)
        q["gold"] = q["y_gold"].notna()
        q["gold_complete"] = q["gold"] & (q["gold_n_parts"] == q["n_parts"]) & q["gold_sampled"]
        notes.append(f"gold: sampled {int(q['gold_sampled'].sum())}, labelled {int(q['gold'].sum())}, "
                     f"complete {int(q['gold_complete'].sum())}")
    else:
        q["y_gold"], q["gold_pi"], q["gold_n_parts"] = np.nan, np.nan, np.nan
        q["gold"] = q["gold_sampled"] = q["gold_complete"] = False
        notes.append("gold labels missing")
    if outcome == "synthetic":      # oracle columns exist only for the synthetic outcome
        q = q.merge(pd.read_parquet(os.path.join(mdir, "truth_scores.parquet")), on="qid", how="left")
    q["outcome_source"] = outcome
    return q, notes


def attach_m2_design(q, outcome, path=None):
    """Merge the M2 sampling design (m2_weight, m2_role, in_m2_sample; see module docstring). Returns (q, note)."""
    path = path or os.path.join(C.measures_dir(outcome), "m2_sample.csv")
    if not os.path.exists(path):
        if outcome == "synthetic":
            return q.assign(m2_weight=1.0, m2_role="census", in_m2_sample=True), "m2 design: synthetic, all weights 1"
        print(f"WARNING: {path} missing -> m2_weight unknown (NaN); weighted M2 models will be empty", file=sys.stderr)
        return q.assign(m2_weight=np.nan, m2_role="unknown", in_m2_sample=False), "m2_sample.csv missing"
    s = pd.read_csv(path, usecols=["qid", "role", "m2_weight"]).rename(columns={"role": "m2_role"})
    if s["qid"].duplicated().any():
        raise ValueError(f"duplicate qids in {path}")
    unk = int((~s["qid"].isin(q["qid"])).sum())
    if unk:
        print(f"WARNING: {unk} qids in {path} are not in the question frame", file=sys.stderr)
    q = q.drop(columns=[c for c in ("m2_weight", "m2_role") if c in q]).merge(s, on="qid", how="left")
    q["m2_role"] = q["m2_role"].fillna("out")
    q["m2_weight"] = q["m2_weight"].fillna(0.0)
    q["in_m2_sample"] = q["m2_role"] != "out"
    note = (f"m2 design: {int(q['in_m2_sample'].sum()):,} sampled ({int((q['m2_role'] == 'gold').sum())} gold-only, "
            f"weight 0), weighted population {q['m2_weight'].sum():,.0f}")
    if "y_m2" in q and q["y_m2"].notna().any():
        out_lab = int((q["y_m2"].notna() & (q["m2_weight"] <= 0)).sum())
        unlab = int((q["y_m2"].isna() & q["in_m2_sample"] & ~q["answer_missing"]).sum())
        note += f"; M2-labelled with weight 0: {out_lab:,}; sampled+answered but unlabelled: {unlab:,}"
    return q, note


def add_sample_flags(q):
    """analysis_sample = pre-specified exclusions (plan §4.5): no answer text, qa_mismatch / answer_garbled flags,
    annexure-only pointer; plus no government period (8 undated questions), unresolved askers, unmapped ministry."""
    flagged = sum((q[c].fillna(False).astype(bool) for c in ("qa_mismatch", "answer_garbled") if c in q),
                  start=pd.Series(False, index=q.index))
    q["analysis_sample"] = (~q["answer_missing"] & ~flagged & ~q["annexure_only"] & q["govt_period"].notna()
                            & (q["asker_bloc"] != "unresolved") & (q["ministry_group"] != "unmapped"))
    return q


def build(outcome):
    q = load_covariates()
    if outcome == "synthetic":
        import synthetic
        synthetic.ensure(q)
    q, notes = attach_outcomes(q, outcome)
    q, note = attach_m2_design(q, outcome)
    notes.append(note)
    q = add_sample_flags(q)
    os.makedirs(os.path.dirname(C.frame_path(outcome)), exist_ok=True)
    q.to_parquet(C.frame_path(outcome), index=False)
    print(f"[{outcome}] frame {q.shape}; analysis sample {int(q['analysis_sample'].sum()):,}; " + "; ".join(notes))
    return q


def load_frame(outcome, rebuild=False):
    p = C.frame_path(outcome)
    if rebuild or not os.path.exists(p):
        return build(outcome)
    return pd.read_parquet(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outcome", choices=C.OUTCOMES, default="synthetic")
    ap.add_argument("--topics", action="store_true", help="(re)build topic clusters and exit")
    a = ap.parse_args()
    if a.topics:
        build_topics()
        return
    build(a.outcome)


if __name__ == "__main__":
    main()
