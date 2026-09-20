"""Tests for analysis/m1_embed.py (no GPU / no model download needed: a hashing stub encoder stands in).

    python -I -X utf8 -m pytest analysis/tests/test_m1.py -q
"""
import os
import sys
import zlib

import numpy as np
import pandas as pd
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "analysis"))
import m1_embed as m1  # noqa: E402


def stub_encode(texts, dim=64):
    """deterministic hashing bag-of-words encoder, L2-normalised"""
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for i, t in enumerate(texts):
        for w in t.lower().split():
            out[i, zlib.crc32(w.encode()) % dim] += 1.0
    n = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.maximum(n, 1e-9)


# ---------------------------------------------------------------------------- cleaning
def test_clean_answer_strips_laid_sentence_but_keeps_content():
    t = "A statement is laid on the Table of the House.\nSTATEMENT REFERRED TO IN REPLY OF PARTS (a) TO (d) OF LOK SABHA STARRED QUESTION NO.714 FOR 16.5.2000\nThe seat factor was 64.5%."
    c = m1.clean_answer(t)
    assert "laid" not in c.lower() and "REFERRED" not in c
    assert c == "The seat factor was 64.5%."
    assert m1.clean_answer("Statement `A` is laid on the Table of the House.\nData follows here.") == "Data follows here."
    assert m1.clean_answer("A statement is laid in the Table of the House. Figures: 5") == "Figures: 5"


def test_clean_answer_keeps_deferral_sentence():
    t = "Information is being collected and will be laid on the Table of the House."
    assert m1.clean_answer(t) == t


def test_clean_answer_minister_header_and_prose():
    hdr = "THE MINISTER OF STATE IN THE MINISTRY OF RAILWAYS (SHRI X)\nThe work is in progress."
    assert m1.clean_answer(hdr) == "The work is in progress."
    prose = "The Prime Minister`s 15 Point Programme is an on-going programme."
    assert m1.clean_answer(prose) == prose
    assert m1.clean_answer("ANSWER\nYes, Sir.") == "Yes, Sir."


def test_clean_answer_stars_labels_whitespace_idempotent():
    assert m1.clean_answer("(a) and (b) : Yes.\n********.") == "Yes."
    assert m1.clean_answer("(a) to (c): The  data\n\nis   here ****") == "The data is here"
    s = "(a) The list is at Annexure. *****"
    assert m1.clean_answer(m1.clean_answer(s)) == m1.clean_answer(s)
    assert m1.clean_answer(None) == "" and m1.clean_answer("  \n ") == ""
    assert m1.clean_answer("A statement is laid on the Table of the House.") == ""


# ---------------------------------------------------------------------------- split
def test_assign_split_by_qid_deterministic_and_stable():
    qids = [f"ls17-s1-S-{i}-20200101" for i in range(20000)]
    s1 = m1.assign_split(qids, 0.05, 0.01, seed=13)
    s2 = m1.assign_split(qids, 0.05, 0.01, seed=13)
    assert (s1 == s2).all()
    vc = pd.Series(s1).value_counts(normalize=True)
    assert abs(vc["train"] - 0.05) < 0.01 and abs(vc["val"] - 0.01) < 0.005
    # stable when the corpus grows: adding qids does not move existing ones
    more = qids + [f"ls18-s1-S-{i}-20250101" for i in range(500)]
    s3 = m1.assign_split(more, 0.05, 0.01, seed=13)
    assert (s3[:len(qids)] == s1).all()
    assert not (m1.assign_split(qids, 0.05, 0.01, seed=14) == s1).all()


def test_split_never_puts_a_qid_in_two_splits():
    parts = pd.DataFrame({"qid": np.repeat([f"q{i}" for i in range(3000)], 3)})
    sp = m1.assign_split(parts["qid"], 0.05, 0.01, 7)
    g = pd.DataFrame({"qid": parts["qid"], "sp": sp}).groupby("qid")["sp"].nunique()
    assert (g == 1).all()


# ---------------------------------------------------------------------------- training data
def test_build_training_pairs_filters():
    df = pd.DataFrame({
        "qid": ["a", "a", "b", "c", "d", "e", "f"],
        "split": ["train", "train", "train", "train", "val", "train", "train"],
        "q": ["what is x?", "what is y?", "what is z?", "what is w?", "what is v?", "", "what is x?"],
        "a": ["This answer is long enough to keep.", "short", "", "This answer is long enough to keep.",
              "A valid long answer text here.", "A valid long answer text here.", "This answer is long enough to keep."]})
    tr = m1.build_training_pairs(df, "train")
    # a/short dropped (<20), b/'' dropped (missing), e dropped (empty question), f duplicates (q,a) of row 0
    assert set(tr["qid"]) == {"a", "c"} and len(tr) == 2
    assert list(m1.build_training_pairs(df, "val")["qid"]) == ["d"]


def test_make_batches_no_false_negatives_and_deterministic():
    rng = np.random.default_rng(0)
    n = 500
    qids = [f"q{i // 3}" for i in range(n)]               # 3 sibling parts per question
    answers = [f"ans{rng.integers(0, 400)}" for _ in range(n)]  # some identical answer texts
    b1 = m1.make_batches(qids, answers, 32, seed=5)
    b2 = m1.make_batches(qids, answers, 32, seed=5)
    assert b1 == b2
    seen = set()
    for b in b1:
        assert len(b) <= 32 and len(b) >= 8
        assert len({qids[i] for i in b}) == len(b)
        assert len({answers[i] for i in b}) == len(b)
        assert not (set(b) & seen)
        seen |= set(b)
    assert len(seen) >= 0.9 * n
    assert m1.make_batches(qids, answers, 32, seed=6) != b1


def test_retrieval_metrics_perfect_and_random():
    e = np.eye(10, dtype=np.float32)
    r = m1.retrieval_metrics(e, e, np.arange(10))
    assert r["acc1"] == 1.0 and r["mrr"] == 1.0 and r["cos_pos"] == 1.0 and r["cos_neg"] == 0.0
    perm = np.roll(e, 1, axis=0)
    r2 = m1.retrieval_metrics(e, perm, np.arange(10))
    assert r2["acc1"] == 0.0
    # duplicate answers: candidates sharing a group count as correct
    r3 = m1.retrieval_metrics(e[:2], np.vstack([e[0], e[0]]), np.array([0, 0]))
    assert r3["acc1"] == 1.0


# ---------------------------------------------------------------------------- universes
def test_index_texts_dedup_and_empty():
    s = pd.Series(["Yes, Sir.", "A statement is laid on the Table of the House.", "Yes,   Sir.", None, "Other text here"])
    idx, texts, clipped = m1.index_texts(s, m1.clean_answer, 1000, prefix="doc: ")
    assert idx[0] == idx[2] >= 0                   # same after whitespace cleaning -> encoded once
    assert idx[1] == -1 and idx[3] == -1           # lay sentence only / None -> no text
    assert len(texts) == 2 and texts[idx[0]] == "doc: Yes, Sir."
    idx2, texts2, clipped2 = m1.index_texts(pd.Series(["x" * 50]), m1.clean_question, 10)
    assert texts2[0] == "x" * 10 and clipped2[0]


def _mini_frames():
    parts = pd.DataFrame({
        "part_id": ["q1#a", "q1#b", "q2#_all", "q3#a"], "qid": ["q1", "q1", "q2", "q3"],
        "part_label": ["a", "b", "_all", "a"],
        "question_part": ["how many railway stations", "the steps taken to increase it", "whole question two", ""],
        "answer_part": ["how many railway stations are there 100 stations", "Measures such as route rationalisation",
                        "", "something"],
        "answer_missing_for_part": [False, False, True, False]})
    q = pd.DataFrame({"qid": ["q1", "q2", "q3"], "question_text": ["how many railway stations and steps", "whole question two", "x"],
                      "answer_text": ["how many railway stations are there. Measures such as route rationalisation", "", "something"]})
    return parts, q


def test_build_universes_and_pipeline_with_stub(tmp_path):
    parts, q = _mini_frames()
    U = m1.build_universes(parts, q, 512)
    assert (U["part_a"][[2]] == -1).all() and U["part_q"][3] == -1
    assert U["ques_a"][1] == -1 and U["ques_a"][0] >= 0
    # full pipeline through the resumable store with the stub encoder
    rows = {}
    for side in "qa":
        texts = U[f"{side}_texts"]
        order = np.argsort([-len(t) for t in texts], kind="stable")
        rank = np.empty(len(order), dtype=np.int64)
        rank[order] = np.arange(len(order))
        store = m1.EmbedStore(str(tmp_path), side, len(texts), 64, "sig", chunk=2)
        m1.encode_universe(stub_encode, texts, store, order, batch_size=2, log_every=99)
        assert store.complete()
        rows[side] = (store, rank)
    toj = lambda idx, rank: np.where(idx >= 0, rank[np.clip(idx, 0, None)], -1)
    pc = m1.pair_cosines(rows["q"][0].mm, rows["a"][0].mm, toj(U["part_q"], rows["q"][1]), toj(U["part_a"], rows["a"][1]))
    assert np.isnan(pc[2]) and np.isnan(pc[3])          # missing answer / missing question -> NaN
    assert pc[0] > 0.5 and pc[1] == pytest.approx(0.0, abs=1e-6)  # lexical overlap vs none
    expected = float((stub_encode(["how many railway stations"]) @ stub_encode(["how many railway stations are there 100 stations"]).T)[0, 0])
    assert pc[0] == pytest.approx(expected, abs=2e-3)     # fp16 storage error only


def test_truncation_report():
    U = {"part_q": np.array([0, 1, -1]), "part_a": np.array([0, 1, 1]), "ques_q": np.array([0]), "ques_a": np.array([0]),
         "q_clipped": np.array([False, False]), "a_clipped": np.array([False, True])}
    r = m1.truncation_report(U, np.array([10, 600]), np.array([100, 100]), 512)
    assert r["part_question_pct_truncated"] == 50.0        # rows 0,1 valid; one truncated
    assert r["part_answer_pct_truncated"] == pytest.approx(200 / 3)  # clipped flag counts
    assert r["question_level_question_pct_truncated"] == 0.0


# ---------------------------------------------------------------------------- resume logic
def test_embedstore_resume_skips_done_chunks(tmp_path):
    texts = [f"text number {i} " * (1 + i % 5) for i in range(23)]
    order = np.argsort([-len(t) for t in texts], kind="stable")
    full = m1.EmbedStore(str(tmp_path / "full"), "x", len(texts), 64, "sigA", chunk=5)
    m1.encode_universe(stub_encode, texts, full, order, 5, log_every=99)
    ref = np.array(full.mm)

    calls = []

    class Boom(Exception):
        pass

    def flaky(batch):
        calls.append(len(batch))
        if len(calls) == 3:
            raise Boom()
        return stub_encode(batch)

    d = str(tmp_path / "res")
    st = m1.EmbedStore(d, "x", len(texts), 64, "sigA", chunk=5)
    with pytest.raises(Boom):
        m1.encode_universe(flaky, texts, st, order, 5, log_every=99)
    assert st.done == {0, 1}
    del st
    calls.clear()
    st2 = m1.EmbedStore(d, "x", len(texts), 64, "sigA", chunk=5)   # "new process"
    assert st2.done == {0, 1}
    m1.encode_universe(lambda b: (calls.append(len(b)), stub_encode(b))[1], texts, st2, order, 5, log_every=99)
    assert sum(calls) == len(texts) - 10                       # only unfinished chunks re-encoded
    assert st2.complete()
    assert np.allclose(np.array(st2.mm), ref, atol=1e-3)
    del st2
    # signature change (different model / texts) -> start afresh, nothing skipped
    st3 = m1.EmbedStore(d, "x", len(texts), 64, "sigB", chunk=5)
    assert st3.done == set()


def test_pair_cosines_blocks_and_nan():
    rng = np.random.default_rng(1)
    q = rng.normal(size=(50, 8)).astype(np.float32)
    a = rng.normal(size=(50, 8)).astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    qr = np.array([0, 5, -1, 7, 9])
    ar = np.array([1, 5, 3, -1, 9])
    c = m1.pair_cosines(q.astype(np.float16), a.astype(np.float16), qr, ar, block=2)
    assert np.isnan(c[2]) and np.isnan(c[3])
    assert c[1] == pytest.approx(float(q[5] @ a[5]), abs=2e-3)


# ---------------------------------------------------------------------------- aggregation
def test_aggregate_question():
    scores = pd.DataFrame({"qid": ["q1", "q1", "q1", "q2", "q3"], "m1_cos": [0.2, 0.6, np.nan, 0.5, np.nan],
                           "m1_cos_untuned": [0.1, 0.5, np.nan, 0.4, np.nan]})
    out = m1.aggregate_question(scores, ["q1", "q2", "q3", "q4"], [0.3, 0.4, np.nan, np.nan], [0.2, 0.3, np.nan, np.nan])
    o = out.set_index("qid")
    assert o.loc["q1", "m1_part_mean"] == pytest.approx(0.4) and o.loc["q1", "m1_part_min"] == pytest.approx(0.2)
    assert o.loc["q1", "n_parts_scored"] == 2 and o.loc["q3", "n_parts_scored"] == 0 and o.loc["q4", "n_parts_scored"] == 0
    assert np.isnan(o.loc["q3", "m1_part_mean"]) and o.loc["q2", "m1_q_cos"] == pytest.approx(0.4)
    assert o.loc["q1", "m1_part_mean_untuned"] == pytest.approx(0.3)


# ---------------------------------------------------------------------------- calibration
def _synthetic(n=400, seed=3):
    rng = np.random.default_rng(seed)
    full = rng.normal(0.70, 0.10, n)
    none = rng.normal(0.30, 0.10, n)
    part = rng.normal(0.50, 0.12, n // 2)
    scores = pd.DataFrame({"part_id": [f"q{i}#a" for i in range(n * 2 + n // 2)],
                           "qid": [f"q{i}" for i in range(n * 2 + n // 2)],
                           "m1_cos": np.concatenate([full, none, part])})
    labels = pd.DataFrame({"part_id": scores["part_id"],
                           "label": ["FULL"] * n + ["NONE"] * n + ["PARTIAL"] * (n // 2)})
    return scores, labels


def test_calibrate_threshold_synthetic():
    scores, labels = _synthetic()
    r = m1.calibrate_threshold(scores, labels, n_boot=50)
    assert 0.42 < r["threshold"] < 0.58
    assert r["auc"] > 0.97 and r["J"] > 0.85
    assert r["n_full"] == 400 and r["n_none"] == 400 and r["n_partial"] == 200
    lo, hi = r["auc_ci95"]
    assert lo <= r["auc"] <= hi + 1e-9
    # PARTIAL sits between FULL and NONE
    m = r["mean_score_by_label"]
    assert m["NONE"] < m["PARTIAL"] < m["FULL"]
    # youden threshold really maximises J: brute-force check
    fn = scores.merge(labels, on="part_id")
    fn = fn[fn["label"] != "PARTIAL"]
    y = (fn["label"] == "FULL").to_numpy()
    s = fn["m1_cos"].to_numpy()
    best = max(((s[y] >= t).mean() - (s[~y] >= t).mean()) for t in np.unique(s))
    assert r["J"] == pytest.approx(best)


def test_calibrate_requires_both_classes_and_handles_multi_coder():
    scores, labels = _synthetic(50)
    with pytest.raises(ValueError):
        m1.calibrate_threshold(scores, labels[labels["label"] != "NONE"], n_boot=0)
    # two coders: second coder agrees on all but disagrees on a few; ties are dropped, majority kept
    l2 = pd.concat([labels, labels.assign(label=labels["label"])])
    r = m1.calibrate_threshold(scores, l2, n_boot=0)
    assert r["multi_coder_majority"] and r["n_full"] == 50
    l3 = pd.concat([labels, labels.assign(label=labels["label"].map({"FULL": "NONE", "NONE": "FULL", "PARTIAL": "PARTIAL"}))])
    with pytest.raises(ValueError):
        m1.calibrate_threshold(scores, l3, n_boot=0)


def test_part_coverage():
    d = pd.DataFrame({"qid": ["a", "a", "a", "b", "c"], "m1_cos": [0.9, 0.2, np.nan, 0.8, np.nan]})
    cov = m1.part_coverage(d, 0.5).set_index("qid")["m1_part_coverage"]
    assert cov["a"] == 0.5 and cov["b"] == 1.0 and "c" not in cov.index


def test_run_calibrate_cli(tmp_path):
    scores, labels = _synthetic(100)
    scores["qid"] = scores["part_id"].str.split("#").str[0]
    sp, _ = m1.out_paths(m1.DEFAULT_MODEL, str(tmp_path))
    scores.to_parquet(sp)
    labels.to_csv(tmp_path / "labels.csv", index=False)
    args = m1.build_parser().parse_args(["--stage", "calibrate", "--out-dir", str(tmp_path),
                                         "--labels", str(tmp_path / "labels.csv")])
    res = m1.run_calibrate(args)
    assert res["auc"] > 0.95
    assert os.path.exists(sp.replace("m1_scores", "m1_coverage"))


# ---------------------------------------------------------------------------- misc
def test_out_paths_and_slug(tmp_path):
    assert os.path.basename(m1.out_paths(m1.DEFAULT_MODEL, str(tmp_path))[0]) == "m1_scores.parquet"
    assert os.path.basename(m1.out_paths("BAAI/bge-base-en-v1.5", str(tmp_path))[1]) == "m1_question__bge-base-en-v1.5.parquet"


def test_estimate_run_scales():
    e = m1.estimate_run(200_000_000, 20_000_000, 1, 2, 150_000)
    assert e["encode_s"] == pytest.approx(2 * 200e6 / 150e3)
    assert e["usd"] == pytest.approx(e["total_s"] / 3600 * m1.A100_USD_PER_HOUR)


@pytest.mark.skipif(not os.path.exists(m1.PARTS_PATH), reason="parts.parquet not built")
def test_real_parts_subset_prep():
    parts, q = m1.load_data(m1.PARTS_PATH, m1.QUESTIONS_PATH, limit_parts=1000)
    assert 1000 <= len(parts) < 1400
    assert set(parts["qid"]) == set(q["qid"]) and len(q) == parts["qid"].nunique()
    split = m1.assign_split(parts["qid"], 0.3, 0.1, 13)
    assert pd.DataFrame({"q": parts["qid"], "s": split}).groupby("q")["s"].nunique().max() == 1
    U = m1.build_universes(parts, q, 512)
    assert len(U["part_q"]) == len(parts) and len(U["ques_a"]) == len(q)
    miss = parts["answer_missing_for_part"].to_numpy()
    assert (U["part_a"][miss] == -1).all()
    # deferral boilerplate survives cleaning; no raw lay sentence is left
    assert not any("statement is laid" in t.lower() for t in U["a_texts"])


# ====================================================================================== parity_check
import parity_check as pc  # noqa: E402


def test_parity_allocate_properties():
    counts = {f"m{i}": c for i, c in enumerate([500, 300, 120, 60, 30, 10, 3, 1])}
    a = pc.allocate(counts, 100, min_each=2)
    assert sum(a.values()) == 100
    assert all(0 <= a[k] <= counts[k] for k in counts)
    assert a["m7"] == 1 and a["m6"] >= 2            # tiny strata: min_each capped at availability
    assert a["m0"] > a["m1"] > a["m2"]               # roughly proportional
    b = pc.allocate(counts, 3, min_each=2)           # fewer picks than strata
    assert sum(b.values()) == 3
    assert sum(pc.allocate({"a": 2, "b": 1}, 10).values()) == 3  # more requested than available


def _fake_ls17(n=2000):
    rng = np.random.default_rng(0)
    mins = rng.choice([f"MIN{i}" for i in range(12)], n, p=np.r_[np.full(11, 0.08), 0.12])
    return pd.DataFrame({
        "qid": [f"ls17-s1-S-{i}-20200101" for i in range(n)], "ministry": mins,
        "answer_text": ["x" * 80] * n, "docx_url": ["https://sansad.in/a.docx"] * (n - 100) + [None] * 100,
        "pdf_url": [" "] * n, "answer_missing": False, "qa_mismatch": False, "question_missing": False})


def test_parity_select_sample_deterministic_stratified():
    df = _fake_ls17()
    s1 = pc.select_sample(df, 120, seed=1)
    s2 = pc.select_sample(df, 120, seed=1)
    assert s1.equals(s2) and len(s1) == 120 and s1["qid"].is_unique
    assert s1["ministry"].nunique() == 12 and s1.groupby("ministry").size().min() >= 2
    assert s1["docx_url"].notna().all()                  # only questions with a link
    assert not s1["qid"].equals(pc.select_sample(df, 120, seed=2)["qid"])
    assert pc.links_of({"docx_url": "https://x/a.docx", "pdf_url": " "}) == [("https://x/a.docx", ".docx")]
    assert [e for _, e in pc.links_of({"docx_url": "https://x/a.docx", "pdf_url": "https://x/a.pdf"})] == [".docx", ".pdf"]


def test_parity_text_metrics():
    a = "The  seat factor was 64.5% in 1999."
    assert pc.char_similarity(a, a.upper()) == 1.0
    assert pc.content_similarity("Yes, Sir. (a) 12", "yes sir a 12") == 1.0
    assert pc.char_similarity("", "") == 1.0 and pc.char_similarity("abc", "") == 0.0
    assert 0.3 < pc.char_similarity("rail safety measures taken", "airline route rationalisation") < 0.7
    ap = [{"label": "a", "labels": ["a", "b"]}, {"label": "c", "labels": ["c"]}, {"label": "_pre", "labels": []}]
    assert pc.n_labelled(ap) == 2 and pc.label_set(ap) == {"a", "b", "c"}
    assert pc.jaccard({"a", "b"}, {"a", "b", "c"}) == pytest.approx(2 / 3)
    row = pc.compare_one({"answer_text": "x y z", "answer_parts": ap, "statement_laid": False},
                         {"answer_text": "x y z", "answer_parts": ap[:1], "statement_laid": True})
    assert row["char_sim"] == 1.0 and not row["parts_equal"]
    assert row["n_parts_file"] == 1 and row["statement_laid_file"] is True


def test_parity_statistics():
    rng = np.random.default_rng(5)
    a = rng.uniform(0.2, 0.8, 300)
    b = a + rng.normal(0.0, 0.01, 300)
    ba = pc.bland_altman(a, b)
    assert abs(ba["bias"]) < 0.005 and ba["loa"][0] < 0 < ba["loa"][1] and ba["loa"][1] < 0.04
    assert ba["bias_ci"][0] <= ba["bias"] <= ba["bias_ci"][1]
    t = pc.paired_tests(a, b, margin=0.03)
    assert t["equivalent"] and t["tost_p"] < 0.05 and t["ccc"] > 0.99 and t["wilcoxon_p"] > 0.01
    t2 = pc.paired_tests(a, a + 0.05 + rng.normal(0, 0.01, 300), margin=0.03)  # a real shift is detected, not equivalent
    assert t2["t_p"] < 1e-6 and t2["wilcoxon_p"] < 1e-6 and not t2["equivalent"]
    assert pc.bland_altman(a, a + 0.05)["bias"] == pytest.approx(0.05)
    b3 = b.copy()
    b3[:10] = np.nan                                                            # NaNs dropped pairwise
    assert pc.paired_tests(a, b3)["n"] == 290
    pb = pc.bland_altman(a, a * 1.2)                                            # proportional bias detected
    assert pb["prop_bias_slope"] > 0.1 and pb["prop_bias_p"] < 1e-6


def test_parity_cluster_bootstrap_and_kappa():
    rng = np.random.default_rng(2)
    df = pd.DataFrame({"qid_base": np.repeat([f"q{i}" for i in range(80)], 3)})
    df["a"] = rng.uniform(0, 1, len(df))
    df["b"] = df["a"] + 0.02
    r = pc.cluster_bootstrap_mean_diff(df, "a", "b", n_boot=300)
    assert r["mean_diff"] == pytest.approx(0.02) and r["n_clusters"] == 80 and r["n_parts"] == 240
    assert r["ci"][0] <= 0.02 + 1e-6 <= r["ci"][1] + 1e-6
    assert pc.cohen_kappa([1, 1, 0, 0], [1, 1, 0, 0]) == 1.0


def test_parity_m1_pairing_and_report(tmp_path):
    """Synthetic m1 outputs for the two versions -> paired analysis + report file (no model needed)."""
    rng = np.random.default_rng(9)
    base = rng.uniform(0.2, 0.8, 60)
    s_rows, q_rows = [], []
    for i in range(60):
        parts_api = np.clip(base[i] + rng.normal(0, 0.1, 2), 0, 1)
        for ver in ("api", "file"):
            vals = parts_api + (rng.normal(0, 0.005, 2) if ver == "file" else 0)
            for j, lab in enumerate("ab"):
                s_rows.append({"part_id": f"q{i}@{ver}#{lab}", "qid": f"q{i}@{ver}", "part_label": lab,
                               "m1_cos": vals[j], "m1_cos_untuned": vals[j] * 0.9})
            q_rows.append({"qid": f"q{i}@{ver}", "m1_q_cos": vals.mean(), "m1_part_mean": vals.mean(), "m1_part_min": vals.min(),
                           "m1_q_cos_untuned": vals.mean() * 0.9, "m1_part_mean_untuned": vals.mean() * 0.9})
    sp, qp = m1.out_paths(m1.DEFAULT_MODEL, str(tmp_path))
    pd.DataFrame(s_rows).to_parquet(sp)
    pd.DataFrame(q_rows).to_parquet(qp)
    res = pc.m1_parity(str(tmp_path), margin=0.03, threshold=0.5)
    assert res["n_questions_scored"] == 60 and res["n_parts_paired"] == 120
    assert abs(res["question_part_mean"]["bland_altman"]["bias"]) < 0.01
    assert res["question_part_mean"]["tests"]["equivalent"]
    assert res["part_level"]["cluster_bootstrap"]["n_clusters"] == 60
    assert res["coverage_classification"]["agreement"] > 0.9
    pairs = pd.DataFrame({"qid": [f"q{i}" for i in range(60)], "ministry": "M", "ext": ["docx"] * 50 + ["pdf"] * 10,
                          "doc_note": [None] * 60, "found_heading": True, "extract_error": None, "doc_pages": 1,
                          "q_char_sim": 0.99, "file_answer_empty": False, "api_len": 100, "file_len": 99, "len_ratio": 0.99,
                          "char_sim": rng.uniform(0.9, 1, 60), "content_sim": rng.uniform(0.95, 1, 60), "n_parts_api": 2,
                          "n_parts_file": 2, "parts_equal": True, "label_jaccard": 1.0, "statement_laid_api": False,
                          "statement_laid_file": False})
    out = tmp_path / "rep.md"
    txt = pc.write_report(pairs, res, str(out), 60, 0.03)
    assert out.exists() and "limits of agreement" in txt and "question_part_mean" in txt
    txt2 = pc.write_report(pairs, None, str(tmp_path / "rep2.md"), 60, 0.03)
    assert "not available yet" in txt2


@pytest.mark.skipif(not os.path.exists(pc.CLEAN_PATH), reason="clean parquet not built")
def test_parity_extract_roundtrip_with_real_ls17_text(tmp_path):
    """Write real 17th-LS API answers into .docx files laid out like the Secretariat's replies, push them through the
    pipeline's 18th-LS document path, and check the extraction reproduces the API answer (machinery check, no network)."""
    docx = pytest.importorskip("docx")
    df = pd.read_parquet(pc.CLEAN_PATH, columns=["qid", "ls", "question_text", "answer_text", "question_parts", "answer_parts",
                                                 "statement_laid", "n_question_parts", "n_answer_parts"],
                         filters=[("ls", "==", 17)])
    df = df[(df["n_question_parts"] >= 2) & (df["n_answer_parts"] >= 2) & (df["answer_text"].str.len().between(200, 1500))
            & ~df["statement_laid"]].head(8)
    assert len(df) == 8
    sims, equal = [], []
    for r in df.itertuples(index=False):
        d = docx.Document()
        for l in ["GOVERNMENT OF INDIA", "MINISTRY OF TEST", "LOK SABHA", "UNSTARRED QUESTION NO. 1", "TO BE ANSWERED ON 1.1.2020",
                  "TEST SUBJECT", "SHRI A B:"]:
            d.add_paragraph(l)
        d.add_paragraph("Will the Minister of TEST be pleased to state:")
        for l in r.question_text.split("\n"):
            d.add_paragraph(l)
        d.add_paragraph("ANSWER")
        d.add_paragraph("THE MINISTER OF STATE IN THE MINISTRY OF TEST (SHRI X Y)")
        for l in r.answer_text.split("\n"):
            d.add_paragraph(l)
        path = tmp_path / f"{r.qid}.docx"
        d.save(str(path))
        fx = pc.extract_file_answer(str(path), r.question_parts, str(tmp_path / "cache"))
        assert fx["found_heading"] and fx["doc_note"] is None
        row = pc.compare_one({"answer_text": r.answer_text, "answer_parts": pc._plain(r.answer_parts), "statement_laid": False}, fx)
        sims.append(row["content_sim"])
        equal.append(row["parts_equal"])
        rows = pc.build_part_rows(r.qid, r.question_text, pc._plain(r.question_parts), fx["answer_text"], fx["answer_parts"], "file")
        assert rows["part_id"].str.contains("@file#").all() and (rows["qid_base"] == r.qid).all()
        assert set(rows.columns) >= {"part_id", "qid", "part_label", "question_part", "answer_part", "answer_missing_for_part"}
    assert np.mean(sims) > 0.97 and np.mean(equal) >= 0.75


def test_parity_list_dry_run_makes_no_network_call(tmp_path, monkeypatch):
    if not os.path.exists(pc.CLEAN_PATH):
        pytest.skip("clean parquet not built")
    monkeypatch.setitem(sys.modules, "collect", None)  # importing collect (the downloader) would now fail
    args = pc.build_parser().parse_args(["--list", "--n", "30", "--sample-csv", str(tmp_path / "s.csv"), "--outdir", str(tmp_path / "o")])
    s = pc.run_list(args)
    assert len(s) == 30 and (tmp_path / "s.csv").exists() and not (tmp_path / "o").exists()
