"""Tests for the robustness additions (plan §4.6 items 12 and 13, coverage check, variant summary).

    python -I -X utf8 -m pytest analysis/tests/test_robustness.py -q
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import robustness as R  # noqa: E402
from test_analysis import _fake_frame  # noqa: E402


def test_unseen_annexure_flag_is_text_based():
    d = pd.DataFrame({"has_annexure": [True, True, False, False, True],
                      "annexure_len": [0, 120, 0, 50, np.nan],
                      "answer_len": [50, 50, 5000, 10, 9000]})
    # has_annexure & no annexure text, whatever the answer length (unlike the <300-char baseline proxy)
    assert list(R.unseen_annexure(d)) == [True, False, False, False, True]


def test_gold_note_flags_match_split_or_garbled_case_insensitive(tmp_path):
    pd.DataFrame({"part_id": ["q1#a", "q1#b", "q2#a", "q3#a", "q4#a"],
                  "qid": ["q1", "q1", "q2", "q3", "q4"],
                  "note": ["Split, no text for (a)", "", "(c) cell GARBLED", "answer fine", np.nan]}
                 ).to_csv(tmp_path / "human_labels.csv", index=False)
    pd.DataFrame({"part_id": ["q5#a", "q6#a"], "qid": ["q5", "q6"],
                  "coder1_note": ["", "ok"], "coder2_note": [np.nan, ""],
                  "adjudicator_note": ["mis-split part", "fine"]}).to_csv(tmp_path / "adjudication.csv", index=False)
    qids, cnt = R.gold_note_flagged_qids(str(tmp_path))
    assert qids == {"q1", "q2", "q5"}
    assert cnt == {"n_notes": 3, "n_questions": 3}
    assert R.gold_note_flagged_qids(str(tmp_path / "nowhere"))[0] == set()


def test_plan_coverage_detects_missing_items():
    full = ["baseline", "fractional logit (AME)", "outcome: M1 cosine", "outcome: M1 part coverage",
            "outcome: second embedding model", "exclude grouped answers", "starred only", "unstarred only",
            "drop ministry: X", "single-asker questions only", "strict party coding", "drop split-party windows",
            "LS15 vs LS14", "LS17 vs LS16", "within-LS17 trend per year", "within-LS18 trend per year",
            "[12] exclude unseen annexures", "[13] PPI++ excluding", "[14] M2 unweighted", "[15a] M1", "[15b] M1",
            "[16] exclude OCR"]
    assert R.plan_coverage(full) == {}
    miss = R.plan_coverage([n for n in full if not n.startswith("[13]") and "LS17 vs LS16" not in n])
    assert miss == {9: ["LS17 vs LS16"], 13: ["[13]"]}


def _frame_with_gold(n=6000, seed=3):
    rng = np.random.default_rng(seed)
    d = _fake_frame(n=n, seed=seed)
    d["qid"] = [f"q{i}" for i in range(n)]
    d["y_m2"] = d["y_true"].clip(0, 1)
    d["m2_weight"] = 1.0
    d["outcome_source"] = "real"
    d["has_annexure"] = rng.random(n) < 0.3
    d["annexure_len"] = np.where(d["has_annexure"] & (rng.random(n) < 0.5), 0, 200)
    d["gold_universe"] = True
    R_ = rng.random(n) < 0.06
    d["gold_sampled"] = R_
    d["gold_complete"] = R_
    d["gold_pi"] = np.where(R_, 0.06, np.nan)
    d["y_gold"] = np.where(R_, (d["y_m2"] + rng.normal(0, 0.15, n)).clip(0, 1), np.nan)
    return d


def test_item12_variant_present_and_excludes_exactly_the_flagged_rows():
    d = _frame_with_gold(n=800)
    d = d.assign(grouped_answer=False, statement_laid=False, asker_bloc="government", n_askers=1,
                 minister_rank="Cabinet", answer_ocr=False)
    V = {v[0]: v for v in R.standard_variants(d, "y_m2", "ols")}
    name = next(k for k in V if k.startswith("[12]"))
    sub = V[name][1]
    assert len(sub) == int((~R.unseen_annexure(d)).sum()) < len(d)
    assert not R.unseen_annexure(sub).any()
    assert V[name][2] == "y_m2"


def test_item13_ppi_variant_drops_flagged_gold_questions():
    d = _frame_with_gold()
    gold_q = list(d.loc[d["gold_complete"], "qid"])
    flagged = set(gold_q[:20]) | {"not-a-question"}
    out = R.ppi_variants(d, "y_m2", "ols", flagged=flagged)
    base = out[out["variant"].str.startswith("PPI++ baseline")]
    v13 = out[out["variant"].str.startswith("[13]")]
    assert set(base["hypothesis"]) == set(v13["hypothesis"]) == {"H1", "H2", "H3", "H4"}
    pooled = ["H1", "H3"]          # all-LS specs: n_gold = every complete gold question (H2/H4 use LS17-18 only)
    assert (base[base["hypothesis"].isin(pooled)]["n_gold"] == len(gold_q)).all()
    assert (v13[v13["hypothesis"].isin(pooled)]["n_gold"] == len(gold_q) - 20).all()
    assert (v13["n_gold"] <= base["n_gold"].values).all() and (v13["n_gold"] < base["n_gold"].values).any()
    assert (v13["n_flagged_gold_dropped"] == 20).all() and (base["n_flagged_gold_dropped"] == 0).all()
    # same code path: with nothing flagged the [13] variant reproduces the baseline exactly
    same = R.ppi_variants(d, "y_m2", "ols", flagged=set())
    a = same[same["variant"].str.startswith("PPI++ baseline")][["hypothesis", "method", "estimate", "se"]].reset_index(drop=True)
    b = same[same["variant"].str.startswith("[13]")][["hypothesis", "method", "estimate", "se"]].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)
    assert R.ppi_variants(d, "y_m1", "ols", flagged=flagged).empty      # M2 outcomes only


def test_summarise_variants_counts_sign_and_extremes():
    rows = []
    for h, base, others in (("H1", -0.01, [-0.02, 0.005, -0.03]), ("H2", 0.1, [0.2, 0.05, 0.3])):
        for nm, e in [("baseline", base)] + [(f"v{i}", o) for i, o in enumerate(others)]:
            rows.append({"variant": nm, "hypothesis": h, "estimate": e, "p": 0.01 if abs(e) > 0.02 else 0.5})
    s = R.summarise_variants(pd.DataFrame(rows)).set_index("hypothesis")
    assert s.loc["H1", "n_same_sign"] == 2 and s.loc["H1", "sign_flips"] == "v1"
    assert s.loc["H1", "min"] == -0.03 and s.loc["H1", "max"] == 0.005
    assert s.loc["H2", "n_same_sign"] == 3 and s.loc["H2", "sign_flips"] == "none"
    assert s.loc["H2", "n_variants"] == 3 and s.loc["H1", "n_p_lt_05"] == 1
