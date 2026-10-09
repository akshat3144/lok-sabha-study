"""Tests for M3 (sampling, coding sheets, validation). All data are synthetic; nothing needs the real corpus.

    python -I -X utf8 -m pytest analysis/tests/test_m3.py -q
"""
import os
import shutil
import sys

import numpy as np
import pandas as pd
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import codebook as cb  # noqa: E402
import coding_sheet as cs  # noqa: E402
import sample as sm  # noqa: E402
import validate as va  # noqa: E402
from openpyxl import load_workbook  # noqa: E402

N = np.nan
# Krippendorff (2011) "Computing Krippendorff's alpha-reliability", 4 coders x 12 units, values 1..5
KRIP = [[1, 2, 3, 3, 2, 1, 4, 1, 2, N, N, N],
        [1, 2, 3, 3, 2, 2, 4, 1, 2, 5, N, 3],
        [N, 3, 3, 3, 2, 3, 4, 2, 2, 5, 1, N],
        [1, 2, 3, 3, 2, 4, 4, 1, 2, 5, 1, N]]


# ============================================================================================ agreement statistics
def test_krippendorff_published_example():
    cats = [1, 2, 3, 4, 5]
    assert va.krippendorff_alpha(KRIP, "nominal", cats) == pytest.approx(0.743, abs=5e-4)
    assert va.krippendorff_alpha(KRIP, "ordinal", cats) == pytest.approx(0.815, abs=5e-4)
    assert va.krippendorff_alpha(KRIP, "interval", cats) == pytest.approx(0.849, abs=5e-4)


def test_alpha_extremes():
    same = [["A", "B", "C", "A"], ["A", "B", "C", "A"]]
    assert va.krippendorff_alpha(same, "nominal", ["A", "B", "C"]) == pytest.approx(1.0)
    # systematic disagreement (a permutation of categories) gives alpha < 0
    swapped = [["A", "B", "C", "A", "B", "C"], ["B", "C", "A", "B", "C", "A"]]
    assert va.krippendorff_alpha(swapped, "nominal", ["A", "B", "C"]) < 0
    # a unit with a single rating is ignored
    one = [["A", "B", None], ["A", "B", "C"]]
    assert va.krippendorff_alpha(one, "nominal", ["A", "B", "C"]) == pytest.approx(1.0)


def test_ordinal_penalises_far_disagreements_more():
    cats = ["NONE", "PARTIAL", "FULL"]
    near = [["NONE", "PARTIAL", "FULL", "NONE", "PARTIAL", "FULL"], ["PARTIAL", "PARTIAL", "FULL", "NONE", "FULL", "FULL"]]
    far = [["NONE", "PARTIAL", "FULL", "NONE", "PARTIAL", "FULL"], ["FULL", "PARTIAL", "FULL", "NONE", "PARTIAL", "NONE"]]
    assert va.krippendorff_alpha(near, "ordinal", cats) > va.krippendorff_alpha(far, "ordinal", cats)


def test_kappa_and_metrics_match_sklearn():
    from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
    rng = np.random.default_rng(0)
    labs = va.ORD_LABELS
    a = rng.choice(labs, 300, p=[.3, .3, .4])
    b = np.where(rng.random(300) < .7, a, rng.choice(labs, 300))
    C = va.confusion(a, b, labs)
    m = va.metrics_from_confusion(C)
    assert m["kappa"] == pytest.approx(cohen_kappa_score(a, b, labels=labs))
    assert m["kappa_lw"] == pytest.approx(cohen_kappa_score(a, b, labels=labs, weights="linear"))
    assert m["accuracy"] == pytest.approx(accuracy_score(a, b))
    assert m["macro_f1"] == pytest.approx(f1_score(a, b, labels=labs, average="macro"))
    # weighted confusion with integer weights == repeated rows
    w = rng.integers(1, 4, 300)
    Cw = va.confusion(a, b, labs, w)
    assert va.kappa_from_confusion(Cw) == pytest.approx(
        cohen_kappa_score(np.repeat(a, w), np.repeat(b, w), labels=labs))


def test_auc_youden_spearman():
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(1)
    pos = rng.random(200) < .5
    score = np.where(pos, rng.normal(.6, .2, 200), rng.normal(.4, .2, 200)).round(2)  # ties on purpose
    assert va.auc_rank(score, pos) == pytest.approx(roc_auc_score(pos, score))
    t, tpr, fpr, J = va.youden_threshold(score, pos)
    assert J == pytest.approx(((score >= t) & pos).sum() / pos.sum() - ((score >= t) & ~pos).sum() / (~pos).sum())
    # brute-force optimum
    best = max(((score >= u) & pos).sum() / pos.sum() - ((score >= u) & ~pos).sum() / (~pos).sum()
               for u in np.unique(score))
    assert J == pytest.approx(best)
    from scipy.stats import spearmanr
    x, y = rng.random(50), rng.random(50)
    assert va.spearman(x, y) == pytest.approx(spearmanr(x, y)[0])


# ============================================================================================ sampling
def synth_universe(n=30000, seed=0):
    rng = np.random.default_rng(seed)
    u = pd.DataFrame({
        "qid": [f"q{i:06d}" for i in range(n)],
        "ls": rng.choice([13, 14, 15, 16, 17, 18], n),
        "qtype": rng.choice(["starred", "unstarred"], n, p=[.08, .92]),
        "ministry": rng.choice(["HOME AFFAIRS", "DEFENCE", "FINANCE", "RAILWAYS", "EXTERNAL AFFAIRS"], n),
        "boilerplate_flag": rng.random(n) < .28,
        "no_answer": rng.random(n) < .02,
        "no_question": False,
        "bad_quality": rng.random(n) < .01,
        "n_parts": rng.choice(range(1, 10), n, p=[.01, .05, .2, .3, .25, .1, .05, .03, .01]),
        "m1_mean": np.nan})
    u["ministry_group"] = u["ministry"].map(sm.provisional_ministry_group)
    return u


def test_allocate_properties():
    sizes = pd.Series({"a": 5, "b": 1, "c": 1000, "d": 40000, "e": 300})
    al = sm.allocate(sizes, 100)
    assert al.sum() == 100 and (al <= sizes).all() and al["b"] == 1
    assert al["d"] > al["c"] > al["e"] > al["a"]
    assert (al >= np.minimum(2, sizes)).all()
    assert sm.allocate(pd.Series({"a": 3, "b": 2}), 100).sum() == 5  # cannot exceed the population
    assert sm.allocate(pd.Series({"a": 10, "b": 10}), 0).empty


def test_sample_properties(tmp_path):
    u = synth_universe()
    pilot, main, meta = sm.run(root=str(tmp_path), universe=u, out_dir=str(tmp_path))
    assert len(pilot) == 50
    est = main[main["in_estimate"]]
    san = main[~main["in_estimate"]]
    assert len(est) == 400 and len(san) == 20
    assert not set(pilot["qid"]) & set(main["qid"])
    # only eligible questions in the estimate sample; sanity only no-answer
    ue = u.set_index("qid")
    assert (ue.loc[est["qid"], "n_parts"] <= 6).all() and (~ue.loc[est["qid"], "no_answer"]).all()
    assert ue.loc[san["qid"], "no_answer"].all()
    # boilerplate oversampled to ~25% (corpus share is 28% here, so check the exact target and strata)
    assert est["boilerplate_flag"].sum() == 100
    # weights: sum of weights == eligible universe size minus the pilot
    elig = (~u["no_answer"]) & (u["n_parts"] <= 6) & (~u["bad_quality"])
    assert (~ue.loc[est["qid"], "bad_quality"]).all()
    assert est["weight"].sum() == pytest.approx(elig.sum() - len(pilot))
    assert pilot["weight"].sum() == pytest.approx(elig.sum())
    # every stratum present in the sample has weight N/n
    g = est.groupby("stratum").agg(n=("qid", "size"), N=("N_stratum", "first"), w=("weight", "first"))
    assert (g["w"] == g["N"] / g["n"]).all() and (g["n"] > 0).all()
    # all six Lok Sabhas, both qtypes, both ministry groups
    assert set(est["ls"]) == {13, 14, 15, 16, 17, 18} and set(est["qtype"]) == {"starred", "unstarred"}
    assert set(est["ministry_group"]) == {"security", "other"}
    # reproducible with the same seed, different with another
    p2, m2, _ = sm.run(root=str(tmp_path), universe=u, out_dir=str(tmp_path), force=True)
    assert m2["qid"].tolist() == main["qid"].tolist()
    _, m3, _ = sm.run(root=str(tmp_path), universe=u, out_dir=str(tmp_path), force=True, seed=1)
    assert m3["qid"].tolist() != main["qid"].tolist()
    # refuses to overwrite silently
    with pytest.raises(SystemExit):
        sm.run(root=str(tmp_path), universe=u, out_dir=str(tmp_path))
    assert (main["seed"] == sm.SEED).all()
    assert (main.loc[~main["in_estimate"], "coders"] == "coder1;coder2").all()
    assert list(main.columns) == sm.MAIN_COLS and "double_coded" not in pilot.columns


def test_sample_with_m1_quintiles_and_reweight(tmp_path):
    u = synth_universe()
    rng = np.random.default_rng(3)
    u["m1_mean"] = rng.random(len(u))
    u.loc[u.sample(150, random_state=1).index, "m1_mean"] = np.nan  # questions without a valid M1 score
    pilot, main, meta = sm.run(root=str(tmp_path), universe=u, out_dir=str(tmp_path))
    est = main[main["in_estimate"]]
    assert est["m1_quintile"].between(0, 5).all() and (est["m1_quintile"] > 0).any()
    assert meta["m1_available"] and meta["n_m1_missing_nonboiler_universe"] > 0
    # questions without M1 get their own stratum (one per budget group), nothing is dropped from the universe
    miss = est[est["stratum"].str.startswith("m1_missing|")]
    assert 1 <= len(miss) <= 3 and miss["N_stratum"].sum() <= 150
    ue = u.set_index("qid")
    assert ue.loc[miss["qid"], "m1_mean"].isna().all() and not ue.loc[miss["qid"], "boilerplate_flag"].any()
    # every non-missing sampled non-boilerplate question sits in a quintile stratum or a merged quintile bin
    nonb = est[(~est["boilerplate_flag"]) & (~est["qid"].isin(miss["qid"]))]
    assert nonb["stratum"].str.contains(r"\|q\d+$").mean() > 0.5
    assert est["weight"].sum() == pytest.approx(((~u["no_answer"]) & (u["n_parts"] <= 6) & (~u["bad_quality"])).sum() - 50)
    # corpus grows (new questions appended): reweight keeps qids and changes weights
    u2 = pd.concat([u, synth_universe(5000, seed=9).assign(qid=lambda d: "z" + d["qid"])], ignore_index=True)
    u2["m1_mean"] = np.where(u2["m1_mean"].isna(), rng.random(len(u2)), u2["m1_mean"])
    p2, m2, rep = sm.reweight(root=str(tmp_path), out_dir=str(tmp_path), universe=u2)
    assert m2["qid"].tolist() == main["qid"].tolist()
    assert m2.loc[m2["in_estimate"], "weight"].sum() == pytest.approx(
        ((~u2["no_answer"]) & (u2["n_parts"] <= 6) & (~u2["bad_quality"])).sum() - 50, rel=1e-9)
    assert m2["coders"].tolist() == main["coders"].tolist() and m2["double_coded"].tolist() == main["double_coded"].tolist()
    # reweight against the unchanged universe reproduces the drawn weights exactly
    _, m3, _ = sm.reweight(root=str(tmp_path), out_dir=str(tmp_path), universe=u)
    assert m3["weight"].sum() > 0
    mm = sm.run(root=str(tmp_path), universe=u, out_dir=str(tmp_path), force=True)[1]
    _, m4, _ = sm.reweight(root=str(tmp_path), out_dir=str(tmp_path), universe=u)
    assert np.allclose(m4["weight"].values, mm["weight"].values)
    assert (m4["N_stratum"].values == mm["N_stratum"].values).all()


def _ls_counts(est):
    return est["ls"].value_counts().to_dict()


def test_ls_share_oversampling_synthetic(tmp_path):
    u = synth_universe()
    pilot, main, meta = sm.run(root=str(tmp_path), universe=u, out_dir=str(tmp_path))  # default "17,18:0.45"
    est = main[main["in_estimate"]]
    c = _ls_counts(est)
    assert abs(c[17] + c[18] - 180) <= 2 and abs(c[17] - c[18]) <= 2
    assert abs(sum(c[k] for k in (13, 14, 15, 16)) - 220) <= 2
    assert len(est) == 400 and est["boilerplate_flag"].sum() == 100
    assert meta["ls_share"] == "17,18:0.45" and meta["ls_share_parsed"] == {"ls": [17, 18], "share": 0.45}
    assert meta["main_estimate_n_by_ls"][17] == c[17]
    # weights are exact: sum to eligible universe minus pilot, overall and within each LS
    elig = (~u["no_answer"]) & (u["n_parts"] <= 6) & (~u["bad_quality"])
    assert est["weight"].sum() == pytest.approx(elig.sum() - len(pilot))
    pilot_ls = pilot["ls"].value_counts()
    for ls in range(13, 19):
        expect = (elig & (u["ls"] == ls)).sum() - pilot_ls.get(ls, 0)
        assert est.loc[est["ls"] == ls, "weight"].sum() == pytest.approx(expect)
    g = est.groupby("stratum").agg(n=("qid", "size"), N=("N_stratum", "first"), w=("weight", "first"))
    assert (g["w"] == g["N"] / g["n"]).all()
    # disjoint from pilot, reproducible with the seed
    assert not set(pilot["qid"]) & set(main["qid"])
    _, m2, _ = sm.run(root=str(tmp_path), universe=u, out_dir=str(tmp_path), force=True)
    assert m2["qid"].tolist() == main["qid"].tolist()
    # a different share / "none" changes the composition
    _, m3, _ = sm.run(root=str(tmp_path), universe=u, out_dir=str(tmp_path), force=True, ls_share="18:0.5")
    c3 = _ls_counts(m3[m3["in_estimate"]])
    assert abs(c3[18] - 200) <= 2
    _, m4, meta4 = sm.run(root=str(tmp_path), universe=u, out_dir=str(tmp_path), force=True, ls_share="none")
    c4 = _ls_counts(m4[m4["in_estimate"]])
    assert c4[17] + c4[18] < 180 and meta4["ls_share_parsed"]["ls"] == []
    # reweight keeps the qids and the exact weight total
    _, m5, _ = sm.reweight(root=str(tmp_path), out_dir=str(tmp_path), universe=u)
    assert m5.loc[m5["in_estimate"], "weight"].sum() == pytest.approx(
        elig.sum() - len(pd.read_csv(tmp_path / "pilot_sample.csv")))


def test_parse_ls_share():
    assert sm.parse_ls_share("17,18:0.45") == ([17, 18], 0.45)
    assert sm.parse_ls_share("none") == ([], 0.0) and sm.parse_ls_share(None) == ([], 0.0)
    for bad in ("17", "17,18:1.5", "x:0.4", ":0.4"):
        with pytest.raises(ValueError):
            sm.parse_ls_share(bad)


def test_load_universe_collapses_eight_group_column(tmp_path):
    q = pd.DataFrame({"qid": ["a", "b", "c"], "ls": 17, "qtype": "unstarred", "ministry": "X",
                      "ministry_group": ["security", "economy_finance", "agriculture_rural"],
                      "boilerplate": [None, None, None], "answer_missing": False, "question_missing": False,
                      "qa_mismatch": False, "answer_garbled": False})
    parts = pd.DataFrame({"qid": ["a", "b", "c"], "n_parts": 1, "part_id": ["a#_all", "b#_all", "c#_all"]})
    os.makedirs(tmp_path / "data" / "final")
    os.makedirs(tmp_path / "data" / "measures")
    q.to_parquet(tmp_path / "data" / "final" / "questions_analysis.parquet")
    parts.to_parquet(tmp_path / "data" / "measures" / "parts.parquet")
    u = sm.load_universe(str(tmp_path))
    assert dict(zip(u["qid"], u["ministry_group"])) == {"a": "security", "b": "other", "c": "other"}
    assert u.attrs["ministry_group_source"] == "ministry_group column"


REAL_Q = os.path.join(os.path.dirname(os.path.dirname(HERE)), "data", "final", "questions_analysis.parquet")
REAL_P = os.path.join(os.path.dirname(os.path.dirname(HERE)), "data", "measures", "parts.parquet")


@pytest.mark.skipif(not (os.path.exists(REAL_Q) and os.path.exists(REAL_P)), reason="real data not available")
def test_ls_share_real_data(tmp_path):
    u = sm.load_universe(os.path.dirname(os.path.dirname(HERE)))
    est, _, _ = sm.eligibility(u)
    pilot, main, meta = sm.run(root=str(tmp_path), universe=u, out_dir=str(tmp_path))
    e = main[main["in_estimate"]]
    c = _ls_counts(e)
    assert abs(c[17] + c[18] - 180) <= 2 and abs(c[17] - c[18]) <= 2
    assert len(e) == 400 and e["boilerplate_flag"].sum() == 100
    assert e["weight"].sum() == pytest.approx(est.sum() - len(pilot))  # M1-less questions are not dropped
    assert e["double_coded"].sum() == 100 and (e["coders"] == "coder1").sum() == 150
    assert set(e["ministry_group"]) <= {"security", "other"} and not set(pilot["qid"]) & set(main["qid"])


def test_load_universe_fallback_and_ministry_group(tmp_path):
    q = pd.DataFrame({"qid": ["a", "b", "c", "d"], "ls": 13, "qtype": "starred",
                      "ministry": ["HOME AFFAIRS", "FINANCE", "DEFENCE", "EXTERNAL AFFAIRS"],
                      "boilerplate": [np.array(["state_subject"]), np.array([], dtype=object), None, [] ],
                      "answer_missing": [False, False, True, False], "question_missing": False,
                      "qa_mismatch": [False, False, False, True], "answer_garbled": [False, True, False, False]})
    parts = pd.DataFrame({"qid": ["a", "a", "b", "c", "d"], "n_parts": [2, 2, 1, 1, 1],
                          "part_id": ["a#a", "a#b", "b#_all", "c#_all", "d#_all"]})
    os.makedirs(tmp_path / "data" / "final")
    os.makedirs(tmp_path / "data" / "measures")
    q.to_parquet(tmp_path / "data" / "final" / "questions_analysis.parquet")
    parts.to_parquet(tmp_path / "data" / "measures" / "parts.parquet")
    u = sm.load_universe(str(tmp_path))
    assert dict(zip(u["qid"], u["ministry_group"])) == {"a": "security", "b": "other", "c": "security", "d": "security"}
    assert dict(zip(u["qid"], u["boilerplate_flag"])) == {"a": True, "b": False, "c": False, "d": False}
    assert dict(zip(u["qid"], u["bad_quality"])) == {"a": False, "b": True, "c": False, "d": True}
    est, san, rep = sm.eligibility(u)
    assert est.sum() == 1 and u.loc[est, "qid"].tolist() == ["a"] and san.sum() == 1
    assert rep["n_excluded_qa_mismatch_or_garbled"] == 2
    assert dict(zip(u["qid"], u["n_parts"])) == {"a": 2, "b": 1, "c": 1, "d": 1} and u["m1_mean"].isna().all()
    # ministry_group column present -> used
    q["ministry_group"] = ["other", "other", "security", "other"]
    q.to_parquet(tmp_path / "data" / "final" / "questions_analysis.parquet")
    assert sm.load_universe(str(tmp_path)).set_index("qid").loc["a", "ministry_group"] == "other"
    # M1 scores present -> question-level mean
    pd.DataFrame({"part_id": ["a#a", "a#b"], "m1_cos": [0.2, 0.4]}).to_parquet(
        tmp_path / "data" / "measures" / "m1_scores.parquet")
    assert sm.load_universe(str(tmp_path)).set_index("qid").loc["a", "m1_mean"] == pytest.approx(0.3)


# ============================================================================================ coding sheets
def synth_corpus(n_q=6, seed=0):
    rng = np.random.default_rng(seed)
    qrows, prows = [], []
    for i in range(n_q):
        qid = f"ls1{i % 6}-s1-U-{i}-20000101"
        labels = list("abc")[: 2 + i % 2]
        qtext = "\n".join(f"({l}) whether the Government will do thing {l}{i}?" for l in labels)
        qrows.append({"qid": qid, "ls": 13 + i % 6, "ministry": "FINANCE", "question_text": qtext,
                      "answer_text": ("(a) =SUM(1,2) answer text. Yes\x07" if i == 0 else f"Answer {i}.")
                      if i != 3 else "",
                      "annexure_text": ("ANNEX " + "x" * 5000) if i == 1 else (None if i != 2 else ""),
                      "has_annexure": i in (1, 2)})
        for l in labels:
            prows.append({"part_id": f"{qid}#{l}", "qid": qid, "part_label": l, "n_parts": len(labels),
                          "question_part": f"whether the Government will do thing {l}{i}?",
                          "answer_part": f"Part answer {l}{i}" if i != 3 else "", "answer_labels": np.array([l]),
                          "shared_answer": False, "answer_missing_for_part": i == 3})
    sample = pd.DataFrame({"qid": [r["qid"] for r in qrows]})
    return pd.DataFrame(prows), pd.DataFrame(qrows), sample


def fill_sheet(path, rule, out=None):
    """Simulate a coder: rule(part_id, row_idx) -> (label, evasion, request, note)."""
    wb = load_workbook(path)
    ws = wb["Coding"]
    for r in range(2, ws.max_row + 1):
        lab, ev, rq, note = rule(ws.cell(row=r, column=1).value, r)
        for col, v in zip((cs.COL["label"], cs.COL["evasion_type"], cs.COL["request_type"], cs.COL["note"]),
                          (lab, ev, rq, note)):
            ws.cell(row=r, column=col).value = v
    wb.save(out or path)


def std_rule(pid, r):
    lab = ["FULL", "PARTIAL", "NONE"][r % 3]
    return lab, ("NONE_APPLICABLE" if lab == "FULL" else "GENERIC"), "INFORMATION", f"n{r}"


def test_sheet_build_blind_randomised_protected(tmp_path):
    parts, qtext, sample = synth_corpus()
    p1, v1, n1 = cs.build_sheet(parts, qtext, sample, "coder1", "pilot", str(tmp_path / "s1.xlsx"))
    p2, v2, n2 = cs.build_sheet(parts, qtext, sample, "coder2", "pilot", str(tmp_path / "s2.xlsx"))
    assert v1 == v2 and n1 == n2 == len(parts)
    wb1, wb2 = load_workbook(p1), load_workbook(p2)
    assert wb1.sheetnames == ["Instructions", "Coding", "Meta"] and wb1["Meta"].sheet_state == "hidden"
    ws1, ws2 = wb1["Coding"], wb2["Coding"]
    hdr = [c.value for c in ws1[1]]
    assert hdr == cs.HEADERS
    # blindness: no asker / party / bloc / date / qtype / boilerplate / model columns anywhere
    flat = " ".join(str(c.value).lower() for row in ws1.iter_rows(max_row=1) for c in row)
    for bad in ("asker", "party", "bloc", "date", "govt", "opposition", "qtype", "boiler", "m1", "m2", "weight"):
        assert bad not in flat
    instr = " ".join(str(r[0].value) for r in wb1["Instructions"].iter_rows())
    assert cb.instructions_text().split("\n")[0] in instr and "WORKED EXAMPLES" in instr
    # coder-specific order, parts of a question stay contiguous and in order
    ids1 = [ws1.cell(row=r, column=1).value for r in range(2, ws1.max_row + 1)]
    ids2 = [ws2.cell(row=r, column=1).value for r in range(2, ws2.max_row + 1)]
    assert sorted(ids1) == sorted(ids2) == sorted(parts["part_id"]) and ids1 != ids2
    qs = [i.rsplit("#", 1)[0] for i in ids1]
    assert [q for i, q in enumerate(qs) if i == 0 or q != qs[i - 1]] == list(dict.fromkeys(qs))
    # deterministic
    assert cs.coder_order(["a", "b", "c", "d"], "coder1", "pilot") == cs.coder_order(["a", "b", "c", "d"], "coder1", "pilot")
    # data validation + lock + illegal char / '=' text handled
    dvs = {d.formula1 for d in ws1.data_validations.dataValidation}
    assert '"FULL,PARTIAL,NONE"' in dvs and any("NONE_APPLICABLE" in d for d in dvs)
    assert ws1.protection.sheet and ws1["A2"].protection.locked and not ws1.cell(row=2, column=cs.COL["label"]).protection.locked
    full_answers = [str(ws1.cell(row=r, column=cs.COL["FULL ANSWER (all parts)"]).value) for r in range(2, ws1.max_row + 1)]
    assert any(a.startswith("(a) =SUM") for a in full_answers) and not any("\x07" in a for a in full_answers)
    assert any("[NO ANSWER TEXT" in a for a in full_answers)
    # annexure column: truncated to codebook.ANNEXURE_CHARS, placeholder when referred to but empty, blank otherwise
    anns = [str(ws1.cell(row=r, column=cs.COL["ANNEXURE (truncated)"]).value or "") for r in range(2, ws1.max_row + 1)]
    assert cs.COL["ANNEXURE (truncated)"] == cs.COL["ANSWER TEXT FOR THIS PART"] + 1
    long_ = [a for a in anns if a.startswith("ANNEX")]
    assert long_ and all(a == "ANNEX " + "x" * (cb.ANNEXURE_CHARS - 6) + "[... truncated ...]" for a in long_)
    assert any(a.startswith("[annexure referred to") for a in anns) and any(a == "" for a in anns)
    assert cs.EVASION_CHOICES is cb.EVASION_CHOICES and va.EVASION_CHOICES is cb.EVASION_CHOICES


def test_import_roundtrip_and_validation(tmp_path):
    parts, qtext, sample = synth_corpus()
    paths = []
    for c in ("coder1", "coder2"):
        p, v, n = cs.build_sheet(parts, qtext, sample, c, "pilot", str(tmp_path / f"sheet_{c}_pilot.xlsx"))
        fill_sheet(p, std_rule if c == "coder1" else (lambda pid, r: ("FULL", "", "reason", "x")))
        paths.append(p)
    out = str(tmp_path / "human_labels.csv")
    df, probs = cs.import_sheets(paths, out_path=out)
    assert list(df.columns) == cs.LABEL_COLS and len(df) == 2 * len(parts) and not probs
    c2 = df[df["coder"] == "coder2"]
    assert (c2["evasion_type"] == "NONE_APPLICABLE").all() and (c2["request_type"] == "REASON").all()
    assert df["sheet_version"].str.startswith(f"{cb.CODEBOOK_VERSION}-pilot-").all() and (df["stage"] == "pilot").all()
    assert (df["qid"] == df["part_id"].str.rsplit("#", n=1).str[0]).all()

    # --- invalid enum / incomplete / FULL with evasion / non-FULL without evasion
    bad = str(tmp_path / "sheet_coder1_pilot.xlsx")

    def broken(pid, r):
        return {2: ("FUL", "GENERIC", "INFORMATION", ""), 3: (None, None, None, None),
                4: ("FULL", "GENERIC", "INFORMATION", ""), 5: ("NONE", "", "INFORMATION", ""),
                6: ("NONE", "GENERIC", "WHY", "")}.get(r, ("PARTIAL", "OTHER", "ACTION", ""))
    fill_sheet(bad, broken)
    with pytest.raises(cs.ImportProblems) as ei:
        cs.import_sheets([bad], out_path=out)
    msgs = " | ".join(p[2] for p in ei.value.problems)
    for frag in ("invalid label", "label missing", "FULL requires", "NONE requires an evasion_type",
                 "invalid request_type"):
        assert frag in msgs
    # nothing written on failure; allow_incomplete keeps only valid rows and keeps the other coder's rows
    before = pd.read_csv(out)
    df2, probs2 = cs.import_sheets([bad], out_path=out, allow_incomplete=True)
    assert len(probs2) >= 5
    assert (df2["coder"] == "coder2").sum() == len(parts)
    assert (df2["coder"] == "coder1").sum() == len(parts) - 5
    assert len(before) == 2 * len(parts)

    # --- tampering: deleted row -> part_id set differs
    wb = load_workbook(str(tmp_path / "sheet_coder2_pilot.xlsx"))
    wb["Coding"].delete_rows(5)
    t = str(tmp_path / "sheet_coder2_tampered.xlsx")
    wb.save(t)
    with pytest.raises(cs.ImportProblems) as ei:
        cs.import_sheets([t], out_path=out)
    assert any("part_id set differs" in p[2] for p in ei.value.problems)


def _assignment_sample():
    """6 questions of synth_corpus: q0,q1 double; q2,q3 coder1 only; q4,q5 coder2 only (+ in_estimate)."""
    parts, qtext, sample = synth_corpus()
    q = sample["qid"].tolist()
    sample["coders"] = ["coder1;coder2", "coder1;coder2", "coder1", "coder1", "coder2", "coder2"]
    sample["in_estimate"] = True
    return parts, qtext, sample, q


def test_per_coder_sheets_and_importer_validation(tmp_path):
    parts, qtext, sample, q = _assignment_sample()
    d = tmp_path / "data" / "coding"
    os.makedirs(d)
    sample.to_csv(d / "main_sample.csv", index=False)
    sets, paths = {}, {}
    for c in ("coder1", "coder2"):
        mine = cs.coder_sample(sample, c)
        p, v, n = cs.build_sheet(parts, qtext, mine, c, "main", str(d / f"sheet_{c}_main.xlsx"))
        ws = load_workbook(p)["Coding"]
        ids = [ws.cell(row=r, column=1).value for r in range(2, ws.max_row + 1)]
        sets[c] = {i.rsplit("#", 1)[0] for i in ids}
        paths[c] = p
        assert n == len(ids) == int(parts["qid"].isin(set(mine["qid"])).sum())
    assert sets["coder1"] == {q[0], q[1], q[2], q[3]} and sets["coder2"] == {q[0], q[1], q[4], q[5]}
    v1 = load_workbook(paths["coder1"])["Meta"]["B1"].value
    v2 = load_workbook(paths["coder2"])["Meta"]["B1"].value
    assert v1 != v2  # the version hash covers the coder's own part set
    for c in paths:
        fill_sheet(paths[c], std_rule)
    out = str(d / "human_labels.csv")
    df, probs = cs.import_sheets(list(paths.values()), out_path=out)
    assert not probs and set(df["coder"]) == {"coder1", "coder2"}
    n_both = df.groupby("part_id")["coder"].nunique()
    assert (n_both == 2).sum() == int(parts["qid"].isin({q[0], q[1]}).sum())
    assert (n_both == 1).sum() == int(parts["qid"].isin({q[2], q[3], q[4], q[5]}).sum())
    # a sheet with a coder-2 question in coder 1's workbook is refused (set differs from the issued/assigned set)
    wrong = cs.coder_sample(sample, "coder2")
    pw, _, _ = cs.build_sheet(parts, qtext, wrong, "coder1", "main", str(d / "sheet_coder1_wrong.xlsx"))
    fill_sheet(pw, std_rule)
    with pytest.raises(cs.ImportProblems) as ei:
        cs.import_sheets([pw], out_path=out)
    assert any("assigned to coder1" in p_[2] for p_ in ei.value.problems)
    # tampering (deleted row) still caught per coder
    wb = load_workbook(paths["coder1"])
    wb["Coding"].delete_rows(3)
    t = str(d / "sheet_coder1_tamper.xlsx")
    wb.save(t)
    with pytest.raises(cs.ImportProblems) as ei:
        cs.import_sheets([t], out_path=out)
    assert any("part_id set differs" in p_[2] for p_ in ei.value.problems)


def test_make_main_per_coder_and_refresh_instructions(tmp_path, monkeypatch):
    parts, qtext, sample, q = _assignment_sample()
    os.makedirs(tmp_path / "data" / "measures")
    os.makedirs(tmp_path / "data" / "final")
    os.makedirs(tmp_path / "data" / "coding")
    parts.to_parquet(tmp_path / "data" / "measures" / "parts.parquet")
    qtext.to_parquet(tmp_path / "data" / "final" / "questions_analysis.parquet")
    sp = tmp_path / "data" / "coding" / "main_sample.csv"
    sample.to_csv(sp, index=False)
    before = sp.read_bytes()
    monkeypatch.setattr(cb, "CODEBOOK_VERSION", "v0.9")
    res1 = cs.make(str(tmp_path), "main")
    assert [r[2] for r in res1] == [int(parts["qid"].isin(set(cs.coder_sample(sample, c)["qid"])).sum())
                                    for c in ("coder1", "coder2")]
    old_sheet = str(tmp_path / "old_coder2.xlsx")
    shutil.copy(res1[1][0], old_sheet)
    monkeypatch.setattr(cb, "CODEBOOK_VERSION", "v1.9")
    monkeypatch.setattr(cb, "instructions_text", lambda: "NEW INSTRUCTIONS TEXT")
    res2 = cs.make(str(tmp_path), "main")  # codebook freeze: re-run make, same sample
    assert sp.read_bytes() == before
    assert [r[2] for r in res2] == [r[2] for r in res1]
    assert res2[0][1].startswith("v1.9-main-") and res2[0][1].split("-")[-1] == res1[0][1].split("-")[-1]
    wb = load_workbook(res2[0][0])
    assert any("NEW INSTRUCTIONS TEXT" in str(r[0].value) for r in wb["Instructions"].iter_rows())
    # a sheet issued under an older codebook version still imports (hash + stage are what is checked)
    fill_sheet(old_sheet, std_rule)
    df, probs = cs.import_sheets([old_sheet], out_path=str(tmp_path / "data" / "coding" / "human_labels.csv"))
    assert not probs and df["sheet_version"].str.startswith("v0.9-main-").all()


def test_assign_coders_counts_and_balance(tmp_path):
    u = synth_universe()
    pilot, main, meta = sm.run(root=str(tmp_path), universe=u, out_dir=str(tmp_path))
    est = main[main["in_estimate"]]
    assert est["double_coded"].sum() == 100
    assert (est["coders"] == "coder1").sum() == 150 and (est["coders"] == "coder2").sum() == 150
    assert (est.loc[est["double_coded"], "coders"] == "coder1;coder2").all()
    assert (est.loc[~est["double_coded"], "coders"].isin(["coder1", "coder2"])).all()
    san = main[~main["in_estimate"]]
    assert (san["coders"] == "coder1;coder2").all() and san["double_coded"].all()
    assert meta["coder_assignment_estimate"] == {"double_coded": 100, "single_coder1": 150, "single_coder2": 150,
                                                 "sanity_coded_by_both": 20}
    # each coder's workload = 100 double + 150 single + 20 sanity questions
    for c in ("coder1", "coder2"):
        assert main["coders"].str.contains(c).sum() == 270
    # stratified double subsample: LS group x boilerplate, proportional (+-1)
    grp = np.where(est["ls"].isin([17, 18]), "g17", "g13") + "|" + est["boilerplate_flag"].astype(str)
    tab = pd.crosstab(grp, est["double_coded"])
    expect = tab.sum(axis=1) * 100 / len(est)
    assert ((tab[True] - expect).abs() <= 1.0).all()
    # single-coded split balanced inside every stratum (+-1)
    sg = est[~est["double_coded"]]
    sgrp = np.where(sg["ls"].isin([17, 18]), "g17", "g13") + "|" + sg["boilerplate_flag"].astype(str)
    t2 = pd.crosstab(sgrp, sg["coders"])
    assert ((t2["coder1"] - t2["coder2"]).abs() <= 1).all()
    # reproducible with the recorded seed; another seed changes the assignment
    _, m2, _ = sm.run(root=str(tmp_path), universe=u, out_dir=str(tmp_path), force=True)
    assert m2["coders"].tolist() == main["coders"].tolist()
    _, m3, _ = sm.run(root=str(tmp_path), universe=u, out_dir=str(tmp_path), force=True, seed=2)
    assert m3["coders"].tolist() != main["coders"].tolist()
    # other sizes via the parameters
    _, m4, _ = sm.run(root=str(tmp_path), universe=u, out_dir=str(tmp_path), force=True, main_n=200, double_n=40)
    e4 = m4[m4["in_estimate"]]
    assert len(e4) == 200 and e4["double_coded"].sum() == 40 and (e4["coders"] == "coder1").sum() == 80


def test_main_only_keeps_pilot_file(tmp_path):
    u = synth_universe()
    sm.run(root=str(tmp_path), universe=u, out_dir=str(tmp_path))
    pb = (tmp_path / "pilot_sample.csv").read_bytes()
    pilot0 = pd.read_csv(tmp_path / "pilot_sample.csv")
    with pytest.raises(SystemExit):
        sm.run(root=str(tmp_path), universe=u, out_dir=str(tmp_path), keep_pilot=True)  # main exists, no --force
    pilot, main, meta = sm.run(root=str(tmp_path), universe=u, out_dir=str(tmp_path), keep_pilot=True, force=True,
                               main_n=300, double_n=60, seed=5)
    assert (tmp_path / "pilot_sample.csv").read_bytes() == pb and meta["pilot_kept_from_existing_file"]
    assert not set(pilot0["qid"]) & set(main["qid"]) and len(main[main["in_estimate"]]) == 300


# ============================================================================================ validation
def synth_human(n_q=60, agree=0.8, seed=0, stage="main"):
    rng = np.random.default_rng(seed)
    rows, sample = [], []
    for i in range(n_q):
        qid = f"ls{13 + i % 6}-q{i}"
        sample.append({"qid": qid, "ls": 13 + i % 6, "ministry_group": "security" if i % 4 == 0 else "other",
                       "weight": 10.0 + i % 3, "in_estimate": i != n_q - 1})
        for l in "abcd"[: 2 + i % 3]:
            truth = rng.choice(va.ORD_LABELS, p=[.35, .3, .35])
            ev = "NONE_APPLICABLE" if truth == "FULL" else rng.choice(cb.EVASION_TYPES[:4])
            rq = rng.choice(cb.REQUEST_TYPES[:2])
            for c in ("coder1", "coder2"):
                lab = truth if rng.random() < agree else rng.choice(va.ORD_LABELS)
                e = "NONE_APPLICABLE" if lab == "FULL" else (ev if ev != "NONE_APPLICABLE" else "GENERIC")
                rows.append({"part_id": f"{qid}#{l}", "qid": qid, "coder": c, "label": lab, "evasion_type": e,
                             "request_type": rq, "note": "", "sheet_version": "v1", "stage": stage})
    return pd.DataFrame(rows), pd.DataFrame(sample)


def test_reliability_values():
    human, sample = synth_human(agree=1.0)
    r = va.reliability(human, B=50)
    f = r["fields"]["label"]
    assert f["pct_agree"] == 1.0 and f["kappa"] == pytest.approx(1.0) and f["alpha_nominal"] == pytest.approx(1.0)
    human, sample = synth_human(agree=0.7)
    r = va.reliability(human, B=100)
    f = r["fields"]["label"]
    assert 0.3 < f["alpha_nominal"] < 0.9 and f["alpha_ordinal"] >= f["alpha_nominal"] - 0.2
    assert f["alpha_nominal_ci"][0] <= f["alpha_nominal"] <= f["alpha_nominal_ci"][1]
    assert f["confusion"][2].values.sum() == f["n_parts"]
    # alpha for two coders and no missing data agrees with the manual formula via sklearn-kappa ballpark
    assert abs(f["alpha_nominal"] - f["kappa"]) < 0.05
    by = va.reliability_by(human, sample, "ls")
    assert set(by["ls"]) == set(range(13, 19)) and by["n_parts"].sum() == f["n_parts"]
    bg = va.reliability_by(human, sample, "ministry_group")
    assert set(bg["ministry_group"]) == {"security", "other"}
    assert "evasion_both_nonfull" in r and "request_type" in r["fields"]


def test_adjudication_and_gold(tmp_path):
    human, sample = synth_human(agree=0.7)
    d, _ = va.disagreements(human)
    assert len(d) > 0 and d["disagree_on"].str.len().gt(0).all()
    wide = human.pivot(index="part_id", columns="coder", values="label")
    n_label_dis = int((wide["coder1"] != wide["coder2"]).sum())
    assert (d["disagree_on"].str.contains("label")).sum() == n_label_dis
    parts, _, _ = synth_corpus()
    adj = va.build_adjudication(human, None, None, None)
    assert {"adjudicated_label", "coder1_label", "coder2_label", "disagree_on"} <= set(adj.columns)
    g0, st0 = va.build_gold(human, adj)
    assert st0["n_open"] > 0 and st0["n_gold"] + st0["n_open"] == st0["n_coded_by_both"]
    # adjudicator resolves every label disagreement as the coder1 label
    adj["adjudicated_label"] = np.where(adj["coder1_label"] != adj["coder2_label"], adj["coder1_label"], "")
    adj["adjudicated_evasion_type"] = "OTHER"
    adj["adjudicated_request_type"] = "ACTION"
    # an existing adjudication survives re-generation
    again = va.build_adjudication(human, None, None, adj)
    assert (again["adjudicated_label"] == adj["adjudicated_label"]).all()
    g, st = va.build_gold(human, adj)
    assert st["n_open"] == 0 and st["n_adjudicated"] == n_label_dis
    wide_full = human.pivot(index="part_id", columns="coder", values="label")
    gi = g.set_index("part_id")
    agree = wide_full[wide_full["coder1"] == wide_full["coder2"]]
    assert (gi.loc[agree.index, "label"] == agree["coder1"]).all()
    dis = wide_full[wide_full["coder1"] != wide_full["coder2"]]
    assert (gi.loc[dis.index, "label"] == dis["coder1"]).all()
    assert (gi.loc[gi["label"] == "FULL", "evasion_type"] == "NONE_APPLICABLE").all()
    assert gi.loc[gi["label"] != "FULL", "evasion_type"].notna().all()
    # invalid adjudicated values are ignored (stay open)
    bad = adj.copy()
    bad["adjudicated_label"] = "MAYBE"
    assert va.build_gold(human, bad)[1]["n_open"] == n_label_dis
    # parts text is attached when supplied
    pt = pd.DataFrame({"part_id": adj["part_id"], "question_part": "Q?", "answer_part": "A" * 3000})
    ad2 = va.build_adjudication(human, None, pt, None)
    assert (ad2["answer_part"].str.len() == 1500).all()


def synth_m2(gold, models):
    rng = np.random.default_rng(5)
    rows = []
    for model, acc in models.items():
        for pid, lab in zip(gold["part_id"], gold["label"]):
            m = lab if rng.random() < acc else rng.choice(va.ORD_LABELS)
            rows.append({"part_id": pid, "label": m, "evasion_type": "NONE_APPLICABLE" if m == "FULL" else "GENERIC",
                         "model_id": model})
    return pd.DataFrame(rows)


def test_m2_evaluation_and_model_selection():
    human, sample = synth_human(n_q=120, agree=0.9, seed=2)
    adj = va.build_adjudication(human)
    adj["adjudicated_label"] = np.where(adj["coder1_label"] != adj["coder2_label"], adj["coder1_label"], "")
    adj["adjudicated_evasion_type"] = "OTHER"
    adj["adjudicated_request_type"] = "ACTION"
    gold, _ = va.build_gold(human, adj)
    m2 = synth_m2(gold, {"cheap": 0.55, "mid": 0.85, "pricey": 0.97})
    tab, confs = va.evaluate_m2(gold, m2, sample, B=100)
    t = tab.set_index("model_id")
    assert t.loc["pricey", "kappa"] > t.loc["mid", "kappa"] > t.loc["cheap", "kappa"]
    assert (t["coverage"] == 1.0).all() and set(confs) == {"cheap", "mid", "pricey"}
    assert confs["mid"].values.sum() == len(gold)
    assert t.loc["mid", "kappa_ci_lo"] <= t.loc["mid", "kappa"] <= t.loc["mid", "kappa_ci_hi"]
    assert 0 <= t.loc["mid", "accuracy_weighted"] <= 1
    # selection: highest kappa per cost among those >= threshold
    costs = {"cheap": 1.0, "mid": 5.0, "pricey": 40.0}
    thr = (t.loc["mid", "kappa"] + t.loc["cheap", "kappa"]) / 2
    tsel, pick, msg = va.select_model(tab, costs, threshold=thr)
    assert pick == "mid" and "mid" in msg and "lower 95% CI bound" in msg
    assert "ci_lo_clears" in tsel and tsel.set_index("model_id").loc["pricey", "ci_lo_clears"] in (True, False)
    _, pick, _ = va.select_model(tab, costs, threshold=0.0)
    assert pick == "cheap"  # every model passes -> best kappa per dollar
    _, pick, msg = va.select_model(tab, costs, threshold=0.999)
    assert pick is None and "no model reaches" in msg
    _, pick, msg = va.select_model(tab, {}, threshold=thr)
    assert pick is None and "no has a cost" in msg.replace("none has", "no has")
    # partial coverage is reported
    tab2, _ = va.evaluate_m2(gold, m2[m2["part_id"].isin(gold["part_id"].iloc[:50])], B=20)
    assert (tab2["coverage"] < 1).all()


def test_m1_evaluation():
    human, sample = synth_human(n_q=100, agree=1.0, seed=4)
    gold, _ = va.build_gold(human)
    rng = np.random.default_rng(1)
    m1 = pd.DataFrame({"part_id": gold["part_id"],
                       "m1_cos": gold["label"].map(cb.LABEL_SCORE) * 0.4 + rng.normal(0, 0.15, len(gold))})
    e = va.evaluate_m1(gold, m1, B=100)
    assert e["auc_full_vs_none"] > 0.9 and e["spearman_label_score"] > 0.6
    assert 0.1 < e["youden_threshold"] < 0.3 and e["youden_j"] > 0.6
    lo, hi = e["auc_full_vs_none_ci"]
    assert lo <= e["auc_full_vs_none"] <= hi
    assert va.evaluate_m1(gold, m1.iloc[0:0], B=10) is None


def _write_inputs(root, human=None, sample=None, adj=None, m2=None, m1=None):
    d = root / "data" / "coding"
    os.makedirs(d, exist_ok=True)
    os.makedirs(root / "data" / "measures", exist_ok=True)
    if human is not None:
        human.to_csv(d / "human_labels.csv", index=False)
    if sample is not None:
        sample.to_csv(d / "main_sample.csv", index=False)
    if adj is not None:
        adj.to_csv(d / "adjudication.csv", index=False)
    if m2 is not None:
        m2.to_parquet(root / "data" / "measures" / "m2_labels.parquet")
    if m1 is not None:
        m1.to_parquet(root / "data" / "measures" / "m1_scores.parquet")


def test_report_with_no_inputs(tmp_path):
    path, text = va.write_report(str(tmp_path), "main", B=10)
    assert os.path.exists(path)
    assert "not found" in text and "No gold labels yet" in text
    assert va.main(["report", "--root", str(tmp_path), "--boot", "10"]) == 0
    assert va.main(["adjudication", "--root", str(tmp_path)]) == 1


def test_report_end_to_end(tmp_path):
    human, sample = synth_human(n_q=100, agree=0.85, seed=7)
    _write_inputs(tmp_path, human, sample)
    # human only: reliability, no gold yet, M2/M1 sections say so
    _, text = va.write_report(str(tmp_path), "main", B=30)
    assert "Krippendorff" not in text and "alpha_nominal" in text and "by ls" in text and "by ministry_group" in text
    assert "open:" in text and "m2_labels.parquet not found" in text
    # adjudicate through the CLI, resolve, then add M2 + M1
    assert va.main(["adjudication", "--root", str(tmp_path)]) == 0
    adj = pd.read_csv(tmp_path / "data" / "coding" / "adjudication.csv", keep_default_na=False)
    adj["adjudicated_label"] = np.where(adj["coder1_label"] != adj["coder2_label"], adj["coder1_label"], "")
    adj["adjudicated_evasion_type"] = "OTHER"
    adj["adjudicated_request_type"] = "ACTION"
    adj.to_csv(tmp_path / "data" / "coding" / "adjudication.csv", index=False)
    assert va.main(["gold", "--root", str(tmp_path)]) == 0
    gold = pd.read_csv(tmp_path / "data" / "coding" / "gold_labels.csv")
    assert gold["label"].notna().all()
    m2 = synth_m2(gold, {"flash-lite": 0.6, "flash": 0.9})
    rng = np.random.default_rng(0)
    m1 = pd.DataFrame({"part_id": gold["part_id"],
                       "m1_cos": gold["label"].map(cb.LABEL_SCORE) * 0.3 + rng.normal(0, .2, len(gold))})
    _write_inputs(tmp_path, m2=m2, m1=m1)
    rc = va.main(["report", "--root", str(tmp_path), "--boot", "30", "--threshold", "0.6",
                  "--cost", "flash-lite=1", "--cost", "flash=5"])
    assert rc == 0
    text = open(tmp_path / "results" / "validation_report.md", encoding="utf-8").read()
    assert "Model rule" in text and "flash" in text and "AUC FULL vs NONE" in text and "Youden" in text
    assert "Sanity stratum" in text  # the last synthetic question is flagged in_estimate=False
    assert "Confusion matrix for `flash`" in text


def test_gold_source_logic():
    def row(pid, coder, lab, ev, rq="INFORMATION"):
        return {"part_id": pid, "qid": pid.rsplit("#", 1)[0], "coder": coder, "label": lab, "evasion_type": ev,
                "request_type": rq, "note": "", "sheet_version": "v", "stage": "main"}
    NA = "NONE_APPLICABLE"
    human = pd.DataFrame([
        row("q1#a", "coder1", "FULL", NA), row("q1#a", "coder2", "FULL", NA),                    # agreed
        row("q2#a", "coder1", "FULL", NA), row("q2#a", "coder2", "NONE", "GENERIC"),             # adjudicated
        row("q3#a", "coder1", "NONE", "GENERIC"), row("q3#a", "coder2", "PARTIAL", "OTHER"),     # open
        row("q4#a", "coder1", "PARTIAL", "OTHER"),                                                # single (coder1)
        row("q5#a", "coder2", "FULL", NA, "REASON"),                                              # single (coder2)
        row("q6#a", "coder1", "NONE", "GENERIC"),                                                 # pending (assigned to both)
        row("q7#a", "coder1", "NONE", "GENERIC"), row("q7#a", "coder2", "NONE", "OTHER"),        # evasion differs only
    ])
    sample = pd.DataFrame({"qid": [f"q{i}" for i in range(1, 8)],
                           "coders": ["coder1;coder2", "coder1;coder2", "coder1;coder2", "coder1", "coder2",
                                      "coder1;coder2", "coder1;coder2"]})
    adj = pd.DataFrame({"part_id": ["q2#a"], "adjudicated_label": ["PARTIAL"], "adjudicated_evasion_type": ["OTHER"],
                        "adjudicated_request_type": ["INFORMATION"]})
    g, st = va.build_gold(human, adj, sample=sample)
    gi = g.set_index("part_id")
    assert set(g.columns) == set(va.GOLD_COLS) and "label_source" not in g.columns
    assert gi.loc["q1#a", "gold_source"] == "agreed" and gi.loc["q1#a", "label"] == "FULL"
    assert gi.loc["q2#a", "gold_source"] == "adjudicated" and gi.loc["q2#a", "label"] == "PARTIAL"
    assert gi.loc["q2#a", "evasion_type"] == "OTHER"
    assert gi.loc["q3#a", "gold_source"] == "adjudicated" and pd.isna(gi.loc["q3#a", "label"])  # still open
    assert gi.loc["q4#a", "gold_source"] == "single" and gi.loc["q4#a", "label"] == "PARTIAL"
    assert gi.loc["q5#a", "gold_source"] == "single" and gi.loc["q5#a", "evasion_type"] == NA
    assert gi.loc["q5#a", "request_type"] == "REASON"
    assert "q6#a" not in gi.index and st["n_pending"] == 1
    assert pd.isna(gi.loc["q7#a", "evasion_type"]) and gi.loc["q7#a", "gold_source"] == "adjudicated"
    assert st["n_single"] == 2 and st["n_coded_by_both"] == 4 and st["n_open"] == 1 and st["n_adjudicated"] == 3
    assert set(g["gold_source"]) <= {"agreed", "adjudicated", "single"}
    # without a sample file the half-coded part counts as single-coded
    g2, st2 = va.build_gold(human, adj)
    assert g2.set_index("part_id").loc["q6#a", "gold_source"] == "single" and st2["n_pending"] == 0
    # disagreement list contains only double-coded parts
    d, _ = va.disagreements(human)
    assert set(d["part_id"]) == {"q2#a", "q3#a", "q7#a"}


def _half_single(human, sample, frac=0.5, seed=0):
    """Make a share of the questions single-coded (drop coder2 or coder1 rows), update the sample's coders column."""
    rng = np.random.default_rng(seed)
    qs = sample["qid"].tolist()
    single = set(rng.choice(qs, int(len(qs) * frac), replace=False))
    which = {q: rng.choice(["coder1", "coder2"]) for q in single}
    keep = [not (q in single and c != which[q]) for q, c in zip(human["qid"], human["coder"])]
    out = human[keep].reset_index(drop=True)
    sample = sample.copy()
    sample["coders"] = [which[q] if q in single else "coder1;coder2" for q in sample["qid"]]
    sample["double_coded"] = ~sample["qid"].isin(single)
    return out, sample, single


def test_reliability_uses_only_double_coded_parts():
    human, sample = synth_human(n_q=100, agree=0.7, seed=11)
    mixed, sample2, single = _half_single(human, sample)
    dbl_parts = mixed.groupby("part_id")["coder"].nunique()
    n_dbl = int((dbl_parts == 2).sum())
    assert 0 < n_dbl < len(dbl_parts)
    r = va.reliability(mixed, B=20)
    assert r["fields"]["label"]["n_parts"] == n_dbl and r["fields"]["label"]["confusion"][2].values.sum() == n_dbl
    # identical to reliability on the double-coded subset alone, and the single-coded labels do not enter
    only = human[~human["qid"].isin(single)]
    r2 = va.reliability(only, B=20)
    for k in ("kappa", "pct_agree", "alpha_nominal", "alpha_ordinal"):
        assert r["fields"]["label"][k] == pytest.approx(r2["fields"]["label"][k])
    by = va.reliability_by(mixed, sample2, "ls")
    assert by["n_parts"].sum() == n_dbl


def test_report_with_single_and_double_coded(tmp_path):
    human, sample = synth_human(n_q=100, agree=0.8, seed=21)
    mixed, sample2, single = _half_single(human, sample, frac=0.6, seed=3)
    _write_inputs(tmp_path, mixed, sample2)
    _, text = va.write_report(str(tmp_path), "main", B=20)
    assert "single-coded parts are excluded" in text and "single-coded parts:" in text
    assert "primary set = double-coded estimate questions" in text
    assert va.main(["adjudication", "--root", str(tmp_path)]) == 0
    adj = pd.read_csv(tmp_path / "data" / "coding" / "adjudication.csv", keep_default_na=False)
    assert not adj["qid"].isin(single).any()  # only double-coded parts go to adjudication
    adj["adjudicated_label"] = np.where(adj["coder1_label"] != adj["coder2_label"], adj["coder1_label"], "")
    adj["adjudicated_evasion_type"] = "OTHER"
    adj["adjudicated_request_type"] = "ACTION"
    adj.to_csv(tmp_path / "data" / "coding" / "adjudication.csv", index=False)
    assert va.main(["gold", "--root", str(tmp_path)]) == 0
    gold = pd.read_csv(tmp_path / "data" / "coding" / "gold_labels.csv")
    assert set(gold["gold_source"]) == {"agreed", "adjudicated", "single"} and gold["label"].notna().all()
    assert set(gold.loc[gold["gold_source"] == "single", "qid"]) == set(single)
    assert gold["part_id"].nunique() == mixed["part_id"].nunique()


def test_gold_adjudicated_evasion_overrides_agreed_value():
    """Regression: both coders chose OTHER (but disagreed on label); the adjudicator re-classified the evasion
    type as PROCEDURAL (renamed WITHHELD in v1.2). The adjudicated value must win, mapped to the current name."""
    import pandas as pd
    import validate as V
    human = pd.DataFrame([
        {"part_id": "q1#a", "qid": "q1", "coder": "coder1", "label": "NONE", "evasion_type": "OTHER",
         "request_type": "INFORMATION", "note": "", "sheet_version": "x", "stage": "pilot"},
        {"part_id": "q1#a", "qid": "q1", "coder": "coder2", "label": "PARTIAL", "evasion_type": "OTHER",
         "request_type": "INFORMATION", "note": "", "sheet_version": "x", "stage": "pilot"},
    ])
    adj = pd.DataFrame([{"part_id": "q1#a", "adjudicated_label": "PARTIAL", "adjudicated_evasion_type": "PROCEDURAL",
                         "adjudicated_request_type": "INFORMATION"}])
    gold, status = V.build_gold(human, adj, stage="pilot")
    row = gold.set_index("part_id").loc["q1#a"]
    assert row["label"] == "PARTIAL" and row["evasion_type"] == "WITHHELD" and row["gold_source"] == "adjudicated"


def test_stage_files_never_collide(tmp_path):
    """pilot and main adjudication/gold live in different files (a main `adjudication` run once overwrote the pilot's)"""
    root = str(tmp_path)
    for name in ("adjudication", "gold_labels"):
        main, pilot = va.coding_file(root, name, "main"), va.coding_file(root, name, "pilot")
        assert main != pilot and main.endswith(f"{name}.csv") and pilot.endswith(f"{name}_pilot.csv")
