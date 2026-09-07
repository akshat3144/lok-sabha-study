"""Unit tests for the part splitter, minister-header parser and friends.

Snippets are real (LS17 API text and LS18 docx text); run with
    python -I -X utf8 -m pytest pipeline/tests -q      (or: python -I -X utf8 pipeline/tests/test_clean.py)
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import clean  # noqa: E402
import textparse as tp  # noqa: E402


# ------------------------------------------------------------------ html / hindi cleaning
def test_html_escaped_and_tabs():
    raw = ("(a)\t whether the Government is aware; &lt;br/&gt;(b) \tif so, the details thereof; &lt;br/&gt;"
           "(c) \tif not, the reasons therefor?&lt;br/&gt;")
    txt, hindi = tp.clean_text(raw_html=raw)
    assert txt == ("(a) whether the Government is aware;\n(b) if so, the details thereof;\n"
                   "(c) if not, the reasons therefor?")
    assert not hindi


def test_garbled_hindi_header_line_removed_and_name_kept():
    raw = ("MINISTER OF AGRICULTURE AND FARMERS WELFARE&lt;br/&gt;&lt;br/&gt;"
           "???? ??? ????? ?????? ??????            ( SHRI NARENDRA SINGH TOMAR)  &lt;br/&gt;&lt;br/&gt;"
           "(a)\t    Yes. Potato is produced.")
    txt, hindi = tp.clean_text(raw_html=raw)
    assert hindi
    info, rest = tp.parse_minister_header(txt)
    assert info["minister_name"] == "Narendra Singh Tomar"
    assert info["minister_title"] == "Minister of Agriculture and Farmers Welfare"
    assert rest.startswith("(a) Yes. Potato")


def test_fully_garbled_line_dropped():
    txt, hindi = tp.clean_text(raw_html="???????? ??? ?????? ?????? (???? ????? ????) &lt;br/&gt;THE MINISTER OF COMMERCE &amp; INDUSTRY")
    assert hindi and txt == "THE MINISTER OF COMMERCE & INDUSTRY"


def test_devanagari_line_dropped():
    txt, hindi = tp.clean_text(lines=["ANSWER", "उत्तर सरकार", "(a) Yes."])
    assert hindi and txt == "ANSWER\n(a) Yes."


def test_page_furniture_removed():
    lines = ["(c): Some text", "Page 2 of 3", "-3-", "L.S.US.Q.NO. 2426 FOR 06.08.2024", "Contd..2/-", "more text"]
    txt, _ = tp.clean_text(lines=lines)
    assert txt == "(c): Some text\nmore text"


# ------------------------------------------------------------------ part splitter
def labels(text, mode="answer", **kw):
    pre, parts = tp.split_parts(text, mode, **kw)
    return pre, [(p["label"], p["labels"]) for p in parts]


def test_question_parts_simple_one_line():
    q = ("(a) whether the Government has any proposal; (b) if so, the details thereof; "
         "(c) if not, the reasons therefor; and (d) the plan outlay?")
    pre, parts = tp.split_parts(q, "question")
    assert [p["label"] for p in parts] == ["a", "b", "c", "d"]
    assert parts[2]["text"] == "if not, the reasons therefor; and"
    assert parts[3]["text"] == "the plan outlay?"


def test_question_parts_multiline_with_tabs_and_missing_space():
    q = "(a)whether the Union Government has disbursed GST;\n(b) the total amount of arrears;\n(c) whether cut"
    _, p = labels(q, "question")
    assert [x[0] for x in p] == ["a", "b", "c"]


def test_amp_group():
    _, p = labels("(a) & (b): Occurrence of events.\n(c) & (d): Employment generation.")
    assert p == [("a", ["a", "b"]), ("c", ["c", "d"])]


def test_amp_group_no_spaces_and_dash_colon():
    _, p = labels("(a)&(b): Mahatma Gandhi NREGS.\n(c)&(d):- As per the provision")
    assert p == [("a", ["a", "b"]), ("c", ["c", "d"])]


def test_to_range():
    _, p = labels("(a) to (e): Government has taken several initiatives.")
    assert p == [("a", ["a", "b", "c", "d", "e"])]
    _, p = labels("(a)to (f) AStatement is laid on the Table of the House.")
    assert p == [("a", list("abcdef"))]


def test_and_and_comma_groups():
    _, p = labels("(a) and (b) No. Presently there is no plan.\n(c) to (e) No. As per the information")
    assert p == [("a", ["a", "b"]), ("c", ["c", "d", "e"])]
    _, p = labels("(a), (b) & (c): Yes Sir.\n(d): No.")
    assert p == [("a", ["a", "b", "c"]), ("d", ["d"])]


def test_space_separated_group():
    _, p = labels("(a) & (b):- As per provision.\n(c) (d) & (e):- Net Borrowing Ceiling.\n(f):- No, Sir.")
    assert p == [("a", ["a", "b"]), ("c", ["c", "d", "e"]), ("f", ["f"])]


def test_compact_paren_forms():
    _, p = labels("(a&b) Yes Sir. Construction of Ro Ro Jetty.\n(c) No.")
    assert p == [("a", ["a", "b"]), ("c", ["c"])]
    _, p = labels("( a to d) Union Government devolves States' share.")
    assert p == [("a", list("abcd"))]


def test_nested_roman_sublist_not_split_inside_alpha_parts():
    t = ("(a): The measures are:\n(i) Implementation of Ne-FMS.\n(ii) Intensive consultation.\n"
         "(b): Funds are released.")
    _, p = labels(t)
    assert p == [("a", ["a"]), ("b", ["b"])]


def test_repeated_a_inside_statement_not_new_part():
    t = "(a) to (c): Ministry notifies the list.\n(a) is about the NLEM.\n(d): Ceiling prices."
    _, p = labels(t)
    assert [x[0] for x in p] == ["a", "d"]


def test_inline_references_not_labels():
    t = "(a) Under section 5(b) of the Act and clause (ii) of the order, the details are given."
    _, p = labels(t)
    assert p == [("a", ["a"])]


def test_roman_primary_scheme():
    _, p = labels("(i) first thing;\n(ii) second thing;\n(iii) third thing.", "question")
    assert [x[0] for x in p] == ["i", "ii", "iii"]


def test_a_paren_style_at_line_start():
    _, p = labels("a) whether the Government;\nb) if so, details;\nc) reasons?", "question")
    assert [x[0] for x in p] == ["a", "b", "c"]


def test_question_duplicate_a_typo_becomes_b():
    q = "(a) whether first;\n\n(a)\twhether second;\n\n(c) the details;"
    _, p = labels(q, "question")
    assert [x[0] for x in p] == ["a", "b", "c"]


def test_preamble_kept_when_answer_has_text_before_first_label():
    pre, p = labels("Government has taken steps.\n(a): Yes.\n(b): No.")
    assert pre == "Government has taken steps." and [x[0] for x in p] == ["a", "b"]


def test_no_labels():
    pre, p = labels("There is no such proposal at present.")
    assert p == [] and pre.startswith("There is no")


def test_answer_may_start_at_c():
    _, p = labels("(c) to (e): Yes.", "answer")
    assert p == [("c", ["c", "d", "e"])]


# ------------------------------------------------------------------ minister header
def test_header_standard_two_line():
    t = "ANSWER\nTHE MINISTER OF HEALTH AND FAMILY WELFARE\n(DR. HARSH VARDHAN)\n(a) to (c): A statement is laid"
    info, rest = tp.parse_minister_header(t)
    assert info["minister_name"] == "Harsh Vardhan"
    assert info["minister_name_raw"] == "DR. HARSH VARDHAN"
    assert info["minister_title"] == "Minister of Health and Family Welfare"
    assert info["minister_rank"] == "cabinet"
    assert rest.startswith("(a) to (c)")


def test_header_mos_in_ministry_docx():
    t = ("MINISTER OF STATE IN THE MINISTRY OF RURAL DEVELOPMENT\n(SHRI KAMLESH PASWAN)\n"
         "(a)&(b): Mahatma Gandhi National Rural Employment Guarantee Scheme")
    info, rest = tp.parse_minister_header(t)
    assert info["minister_name"] == "Kamlesh Paswan"
    assert info["minister_title"] == "Minister of State in the Ministry of Rural Development"
    assert info["minister_rank"] == "mos"
    assert rest.startswith("(a)&(b)")


def test_header_independent_charge_paren_in_title():
    t = "THE MINISTER OF STATE (IC) OF THE MINISTRY OF AYUSH\n(SHRI PRATAPRAO JADHAV)\n(a) Yes."
    info, rest = tp.parse_minister_header(t)
    assert info["minister_name"] == "Prataprao Jadhav"
    assert info["minister_rank"] == "mos_ic"
    assert "(IC)" in info["minister_title"]


def test_header_multi_line_multi_portfolio_title():
    t = ("MINISTER OF RAILWAYS, INFORMATION & BROADCASTING AND\nELECTRONICS & INFORMATION TECHNOLOGY\n"
         "(SHRI ASHWINI VAISHNAW)\n(a) to (e) The Amrit Bharat Station Scheme")
    info, rest = tp.parse_minister_header(t)
    assert info["minister_name"] == "Ashwini Vaishnaw"
    assert info["minister_title"].startswith("Minister of Railways, Information & Broadcasting and")
    assert rest.startswith("(a) to (e)")


def test_header_square_brackets_and_leading_br():
    t = "THE MINISTER OF STATE IN THE MINISTRYOF EXTERNAL AFFAIRS\n[SHRI V. MURALEEDHARAN]\n(a) The Government"
    info, rest = tp.parse_minister_header(t)
    assert info["minister_name"] == "V. Muraleedharan"
    assert rest.startswith("(a)")


def test_header_title_only_does_not_steal_paren_acronym():
    t = ("The Finance Minister\n(a) to (d): A statement is laid on the Table of the House.\n***\n"
         "STATEMENT FOR LOK SABHA STARRED QUESTION NO. 401 REGARDING LOSS OF PSBS\n(a) to (d): As per RBI data on "
         "Public Sector Banks (PSBs) increased")
    info, rest = tp.parse_minister_header(t)
    assert info["minister_name"] is None
    assert info["minister_title"] == "Finance Minister"
    assert rest.startswith("(a) to (d): A statement")


def test_header_absent():
    info, rest = tp.parse_minister_header("(a) Yes. Potato, Onion and Tomato are produced in Rajasthan.")
    assert info["minister_name"] is None and info["minister_title"] is None
    assert rest.startswith("(a) Yes")


def test_header_typo_ministr_and_glued():
    info, _ = tp.parse_minister_header("THE MINISTR OF FINANCE\n(SMT. NIRMALA SITHARAMAN)\n(a) & (b) : Yes")
    assert info["minister_name"] == "Nirmala Sitharaman"
    info, _ = tp.parse_minister_header("MINISTEROF STATE FOR JAL SHAKTI\n(SHRI RATTAN LAL KATARIA)\n(a) to (c) To provide")
    assert info["minister_name"] == "Rattan Lal Kataria"


def test_header_name_on_own_line_without_parens():
    info, rest = tp.parse_minister_header("Shri Ashwini Kumar Choubey\n(a) Sir, Global opinions")
    assert info["minister_name"] == "Ashwini Kumar Choubey" and rest.startswith("(a) Sir")


def test_header_answer_prefix():
    info, _ = tp.parse_minister_header("ANSWER\nMINISTER OF DEFENCE                  (SHRI RAJNATH SINGH)\n(a) Yes")
    assert info["minister_name"] == "Rajnath Singh"


# ------------------------------------------------------------------ statement laid / annexure / truncation
def test_statement_laid_with_heading_and_nested_labels():
    rest = ("(a) to (f): A statement is laid on the Table of the House.\n*****\n"
            "STATEMENT REFERRED TO IN REPLY TO PARTS (a) TO (f) OF STARRED QUESTION NO. 12 REGARDING \"DRUGS\".\n"
            "(a) to (c): Ministry notifies NLEM.\n(d): Ceiling prices.\n(e) & (f): All drugs regulated.\n*****")
    r = clean.process_answer("MINISTER OF HEALTH\n(SHRI X Y)\n" + rest)
    assert r["statement_laid"] and not r["statement_empty"]
    assert [p["labels"] for p in r["answer_parts"]] == [["a", "b", "c"], ["d"], ["e", "f"]]
    assert r["grouped_answer"]
    assert "STATEMENT REFERRED" not in r["answer_text"] and "laid on the Table" not in r["answer_text"]


def test_statement_laid_unlabeled_statement_covers_all_laid_labels():
    r = clean.process_answer("THE MINISTER OF HEALTH\n(DR. A B)\n(a) to (f) A Statement is laid on the Table of the House.\n"
                             "STATEMENT REFERRED TO IN REPLY TO LOK SABHA\nSTARRED QUESTION NO. 162 FOR 2ND AUGUST, 2024\n"
                             "As on 30.06.2024, a total number of 29,281 hospitals.\n(i) They can file grievances.\n(ii) Approach SHA.")
    assert r["statement_laid"]
    assert len(r["answer_parts"]) == 1 and r["answer_parts"][0]["labels"] == list("abcdef")
    assert "29,281 hospitals" in r["answer_text"]


def test_statement_laid_but_content_missing():
    r = clean.process_answer("MINISTER OF DEFENCE (SHRI RAJNATH SINGH)\n\n(a) to (g):\tA Statement is laid on the Table of the House.\n*******")
    assert r["statement_laid"] and r["statement_empty"] and r["answer_text"] == ""


def test_stars_right_after_header_do_not_end_answer():
    r = clean.process_answer("MINISTER OF PETROLEUM & NATURAL GAS\n(SHRI DHARMENDRA PRADHAN)\n******\n"
                             "(a) to (c): At present, there is no agreement between OMCs and the farmers for plantation.")
    assert "no agreement" in r["answer_text"]


def test_annexure_split():
    main, annex, ref = tp.split_annexure("(a): Details at Annexure.\n*****\nAnnexure-I\nState-wise details\nAssam 3")
    assert main == "(a): Details at Annexure." and annex.startswith("Annexure-I") and ref


def test_truncated_start_repair():
    t, rep = tp.repair_truncated_start("):  No, Sir.  As per NASSCOM\n(b) and (c):  Does not arise.")
    assert rep and t.startswith("(a): No, Sir.")
    t, rep = tp.repair_truncated_start("to (c):  Ministry of Electronics")
    assert rep and t.startswith("(a) to (c): Ministry")
    t, rep = tp.repair_truncated_start(") to (c): 'Police' and 'Public Order'")
    assert rep and t.startswith("(a) to (c):")
    _, rep = tp.repair_truncated_start("(a) Yes.")
    assert not rep


def test_inferred_first_label():
    r = clean.process_answer("Yes, Sir. The Government has launched it.\n(b): 11611 Common Services Centres are set up.")
    assert r["answer_parts"][0]["labels"] == ["a"] and r["answer_parts"][0]["inferred"]
    assert r["answer_labels"] == ["a", "b"]


def test_question_preamble_stripped():
    q, parts, labs = clean.process_question("Will the Minister of RAILWAYS be pleased to state:\n(a) whether x;\n(b) if so, y?")
    assert labs == ["a", "b"] and q.startswith("(a) whether x")


def test_reflow_joins_hard_wraps_but_not_parts():
    t = "(a) whether the Government is aware of the need to improve the\nfinancial management system;\n(b) if so, the details thereof;"
    assert tp.reflow(t) == "(a) whether the Government is aware of the need to improve the financial management system;\n(b) if so, the details thereof;"


# ------------------------------------------------------------------ boilerplate / coverage
def test_boilerplate_flags():
    f = tp.boilerplate_flags("Information is being collected and will be laid on the Table. The question does not arise. "
                             "There is no such proposal. Data is not maintained centrally. It is under consideration.")
    assert {"info_being_collected", "does_not_arise", "no_such_proposal", "not_maintained_centrally",
            "under_consideration"} <= set(f)


def test_build_record_api_roundtrip():
    q = {"quesNo": 2898, "subjects": "Closing of  Firecracker", "lokNo": "17", "member": ["Manickam Tagore B"],
         "ministry": "COMMERCE & INDUSTRY", "type": "UNSTARRED ", "date": "10.07.2019",
         "questionText": "(a)\t whether aware; &lt;br/&gt;(b) \tif so, the details thereof; &lt;br/&gt;(c) \treasons?&lt;br/&gt;",
         "answerText": "???????? ??? (???? ????) &lt;br/&gt;THE MINISTER OF COMMERCE & INDUSTRY&lt;br/&gt;(SHRI  PIYUSH GOYAL)&lt;br/&gt;"
                       "&lt;br/&gt;(a) \tDepartment does not collect information. &lt;br/&gt;(b) to (c) : \tDo not arise.   &lt;br/&gt;&lt;br/&gt;******&lt;br/&gt;",
         "sessionNo": "1"}
    r = clean.build_record(17, q)
    assert r["qid"] == "ls17-s1-U-2898-20190710" and r["qtype"] == "unstarred"
    assert r["ministry"] == "COMMERCE AND INDUSTRY" and r["ministry_raw"] == "COMMERCE & INDUSTRY"
    assert r["minister_name"] == "Piyush Goyal" and r["minister_rank"] == "cabinet"
    assert r["question_labels"] == ["a", "b", "c"] and r["answer_labels"] == ["a", "b", "c"]
    assert r["all_parts_covered"] is True and r["grouped_answer"] and r["answer_has_hindi"]
    assert "does_not_arise" in r["boilerplate"] and not r["answer_missing"]
    assert r["answer_text"].startswith("(a) Department does not collect") and "******" not in r["answer_text"]
    aligned = clean.align_parts(r["question_parts"], r["answer_parts"])
    assert aligned["c"]["answer"] == "Do not arise." and aligned["c"]["shared"]


def test_ls16_hard_wrapped_html_and_typo_header():
    raw = ("\r<br>\n<br>\n\nMINSTER OF STATE FOR SOCIAL JUSTICE AND EMPOWERMENT (SHRI SUDARSHAN BHAGAT)\r<br>\n<br>\n\n"
           "(a) & (b): Scholarships are provided to mentally retarded students \r\nto pursue their studies under three \r\n"
           "schemes.\r<br>\n<br>\n\n(c) & (d):  Do not arise.\r\n<br>")
    txt, _ = tp.clean_text(raw_html=raw)
    assert "students to pursue their studies under three schemes." in txt
    r = clean.process_answer(txt)
    assert r["minister_name"] == "Sudarshan Bhagat" and r["minister_rank"] == "mos"
    assert r["answer_labels"] == ["a", "b", "c", "d"]


def test_docx_numbering_labels():
    import docx_extract as dx
    from lxml import etree
    num = (b'<w:numbering xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
           b'<w:abstractNum w:abstractNumId="0"><w:lvl w:ilvl="0"><w:start w:val="1"/><w:numFmt w:val="lowerLetter"/>'
           b'<w:lvlText w:val="(%1)"/></w:lvl><w:lvl w:ilvl="1"><w:start w:val="1"/><w:numFmt w:val="lowerRoman"/>'
           b'<w:lvlText w:val="(%2)"/></w:lvl></w:abstractNum><w:num w:numId="1"><w:abstractNumId w:val="0"/></w:num></w:numbering>')
    n = dx.Numbering(num, None)
    W = dx.W

    def para(il):
        return etree.fromstring(f'<w:p xmlns:w="{W[1:-1]}"><w:pPr><w:numPr><w:ilvl w:val="{il}"/><w:numId w:val="1"/>'
                                f'</w:numPr></w:pPr><w:r><w:t>x</w:t></w:r></w:p>')
    assert [n.label(para(i)) for i in (0, 0, 1, 1, 0)] == ["(a)", "(b)", "(i)", "(ii)", "(c)"]


def test_dotted_and_inline_letter_labels():
    q = "a. whether the Government is promoting;\nb. the steps taken;\nc. the details of changes; and d. the impact?"
    _, p = labels(q, "question")
    assert [x[0] for x in p] == ["a", "b", "c", "d"]
    # '1.' enumerations inside an answer are not part labels
    _, p = labels("(a): The measures are:\n1. First measure.\n2. Second measure.\n(b): No.")
    assert p == [("a", ["a"]), ("b", ["b"])]


def test_header_title_split_around_name():
    info, rest = tp.parse_minister_header(
        "THE MINISTER OF STATE IN THE (SHRI BHUPATHIRAJU SRINIVASA VARMA)\nMINISTRY OF STEEL\n(a)to(d):  Steel is")
    assert info["minister_title"] == "Minister of State in the Ministry of Steel"
    assert info["minister_name"] == "Bhupathiraju Srinivasa Varma" and rest.startswith("(a)to(d)")


def test_statement_heading_without_lay_sentence_and_placed_variant():
    r = clean.process_answer("[\n\nSTATEMENT REFERRED TO IN REPLY TO PARTS (a) TO (d) OF\nUNSTARRED QUESTION NO.1582 BY SHRI X TO BE\n"
                             "ANSWERED IN LOK SABHA ON 31.07.2024 REGARDING RAILWAY\nDIVISION\n(a) to (d): No work has been sanctioned.")
    assert r["statement_laid"] and r["answer_text"].startswith("(a) to (d): No work")
    r = clean.process_answer("MINISTER OF OIL (SHRI X Y)\n(a), (b) and (c): A statement is placed on the Table of the House.\n"
                             "STATEMENT IN REPLY TO THE LOK SABHA STARRED QUESTION NO. 213\nFOR ANSWER ON 6TH AUGUST, 2024 RAISED BY SHRI V T\n"
                             "REGARDING REPEALING OF ARTICLE 370\n(a): Before abrogation.\n(b)&(c): No.")
    assert r["statement_laid"] and r["answer_labels"] == ["a", "b", "c"] and r["answer_text"].startswith("(a): Before")


def test_glued_statement_lay_sentence():
    r = clean.process_answer("THE MINISTER OF HEALTH\n(DR. A B)\n(a)to (c) AStatement is laid on the Table of the House.\n"
                             "STATEMENT REFERRED TO IN REPLYTO LOK SABHA\nSTARRED QUESTION NO. 173 FOR 2ND AUGUST, 2024\n"
                             "As on 15.07.2024, a total number of 3029 beneficiaries.")
    assert r["statement_laid"] and r["answer_parts"][0]["labels"] == ["a", "b", "c"]


def test_superscript_ordinal_and_hindi_mixed_line():
    txt, hindi = tp.clean_text(lines=["(To be answered on the 1 st August 2024)",
                                      "कलयाण मंती (SHRI SHIVRAJ SINGH CHOUHAN) ृ"])
    assert hindi and txt == "(To be answered on the 1st August 2024)\n(SHRI SHIVRAJ SINGH CHOUHAN)"


def test_out_of_order_groups_need_question_labels():
    t = "(a) & (d): Sam Pitroda report.\nMore text.\n(b) & (c): No, Madam."
    r = clean.process_answer(t, q_labels=list("abcd"), q_scheme="alpha")
    assert r["answer_labels"] == ["a", "d", "b", "c"]
    assert clean.process_answer(t)["answer_labels"] == ["a", "d"]  # without the question's labels: stay strict


def test_fuzzy_canonical_minister_variants():
    from collections import Counter
    v = {"anupriyapatel": Counter({"Anupriyapatel": 3, "Anupriya Patel": 90}),
         "ministerofstateintheminstryofrailways": Counter({"Minister of State in the Minstry of Railways": 2}),
         "ministerofstateintheministryofrailways": Counter({"Minister of State in the Ministry of Railways": 300})}
    m = clean.fuzzy_canonical(v, cutoff=94, rare=20, frequent=40)
    assert m["anupriyapatel"] == "Anupriya Patel"
    assert m["ministerofstateintheminstryofrailways"] == "Minister of State in the Ministry of Railways"


def test_qa_fields_swapped_or_misfiled():
    base = {"quesNo": 1, "type": "UNSTARRED", "date": "02.03.2015", "sessionNo": "5", "member": ["X"], "ministry": "COAL",
            "subjects": "S"}
    q = dict(base, questionText="MINISTER OF COAL&lt;br/&gt;(SHRI A B)&lt;br/&gt;(a) Yes.&lt;br/&gt;(b) No.",
             answerText="Will the Minister of COAL be pleased to state:&lt;br/&gt;(a) whether x;&lt;br/&gt;(b) if so, y?")
    r = clean.build_record(16, q)
    assert r["qa_mismatch"] and r["question_labels"] == ["a", "b"] and r["minister_name"] == "A B"
    assert r["answer_text"].startswith("(a) Yes.")
    q2 = dict(base, questionText="(a) whether x;&lt;br/&gt;(b) y?",
              answerText="GOVERNMENT OF INDIA&lt;br/&gt;LOK SABHA&lt;br/&gt;Will the Minister of COAL be pleased to state:&lt;br/&gt;(a) whether x;")
    r = clean.build_record(16, q2)
    assert r["qa_mismatch"] and r["answer_missing"] and not r["question_missing"]


def test_statement_enclosed_and_double_dash_page_marker():
    r = clean.process_answer("MINISTER OF DEFENCE (SHRI A B)\n(a) to (c): A Statement is enclosed.\n"
                             "STATEMENT REFERRED IN REPLY TO LOK SABHA UNSTARRED QUESTION NO. 3242 DUE FOR ANSWER ON 28TH JULY, 2009.\n"
                             "(a) to (c): During the last three years, two incidents.")
    assert r["statement_laid"] and r["answer_text"].startswith("(a) to (c): During")
    txt, _ = tp.clean_text(lines=["--2--", "(a): text", "-3-"])
    assert txt == "(a): text"


if __name__ == "__main__":  # plain-assert runner when pytest is unavailable
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("ok  ", name)
            except Exception as e:  # noqa: BLE001
                fails += 1
                print("FAIL", name, repr(e))
    sys.exit(1 if fails else 0)
