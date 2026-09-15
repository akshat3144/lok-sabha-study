"""Tests for the replacement-file (docs_fallback) paths, the HTML extractor and the ministry map.

Snippets are real: .htm bodies from data/raw/docs_fallback (LS14/LS15/LS16), docx/pdf line lists as produced by
docx_extract / pypdf for LS17 and LS18 files, API answers from LS14-LS17.
    python -I -X utf8 -m pytest pipeline/tests -q
"""
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import clean  # noqa: E402
import htm_extract as hx  # noqa: E402
import textparse as tp  # noqa: E402

ROOT = clean.ROOT
ALLOWED_GROUPS = {"security", "economy_finance", "infrastructure_transport", "social_welfare_health_education",
                  "agriculture_rural", "energy_environment_resources", "industry_commerce_labour",
                  "governance_law_other"}


def fb_path(ls, name):
    return os.path.join(clean.RAW_FALLBACK, f"ls{ls}", name)


def mk_q(ls_session="1", typ="UNSTARRED ", no=5, date="10.07.2019", ministry="RAILWAYS", q=None, a=None):
    return {"quesNo": no, "subjects": "S", "member": ["A B"], "ministry": ministry, "type": typ, "date": date,
            "sessionNo": ls_session, "questionText": q, "answerText": a}


Q_PARTS = "(a) the details of the railway workshops;\n(b) the present status thereof; and\n(c) the time frame?"
LAY_API = ("MINISTER OF STATE IN THE MINISTRY OF HOME AFFAIRS  (SHRI SRIPRAKASH JAISWAL)\r<br>\n<br>\n\n"
           "(a) to (e):  A Statement is laid on the Table of the House.\r<br>\n<br>")


# ------------------------------------------------------------------ html extractor
def test_decode_cp1252_declared_and_curly_quotes():
    raw = (b'<html><head><meta http-equiv="Content-Type" content="text/html; charset=windows-1252"></head><body>'
           b'<p>People\x92s Liberation Army (PLA) \x96 \x93banned\x94</p></body></html>')
    lines, st = hx.html_bytes_to_lines(raw)
    assert lines == ["People’s Liberation Army (PLA) – “banned”"]
    assert st["encoding"] == "cp1252"


def test_decode_utf8_bom_and_utf8_declared_cp1252():
    body = "<html><body><p>Details – Rs. 5 crore ‘x’</p></body></html>"
    lines, st = hx.html_bytes_to_lines(b"\xef\xbb\xbf" + body.encode("utf8"))
    assert lines == ["Details – Rs. 5 crore ‘x’"] and st["encoding"] == "utf-8-sig"
    # declared windows-1252 but the bytes are valid UTF-8 (seen in LS15/LS16 files): trust the bytes
    hdr = '<meta http-equiv="Content-Type" content="text/html; charset=windows-1252">'
    lines, st = hx.html_bytes_to_lines((hdr + body).encode("utf8"))
    assert lines == ["Details – Rs. 5 crore ‘x’"] and st["encoding"] == "utf-8"


def test_word_html_strips_comments_styles_tags_and_unwraps_source_lines():
    raw = b"""<html xmlns:o="urn:schemas-microsoft-com:office:office">
<head><meta http-equiv=Content-Type content="text/html; charset=windows-1252"><title>L</title>
<!--[if gte mso 9]><xml><o:DocumentProperties><o:Author>R&amp;I</o:Author></o:DocumentProperties></xml><![endif]-->
<style><!-- p.MsoNormal {margin:0in} --></style></head>
<body><script>var x = 1;</script>
<p class="MsoNormal" align="right"><b>ANNEXURE</b></p>
<p class="MsoNormal" style="text-align:justify"><span style="font-size:12.0pt;
font-family:&quot;Arial Black&quot;">(a)<span style="mso-spacerun: yes">&nbsp; </span>to
(e):<span style="mso-tab-count:1">&nbsp;&nbsp;&nbsp;</span>Yes, Sir. The Joint Secretary level talks between India and
Bangladesh were held on 29-30 May, 2008 at New Delhi.<o:p>
</o:p>
</span></p></body></html>"""
    lines, _ = hx.html_bytes_to_lines(raw)
    assert lines == ["ANNEXURE",
                     "(a) to (e): Yes, Sir. The Joint Secretary level talks between India and Bangladesh were held "
                     "on 29-30 May, 2008 at New Delhi."]


def test_html_table_rows_become_pipe_lines_and_cell_paragraphs_are_merged():
    raw = b"""<html><body><p>Details are given below:-</p>
<table border=1>
<tr><td><p>Railway</p></td><td><p>Workshop</p></td></tr>
<tr><td rowspan=2><p>CENTRAL</p></td><td><p>Locomotive Workshop,
<span class="SpellE">Parel</span></p></td></tr>
<tr><td><p>Carriage Workshop, Matunga</p></td></tr>
<tr><td>Total</td><td>102.42</td><td>5446</td></tr>
</table></body></html>"""
    lines, _ = hx.html_bytes_to_lines(raw)
    assert lines == ["Details are given below:-", "Railway | Workshop", "CENTRAL | Locomotive Workshop, Parel",
                     "Carriage Workshop, Matunga", "Total | 102.42 | 5446"]


def test_single_cell_layout_table_keeps_line_breaks():
    raw = b"<html><body><table><tr><td><p>STATEMENT</p><p>(a) Yes.</p><p>(b) No.</p></td></tr></table></body></html>"
    lines, _ = hx.html_bytes_to_lines(raw)
    assert lines == ["STATEMENT", "(a) Yes.", "(b) No."]


def test_legacy_hindi_glyph_soup_removed_from_english_header():
    raw = ("<p>Will the Minister of RAILWAYS (¸êü»Ö ´ÖÓ¡Öß) "
           "be pleased to state:</p><p>¸êü»Ö ´ÖÓ¡Öß</p>").encode("utf8")
    lines, st = hx.html_bytes_to_lines(raw)
    assert lines == ["Will the Minister of RAILWAYS be pleased to state:"] and st["legacy_hindi"]


def test_real_fallback_htm_files_parse_if_present():
    p = fb_path(14, "ls14-s14-U-2647-20081216.htm")
    if not os.path.exists(p):
        return
    lines, st = hx.htm_lines(p)
    assert lines[0] == "ANNEXURE" and lines[1].startswith("STATEMENT IN REPLY TO PARTS (a) TO (e)")
    assert lines[2].startswith("(a) to (e): Yes, Sir.") and "Insurgent Groups" in lines[2] or "Insurgent" in lines[2]


# ------------------------------------------------------------------ fallback_index / load_doc_lines
def test_fallback_index_prefers_docx_over_pdf_over_htm_and_skips_junk(tmp_path):
    d = tmp_path / "ls16"
    d.mkdir()
    for fn in ("a.htm", "a.pdf", "a.docx", "b.pdf", "b.htm", "c.htm", "c.htm.part", "d.txt", "e.docx.tmp"):
        (d / fn).write_bytes(b"x")
    (tmp_path / "stray.docx").write_bytes(b"x")
    idx = clean.fallback_index(str(tmp_path))
    assert {k: os.path.splitext(v)[1] for k, v in idx.items()} == {"a": ".docx", "b": ".pdf", "c": ".htm"}


def test_load_doc_lines_htm_cached_and_namespaced(tmp_path, monkeypatch):
    monkeypatch.setattr(clean, "CACHE_DIR", str(tmp_path / "cache"))
    d = tmp_path / "docs_fallback" / "ls15"
    d.mkdir(parents=True)
    f = d / "ls15-s4-S-502-20100430.htm"
    f.write_bytes(b"<html><body><p>STATEMENT REFERRED TO IN REPLY TO LOK SABHA STARRED QUESTION NO. 502 FOR 30TH APRIL, 2010</p>"
                  b"<p>(a) &amp; b) Yes, Madam.</p></body></html>")
    lines, st, err = clean.load_doc_lines(str(f))
    assert err is None and lines[1] == "(a) & b) Yes, Madam."
    assert os.path.exists(tmp_path / "cache" / "fb_ls15" / "ls15-s4-S-502-20100430.json.gz")
    assert clean.load_doc_lines(str(f))[0] == lines  # second call served from cache


# ------------------------------------------------------------------ build_record: htm statement body
def htm_fb(lines, name="ls14-s8-S-135-20060801.htm"):
    return (fb_path(14, name), lines, {"n_pages": 1, "n_media": 0}, None)


def test_htm_statement_body_replaces_empty_statement_stub():
    q = mk_q(no=135, typ="STARRED", date="01.08.2006", ministry="HOME AFFAIRS",
             q="(a) whether banned organisations exist;\n(b) the funds received; and\n(c) the steps taken?", a=LAY_API)
    r0 = clean.build_record(14, q)
    assert r0["statement_laid"] and r0["statement_empty"] and r0["answer_missing"]
    fb = htm_fb(["STATEMENT REFERRED TO IN REPLY TO LOK SABHA STARRED QUESTION NO. 135 FOR ANSWER ON 01.08.2006 "
                 "REGARDING BANNED TERRORIST ORGANISATIONS BY S/SHRI CHANDRA MANI TRIPATHI AND VIJAY KUMAR MALHOTRA, MPs.",
                 "(a) Thirty two organizations continue to be banned as terrorist organisations under the Unlawful "
                 "Activities (Prevention) Act 1967. A list of these organizations is annexed.",
                 "(b) & (c) According to available reports, terrorist groups are receiving funds through hawala.",
                 "ANNEXURE", "1. Babbar Khalsa International", "2. Khalistan Commando Force"])
    r = clean.build_record(14, q, None, fb)
    assert not r["answer_missing"] and not r["statement_empty"] and r["statement_laid"]
    assert r["source_a"] == "fallback_htm" and r["source"] == "fallback_htm"
    assert r["fallback_file"].endswith("docs_fallback/ls14/ls14-s8-S-135-20060801.htm") and not r["needs_ocr"]
    assert r["answer_text"].startswith("(a) Thirty two organizations") and "REFERRED TO IN REPLY" not in r["answer_text"]
    assert r["answer_labels"] == ["a", "b", "c"] and r["all_parts_covered"] is True
    assert r["has_annexure"] and "Babbar Khalsa" in r["annexure_text"] and "Babbar" not in r["answer_text"]
    # API metadata is kept
    assert r["minister_name"] == "Sriprakash Jaiswal" and r["minister_rank"] == "mos"
    assert r["ministry_canonical"] == "Home Affairs" and r["ministry_group"] == "security"


def test_htm_that_starts_with_annexure_heading_before_statement_heading():
    # real LS14 file: ANNEXURE / STATEMENT IN REPLY TO PARTS (a) TO (e) ... / body
    q = mk_q(no=2647, date="16.12.2008", ministry="EXTERNAL AFFAIRS",
             q="(a) whether talks were held;\n(b) the details;\n(c) the decisions;\n(d) the outcome; and\n(e) the follow-up?",
             a="MINISTER OF STATE IN THE MINISTRY OF EXTERNAL AFFAIRS (SHRI E AHAMED)<br>(a) to (e): A Statement is laid on the Table of the House.")
    fb = htm_fb(["ANNEXURE",
                 "STATEMENT IN REPLY TO PARTS (a) TO (e) OF LOK SABHA UNSTARRED QUESTION NO.2647 FOR 16.12.2008 "
                 "REGARDING JOINT SECRETARY LEVEL TALKS WITH BANGLADESH.",
                 "(a) to (e): Yes, Sir. The Joint Secretary level talks between India and Bangladesh were held on "
                 "29-30 May, 2008 at New Delhi."], "ls14-s14-U-2647-20081216.htm")
    r = clean.build_record(14, q, None, fb)
    assert r["answer_text"].startswith("(a) to (e): Yes, Sir.") and not r["statement_empty"]
    assert r["answer_labels"] == ["a", "b", "c", "d", "e"] and not r["has_annexure"]


def test_htm_that_is_only_an_annexure_table_becomes_the_answer():
    # LS13: the API reply points at a statement; the file is "ANNEXURE OF UNSTARRED QUESTION NO-4554" + table
    q = mk_q(no=4554, date="22.04.2003", ministry="HUMAN RESOURCE DEVELOPMENT", q="(a) the details of grants;\n(b) the names?",
             a="MINISTER OF STATE IN THE MINISTRY OF HUMAN RESOURCE DEVELOPMENT (SHRI X Y)<br>(a) to (b): A statement is laid on the Table of the House.")
    fb = htm_fb(["ANNEXURE OF UNSTARRED QUESTION NO-4554", "Statement",
                 "Grants-in-Aid released to NGOs under STEP scheme during 2002-03",
                 "1. Biswa Santi Sakti Parishad, Guwahati | 1.6.200 | 600"], "ls13-s12-U-4554-20030422.htm")
    r = clean.build_record(13, q, None, fb)
    assert not r["answer_missing"] and "Biswa Santi" in r["answer_text"]
    assert r["has_annexure"] and r["annexure_text"] == "" and "fallback_annexure_only" in r["doc_note"]


def test_complete_reply_inside_an_htm_is_treated_like_a_docx_reply():
    # real LS14 file (ls14-s13-S-305): question + ANSWER + minister header + lay sentence; API answer is a stub
    q = mk_q(no=305, typ="STARRED", date="20.03.2008", ministry="RAILWAYS", q="(a) the details of surveys;\n(b) the proposals; and\n(c) the status?",
             a="MINISTER OF RAILWAYS(SHRI LALU PRASAD)<br>(a) to (c): A Statement is laid on the Table of the Sabha.")
    lines = ["LOK SABHA", "STARRED QUESTION NO. 305", "TO BE ANSWERED ON 20.03.2008", "SURVEYS FOR NEW RAILWAY PROJECTS",
             "*305. SHRI G.M. SIDDESWARA:", "Will the Minister of RAILWAYS be pleased to state:",
             "(a) the details of the on-going surveys for new rail projects;", "(b the details of the proposals for the new surveys; and",
             "(c) the present status of the on-going surveys?", "ANSWER", "MINISTER OF RAILWAYS (SHRI LALU PRASAD)",
             "(a) to (c): A Statement is laid on the Table of the Sabha.", "STATEMENT REFERRED TO IN REPLY TO PARTS (a) TO (c) OF STARRED QUESTION NO. 305",
             "(a) to (c): Surveys have been taken up for 12 new lines in the southern region."]
    r = clean.build_record(14, q, None, htm_fb(lines, "ls14-s13-S-305-20080320.htm"))
    assert r["source_a"] == "fallback_htm" and not r["answer_missing"]
    assert r["answer_text"].startswith("(a) to (c): Surveys have been taken up")
    assert r["minister_name"] == "Lalu Prasad" and not r["qa_mismatch"]


def test_htm_not_used_when_api_answer_is_complete_but_annexure_is_attached():
    api = ("MINISTER OF HEALTH (SHRI A B)<br>(a) & (b): The scheme provides support to all States. The data are "
           "given in the statement enclosed.")
    q = mk_q(no=77, ministry="HEALTH AND FAMILY WELFARE", q="(a) whether x;\n(b) the details?", a=api)
    fb = htm_fb(["STATEMENT", "State | Amount", "Kerala | 10"])
    r = clean.build_record(14, q, None, fb)
    assert r["source_a"] == "api" and r["answer_text"].startswith("(a) & (b): The scheme provides support")
    assert not r["statement_laid"] and r["has_annexure"] and "Kerala | 10" in r["annexure_text"]
    assert r["fallback_file"]


# ------------------------------------------------------------------ build_record: docx / pdf replacement of a whole reply
DOCX_LINES = [
    "GOVERNMENT OF INDIA", "MINISTRY OF RAILWAYS", "LOK SABHA", "UNSTARRED QUESTION NO. 2536", "TO BE ANSWERED ON 10.03.2021",
    "TERMINATION OF CONTRACTS", "2536. SHRI MARGANI BHARAT:", "Will the Minister of RAILWAYS be pleased to state:",
    "(a) whether the Indian Railways has approached the World Bank to", "terminate its contracts with Chinese companies recently;",
    "(b) if so, the details of each contract given to Chinese companies by", "Railways and the contracts that have been terminated, project-wise;",
    "(c) the status of progress of each of the projects; and", "(d) the details of projects that are likely to be affected in AP?",
    "ANSWER", "MINISTER OF RAILWAYS, COMMERCE & INDUSTRY AND", "CONSUMER AFFAIRS, FOOD & PUBLIC DISTRIBUTION", "(SHRI PIYUSH GOYAL)",
    "(a) to (d): A statement is laid on the Table of the House.",
    "STATEMENT REFERRED TO IN REPLY TO PARTS (a) TO (d) OF", "UNSTARRED QUESTION NO.2536 BY SHRI MARGANI BHARAT TO BE",
    "ANSWERED IN LOK SABHA ON 10.03.2021 REGARDING TERMINATION", "OF CONTRACTS",
    "(a) and (b): Dedicated Freight Corridor Corporation of India Limited", "(DFCCIL) had approached the World Bank for termination of one",
    "Signal & Telecommunication (S&T) Contract. The same was terminated on", "17.07.2020 on account of very poor progress.",
    "(c): No further Contracts in EDFC are slated to be terminated as of now.",
    "(d): No projects funded by World Bank in Railway sector is likely to be", "affected in AP.", "*****",
]


def test_docx_fallback_recovers_statement_when_api_has_only_the_stub():
    api = ("MINISTER OF RAILWAYS, COMMERCE & INDUSTRY AND &lt;br/&gt;CONSUMER AFFAIRS, FOOD & PUBLIC DISTRIBUTION&lt;br/&gt;&lt;br/&gt;"
           "(SHRI PIYUSH GOYAL)&lt;br/&gt; &lt;br/&gt;(a) to (d): A statement is laid on the Table of the House.&lt;br/&gt;&lt;br/&gt;*****")
    q = mk_q("5", "UNSTARRED ", 2536, "10.03.2021", "RAILWAYS", q="(a) whether x;\n(b) if so;\n(c) status; and\n(d) AP?", a=api)
    assert clean.build_record(17, q)["statement_empty"]
    fb = (fb_path(17, "ls17-s5-U-2536-20210310.docx"), DOCX_LINES, {"n_pages": 2, "n_media": 0}, None)
    r = clean.build_record(17, q, None, fb)
    assert r["source_a"] == "fallback_docx" and not r["answer_missing"] and not r["statement_empty"]
    assert r["answer_labels"] == ["a", "b", "c", "d"] and r["all_parts_covered"] is True
    assert "DFCCIL" in r["answer_text"] and "REFERRED TO IN REPLY" not in r["answer_text"]
    assert r["minister_name"] == "Piyush Goyal" and r["minister_rank"] == "cabinet"
    # the API question is kept (the fallback question body is only used when the API question is empty)
    assert r["source_q"] == "api" and r["question_text"].startswith("(a) whether x;")


def test_docx_fallback_fills_missing_question_and_answer():
    q = mk_q("5", "UNSTARRED ", 2536, "10.03.2021", "RAILWAYS", q="", a="")
    fb = (fb_path(17, "ls17-s5-U-2536-20210310.docx"), DOCX_LINES, {"n_pages": 2, "n_media": 0}, None)
    r = clean.build_record(17, q, None, fb)
    assert r["source_q"] == "fallback_docx" and r["question_labels"] == ["a", "b", "c", "d"]
    assert r["source_a"] == "fallback_docx" and not r["answer_missing"]


def test_fallback_overrides_answer_filed_under_question_text_mismatch():
    # the API put the question under answerText; the reply is recovered from the file
    q = mk_q("5", "UNSTARRED ", 2536, "10.03.2021", "RAILWAYS", q="(a) whether x;\n(b) y?",
             a="Will the Minister of RAILWAYS be pleased to state:<br/>(a) whether x;<br/>(b) y?")
    fb = (fb_path(17, "ls17-s5-U-2536-20210310.docx"), DOCX_LINES, {"n_pages": 2, "n_media": 0}, None)
    r = clean.build_record(17, q, None, fb)
    assert r["qa_mismatch"] and r["source_a"] == "fallback_docx" and not r["answer_missing"]


PDF_LINES = [
    "GOVERNMENT OF INDIA ", "MINISTRY OF HOUSING AND URBAN AFFAIRS ", "LOK SABHA ", "UNSTARRED QUESTION NO.555 ",
    "TO BE ANSWERED ON NOVEMBER 28, 2024 ", " ", "HOUSING REQUIREMENT IN HARYANA ", " ", " ", "NO. 555. SHRI VARUN CHAUDHRY:    ", " ",
    "Will the Minister of HOUSING AND URBAN AFFAIRS be pleased to state: ",
    "(a) whether any survey has been conducted or applications invited to ",
    "ascertain the housing requirement of the urban and rural poor and if so, ", "the details thereof; and ",
    "(b) the number of applications pending at present for housing requirement in ",
    "Haryana and the time by which the pendency would be cleared? ", " ", "ANSWER ", "THE MINISTER OF STATE IN THE ",
    " MINISTRY OF HOUSING AND URBAN AFFAIRS ", "(SHRI TOKHAN SAHU) ", " ",
    "(a) &(b): ‘Land’ and ‘Colonisation’ are State subjects. Therefore, schemes ",
    "related to housing for their citizens are implemented by States/Union ", "Territories (UTs). However, Ministry of Housing and Urban Affairs (MoHUA) ",
    "supplements the efforts of States/UTs by providing Central Assistance ", "under Pradhan Mantri Awas Yojana – Urban (PMAY-U) since June 25, 2015 ",
]


def test_pdf_fallback_replaces_unusable_ls18_docx():
    q = mk_q("3", "UNSTARRED ", 555, "28.11.2024", "HOUSING AND URBAN AFFAIRS", q=None, a=None)
    bad_doc = (os.path.join(clean.RAW_DOCS, "ls18", "x.docx"), [], {}, "BadDocx:no_document_xml")
    fb = (fb_path(18, "ls18-s3-U-555-20241128.pdf"), PDF_LINES, {"n_pages": 1, "n_media": 0}, None)
    r0 = clean.build_record(18, q, bad_doc)
    assert r0["answer_missing"] and r0["doc_note"] == "bad_no_document_xml" and not r0["needs_ocr"]
    r = clean.build_record(18, q, bad_doc, fb)
    assert r["source_a"] == "fallback_pdf" and r["source"] == "fallback_pdf" and not r["answer_missing"]
    assert r["doc_note"] == "bad_no_document_xml"  # the original doc's problem stays recorded
    assert r["fallback_file"].endswith("ls18-s3-U-555-20241128.pdf") and not r["needs_ocr"]
    assert r["minister_name"] == "Tokhan Sahu" and r["minister_rank"] == "mos"
    assert r["answer_labels"] == ["a", "b"] and "State subjects" in r["answer_text"]
    assert r["source_q"] == "fallback_pdf" and r["question_labels"] == ["a", "b"]
    assert r["ministry_canonical"] == "Housing and Urban Affairs"


def test_image_only_pdf_sets_needs_ocr_and_never_invents_text():
    q = mk_q("2", "STARRED", 211, "06.08.2024", "RAILWAYS", q=None, a=None)
    img_doc = (os.path.join(clean.RAW_DOCS, "ls18", "x.docx"), [], {"n_pages": 1, "n_media": 2}, None)
    fb = (fb_path(18, "ls18-s2-S-211-20240806.pdf"), ["", " "], {"n_pages": 3, "n_media": 0}, None)
    r = clean.build_record(18, q, img_doc, fb)
    assert r["answer_missing"] and r["needs_ocr"] and r["answer_text"] == "" and r["source_a"] is None
    assert r["doc_note"] == "image_only;fallback_image_only" and r["fallback_file"]


def test_needs_ocr_false_when_answer_recovered_or_pdf_is_hindi_only():
    q = mk_q("2", "UNSTARRED ", 1790, "01.08.2024", "CIVIL AVIATION", q=None, a=None)
    hindi = ["भारत सरकार", "लोक सभा",
             "उतर नागर विमानन मंत्रालय"]
    fb = (fb_path(18, "ls18-s2-U-1790-20240801.docx"), hindi, {"n_pages": 1, "n_media": 0}, None)
    r = clean.build_record(18, q, None, fb)
    assert r["answer_missing"] and not r["needs_ocr"] and r["doc_note"] == "fallback_hindi_only"


def test_corrupt_fallback_records_error_note_without_crashing():
    q = mk_q("2", "UNSTARRED ", 1, "01.08.2024", "POWER", q="(a) x?", a=None)
    fb = (fb_path(18, "ls18-s2-U-1-20240801.pdf"), [], {}, "PdfReadError:EOF marker not found")
    r = clean.build_record(18, q, None, fb)
    assert r["answer_missing"] and "fallback_bad_" in r["doc_note"] and not r["needs_ocr"]


def test_fallback_not_applied_when_api_answer_is_fine():
    q = mk_q("1", "UNSTARRED ", 2898, "10.07.2019", "COMMERCE & INDUSTRY", q="(a) whether aware;\n(b) details?",
             a="THE MINISTER OF COMMERCE & INDUSTRY<br>(SHRI PIYUSH GOYAL)<br><br>(a) Yes.<br>(b) Details are as follows.")
    fb = (fb_path(17, "ls17-s1-U-2898-20190710.docx"), DOCX_LINES, {"n_pages": 2, "n_media": 0}, None)
    r = clean.build_record(17, q, None, fb)
    assert r["source_a"] == "api" and r["fallback_file"] is None and r["answer_text"].startswith("(a) Yes.")


# ------------------------------------------------------------------ statement parsing fixes
def test_statement_enclosed_after_real_answer_is_not_a_lay_sentence():
    # real LS13 answer: the whole reply used to be discarded because of 'given in the statement enclosed'
    api = ("THE MINISTER OF TRIBAL AFFAIRS (SHRI JUAL ORAM):\n(a) and (b) No, `Sir. However, the subject of Panchayat Raj is "
           "being dealt in Ministry of Rural Development as per the Allocation of Business Rules.\n(c) Various States "
           "Governments have enacted Laws to restrict transfer of tribal land in the Scheduled Areas. A list of Land "
           "Alienation Laws in TSP States is given in the statement enclosed.")
    r = clean.process_answer(api)
    assert not r["statement_laid"] and not r["statement_empty"]
    assert r["answer_labels"] == ["a", "b", "c"] and "Land Alienation Laws" in r["answer_text"]
    # a genuine short lay sentence is still a statement
    r2 = clean.process_answer("MINISTER OF DEFENCE (SHRI A B)\n(a) to (c): A Statement is enclosed.\n(a) to (c): Details.")
    assert r2["statement_laid"] and r2["answer_text"].startswith("(a) to (c): Details")


def test_statement_heading_flattened_onto_one_line_with_body():
    # real LS18 docx: heading and body arrive as one long line
    line = ('Statement referred to in reply to parts (a) to (d) in respect of Lok Sabha Starred Question No. 399 for reply on '
            '27.03.2025 regarding "Point of Call Status to Kannur International Airport" asked by Shri K Sudhakaran.  '
            '(a) to (d) Grant of point of call (PoC) to the designated carriers of foreign countries is a continuous process.')
    r = clean.process_answer("Minister of CIVIL AVIATION (Shri Kinjarapu Rammohan Naidu)\n"
                             "(a) to (d): A Statement is laid on the Table of the House.\n******\n" + line)
    assert r["statement_laid"] and not r["statement_empty"]
    assert r["answer_text"].startswith("(a) to (d) Grant of point of call") and "Starred Question" not in r["answer_text"]


# ------------------------------------------------------------------ ministry map
def _map_rows():
    with open(clean.MINISTRY_MAP_CSV, encoding="utf8", newline="") as f:
        return list(csv.DictReader(f))


def test_ministry_map_file_is_complete_and_uses_the_eight_groups():
    rows = _map_rows()
    assert rows and len({r["ministry"] for r in rows}) == len(rows)
    for r in rows:
        assert r["ministry_canonical"] and r["ministry_group"] in ALLOWED_GROUPS, r
        assert r["ministry"] == clean.norm_ministry(r["ministry"])  # keys are in clean.py's normalised form
    assert {r["ministry_group"] for r in rows} == ALLOWED_GROUPS


def test_ministry_variants_and_typos_collapse():
    f = clean.ministry_info
    for raw in ("WATER RESOURCES, RIVER DEVELOPMENT AND GANAGA REJUVENATION", "WATER RESOURCES  "):
        assert f(raw) == ("Water Resources, River Development and Ganga Rejuvenation", "energy_environment_resources")
    assert f("MICRO,SMALL AND MEDIUM ENTERPRISES") == f("MICRO, SMALL AND MEDIUM ENTERPRISES") == \
        ("Micro, Small and Medium Enterprises", "industry_commerce_labour")
    assert f("COMMERCE & INDUSTRY")[0] == f("COMMERCE AND INDUSTRY  ")[0] == "Commerce and Industry"
    assert f("Education") == f("EDUCATION") == ("Education", "social_welfare_health_education")
    assert f("PERSONNEL,PUBLIC GRIEVANCES AND PENSIONS") == f("PERSONNEL, PUBLIC GRIEVANCES AND PENSIONS")
    assert f("URBAN  DEVELOPMENT AND POVERTY  ALLEVATION")[0] == "Urban Development and Poverty Alleviation"
    assert f("LAW , JUSTICE AND COMPANY AFFAIRS")[0] == "Law, Justice and Company Affairs"
    assert f("AYURVEDA,YOGA & NATUROPATHY,UNANI,SIDDHA AND HOMEOPATHY (AYUSH)")[0] == f("AYUSH")[0] == "AYUSH"
    assert f("COMMUNICATION")[0] == f("COMMUNICATIONS")[0]
    assert f("SMALL SCALE INDUSTRIES AND RURAL AND AGRO INDUSTR ")[0] == "Small Scale Industries and Agro and Rural Industries"


def test_ministry_renames_collapse_but_mergers_stay_separate():
    f = clean.ministry_info
    assert f("HUMAN RESOURCE DEVELOPMENT")[0] == f("EDUCATION")[0] == "Education"
    assert f("ENVIRONMENT AND FORESTS")[0] == f("ENVIRONMENT, FOREST AND CLIMATE CHANGE")[0] == \
        f("ENVIRONMENT,  FORESTS AND CLIMATE CHANGE")[0] == "Environment, Forest and Climate Change"
    assert f("NON-CONVENTIONAL ENERGY SOURCES")[0] == f("NEW AND RENEWABLE ENERGY")[0]
    assert f("COMPANY AFFAIRS")[0] == f("CORPORATE AFFAIRS")[0]
    assert f("LABOUR")[0] == f("LABOUR AND EMPLOYMENT")[0]
    assert f("SHIPPING")[0] == f("PORTS, SHIPPING AND WATERWAYS")[0]
    assert f("AGRICULTURE")[0] == f("AGRICULTURE AND FARMERS WELFARE")[0]
    # mergers / splits are NOT collapsed
    assert len({f(x)[0] for x in ("JAL SHAKTI", "WATER RESOURCES", "DRINKING WATER AND SANITATION")}) == 3
    assert len({f(x)[0] for x in ("COMMUNICATIONS", "COMMUNICATIONS AND INFORMATION TECHNOLOGY",
                                  "ELECTRONICS AND INFORMATION TECHNOLOGY")}) == 3
    assert len({f(x)[0] for x in ("URBAN DEVELOPMENT", "HOUSING AND URBAN POVERTY ALLEVIATION",
                                  "HOUSING AND URBAN AFFAIRS")}) == 3
    assert f("SHIPPING, ROAD TRANSPORT AND HIGHWAYS")[0] != f("ROAD TRANSPORT AND HIGHWAYS")[0]


def test_ministry_groups_security_and_lineage_columns():
    f = clean.ministry_info
    for m in ("HOME AFFAIRS", "DEFENCE", "EXTERNAL AFFAIRS"):
        assert f(m)[1] == "security"
    assert f("FINANCE")[1] == "economy_finance" and f("RAILWAYS")[1] == "infrastructure_transport"
    assert f("HEALTH AND FAMILY WELFARE")[1] == "social_welfare_health_education"
    assert f("RURAL DEVELOPMENT")[1] == "agriculture_rural" and f("POWER")[1] == "energy_environment_resources"
    assert f("TEXTILES")[1] == "industry_commerce_labour" and f("LAW AND JUSTICE")[1] == "governance_law_other"
    rows = {r["ministry"]: r for r in _map_rows()}
    assert rows["HUMAN RESOURCE DEVELOPMENT"]["predecessor_of"] == "Education"
    assert rows["EDUCATION"]["predecessor_of"] == ""
    assert rows["ENVIRONMENT AND FORESTS"]["predecessor_of"] == "Environment, Forest and Climate Change"
    assert "2014" in rows["ENVIRONMENT AND FORESTS"]["notes"]


def test_unmapped_ministry_is_reported_not_guessed():
    before = clean.UNMAPPED_MINISTRIES["MINISTRY OF TESTING"]
    assert clean.ministry_info("MINISTRY OF TESTING") == (None, None)
    assert clean.UNMAPPED_MINISTRIES["MINISTRY OF TESTING"] == before + 1
    q = mk_q(ministry="MINISTRY OF TESTING", q="(a) x?", a="THE MINISTER OF X (SHRI A B)<br>(a) Yes.")
    r = clean.build_record(17, q)
    assert r["ministry_canonical"] is None and r["ministry_group"] is None
    clean.UNMAPPED_MINISTRIES.pop("MINISTRY OF TESTING", None)


def test_every_ministry_in_the_built_table_is_mapped():
    path = os.path.join(clean.CLEAN_DIR, "questions.parquet")
    if not os.path.exists(path):
        return
    import pyarrow.parquet as pq
    t = pq.read_table(path, columns=["ministry_raw"]).column("ministry_raw").to_pylist()
    bad = {m for m in set(t) if clean.ministry_info(m) == (None, None)}
    clean.UNMAPPED_MINISTRIES.clear()
    assert not bad, bad


def test_schema_has_new_columns():
    names = [f.name for f in clean.schema()]
    for c in ("needs_ocr", "fallback_file", "ministry_canonical", "ministry_group"):
        assert c in names
    r = clean.build_record(17, mk_q(q="(a) x?", a="THE MINISTER OF X (SHRI A B)<br>(a) Yes."))
    assert set(r) == set(names)
