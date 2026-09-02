"""coalition_lookup.py -- classify an MP's party relative to the Union government on a given date.

    role(party_name, date, mp=None, pc=None) -> "government" | "outside_support" | "opposition"

* party_name : ECI party name as in data/ref/source/eci0414.csv / eci1924.csv (whitespace, case,
               punctuation and a few known spelling variants are normalised).
* date       : "YYYY-MM-DD" string, datetime.date or datetime.datetime (the question date).
* mp / pc    : optional MP name / constituency (ECI spelling). Needed ONLY for parties that split
               while in office (Shiv Sena 2022, NCP 2023, LJP 2021, TMC-rebels 2026); see FACTIONS
               and coalitions_notes.md. Without them a split party on a date inside its split
               window returns the party-level answer (see needs_manual()).

Data: coalitions.csv next to this file. Intervals are inclusive [start_date, end_date]; empty end_date =
open (as of 2026-10-17). Precedence if several rows cover a date: government > outside_support >
opposition (rows with role opposition_formally_noted count as opposition). Parties not listed ->
"opposition". Rows with confidence "low" are included by default; pass min_conf="medium" or "high" to
ignore weaker rows (they then fall back to "opposition").

Run `python -I coalition_lookup.py` to execute the self-test.
"""
import csv
import datetime as _dt
import os
import re

_HERE = os.path.dirname(os.path.abspath(__file__))
_CONF_RANK = {"low": 0, "medium": 1, "high": 2}
_ROLE_RANK = {"government": 2, "outside_support": 1, "opposition_formally_noted": 0}

# normalised-name aliases (ECI spelling drift / renames) -> key used in coalitions.csv
_ALIASES = {
    "indianunionmuslimleague": "muslimleaguekeralastatecommittee",
    "nagalandpeoplesfront": "nagapeoplesfront",
    "nagapeoplesfront": "nagapeoplesfront",
    "apnadal": "apnadal(soneylal)",
    "lokjanshaktiparty": "lokjanshaktiparty",
    "janatadalunited": "janatadalunited",
    "shivsena": "shivsena",           # 'Shivsena' (2004-19) and 'Shiv Sena' (2024) share one key; dates separate them
}


def _norm(name):
    s = str(name).lower()
    s = re.sub(r"[^a-z0-9()]+", "", s)      # drops spaces, apostrophes, commas, mojibake characters
    return _ALIASES.get(s, s)


def _date(d):
    if isinstance(d, _dt.datetime):
        return d.date()
    if isinstance(d, _dt.date):
        return d
    return _dt.date.fromisoformat(str(d)[:10])


def _load():
    rows = []
    with open(os.path.join(_HERE, "coalitions.csv"), newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(dict(
                key=_norm(r["party_name"]),
                role=r["role"],
                start=_date(r["start_date"]),
                end=_date(r["end_date"]) if r["end_date"] else _dt.date.max,
                conf=_CONF_RANK[r["confidence"]],
                label=r["party_name"],
            ))
    return rows


_ROWS = _load()

# ---------------------------------------------------------------------------------------------
# MP-level overrides for parties that split in office.  Matching: every token of one tuple in
# name_tokens must occur among the lower-cased words of `mp`, OR the constituency `pc` equals.
# Names/constituencies are the ECI spellings of eci1924.csv (2019/2024).  See coalitions_notes.md.
# ---------------------------------------------------------------------------------------------
_END17 = _dt.date(2024, 6, 8)
_FOREVER = _dt.date.max
FACTIONS = [
    # Shiv Sena (ECI 2019 'Shivsena'): Shinde side recognised in LS 2022-07-19 (12 MPs)
    dict(key="shivsena", start=_dt.date(2022, 7, 19), end=_END17, role="government",
         members=[
             (("shewale",), "Mumbai South-Central"), (("krupal", "tumane"), "Ramtek"),
             (("bhavana", "gawali"), "Yavatmal-Washim"), (("hemant", "patil"), "Hingoli"),
             (("godse", "hemant"), "Nashik"), (("rajendra", "gavit"), "Palghar"),
             (("shrikant", "shinde"), "Kalyan"), (("shrirang", "barne"), "Maval"),
             (("sadashiv", "lokhande"), "Shirdi"), (("sanjay", "mandlik"), "Kolhapur"),
             (("dhairyasheel", "mane"), "Hatkanangle"), (("prataprao", "jadhav"), "Buldhana"),
         ]),
    # 13th Shinde MP: Gajanan Kirtikar (Mumbai North-West) -- joined Nov 2022 (day UNVERIFIED, from memory)
    dict(key="shivsena", start=_dt.date(2022, 11, 11), end=_END17, role="government",
         members=[(("gajanan", "kirtikar"), "Mumbai North-West")]),
    # NCP: Ajit Pawar side = Sunil Tatkare only (Amol Kolhe returned to Sharad Pawar within days)
    dict(key="nationalistcongressparty", start=_dt.date(2023, 7, 2), end=_END17, role="government",
         members=[(("tatkare",), "Raigad")]),
    # LJP: Paras faction (5 MPs, RLJP) until Paras quit the NDA-led cabinet 2024-03-19
    dict(key="lokjanshaktiparty", start=_dt.date(2021, 6, 14), end=_dt.date(2024, 3, 19), role="government",
         members=[(("paras",), "Hajipur"), (("prince", "raj"), "Samastipur"), (("chandan", "singh"), "Nawada"),
                  (("veena", "devi"), "Vaishali"), (("mehboob", "kaiser"), "Khagaria")]),
    # LJP: Chirag Paswan (low confidence, see notes)
    dict(key="lokjanshaktiparty", start=_dt.date(2021, 6, 14), end=_END17, role="government",
         members=[(("chirag",), "Jamui")], conf=0),
    # AITC rebels (20) -> NCPI, support to NDA from 2026-06-08 (low confidence)
    dict(key="alltrinamoolcongress", start=_dt.date(2026, 6, 8), end=_FOREVER, role="outside_support", conf=0,
         members=[(("basunia",), "Coochbehar"), (("khalilur",), "Jangipur"), (("pathan",), "Baharampur"),
                  (("abu", "taher"), "Murshidabad"), (("partha", "bhowmick"), "Barrackpur"),
                  (("kakoli",), "Barasat"), (("bapi",), "Mathurapur"), (("sayani",), "Jadavpur"),
                  (("saayoni",), None), (("mala", "roy"), "Kolkata Dakshin"), (("sudip",), "Kolkata Uttar"),
                  (("prasun", "banerjee"), "Howrah"), (("rachna",), "Hooghly"), (("rachana",), None),
                  (("mitali", "bag"), "Arambag"), (("deepak", "adhikari"), "Ghatal"),
                  (("kalipada",), "Jhargram"), (("june",), "Medinipur"), (("arup", "chakraborty"), "Bankura"),
                  (("sharmila", "sarkar"), "Bardhaman Purba"), (("asit", "mal"), "Bolpur"),
                  (("satabdi",), "Birbhum")]),
]
# the 'alltrinamoolcongress' key is the normalised form of 'All India Trinamool Congress'
for _f in FACTIONS:
    if _f["key"] == "alltrinamoolcongress":
        _f["key"] = _norm("All India Trinamool Congress")

_SPLIT_WINDOWS = {}
for _f in FACTIONS:
    lo, hi = _SPLIT_WINDOWS.get(_f["key"], (_dt.date.max, _dt.date.min))
    _SPLIT_WINDOWS[_f["key"]] = (min(lo, _f["start"]), max(hi, _f["end"]))


def _tokens(s):
    return set(re.findall(r"[a-z]+", str(s).lower()))


def needs_manual(party_name, date):
    """True if party_name on `date` is inside a split window (MP-level classification required)."""
    k = _norm(party_name)
    d = _date(date)
    w = _SPLIT_WINDOWS.get(k)
    return bool(w and w[0] <= d <= w[1])


def _faction_role(key, d, mp, pc, min_conf):
    for f in FACTIONS:
        if f["key"] != key or not (f["start"] <= d <= f["end"]) or f.get("conf", 2) < min_conf:
            continue
        for toks, fpc in f["members"]:
            if mp is not None and set(toks) <= _tokens(mp):
                return f["role"]
            if pc is not None and fpc is not None and _norm(fpc) == _norm(pc):
                return f["role"]
    return None


def role(party_name, date, mp=None, pc=None, min_conf="low"):
    """Return 'government', 'outside_support' or 'opposition' for the party (or split-faction MP) on `date`."""
    mc = _CONF_RANK[min_conf]
    d = _date(date)
    key = _norm(party_name)
    if mp is not None or pc is not None:
        fr = _faction_role(key, d, mp, pc, mc)
        if fr is not None:
            return fr
    best = None
    for r in _ROWS:
        if r["key"] != key or r["conf"] < mc or not (r["start"] <= d <= r["end"]):
            continue
        # faction-label rows ("... [..]") are only reachable via FACTIONS
        if "[" in r["label"]:
            continue
        if best is None or _ROLE_RANK[r["role"]] > _ROLE_RANK[best]:
            best = r["role"]
    if best in ("government", "outside_support"):
        return best
    return "opposition"


def _selftest():
    R = role
    # NDA 1999-2004
    assert R("Bharatiya Janata Party", "2001-01-01") == "government"
    assert R("Telugu Desam", "2002-06-01") == "outside_support"
    assert R("Dravida Munnetra Kazhagam", "2003-12-20") == "government"
    assert R("Dravida Munnetra Kazhagam", "2003-12-21") == "opposition"
    assert R("All India Trinamool Congress", "2002-01-01") == "opposition"      # out Mar 2001 - Sep 2003
    assert R("All India Trinamool Congress", "2003-12-01") == "government"
    assert R("Jammu & Kashmir National Conference", "2003-07-11") == "government"
    assert R("Jammu & Kashmir National Conference", "2003-07-13") == "opposition"
    assert R("Lok Jan Shakti Party", "2002-04-28") == "government"
    assert R("Lok Jan Shakti Party", "2002-04-30") == "opposition"
    # UPA
    assert R("Indian National Congress", "2007-01-01") == "government"
    assert R("Indian National Congress", "2015-01-01") == "opposition"
    assert R("Bharatiya Janata Party", "2010-01-01") == "opposition"
    assert R("Communist Party Of India  (Marxist)", "2006-03-01") == "outside_support"   # double-space ECI variant
    assert R("Communist Party Of India (Marxist)", "2008-07-09") == "opposition"
    assert R("Bahujan Samaj Party", "2005-01-01") == "outside_support"
    assert R("Bahujan Samaj Party", "2008-07-01") == "opposition"
    assert R("Samajwadi Party", "2011-06-01") == "outside_support"
    assert R("Telangana Rashtra Samithi", "2006-01-01") == "government"
    assert R("Telangana Rashtra Samithi", "2006-10-01") == "opposition"
    assert R("All India Trinamool Congress", "2012-09-20") == "government"
    assert R("All India Trinamool Congress", "2012-12-01") == "opposition"
    assert R("Dravida Munnetra Kazhagam", "2013-03-19") == "government"
    assert R("Dravida Munnetra Kazhagam", "2013-03-20") == "opposition"
    assert R("Rashtriya Lok Dal", "2010-01-01") == "opposition"
    assert R("Rashtriya Lok Dal", "2013-01-01") == "government"
    assert R("Indian Union Muslim League", "2012-01-01") == "government"          # alias of 'Muslim League Kerala State Committee'
    # NDA 2014-
    assert R("Shivsena", "2016-01-01") == "government"
    assert R("Shivsena", "2020-01-01") == "opposition"
    assert R("Shivsena", "2022-08-01") == "opposition"                              # split window, MP unknown
    assert needs_manual("Shivsena", "2022-08-01") and not needs_manual("Shivsena", "2016-01-01")
    assert R("Shivsena", "2022-08-01", mp="Rahul Ramesh Shewale") == "government"
    assert R("Shivsena", "2022-08-01", mp="Arvind Ganpat Sawant") == "opposition"
    assert R("Shiv Sena", "2024-07-01") == "government"
    assert R("Shiv Sena (Uddhav Balasaheb Thackrey)", "2024-07-01") == "opposition"
    assert R("Telugu Desam", "2018-03-15") == "government" and R("Telugu Desam", "2018-03-17") == "opposition"
    assert R("Telugu Desam", "2021-01-01") == "opposition" and R("Telugu Desam", "2024-07-01") == "government"
    assert R("Janata Dal  (United)", "2015-01-01") == "opposition"
    assert R("Janata Dal (United)", "2018-01-01") == "government"
    assert R("Janata Dal  (United)", "2022-09-01") == "opposition"
    assert R("Janata Dal (United)", "2024-02-01") == "government"
    assert R("Shiromani Akali Dal", "2020-09-25") == "government" and R("Shiromani Akali Dal", "2020-09-27") == "opposition"
    assert R("All India Anna Dravida Munnetra Kazhagam", "2016-01-01") == "opposition"
    assert R("All India Anna Dravida Munnetra Kazhagam", "2020-01-01") == "government"
    assert R("All India Anna Dravida Munnetra Kazhagam", "2023-10-01") == "opposition"
    assert R("Apna Dal", "2015-01-01") == "government"                              # ECI 2014 label
    assert R("Nationalist Congress Party", "2020-01-01") == "opposition"
    assert R("Nationalist Congress Party", "2023-09-01", mp="Sunil Dattatrey Tatkare") == "government"
    assert R("Nationalist Congress Party", "2023-09-01", mp="Supriya Sule") == "opposition"
    assert R("Nationalist Congress Party", "2024-07-01") == "government"            # ECI 2024 NCP = Ajit Pawar's
    assert R("Nationalist Congress Party � Sharadchandra Pawar", "2024-07-01") == "opposition"
    assert R("Lok Jan Shakti Party", "2021-09-01", mp="Pashu Pati Kumar Paras") == "government"
    assert R("Lok Janshakti Party(Ram Vilas)", "2024-07-01") == "government"
    assert R("Janata Dal (Secular)", "2022-01-01") == "opposition" and R("Janata Dal (Secular)", "2024-07-01") == "government"
    assert R("All India Trinamool Congress", "2026-07-01", mp="Dr. Kakoli Ghosh Dastidar") == "outside_support"
    assert R("All India Trinamool Congress", "2026-07-01", mp="Mahua Moitra") == "opposition"
    assert R("Some Unknown Party", "2010-01-01") == "opposition"
    assert R("Telugu Desam", "2002-06-01", min_conf="high") == "opposition"           # medium-confidence row ignored
    import inspect
    n = sum(1 for ln in inspect.getsource(_selftest).splitlines() if ln.strip().startswith("assert"))
    print("coalition_lookup self-test OK (%d assert statements)" % n)


if __name__ == "__main__":
    _selftest()
