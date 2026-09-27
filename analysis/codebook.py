"""Codebook v1.2 — single source of truth for labels used by human coders (M3), Gemini (M2) and the analysis.

Unit of coding: one QUESTION PART (a), (b), ... paired with the answer text that addresses it
(clean.align_parts(); an answer covering "(a) to (c)" is the answer text for each of a, b, c).
Questions with no parsed parts are coded as a single part labelled "_all".

Version history:
  v1   (2026-10-17) drafted before any human coding.
  v1.1 (2026-10-17) after the 50-question pilot (194 parts, 2 coders; label kappa 0.84, ordinal alpha 0.88) and
       the adjudication of its 30 disagreements: sharper request types, PROCEDURAL, rules R8–R17, evasion
       precedence. Snapshot: data/coding/codebook_v1.1_frozen_snapshot.py.
  v1.2 (2026-10-17) after the coders' written pilot feedback, BEFORE any main-sample or Gemini label existed:
       new evasion type INCOMPLETE (both coders independently reported that ~27 OTHER codes each were really
       "a requested breakdown/period/element is missing"); PROCEDURAL renamed WITHHELD (same meaning, clearer);
       R1 generalised (both directions + failed preconditions); explicit premise-denial rule; R5 no longer
       penalises fully answered parts that open with "State subject"; bare "As annexed"; not-maintained + proxy
       data; implied answers to "whether" parts; deterministic request-type tie-break; worked examples on the
       FULL/GENERIC line; per-part evasion judged only on the text addressing that part; mis-split cells.
       Rules renumbered R1–R18. FROZEN for the main coding and the Gemini runs.
Human-readable version with examples: CODEBOOK.md (same folder). Keep the two in sync.
"""

CODEBOOK_VERSION = "v1.2"

# ---------- primary label: did the answer provide what this part asked for? ----------
RESPONSIVENESS = {
    "FULL": "The answer provides the specific information, reason or commitment the part asks for "
            "(e.g. the requested numbers, the list, a clear yes/no with the asked-for details, or a clear factual "
            "denial of the premise). A short answer is FULL if it gives the specific items asked for. An annexure "
            "counts if it is presented as containing the requested data.",
    "PARTIAL": "The answer addresses the part but leaves out a requested element (a breakdown, a year, one of "
               "several items) or substitutes a related but different scope/quantity while also giving some "
               "requested content.",
    "NONE": "The answer does not provide what was asked: it defers, deflects, says the data is not maintained, "
            "withholds it, answers something else, or gives only generic policy description without any of the "
            "requested specifics.",
}
LABELS = list(RESPONSIVENESS)
LABEL_SCORE = {"FULL": 1.0, "PARTIAL": 0.5, "NONE": 0.0}  # part-level score used in analysis

# ---------- secondary: how the part was not (fully) answered; only for PARTIAL / NONE ----------
EVASION = {
    "DEFERRAL": "Information 'is being collected and will be laid on the Table of the House' or similar.",
    "NOT_MAINTAINED": "Data 'not maintained centrally' / 'no such data is available', where such data could exist.",
    "JURISDICTION": "Deflects to another authority ('State subject', 'matter of the State Government', another "
                    "ministry or body) INSTEAD of giving some or all of the requested content.",
    "WITHHELD": "Explicitly declines to give the substance: 'not in public interest to divulge', confidential, "
                "sub judice, report still in draft / not yet tabled, under examination or consideration.",
    "DOES_NOT_ARISE": "'Does not arise' / 'not applicable' where the part does follow (see R1).",
    "DIFFERENT_QUESTION": "Substitutes a related but DIFFERENT quantity, scope, period or subject for the one asked "
                          "(spending instead of allocation; last year instead of this year; all patients instead of "
                          "those outside the scheme).",
    "INCOMPLETE": "Gives the requested content but omits a requested element: a breakdown (state-, company-, "
                  "year-wise), a period, a date for each item, one of several bundled items, or the details after "
                  "a count.",
    "ANNEXURE_ONLY": "Refers to an annexure/statement that is vague or clearly does not match the request.",
    "GENERIC": "Describes schemes or policy in general terms without any of the specifics requested.",
    "OTHER": "Any other way of not providing the requested information (describe in note).",
}
EVASION_TYPES = list(EVASION)
EVASION_CHOICES = EVASION_TYPES + ["NONE_APPLICABLE"]  # full enum incl. the value used for FULL parts
# when several evasion types fit, choose the FIRST in this order
EVASION_PRECEDENCE = ["DEFERRAL", "NOT_MAINTAINED", "JURISDICTION", "WITHHELD", "DOES_NOT_ARISE",
                      "DIFFERENT_QUESTION", "INCOMPLETE", "ANNEXURE_ONLY", "GENERIC", "OTHER"]
assert sorted(EVASION_PRECEDENCE) == sorted(EVASION_TYPES)
# codes renamed between versions (used to migrate pilot labels coded under older versions)
EVASION_RENAMES = {"PROCEDURAL": "WITHHELD"}
ANNEXURE_CHARS = 2000  # annexure text shown to coders AND to Gemini is truncated identically

DOES_NOT_ARISE_RULE = (
    "R1 'Does not arise' is FULL when the part's condition is not met by the answer to an earlier part ('if so…' "
    "after a No; 'if not…' after a Yes), or when the answer shows that the part's precondition does not hold (e.g. "
    "exploration has not begun, so quantity or viability cannot be known). Otherwise code NONE with DOES_NOT_ARISE.")

# ---------- question-part request type (Ojha & Mishra 2010 distinction, extended) ----------
REQUEST_TYPE = {
    "INFORMATION": "asks for facts, figures, lists, incidents or the status of things in the world "
                   "(e.g. numbers, whether something happened, whether data exists)",
    "REASON": "asks why / the reasons therefor",
    "ACTION": "asks about the GOVERNMENT'S OWN decisions, schemes, steps or measures — taken, ongoing or proposed "
              "(e.g. 'whether the Government has decided/proposes to', 'steps taken', 'measures to', "
              "'action taken thereon', 'whether the Government has set up/launched')",
    "OPINION": "asks for the government's view or assessment",
}
REQUEST_TYPES = list(REQUEST_TYPE)
REQUEST_TYPE_PRECEDENCE = ["REASON", "ACTION", "OPINION", "INFORMATION"]
REQUEST_TYPE_RULE = ("R17 Request type of mixed parts: if a part asks for more than one kind of thing, choose the "
                     "first that applies in the order " + " > ".join(REQUEST_TYPE_PRECEDENCE) + " (e.g. 'details "
                     "and the reasons therefor' = REASON; 'steps taken and the number of prosecutions' = ACTION). "
                     "'If so, the details thereof' takes the request type of the part it refers to. 'The "
                     "Government' includes ministries, departments, attached offices, public-sector undertakings "
                     "and statutory or regulatory bodies (e.g. CIL, NSDC, BIS, RBI).")

# Mapping to Poljak (2024, J. Legislative Studies) three-way reply typology for comparison.
# TODO(verify against Poljak's published codebook before the paper is written).
POLJAK_MAP = {"FULL": "reply", "PARTIAL": "partial reply", "NONE": "non-reply"}

# ---------- decision rules (shown to coders and to Gemini) ----------
RULES = [
    DOES_NOT_ARISE_RULE,
    "R2 Judge each part only against the text that addresses it. In a shared answer ('(a) to (c)'), a part may be "
    "FULL while another is not; paragraphs that clearly address OTHER parts are ignored for this part, including "
    "when choosing the evasion type.",
    "R3 Annexures: 'details are at Annexure-I' or a bare 'As annexed' in reply to a data request is FULL (the "
    "annexure is presented as the answer, even if you cannot see it). ANNEXURE_ONLY only when the reference is "
    "vague or clearly does not match the request.",
    "R4 'Not maintained' plus substitute data: if the substitute measures the SAME quantity from another source "
    "(e.g. industry-reported consumption when consumption is 'not maintained'), PARTIAL with NOT_MAINTAINED; if it "
    "is a DIFFERENT quantity (e.g. BPL population when unemployed BPL youth were asked), NONE with NOT_MAINTAINED.",
    "R5 A 'State subject' / other-authority opener does not lower the label when the requested content is then "
    "fully given (FULL). Use JURISDICTION only when the deflection replaces requested content: PARTIAL if some "
    "requested content is given, NONE for a bare 'State subject'.",
    "R6 If the answer text is empty or missing, every part is NONE with OTHER.",
    "R7 Ignore the ministry header, the minister's name and the sentence 'A statement is laid on the Table'. Judge "
    "only whether the requested content is present, not whether it is true or good policy.",
    "R8 Brevity is not evasion. FULL: a short answer naming the specific measures, numbers or decision asked for "
    "(e.g. 'recruitment rallies and media campaigns' for measures to attract applicants). GENERIC: an action or "
    "policy stated without the specifics asked for (e.g. 'the matter has been taken up with the concerned "
    "authorities' — no who, what or outcome).",
    "R9 Substitution: giving a related but DIFFERENT quantity, scope or period is DIFFERENT_QUESTION — PARTIAL if "
    "some requested content is also given, else NONE. Giving the SAME quantity at a coarser breakdown or for fewer "
    "periods is INCOMPLETE (R14), not substitution.",
    "R10 Premise: a clear factual denial or correction that answers the part is FULL (e.g. 'no repeated breakdowns "
    "occurred'; '15 washeries, not 20'). A denial that concedes a related issue and leaves the requested item "
    "unaddressed is PARTIAL with DIFFERENT_QUESTION.",
    "R11 If the requested data or timeline cannot exist yet (season or survey in progress, required approval not "
    "yet given) and the answer says why, FULL. NOT_MAINTAINED is only for data that could exist but is not kept.",
    "R12 'Whether …' parts: a description that clearly entails the answer counts as answering (e.g. details of a "
    "sanctioned scheme answer 'whether the Government has decided to set it up' → FULL). Only when the description "
    "entails neither yes nor no is the part at most PARTIAL: PARTIAL with GENERIC if related policy is described, "
    "NONE if nothing relevant is said. For 'whether … proposes a (new) scheme' answered only with existing schemes: "
    "PARTIAL with GENERIC (this rule takes precedence over R9).",
    "R13 Garbled, scrambled or mis-split text: code what is legible; if the part's answer cell is mis-split, judge "
    "against the full answer; write 'garbled' or 'split' in the note.",
    "R14 Incomplete: the requested content is given but a requested element is missing (a state-, company- or "
    "year-wise breakdown, a period, a date for each item, one of several bundled items, details after a count): "
    "PARTIAL with INCOMPLETE.",
    "R15 Withheld: an explicit refusal ('not in public interest', confidential, sub judice, draft report not "
    "tabled) is WITHHELD — PARTIAL if relevant facts are acknowledged, NONE otherwise.",
    "R16 When several evasion types fit, use the first in this order: " + " > ".join(EVASION_PRECEDENCE) + ".",
    REQUEST_TYPE_RULE,
    "R18 Do not use OTHER if any listed evasion type fits; when you use OTHER, explain in the note.",
]

# ---------- JSON schema for one coded part (Gemini structured output and human sheet export) ----------
RATIONALE_MAX_WORDS = 15
PART_SCHEMA = {
    "type": "object",
    "properties": {
        "part_label": {"type": "string"},
        "request_type": {"type": "string", "enum": REQUEST_TYPES},
        # rationale comes BEFORE the label so the model states the decisive evidence first (cheap reasoning)
        "rationale": {"type": "string",
                      "description": f"at most {RATIONALE_MAX_WORDS} words: the decisive phrase from the answer"},
        "label": {"type": "string", "enum": LABELS},
        "evasion_type": {"type": "string", "enum": EVASION_TYPES + ["NONE_APPLICABLE"]},
    },
    "required": ["part_label", "request_type", "rationale", "label", "evasion_type"],
    "propertyOrdering": ["part_label", "request_type", "rationale", "label", "evasion_type"],
}
RESPONSE_SCHEMA = {"type": "object", "properties": {"parts": {"type": "array", "items": PART_SCHEMA}},
                   "required": ["parts"]}


def instructions_text():
    """Codebook rendered as plain text (used verbatim in the Gemini prompt and printed on the coding sheet)."""
    lines = [f"CODEBOOK {CODEBOOK_VERSION}: responsiveness of written ministerial answers (Lok Sabha).",
             "For EACH question part, judge whether the answer text provides what that part asks for.", "",
             "LABEL (choose one):"]
    lines += [f"- {k}: {v}" for k, v in RESPONSIVENESS.items()]
    lines += ["", "EVASION TYPE (only if PARTIAL or NONE; else NONE_APPLICABLE):"]
    lines += [f"- {k}: {v}" for k, v in EVASION.items()]
    lines += ["", "REQUEST TYPE of the part:"] + [f"- {k}: {v}" for k, v in REQUEST_TYPE.items()]
    lines += ["", "RULES:"] + [f"- {r}" for r in RULES]
    return "\n".join(lines)
