"""Codebook v1.1 — single source of truth for labels used by human coders (M3), Gemini (M2) and the analysis.

Unit of coding: one QUESTION PART (a), (b), ... paired with the answer text that addresses it
(clean.align_parts(); an answer covering "(a) to (c)" is the answer text for each of a, b, c).
Questions with no parsed parts are coded as a single part labelled "_all".

Version history:
  v1   (2026-10-17) drafted before any human coding.
  v1.1 (2026-10-17) after the 50-question pilot (194 parts, 2 coders; label kappa 0.84, ordinal alpha 0.88).
       Changes, each motivated by pilot disagreements: request-type definitions sharpened (INFORMATION vs
       ACTION was the most frequent request-type disagreement); new evasion type PROCEDURAL; rules R8–R13
       (brevity vs genericness, substitution, premise correction, not-yet-available data, unanswered yes/no,
       garbled text); evasion-type precedence order to resolve GENERIC vs OTHER vs DIFFERENT_QUESTION.
       FROZEN for the main coding and the Gemini runs once the pilot adjudication is complete.
Human-readable version with examples: CODEBOOK.md (same folder). Keep the two in sync.
"""

CODEBOOK_VERSION = "v1.1"

# ---------- primary label: did the answer provide what this part asked for? ----------
RESPONSIVENESS = {
    "FULL": "The answer provides the specific information, reason or commitment the part asks for "
            "(e.g. the requested numbers, the list, a clear yes/no with the asked-for details). A short answer "
            "is FULL if it gives the specific items asked for. An annexure counts only if it is described as "
            "containing the requested data.",
    "PARTIAL": "The answer addresses the part but omits a substantive element of the request, or substitutes a "
               "related but different scope/quantity: e.g. national figures when state-wise were asked, spending "
               "when allocation was asked, existing schemes when new ones were asked, some years but not all, "
               "'yes' without the asked-for details, or one of two bundled questions answered.",
    "NONE": "The answer does not provide what was asked: it defers, deflects, says the data is not "
            "maintained, withholds it on procedural grounds, answers something else, or gives only generic "
            "policy description without any of the requested specifics.",
}
LABELS = list(RESPONSIVENESS)
LABEL_SCORE = {"FULL": 1.0, "PARTIAL": 0.5, "NONE": 0.0}  # part-level score used in analysis

# ---------- secondary: how the part was not (fully) answered; only for PARTIAL / NONE ----------
EVASION = {
    "DEFERRAL": "Information 'is being collected and will be laid on the Table of the House' or similar.",
    "NOT_MAINTAINED": "Data 'not maintained centrally' / 'no such data is available', where such data could exist.",
    "JURISDICTION": "Deflects to another authority: 'State subject', 'matter of the State Government', "
                    "other ministry or body.",
    "PROCEDURAL": "Withholds the substance on procedural grounds: report not yet tabled, under examination or "
                  "consideration, matter sub judice, confidential / not in public interest.",
    "DOES_NOT_ARISE": "'Does not arise' / 'not applicable' for a part that does follow from the previous parts.",
    "DIFFERENT_QUESTION": "Answers a related but different question than the one asked (different scope, "
                          "quantity or subject is the main move).",
    "ANNEXURE_ONLY": "Refers to an annexure/statement whose content is not present or does not contain the "
                     "requested information.",
    "GENERIC": "Describes schemes or policy in general terms without the specifics requested.",
    "OTHER": "Any other way of not providing the requested information (describe in note).",
}
EVASION_TYPES = list(EVASION)
EVASION_CHOICES = EVASION_TYPES + ["NONE_APPLICABLE"]  # full enum incl. the value used for FULL parts
# when several evasion types fit, choose the FIRST in this order (pilot: GENERIC/OTHER/DIFFERENT_QUESTION confusion)
EVASION_PRECEDENCE = ["DEFERRAL", "NOT_MAINTAINED", "JURISDICTION", "PROCEDURAL", "DOES_NOT_ARISE",
                      "DIFFERENT_QUESTION", "ANNEXURE_ONLY", "GENERIC", "OTHER"]
assert sorted(EVASION_PRECEDENCE) == sorted(EVASION_TYPES)
ANNEXURE_CHARS = 2000  # annexure text shown to coders AND to Gemini is truncated identically

# Legitimate "does not arise": if an earlier part was answered "No" and this part asks "if so, details",
# then "does not arise" is FULL (the question was answered), not evasion.
DOES_NOT_ARISE_RULE = ("If a previous part was answered in the negative and this part is conditional on a "
                       "positive answer ('if so, the details thereof'), 'does not arise' is coded FULL.")

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
REQUEST_TYPE_RULE = ("'If so, the details thereof' takes the request type of the part it refers to. A part that "
                     "asks both why and what: use the dominant request; if equal, REASON.")

# Mapping to Poljak (2024, J. Legislative Studies) three-way reply typology for comparison.
# TODO(verify against Poljak's published codebook before the paper is written).
POLJAK_MAP = {"FULL": "reply", "PARTIAL": "partial reply", "NONE": "non-reply"}

# ---------- decision rules (shown to coders and to Gemini) ----------
RULES = [
    DOES_NOT_ARISE_RULE,
    "Judge only whether the requested content is present, not whether it is true or adequate policy.",
    "An answer covering several parts jointly ('(a) to (c)') may be FULL for some of those parts and not others; "
    "judge each part separately against the shared text.",
    "Ignore the ministry header, the minister's name and the sentence 'A statement is laid on the Table'.",
    "If the answer text is empty or missing, every part is NONE with evasion_type OTHER.",
    "Annexures: 'details are at Annexure-I' is FULL if the annexure is described as holding the requested data; "
    "ANNEXURE_ONLY only if the reference is vague or clearly does not match the request.",
    "Partial data plus a non-maintenance statement (e.g. production given, consumption 'not maintained') is PARTIAL.",
    "Jurisdiction plus some requested facts is PARTIAL (JURISDICTION); a bare 'State subject' is NONE.",
    # v1.1 rules from the pilot
    "R8 Brevity is not evasion: a short answer naming the specific measures, numbers or decision asked for is FULL; "
    "GENERIC applies only when no specifics of the requested kind are given.",
    "R9 Substitution: giving a related but different scope or quantity (national vs state-wise, spending vs "
    "allocation, existing vs proposed schemes) is PARTIAL if some requested content is also given, else NONE.",
    "R10 Correcting a factual premise with accurate specifics (e.g. '15 washeries, not 20') is FULL.",
    "R11 If the requested data cannot exist yet (season or survey still in progress) and the answer says so, FULL; "
    "NOT_MAINTAINED is only for data that could exist but is not kept.",
    "R12 A 'whether …' part that gets no yes/no and no direct statement either way is at most PARTIAL; if only "
    "related policy is described, PARTIAL with GENERIC, or NONE if nothing relevant is said.",
    "R13 Garbled or scrambled text: code what is legible and write 'garbled' in the note.",
    # added during the pilot adjudication (gaps found by the coders)
    "R14 Incomplete specifics: when some of the requested specifics are given but others are missing (e.g. a count "
    "without the asked-for details, one state of two) and no other evasion type applies, code PARTIAL with OTHER; "
    "GENERIC is reserved for answers that give no requested specifics at all.",
    "R15 'Whether … proposes/has any scheme' answered only with a list of EXISTING schemes: apply R12 (no yes/no, "
    "related policy) → PARTIAL with GENERIC; R12 takes precedence over R9 for 'whether' parts.",
    "R16 'The Government' includes ministries, departments, attached offices, public-sector undertakings and "
    "statutory or regulatory bodies (e.g. CIL, NSDC, BIS, RBI) when deciding whether a part is ACTION.",
    "R17 A part bundling facts with 'the action taken by the Government thereon' is ACTION.",
    "When several evasion types fit, use the first in this order: " + " > ".join(EVASION_PRECEDENCE) + ".",
    REQUEST_TYPE_RULE,
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
