# Codebook v1.2: Responsiveness of written ministerial answers (Lok Sabha, 1999–2026)
*Frozen 2026-10-17, before any main-sample or Gemini label existed. The machine-readable version is `codebook.py`; keep the two in sync.*

**History:**
- **v1** was drafted before any coding.
- **v1.1** came out of the 50-question pilot (194 parts, two coders; label κ 0.84, ordinal α 0.88) and the adjudication of its 30 disagreements.
- **v1.2** adds the coders' written pilot feedback:
  - a new `INCOMPLETE` code (both coders independently found OTHER overused for missing breakdowns);
  - `PROCEDURAL` renamed to `WITHHELD`;
  - clearer rules for "does not arise", premise denials, jurisdiction openers, annexures, substitute data, implied answers and mixed request types.

Earlier versions are kept in `data/coding/` for the audit trail.

## 1. What you code
- **Unit:** one **question part**, for example "(a)", together with the answer text that addresses it.
  - Where the answer groups parts, as in "(a) to (c): …", that text is the answer for **each** of those parts.
  - A question with no parts is coded as one unit, `_all`.
- **The question you ask yourself:** *Does the answer provide what this part asks for?*
  - This is about **responsiveness**: is the requested content there?
  - It is **not** about whether the answer is true, whether the policy is good, or whether the tone is polite.
- **You are blind** to who asked the question. Don't look it up.

## 2. Labels (choose one per part)
| Label | Use when | Score |
|---|---|---|
| **FULL** | The specific information, reason or commitment asked for is there: the numbers, the list, a clear yes/no plus the requested details, or a **clear factual denial of the premise**. A short answer is FULL if it gives the specific items asked for. An annexure counts if it is presented as the answer | 1 |
| **PARTIAL** | The part is addressed, but a **requested element is missing** (a breakdown, a year, one of several items), **or** a different scope or quantity is substituted while some requested content is still given | 0.5 |
| **NONE** | None of the requested content is there: the answer defers, deflects, says the data isn't maintained, withholds it, answers something else, or gives only general description | 0 |

## 3. Evasion type (only for PARTIAL or NONE; otherwise `NONE_APPLICABLE`)
| Code | Use when | Typical wording |
|---|---|---|
| `DEFERRAL` | Promises the information later | "Information is being collected and will be laid on the Table of the House" |
| `NOT_MAINTAINED` | Says data that *could* exist isn't kept | "Not maintained centrally"; "no such data is available" |
| `JURISDICTION` | Deflection to another authority **replaces** requested content | "Health is a State subject…" |
| `WITHHELD` | Explicitly declines to give the substance | "Not in public interest to divulge"; confidential; sub judice; "the CAG report is still a draft" |
| `DOES_NOT_ARISE` | "Does not arise" where the part **does** follow (see R1) | "Does not arise." |
| `DIFFERENT_QUESTION` | A **different** quantity, scope, period or subject is substituted | Spending instead of allocation; last year instead of this year; all patients instead of those outside the scheme |
| `INCOMPLETE` *(new)* | The **same** content is given, but a requested element is missing | National total instead of state-wise; one year when two were asked; no company-wise split; one date for all instead of each; a count with no details |
| `ANNEXURE_ONLY` | The annexure reference is vague, or clearly doesn't match the request | — |
| `GENERIC` | Policy or schemes described with **no** requested specifics at all | "The matter has been taken up with the concerned authorities" |
| `OTHER` | Only if nothing above fits. Explain in the note | — |

**If several codes fit, use the first in this order:**
DEFERRAL > NOT_MAINTAINED > JURISDICTION > WITHHELD > DOES_NOT_ARISE > DIFFERENT_QUESTION > INCOMPLETE > ANNEXURE_ONLY > GENERIC > OTHER.

**The key distinction:**
- **INCOMPLETE:** they gave *what* you asked for, but not *all of it*.
- **DIFFERENT_QUESTION:** they gave *something else* instead.

## 4. Request type (for every part)
| Code | The part asks… |
|---|---|
| `INFORMATION` | for facts, figures, lists, incidents, or the status of things in the world: numbers, whether something happened, whether data exists |
| `REASON` | why ("the reasons therefor") |
| `ACTION` | about the **government's own** decisions, schemes, steps or measures, whether taken, ongoing or proposed. For example: "whether the Government has decided / proposes to…", "steps taken", "measures to…", "action taken thereon" |
| `OPINION` | for the government's view or assessment |

- **Mixed parts:** choose the first that applies, **REASON > ACTION > OPINION > INFORMATION**.
  - "Details and the reasons therefor" = REASON.
  - "Steps taken and the number of prosecutions" = ACTION.
- "If so, the details thereof" takes the type of the part it refers to.
- **"The Government"** includes ministries, departments, attached offices, PSUs, and statutory or regulatory bodies (for example CIL, NSDC, BIS, RBI).

## 5. Rules
- **R1, "does not arise".** It is **FULL** when the part's condition fails because of an earlier answer: "if so…" after a **No**, or "if not…" after a **Yes**. It is also FULL when the answer shows the part's precondition doesn't hold. *Example:* exploration hasn't started, so "how much gold is there?" or "is it viable?" can't be known. Otherwise it is **NONE + DOES_NOT_ARISE**. *Pilot:* rice-export (b) is NONE, because (a) was never answered No.
- **R2, judge each part only on the text that addresses it.** In a shared answer, one part can be FULL and another not. Ignore paragraphs that clearly address *other* parts, including when you choose the evasion type.
- **R3, annexures.** "Details at Annexure-I", or a bare "As annexed", in reply to a data request is **FULL**, even if you can't see the annexure. Use `ANNEXURE_ONLY` only if the reference is vague or mismatched. *You don't need to flag unseen annexures; the analysis does that automatically.*
- **R4, "not maintained" plus substitute data.** If the substitute measures the **same** quantity from another source, it is **PARTIAL + NOT_MAINTAINED**. *Iron ore:* consumption is "not maintained", but industry-reported consumption is given. If it's a **different** quantity, it is **NONE + NOT_MAINTAINED**. *Example:* BPL population given when *unemployed* BPL youth were asked about.
- **R5, "State subject" openers.** They **don't lower** the label if the requested content is then fully given (**FULL**). Use JURISDICTION only when the deflection *replaces* content: PARTIAL if some content is given, NONE for a bare "State subject".
- **R6, empty answers.** If the answer is empty or missing, every part is **NONE + OTHER**.
- **R7, ignore boilerplate.** Ignore the ministry header, the minister's name and "A statement is laid on the Table". Judge presence, not truth or quality.
- **R8, brevity isn't evasion.**
  - **FULL:** a short answer that names the specific measures, numbers or decision. *Example:* "recruitment rallies and media campaigns" as measures to attract applicants.
  - **GENERIC:** an action stated with no specifics. *Example:* "taken up with the concerned authorities", with no who, what or outcome.
- **R9, substitution.** A **different** quantity, scope or period counts as DIFFERENT_QUESTION: PARTIAL if some requested content is also given, otherwise NONE. The **same** quantity at a coarser breakdown or for fewer periods is **INCOMPLETE** (R14), not substitution.
- **R10, premise.**
  - A **clear factual denial or correction** that answers the part is **FULL**. *Examples:* "no repeated breakdowns occurred"; "15 washeries are being set up, not 20".
  - A denial that **concedes a related issue and leaves the requested item unaddressed** is **PARTIAL + DIFFERENT_QUESTION**.
- **R11, not possible yet.** If the data or timeline **cannot exist yet** and the answer says why, it is **FULL**. *Examples:* the season is still running; a required approval hasn't been given. NOT_MAINTAINED is only for data that could exist.
- **R12, "whether…" parts.**
  - A description that **clearly implies** the answer counts as answering. *Example:* details of a sanctioned institute answer "whether the Government has decided to set it up", so FULL.
  - Only when it implies **neither** yes nor no is the part at most **PARTIAL + GENERIC** (related policy described), or **NONE** (nothing relevant).
  - "Whether … proposes a new scheme" answered with existing schemes only is **PARTIAL + GENERIC**.
- **R13, garbled or mis-split text.** Code what's legible. If the part's answer cell is mis-split, judge against the full answer, and write "garbled" or "split" in the note.
- **R14, incomplete.** The content is there but a requested element is missing: a state-, company- or year-wise breakdown, a period, a date for each item, one of several items, or details after a count. Code it **PARTIAL + INCOMPLETE**.
- **R15, withheld.** An explicit refusal is **WITHHELD**: **PARTIAL** if relevant facts are acknowledged (*e.g.* "the CAG draft commented on these blocks…"), otherwise **NONE**.
- **R16, precedence.** If several evasion codes fit, use the precedence order in §3.
- **R17, request type of mixed parts.** Use the tie-break order in §4.
- **R18, OTHER is a last resort.** Never use OTHER if a listed code fits. Always explain it in the note.

## 6. Worked examples (real answers, shortened)
| # | Part asks | Answer says | Code |
|---|---|---|---|
| 1 | No. of PILs on medical termination of pregnancy, last ten years | "Information is being collected and will be laid on the Table" | NONE, DEFERRAL, INFORMATION |
| 2 | Whether commercialisation of drinking water caused scarcity… | "Water supply … is a State subject…", and nothing else | NONE, JURISDICTION, INFORMATION |
| 3 | Whether quackery is flourishing… | "Health is a State subject … Delhi reported raids in 283 cases…" | PARTIAL, JURISDICTION, INFORMATION |
| 4 | Iron ore produced and consumed, state-wise | Production at Annexure-I; consumption not maintained, but industry-reported consumption given | PARTIAL, NOT_MAINTAINED, INFORMATION (R4) |
| 5 | "If so, the reasons therefor" | "Does not arise in view of (b) above", where (b) = "No" | FULL (R1) |
| 6 | Total power generated in Gujarat | "4593 Million Units in October 2002" | FULL, INFORMATION |
| 7 | Measures to attract applicants to the armed forces | "Recruitment rallies and media campaigns" | FULL, ACTION (R8) |
| 8 | Whether about 20 washeries are being added | "15 are being set up, 9 for non-coking coal" | FULL, ACTION (R10) |
| 9 | Ongoing educational projects, state-wise | National list of schemes; annexures give state-wise *funds*, not projects | PARTIAL, DIFFERENT_QUESTION, INFORMATION (R9) |
| 10 | Funds allocated per year for two years | Only the 2020-21 figure | PARTIAL, INCOMPLETE, INFORMATION (R14) |
| 11 | Whether the CAG flagged cost escalation | "CAG draft commented on these blocks … no comments can be made as the report is in draft" | PARTIAL, WITHHELD, INFORMATION (R15) |
| 12 | Whether the scheme covers sharecroppers, and details | "Yes … details of coverage not available as the season is in progress" | FULL, INFORMATION (R11) |
| 13 | Whether the Government proposes a scheme for youth employment | A list of existing schemes | PARTIAL, GENERIC, ACTION (R12) |

## 7. Comparability
FULL / PARTIAL / NONE map onto the reply / partial reply / non-reply categories of Poljak (2024, *J. Legislative Studies*).
**To do:** check this mapping against Poljak's replication materials (doi:10.7910/dvn/0epyvn) before writing the paper.

## 8. Procedure for the main coding
- **Code independently.** Don't discuss the 100 shared questions until both of you are done.
- **Use the note field** for every uncertain case, every OTHER, and any "garbled" or "split" text.
- **The codebook is frozen.** Disagreements in the 100 shared questions are settled by discussion, applying v1.2 as written.
