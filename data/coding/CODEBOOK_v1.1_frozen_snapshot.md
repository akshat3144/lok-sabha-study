# Codebook v1.1: Responsiveness of written ministerial answers (Lok Sabha, 1999–2026)
*v1 was drafted on 2026-10-17, before any coding. **v1.1** revises it after the 50-question pilot (194 parts, two coders; label κ 0.84, ordinal α 0.88; 30 parts disagreed on at least one field). Every change below fixes a pattern seen in those disagreements. v1.1 is **frozen** once the pilot adjudication is complete. The machine-readable version is `codebook.py`; keep the two in sync.*

**What changed in v1.1:**
- **Request types** are defined more sharply. INFORMATION vs ACTION was the most common request-type disagreement.
- A new evasion type, **`PROCEDURAL`**.
- An **evasion precedence order**, to settle GENERIC vs OTHER vs DIFFERENT_QUESTION.
- New rules **R8–R13**, covering brevity, substitution, premise correction, not-yet-available data, unanswered yes/no parts, and garbled text.

## 1. What you code
- **Unit:** one **question part**, for example "(a)", together with the answer text that addresses it.
  - Where the answer groups parts, as in "(a) to (c): …", that text is the answer for **each** of those parts.
  - A question with no parts is coded as one unit, `_all`.
- **The question you ask yourself:** *Does the answer provide what this part asks for?*
  - This is about **responsiveness**: is the requested content there?
  - It is **not** about whether the answer is true, whether the policy is good, or whether the tone is polite.
- **You are blind** to who asked the question (party, government or opposition). Don't look it up.

## 2. Labels (choose one per part)
| Label | Use when | Score |
|---|---|---|
| **FULL** | The specific information, reason or commitment asked for is provided: the numbers, the list, a clear yes/no plus the requested details. An annexure counts **only if** the answer says it contains the requested data | 1 |
| **PARTIAL** | The part is addressed, but a substantive element is missing. Examples: national figures when state-wise were asked; some years, not all; "Yes" without the asked-for details; one of two bundled questions answered | 0.5 |
| **NONE** | The requested content is not provided: the answer defers, deflects, says the data isn't maintained, answers something else, or gives only general description | 0 |

## 3. Evasion type (only for PARTIAL or NONE)
| Code | Typical wording |
|---|---|
| `DEFERRAL` | "Information is being collected and will be laid on the Table of the House." |
| `NOT_MAINTAINED` | "Such data is not maintained centrally" / "no such data is available." |
| `JURISDICTION` | "Health is a State subject…"; pushes the question to another authority |
| `PROCEDURAL` *(new in v1.1)* | Withholds the substance on procedural grounds: "report not yet tabled", "under examination/consideration", "sub judice", "confidential" |
| `DOES_NOT_ARISE` | "Does not arise", when the part **does** follow from earlier parts (see rule R1) |
| `DIFFERENT_QUESTION` | Answers a related but different question: a different scope, quantity or subject is the main move |
| `ANNEXURE_ONLY` | Points to an annexure whose content is missing, or doesn't contain what was asked |
| `GENERIC` | Describes schemes or policy in general terms, without the specifics requested |
| `OTHER` | Anything else; explain in the note |

**When several evasion types fit, use the first that applies in this order:**
DEFERRAL > NOT_MAINTAINED > JURISDICTION > PROCEDURAL > DOES_NOT_ARISE > DIFFERENT_QUESTION > ANNEXURE_ONLY > GENERIC > OTHER.

## 4. Request type (for every part)
| Code | The part asks… |
|---|---|
| `INFORMATION` | for facts, figures, lists, incidents, or the status of things in the world: numbers, whether something happened, whether data exists |
| `REASON` | why ("the reasons therefor") |
| `ACTION` | about the **government's own** decisions, schemes, steps or measures, whether taken, ongoing or proposed. For example: "whether the Government has decided / proposes to…", "steps taken", "measures to…", "action taken thereon", "whether the Government has set up / launched…" |
| `OPINION` | for the government's view or assessment |

- "If so, the details thereof" **takes the request type of the part it refers to**.
- "Details along with the reasons" asks what and why equally, so it is `REASON`.

This follows the information-vs-action distinction in Ojha & Mishra (2010).

## 5. Decision rules
- **R1, legitimate "does not arise".** If an earlier part was answered "No" and this part is conditional ("if so, the details thereof"), then "Does not arise" is **FULL**. The question has been answered.
- **R2, grouped answers.** "(a) to (c): …" can be FULL for some of those parts and PARTIAL or NONE for others. Judge each part separately against the shared text.
- **R3, annexures.** "Details are at Annexure-I" is FULL if the annexure is described as holding the requested data, even when we can't see the annexure. Use `ANNEXURE_ONLY` only when the reference is vague, or clearly doesn't match the request.
- **R4, partial data plus a reason.** "Production is at Annexure-I; consumption is not maintained centrally" answers half the request, so it is **PARTIAL**, coded `NOT_MAINTAINED`.
- **R5, jurisdiction with substance.** "X is a State subject; however, Delhi reported 283 raids…" gives *some* requested facts, so it is **PARTIAL**, coded `JURISDICTION`. A bare "State subject" is NONE.
- **R6, empty answer.** If the answer is missing or empty, every part is NONE, coded `OTHER`, with the note "no answer text".
- **R7, ignore boilerplate.** Ignore the ministry header, the minister's name and "A statement is laid on the Table". Judge the statement's content.

### New in v1.1 (from the pilot disagreements)
- **R8, brevity isn't evasion.** A short answer that names the specific measures, numbers or decision asked for is **FULL**. *Pilot example:* "measures to attract applicants" answered with "recruitment rallies and media campaigns" is FULL. `GENERIC` is only for answers that give **no** specifics of the kind requested.
- **R9, substitution.** If the answer gives a related but different scope or quantity, it is **PARTIAL** when some requested content is also given, and **NONE** otherwise. Examples: national instead of state-wise figures; spending instead of allocation; existing schemes when a *proposed* scheme was asked about.
- **R10, premise correction.** Correcting a factual premise with accurate specifics is **FULL**. *Pilot:* "about 20 washeries?" answered with "15 are being set up, 9 for non-coking coal".
- **R11, data that can't exist yet.** If the requested data cannot exist yet (for example, the season or survey is still in progress) and the answer says so, it is **FULL**. `NOT_MAINTAINED` is only for data that could exist but isn't kept.
- **R12, unanswered "whether…".** A "whether…" part that gets no yes/no and no direct statement either way is **at most PARTIAL**. If only related policy is described, it is PARTIAL + `GENERIC`; if nothing relevant is said, it is NONE.
- **R13, garbled text.** Code whatever is legible and write "garbled" in the note.

### Added during the pilot adjudication (gaps the coders found)
- **R14, incomplete specifics.** Some requested specifics are given, but others are missing, and no other evasion type applies. Code it **PARTIAL + `OTHER`**. Examples: a count without the asked-for details; one state of two. `GENERIC` is reserved for answers that give **no** requested specifics.
- **R15, "whether … proposes any scheme" answered with existing schemes.** Apply R12 (no yes/no, only related policy), so **PARTIAL + `GENERIC`**. For "whether" parts, R12 takes precedence over R9.
- **R16, who counts as "the Government".** For ACTION, it includes ministries, departments, attached offices, public-sector undertakings, and statutory or regulatory bodies (for example CIL, NSDC, BIS, RBI).
- **R17, bundled parts.** A part that bundles facts with "the action taken by the Government thereon" is **ACTION**.

## 6. Worked examples (real answers from the corpus, shortened)
1. **Q:** "the number of PILs filed in the Supreme Court for medical termination of pregnancy for the last ten years"
   **A:** "Information is being collected and will be laid on the Table of the House."
   **→ NONE, DEFERRAL, INFORMATION**
2. **Q:** "whether the commercialisation of drinking water in peri-urban areas has created water scarcity … and if so, the facts thereof"
   **A:** "Water supply and its regulation is a State subject and State Governments are responsible for its management…"
   **→ NONE, JURISDICTION, INFORMATION**
3. **Q:** "whether the Government are aware of the unabated flourishing of quackery…"
   **A:** "Health is a State subject … Government of NCT of Delhi has reported that raids have been conducted in 283 cases…"
   **→ PARTIAL, JURISDICTION, INFORMATION** (rule R5)
4. **Q:** "the details of quantity of iron ore produced/mined and consumed … State-wise"
   **A:** "The statewise production … is given at Annexure-I. Statewise consumption … is not maintained centrally. However, the reported consumption (by Industry) … is given at Annexure-I"
   **→ PARTIAL, NOT_MAINTAINED, INFORMATION** (rule R4; arguably FULL because a substitute figure is given; **discuss in the pilot**)
5. **Q:** "if so, the reasons therefor"
   **A:** "Does not arise in view of (b) above."
   **→ FULL if (b) was answered "No"** (rule R1). Otherwise **NONE, DOES_NOT_ARISE**.
6. **Q:** "the total quantity of power generated in Gujarat as on date"
   **A:** "The total power … generated in Gujarat during October, 2002 was 4593 Million Units."
   **→ FULL, INFORMATION**

## 7. Comparability
FULL / PARTIAL / NONE map onto the reply / partial reply / non-reply categories used in comparative question-time research (Poljak 2024, *J. Legislative Studies*).
**To do before freezing v1.1:** check this mapping against Poljak's published codebook and replication data (Harvard Dataverse, doi:10.7910/dvn/0epyvn).

## 8. Pilot procedure
1. Both coders code the same 50 pilot questions independently.
2. Compute agreement, then discuss every disagreement.
3. Revise the definitions and add examples **once**.
4. Freeze v1.1 and register it on OSF together with the hypotheses.
5. Pilot labels are **not** used in the main estimates.
