# Coalition reference table: method, sources, uncertainties, split handling

Built 2026-10-17 for the Lok Sabha Question Hour project (questions Oct 1999 - Oct 2026).
Files: `govt_periods.csv`, `coalitions.csv`, `coalition_lookup.py` (all in `data/ref/`).

## 1. Conventions

* **Dates** are ISO, intervals **inclusive** `[start_date, end_date]`; empty `end_date` = still open on 2026-10-17.
  Government periods are partitioned by swearing-in: a government's `end_date` is the day before its successor was sworn in.
* **Rows only describe relation to the government of the day.** The NDA 2004-2014 and UPA 2014-2026 are out of power,
  so NDA/UPA members in those windows are "opposition" (no row). Rows are clipped so that a party that joined an alliance
  before it took office starts on the government's first day (join date is in `notes`), and rows end at the government's end
  where the alliance left office (e.g. BJD stayed in NDA until 2009-03-07 but its row ends 2004-05-21).
* **role**: `government` = in council of ministers or formal alliance member; `outside_support` = declared support from outside / no ministers
  and described as outside support in sources; `opposition_formally_noted` (1 row, JMM 2009-13) = ambiguous, treat as opposition.
* Extra columns beyond the spec: `in_council_of_ministers` (yes/no/partial/unknown) and `confidence` (high/medium/low). `coalition_lookup.role(..., min_conf=)` can ignore weak rows.
* `party_name` uses Election Commission spellings from `eci0414.csv`/`eci1924.csv` (whitespace-insensitive; the lookup normalises
  case/spaces/punctuation and aliases `Indian Union Muslim League`<->`Muslim League Kerala State Committee`, `Nagaland Peoples Front`<->`Naga Peoples Front`, `Apna Dal`->`Apna Dal (Soneylal)`).
  `Shivsena` (2004-19) and `Shiv Sena` (2024) normalise to one key; dates separate them. Four 1999-only parties are not in the 2004-24 files
  (Himachal Vikas Congress, Akhil Bharatiya Loktantrik Congress, MGR Anna Dravida Munnetra Kazhagam, Manipur State Congress Party): common names used, **ECI 1999 spelling not verified**.
  Five rows have bracketed labels (`Shivsena [Shinde-faction MPs]` etc.): these are faction labels, not ECI names, reachable only through the MP lists in `coalition_lookup.py` (`FACTIONS`).
* **Question Hour only happens on sitting days.** Many exit dates probably fall in inter-session gaps or in the pre-election period after a Lok Sabha's last session (e.g. the Mar 2009 break-ups of LJP/RJD/SP with Congress; Feb-May 2024 joins), which would make their day-level uncertainty moot. I did NOT check the sitting calendar; do so before worrying about any single date.

## 2. Method and sources

1. Leads (user's list, Wikipedia NDA/UPA/List-of-members pages) were treated as hypotheses. Each dated entry/exit below was checked against at least one news/official source retrieved in this session
   (Tribune archive 1999-2008, Gulf News, Outlook, Business Standard (snippet only; page returned 403), PIB-type pages (President's press release), Deccan Herald, Scroll, The Hindu-type reports via search, newsonair, ETV Bharat, Congressional Research Service RL32465 for May 2004).
2. Wikipedia infoboxes were used to enumerate candidate parties and for **year-level** facts only. Where a row rests on Wikipedia alone it is `confidence=low` (or `medium` if the claim is
   uncontroversial) and the source column says so.
3. Seat holding: parties were enumerated from ECI winners of each general election (recomputed from the local ECI CSVs for 2004-2024, plus 1999 results from Wikipedia) and from the Wikipedia 1999 NDA candidate table.
   Coverage target: every party that held >=1 LS seat while in the governing alliance or giving formal outside support. Parties with zero LS seats in the relevant period (e.g. AIADMK 2025-, TMC(M), PMK 2019-26, AMMK, Twenty20) were NOT coded.
4. Search engine results for pre-2005 events were thin; several 1999-2004 small parties could only be sourced to the 1999 Wikipedia NDA candidate table and the Tribune 14 Oct 1999 report.

Main sources (full URLs in `source_urls`): Tribune 14 Oct 1999 (ministry composition and parties outside it), 16 Mar 2001 (Trinamool exit), 28 Jul 2001, 30 Apr 2002 (Paswan), 24 May 2003 (Ajit Singh), 13 Jul 2003 (NC), 10 Feb 2004 (INLD),
22 Jun 2008 (BSP), 4 Jul and 9 Jul 2008 (SP/Left), 24 Sep 2006 (TRS), 21 Sep 2012 (TMC); Outlook (Dec 2003, DMK); Gulf News 13 Jan 2004 (PMK/MDMK); CRS RL32465 (May 2004);
President of India press release 9 Jul 2008; Deccan Herald/Governance Now 2013 (DMK); Bloomberg Quint/Udayavani 2018 (TDP); Deccan Herald/Scroll/Tribune 2019-2020 (Sena, SAD); Business Today/Madhyamam (JD(U)); Tribune/Quint/Onmanorama (AIADMK);
newsonair (JD(S), LJP(RV)); Scroll (NCP/ECI); Outlook/FPJ (Shiv Sena MPs 2022); Tribune (LJP 2021); Wikipedia "2026 Trinamool Congress split", Tribune 14 Jun 2026, Wikipedia NCPI (2026).

## 3. Corrections to the leads

| Lead | Finding |
|---|---|
| DMK left NDA Dec 2003 | Confirmed: decision 2003-12-20 (ministers Baalu, Raja resigned). Not 2002. |
| National Conference left 2002? | Omar Abdullah resigned as minister 2002-10-14 but NC stayed; **NC quit NDA 2003-07-12**. |
| TMC in/out | Out 2001-03-15 (Tehelka), back **2003-09-08** (Mamata minister without portfolio; coal from 2004-01-09), not Jan 2004. |
| LJP | Paswan resigned/LJP out 2002-04-29. Joined UPA 2004; left UPA by 2009; back in NDA 2014-02-27. |
| TDP 1999-2004 | No ministers; Speaker's post (Balayogi). Coded `outside_support` (alternative: formal NDA constituent => government). |
| Left Front to July 2008 | Confirmed: decision 2008-07-08, letters to President 2008-07-09. |
| TMC left UPA-II Sept 2012 | Announced 2012-09-18; ministers resigned 2012-09-21 (coded to the 21st). |
| DMK left UPA-II Mar 2013 | Withdrawal letter 2013-03-19; ministers resigned 2013-03-20 (not 21). |
| TDP left NDA Mar 2018 | Ministers quit 2018-03-08; NDA exit decision 2018-03-16 (coded to 16th). |
| Sena left NDA Nov 2019 | 2019-11-11 (Sawant resigned; party decision same day). |
| SAD Sept 2020 | Minister quit 2020-09-17; NDA exit 2020-09-26. |
| JD(U) Aug 2022 / Jan 2024 | 2022-08-09 confirmed; rejoin Jan 2024 (28 Jan per Wikipedia; Factly confirms 'January 2024'). Earlier: left 2013-06-16, formally rejoined 2017-08-19. |
| AIADMK left Sept 2023 | 2023-09-25 confirmed. Joined NDA 2019-02-19; rejoined 2025-04-11 (no LS seats in 2024). In 2014-19 AIADMK was **not** NDA. |
| JD(S) Sept 2023 | 2023-09-22 confirmed. |
| Shinde Sena from 2022 | State govt 2022-06-30; in Lok Sabha Speaker recognised Shewale 2022-07-19. |
| NCP Ajit from July 2023 | 2023-07-02; ECI order 2024-02-06 recognising Ajit's NCP. |
| SP/BSP/RJD for UPA | UPA-I: BSP (to 2008-06-21), SP (tacit; formal letter 2008-07-09). UPA-II: SP, BSP, RJD, JD(S) gave outside support in May 2009. |

## 4. Uncertain dates (month-level or weaker; `confidence` low/medium in the csv)

* **PMK/MDMK 2001 gap**: both left NDA before the 2001 TN assembly polls (BJP president, 2001-07-27: "two allies left", PMK re-entering); dates and whether their ministers resigned not found. Coded continuous until 2004-01-12. MDMK's exact 2004 exit day (<= 2004-01-12) uncertain.
* **TDP formal exit from NDA after May 2004**: not needed (end of Vajpayee govt); Wikipedia says 2004, month unknown.
* **RLD 2001 entry**: ministry induction ~2001-07-23 (Tribune said "tomorrow" on 22 Jul); RLD's 2 seats in 1999 listed by Wikipedia as supporters from Oct 1999, unverified.
* **LJP start 2000-11-28**: party founding date from Wikipedia; Paswan was already a minister.
* **TRS UPA exit**: ministers out 2006-08-24; support-withdrawal letter 2006-09-23..26 (sources disagree).
* **MDMK in UPA-I**: outside support after 2006 break; formal exit March 2007 (one source). 2004-06 status (inside DPA, no ministers) coded the same.
* **PDP, JMM, AIMIM, BPF, KC(M), VCK, JVM(P), JD(S)** in UPA-I/II: support/membership partly from Wikipedia lists; dates year-level (placeholders: JVM(P) 2012-12-31, JMM 2013-01-08). JMM 2009-13 is ambiguous (Wikipedia: NDA 2009-2013 via Jharkhand govt, but voted with UPA 2010-04-27).
* **SP in UPA-I**: tacit (CRS) 2004-2008, formal letter 2008-07-09; SP in UPA-II: letter May 2009, withdrawal threatened (2012-13) but never made.
* **PMK 2014-16** (exit placeholder 2016-03-31), **Swabhimani Paksha** (exit ~2017-08-31/09-01), **SDF 2014-19** (Wikipedia year-level), **PDP 2015-18** (state alliance; Centre status ambiguous), **TDP join 2014-04-06**, **RLSP join 2014-02-23** (clipped to govt start).
* **AJSU Nov 2019-2020 / NPF May 2019 / NPP Feb 2023**: state-level breaks; Centre-level status assumed continuous.
* **RLP join 2019-04-04** (Wikipedia, day unverified; exit 2020-12-26 verified).
* **MNF**: `end_date` 2023-07-26 is the *last confirmed* NDA date, exit (2023) not dated. **NDPP 2019-24**: ally in Nagaland, formal NDA status unverified.
* **UPPL**: Wikipedia shows NDA 2020-2026; left open-ended, exit not found (low).
* **RLD 2024-02-09 and HAM(S) June 2023** join dates (Wikipedia). **NCPI (TMC rebels) status**: see section 5.
* **Sources with 403/snippet only**: Business Standard (PMK 2004-01-13) and Deccan Herald (BSP 2009 outside support) pages could not be fetched; relied on search-result text.
* **Shiv Sena (Shinde) MP list**: the 12 names are from a 2022-07-19 press report; Gajanan Kirtikar's joining (Nov 2022) is from memory, unverified.

## 5. Handling party splits (proposed rule)

ECI party names reflect the election, not later splits, so a party-level lookup is wrong for MPs elected under a party that later split. Proposed rule:

1. Party-level classification (`coalitions.csv` rows keyed by ECI name) applies **until the split date**; for the *split window* only, classify each MP by his/her **documented side**, using the Lok Sabha Speaker's recognition (leader/whip change) as the effective date; where the Speaker did not rule, use the MP's public letter/declaration date.
2. Use a **manual MP list** (`coalition_lookup.FACTIONS`) for the window; MPs not on the list default to the other (opposition) side. Match on MP name or ECI constituency; constituency is safer (ECI spellings in `eci1924.csv`).
3. After the next general election the ECI names (2024) already encode the sides; see `Shiv Sena`, `Shiv Sena (Uddhav Balasaheb Thackrey)`, `Nationalist Congress Party` (Ajit), `Nationalist Congress Party - Sharadchandra Pawar`, `Lok Janshakti Party(Ram Vilas)`.

| Split | Window | Government side (MPs) | Other side |
|---|---|---|---|
| Shiv Sena | 2022-07-19 -> 2024-06-08 (before it: party left NDA 2019-11-11 so all Sena MPs are opposition 2019-11-12..2022-07-18) | 12 on 2022-07-19: Rahul Shewale (Mumbai South-Central), Krupal Tumane (Ramtek), Bhavana Gawali (Yavatmal-Washim), Hemant Patil (Hingoli), Hemant Godse (Nashik), Rajendra Gavit (Palghar), Shrikant Shinde (Kalyan), Shrirang Barne (Maval), Sadashiv Lokhande (Shirdi), Sanjay Mandlik (Kolhapur), Dhairyasheel Mane (Hatkanangle), Prataprao Jadhav (Buldhana); + Gajanan Kirtikar (Mumbai North-West) from ~Nov 2022 (unverified) | Vinayak Raut, Arvind Sawant, Omraje Nimbalkar, Sanjay Jadhav, Rajan Vichare (opposition). Names/constituencies cross-checked with the ECI 2019 file. |
| NCP | 2023-07-02 -> 2024-06-08 | Sunil Tatkare (Raigad) only. Amol Kolhe attended 2 Jul swearing-in but returned to Sharad Pawar within days (no sittings in between): opposition. | Supriya Sule, Shriniwas Patil (Satara), Mohammed Faizal (Lakshadweep), Kolhe |
| LJP | 2021-06-14 -> 2024-03-19 (Paras faction); Chirag to 2024-06-08 | Paras faction (RLJP, Paras a Cabinet minister Jul 2021-Mar 2024): Pashupati Kumar Paras (Hajipur), Prince Raj (Samastipur by-poll Nov 2019), Chandan Singh (Nawada), Veena Devi (Vaishali), Mehboob Ali Kaiser (Khagaria). Chirag Paswan (Jamui): coded government, low confidence (stated loyalty to Modi, but fought JD(U) in 2020 and was outside NDA meetings until 2023-24; formal NDA seat-sharing 2024-03-14). | After 2024-03-19 Paras group = opposition. |
| AITC (2026) | from 2026-06-08 | 20 rebels led by Kakoli Ghosh Dastidar (list in `FACTIONS`; Wikipedia lists 20, Tribune says 19 on 14 Jun). Coded `outside_support`, **low confidence**. They announced a merger with the Nationalist Citizens Party of India and "support" for NDA (Tribune: "join NDA"; Outlook (Sep 2026): 17 may join BJP, 3 to back from outside; Wikipedia NCPI infobox: NDA since 2026). No Speaker ruling or cabinet berth found. If your research design treats formal alliance membership as government, recode to `government`. | 8 loyal MPs (Abhishek Banerjee, Mahua Moitra, Kalyan Banerjee, Shatrughan Sinha, Kirti Azad, Sougata Roy, Pratima Mondal, Sajda Ahmed; Basirhat vacant/changed) = opposition |
| Apna Dal | 2016 | ECI 2014 label `Apna Dal` (Anupriya Patel, Harivansh Singh); Anupriya's faction (later `Apna Dal (Soneylal)`) stayed NDA; Krishna Patel's faction cut ties 2016-08 (no MPs). Both 2014 MPs presumed Soneylal side (unverified). | |
| RLSP | 2015 | MP Arun Kumar broke away (not verified); party stays NDA until 2018-12-10. | |

Independents who backed NDA/UPA (e.g. Navneet Rana, Sumalatha Ambareesh in 2019) were not coded.

## 6. Parties deliberately NOT coded as support (sensitivity-analysis candidates)

Issue-based supporters of the Modi governments without formal ties: **YSRCP (2014-24), BJD (2014-24, after leaving NDA 2009-03-07), BRS/TRS (2014-24)** and **AIADMK 2014-19** (37 MPs, Deputy Speaker M. Thambidurai; widely described as issue-based; not a formal NDA member). Also JD(S) before 2023 and the SAD after 2020 (Aug 2026 reports of an imminent SAD-BJP realignment are unconfirmed; SAD 1 MP is coded opposition). I did not collect sources for these characterisations, so they are not in the csv.
Parties with LS seats in the 1999-2004 NDA that I could not source beyond the 1999 candidate table are included at `low` confidence (HVC, MSCP, MADMK, SDF, ABLTC, Samata, Lok Shakti). Not researched at all: Kerala Congress/P.C. Thomas (Vajpayee minister 2003-04?), Indian Federal Democratic Party, Bharatiya Navshakti Party, Republican Party of India (A), Samajwadi Janata Party (Rashtriya) and other 1-seat 2004/2009 parties; their MPs default to `opposition`.

## 7. Problems found in the local ECI files (for the other agents)

* `eci0414.csv` **2014 lacks all of Odisha (21) and Chhattisgarh (11)**: only 511 constituencies, so recomputed BJP = 271 (official 282). 2009 has 541 of 543; 2024 has 542 (Surat unopposed, so BJP recomputes to 239 vs official 240).
* Mojibake in party names (e.g. `Nationalist Congress Party � Sharadchandra Pawar`, `United People�S Party, Liberal`); double spaces (`Janata Dal  (United)`); the same party appears under several labels (`Muslim League Kerala State Committee` -> `Indian Union Muslim League`; `Nagaland Peoples Front` -> `Naga Peoples Front`; `Telugu Desam`; `Lok Jan Shakti Party` vs `Lok Janshakti Party(Ram Vilas)`).
* ECI 1999 data not available locally; 1999 seat counts come from Wikipedia (BJP 182, INC 114, TDP 29, JD(U) 21, Shiv Sena 15, DMK 12, BJD 10, AITC 8, PMK 5, INLD 5, MDMK 4, NC 4, SAD 2, ABLTC 2, HVC 1, MSCP 1, MADMK 1, SDF 1); Wikipedia's 13th-LS page shows slightly different counts (BJP 186, PMK 8) because of by-elections/defections, so 1999 should be re-based on ECI data if available.

## 8. Suggestions

* Re-check the low-confidence rows (about 26 of 90) with Lok Sabha Secretariat/PRS data if any bear on a large share of questions: PDP, JMM, AIMIM, BPF, KC(M), VCK, JVM(P), SDF, MNF, NDPP, UPPL, HVC, MSCP, MADMK, ABLTC.
* Treat TDP 1999-2004 and INLD 1999-2004 as a sensitivity switch (`outside_support` vs `government`), as well as NCPI after 2026-06-08.
