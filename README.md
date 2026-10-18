# Answering or Evading? Ministerial Responsiveness in India's Lok Sabha, 1999 to 2026

Code and data for the paper of the same name (Akshat Gupta, Plaksha University).

Parliamentary questions oblige ministers to answer. This project measures whether they do: for every part of 145,077 questions, sampled from 393,501 questions answered in the 13th to 18th Lok Sabha, an LLM labels the answer as full, partial or no answer and classifies the form of evasion. The labels are validated against a stratified human gold standard, where the model agrees as closely as two human coders agree with each other (κ 0.711 vs 0.708), and remaining bias is corrected with prediction powered inference.

**This repository contains no question or answer text.** The texts are records of the Lok Sabha Secretariat. The code below rebuilds them from the public source (sansad.in).

## Repository layout

| Path | Content |
|---|---|
| `collect.py`, `fetch_fallback.py` | Resumable collector for the official question and answer records and answer files |
| `pipeline/` | Cleaning, splitting questions into parts, aligning answers, OCR of scanned answers, party and coalition role of every asker |
| `analysis/` | Codebook (`CODEBOOK.md`), human coding sheets and validation, embedding measure (M1), Gemini labelling (M2, prompt in `m2_prompt.py`), models, bias correction, robustness, figures |
| `data/measures/release/` | **Public data release** (see its README): question metadata, labels for each question part, similarity scores, OCR provenance |
| `data/coding/` | Codebook versions, sampling frames and human gold labels |
| `data/ref/` | Party, coalition and ministry reference tables |
| `results/` | Tables, figures, validation, parity and robustness reports |

## Reproducing the results

```
pip install -r requirements.txt

# 1. Collect (hours; polite, resumable)
python collect.py api
python collect.py docs
python fetch_fallback.py

# 2. Build the corpus
python -I -X utf8 pipeline/clean.py
python -I -X utf8 pipeline/mp_party.py
python -I -X utf8 pipeline/link_roles.py        # also applies the OCR answers (pipeline/apply_ocr.py)
python -I -X utf8 analysis/parts.py

# 3. Measures
python -I -X utf8 analysis/m1_embed.py --stage all          # embedding similarity (GPU recommended)
python -I -X utf8 analysis/m2_subsample.py                  # probability sample for LLM labels
python -I -X utf8 analysis/m2_gemini.py batch --project <GCP project> --gcs-bucket <bucket> \
       --model gemini-3.8-flash --no-rationale --qids-file data/measures/m2_all_qids.csv

# 4. Validation and analysis
python -I -X utf8 analysis/validate.py report --stage main
python -I -X utf8 analysis/build_frame.py --outcome real
python -I -X utf8 analysis/models.py --outcome real --y y_m2 --family ols
python -I -X utf8 analysis/bias_correct.py --outcome real --family ols
python -I -X utf8 analysis/robustness.py --outcome real --y y_m2 --family ols
python -I -X utf8 analysis/explore_rq2.py
python -I -X utf8 analysis/explore_rq3.py
python -I -X utf8 analysis/explore_rq4_rq1.py
python -I -X utf8 analysis/paper_figures.py

# 5. Public release
python -I -X utf8 analysis/make_release.py
```

Tests: `python -I -X utf8 -m pytest analysis/tests pipeline/tests -q`.

The human gold labels in `data/coding/gold_labels.csv` let you validate any other labeller against the same standard without repeating the human coding.

## Collector details

- **Pause and resume:** press `Ctrl+C` or run `python collect.py stop`. Re-running a job skips everything already on disk. Files are written to `.part` and renamed when complete, and failed items are retried on the next run.
- **Progress:** `python collect.py status`. Logs are in `data/logs/`.
- **Politeness:** few workers, a pause between requests and exponential backoff on errors.

## License and citation

- **Code:** MIT License (`LICENSE`).
- **Derived data:** CC BY 4.0 (`LICENSE-DATA.md`).
- **Parliamentary texts:** not included and not licensed here.

Please cite: Gupta, A. (2026). *Answering or Evading? Ministerial Responsiveness in India's Lok Sabha from 1999 to 2026.*
