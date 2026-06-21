# Ukrainian Air-Raid Alert Analysis

End-to-end time-series analysis of official Ukrainian air-raid siren data,
built to surface temporal, regional, and special-date patterns relevant to
defense analytics — and to honestly assess what is and isn't predictable.

**Data source:** [Vadimkin/ukrainian-air-raid-sirens-dataset](https://github.com/Vadimkin/ukrainian-air-raid-sirens-dataset)
(official feed, oblast level)
**Trustworthy analysis window:** 2022-03-15 → 2025-10-31
*(the raw feed both duplicates history and degrades after Oct 2025 — see caveats below)*

---

## One-line verdict

> This data is **excellent for describing and detecting patterns**, and **poor
> for forecasting the timing of individual alerts**. Both halves are backed by
> metrics in this project, not asserted.

---

## Headline findings

| Confidence | Finding |
|---|---|
| **Confident** | Strong **hour-of-day seasonality** (dawn trough ~05–07h, midday peak ~12h, evening bump ~21h) |
| **Confident** | Alerts **concentrated in front-line/eastern-southern oblasts** (Donetsk, Zaporizhzhia, Kharkiv, Dnipro) |
| **Confident** | **War-phase trend** — non-stationary, peaking early–mid 2024 |
| **Confident** | **Anomaly detection works** as a retrospective tool (37 unusual days flagged) |
| **Suggestive** | **Holiday effects** — New Year (+28%) and Western Christmas (+19%) elevated; **24 Feb invasion anniversary suppressed (−11%)**. Only 3–4 samples each. |
| **Conclusively limited** | **Forecasting alert timing does NOT work** from calendar features (daily NB-GLM doesn't beat naive; hourly AUC ≈ 0.57) |

Full detail, caveats, and next steps: **[`CONCLUSIONS.pdf`](CONCLUSIONS.pdf)**
(color-coded report) or **[`CONCLUSIONS.txt`](CONCLUSIONS.txt)** (plain text).

---

## Project structure

```
.
├── src/
│   ├── ingest.py      Phase 1 — load, dedupe, censor-flag, tz-convert, clean
│   ├── eda.py         Phase 2 — seasonality, regional & duration EDA, STL
│   ├── anomaly.py     Phase 3 — STL-residual anomaly detection + holiday windows
│   └── forecast.py    Phase 4 — Kyiv City daily & hourly forecasting + honest limits
├── figures/           generated plots + holiday_effects.csv
├── make_pdf.py        builds the styled CONCLUSIONS.pdf
├── CONCLUSIONS.pdf    color-coded conclusions report
├── CONCLUSIONS.txt    plain-text conclusions (full detail + caveats)
├── requirements.txt
└── README.md
```

> **Note:** the raw and cleaned CSVs are **not committed** (see `.gitignore`).
> Regenerate `data/clean_oblast.csv` by running Phase 1 (below).

---

## Quick start

```bash
# 1. environment
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. fetch the raw dataset
mkdir -p data
curl -sL https://raw.githubusercontent.com/Vadimkin/ukrainian-air-raid-sirens-dataset/main/datasets/official_data_en.csv \
  -o data/official_data_en.csv

# 3. run the pipeline (from the src/ directory)
cd src
python ingest.py      # -> data/clean_oblast.csv
python eda.py         # -> figures/01..05
python anomaly.py     # -> figures/06,07 + holiday_effects.csv
python forecast.py    # -> figures/08,09 + printed metrics

# 4. (optional) rebuild the PDF
cd .. && python make_pdf.py
```

---

## Methodology notes (the assumptions a domain expert should verify)

- **Metric = alert onset counts**, not durations — robust to right-censoring.
- **Oblast level only**; raion/hromada rows double-count and hold outliers.
- **24h duration cap**, flag-only (13 alerts, all front-line — kept in counts,
  excluded from duration stats).
- **Kyiv local time** (`Europe/Kyiv`, DST-aware) for all hour/day analysis.
- **STL period = 7** (weekly); annual seasonality is weak/irregular here.
- **Anomalies** = STL residual ÷ 30-day rolling local std, |z| ≥ 3.
- **Holiday windows** = ±5 days vs same-year ±21-day local baseline (global-mean
  pooling is invalid because of the war-phase trend).
- **Forecasting** uses strict time-based splits (no shuffling); count-appropriate
  models (Negative-Binomial / logistic) and metrics (Poisson deviance, AUC, Brier).

## Critical data caveats

1. **~50% of raw rows are exact duplicates** — a publisher-side regeneration
   artifact. Removed losslessly; dedup must stay a permanent step.
2. **Feed collapses after Oct 2025** (only Kyiv City reports from Dec 2025).
   Not a ceasefire — a collection failure. Analysis is hard-cut at 2025-10-31.
3. **Durations are right-censored** when an "all-clear" was missed.
4. **Luhansk** has 1 record total — excluded from per-region work.

## Most promising next step

**Spatial nowcasting** — predict a region's imminent alert from its *neighbours'*
current state (alerts sweep geographically). This is the one framing the evidence
suggests could actually *predict*, rather than merely describe. See
`CONCLUSIONS.pdf` §"Next steps".

---

## License & attribution

Analysis code: MIT (suggested). Underlying data: see the
[source dataset's license](https://github.com/Vadimkin/ukrainian-air-raid-sirens-dataset)
(MIT). Please credit the dataset author (Vadym Klymenko / Vadimkin).

🇺🇦
