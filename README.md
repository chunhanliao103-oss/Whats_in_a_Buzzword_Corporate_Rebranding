# What's in a Buzzword? Corporate Rebranding in the Dot Com, Blockchain and AI Waves
We study US exchange-listed firms that renamed themselves with a technology buzzword during three waves:
dot com (1995–2002), blockchain (2016–2020) and AI (2021–2025).

## Research questions
| | Question | Firms |
|---|---|---|
| RQ1 | How common was buzzword rebranding in each wave? | 79 (97 over 1990–2025) |
| RQ2 | How did the stock market react to the announcement of the new name? | 47 |
| RQ3 | Did investors react differently to AI than to dot com rebranding? | 45 |
| RQ4 | Which firms rebrand (size, profitability, industry)? | 79 / 59 with accounts |

## Repository structure
| Folder | Content |
|---|---|
| `code/` | Notebooks 01–08 and the shared module `buzzword_event_study.py` |
| `data/` | Raw inputs (see Data) |
| `event_dates/` | Event register with announcement dates and deal types |
| `appendix/` | Blockchain case studies |

The notebooks write intermediate tables to `processed/` and result tables and figures to `output/`. These folders
are not included, because they contain data derived from licensed CRSP and Compustat files; they are created when
the notebooks are run.

Run the notebooks in order (01 → 08) from inside `code/`. Requirements: Python 3.11, pandas, numpy, scipy,
statsmodels, scikit-learn, matplotlib, openpyxl.

## Data
| Source | Use | Access |
|---|---|---|
| CRSP US Stock Database, CRSP/Compustat Merged | Name histories, daily returns and volume, identifier links | WRDS (licensed) |
| Compustat North America | Pre-event firm accounts and industry codes | WRDS (licensed) |
| SEC EDGAR | Announcement dates of new names; 8-K filings | Public |
| Kenneth R. French Data Library | Fama-French 12 industry definitions | Public |

- `data/raw_data/wrds/`: place for the CRSP, Compustat and CCM extracts (retrieved on 2026-10-15; not included,
  see the README in this folder)
- `data/raw_data/edgar_fts/`: SEC EDGAR full-text search results for 8-K filings
- `data/event_dates/`: hand-collected announcement dates, confidence grades and deal types per event

CRSP and Compustat data are licensed through WRDS and are not included in this repository; the source tables
are listed in `code/01_data_overview.ipynb`.
