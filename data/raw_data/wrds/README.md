# WRDS data (not included)

CRSP and Compustat data are licensed through WRDS (Wharton Research Data Services) and cannot be shared.
To run the notebooks, download the following tables from WRDS and save them in this folder as gzip CSV files
with these names:

| File | WRDS table |
|---|---|
| `crsp_security_info_history.csv.gz` | `crsp_a_stock.stksecurityinfohist` |
| `crsp_issuer_info_history.csv.gz` | `crsp_a_stock.stkissuerinfohist` |
| `crsp_daily_market_index.csv.gz` | CRSP daily market indexes (`vwretd`, `ewretd`, `sprtrn`) |
| `crsp_daily_event_firms.csv.gz` | `crsp_a_stock.stkdlysecuritydata` (event firms) |
| `crsp_daily_rename_firms.csv.gz` | `crsp_a_stock.stkdlysecuritydata` (all renaming firms, 1994–2025) |
| `crsp_daily_announcement_firms.csv.gz` | `crsp_a_stock.stkdlysecuritydata` (8-K announcement firms) |
| `crsp_monthly_announcement_firms.csv.gz` | `crsp_a_stock.stkmthsecuritydata` (8-K announcement firms) |
| `ccm_link_history.csv.gz` | `crsp_a_ccm.ccmxpf_lnkhist` |
| `ccm_permno_gvkey_cik.csv.gz` | `crsp_a_ccm.ccmxpf_lnkhist` joined with `comp.company` |
| `compustat_funda_1987_2025.csv.gz` | `comp.funda` joined with `comp.company` |
| `ff_factors_daily.csv.gz`, `ff_factors_monthly.csv.gz` | `ff_all.factors_daily`, `ff_all.factors_monthly` |

Columns and coverage of each file are described in `code/01_data_overview.ipynb`.
