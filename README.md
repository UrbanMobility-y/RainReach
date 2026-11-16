# Extreme rainfall reaches farther than Chinese cities' regional markets

Code and source data for the study "Extreme rainfall reaches farther than Chinese cities' regional markets"

Repository: https://github.com/UrbanMobility-y/RainReach

```
git clone https://github.com/UrbanMobility-y/RainReach.git
```

## Repository contents

| Path | Contents |
|---|---|
| `01_spending_panel.py` | City-day spending panel for 225 cities, 2017-2019: local, inbound, outbound and total spending on an additive scale |
| `02_rainfall_and_events.py` | ERA5-Land daily rainfall for 369 prefecture-level units, 3-year return levels, extreme-rain days, city events and rainfall systems |
| `03_counterfactual_and_event_panel.py` | Event-time panel and counterfactual spending for every city event |
| `04_ties_and_rainfall_cooccurrence.py` | Intercity ties, same-day co-occurrence of extreme rain (ERA5-Land, CPC, CHIRPS), shared-rain shares and the event table |
| `05_event_estimates_and_robustness.py` | Spending declines during events, specification variants, decline ratios and the gradient with rainfall-system extent |
| `06_shared_rain_exposure.py` | Shared-rain share, its validation across rainfall products, the decay of co-occurrence with distance and the decoupling distance |
| `07_exante_prediction_and_transmission.py` | Ex-ante prediction of shared exposure, random reassignment of partners and the test for transmission |
| `08_city_clusters_and_attributes.py` | City clusters, exposure inside and outside clusters and differences by city attributes |
| `09_sector_deficits.py` | Cumulative spending deficits after the last day of extreme rain by sector group |
| `10_grouping_assessment.py` | Assessment of regional groupings: daily shortfalls, group shared-rain share, synchrony and excess peak shortfall |
| `12_sector_shocks_and_decline_ratios.py` | Supplementary results for Fig. 1: sector-group shocks, decline ratios, dose-response and extent slopes |
| `13_rainfall_products_and_origins.py` | Supplementary results for Figs 2 and 3: rainfall-product checks, origin counts and lead-lag profiles |
| `14_clusters_and_sector_deficits.py` | Supplementary results for Fig. 4: exposure inside and outside the 19 city clusters and deficits by sector |
| `15_group_variance_ratios.py` | Supplementary results for Fig. 5: variance ratios by grouping and group size with bootstrap intervals |
| `Reproducibility/Figure_1.ipynb` ... `Figure_5.ipynb` | Notebooks that redraw the data panels of Figs 1-5 from `Source data/`, one cell per panel |
| `Source data/Figure_1.xlsx` ... `Figure_5.xlsx` | Source data for Figs 1-5: one sheet per panel and a README sheet that defines every column |
| `Source data/City and event results/` | Event catalogue with event-level estimates, city-year shared-rain shares, decoupling distance of each rainfall unit and the East-West cooperation pairs; columns are defined in `README.txt` |
| `Source data/Supplementary tables/` | Complete versions of Supplementary Tables 1-20 as CSV files; titles and notes in `Supplementary_Tables_titles_and_notes.csv` |
| `requirements.txt` | Python package versions |
| `LICENSE`, `Source data/LICENSE` | Licences for the code (MIT) and the source data (CC BY 4.0) |

## Data

The daily China UnionPay spending indices and intercity spending records were obtained from a commercial data service provider under a licence that does not permit redistribution, so they are not included. Scripts that read them are provided in full; running them requires access to the records. All values in `Source data/` are aggregates (panel-level values, event-level summaries and city-level measures); no daily spending series or city-to-city spending ties are included.

Public inputs:

- ERA5-Land hourly total precipitation, Copernicus Climate Change Service Climate Data Store (https://cds.climate.copernicus.eu). `02_rainfall_and_events.py` can download it with the `cdsapi` client and a personal CDS key.
- CPC Unified Gauge-Based Analysis of Daily Precipitation, NOAA Physical Sciences Laboratory (https://psl.noaa.gov).
- CHIRPS daily precipitation, Climate Hazards Center (https://www.chc.ucsb.edu/data/chirps).
- IBTrACS version 4 (release v04r01, Western Pacific), NOAA National Centers for Environmental Information, used for supplementary robustness checks.
- Prefecture-level boundaries; city attributes from the datasets and official documents cited in the article; city-cluster membership and the East-West cooperation pairs compiled from official plans (provided as small tables).

## How to run

1. Set `ROOT` at the top of each script to the folder that holds `Dataset`, `External data`, `Intermediate` and this repository. Scripts read raw inputs from `Dataset` and `External data` and write intermediate files to `Intermediate`.
2. Run the scripts in numerical order. Each script lists its inputs and outputs and can be run on its own once its inputs exist. Run times on a 32-core workstation range from seconds to about 85 minutes (`10_grouping_assessment.py`).
3. Open the notebooks in `Reproducibility/` to redraw the main-figure panels from `Source data/` (tested with Python 3.12 and the same package versions plus Jupyter). Map panels (Figs 1a, 4a and 5b) also need the prefecture-level boundary files, which are not redistributed; set `GIS` in the first code cell of those notebooks to a local copy. Schematic panels are not redrawn, and in Fig. 1a the background intercity ties are omitted and partner ties are drawn by weight class, because city-to-city spending ties are not released.

## Software

Analyses were run with Python 3.11.17 and the package versions in `requirements.txt` (pandas 3.1.0, numpy 2.4.6, scipy 1.17.1, pyfixest 0.60.0, scikit-learn 1.9.1, geopandas 1.2.0, shapely 2.2.0, pyproj 3.7.2, matplotlib 3.11.2, xarray 2026.9.0, netCDF4 1.7.4, pyarrow 25.0.0, openpyxl 3.1.5).

Notes on reproduction:

- Some intermediate results were first produced with pandas 2.3.3. Under pandas 3.1, three files differ only where a value lies exactly on a group boundary (one tercile label among 643 events, three percentile bands among 448 city-years, and tercile medians in one table of within-grouping slopes, whose slopes and intervals are unchanged); with pandas 2.3.3 they reproduce byte for byte.
- Standard errors from two-way clustered fixed-effects models vary between runs at the order of 1e-8 because of iterative demeaning; point estimates are unchanged.
- Daily rainfall for 1 January 1981 requires ERA5-Land data for 31 December 1980 (UTC+8 aggregation).

## Licence

Code: MIT licence (`LICENSE`). Source data: Creative Commons Attribution 4.0 International (`Source data/LICENSE`).
