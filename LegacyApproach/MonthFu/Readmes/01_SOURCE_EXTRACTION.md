# 01 — Source extraction and the supplied foundation

The project began with three market and environment sources: Yahoo Finance, CFTC COT and Open-Meteo. These acquisition pipelines were already present in the supplied repository. MonthFu reused their local inputs and preserved the original work; monthly model training did not download replacement data. A fourth GDELT event layer was supplied and later removed.

| Source | Implemented extraction | Local storage |
| --- | --- | --- |
| Yahoo Finance | `yfinance.download("KC=F", ...)` retrieves daily Coffee C futures OHLCV | `../DATA/yahoo/arabica_coffee_futures_history.csv` |
| CFTC COT | Discover historical compressed-report links, download ZIPs, extract tables, filter `COFFEE C` rows | `../DATA/cftc_cot/`, `../DATA/raw_cot_data/`, `../DATA/COT/` |
| Open-Meteo | Request daily historical weather for three configured coordinates | Combined and regional CSV caches under `../DATA/weather/` |

The GDELT row and its artifacts were removed with the news layer; only the three market
and environment sources above remain in use.

The Yahoo and CFTC stages are implemented in [DataExtraction.ipynb](../../Scripts/notebooks/DataExtraction.ipynb). Yahoo columns are flattened when needed and `Date` becomes an ordinary CSV column. The input is a continuous futures history: it does not supply individual-contract roll decisions, a futures curve, physical delivery records, or certified executable returns.

CFTC downloads default to legacy futures-only reports from 2000 onward and disaggregated futures-only history, including the 2006–2016 bundle and yearly archives from 2017. Futures-and-options and commodity-index supplemental families are configured optional choices. Extraction separates report families and archive names to avoid filename collisions. Coffee rows retain source-file, report-family and archive identifiers; combined tables and filter manifests record the processing results. Early disaggregated coverage in an archive is not proof that it was published contemporaneously; monthly availability rules are explained later.

[refresh_weather_cache.py](../../Scripts/project/scripts/refresh_weather_cache.py) calls `https://archive-api.open-meteo.com/v1/archive` for Minas Gerais, Huila and Dak Lak, using each region's local timezone. JSON daily arrays become date-indexed CSV columns with region prefixes. Variables include temperatures, precipitation, humidity, wind, radiation and evapotranspiration, plus soil moisture, soil temperature and vapor-pressure deficit when supported. Failed extended-variable requests can fall back to core variables. The script saves regional caches, combined data, refresh metadata and a backup.

[fetch_missing_data.sh](../../Scripts/fetch_missing_data.sh) is a local integrity/rebuild entry point. It requires the committed Yahoo/COT base CSV, then refreshes weather or rebuilds downstream tables when absent or forced. It does not itself redownload Yahoo and CFTC. Its former `--with-events` GDELT stage was removed with the news layer.

Historical weather is reanalysis, regional coverage is partial, and the supplied sources lack complete publication-vintage snapshots. These remain limitations after feature engineering.

[Previous: 00 — Index](00_INDEX.md) · [Next: 02 — ELT/ETL data preparation](02_ELT_ETL_DATA_PREPARATION.md)
