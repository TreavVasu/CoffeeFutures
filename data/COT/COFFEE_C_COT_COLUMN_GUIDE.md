# Coffee C COT Column Guide

I use this guide to interpret the CFTC Commitments of Traders fields in `data/COT/coffee_c_all_cot_data.csv`. The file combines Legacy Futures Only and Disaggregated Futures Only observations for Coffee C. Some fields are blank by design because the two report formats define different trader groups.

My modeling use is specific: I treat COT values as lagged explanatory variables for future Arabica Coffee C returns. I do not treat them as records of physical bean purchases or as matched buyer-seller transactions.

## Report Families

| Local report family | Categories I use | Role |
| --- | --- | --- |
| `legacy_futures_only` | Commercial, Noncommercial, Nonreportable | Provides the longest continuous history |
| `disaggregated_futures_only` | Producer/Merchant, Swap Dealer, Managed Money, Other Reportables, Nonreportable | Provides more detailed participant categories |

I preserve both families for traceability. I do not assume that a similarly named category is identical across report definitions.

## How I Interpret Positions

- `Long` records buyer-side futures exposure.
- `Short` records seller-side futures exposure.
- `Spread` records offsetting long and short positions, commonly across contract months.
- `Open Interest` is the number of outstanding contracts.
- `Change_in_...` is the change from the prior weekly report.
- `Pct_of_OI_...` normalizes a position by total open interest.
- `Traders_...` counts reportable traders in a bucket.
- `Conc_Gross_...` and `Conc_Net_...` measure concentration among the largest four or eight traders.
- `All` covers all contract months; `Old` and `Other` are CFTC crop-year buckets.

Coffee C contracts represent 37,500 pounds of exchange-grade green coffee. I use that conversion only for economic interpretation; the model operates on contracts, ratios, changes, and normalized features.

## Trader Categories

| Category | My interpretation |
| --- | --- |
| Commercial / `Comm` | Broad legacy-report commercial hedgers |
| Noncommercial / `NonComm` | Large traders not classified as commercial, often speculative or investment-driven |
| Producer/Merchant/Processor/User / `Prod_Merc` | Physical supply-chain firms managing price risk |
| Swap Dealer / `Swap` | Dealers hedging commodity-swap exposure |
| Managed Money / `M_Money` | Funds, CTAs, CPOs, and similar managers |
| Other Reportables / `Other_Rept` | Large reportable traders outside the named disaggregated categories |
| Nonreportable / `NonRept` | Smaller positions below reporting thresholds, derived as a residual |
| Total Reportable / `Tot_Rept` | Aggregate reportable positioning |

I interpret these as positioning regimes, not fixed directional identities. A commercial long may hedge future purchases, a commercial short may hedge production or inventory, and a managed-money position may form part of a spread or broader portfolio.

## Column Grammar

Most numerical names combine a measure, participant group, side, and maturity bucket:

```text
<Measure>_<Group>_<Side>_<Bucket>
```

For example, `Pct_of_OI_M_Money_Long_All` is the percentage of total Coffee C open interest held long by managed money across all contract months.

| Name component | Values |
| --- | --- |
| Measure | `Positions`, `Change_in`, `Pct_of_OI`, `Traders`, `Conc`, `Open_Interest` |
| Group | `Comm`, `NonComm`, `Prod_Merc`, `Swap`, `M_Money`, `Other_Rept`, `Tot_Rept`, `NonRept` |
| Side | `Long`, `Short`, `Spread` |
| Bucket | `All`, `Old`, `Other` |

## Metadata I Exclude From Modeling

I retain these fields for filtering and lineage but exclude them from the numerical feature set:

| Field | My use |
| --- | --- |
| `source_file` | Trace the row to a local source file |
| `source_dataset` | Identify the report family |
| `source_archive` | Identify the archive or historical bundle |
| `Market_and_Exchange_Names` | Verify the Coffee C market |
| `As_of_Date_In_Form_YYMMDD` | Compact report-date key |
| `Report_Date_as_MM_DD_YYYY` | Primary date for availability alignment |
| `CFTC_Contract_Market_Code` | Stable contract-market identifier |
| `CFTC_Market_Code` | Exchange or market metadata |
| `CFTC_Region_Code` | CFTC regional metadata |
| `CFTC_Commodity_Code` | Commodity identifier |
| `CFTC_SubGroup_Code` | Disaggregated-report subgroup metadata |
| `FutOnly_or_Combined` | Futures-only versus combined report flag |
| `Contract_Units` | Human-readable unit description |

## Numerical Measures I Use

| Column pattern | Meaning in my feature set |
| --- | --- |
| `Open_Interest_*` | Total market participation |
| `*_Positions_Long_*` | Long exposure for a participant group |
| `*_Positions_Short_*` | Short exposure for a participant group |
| `*_Positions_Spread_*` | Offset exposure across sides or maturities |
| `Change_in_*` | Weekly positioning impulse |
| `Pct_of_OI_*` | Position normalized by market size |
| `Traders_*` | Breadth of reportable participation |
| `Conc_Gross_*` | Gross concentration among the largest traders |
| `Conc_Net_*` | Net concentration among the largest traders |

## Derived Features

I derive features that summarize directional pressure and make different historical regimes more comparable:

| Feature | Formula |
| --- | --- |
| Managed-money net | `M_Money_Positions_Long_ALL - M_Money_Positions_Short_ALL` |
| Managed-money net share | `Pct_of_OI_M_Money_Long_All - Pct_of_OI_M_Money_Short_All` |
| Producer/merchant net | `Prod_Merc_Positions_Long_ALL - Prod_Merc_Positions_Short_ALL` |
| Commercial net | `Comm_Positions_Long_All - Comm_Positions_Short_All` |
| Noncommercial net | `NonComm_Positions_Long_All - NonComm_Positions_Short_All` |
| Nonreportable net | `NonRept_Positions_Long_All - NonRept_Positions_Short_All` |
| Managed-money weekly net change | `Change_in_M_Money_Long_All - Change_in_M_Money_Short_All` |
| Commercial weekly net change | `Change_in_Comm_Long_All - Change_in_Comm_Short_All` |
| Open-interest change rate | `Change_in_Open_Interest_All / Open_Interest_All` |

I prioritize `All` buckets for the first-pass model because they are the most consistent aggregate representation. I add `Old` and `Other` only when the research question requires crop-year structure and the historical coverage is adequate.

## Availability Rule

COT positions are measured on a report date and published later. I shift the report by the implemented release lag before forward-filling it onto the daily Coffee C calendar. I never join the Tuesday snapshot directly to Tuesday's forecast as if it were already public.

This timing rule is more important than small differences in feature engineering. A strong backtest with incorrectly aligned COT data would not be credible.

## Historical Naming Variants

I preserve source spellings such as `Postions`, `Spead`, and `Othr` in the raw and filtered files so I can trace columns back to CFTC archives. In feature code, I map or reference these names explicitly rather than silently renaming the source data.

## Reference Sources

- CFTC Legacy COT explanatory notes: https://www.cftc.gov/MarketReports/CommitmentsofTraders/ExplanatoryNotes/index.htm
- CFTC Disaggregated COT explanatory notes: https://www.cftc.gov/MarketReports/CommitmentsofTraders/DisaggregatedExplanatoryNotes/index.htm
- CFTC historical variable names: https://www.cftc.gov/MarketReports/CommitmentsofTraders/HistoricalViewable/deanexplanatory.html
- ICE Coffee C contract specifications: https://www.ice.com/products/15/Coffee-C-Futures/specs

