# Dependent And Independent Values For Arabica Coffee Modeling

This note explains how to think about dependent and independent values in this project.

In simple terms:

- The **dependent value** is what you want to predict.
- The **independent values** are the inputs used to predict it.

For this repo, the most natural dependent value is usually a future Arabica Coffee C price or return, and the independent values are past price data plus COT positioning data.

## 1. Yahoo Finance Coffee Futures Data

File: `data/yahoo/arabica_coffee_futures_history.csv`

Columns:

| Column | What it means | Dependent or independent? | How to use it |
| --- | --- | --- | --- |
| `Date` | Trading date. | Index / time key | Use for sorting, joining, and creating lagged features. Not a prediction target. |
| `Close` | Final settlement/closing price for Coffee C futures that day. | Usually dependent | Best target if you want to predict price. More often, predict `next_day_close`, `next_week_close`, or future return based on this. |
| `High` | Highest traded price during the day. | Independent if lagged | Use previous-day or previous-week high as an input. Do not use same-day high to predict same-day close because that leaks information. |
| `Low` | Lowest traded price during the day. | Independent if lagged | Use previous-day or previous-week low as an input. It shows daily downside pressure or volatility. |
| `Open` | Opening price for the day. | Independent if lagged, sometimes same-day input | If predicting end-of-day close during the trading day, same-day open can be input. If predicting future prices from historical data, use lagged open. |
| `Volume` | Number of contracts traded that day. | Independent if lagged | Use as an activity/liquidity signal. Rising volume can confirm stronger market participation. |

## Recommended Yahoo Finance Modeling Setup

The cleanest setup is:

| Modeling goal | Dependent value | Independent values |
| --- | --- | --- |
| Predict next day's Coffee C close | `Close.shift(-1)` | Today's or previous days' `Open`, `High`, `Low`, `Close`, `Volume` |
| Predict next week's Coffee C close | `Close.shift(-5)` | Lagged OHLCV features over the last several trading days |
| Predict return instead of price | `future_return = Close.shift(-n) / Close - 1` | Lagged OHLCV and COT features |
| Predict direction | `future_return > 0` | Lagged OHLCV and COT features |

For machine learning, avoid using `High`, `Low`, `Close`, or `Volume` from the same date if your target is also the same date's `Close`. That creates leakage because the model sees information that would not have been fully known before the close.

## Useful Derived Yahoo Features

These are independent variables you can create:

| Feature | Meaning |
| --- | --- |
| `return_1d` | Daily percent change in close. |
| `return_5d` | Weekly percent change in close. |
| `range_pct = (High - Low) / Close` | Intraday volatility. |
| `close_to_open_pct = (Close - Open) / Open` | Strength from open to close. |
| `volume_change_pct` | Whether trading activity is increasing or decreasing. |
| `moving_avg_5`, `moving_avg_20` | Short-term and medium-term trend. |
| `close_vs_ma_20` | Whether price is above or below its recent average. |

## 2. COT Coffee C Data

Main file: `data/COT/coffee_c_all_cot_data.csv`

The COT data describes weekly positioning in Coffee C futures. It tells you how different trader groups are positioned: commercial hedgers, noncommercial speculators, managed money, producers/merchants, swap dealers, other reportable traders, and smaller nonreportable traders.

For predicting Arabica coffee futures price, COT columns are usually **independent variables**. They are explanatory inputs, not the target.

## Recommended COT Modeling Setup

| Modeling goal | Dependent value | Independent values |
| --- | --- | --- |
| Predict future Coffee C price | Future Yahoo `Close` | COT positioning values known at the latest report date |
| Predict future Coffee C return | Future Yahoo return | COT net positions, open interest, trader counts, and concentration ratios |
| Predict bullish/bearish direction | Future return direction | COT positioning changes and percent-of-open-interest columns |
| Analyze trader behavior itself | A chosen COT column | Other lagged COT columns |

Most of the time, do this:

- Dependent value: future `Close` or future return from Yahoo Finance.
- Independent values: lagged COT columns.
- Join key: align COT `Report_Date_as_MM_DD_YYYY` to the next available trading date or weekly price date.
- Important: COT is weekly and released after the report date, so use it with a reporting lag to avoid lookahead bias.

## COT Columns That Are Not Features

These are identifiers, dates, or traceability fields. They should usually not be independent variables in a model.

| Column | Role |
| --- | --- |
| `source_file` | Traceability only. |
| `source_dataset` | Report family label. Useful for filtering, not usually a numeric feature. |
| `source_archive` | Archive/year source. Traceability only. |
| `Market_and_Exchange_Names` | Confirms this row is Coffee C. Not a numeric feature after filtering. |
| `As_of_Date_In_Form_YYMMDD` | Date key. |
| `Report_Date_as_MM_DD_YYYY` | Main date key for joining to price data. |
| `CFTC_Contract_Market_Code` | Identifier for the Coffee C contract market. |
| `CFTC_Market_Code` | Exchange/market identifier. |
| `CFTC_Region_Code` | Metadata. |
| `CFTC_Commodity_Code` | Commodity identifier for Coffee C. |
| `CFTC_SubGroup_Code` | Commodity subgroup metadata. |
| `FutOnly_or_Combined` | Report type flag. |
| `Contract_Units` | Unit text, e.g. contracts of 37,500 pounds. Useful for interpretation, not modeling. |

## COT Columns That Are Independent Variables

These are the main feature groups to use for modeling Coffee C price or returns.

| COT column pattern | Independent or dependent? | What it means |
| --- | --- | --- |
| `Open_Interest_*` | Independent | Total outstanding Coffee C futures contracts. Rising open interest can mean more participation or stronger conviction. |
| `*_Positions_Long_*` | Independent | Number of long contracts held by a trader group. Long means buyer-side exposure. |
| `*_Positions_Short_*` | Independent | Number of short contracts held by a trader group. Short means seller-side exposure. |
| `*_Positions_Spread_*` | Independent | Offset long and short positions, often across contract months. Less directional than outright long or short. |
| `Change_in_*` | Independent | Weekly change in a position or open-interest value. Useful for detecting new buying or selling pressure. |
| `Pct_of_OI_*` | Independent | Position as a percent of total open interest. Useful because it normalizes positions across time. |
| `Traders_*` | Independent | Number of reportable traders in a group/side. Useful for participation breadth. |
| `Conc_Gross_*` | Independent | How concentrated positions are among the largest 4 or 8 traders, using gross positions. |
| `Conc_Net_*` | Independent | How concentrated positions are among the largest 4 or 8 traders, using net positions. |

## Key COT Independent Variables For Coffee C

These are usually the most useful COT inputs to start with:

| Feature | Why it matters |
| --- | --- |
| `Open_Interest_All` | Shows total market participation in Coffee C futures. |
| `M_Money_Positions_Long_ALL` | Managed money long exposure. Often linked to speculative bullish pressure. |
| `M_Money_Positions_Short_ALL` | Managed money short exposure. Often linked to speculative bearish pressure. |
| `Change_in_M_Money_Long_All` | New weekly managed-money buying. |
| `Change_in_M_Money_Short_All` | New weekly managed-money selling or short building. |
| `Pct_of_OI_M_Money_Long_All` | Managed-money long exposure normalized by market size. |
| `Pct_of_OI_M_Money_Short_All` | Managed-money short exposure normalized by market size. |
| `Prod_Merc_Positions_Long_ALL` | Physical coffee firms with long-side exposure, often users or hedgers of future purchases. |
| `Prod_Merc_Positions_Short_ALL` | Physical coffee firms with short-side exposure, often producers, merchants, inventory holders, or sellers hedging price risk. |
| `Comm_Positions_Long_All` | Legacy commercial long exposure. |
| `Comm_Positions_Short_All` | Legacy commercial short exposure. |
| `NonComm_Positions_Long_All` | Legacy noncommercial long exposure, often speculative. |
| `NonComm_Positions_Short_All` | Legacy noncommercial short exposure, often speculative. |
| `Swap_Positions_Long_All` | Swap dealer long exposure used to hedge client swap books. |
| `Swap__Positions_Short_All` | Swap dealer short exposure used to hedge client swap books. |
| `NonRept_Positions_Long_All` | Smaller-trader long exposure, derived by CFTC. |
| `NonRept_Positions_Short_All` | Smaller-trader short exposure, derived by CFTC. |
| `Conc_Gross_LE_4_TDR_Long_All` | Whether long exposure is concentrated among the largest 4 traders. |
| `Conc_Gross_LE_4_TDR_Short_All` | Whether short exposure is concentrated among the largest 4 traders. |

## Useful Derived COT Features

These are often better than raw long and short columns:

| Derived feature | Formula | Meaning |
| --- | --- | --- |
| `managed_money_net` | `M_Money_Positions_Long_ALL - M_Money_Positions_Short_ALL` | Net speculative fund positioning. Positive means funds are net long Coffee C. |
| `managed_money_net_pct_oi` | `Pct_of_OI_M_Money_Long_All - Pct_of_OI_M_Money_Short_All` | Managed-money net position normalized by open interest. |
| `producer_merchant_net` | `Prod_Merc_Positions_Long_ALL - Prod_Merc_Positions_Short_ALL` | Net physical supply-chain hedging pressure. |
| `commercial_net` | `Comm_Positions_Long_All - Comm_Positions_Short_All` | Legacy commercial net hedging position. |
| `noncommercial_net` | `NonComm_Positions_Long_All - NonComm_Positions_Short_All` | Legacy speculative net position. |
| `nonreportable_net` | `NonRept_Positions_Long_All - NonRept_Positions_Short_All` | Net small-trader position. |
| `managed_money_weekly_net_change` | `Change_in_M_Money_Long_All - Change_in_M_Money_Short_All` | Weekly shift in fund positioning. |
| `commercial_weekly_net_change` | `Change_in_Comm_Long_All - Change_in_Comm_Short_All` | Weekly shift in commercial hedging. |
| `open_interest_change_pct` | `Change_in_Open_Interest_All / Open_Interest_All` | Whether participation is expanding or shrinking. |

## COT Columns That Can Be Dependent Values

If your goal is price prediction, COT columns are independent variables.

But if your goal is to model positioning itself, then a COT column can become the dependent value. Examples:

| Modeling question | Dependent value |
| --- | --- |
| Will funds increase Coffee C longs next week? | `Change_in_M_Money_Long_All.shift(-1)` |
| Will managed money become more bullish? | `managed_money_net.shift(-1)` |
| Will commercial hedgers add shorts? | `Change_in_Comm_Short_All.shift(-1)` |
| Will open interest expand? | `Change_in_Open_Interest_All.shift(-1)` |

So the same COT field can be independent or dependent depending on the question. For this Arabica futures project, the recommended default is:

- Dependent: future Coffee C price, return, or direction from Yahoo Finance.
- Independent: historical/lagged Yahoo OHLCV plus COT positioning features.

## Practical Feature Selection Recommendation

Start simple:

1. Use Yahoo Finance to create the target:
   - `future_return_5d = Close.shift(-5) / Close - 1`
   - or `future_direction_5d = future_return_5d > 0`

2. Use lagged Yahoo features:
   - `Close`
   - `Volume`
   - `return_1d`
   - `return_5d`
   - `range_pct`

3. Use COT features:
   - `Open_Interest_All`
   - `managed_money_net`
   - `managed_money_net_pct_oi`
   - `producer_merchant_net`
   - `commercial_net`
   - `noncommercial_net`
   - `Change_in_Open_Interest_All`
   - `managed_money_weekly_net_change`

4. Lag COT features by at least one report cycle before using them to predict price.

## Final Rule Of Thumb

For Arabica Coffee C futures:

- **Dependent value**: what happens next to price.
- **Independent values**: what was already known before that price move.
- Yahoo OHLCV gives market price and trading activity.
- COT gives who is positioned long or short in Coffee C futures.
- COT does not say who physically bought coffee beans. It shows futures-market exposure by trader category.
