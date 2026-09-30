"""Rebuild cached-origin features or replay saved monthly holdout predictions."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent))
from train import ROOT, MONTHFU, latest_forecast
from features import build_feature_frame
from modeling import add_targets
import joblib
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle",type=Path,default=MONTHFU / "artifacts/30calendar/model.joblib")
    parser.add_argument("--output",type=Path)
    parser.add_argument("--replay-date",help="Use saved out-of-sample prediction for a historical holdout origin")
    args = parser.parse_args()
    if args.replay_date:
        rows = pd.read_csv(args.bundle.parent / "holdout_predictions.csv",parse_dates=["Date"])
        row = rows.loc[rows.Date.eq(pd.Timestamp(args.replay_date))]
        if row.empty:
            raise ValueError("Requested date is not a saved evaluated holdout origin.")
        result = row[["Date","Close","target_end_date","selected","empirical_lower_80","empirical_upper_80"]].copy()
        result["mode"] = "Saved walk-forward replay; no use of final refitted model"
    else:
        bundle = joblib.load(args.bundle)
        base,_,_ = build_feature_frame(ROOT)
        frame = add_targets(base,bundle["horizon"],bundle["horizon_unit"])
        if frame.Date.max().strftime("%Y-%m-%d") < bundle["fitted_as_of"]:
            raise ValueError("Cache predates the fitted model; use saved holdout replay for history.")
        # When a stateful member is used, update every intervening observed price.
        inputs = frame.loc[frame.Date.ge(pd.Timestamp(bundle["fitted_as_of"]))]
        if any(m["spec"]["kind"]=="holt_winters" for m in bundle["members"]) and len(inputs)>1:
            from modeling import predict_bundle
            prediction = float(predict_bundle(bundle,inputs)[-1])
            result = latest_forecast(bundle,frame)
            result.loc[0,"predicted_return"] = prediction
            result.loc[0,"predicted_return_pct"] = 100*prediction
            result.loc[0,"implied_close"] = result.close.iloc[0]*(1+prediction)
            result.loc[0,"cv_empirical_lower_80"] = prediction-bundle["cv_absolute_error_q80"]
            result.loc[0,"cv_empirical_upper_80"] = prediction+bundle["cv_absolute_error_q80"]
        else:
            result = latest_forecast(bundle,frame)
    if args.output:
        out = args.output.resolve()
        if not out.is_relative_to(MONTHFU):
            raise ValueError("Keep outputs inside MonthFu.")
        out.parent.mkdir(parents=True,exist_ok=True)
        result.to_csv(out,index=False)
    print(result.to_string(index=False))


if __name__=="__main__":
    main()
