"""Independently verify saved monthly targets, scores, fit boundaries and replay.

This auditor recalculates scalar metrics and target endpoints without importing
the model's metric/target helpers. Final model replay intentionally exercises the
saved inference code. Historical replay remains the saved quarterly prediction
table: final refitted weights are never used to recreate old holdout forecasts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ.setdefault(key, "1")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MONTHFU = ROOT / "MonthFu"
sys.path.insert(0, str(MONTHFU / "src"))

EXPECTED_EXPERIMENTS = ["30calendar", "21sessions", "28calendar", "21calendar", "30sessions"]
METRIC_KEYS = ["rows", "rmse", "mae", "direction_accuracy", "correlation", "r2_vs_zero",
               "mean_prediction", "mean_actual"]


def scalar_metrics(y, prediction):
    actual, predicted = np.asarray(y, dtype=float), np.asarray(prediction, dtype=float)
    assert len(actual) and actual.shape == predicted.shape
    assert np.isfinite(actual).all() and np.isfinite(predicted).all()
    error = actual - predicted
    mse = float(np.square(error).mean())
    reference = float(np.square(actual).mean())
    return {"rows": len(actual), "rmse": np.sqrt(mse), "mae": np.abs(error).mean(),
            "direction_accuracy": (np.sign(actual) == np.sign(predicted)).mean(),
            "correlation": (np.corrcoef(actual, predicted)[0, 1]
                            if np.std(predicted) > 1e-12 and np.std(actual) > 0 else None),
            "r2_vs_zero": 1-mse/reference if reference else None,
            "mean_prediction": predicted.mean(), "mean_actual": actual.mean()}


def assert_scores(expected, observed, label):
    for key in METRIC_KEYS:
        if key not in observed:
            continue
        actual = observed[key]
        if expected[key] is None:
            assert actual is None or pd.isna(actual), f"{label}: {key} should be absent"
        else:
            # Some candidate learners return float32, whose CSV text round-trip
            # slightly changes derived correlation while preserving economics.
            assert np.isclose(expected[key], actual, rtol=1e-7,
                              atol=1e-8 if key in {"correlation", "r2_vs_zero"} else 1e-10), (
                f"{label}: {key} expected {expected[key]}, saved {actual}")


def read_prices():
    prices = pd.read_csv(ROOT / "data/yahoo/arabica_coffee_futures_history.csv")
    prices["Date"] = pd.to_datetime(prices.Date, errors="coerce").dt.normalize()
    prices["Close"] = pd.to_numeric(prices.Close, errors="coerce")
    prices = prices.loc[prices.Date.notna() & np.isfinite(prices.Close) & prices.Close.gt(0)]
    prices = prices.sort_values("Date").reset_index(drop=True)
    assert not prices.Date.duplicated().any(), "Duplicate raw price origins"
    return prices


def independent_labels(prices, horizon, unit):
    dates = pd.DatetimeIndex(prices.Date)
    indices = np.arange(len(prices))
    if unit == "calendar":
        endpoints = dates.searchsorted(dates + pd.Timedelta(days=horizon))
    else:
        assert unit == "sessions"
        endpoints = indices+horizon
    mature = endpoints < len(prices)
    result = prices[["Date", "Close"]].copy()
    result["target_end_date"] = pd.NaT
    result["target_close"] = np.nan
    result.loc[mature, "target_end_date"] = dates.take(endpoints[mature]).to_numpy()
    result.loc[mature, "target_close"] = prices.Close.to_numpy()[endpoints[mature]]
    result["target_return"] = result.target_close/result.Close-1
    result["target_sessions"] = np.where(mature, endpoints-indices, np.nan)
    return result


def verify_prediction_targets(predictions, labels, label):
    assert predictions.Date.is_monotonic_increasing and not predictions.Date.duplicated().any(), label
    matched = predictions.merge(labels, on="Date", suffixes=("_saved", "_raw"), validate="one_to_one")
    assert len(matched) == len(predictions), f"{label}: origin missing from source"
    assert matched.target_end_date_saved.eq(matched.target_end_date_raw).all(), f"{label}: wrong target end"
    for key in ["Close", "target_return", "target_close", "target_sessions"]:
        if key+"_saved" in matched:
            np.testing.assert_allclose(matched[key+"_saved"], matched[key+"_raw"],
                                       rtol=1e-10, atol=1e-11, err_msg=label+": "+key)


def mature_labels(labels, cutoff, years=0):
    result = labels.loc[labels.target_end_date.lt(cutoff) & labels.Date.lt(cutoff) & labels.target_return.notna()]
    if years:
        result = result.loc[result.Date.ge(cutoff-pd.DateOffset(years=years))]
    return result


def disjoint_positions(predictions, offset):
    positions, previous_end = [], pd.Timestamp.min
    for index, row in enumerate(predictions.iloc[offset:].itertuples(), start=offset):
        if row.Date >= previous_end:
            positions.append(index)
            previous_end = row.target_end_date
    return positions


def independent_block_interval(actual, selected, baseline, block, draws):
    """Vectorized paired moving blocks, independent of the model helper."""
    actual, selected, baseline = map(lambda values: np.asarray(values, dtype=float),
                                     [actual, selected, baseline])
    block = min(block, len(actual))
    rng = np.random.default_rng(42)
    starts = rng.integers(0, len(actual)-block+1,
                         size=(draws, int(np.ceil(len(actual)/block))))
    rows = (starts[:, :, None]+np.arange(block)).reshape(draws, -1)[:, :len(actual)]
    selected_losses = np.square(actual-selected)[rows].mean(axis=1)
    baseline_losses = np.square(actual-baseline)[rows].mean(axis=1)
    return np.quantile(np.sqrt(selected_losses)-np.sqrt(baseline_losses), [.025, .975])


def verify_all_cot_dataset(folder, prices, labels, summary, selection, specs):
    """Verify persisted training/inference matrices and every required fit schema."""
    schema = json.loads((folder / "feature_schema.json").read_text())
    assert schema["cot_policy"] == selection["cot_policy"] == "all"
    required = schema["required_cot_features"]
    assert required and len(required) == len(set(required))
    assert set(required) == set(selection["required_cot_features"])
    source_manifest = schema["source_field_manifest"]
    assert source_manifest["feature_schema_version"] >= 2
    assert source_manifest["features"] == len(required)
    assert source_manifest["excluded_prelaunch_disaggregated_backcasts"] == 0
    raw = pd.read_csv(ROOT / "data/COT/coffee_c_all_cot_data.csv", low_memory=False)
    raw = raw.loc[raw.source_dataset.isin(["legacy_futures_only", "disaggregated_futures_only"])]
    raw_dates = pd.to_datetime(raw.Report_Date_as_MM_DD_YYYY).dt.normalize()
    backcast = raw.source_dataset.eq("disaggregated_futures_only") & raw_dates.lt("2009-09-01")
    assert source_manifest["reports"] == len(raw), "COT archive reports were dropped"
    assert source_manifest["retained_prelaunch_disaggregated_backcasts"] == int(backcast.sum())
    for family, family_manifest in source_manifest["source_field_coverage"].items():
        included = family_manifest["included_source_fields"]
        excluded = family_manifest["excluded_source_fields"]
        assert all(value["feature"] in required for value in included.values())
        assert not any(value == "unrecognized_nonmeasurement_field" for value in excluded.values()), "Unrecognized COT source fields need review"
        assert not any(name.lower().startswith("cftc_") or "date" in name.lower() for name in included)
        observed = {name for name in included if pd.to_numeric(raw.loc[raw.source_dataset.eq(family), name], errors="coerce").notna().any()}
        assert observed == set(family_manifest["source_fields_with_observations"])
    prediction_inputs = pd.read_csv(folder / "prediction_inputs.csv.gz", parse_dates=["Date"])
    assert prediction_inputs.Date.eq(prices.Date).all() and len(prediction_inputs) == len(prices)
    actual_cot = [name for name in prediction_inputs if name.startswith("cot_") and pd.api.types.is_numeric_dtype(prediction_inputs[name])]
    assert set(actual_cot) == set(required), "Prediction matrix omits COT features"
    training = pd.read_csv(folder / "training_dataset.csv.gz", parse_dates=["Date", "target_end_date"])
    mature = mature_labels(labels, prices.Date.max()+pd.Timedelta(days=1))
    assert training.Date.tolist() == mature.Date.tolist(), "Persisted training rows have incorrect label maturity"
    verify_prediction_targets(training, labels, "Persisted training dataset")
    assert set(required).issubset(training.columns)
    prediction_by_date = prediction_inputs.set_index("Date").loc[training.Date]
    for name in required:
        np.testing.assert_allclose(training[name], prediction_by_date[name], equal_nan=True, rtol=1e-10, atol=1e-11, err_msg="Training/inference feature disagreement: "+name)
    expected_eligible = {name for name, spec in specs.items() if spec.get("require_all_cot")}
    assert expected_eligible == set(selection["eligible_candidates"])
    assert set(selection["recipe"]["weights"]).issubset(expected_eligible)
    inclusion = pd.read_csv(folder / "cot_feature_inclusion.csv.gz", parse_dates=["cutoff"])
    selected = inclusion.loc[inclusion.member.isin(expected_eligible)]
    assert len(selected) and selected.required.eq(True).all() and selected.retained.eq(True).all()
    assert selected.observed_rows.between(0, selected.training_rows).all()
    for _, fit in selected.groupby(["stage", "cutoff", "member", "comparison"], dropna=False):
        assert len(fit) == len(required) and set(fit.feature) == set(required), "A fit lacks required COT predictors"
    cv_fits = selected.loc[selected.stage.eq("cv")]
    for member in expected_eligible:
        member_rows = cv_fits.loc[cv_fits.member.eq(member)]
        assert member_rows.cutoff.nunique() == summary["cv_folds"], "Missing CV COT inclusion audit"
    holdout = selected.loc[selected.stage.eq("holdout") & selected.comparison.eq("selected")]
    for member in selection["recipe"]["weights"]:
        assert holdout.loc[holdout.member.eq(member)].cutoff.nunique() == pd.read_csv(folder / "holdout_predictions.csv", parse_dates=["Date"]).Date.dt.to_period("Q").nunique(), "Missing quarterly COT inclusion audit"
    final = selected.loc[selected.stage.eq("final")]
    assert set(final.member) == set(selection["recipe"]["weights"]), "Missing final COT inclusion audit"
    return {"features": len(required), "reports": len(raw), "retained_backcasts": int(backcast.sum()),
            "training_rows": len(training), "prediction_input_rows": len(prediction_inputs),
            "verified_feature_fit_rows": len(selected)}


def verify_experiment(folder, prices, replay=True):
    summary = json.loads((folder / "metrics.json").read_text())
    selection = json.loads((folder / "selection.json").read_text())
    all_cot = selection.get("cot_policy") == "all"
    labels = independent_labels(prices, summary["horizon"], summary["horizon_unit"])
    holdout = pd.read_csv(folder / "holdout_predictions.csv", parse_dates=["Date", "target_end_date"])
    cv = pd.read_csv(folder / "cv_predictions.csv", parse_dates=["Date", "target_end_date"])
    assert selection["recipe"] == summary["selection"], "Saved selection recipe changed"
    cutoff = pd.Timestamp(selection["holdout_start"])
    assert holdout.Date.ge(cutoff).all() and cv.target_end_date.lt(cutoff).all()
    verify_prediction_targets(holdout, labels, "holdout")
    verify_prediction_targets(cv, labels, "CV")
    assert len(cv) == summary["cv_rows"]
    assert str(holdout.Date.min().date()) == summary["holdout_start"]
    assert str(holdout.Date.max().date()) == summary["holdout_end"]
    metrics_table = pd.read_csv(folder / "holdout_metrics.csv")
    for row in summary["holdout_metrics"]:
        scores = scalar_metrics(holdout.target_return, holdout[row["model"]])
        assert_scores(scores, row, "JSON "+row["model"])
        saved = metrics_table.loc[metrics_table.model.eq(row["model"])]
        assert len(saved) == 1
        assert_scores(scores, saved.iloc[0], "CSV "+row["model"])
    for column, summary_key in [("selected", "cv_selected"), ("zero", "cv_zero"),
                                ("previous_transferred_ensemble", "cv_previous_ensemble"),
                                ("previous_cv_ensemble", "cv_previous_retuned")]:
        assert_scores(scalar_metrics(cv.target_return, cv[column]), summary[summary_key], "CV "+column)
    candidate_kinds = {item["name"]: item["kind"] for item in selection["candidates"]}
    for key in ["recipe", "previous_cv_recipe"]:
        recipe = selection[key]
        weights = recipe["weights"]
        assert weights and all(0 <= weight <= 1 for weight in weights.values())
        assert sum(weights.values()) <= 1+1e-9, "Shrinkage/convex weights exceed one"
        column = "selected" if key == "recipe" else "previous_cv_ensemble"
        # XGBoost emits float32. CSV parsing promotes its shortest decimal
        # representation to float64, so recomputed shrinkage differs by up to
        # float32 rounding. Preserve tighter checks for float64-only recipes.
        float32_member = any(candidate_kinds[name] == "xgb" for name in weights)
        np.testing.assert_allclose(cv[column], sum(cv[name]*weight for name, weight in weights.items()),
                                   rtol=2e-7 if float32_member else 1e-10,
                                   atol=1e-10 if float32_member else 1e-11)
        assert np.isclose(scalar_metrics(cv.target_return, cv[column])["rmse"], recipe["cv_rmse"])
    assert set(selection["previous_cv_recipe"]["weights"]) <= {"previous_rf", "previous_ar", "previous_hw"}
    folds = pd.read_csv(folder / "cv_folds.csv")
    for row in folds.itertuples():
        valid = cv.loc[cv.fold.eq(row.fold)]
        train = mature_labels(labels, valid.Date.min())
        assert train.target_end_date.lt(valid.Date.min()).all()
        assert train.Date.min() == pd.Timestamp(row.train_start)
        assert train.Date.max() == pd.Timestamp(row.train_end)
        assert train.target_end_date.max() == pd.Timestamp(row.latest_training_label_end)
        assert valid.Date.min() == pd.Timestamp(row.validation_start)
        assert valid.Date.max() == pd.Timestamp(row.validation_end)
        assert valid.target_end_date.max() == pd.Timestamp(row.latest_validation_label_end)
    assert len(folds) == summary["cv_folds"]
    assert cv.target_end_date.lt(cutoff).all()
    specs = {row["name"]: row for row in selection["candidates"]}
    cv_metrics = pd.read_csv(folder / "cv_metrics.csv")
    for row in cv_metrics.to_dict(orient="records"):
        valid = cv.loc[cv.fold.eq(row["fold"])]
        train = mature_labels(labels, valid.Date.min(), specs[row["candidate"]]["train_years"])
        assert len(train) == row["train_rows"]
        assert_scores(scalar_metrics(valid.target_return, valid[row["candidate"]]), row, "CV candidate")
    ranking = pd.read_csv(folder / "cv_ranking.csv")
    for row in ranking.to_dict(orient="records"):
        weights = json.loads(row["weights"])
        prediction = sum(cv[name]*weight for name, weight in weights.items())
        assert_scores(scalar_metrics(cv.target_return, prediction), row, "CV ranking")
    eligible_ranking = ranking.loc[ranking.eligible_for_selection.eq(True)] if all_cot else ranking
    assert np.isclose(eligible_ranking.rmse.min(), summary["selection"]["cv_rmse"]), "Recipe was not eligible CV-minimum"
    cot_dataset_audit = verify_all_cot_dataset(folder, prices, labels, summary, selection, specs) if all_cot else None
    refits = pd.read_csv(folder / "holdout_refits.csv")
    refit_count = 0
    for quarter, block in holdout.groupby(holdout.Date.dt.to_period("Q"), sort=True):
        start = block.Date.min()
        entries = refits.loc[refits.quarter.eq(str(quarter))]
        assert len(entries), "Missing quarter fit audit"
        for row in entries.itertuples():
            train = mature_labels(labels, start, specs[row.member]["train_years"])
            assert len(train) == row.train_rows
            assert pd.Timestamp(row.first_prediction_date) == start
            assert pd.Timestamp(row.train_last_target_date) == train.target_end_date.max() < start
            assert pd.Timestamp(row.train_last_date) == train.Date.max() < start
            if row.member == "previous_hw" and hasattr(row, "state_price_fit_end"):
                assert pd.Timestamp(row.state_price_fit_end) == prices.loc[prices.Date.lt(start), "Date"].max()
                assert pd.Timestamp(row.state_price_fit_start) == prices.Date.min()
        previous = mature_labels(labels, start)
        np.testing.assert_allclose(block.historical_mean, previous.target_return.mean(), rtol=1e-10, atol=1e-11)
        majority = np.sign(np.sign(previous.target_return).mean())
        assert block.training_majority_direction.eq(majority).all()
        assert block.zero.eq(0.).all()
        refit_count += 1
    year_metrics = pd.read_csv(folder / "holdout_by_year.csv")
    for row in year_metrics.to_dict(orient="records"):
        year = holdout.loc[holdout.Date.dt.year.eq(row["year"])]
        assert_scores(scalar_metrics(year.target_return, year[row["model"]]), row, "Yearly "+row["model"])
    disjoint = pd.read_csv(folder / "nonoverlapping_metrics.csv")
    for row in disjoint.to_dict(orient="records"):
        sampled = holdout.iloc[disjoint_positions(holdout, int(row["offset"]))]
        assert_scores(scalar_metrics(sampled.target_return, sampled[row["model"]]), row, "Non-overlap "+row["model"])
    band = np.quantile(np.abs(cv.target_return-cv.selected), .8)
    assert np.isclose(summary["cv_absolute_error_q80"], band)
    # Pandas also retains float32 when adding a scalar band to float32
    # predictions. Apply the same scoped serialization tolerance to bounds.
    selected_float32 = any(candidate_kinds[name] == "xgb" for name in selection["recipe"]["weights"])
    band_rtol = 2e-7 if selected_float32 else 1e-10
    band_atol = 1e-10 if selected_float32 else 1e-11
    np.testing.assert_allclose(holdout.empirical_lower_80, holdout.selected-band, rtol=band_rtol, atol=band_atol)
    np.testing.assert_allclose(holdout.empirical_upper_80, holdout.selected+band, rtol=band_rtol, atol=band_atol)
    coverage = ((holdout.target_return >= holdout.empirical_lower_80) &
                (holdout.target_return <= holdout.empirical_upper_80)).mean()
    assert np.isclose(summary["empirical_80pct_band_holdout_coverage"], coverage)
    for baseline, result in summary["block_comparisons"].items():
        difference = scalar_metrics(holdout.target_return, holdout.selected)["rmse"] - scalar_metrics(holdout.target_return, holdout[baseline])["rmse"]
        assert np.isclose(difference, result["rmse_difference"])
        interval = result["rmse_difference_95pct_block_interval"]
        assert len(interval) == 2 and interval[0] <= interval[1]
        assert result["block_sessions"] >= max(holdout.target_sessions.median(), 1)
        rebuilt = independent_block_interval(holdout.target_return, holdout.selected, holdout[baseline],
                                             result["block_sessions"], result["bootstrap_draws"])
        np.testing.assert_allclose(rebuilt, interval, rtol=1e-6, atol=1e-9)
    for result in summary["zero_bootstrap_block_sensitivity"].values():
        rebuilt = independent_block_interval(holdout.target_return, holdout.selected, holdout.zero,
                                             result["block_sessions"], result["bootstrap_draws"])
        np.testing.assert_allclose(rebuilt, result["rmse_difference_95pct_block_interval"], rtol=1e-6, atol=1e-9)
    cot = pd.read_csv(folder / "cot_release_audit.csv", parse_dates=["report_date", "publication_date",
            "feature_publication_date", "first_usable_session"])
    assert cot.publication_date.ge(cot.report_date).all()
    assert cot.feature_publication_date.ge(cot.publication_date).all()
    if all_cot:
        assert "historical_backcast" in cot and "expanded_history_publication_date" in cot
        cot["expanded_history_publication_date"] = pd.to_datetime(cot.expanded_history_publication_date)
        cot["raw_value_available_date"] = pd.to_datetime(cot.raw_value_available_date)
        assert cot.expanded_history_publication_date.ge(cot.raw_value_available_date).all()
        assert cot.feature_publication_date.ge(cot.raw_value_available_date).all()
        assert cot.loc[cot.historical_backcast, "publication_date"].ge("2009-10-20").all()
        assert len(cot) == cot_dataset_audit["reports"]
        assert int(cot.historical_backcast.sum()) == cot_dataset_audit["retained_backcasts"]
    usable = cot.loc[cot.first_usable_session.notna()]
    dates = pd.DatetimeIndex(prices.Date)
    first_usable = dates.take(dates.searchsorted(usable.feature_publication_date, side="right"))
    np.testing.assert_equal(usable.first_usable_session.to_numpy(), first_usable.to_numpy())
    for _, family in cot.groupby("source_dataset"):
        if all_cot:
            family = family.loc[~family.historical_backcast]
        ordered = family.sort_values("report_date")
        assert ordered.feature_publication_date.is_monotonic_increasing
    availability = pd.read_csv(folder / "availability_rows.csv", parse_dates=["Date"])
    assert availability.Date.eq(prices.Date).all() and len(availability) == len(prices)
    age_columns = [name for name in availability if name.endswith("age_days")]
    assert all(availability[name].dropna().ge(0).all() for name in age_columns), "Source has negative age"
    for path, digest in summary["input_sha256"].items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest() == digest, "Input changed: "+path
    code_changed = [path for path, digest in summary.get("code_sha256", {}).items()
                    if not (ROOT/path).exists() or hashlib.sha256((ROOT/path).read_bytes()).hexdigest() != digest]
    critical = [path for path in code_changed if path.startswith("MonthFu/src/") or path in {
        "MonthFu/scripts/train.py", "MonthFu/scripts/resume_export.py", "MonthFu/scripts/predict.py"}]
    assert not critical, "Training/inference code changed since artifact: "+str(critical)
    replay_error = None
    latest = pd.read_csv(folder / "latest_forecast.csv", parse_dates=["as_of_date"])
    assert len(latest) == 1 and latest.as_of_date.iloc[0] == prices.Date.max()
    if replay:
        import joblib
        from features import build_feature_frame
        from modeling import add_targets, predict_bundle
        bundle = joblib.load(folder / "model.joblib")
        assert bundle["recipe"] == summary["selection"]
        assert bundle["horizon"] == summary["horizon"] and bundle["horizon_unit"] == summary["horizon_unit"]
        base, feature_manifest, _ = build_feature_frame(ROOT)
        frame = add_targets(base, bundle["horizon"], bundle["horizon_unit"])
        final_cutoff = frame.Date.max()+pd.Timedelta(days=1)
        if all_cot:
            assert bundle["cot_policy"] == "all"
            assert set(bundle["required_cot_features"]) == set(selection["required_cot_features"])
        for member in bundle["members"]:
            spec = member["spec"]
            assert set(member["features"]) <= set(feature_manifest["groups"][spec["group"]])
            assert not any(name.startswith("target_") or "future_return" in name for name in member["features"])
            if bundle["status"] != "experimental_news":
                assert not any(name.startswith("news_") for name in member["features"])
            if "estimator" in member:
                if all_cot:
                    assert spec["require_all_cot"]
                    assert set(member["required_cot_features"]) == set(selection["required_cot_features"])
                    assert set(selection["required_cot_features"]).issubset(member["features"])
                final_train_dates = mature_labels(labels, final_cutoff, spec["train_years"]).Date
                fit_rows = frame.loc[frame.Date.isin(final_train_dates), member["features"]]
                fitted_medians = member["estimator"].named_steps["impute"].statistics_
                expected_medians = fit_rows.median().fillna(0).to_numpy() if all_cot else fit_rows.median().to_numpy()
                if all_cot:
                    assert member["estimator"].named_steps["impute"].keep_empty_features
                np.testing.assert_allclose(expected_medians, fitted_medians,
                                           rtol=1e-9, atol=1e-10, err_msg="Final imputer uses immature/future rows")
        prediction = float(predict_bundle(bundle, frame.tail(1))[0])
        replay_error = float(abs(prediction-latest.predicted_return.iloc[0]))
        assert replay_error < 1e-10, "Final saved bundle does not reproduce latest saved forecast"
    selected = scalar_metrics(holdout.target_return, holdout.selected)
    transferred = scalar_metrics(holdout.target_return, holdout.previous_transferred_ensemble)
    retuned = scalar_metrics(holdout.target_return, holdout.previous_cv_ensemble)
    return {"experiment": folder.name, "status": "passed", "cv_rows": len(cv), "holdout_rows": len(holdout),
            "quarterly_refits": refit_count, "verified_candidate_fold_scores": len(cv_metrics),
            "verified_disjoint_sample_scores": len(disjoint), "cot_audit_rows": len(cot),
            "latest_bundle_replay_absolute_error": replay_error,
            "noncritical_code_changes": code_changed,
            "rmse": float(selected["rmse"]), "skill_vs_zero": float(selected["r2_vs_zero"]),
            "rmse_improvement_vs_transferred": float(1-selected["rmse"]/transferred["rmse"]),
            "rmse_improvement_vs_retuned_previous": float(1-selected["rmse"]/retuned["rmse"]),
            "majority_direction_accuracy": float((np.sign(holdout.target_return) == holdout.training_majority_direction).mean()),
            "selected_direction_accuracy": float(selected["direction_accuracy"]),
            "empirical_band_coverage": float(coverage), "selected_recipe": summary["selection"],
            "selected_groups": sorted({specs[name]["group"] for name in summary["selection"]["weights"]}),
            "cot_policy": "all" if all_cot else "benchmark", "all_cot_dataset_audit": cot_dataset_audit,
            "selected_minus_zero_interval": summary["block_comparisons"]["zero"]["rmse_difference_95pct_block_interval"],
            "selected_minus_retuned_interval": summary["block_comparisons"]["previous_cv_ensemble"]["rmse_difference_95pct_block_interval"],
            "latest_origin": str(latest.as_of_date.iloc[0].date()),
            "checkpoint_export": summary.get("checkpoint_export")}, cv, holdout


def verify_horizon_table(artifact_dir, summaries, predictions):
    path = artifact_dir / "horizon_comparison.csv"
    if not path.exists():
        return "not yet generated"
    table = pd.read_csv(path)
    assert set(table.experiment) == set(summaries), "Horizon comparison does not match completed experiments"
    dates = set.intersection(*(set(p.Date) for p in predictions.values()))
    for row in table.itertuples():
        selected = predictions[row.experiment].loc[predictions[row.experiment].Date.isin(dates)]
        scores = scalar_metrics(selected.target_return, selected.selected)
        previous = scalar_metrics(selected.target_return, selected.previous_transferred_ensemble)
        assert len(selected) == row.common_holdout_rows
        assert np.isclose(scores["rmse"], row.holdout_rmse)
        assert np.isclose(scores["r2_vs_zero"], row.holdout_skill_vs_zero)
        assert np.isclose(scores["direction_accuracy"], row.direction_accuracy)
        assert np.isclose(1-scores["rmse"]/previous["rmse"], row.holdout_rmse_improvement_vs_previous)
        if hasattr(row, "retuned_previous_rmse"):
            retuned = scalar_metrics(selected.target_return, selected.previous_cv_ensemble)
            assert np.isclose(retuned["rmse"], row.retuned_previous_rmse)
            assert np.isclose(1-scores["rmse"]/retuned["rmse"], row.holdout_rmse_improvement_vs_retuned_previous)
        assert np.isclose((np.sign(selected.target_return) == selected.training_majority_direction).mean(), row.training_majority_accuracy)
        if hasattr(row, "always_up_accuracy"):
            assert np.isclose(selected.target_return.gt(0).mean(), row.always_up_accuracy)
    return {"status": "passed", "common_origin_rows": len(dates)}


def write_findings(report, path):
    all_cot = any(row.get("cot_policy") == "all" for row in report["experiments"])
    lines = ["# Independent MonthFu all-COT artifact audit" if all_cot else "# Independent MonthFu artifact audit", "",
             "This audit independently reconstructs monthly labels from the raw price calendar and recalculates saved scores. It checks split/refit maturity and replays the latest forecast from the saved final bundle. Passing these checks establishes artifact consistency and tested timing behavior; it does not certify every historical source vintage or prove future forecasting skill.", "",
             f"Audit status: **{report['status']}**. Completed experiments checked: {len(report['experiments'])}. Pending experiments: {', '.join(report['pending']) or 'none'}.", "",
             "| Experiment | Holdout rows | RMSE | Skill vs zero | RMSE improvement vs transferred old recipe | RMSE improvement vs monthly-retuned old components | Latest replay error |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in report["experiments"]:
        if row["status"] == "passed":
            replay = row["latest_bundle_replay_absolute_error"]
            lines.append(f"| {row['experiment']} | {row['holdout_rows']} | {row['rmse']:.3%} | {row['skill_vs_zero']:+.2%} | {row['rmse_improvement_vs_transferred']:+.2%} | {row['rmse_improvement_vs_retuned_previous']:+.2%} | {'below 1e-10' if replay is not None else 'not requested'} |")
        else:
            lines.append(f"| {row['experiment']} | failed: {row['error']} | | | | | |")
    primary = next((row for row in report["experiments"] if row["experiment"] == "30calendar" and row["status"] == "passed"), None)
    if primary:
        interval = primary["selected_minus_zero_interval"]
        retuned = primary["selected_minus_retuned_interval"]
        lines += ["", "## Interpretation of the measured primary result", "",
                  f"The primary winner is `{primary['selected_recipe']['weights']}`, using feature groups `{primary['selected_groups']}`.", ""]
        if primary["selected_groups"] == ["price"]:
            lines += ["The broad COT, weather and interaction candidates were evaluated, but the selected recipe contains only the engineered price feature group. Their availability improvements remain useful even though these extra sources did not enter the winning forecast.", ""]
        if primary.get("all_cot_dataset_audit"):
            audit = primary["all_cot_dataset_audit"]
            lines += [f"The all-COT policy retains all {audit['features']} numeric COT predictors in each eligible fit and final member. Its source audit contains all {audit['reports']} supported Coffee C reports, including {audit['retained_backcasts']} prelaunch disaggregated backcasts. The saved final training matrix has {audit['training_rows']} mature-label rows and the inference matrix has {audit['prediction_input_rows']} observed sessions. Backcasts first enter historical windows after their October 20, 2009 public release; legacy reports retain their earlier availability. Predictor inclusion does not require a model to assign nonzero importance.", ""]
        lines += [f"On all {primary['holdout_rows']} primary holdout origins, selected-minus-retuned-previous RMSE has a 60-session block interval [{retuned[0]*100:+.3f}, {retuned[1]*100:+.3f}] percentage points. The corresponding interval versus zero is [{interval[0]*100:+.3f}, {interval[1]*100:+.3f}] percentage points, which {'includes zero' if interval[0] <= 0 <= interval[1] else 'does not include zero'}. Negative intervals favor the selected recipe; an interval containing zero leaves the improvement uncertain.", "",
                  f"Primary direction accuracy is {primary['selected_direction_accuracy']:.2%}, compared with {primary['majority_direction_accuracy']:.2%} for the training-only majority benchmark. The empirical nominal 80% band covers {primary['empirical_band_coverage']:.2%} of primary holdout returns{' and falls below nominal coverage' if primary['empirical_band_coverage'] < .8 else ''}. Latest origin is {primary['latest_origin']}, which is the cache date rather than today's market date.", "",
                  "Lower absolute error at a shorter horizon cannot by itself identify a better forecast. The common-origin horizon table can use a smaller shared sample than the full per-horizon table above; do not mix those scopes when reporting gains."]
        negative = [row["experiment"] for row in report["experiments"] if row["status"] == "passed" and row["skill_vs_zero"] < 0]
        if negative:
            lines += ["", f"Experiments failing the zero-return point benchmark: `{negative}`."]
        if primary.get("checkpoint_export"):
            lines += ["", "Training completed its CSV evaluation before strict JSON export rejected undefined neutral-forecast correlations. Recovery preserved the already selected recipe and all saved CV/holdout predictions, then regenerated reports and final inference bundles. This audit checked the recovered labels, recipes, scores, maturity boundaries and bundle replay independently. Some non-selected Elastic Net fits reached their iteration limit; no winning recipe includes Elastic Net."]
    lines += ["", "## Verification scope", "", "Checks performed for every completed experiment:", "",
              "- Raw-price origin, target close, target date and return reconstruction for CV and holdout rows.",
              "- Scalar JSON/CSV metrics, every candidate/fold score, CV ranking and frozen recipe weights.",
              "- Exact training rows and label endpoints before every CV/refit boundary; mature historical means and direction benchmarks.",
              "- Year-specific and true-date non-overlapping metrics, empirical error-band construction and observed coverage.",
              "- Paired 60-session moving-block RMSE intervals independently reconstructed, with 30/90-session zero-reference sensitivity.",
              "- COT publication/first-usable-session chronology, raw input hashes and unchanged core training/inference source hashes.",
              "- Full availability-row calendar, nonnegative source ages, final bundle feature membership and imputation medians from mature fitting rows.",
              "- Latest final-bundle prediction replay; historical holdout scores use saved out-of-sample rows rather than final refitted weights.", "",
              f"Cross-horizon comparison audit: `{report['horizon_comparison']}`.", "",
              "The 30-calendar-day target remains the fixed primary experiment. Alternatives are sensitivity studies. The historical holdout was reused in earlier project research; daily monthly labels overlap, and current news retention/reanalysis/COT release-estimate limitations remain. Prior baselines adapt old methods and feature families rather than reproduce the original saved five-day model. Read moving-block intervals and yearly results before calling a point gain reliable.", "",
              "Re-run from the repository root:", "", "```sh", report.get("rerun_command", ".venv/bin/python MonthFu/scripts/audit_artifacts.py"), "```", ""]
    path.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", type=Path, default=MONTHFU / "artifacts")
    parser.add_argument("--skip-bundle-replay", action="store_true")
    args = parser.parse_args()
    artifact_dir = args.artifacts_dir.resolve()
    if not artifact_dir.is_relative_to(MONTHFU):
        raise ValueError("Keep artifact audit outputs inside MonthFu")
    prices = read_prices()
    reports, summaries, predictions, cv_dates, pending = [], {}, {}, [], []
    required = ["metrics.json", "model.joblib", "holdout_predictions.csv", "report.md", "holdout.png"]
    for key in EXPECTED_EXPERIMENTS:
        folder = artifact_dir / key
        if not all((folder/file).exists() for file in required):
            pending.append(key)
            continue
        try:
            result, cv, holdout = verify_experiment(folder, prices, replay=not args.skip_bundle_replay)
            reports.append(result)
            summaries[key], predictions[key] = result, holdout
            cv_dates.append(list(cv.Date))
            print(f"{key}: audit passed; latest bundle replay error {result['latest_bundle_replay_absolute_error']}", flush=True)
        except Exception as error:
            reports.append({"experiment": key, "status": "failed", "error": str(error)})
            print(f"{key}: audit FAILED: {error}", flush=True)
    failures = any(row["status"] == "failed" for row in reports)
    comparison = "not yet generated"
    if predictions:
        try:
            assert all(dates == cv_dates[0] for dates in cv_dates[1:]), "CV origins differ across horizons"
            comparison = verify_horizon_table(artifact_dir, summaries, predictions)
        except Exception as error:
            failures = True
            comparison = {"status": "failed", "error": str(error)}
    relative_artifacts = str(artifact_dir.relative_to(ROOT))
    rerun_command = ".venv/bin/python MonthFu/scripts/audit_artifacts.py"
    if artifact_dir != MONTHFU / "artifacts":
        rerun_command += " --artifacts-dir " + relative_artifacts
    report = {"status": "failed" if failures else "partial" if pending else "passed",
              "artifacts_dir": relative_artifacts, "rerun_command": rerun_command,
              "experiments": reports, "pending": pending, "horizon_comparison": comparison}
    (artifact_dir / "independent_audit.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    findings_path = (MONTHFU / "docs/INDEPENDENT_AUDIT.md" if artifact_dir == MONTHFU / "artifacts"
                     else MONTHFU / "docs/ALL_COT_AUDIT.md" if artifact_dir == MONTHFU / "artifacts/all_cot"
                     else artifact_dir / "independent_audit.md")
    write_findings(report, findings_path)
    if failures:
        raise SystemExit(1)
    if not reports:
        print("No completed experiments yet; written audit records pending work.", flush=True)


if __name__ == "__main__":
    main()
