"""Independently check saved direction experiments without their metric helpers.

This recomputes confusion/precision/recall, labels from raw closes, original
MonthFu comparisons and chronological fit/selection boundaries. It deliberately
does not replay a final refit on historical holdout rows.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import joblib

ROOT = Path(__file__).resolve().parents[3]
EXPERIMENTS = ROOT / "MonthFu" / "experiments"
CLASS_KEYS = ["rows", "accuracy", "balanced_accuracy", "macro_f1", "up_precision", "up_recall",
              "up_f1", "down_precision", "down_recall", "down_f1", "tp", "tn", "fp", "fn",
              "support_up", "support_down", "predicted_up", "predicted_down", "pr_auc_up",
              "pr_auc_down", "roc_auc", "brier", "log_loss"]


def _divide(first, second):
    return float(first / second) if second else 0.


def _average_precision(actual, score):
    """Integrate precision against positive recall increments, respecting ties."""
    positives = int(np.sum(actual))
    if not positives:
        return None
    order = np.argsort(-score, kind="stable")
    truth, sorted_score = actual[order], score[order]
    last_at_score = np.r_[np.flatnonzero(np.diff(sorted_score)), len(actual) - 1]
    true_positives = np.cumsum(truth)[last_at_score]
    precision = true_positives / (last_at_score + 1)
    positive_increments = np.diff(np.r_[0., true_positives]) / positives
    return float(np.sum(precision * positive_increments))


def independent_metrics(actual, probability, threshold=.5):
    y, p = np.asarray(actual, dtype=int), np.asarray(probability, dtype=float)
    assert len(y) and y.shape == p.shape and np.isin(y, [0, 1]).all()
    assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    assert np.isfinite(threshold) and 0 <= threshold <= 1
    predicted = p >= threshold
    tp = int(sum((y == 1) & predicted))
    tn = int(sum((y == 0) & ~predicted))
    fp = int(sum((y == 0) & predicted))
    fn = int(sum((y == 1) & ~predicted))
    up_recall, down_recall = _divide(tp, tp + fn), _divide(tn, tn + fp)
    up_f1, down_f1 = _divide(2 * tp, 2 * tp + fp + fn), _divide(2 * tn, 2 * tn + fp + fn)
    # Mann-Whitney form of ROC AUC via tied average score ranks.
    ranks = pd.Series(p).rank(method="average").to_numpy()
    positives, negatives = int(y.sum()), int((1 - y).sum())
    auc = ((ranks[y == 1].sum() - positives * (positives + 1) / 2) /
           (positives * negatives)) if positives and negatives else None
    clipped = np.clip(p, np.finfo(float).eps, 1 - np.finfo(float).eps)
    return {"rows": len(y), "accuracy": (tp + tn) / len(y),
            "balanced_accuracy": (up_recall + down_recall) / 2,
            "macro_f1": (up_f1 + down_f1) / 2,
            "up_precision": _divide(tp, tp + fp), "up_recall": up_recall, "up_f1": up_f1,
            "down_precision": _divide(tn, tn + fn), "down_recall": down_recall, "down_f1": down_f1,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "support_up": positives, "support_down": negatives,
            "predicted_up": tp + fp, "predicted_down": tn + fn,
            "pr_auc_up": _average_precision(y, p),
            "pr_auc_down": _average_precision(1 - y, 1 - p),
            "roc_auc": float(auc) if auc is not None else None,
            "brier": float(((y - p) ** 2).mean()),
            "log_loss": float(-(y * np.log(clipped) + (1 - y) * np.log1p(-clipped)).mean())}


def independent_selective(actual, probability, lower, upper):
    y, p = np.asarray(actual, dtype=int), np.asarray(probability, dtype=float)
    assert 0 <= lower <= upper <= 1
    accepted = (p < lower) | (p >= upper)
    if accepted.any():
        result = independent_metrics(y[accepted], p[accepted], upper)
    else:
        result = {key: 0. for key in CLASS_KEYS}
        result["accuracy"] = result["balanced_accuracy"] = None
    for key in ["pr_auc_up", "pr_auc_down", "roc_auc", "brier", "log_loss"]:
        result.pop(key, None)
    result.update({"rows": len(y), "accepted_rows": int(accepted.sum()),
                   "abstained_rows": int((~accepted).sum()), "coverage": float(accepted.mean()),
                   "abstained_up": int(((y == 1) & ~accepted).sum()),
                   "abstained_down": int(((y == 0) & ~accepted).sum()),
                   "support_up_all": int((y == 1).sum()),
                   "support_down_all": int((y == 0).sum())})
    for label, value, correct in [("up", 1, result["tp"]), ("down", 0, result["tn"])]:
        total = int((y == value).sum())
        result[f"{label}_unconditional_recall"] = _divide(correct, total)
        result[f"{label}_class_coverage"] = _divide(((y == value) & accepted).sum(), total)
    return result


def compare_scores(expected, saved, label, keys=CLASS_KEYS):
    checked = []
    for key in keys:
        if key not in saved or key not in expected:
            continue
        actual = saved[key]
        if expected[key] is None:
            assert actual is None or pd.isna(actual), f"{label}: {key} should be absent"
        elif actual is None or pd.isna(actual):
            # Return-direction scores deliberately omit proper probability losses.
            assert key in {"brier", "log_loss"}, f"{label}: missing {key}"
        else:
            assert np.isclose(expected[key], actual, rtol=1e-6, atol=1e-8), (
                f"{label}: {key}: expected {expected[key]}, saved {actual}")
        checked.append(key)
    assert checked, f"{label}: no comparable saved metrics"
    return len(checked)


def raw_labels(horizon, unit):
    frame = pd.read_csv(ROOT / "data/yahoo/arabica_coffee_futures_history.csv")
    frame["Date"] = pd.to_datetime(frame.Date, errors="coerce").dt.normalize()
    frame["Close"] = pd.to_numeric(frame.Close, errors="coerce")
    frame = frame.loc[frame.Date.notna() & np.isfinite(frame.Close) & frame.Close.gt(0)]
    frame = frame.sort_values("Date").reset_index(drop=True)
    assert not frame.Date.duplicated().any()
    dates = pd.DatetimeIndex(frame.Date)
    endpoints = (dates.searchsorted(dates + pd.Timedelta(days=horizon)) if unit == "calendar"
                 else np.arange(len(frame)) + horizon)
    mature = endpoints < len(frame)
    frame["target_end_date"] = pd.NaT
    frame["target_return"] = np.nan
    frame.loc[mature, "target_end_date"] = dates.take(endpoints[mature]).to_numpy()
    frame.loc[mature, "target_return"] = frame.Close.to_numpy()[endpoints[mature]] / frame.Close.to_numpy()[mature] - 1
    return frame[["Date", "Close", "target_end_date", "target_return"]]


def check_targets(saved, raw, label):
    assert saved.Date.is_monotonic_increasing and not saved.Date.duplicated().any(), label
    matched = saved.merge(raw, on="Date", suffixes=("_saved", "_raw"), validate="one_to_one")
    assert len(matched) == len(saved), f"{label}: an origin is missing from raw prices"
    assert matched.target_end_date_saved.eq(matched.target_end_date_raw).all(), f"{label}: wrong actual endpoint"
    np.testing.assert_allclose(matched.target_return_saved, matched.target_return_raw,
                               rtol=1e-10, atol=1e-11, err_msg=label)


def _check_selection_windows(value, cutoff, checked=None):
    """Recursively find threshold/band/calibration selection audits."""
    checked = [] if checked is None else checked
    if isinstance(value, dict):
        if "selection_end" in value:
            assert pd.Timestamp(value["selection_end"]) < cutoff, "Selection used holdout origins"
            if "holdout_start" in value:
                assert pd.Timestamp(value["holdout_start"]) == cutoff, "Selection cutoff differs"
            if "selection_source" in value:
                assert value["selection_source"] == "pre_holdout_oof"
            checked.append(value.get("objective", "selection"))
        for child in value.values():
            _check_selection_windows(child, cutoff, checked)
    elif isinstance(value, list):
        for child in value:
            _check_selection_windows(child, cutoff, checked)
    return checked


def _disjoint_positions(frame, offset=0):
    result, last_end = [], pd.Timestamp.min
    for position in range(offset, len(frame)):
        row = frame.iloc[position]
        if row.Date >= last_end:
            result.append(position)
            last_end = row.target_end_date
    return result


def _score_probability(frame, column, score_type="probability"):
    values = frame[column].to_numpy(dtype=float)
    if score_type in {"regression", "return", "reg", "uncalibrated_return_rank_score"}:
        # Fixed scale matching the documented regression_direction_score.
        return 1 / (1 + np.exp(-np.clip(values / .10, -700, 700)))
    return values


def _probability_for_row(frame, row):
    name = row["model"]
    if name == "always_up":
        return np.ones(len(frame))
    if name == "always_down":
        return np.zeros(len(frame))
    if name == "baseline_majority":
        return frame.baseline_majority_probability.to_numpy()
    assert name in frame, f"Unknown saved prediction column: {name}"
    return _score_probability(frame, name, row["score_type"])


def independent_regression(actual, prediction):
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(prediction, dtype=float)
    mse = float(np.square(actual - predicted).mean())
    reference = float(np.square(actual).mean())
    return {"rows": len(actual), "rmse": np.sqrt(mse), "mae": np.abs(actual - predicted).mean(),
            "direction_accuracy": (np.sign(actual) == np.sign(predicted)).mean(),
            "correlation": np.corrcoef(actual, predicted)[0, 1] if np.std(predicted) > 1e-12 and np.std(actual) > 0 else None,
            "r2_vs_zero": 1 - mse / reference if reference else None,
            "mean_prediction": predicted.mean(), "mean_actual": actual.mean()}


def _audit_tables(frame, class_table, reg_table, label):
    direction = frame.loc[frame.target_return.ne(0)]
    y = direction.target_return.gt(0).astype(int).to_numpy()
    checked = 0
    for row in class_table.to_dict(orient="records"):
        probability = _probability_for_row(direction, row)
        checked += compare_scores(independent_metrics(y, probability, float(row["threshold"])),
                                  row, label + " " + row["model"])
        assert row["excluded_neutral_rows"] == len(frame) - len(direction)
        if row["task"] == "reg":
            assert pd.isna(row["brier"]) and pd.isna(row["log_loss"])
    for row in reg_table.to_dict(orient="records"):
        expected = independent_regression(frame.target_return, frame[row["model"]])
        checked += compare_scores(expected, row, label + " regression " + row["model"], list(expected))
    return checked


def _audit_decisions(selection, cv, recipes):
    direction = cv.loc[cv.target_return.ne(0)]
    y = direction.target_return.gt(0).astype(int).to_numpy()
    decisions = selection["threshold_decisions"]
    checked = 0
    for name, decision in decisions.items():
        probability = direction[name].to_numpy()
        assert decision["selection_rows"] == len(direction)
        assert pd.Timestamp(decision["selection_start"]) == direction.Date.min()
        assert pd.Timestamp(decision["selection_end"]) == direction.Date.max()
        table = []
        for row in decision["search"]:
            expected = independent_metrics(y, probability, row["threshold"])
            checked += compare_scores(expected, row, "Threshold search " + name)
            table.append({**expected, "threshold": row["threshold"]})
        def rank(row):
            return (row["macro_f1"], min(row["up_recall"], row["down_recall"]),
                    row["balanced_accuracy"], -abs(row["up_recall"] - row["down_recall"]),
                    -abs(row["threshold"] - .5), -row["threshold"])
        winner = max(table, key=rank)
        assert winner["threshold"] == decision["threshold"], "Wrong development threshold"
        checked += compare_scores(winner, decision["selection_metrics"], "Threshold selected " + name)
    for name, recipe in recipes.items():
        if recipe["task"] == "class":
            candidate = next(iter(recipe["weights"])) if len(recipe["weights"]) == 1 else recipe["name"]
            assert recipe["threshold"] == decisions[candidate]["threshold"], "Recipe threshold changed"
    band = selection["selected_abstention"]
    assert band["selection_rows"] == len(direction)
    candidates = []
    for row in band["search"]:
        expected = independent_selective(y, direction.selected_direction, row["lower"], row["upper"])
        checked += compare_scores(expected, row, "Abstention development search", list(expected))
        passes = bool(row["coverage"] >= band["min_coverage"] and
                      min(row["up_class_coverage"], row["down_class_coverage"]) >= band["min_class_coverage"] and
                      min(row["up_unconditional_recall"], row["down_unconditional_recall"]) >= band["min_unconditional_recall"] and
                      min(row["predicted_up"], row["predicted_down"]) >= band["min_predicted_per_class"])
        assert row["gates_passed"] == passes, "Wrong confidence-band coverage gate"
        if passes:
            candidates.append(row)
    def rank_band(row):
        return (row["macro_f1"], min(row["up_precision"], row["down_precision"]),
                min(row["up_unconditional_recall"], row["down_unconditional_recall"]),
                row["coverage"], -row["width"])
    winner = max(candidates, key=rank_band) if candidates else band["search"][0]
    assert winner["lower"] == band["lower"] and winner["upper"] == band["upper"]
    assert winner["gates_passed"] == band["gates_passed"]
    return checked


def _audit_recipe_selection(selection, cv, definitions):
    """Verify family parameters and overall winners came from OOF rankings."""
    recipes = selection["recipes"]
    binary = cv.loc[cv.target_return.ne(0)]
    y = binary.target_return.gt(0).astype(int).to_numpy()
    scores = {}
    for name, spec in definitions.items():
        if spec["task"] == "class":
            threshold = selection["threshold_decisions"][name]["threshold"]
            scores[name] = independent_metrics(y, binary[name], threshold)
        else:
            scores[name] = independent_regression(cv.target_return, cv[name])
    class_order = lambda name: (-scores[name]["macro_f1"], -scores[name]["balanced_accuracy"], name)
    reg_order = lambda name: (scores[name]["rmse"], name)
    for name, recipe in recipes.items():
        if name in {"selected_direction", "selected_return"} or name.endswith("equal_ensemble"):
            continue
        task, group, family = name.split("__")
        pool = [candidate for candidate, spec in definitions.items()
                if spec["task"] == task and spec["group"] == group and spec["family"] == family]
        winner = min(pool, key=class_order if task == "class" else reg_order)
        assert recipe["weights"] == {winner: 1.}, "A family variant was chosen outside development scores"
    deployable = {}
    for name, recipe in recipes.items():
        if name in {"selected_direction", "selected_return"}:
            continue
        candidate = next(iter(recipe["weights"])) if len(recipe["weights"]) == 1 else name
        deployable[candidate] = recipe
        if candidate not in scores:
            scores[candidate] = (independent_metrics(y, binary[candidate], recipe["threshold"])
                                 if recipe["task"] == "class" else
                                 independent_regression(cv.target_return, cv[candidate]))
    for task, alias, order in [("class", "selected_direction", class_order), ("reg", "selected_return", reg_order)]:
        pool = [name for name, recipe in deployable.items() if recipe["task"] == task]
        winner = min(pool, key=order)
        assert recipes[alias] == deployable[winner], "Overall deployment winner differs from development scores"
    return len(definitions)


def _independent_bootstrap(y, selected, baseline, selected_cut, block, draws, metric_names, baseline_cut=.5):
    length = min(block, len(y))
    rng = np.random.default_rng(42)
    selected_up, baseline_up = selected >= selected_cut, baseline >= baseline_cut
    samples = {name: [] for name in metric_names}
    # Confusion counts are sufficient for these fixed-threshold rate metrics.
    def rates(truth, up):
        tp, tn = int(((truth == 1) & up).sum()), int(((truth == 0) & ~up).sum())
        fp, fn = int(((truth == 0) & up).sum()), int(((truth == 1) & ~up).sum())
        return {"macro_f1": (_divide(2 * tp, 2 * tp + fp + fn) + _divide(2 * tn, 2 * tn + fp + fn)) / 2,
                "accuracy": (tp + tn) / len(truth),
                "balanced_accuracy": (_divide(tp, tp + fn) + _divide(tn, tn + fp)) / 2,
                "up_precision": _divide(tp, tp + fp), "up_recall": _divide(tp, tp + fn),
                "down_precision": _divide(tn, tn + fn), "down_recall": _divide(tn, tn + fp)}
    for _ in range(draws):
        starts = rng.integers(0, len(y) - length + 1, size=int(np.ceil(len(y) / length)))
        positions = (starts[:, None] + np.arange(length)).ravel()[:len(y)]
        first, second = rates(y[positions], selected_up[positions]), rates(y[positions], baseline_up[positions])
        for name in metric_names:
            samples[name].append(first[name] - second[name])
    return {name: np.quantile(sample, [.025, .975]) for name, sample in samples.items()}


def _audit_final_replay(folder, bundle, latest_saved, final_cutoff):
    """Replay only the latest cached origin; verify train-only transform state."""
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(EXPERIMENTS / "src"))
    # Inference feature reconstruction intentionally exercises the delivered
    # feature code. Target/metric/selection checks above remain independent.
    from direction_features import build_experiment_frame
    frame, groups, metadata = build_experiment_frame(ROOT, bundle["horizon"], bundle["unit"],
                                                    cot_policy=bundle.get("cot_policy", "benchmark"))
    if bundle.get("cot_policy") == "all":
        from MonthFu.experiments.scripts.predict import validate_source_schema
        validate_source_schema(bundle, frame, metadata)
    training = frame.loc[frame.Date.lt(final_cutoff) & frame.target_end_date.lt(final_cutoff) & frame.target_return.notna()]
    latest_origin = frame.tail(1)
    member_predictions = {}
    for name, member in bundle["members"].items():
        train = training.loc[training.target_return.ne(0)] if member["spec"]["task"] == "class" else training
        selected = [column for column in groups[member["spec"]["group"]]
                    if (member["spec"].get("require_all_cot", False) and column.startswith("cot_"))
                    or (train[column].notna().mean() >= .2 and train[column].nunique(dropna=True) > 1)]
        assert member["features"] == selected, "Final feature screening differs from training-only values"
        values = train[selected].to_numpy(dtype=float)
        # sklearn keep_empty_features represents a completely unknown fitting
        # column with zero plus its missingness flag, not a source observation.
        medians = np.zeros(values.shape[1])
        nonempty = ~np.isnan(values).all(axis=0)
        medians[nonempty] = np.nanmedian(values[:, nonempty], axis=0)
        imputer = member["estimator"].named_steps["impute"]
        np.testing.assert_allclose(imputer.statistics_, medians, rtol=1e-10, atol=1e-11)
        missing = np.isnan(values)
        indicators = np.flatnonzero(missing.any(axis=0))
        np.testing.assert_array_equal(imputer.indicator_.features_, indicators)
        filled = np.where(missing, medians, values)
        with_indicators = np.column_stack([filled, missing[:, indicators]])
        scaler = member["estimator"].named_steps["scale"]
        np.testing.assert_allclose(scaler.mean_, with_indicators.mean(axis=0), rtol=1e-9, atol=1e-10)
        np.testing.assert_allclose(scaler.var_, with_indicators.var(axis=0), rtol=1e-8, atol=1e-10)
        assert scaler.n_samples_seen_ == len(train)
        pipeline = member["estimator"]
        inputs = latest_origin[member["features"]]
        prediction = (pipeline.predict_proba(inputs)[:, 1] if member["spec"]["task"] == "class"
                      else pipeline.predict(inputs))
        # Preserve each learner's output dtype. XGBoost can return float32;
        # weighting its array rounds before addition to float64 linear/forest
        # arrays, exactly as in the delivered ensemble inference path.
        member_predictions[name] = prediction
    for row in latest_saved.to_dict(orient="records"):
        recipe = bundle["recipes"][row["model"]]
        replayed = float(sum(weight * member_predictions[name] for name, weight in recipe["weights"].items())[0])
        assert np.isclose(replayed, row["value"], rtol=1e-8, atol=1e-10), "Latest saved forecast replay differs"
        prediction_cutoff = recipe["threshold"] if recipe["task"] == "class" else 0.
        expected_direction = "Up" if replayed >= prediction_cutoff else "Down"
        assert row["direction"] == expected_direction
        assert np.isclose(row["threshold"], prediction_cutoff)
    return {"latest_forecasts_replayed": len(latest_saved),
            "train_only_preprocessors_checked": len(bundle["members"]),
            "latest_inference_feature_check": "Delivered causal feature code rebuilt; metrics/targets independently checked"}


def _audit_precision_recall_policies(folder, cv, holdout, original_selection):
    selection_path = folder / "precision_recall_policy_selection.json"
    if not selection_path.exists():
        return {}
    selection = json.loads(selection_path.read_text())
    cutoff = pd.Timestamp(selection["holdout_start"])
    assert selection["selection_source"] == "mature_pre_holdout_oof_only"
    assert pd.Timestamp(selection["selection_end"]) < cutoff
    assert pd.Timestamp(selection["latest_selection_label_end"]) < cutoff
    assert pd.Timestamp(selection["selection_start"]) == cv.Date.min()
    assert pd.Timestamp(selection["selection_end"]) == cv.Date.max()
    assert pd.Timestamp(selection["latest_selection_label_end"]) == cv.target_end_date.max()
    for name, digest in selection["input_sha256"].items():
        assert hashlib.sha256((folder / name).read_bytes()).hexdigest() == digest
    retained = selection["retained_recipes"]
    assert len(retained) == selection["candidate_recipe_count"]
    signatures = []
    for name, recipe in retained.items():
        assert recipe["weights"] == original_selection["recipes"][name]["weights"]
        signature = tuple(sorted(recipe["weights"].items()))
        assert signature not in signatures, "Equivalent recipes were counted more than once"
        signatures.append(signature)
    binary = cv.loc[cv.target_return.ne(0)]
    y = binary.target_return.gt(0).astype(int).to_numpy()
    grid = pd.read_csv(folder / "precision_recall_policy_cv_grid.csv")
    assert len(grid) == selection["grid_rows"] == len(retained) * len(selection["thresholds"])
    assert selection["selection_rows"] == len(binary)
    checked = 0
    rows = grid.to_dict(orient="records")
    for row in rows:
        expected = independent_metrics(y, binary[row["recipe"]], row["threshold"])
        checked += compare_scores(expected, row, "Precision/recall development grid")
        assert np.isclose(row["minimum_precision"], min(expected["up_precision"], expected["down_precision"]))
        assert np.isclose(row["minimum_recall"], min(expected["up_recall"], expected["down_recall"]))
    for policy, frozen in selection["policies"].items():
        if policy == "precision_first":
            eligible = [row for row in rows if row["minimum_recall"] >= frozen["minimum_required_recall"]]
            primary = "minimum_precision"
        else:
            eligible = rows
            primary = "minimum_recall"
        def rank(row):
            return (row[primary], row["macro_f1"], -abs(row["threshold"] - .5), -row["threshold"])
        if eligible:
            best = max(eligible, key=rank)
            assert frozen["gates_passed"]
            assert frozen["recipe"] == best["recipe"] and np.isclose(frozen["threshold"], best["threshold"])
        else:
            fallback = selection["original_direction_policy"]
            assert not frozen["gates_passed"]
            assert frozen["recipe"] == fallback["recipe"] and np.isclose(frozen["threshold"], fallback["threshold"])
        assert frozen["weights"] == retained[frozen["recipe"]]["weights"]
    metrics_path = folder / "precision_recall_policy_metrics.csv"
    if metrics_path.exists():
        binary_holdout = holdout.loc[holdout.target_return.ne(0)]
        labels = binary_holdout.target_return.gt(0).astype(int).to_numpy()
        table = pd.read_csv(metrics_path)
        for row in table.to_dict(orient="records"):
            expected = independent_metrics(labels, binary_holdout[row["recipe"]], row["threshold"])
            checked += compare_scores(expected, row, "Precision/recall holdout " + row["policy"])
            assert row["coverage"] == 1. and row["excluded_neutral_rows"] == len(holdout) - len(binary_holdout)
            if row["policy"] in selection["policies"]:
                assert np.isclose(row["threshold"], selection["policies"][row["policy"]]["threshold"])
    else:
        table = pd.DataFrame()
    uncertainty_path = folder / "precision_recall_policy_block_uncertainty.json"
    block_comparisons = 0
    if uncertainty_path.exists():
        uncertainty = json.loads(uncertainty_path.read_text())
        reference = selection["original_direction_policy"]
        assert uncertainty["reference"] == reference
        binary_holdout = holdout.loc[holdout.target_return.ne(0)]
        labels = binary_holdout.target_return.gt(0).astype(int).to_numpy()
        reference_probability = binary_holdout[reference["recipe"]].to_numpy()
        reference_score = independent_metrics(labels, reference_probability, reference["threshold"])
        for policy, comparisons in uncertainty["policies"].items():
            frozen = selection["policies"][policy]
            probability = binary_holdout[frozen["recipe"]].to_numpy()
            score = independent_metrics(labels, probability, frozen["threshold"])
            for block, result in comparisons.items():
                assert result["block_sessions"] >= holdout.target_sessions.max()
                intervals = _independent_bootstrap(labels, probability, reference_probability,
                                                  frozen["threshold"], int(block), result["bootstrap_draws"],
                                                  result["metrics"], baseline_cut=reference["threshold"])
                for name, metric in result["metrics"].items():
                    assert np.isclose(metric["difference"], score[name] - reference_score[name])
                    np.testing.assert_allclose(intervals[name], metric["difference_95pct_block_interval"], rtol=1e-8, atol=1e-10)
                block_comparisons += 1
    return {"precision_recall_policy_values_checked": checked,
            "precision_recall_policy_holdout_rows_checked": len(table),
            "precision_recall_policy_block_comparisons_checked": block_comparisons}


def _audit_all_cot_inclusion(folder, bundle, fits, independent_labels=None):
    """Check complete input membership and exported observations, not importance."""
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(EXPERIMENTS / "src"))
    from direction_features import build_experiment_frame, cot_feature_columns, cot_schema_hash
    frame, groups, metadata = build_experiment_frame(ROOT, bundle["horizon"], bundle["unit"], cot_policy="all")
    required = cot_feature_columns(frame)
    expected = set(required)
    assert expected and expected == set(bundle["required_cot_features"])
    assert bundle["cot_feature_schema_sha256"] == cot_schema_hash(required)
    assert all(expected.issubset(columns) for columns in groups.values())
    inclusion = json.loads((folder/"cot_inclusion_audit.json").read_text())
    assert inclusion["required_policy"] and inclusion["all_required_fields_retained_every_fit"]
    assert inclusion["fit_count"] == len(fits)
    caches, empty_fields = {}, set()
    for fit in fits:
        assert expected == set(fit["required_cot_features"])
        assert expected.issubset(fit["features"])
        task = "class" if fit["candidate"].startswith("class__") else "reg"
        key = (fit["cutoff"], task)
        if key not in caches:
            cutoff = pd.Timestamp(fit["cutoff"])
            rows = frame.loc[frame.Date.lt(cutoff) & frame.target_end_date.lt(cutoff) & frame.target_return.notna()]
            if task == "class":
                rows = rows.loc[rows.target_return.ne(0)]
            caches[key] = (len(rows), rows[required].count(), rows[required].nunique(dropna=True))
        count, observed, unique = caches[key]
        columns = {row["feature"]: row for row in fit["cot_inclusion_audit"]}
        assert set(columns) == expected
        for name, row in columns.items():
            assert row["included"] and row["required"]
            assert row["observed_rows"] == observed[name]
            assert row["unique_observed_values"] == unique[name]
            assert np.isclose(row["observed_fraction"], observed[name]/count)
            assert row["empty_training_column"] == (observed[name] == 0)
            if row["empty_training_column"]:
                empty_fields.add(name)
    cutoff = frame.Date.max() + pd.Timedelta(days=1)
    expected_training = frame.loc[frame.Date.lt(cutoff) & frame.target_end_date.lt(cutoff) & frame.target_return.notna()]
    training = pd.read_csv(folder/"training_dataset.csv.gz", parse_dates=["Date", "target_end_date"])
    origins = pd.read_csv(folder/"features_and_targets.csv.gz", parse_dates=["Date", "target_end_date"])
    assert training.Date.tolist() == expected_training.Date.tolist()
    assert origins.Date.tolist() == frame.Date.tolist()
    assert expected.issubset(training) and expected.issubset(origins)
    assert training.target_end_date.lt(cutoff).all()
    if independent_labels is not None:
        check_targets(training, independent_labels, "exported training dataset")
        check_targets(origins.loc[origins.target_return.notna()], independent_labels, "exported labelled origins")
    np.testing.assert_allclose(training[required].to_numpy(dtype=float),
                               expected_training[required].to_numpy(dtype=float),
                               rtol=1e-10, atol=1e-11, equal_nan=True)
    np.testing.assert_allclose(origins[required].to_numpy(dtype=float), frame[required].to_numpy(dtype=float),
                               rtol=1e-10, atol=1e-11, equal_nan=True)
    dataset_audit = json.loads((folder/"training_dataset_audit.json").read_text())
    assert dataset_audit["origin_rows"] == len(frame)
    assert dataset_audit["latest_mature_training_rows"] == len(training)
    assert set(dataset_audit["cot_features"]) == expected
    assert all(dataset_audit["groups_include_entire_cot_bank"].values())
    source = pd.read_csv(ROOT/"data/COT/coffee_c_all_cot_data.csv", usecols=["source_dataset", "Report_Date_as_MM_DD_YYYY"])
    assert metadata["cot"]["reports"] == len(source), "A cached COT report was omitted from the source audit"
    pre_2009 = training.loc[training.Date.lt("2009-01-01")]
    legacy = [name for name in required if name.startswith("cot_legacy_")]
    assert dataset_audit["pre_2009_training_rows"] == len(pre_2009)
    assert dataset_audit["pre_2009_observed_legacy_rows"] == int(pre_2009[legacy].notna().any(axis=1).sum())
    return {"all_cot_source_reports_checked": len(source), "all_cot_required_features_per_fit": len(required),
        "all_cot_fit_inclusion_checked": len(fits), "all_cot_empty_training_fields": sorted(empty_fields),
        "all_cot_training_dataset_rows_checked": len(training), "all_cot_origin_rows_checked": len(origins),
        "all_cot_pre_2009_training_rows": len(pre_2009),
        "all_cot_input_note": "Complete input inclusion verified; constant/empty inputs need not influence learned forecasts."}


def audit_folder(folder):
    selection = json.loads((folder / "selection.json").read_text())
    metadata_path = folder / "metrics.json"
    summary = json.loads(metadata_path.read_text()) if metadata_path.exists() else selection
    horizon = int(summary.get("horizon", selection.get("horizon", 30)))
    unit = summary.get("horizon_unit", summary.get("unit", selection.get("horizon_unit", "calendar")))
    cutoff = pd.Timestamp(selection.get("holdout_start", "2022-01-01"))
    raw = raw_labels(horizon, unit)
    holdout = pd.read_csv(folder / "holdout_predictions.csv", parse_dates=["Date", "target_end_date"])
    cv = pd.read_csv(folder / "cv_predictions.csv", parse_dates=["Date", "target_end_date"])
    check_targets(holdout, raw, "holdout")
    check_targets(cv, raw, "CV")
    assert holdout.Date.ge(cutoff).all()
    assert cv.Date.lt(cutoff).all() and cv.target_end_date.lt(cutoff).all()
    windows = _check_selection_windows(selection, cutoff)
    assert windows, "No explicit pre-holdout threshold/band selection audit found"
    keep = holdout.target_return.ne(0).to_numpy()
    direction = holdout.loc[keep].copy()
    y = direction.target_return.gt(0).astype(int).to_numpy()
    checks = {"horizon": horizon, "unit": unit, "holdout_rows": len(holdout),
              "binary_rows": len(direction), "neutral_rows_excluded": int((~keep).sum()),
              "cv_rows": len(cv), "selection_windows_checked": len(windows),
              "nonoverlap_rows_offset0": len(_disjoint_positions(holdout)),
              "metrics_checked": 0}
    recipes = selection["recipes"]
    class_table = pd.read_csv(folder / "holdout_classification_metrics.csv")
    reg_table = pd.read_csv(folder / "holdout_regression_metrics.csv")
    checks["metrics_checked"] += _audit_tables(holdout, class_table, reg_table, "holdout")
    checks["metrics_checked"] += _audit_tables(cv, pd.read_csv(folder / "cv_classification_metrics.csv"),
                                              pd.read_csv(folder / "cv_regression_metrics.csv"), "CV")
    checks["metrics_checked"] += _audit_decisions(selection, cv, recipes)
    # Reconstruct all development ensemble/alias forecasts directly from members.
    for name, recipe in recipes.items():
        weights = recipe["weights"]
        assert weights and np.isclose(sum(weights.values()), 1.) and all(weight > 0 for weight in weights.values())
        rebuilt = sum(cv[member] * weight for member, weight in weights.items())
        # Candidate columns can be float32 from XGBoost; their textual CSV
        # roundtrip is less precise than the float64 saved ensemble column.
        np.testing.assert_allclose(rebuilt, cv[name], rtol=1e-7, atol=1e-8)
        if name.endswith("equal_ensemble"):
            assert len(weights) == 3 and all(np.isclose(weight, 1 / 3) for weight in weights.values())
            assert {member.split("__")[-1].rsplit("_", 1)[0] for member in weights} == {"linear", "xgb", "forest"}
    band = selection["selected_abstention"]
    expected_selective = independent_selective(y, direction.selected_direction, band["lower"], band["upper"])
    saved_selective = json.loads((folder / "holdout_selective_metrics.json").read_text())
    checks["metrics_checked"] += compare_scores(expected_selective, saved_selective, "holdout selective", list(expected_selective))
    # Stored original forecasts provide an exact paired comparator, never a replay.
    prior_path = ROOT / "MonthFu" / "artifacts" / f"{horizon}{unit}" / "holdout_predictions.csv"
    if prior_path.exists():
        prior = pd.read_csv(prior_path, parse_dates=["Date", "target_end_date"])
        assert holdout.Date.equals(prior.Date), "New and prior holdout origin rows differ"
        assert holdout.target_end_date.equals(prior.target_end_date), "Prior target endpoints differ"
        np.testing.assert_allclose(holdout.target_return, prior.target_return, rtol=1e-10, atol=1e-11)
        checks["prior_exact_rows_matched"] = len(prior)
        for prior_name, saved_name in [("selected", "baseline_monthly_return"), ("previous_cv_ensemble", "baseline_paper_return")]:
            np.testing.assert_allclose(holdout[saved_name], prior[prior_name], rtol=1e-10, atol=1e-11)
            checks["prior_" + prior_name + "_copied_from_saved"] = True
        prior_cv = pd.read_csv(prior_path.with_name("cv_predictions.csv"), parse_dates=["Date", "target_end_date"])
        assert cv.Date.equals(prior_cv.Date) and cv.target_end_date.equals(prior_cv.target_end_date)
        assert cv.fold.equals(prior_cv.fold)
        np.testing.assert_allclose(cv.target_return, prior_cv.target_return, rtol=1e-10, atol=1e-11)
        for prior_name, saved_name in [("selected", "baseline_monthly_return"), ("previous_cv_ensemble", "baseline_paper_return")]:
            np.testing.assert_allclose(cv[saved_name], prior_cv[prior_name], rtol=1e-10, atol=1e-11)
        checks["prior_exact_cv_rows_matched"] = len(prior_cv)
    definitions = {row["name"]: row for row in json.loads((folder / "candidate_parameters.json").read_text())}
    checks["candidate_selection_scores_checked"] = _audit_recipe_selection(selection, cv, definitions)
    groups = json.loads((folder / "feature_groups.json").read_text())
    fits = json.loads((folder / "fit_audit.json").read_text())
    fitted_final = [row for row in fits if row["phase"] == "final"]
    assert len(fits) == summary["cv_fits"] + summary["holdout_fits"] + len(fitted_final)
    fitted_cv = [row for row in fits if row["phase"] == "cv"]
    fitted_holdout = [row for row in fits if row["phase"] == "holdout"]
    needed = {name for recipe in recipes.values() for name in recipe["weights"]}
    assert len(fitted_cv) == len(definitions) * cv.fold.nunique()
    assert len(fitted_holdout) == len(needed) * holdout.quarter.nunique()
    identities = {(row["phase"], row["cutoff"], row["candidate"]) for row in fits}
    assert len(identities) == len(fits), "Duplicate fit-audit record"
    for fold in cv.fold.unique():
        assert {row["candidate"] for row in fitted_cv if row["fold"] == fold} == set(definitions)
    for quarter in holdout.quarter.unique():
        assert {row["candidate"] for row in fitted_holdout if row["quarter"] == quarter} == needed
    for row in fits:
        start = pd.Timestamp(row["cutoff"])
        spec = definitions[row["candidate"]]
        train = raw.loc[raw.Date.lt(start) & raw.target_end_date.lt(start) & raw.target_return.notna()]
        if spec["task"] == "class":
            train = train.loc[train.target_return.ne(0)]
        assert len(train) == row["train_rows"], "Fit used unexpected mature training count"
        assert train.Date.max() == pd.Timestamp(row["train_last_date"]) < start
        assert train.target_end_date.max() == pd.Timestamp(row["train_last_target_date"]) < start
        assert set(row["features"]).issubset(groups[spec["group"]]), "Fit features exceed fixed group"
        assert not any(name.startswith(("target", "news_")) or "future_return" in name for name in row["features"])
        if row["phase"] == "holdout":
            quarter = holdout.loc[holdout.quarter.eq(row["quarter"])]
            assert quarter.Date.min() == start, "Refit cutoff differs from first quarterly forecast"
        elif row["phase"] == "cv":
            valid = cv.loc[cv.fold.eq(row["fold"])]
            assert valid.Date.min() == start, "CV fit cutoff differs from validation start"
        else:
            assert row["phase"] == "final", "Unknown fit-audit phase"
            assert start == raw.Date.max() + pd.Timedelta(days=1), "Final fit cutoff differs from cached origin"
    checks["fit_boundaries_and_counts_checked"] = len(fits)
    folds_path = folder / "cv_folds.csv"
    if folds_path.exists():
        folds = pd.read_csv(folds_path)
        for row in folds.to_dict(orient="records"):
            start = pd.Timestamp(row["validation_start"])
            for key in ["train_end", "latest_training_label_end", "train_last_date", "train_last_target_date"]:
                if key in row and pd.notna(row[key]):
                    assert pd.Timestamp(row[key]) < start, "CV training boundary leak"
            if "latest_validation_label_end" in row:
                assert pd.Timestamp(row["latest_validation_label_end"]) < cutoff
        checks["cv_folds_checked"] = len(folds)
    # Re-evaluate every year and every disjoint starting offset, on the exact rows.
    for year, rows in holdout.groupby(holdout.Date.dt.year):
        yearly_class = pd.read_csv(folder / "holdout_classification_by_year.csv")
        yearly_reg = pd.read_csv(folder / "holdout_regression_by_year.csv")
        checks["metrics_checked"] += _audit_tables(rows, yearly_class.loc[yearly_class.year.eq(year)],
                                                  yearly_reg.loc[yearly_reg.year.eq(year)], f"year {year}")
    disjoint = pd.read_csv(folder / "nonoverlapping_classification_metrics.csv")
    for offset, table in disjoint.groupby("offset"):
        rows = holdout.iloc[_disjoint_positions(holdout, int(offset))]
        checks["metrics_checked"] += _audit_tables(rows, table, pd.DataFrame(), f"nonoverlap {offset}")
    bootstrap = json.loads((folder / "direction_comparison_bootstrap.json").read_text())
    selected_cut = recipes["selected_direction"]["threshold"]
    selected_score = independent_metrics(y, direction.selected_direction, selected_cut)
    baseline_probability = _score_probability(direction, "baseline_monthly_return", "reg")
    baseline_score = independent_metrics(y, baseline_probability)
    for length, result in bootstrap.items():
        assert result["difference_sign"] == "positive_favors_selected"
        assert result["block_sessions"] >= holdout.target_sessions.max(), "Bootstrap block shorter than overlap span"
        intervals = _independent_bootstrap(y, direction.selected_direction.to_numpy(), baseline_probability,
                                          selected_cut, int(length), result["bootstrap_draws"], result["metrics"])
        for name, metric in result["metrics"].items():
            assert np.isclose(metric["difference"], selected_score[name] - baseline_score[name])
            np.testing.assert_allclose(intervals[name], metric["difference_95pct_block_interval"], rtol=1e-8, atol=1e-10)
    checks["bootstrap_block_lengths_checked"] = list(bootstrap)
    simple_bootstrap_path = folder / "simple_ensemble_direction_bootstrap.json"
    if simple_bootstrap_path.exists():
        simple = json.loads(simple_bootstrap_path.read_text())
        recipe = recipes["class__basic__equal_ensemble"]
        probability = direction["class__basic__equal_ensemble"].to_numpy()
        score = independent_metrics(y, probability, recipe["threshold"])
        for block, result in simple.items():
            assert result["difference_sign"] == "positive_favors_selected"
            assert result["block_sessions"] >= holdout.target_sessions.max()
            intervals = _independent_bootstrap(y, probability, baseline_probability, recipe["threshold"],
                                              int(block), result["bootstrap_draws"], result["metrics"])
            for name, metric in result["metrics"].items():
                assert np.isclose(metric["difference"], score[name] - baseline_score[name])
                np.testing.assert_allclose(intervals[name], metric["difference_95pct_block_interval"], rtol=1e-8, atol=1e-10)
        checks["simple_ensemble_bootstrap_block_lengths_checked"] = list(simple)
    checks["probability_calibration"] = "None fitted; model probabilities and development thresholds are reported directly"
    bundle = joblib.load(folder / "model.joblib")
    if fitted_final:
        assert {row["candidate"] for row in fitted_final} == set(bundle["members"])
    if bundle.get("cot_policy") == "all":
        checks.update(_audit_all_cot_inclusion(folder, bundle, fits, raw))
    final_cutoff = pd.Timestamp(bundle["fitted_as_of"]) + pd.Timedelta(days=1)
    assert pd.Timestamp(bundle["fitted_as_of"]) == raw.Date.max()
    for name, member in bundle["members"].items():
        spec = definitions[name]
        train = raw.loc[raw.Date.lt(final_cutoff) & raw.target_end_date.lt(final_cutoff) & raw.target_return.notna()]
        if spec["task"] == "class":
            train = train.loc[train.target_return.ne(0)]
        assert member["train_rows"] == len(train)
        assert pd.Timestamp(member["train_last_date"]) == train.Date.max()
        assert pd.Timestamp(member["train_last_target_date"]) == train.target_end_date.max() < final_cutoff
        estimator = member["estimator"].named_steps["model"]
        parameters = estimator.get_params()
        assert not parameters.get("early_stopping", False), "Random internal early stopping is active"
        assert parameters.get("early_stopping_rounds") is None, "Unrecorded early stopping is active"
        assert set(member["features"]).issubset(groups[spec["group"]])
    for name, recipe in bundle["recipes"].items():
        assert recipe == recipes[name], "Final deployment recipe or threshold changed"
    assert bundle["abstention"] == band, "Final confidence policy changed"
    latest = pd.read_csv(folder / "latest_forecast.csv")
    assert pd.to_datetime(latest.as_of_date).eq(raw.Date.max()).all()
    assert latest.cached_close.eq(raw.Close.iloc[-1]).all()
    checks["final_bundle_members_checked"] = len(bundle["members"])
    checks["final_bundle_policy_frozen"] = True
    checks.update(_audit_final_replay(folder, bundle, latest, final_cutoff))
    checks.update(_audit_precision_recall_policies(folder, cv, holdout, selection))
    checks["sha256"] = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in folder.iterdir() if path.suffix in {".csv", ".json"}}
    return checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, default=EXPERIMENTS / "artifacts" / "independent_direction_audit.json")
    arguments = parser.parse_args()
    folders = arguments.folder or sorted((EXPERIMENTS / "artifacts" / "direction").glob("*calendar"))
    assert folders, "No saved direction experiments available to audit"
    result = {"status": "passed", "experiments": {folder.name: audit_folder(folder) for folder in folders},
              "independent_recomputation": "Metrics, targets, threshold choices and bootstrap intervals recomputed independently; latest replay exercises delivered feature code"}
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
