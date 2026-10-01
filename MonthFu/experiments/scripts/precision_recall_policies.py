"""Frozen precision/recall threshold policies from stored pre-2022 predictions.

No models are fitted. Selection is persisted before the holdout CSV is read.
The policies are exploratory alternatives and do not replace the main recipe.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from MonthFu.experiments.src.direction_metrics import (
    binary_labels_from_returns, classification_metrics, paired_block_bootstrap)

THRESHOLDS = np.round(np.arange(0.35, 0.651, 0.025), 3)


def ready(value):
    if isinstance(value, dict):
        return {str(key): ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [ready(item) for item in value]
    if isinstance(value, np.generic):
        return ready(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def dump(path, value):
    Path(path).write_text(json.dumps(ready(value), indent=2, allow_nan=False)+"\n")


def recipe_signature(recipe):
    """Identify probability-equivalent aliases by frozen member weights."""
    weights = recipe.get("weights", {})
    if not weights or not all(np.isfinite(weight) and weight > 0 for weight in weights.values()):
        raise ValueError("Frozen recipes require finite positive member weights")
    if not np.isclose(sum(weights.values()), 1.0, atol=1e-10):
        raise ValueError("Classifier recipe weights must sum to one")
    return tuple(sorted((name, float(weight)) for name, weight in weights.items()))


def fixed_classifier_recipes(selection):
    """Retain only the frozen per-family/group and equal-weight class recipes."""
    retained, signatures, aliases = {}, {}, {}
    for name, recipe in sorted(selection["recipes"].items()):
        if recipe.get("task") != "class" or not name.startswith("class__"):
            continue
        parts = name.split("__")
        if len(parts) != 3 or parts[2] not in {"linear", "xgb", "forest", "equal_ensemble"}:
            continue
        signature = recipe_signature(recipe)
        if signature in signatures:
            aliases[name] = signatures[signature]
            continue
        signatures[signature] = name
        retained[name] = recipe
    if not retained:
        raise ValueError("No frozen per-family/group classifier recipes available")
    for name, recipe in selection["recipes"].items():
        if recipe.get("task") == "class" and name not in retained:
            signature = recipe_signature(recipe)
            if signature in signatures:
                aliases[name] = signatures[signature]
    return retained, aliases


def validate_development(oof, cutoff):
    required = {"Date", "target_end_date", "target_return"}
    if not required.issubset(oof):
        raise ValueError("OOF selection requires origins, target endpoints and returns")
    dates = pd.DatetimeIndex(pd.to_datetime(oof.Date))
    ends = pd.DatetimeIndex(pd.to_datetime(oof.target_end_date))
    cutoff = pd.Timestamp(cutoff)
    if (not len(oof) or dates.hasnans or dates.has_duplicates or not dates.is_monotonic_increasing
            or ends.hasnans or pd.isna(cutoff)):
        raise ValueError("Development dates must be unique, valid and chronological")
    if np.any(dates >= cutoff) or np.any(ends >= cutoff):
        raise ValueError("Only OOF origins and labels strictly before holdout are permitted")
    if np.any(ends <= dates):
        raise ValueError("Every target endpoint must follow its origin")
    if not np.isfinite(oof.target_return.to_numpy(dtype=float)).all():
        raise ValueError("OOF target returns must be finite before consistently excluding neutral rows")
    labels = binary_labels_from_returns(oof.target_return)
    mask = np.isfinite(labels)
    if not mask.any() or len(np.unique(labels[mask])) < 2:
        raise ValueError("Precision/recall policy selection requires both realized direction classes")
    return labels[mask], mask


def oof_recipe_probability(oof, name, recipe):
    if name not in oof:
        raise ValueError(f"Missing frozen OOF recipe prediction: {name}")
    probability = oof[name].to_numpy(dtype=float)
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError(f"Invalid OOF UP probability: {name}")
    weights = recipe["weights"]
    if all(member in oof for member in weights):
        rebuilt = sum(float(weight)*oof[member].to_numpy(dtype=float) for member, weight in weights.items())
        if not np.allclose(probability, rebuilt, rtol=1e-7, atol=1e-8):
            raise ValueError(f"OOF probability does not match frozen recipe: {name}")
    return probability


def _policy_rank(row, policy):
    if policy == "precision_first":
        primary = min(row["up_precision"], row["down_precision"])
    elif policy == "recall_balanced":
        primary = min(row["up_recall"], row["down_recall"])
    else:
        raise ValueError("Unknown precision/recall policy")
    # Remaining deterministic ties never consult the holdout.
    return (primary, row["macro_f1"], -abs(row["threshold"]-0.5), -row["threshold"])


def select_precision_recall_policies(oof, original_selection, min_recall=0.40, thresholds=None):
    """Choose two model/threshold policies using mature OOF data exclusively."""
    if not np.isfinite(min_recall) or not 0 <= min_recall <= 1:
        raise ValueError("Minimum recall must lie in [0,1]")
    cutoff = original_selection["holdout_start"]
    labels, mask = validate_development(oof, cutoff)
    recipes, aliases = fixed_classifier_recipes(original_selection)
    grid = THRESHOLDS if thresholds is None else np.asarray(thresholds, dtype=float)
    if grid.ndim != 1 or not len(grid) or not np.isfinite(grid).all() or ((grid < 0) | (grid > 1)).any():
        raise ValueError("Thresholds must be nonempty bounded finite probabilities")
    grid = np.unique(grid)
    records = []
    for name, recipe in recipes.items():
        probability = oof_recipe_probability(oof, name, recipe)[mask]
        for threshold in grid:
            row = {"recipe": name, "group": recipe.get("group"),
                   **classification_metrics(labels, probability, float(threshold))}
            row["minimum_precision"] = min(row["up_precision"], row["down_precision"])
            row["minimum_recall"] = min(row["up_recall"], row["down_recall"])
            row["precision_first_gates_passed"] = bool(row["minimum_recall"] >= min_recall)
            records.append(row)
    fallback = original_selection["recipes"].get("selected_direction")
    fallback_name = aliases.get("selected_direction")
    if fallback_name is None and fallback is not None:
        signature = recipe_signature(fallback)
        fallback_name = next((name for name, recipe in recipes.items() if recipe_signature(recipe) == signature), None)
    if fallback_name is None:
        # If the upstream alias is unavailable, preserve its objective using its
        # development-selected recipe thresholds, never loosen the precision gate.
        frozen = [{"recipe": name, **classification_metrics(labels, oof_recipe_probability(oof, name, recipe)[mask],
                                                             float(recipe.get("threshold", 0.5)))}
                  for name, recipe in recipes.items()]
        fallback_name = max(frozen, key=lambda row: (row["macro_f1"], row["balanced_accuracy"]))["recipe"]
        fallback = recipes[fallback_name]
    per_recipe, policies = {}, {}
    for policy in ["precision_first", "recall_balanced"]:
        eligible = [row for row in records if policy != "precision_first" or row["precision_first_gates_passed"]]
        if eligible:
            best = max(eligible, key=lambda row: _policy_rank(row, policy))
            gates_passed, fallback_reason = True, None
        else:
            threshold = float(fallback.get("threshold", 0.5))
            best = {"recipe": fallback_name, "group": recipes[fallback_name].get("group"),
                    **classification_metrics(labels, oof_recipe_probability(oof, fallback_name, recipes[fallback_name])[mask], threshold)}
            best.update(minimum_precision=min(best["up_precision"], best["down_precision"]),
                        minimum_recall=min(best["up_recall"], best["down_recall"]))
            gates_passed = False
            fallback_reason = "No model/threshold passed both recall gates; retain original frozen macro-F1 policy without claiming precision qualification"
        policies[policy] = {"recipe": best["recipe"], "threshold": best["threshold"],
                            "weights": recipes[best["recipe"]]["weights"],
                            "group": best["group"], "objective": "minimum_class_precision" if policy == "precision_first" else "minimum_class_recall",
                            "minimum_required_recall": min_recall if policy == "precision_first" else None,
                            "gates_passed": gates_passed, "fallback_reason": fallback_reason,
                            "selection_metrics": best, "coverage": 1.0}
        recipe_choices = {}
        for name in recipes:
            candidates = [row for row in eligible if row["recipe"] == name]
            if candidates:
                winner = max(candidates, key=lambda row: _policy_rank(row, policy))
                recipe_choices[name] = {"threshold": winner["threshold"], "gates_passed": True,
                                        "selection_metrics": winner}
            else:
                recipe_choices[name] = {"threshold": None, "gates_passed": False,
                                        "reason": "No threshold passed both recall gates"}
        per_recipe[policy] = recipe_choices
    selection = {"holdout_start": str(pd.Timestamp(cutoff).date()), "thresholds": grid.tolist(),
                 "selection_source": "mature_pre_holdout_oof_only", "policy_names": list(policies),
                 "selection_rows": int(mask.sum()), "excluded_neutral_rows": int((~mask).sum()),
                 "selection_start": str(pd.Timestamp(oof.Date.min()).date()),
                 "selection_end": str(pd.Timestamp(oof.Date.max()).date()),
                 "latest_selection_label_end": str(pd.Timestamp(oof.target_end_date.max()).date()),
                 "candidate_recipe_count": len(recipes), "grid_rows": len(records),
                 "original_direction_policy": {"recipe": fallback_name, "threshold": float(fallback.get("threshold", 0.5))},
                 "retained_recipes": {name: {"weights": recipe["weights"], "original_threshold": recipe.get("threshold", 0.5),
                                              "group": recipe.get("group")} for name, recipe in recipes.items()},
                 "deduplicated_aliases": aliases, "policies": policies, "per_recipe_policies": per_recipe,
                 "tie_rule": "Primary objective, then macro-F1, then distance to0.5, then lowerthreshold, then stable alphabetical recipe order",
                 "zero_rule": "Zero realized returns excluded; probability>=threshold predicts Up",
                 "interpretation": "Exploratory no-fit alternatives; no policy is promoted automatically and holdout scores do not select a winner"}
    return selection, pd.DataFrame(records)


def evaluate_policies(holdout, selection):
    dates = pd.DatetimeIndex(pd.to_datetime(holdout.Date))
    if (dates.hasnans or dates.has_duplicates or not dates.is_monotonic_increasing
            or np.any(dates < pd.Timestamp(selection["holdout_start"]))):
        raise ValueError("Policy evaluation requires unique chronological holdout origins")
    actual = holdout.target_return.to_numpy(dtype=float)
    if not np.isfinite(actual).all():
        raise ValueError("Holdout returns must be finite")
    labels = binary_labels_from_returns(actual)
    mask = np.isfinite(labels)
    result = []
    # Compare new policies to the original macro-F1 decision on the same recipe,
    # plus the original overall selected direction probability/threshold.
    for policy, frozen in selection["policies"].items():
        name = frozen["recipe"]
        if name not in holdout:
            raise ValueError(f"Frozen recipe absent from completed holdout predictions: {name}")
        probabilities = holdout[name].to_numpy(dtype=float)
        original_threshold = selection["retained_recipes"][name]["original_threshold"]
        for label, threshold in [(policy, frozen["threshold"]), (f"{policy}__original_macro_f1_threshold", original_threshold)]:
            metrics = classification_metrics(labels[mask], probabilities[mask], threshold)
            result.append({"policy": label, "recipe": name, "group": frozen["group"],
                           "development_gates_passed": frozen["gates_passed"], "coverage": 1.0,
                           "excluded_neutral_rows": int((~mask).sum()), **metrics})
    original = selection["original_direction_policy"]
    if original["recipe"] not in holdout:
        raise ValueError("Original frozen direction reference is missing from holdout")
    metrics = classification_metrics(labels[mask], holdout[original["recipe"]].to_numpy(dtype=float)[mask], original["threshold"])
    result.append({"policy": "original_selected_direction", "recipe": original["recipe"],
                   "group": selection["retained_recipes"][original["recipe"]]["group"],
                   "development_gates_passed": True, "coverage": 1.0,
                   "excluded_neutral_rows": int((~mask).sum()), **metrics})
    return pd.DataFrame(result)


def policy_block_uncertainty(holdout, selection, draws=1000):
    """Paired policy-minus-original intervals; all thresholds remain frozen."""
    labels = binary_labels_from_returns(holdout.target_return)
    mask = np.isfinite(labels)
    original = selection["original_direction_policy"]
    reference = holdout[original["recipe"]].to_numpy(dtype=float)[mask]
    results = {}
    for policy, frozen in selection["policies"].items():
        probabilities = holdout[frozen["recipe"]].to_numpy(dtype=float)[mask]
        results[policy] = {str(block): paired_block_bootstrap(
            labels[mask], probabilities, reference, threshold_selected=frozen["threshold"],
            threshold_baseline=original["threshold"], block=block, draws=draws)
            for block in [30, 60, 90]}
    return {"reference": original, "coverage": 1.0, "primary_block_sessions": 60,
            "interpretation": "Positive policy-minus-original differences favor policy; intervals preserve local dependence from overlapping outcomes and do not establish prospective skill",
            "policies": results}


def run_horizon(destination, select_only=False):
    destination = Path(destination)
    original_selection = json.loads((destination/"selection.json").read_text())
    oof = pd.read_csv(destination/"cv_predictions.csv", parse_dates=["Date", "target_end_date"])
    selection, grid = select_precision_recall_policies(oof, original_selection)
    selection["input_sha256"] = {name: hashlib.sha256((destination/name).read_bytes()).hexdigest()
                                 for name in ["selection.json", "cv_predictions.csv"]}
    selection["code_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    grid.to_csv(destination/"precision_recall_policy_cv_grid.csv", index=False)
    # Critical ordering: freeze and persist before opening any holdout CSV.
    dump(destination/"precision_recall_policy_selection.json", selection)
    if select_only:
        return selection
    holdout = pd.read_csv(destination/"holdout_predictions.csv", parse_dates=["Date", "target_end_date"])
    metrics = evaluate_policies(holdout, selection)
    metrics.to_csv(destination/"precision_recall_policy_metrics.csv", index=False)
    dump(destination/"precision_recall_policy_block_uncertainty.json", policy_block_uncertainty(holdout, selection))
    print(destination.name)
    print(metrics[["policy", "recipe", "threshold", "accuracy", "macro_f1", "up_precision", "up_recall", "down_precision", "down_recall"]].to_string(index=False))
    return selection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizons", nargs="+", type=int, default=[30, 28])
    parser.add_argument("--output-root", type=Path, default=ROOT/"MonthFu/experiments/artifacts/direction")
    parser.add_argument("--select-only", action="store_true")
    args = parser.parse_args()
    for horizon in args.horizons:
        run_horizon(args.output_root/f"{horizon}calendar", args.select_only)


if __name__ == "__main__":
    main()
