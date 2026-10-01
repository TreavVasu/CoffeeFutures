"""Direction metrics, frozen pre-holdout decisions, and dependent uncertainty.

UP means a strictly positive realized return; DOWN a strictly negative return.
Exact-zero returns have no binary direction and must be filtered *once* before
comparing any models. Metrics never silently discard rows or invalid predictions.
All threshold and abstention searches require explicit pre-holdout origin dates.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def binary_labels_from_returns(actual) -> np.ndarray:
    """Return 1/0 for positive/negative returns and NaN for zero/missing labels."""
    values = np.asarray(actual, dtype=float)
    if values.ndim != 1 or np.isinf(values).any():
        raise ValueError("Returns must be one-dimensional and contain no infinities.")
    labels = np.full(len(values), np.nan)
    labels[values > 0] = 1.
    labels[values < 0] = 0.
    return labels


def _pairs(y_binary, prob_up):
    y, p = np.asarray(y_binary, dtype=float), np.asarray(prob_up, dtype=float)
    if y.ndim != 1 or p.ndim != 1 or not len(y) or y.shape != p.shape:
        raise ValueError("Metrics require nonempty paired one-dimensional inputs.")
    if not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("Labels and probabilities must be finite; filter neutral rows consistently first.")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("Direction labels must be binary 0/1, with exact-zero returns excluded.")
    if np.any((p < 0) | (p > 1)):
        raise ValueError("UP probabilities must lie in [0, 1].")
    return y.astype(np.int8), p


def _valid_threshold(threshold):
    if not np.isscalar(threshold) or not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Probability thresholds must be finite numbers in [0, 1].")
    return float(threshold)


def _count_metrics(y, predicted_up):
    tp = int(np.count_nonzero((y == 1) & predicted_up))
    tn = int(np.count_nonzero((y == 0) & ~predicted_up))
    fp = int(np.count_nonzero((y == 0) & predicted_up))
    fn = int(np.count_nonzero((y == 1) & ~predicted_up))
    divide = lambda a, b: float(a / b) if b else 0.
    up_precision, up_recall = divide(tp, tp + fp), divide(tp, tp + fn)
    down_precision, down_recall = divide(tn, tn + fn), divide(tn, tn + fp)
    up_f1, down_f1 = divide(2 * tp, 2 * tp + fp + fn), divide(2 * tn, 2 * tn + fp + fn)
    return {
        "rows": len(y), "accuracy": divide(tp + tn, len(y)),
        "balanced_accuracy": (up_recall + down_recall) / 2,
        "macro_f1": (up_f1 + down_f1) / 2,
        "up_precision": up_precision, "up_recall": up_recall, "up_f1": up_f1,
        "down_precision": down_precision, "down_recall": down_recall, "down_f1": down_f1,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "support_up": tp + fn, "support_down": tn + fp,
        "predicted_up": tp + fp, "predicted_down": tn + fn,
    }


def classification_metrics(y_binary, prob_up, threshold=.5) -> dict:
    """Evaluate both classes; UP is predicted when probability >= threshold.

    Undefined per-class precision/recall/F1 are zero. Macro scores always give
    UP and DOWN equal weight, even for single-class slices. PR-AUC is average
    precision, not trapezoidal interpolation; it is None when its class is absent.
    ROC-AUC is None unless both classes occur. Proper probability losses clip
    only for the logarithm; Brier uses the original bounded probabilities.
    """
    y, p = _pairs(y_binary, prob_up)
    threshold = _valid_threshold(threshold)
    result = _count_metrics(y, p >= threshold)
    clipped = np.clip(p, np.finfo(float).eps, 1 - np.finfo(float).eps)
    result.update({
        "threshold": threshold,
        "pr_auc_up": float(average_precision_score(y, p)) if np.any(y == 1) else None,
        "pr_auc_down": float(average_precision_score(1 - y, 1 - p)) if np.any(y == 0) else None,
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
        "brier": float(np.square(y - p).mean()),
        "log_loss": float(-(y * np.log(clipped) + (1 - y) * np.log1p(-clipped)).mean()),
    })
    return result


def _development_dates(dates, size, holdout_start, target_end_dates=None):
    origins = pd.DatetimeIndex(pd.to_datetime(dates))
    cutoff = pd.Timestamp(holdout_start)
    if len(origins) != size or origins.hasnans or pd.isna(cutoff):
        raise ValueError("Every development prediction requires a valid origin and holdout cutoff.")
    if not origins.is_monotonic_increasing or origins.has_duplicates:
        raise ValueError("Threshold selection requires unique chronological OOF predictions.")
    if np.any(origins >= cutoff):
        raise ValueError("Threshold/band selection may use only pre-holdout OOF origins.")
    if target_end_dates is not None:
        endpoints = pd.DatetimeIndex(pd.to_datetime(target_end_dates))
        if len(endpoints) != size or endpoints.hasnans or np.any(endpoints >= cutoff):
            raise ValueError("Every OOF target must be mature strictly before holdout starts.")
        if np.any(endpoints <= origins):
            raise ValueError("Target endpoints must follow their forecast origins.")
    return origins, cutoff


def select_threshold(y_binary, prob_up, *, dates, holdout_start,
                     target_end_dates=None, thresholds=None) -> dict:
    """Freeze a macro-F1 threshold using pre-holdout OOF rows exclusively.

    Ties prefer higher minimum class recall, then balanced accuracy, smaller
    recall gap, and proximity to 0.5. A fixed modest grid avoids searching every
    observed probability. The returned audit describes the selection window.
    """
    y, p = _pairs(y_binary, prob_up)
    origins, cutoff = _development_dates(dates, len(y), holdout_start, target_end_dates)
    grid = np.linspace(.25, .75, 51) if thresholds is None else np.asarray(thresholds, dtype=float)
    if grid.ndim != 1 or not len(grid):
        raise ValueError("Threshold search requires a nonempty one-dimensional grid.")
    table = []
    for value in np.unique(grid):
        threshold = _valid_threshold(value)
        table.append({**_count_metrics(y, p >= threshold), "threshold": threshold})
    def rank(row):
        return (row["macro_f1"], min(row["up_recall"], row["down_recall"]),
                row["balanced_accuracy"], -abs(row["up_recall"] - row["down_recall"]),
                -abs(row["threshold"] - .5), -row["threshold"])
    best = max(table, key=rank)
    return {"threshold": best["threshold"], "objective": "macro_f1",
            "selection_rows": len(y), "selection_start": str(origins.min().date()),
            "selection_end": str(origins.max().date()), "holdout_start": str(cutoff.date()),
            "selection_metrics": classification_metrics(y, p, best["threshold"]),
            "search": table, "selection_source": "pre_holdout_oof"}


def selective_metrics(y_binary, prob_up, lower=.4, upper=.6) -> dict:
    """Evaluate an abstention policy: DOWN below lower, UP at/above upper.

    Per-class recall is conditional on accepted rows. Unconditional class recall
    divides correctly accepted class predictions by *all* actual class rows, so
    withholding difficult predictions cannot hide a loss in total recall.
    """
    y, p = _pairs(y_binary, prob_up)
    lower, upper = _valid_threshold(lower), _valid_threshold(upper)
    if lower > upper:
        raise ValueError("The lower abstention boundary cannot exceed the upper boundary.")
    accepted = (p < lower) | (p >= upper)
    accepted_y = y[accepted]
    result = _count_metrics(accepted_y, p[accepted] >= upper)
    # An empty accepted set has no observed conditional accuracy; macro-F1 is 0.
    if not accepted.any():
        result["accuracy"] = None
        result["balanced_accuracy"] = None
    result.update({
        "rows": len(y), "accepted_rows": int(accepted.sum()),
        "abstained_rows": int((~accepted).sum()), "coverage": float(accepted.mean()),
        "lower": lower, "upper": upper,
        "abstained_up": int(np.count_nonzero((y == 1) & ~accepted)),
        "abstained_down": int(np.count_nonzero((y == 0) & ~accepted)),
        "support_up_all": int(np.count_nonzero(y == 1)),
        "support_down_all": int(np.count_nonzero(y == 0)),
    })
    for label, value in [("up", 1), ("down", 0)]:
        support = int(np.count_nonzero(y == value))
        correct = result["tp" if value else "tn"]
        result[f"{label}_unconditional_recall"] = float(correct / support) if support else 0.
        result[f"{label}_class_coverage"] = float(np.count_nonzero(accepted_y == value) / support) if support else 0.
    return result


def select_abstention_band(y_binary, prob_up, threshold=.5, *, dates,
                          holdout_start, target_end_dates=None, widths=None,
                          min_coverage=.6, min_class_coverage=.3,
                          min_unconditional_recall=.15,
                          min_predicted_per_class=1) -> dict:
    """Select a symmetric probability band from pre-holdout OOF data only.

    Coverage and class coverage gates are mandatory. Primary optimization is
    conditional macro-F1; ties favor balanced precision, unconditional recall,
    and coverage. If every band fails gates, return a frozen zero-width fallback
    marked ``gates_passed=False`` instead of claiming a valid selective policy.
    """
    y, p = _pairs(y_binary, prob_up)
    threshold = _valid_threshold(threshold)
    origins, cutoff = _development_dates(dates, len(y), holdout_start, target_end_dates)
    for value in [min_coverage, min_class_coverage, min_unconditional_recall]:
        _valid_threshold(value)
    if not isinstance(min_predicted_per_class, (int, np.integer)) or min_predicted_per_class < 1:
        raise ValueError("At least one accepted prediction for each class is required.")
    grid = np.array([0., .025, .05, .075, .10, .125, .15, .175, .20]) if widths is None else np.asarray(widths, dtype=float)
    if grid.ndim != 1 or not len(grid) or not np.isfinite(grid).all() or np.any(grid < 0):
        raise ValueError("Band widths must be a nonempty finite nonnegative grid.")
    table = []
    for width in np.unique(np.r_[0., grid]):
        row = selective_metrics(y, p, max(0., threshold - width), min(1., threshold + width))
        row["width"] = float(width)
        row["gates_passed"] = bool(
            row["coverage"] >= min_coverage and
            min(row["up_class_coverage"], row["down_class_coverage"]) >= min_class_coverage and
            min(row["up_unconditional_recall"], row["down_unconditional_recall"]) >= min_unconditional_recall and
            min(row["predicted_up"], row["predicted_down"]) >= min_predicted_per_class)
        table.append(row)
    passing = [row for row in table if row["gates_passed"]]
    def rank(row):
        return (row["macro_f1"], min(row["up_precision"], row["down_precision"]),
                min(row["up_unconditional_recall"], row["down_unconditional_recall"]),
                row["coverage"], -row["width"])
    best = max(passing, key=rank) if passing else table[0]
    return {"lower": best["lower"], "upper": best["upper"], "width": best["width"],
            "threshold": threshold, "objective": "conditional_macro_f1_with_coverage_gates",
            "gates_passed": best["gates_passed"], "selection_rows": len(y),
            "selection_start": str(origins.min().date()), "selection_end": str(origins.max().date()),
            "holdout_start": str(cutoff.date()), "selection_source": "pre_holdout_oof",
            "min_coverage": min_coverage, "min_class_coverage": min_class_coverage,
            "min_unconditional_recall": min_unconditional_recall,
            "min_predicted_per_class": min_predicted_per_class,
            "selection_metrics": best, "search": table}


def paired_block_bootstrap(y_binary, prob_selected, prob_baseline, *,
                           threshold_selected=.5, threshold_baseline=.5,
                           block=40, draws=1000, seed=42,
                           metric_names=("macro_f1", "balanced_accuracy", "accuracy",
                                         "up_precision", "up_recall", "down_precision", "down_recall")) -> dict:
    """Paired moving-block intervals for selected-minus-baseline scores.

    Positive differences favor the selected policy. Whole chronological blocks
    preserve local dependence from overlapping monthly labels. Thresholds stay
    frozen for every draw; thresholds are never optimized inside holdout draws.
    The block length should be at least the largest observed target-session span.
    """
    y, chosen = _pairs(y_binary, prob_selected)
    _, baseline = _pairs(y_binary, prob_baseline)
    selected_cut = _valid_threshold(threshold_selected)
    baseline_cut = _valid_threshold(threshold_baseline)
    if (not isinstance(block, (int, np.integer)) or block < 1 or
            not isinstance(draws, (int, np.integer)) or draws < 1):
        raise ValueError("Bootstrap block length and draw count must be positive integers.")
    selected_up, baseline_up = chosen >= selected_cut, baseline >= baseline_cut
    selected_score, baseline_score = _count_metrics(y, selected_up), _count_metrics(y, baseline_up)
    if not metric_names or any(name not in selected_score or name in {"rows", "tp", "tn", "fp", "fn", "support_up", "support_down", "predicted_up", "predicted_down"} for name in metric_names):
        raise ValueError("Bootstrap requires supported classification rate metrics.")
    length = min(int(block), len(y))
    rng = np.random.default_rng(seed)
    samples = {name: np.empty(draws) for name in metric_names}
    for iteration in range(draws):
        starts = rng.integers(0, len(y) - length + 1, size=int(np.ceil(len(y) / length)))
        indices = (starts[:, None] + np.arange(length)).ravel()[:len(y)]
        first = _count_metrics(y[indices], selected_up[indices])
        second = _count_metrics(y[indices], baseline_up[indices])
        for name in metric_names:
            samples[name][iteration] = first[name] - second[name]
    results = {}
    for name in metric_names:
        interval = np.quantile(samples[name], [.025, .975]).tolist()
        results[name] = {"selected": selected_score[name], "baseline": baseline_score[name],
                         "difference": selected_score[name] - baseline_score[name],
                         "difference_95pct_block_interval": interval}
    return {"rows": len(y), "block_sessions": length, "requested_block_sessions": int(block),
            "bootstrap_draws": int(draws), "seed": seed,
            "difference_sign": "positive_favors_selected", "metrics": results}


def nonoverlap_positions(frame: pd.DataFrame, offset=0) -> list[int]:
    """Greedily select origins at/after the previous *actual* target endpoint."""
    if not isinstance(offset, (int, np.integer)) or offset < 0:
        raise ValueError("Nonoverlap offset must be a nonnegative integer.")
    dates = pd.DatetimeIndex(pd.to_datetime(frame.Date))
    ends = pd.DatetimeIndex(pd.to_datetime(frame.target_end_date))
    if (dates.hasnans or ends.hasnans or dates.has_duplicates or
            not dates.is_monotonic_increasing or np.any(ends <= dates)):
        raise ValueError("Nonoverlap sampling requires valid increasing origins and future endpoints.")
    selected, last_end = [], pd.Timestamp.min
    for position in range(offset, len(frame)):
        if dates[position] >= last_end:
            selected.append(position)
            last_end = ends[position]
    return selected
