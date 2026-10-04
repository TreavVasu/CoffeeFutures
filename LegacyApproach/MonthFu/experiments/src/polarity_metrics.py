"""Metrics for two separately-trained directional models.

The two heads are fitted independently, so their probabilities are not on a
shared calibration and must not be compared as if they were.  The decision rule
therefore compares each score against **its own frozen threshold**:

    UP      when ``p_up >= up_threshold``   and ``p_down <  down_threshold``
    DOWN    when ``p_down >= down_threshold`` and ``p_up <  up_threshold``
    NEUTRAL when neither fires (abstention; no call is made)

A margin rule is reported alongside as a baseline.  Every reported rate carries
its unconditional counterpart, because a policy can raise apparent recall by
withholding hard rows unless the denominators stay fixed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from direction_metrics import binary_labels_from_returns


def _counts(y: np.ndarray, called_up: np.ndarray) -> dict:
    up, down = y == 1, y == 0
    tp = int(np.count_nonzero(up & called_up))
    tn = int(np.count_nonzero(down & ~called_up))
    fp = int(np.count_nonzero(down & called_up))
    fn = int(np.count_nonzero(up & ~called_up))
    divide = lambda a, b: float(a / b) if b else 0.0
    precision, recall = divide(tp, tp + fp), divide(tp, tp + fn)
    down_precision, down_recall = divide(tn, tn + fn), divide(tn, tn + fp)
    return {
        "rows": int(len(y)), "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": divide(tp + tn, len(y)),
        "balanced_accuracy": (recall + down_recall) / 2,
        "macro_f1": (divide(2 * tp, 2 * tp + fp + fn) + divide(2 * tn, 2 * tn + fp + fn)) / 2,
        "up_precision": precision, "up_recall": recall,
        "up_f1": divide(2 * tp, 2 * tp + fp + fn),
        "down_precision": down_precision, "down_recall": down_recall,
        "down_f1": divide(2 * tn, 2 * tn + fp + fn),
    }


def _validate(score, threshold, name):
    values = np.asarray(score, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError(f"{name} scores must be a nonempty finite 1-D sequence.")
    if not np.isscalar(threshold) or not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError(f"The {name} threshold must be a finite number in [0, 1].")
    return values, float(threshold)


def decide(p_up, p_down, up_threshold, down_threshold) -> np.ndarray:
    """Per-row call in {+1 UP, -1 DOWN, 0 NEUTRAL} from both frozen thresholds."""
    up, up_cut = _validate(p_up, up_threshold, "up")
    down, down_cut = _validate(p_down, down_threshold, "down")
    if up.shape != down.shape:
        raise ValueError("Both models must score the same rows.")
    call = np.zeros(len(up))
    call[(up >= up_cut) & (down < down_cut)] = 1.0
    call[(down >= down_cut) & (up < up_cut)] = -1.0
    return call


def evaluate_calls(y_binary, call) -> dict:
    """Score a call vector, reporting abstention cost honestly.

    ``accuracy`` and ``macro_f1`` are conditional on a call being made;
    ``unconditional_*`` divide the same correct counts by *all* actual class
    rows, so withholding difficult origins cannot inflate the headline.
    """
    y, decisions = np.asarray(y_binary, dtype=float), np.asarray(call, dtype=float)
    if y.ndim != 1 or decisions.shape != y.shape:
        raise ValueError("Labels and calls must be aligned 1-D sequences.")
    if not np.isfinite(y).all() or not np.isin(y, [0, 1]).all():
        raise ValueError("Labels must be binary 0/1 with exact-zero returns excluded.")
    if not np.isin(decisions, [-1.0, 0.0, 1.0]).all():
        raise ValueError("Calls must be +1, -1 or 0.")
    made = decisions != 0
    called_up = decisions == 1.0
    result = _counts(y, called_up)
    result["coverage"] = float(made.mean())
    result["neutral_rows"] = int((~made).sum())
    support_up, support_down = int(np.count_nonzero(y == 1)), int(np.count_nonzero(y == 0))
    correct_up = int(np.count_nonzero((y == 1) & (decisions == 1.0)))
    correct_down = int(np.count_nonzero((y == 0) & (decisions == -1.0)))
    result.update({
        "support_up": support_up, "support_down": support_down,
        "up_unconditional_recall": float(correct_up / support_up) if support_up else 0.0,
        "down_unconditional_recall": float(correct_down / support_down) if support_down else 0.0,
        "up_class_coverage": float(np.count_nonzero((y == 1) & made) / support_up) if support_up else 0.0,
        "down_class_coverage": float(np.count_nonzero((y == 0) & made) / support_down) if support_down else 0.0,
        "neutral_accuracy": float((correct_up + correct_down) / len(y)) if len(y) else 0.0,
    })
    return result


def margin_baseline(y_binary, p_up, p_down) -> dict:
    """Reference rule that simply compares the two raw scores.

    Reported only as a sanity check that the threshold rule is not merely
    re-deriving ``p_up > p_down``.
    """
    y, up, down = np.asarray(y_binary, dtype=float), np.asarray(p_up, dtype=float), np.asarray(p_down, dtype=float)
    return evaluate_calls(y, np.where(up >= down, 1.0, -1.0))


def polarity_report(frame: pd.DataFrame, p_up, p_down, up_threshold,
                    down_threshold) -> dict:
    """Evaluate one row set, honouring the single shared label filter.

    Exact-zero forward returns have no direction and are excluded once, for
    both sides, so the two heads are always compared on identical rows.
    """
    labels = binary_labels_from_returns(frame.target_return)
    keep = np.isfinite(labels)
    if not keep.any():
        raise ValueError("No rows with an observed direction in this slice.")
    up, down = np.asarray(p_up)[keep], np.asarray(p_down)[keep]
    result = evaluate_calls(labels[keep], decide(up, down, up_threshold, down_threshold))
    result["excluded_neutral_rows"] = int((~keep).sum())
    result["up_threshold"] = float(up_threshold)
    result["down_threshold"] = float(down_threshold)
    result["margin_baseline"] = margin_baseline(labels[keep], up, down)
    return result


THRESHOLD_GRID = np.round(np.arange(.30, .701, .025), 3)


def select_side_threshold(y_binary, score, *, dates, target_end_dates,
                          holdout_start, polarity, grid=THRESHOLD_GRID) -> dict:
    """Freeze one side's threshold on pre-holdout out-of-fold rows only.

    The objective is F1 for *this side's own* positive class, so the UP and
    DOWN searches optimise genuinely different criteria rather than sharing one
    macro objective.  Ties prefer higher balanced accuracy, then a threshold
    closer to the training prevalence, so the rule stays near neutral unless
    the data demands otherwise.

    Both the origins and their label endpoints must mature strictly before
    ``holdout_start``; otherwise the threshold would be tuned on holdout
    outcomes.
    """
    y = np.asarray(y_binary, dtype=float)
    score = np.asarray(score, dtype=float)
    origins = pd.DatetimeIndex(pd.to_datetime(dates))
    ends = pd.DatetimeIndex(pd.to_datetime(target_end_dates))
    cutoff = pd.Timestamp(holdout_start)
    if len(y) != len(score) or len(origins) != len(y) or len(ends) != len(y):
        raise ValueError("Threshold selection requires aligned labels, scores, origins and endpoints.")
    if not np.isfinite(y).all() or not np.isfinite(score).all():
        raise ValueError("Threshold selection requires finite labels and scores.")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("Direction labels must be binary 0/1.")
    if origins.hasnans or ends.hasnans or not origins.is_monotonic_increasing or origins.has_duplicates:
        raise ValueError("Threshold selection requires unique chronological origins.")
    if np.any(origins >= cutoff) or np.any(ends >= cutoff):
        raise ValueError("Threshold selection may use only pre-holdout origins with matured labels.")
    if np.any(ends <= origins):
        raise ValueError("Label endpoints must follow their forecast origins.")
    grid = np.asarray(grid, dtype=float)
    if grid.ndim != 1 or not len(grid) or not np.isfinite(grid).all() or np.any((grid < 0) | (grid > 1)):
        raise ValueError("The threshold grid must be finite and inside [0, 1].")

    prevalence = float(y.mean())
    table = [{**_counts(y, score >= threshold), "threshold": float(threshold)}
             for threshold in np.unique(grid)]
    # F1 for this side's positive class; ties favour balance, then proximity
    # to the training prevalence so an unbalanced sample does not drift.
    rank = lambda row: (row[f"{polarity}_f1"], row["balanced_accuracy"],
                        -abs(row["threshold"] - prevalence), -row["threshold"])
    best = max(table, key=rank)
    return {"threshold": best["threshold"], "objective": f"{polarity}_f1",
            "selection_rows": int(len(y)), "training_prevalence": prevalence,
            "selection_start": str(origins.min().date()), "selection_end": str(origins.max().date()),
            "latest_label_end": str(ends.max().date()), "holdout_start": str(cutoff.date()),
            "selection_source": "pre_holdout_oof",
            "selection_metrics": best, "search": table}


def select_threshold_pair(y_binary, p_up, p_down, *, dates, target_end_dates,
                          holdout_start, up_grid=THRESHOLD_GRID,
                          down_grid=THRESHOLD_GRID) -> dict:
    """Choose *both* thresholds together on pre-holdout development rows.

    Why this is joint and not two independent searches
    ----------------------------------------------------
    The two heads are fitted separately, but their thresholds are not
    independent decisions.  They feed one rule that fires UP only when the UP
    score clears its cut *and* the DOWN score stays below its own.  Maximising
    each side's own F1 pushes every cut in the direction that calls that side
    more often, which compounds: the UP cut drifts low while the DOWN cut drifts
    high, and the pair then calls UP on almost every origin while DOWN recall
    collapses.  Each side looks well scored while the joint rule is degenerate.

    The pair is therefore selected against the quantity actually deployed - the
    macro-F1 and balanced accuracy of the combined call - while the two heads,
    their features and their hyper-parameters remain fully independent.
    """
    y = np.asarray(y_binary, dtype=float)
    up = np.asarray(p_up, dtype=float)
    down = np.asarray(p_down, dtype=float)
    origins = pd.DatetimeIndex(pd.to_datetime(dates))
    ends = pd.DatetimeIndex(pd.to_datetime(target_end_dates))
    cutoff = pd.Timestamp(holdout_start)
    if not (len(y) == len(up) == len(down) == len(origins) == len(ends)):
        raise ValueError("Threshold selection requires aligned scores, labels, origins and endpoints.")
    if not np.isfinite(y).all() or not np.isfinite(up).all() or not np.isfinite(down).all():
        raise ValueError("Threshold selection requires finite scores and labels.")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("Direction labels must be binary 0/1.")
    if origins.hasnans or ends.hasnans or not origins.is_monotonic_increasing or origins.has_duplicates:
        raise ValueError("Threshold selection requires unique chronological origins.")
    if np.any(origins >= cutoff) or np.any(ends >= cutoff):
        raise ValueError("Threshold selection may use only pre-holdout origins with matured labels.")
    if np.any(ends <= origins):
        raise ValueError("Label endpoints must follow their forecast origins.")
    up_grid = np.asarray(up_grid, dtype=float)
    down_grid = np.asarray(down_grid, dtype=float)
    for grid in (up_grid, down_grid):
        if grid.ndim != 1 or not len(grid) or not np.isfinite(grid).all() or np.any((grid < 0) | (grid > 1)):
            raise ValueError("Each threshold grid must be finite and inside [0, 1].")

    table = []
    for up_cut in np.unique(up_grid):
        for down_cut in np.unique(down_grid):
            score = evaluate_calls(y, decide(up, down, up_cut, down_cut))
            table.append({"up_threshold": float(up_cut), "down_threshold": float(down_cut),
                          "macro_f1": score["macro_f1"],
                          "balanced_accuracy": score["balanced_accuracy"],
                          "accuracy": score["accuracy"],
                          "neutral_accuracy": score["neutral_accuracy"],
                          "coverage": score["coverage"],
                          "up_recall": score["up_recall"], "down_recall": score["down_recall"]})
    # Neutral accuracy keeps the denominators fixed, so abstaining cannot win by
    # shrinking the sample it is judged on.
    best = max(table, key=lambda row: (row["macro_f1"], row["balanced_accuracy"],
                                       row["neutral_accuracy"], row["coverage"],
                                       -abs(row["up_threshold"] - .5),
                                       -abs(row["down_threshold"] - .5)))
    return {"up_threshold": best["up_threshold"], "down_threshold": best["down_threshold"],
            "objective": "joint_macro_f1_of_the_combined_call",
            "selection_rows": int(len(y)), "selection_start": str(origins.min().date()),
            "selection_end": str(origins.max().date()),
            "latest_label_end": str(ends.max().date()),
            "holdout_start": str(cutoff.date()), "selection_source": "pre_holdout_development",
            "search_size": len(table), "selection_metrics": best, "search": table}


def block_bootstrap_difference(y_binary, call_selected, call_baseline, *,
                               block: int, draws: int = 1000, seed: int = 42,
                               metric_names=("macro_f1", "balanced_accuracy", "accuracy",
                                             "neutral_accuracy", "up_recall", "down_recall")) -> dict:
    """Moving-block interval for a selected-minus-baseline metric difference.

    Origins inside a cell share future observations and origins one session
    apart overlap heavily, so an independent-sample interval would be far too
    narrow.  Whole chronological blocks preserve that local dependence.  The
    block length must cover the cell's forward offset or the interval will
    understate uncertainty.
    """
    y = np.asarray(y_binary, dtype=float)
    selected, baseline = np.asarray(call_selected, dtype=float), np.asarray(call_baseline, dtype=float)
    if selected.shape != y.shape or baseline.shape != y.shape or not len(y):
        raise ValueError("Paired bootstrap requires aligned nonempty label and call vectors.")
    if not np.isfinite(y).all() or not np.isfinite(selected).all() or not np.isfinite(baseline).all():
        raise ValueError("Paired bootstrap requires finite inputs.")
    if not isinstance(block, (int, np.integer)) or block < 1:
        raise ValueError("The bootstrap block length must be a positive integer.")
    if not isinstance(draws, (int, np.integer)) or draws < 1:
        raise ValueError("The bootstrap draw count must be a positive integer.")
    length = min(int(block), len(y))
    rng = np.random.default_rng(seed)
    samples = {name: np.empty(draws) for name in metric_names}
    for iteration in range(draws):
        starts = rng.integers(0, len(y) - length + 1, size=int(np.ceil(len(y) / length)))
        index = (starts[:, None] + np.arange(length)).ravel()[:len(y)]
        first, second = evaluate_calls(y[index], selected[index]), evaluate_calls(y[index], baseline[index])
        for name in metric_names:
            samples[name][iteration] = first[name] - second[name]
    chosen, reference = evaluate_calls(y, selected), evaluate_calls(y, baseline)
    return {
        "rows": int(len(y)), "block_sessions": length, "requested_block_sessions": int(block),
        "bootstrap_draws": int(draws), "seed": int(seed),
        "difference_sign": "positive_favors_selected",
        "interpretation": "Exploratory dependence-aware interval; dependent on the chosen block length.",
        "metrics": {name: {"selected": chosen[name], "baseline": reference[name],
                           "difference": chosen[name] - reference[name],
                           "difference_95pct_block_interval": np.quantile(samples[name], [.025, .975]).tolist()}
                    for name in metric_names},
    }