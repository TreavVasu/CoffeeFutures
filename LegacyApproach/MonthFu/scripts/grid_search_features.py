"""Grid search over feature subsets and hyperparameters, validated out of sample.

Three selection strategies are run and compared on identical pre-2022 expanding
folds:

``grouped``     Evaluate each predefined feature group on its own, then greedily
                add the group that most improves pooled out-of-fold score.
``forward``     Greedy forward selection over a screened candidate pool, scored
                only on each fold's own validation rows.
``nested``      Run the whole ``grouped`` procedure *inside* each outer fold and
                score the resulting pipeline on data the inner search never saw.
                This is the only strategy that measures whether selection itself
                generalizes, so it is the basis for ranking the approaches.

Nothing here reads the 2022+ holdout. Ranking is decided by nested outer-fold
score; the holdout is reported afterwards as a one-shot observation.

Both tasks are covered. Returns minimize RMSE; directions optimize macro F1 with
balanced accuracy as the tie-break, because raw accuracy rewards a majority-class
predictor on this series.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

# Early folds contain legitimately constant and all-missing COT columns, so the
# univariate screen warns on every candidate. Those columns score as unusable and
# are filtered out anyway; the warnings drown out real progress.
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="Features .* are constant")
warnings.filterwarnings("ignore", message=".*divide by zero.*")

# BLAS threading is capped at one per worker because parallelism here comes from
# separate processes, not from threads inside one fit. Left to itself, each
# worker would spawn a full thread pool and oversubscribe the machine.
for _key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
    os.environ.setdefault(_key, "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier  # noqa: E402
from sklearn.feature_selection import (SelectKBest, f_classif,  # noqa: E402
                                       f_regression)
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

def find_project_root(start: Path) -> Path:
    """Return the directory that contains the ``MonthFu`` package.

    The repository was reorganised, so the package is not always a fixed number
    of levels below the script. Walk upwards until it is found rather than
    assuming a depth.
    """
    for candidate in [start, *start.parents]:
        if (candidate / "MonthFu" / "src").is_dir():
            return candidate
    raise FileNotFoundError(f"Could not find the MonthFu package above {start}")


ROOT = find_project_root(Path(__file__).resolve().parent)
sys.path.insert(0, str(ROOT))

from MonthFu.src.features import build_feature_frame  # noqa: E402
from MonthFu.src.modeling import add_targets, expanding_folds, mature_training_rows  # noqa: E402

HOLDOUT_START = pd.Timestamp("2022-01-01")
# Worker count for the independent candidate fits. Every estimator is seeded and
# BLAS is pinned to one thread per worker, so results do not depend on this value.
N_JOBS = max(1, min(10, (os.cpu_count() or 2)))
# The loky backend reuses its process pool across calls, which matters here
# because the search issues thousands of small fits.
_POOL = Parallel(n_jobs=N_JOBS, backend="loky", batch_size=1,
                 max_nbytes=None, timeout=None, verbose=0)
# Column order shared by every Fold. Workers receive column *positions*, not
# labels, so a candidate ships a short list of integers instead of strings.
frame_column_index: dict[str, int] = {}


# --------------------------------------------------------------------------- #
# Feature families
# --------------------------------------------------------------------------- #
def feature_families(frame: pd.DataFrame) -> dict[str, list[str]]:
    """Partition predictors by source family. Disjoint by construction."""
    families: dict[str, list[str]] = {}
    for column in frame.columns:
        if column in {"Date", "Open", "High", "Low", "Close", "Volume", "target_return",
                      "target_close", "target_end_date", "target_sessions", "forecast_steps"}:
            continue
        if column.startswith("cot_"):
            key = "cot"
        elif column.startswith("weather_"):
            key = "weather"
        elif column.startswith("interaction_"):
            key = "interaction"
        elif column.startswith("price_"):
            key = "price"
        else:
            key = "other"
        families.setdefault(key, []).append(column)
    return {name: sorted(columns) for name, columns in families.items() if columns}


# --------------------------------------------------------------------------- #
# Estimators
# --------------------------------------------------------------------------- #
def build_estimator(task: str, kind: str, params: dict) -> Pipeline:
    """Estimator pipeline. Imputation and scaling are fitted inside each fold only."""
    steps = [("impute", SimpleImputer(strategy="median", keep_empty_features=True,
                                      add_indicator=True)),
             ("scale", StandardScaler())]
    if task == "direction":
        if kind == "logistic":
            model = LogisticRegression(C=params.get("C", 1.0), class_weight="balanced",
                                       max_iter=3000, solver="lbfgs", random_state=42)
        elif kind == "hist":
            model = HistGradientBoostingClassifier(
                max_iter=params.get("n_estimators", 200),
                learning_rate=params.get("learning_rate", .05),
                max_leaf_nodes=params.get("max_leaf_nodes", 7),
                min_samples_leaf=params.get("min_samples_leaf", 40),
                l2_regularization=params.get("l2", 1.0),
                early_stopping=False, random_state=42)
        else:
            model = ExtraTreesClassifier(
                n_estimators=params.get("n_estimators", 200),
                max_depth=params.get("max_depth", 8),
                min_samples_leaf=params.get("min_samples_leaf", 30),
                max_features=params.get("max_features", .5),
                class_weight="balanced_subsample", n_jobs=1, random_state=42)
    else:
        from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
        from sklearn.linear_model import Ridge
        if kind == "ridge":
            model = Ridge(alpha=params.get("alpha", 1000.))
        elif kind == "hist":
            model = HistGradientBoostingRegressor(
                max_iter=params.get("n_estimators", 200),
                learning_rate=params.get("learning_rate", .05),
                max_leaf_nodes=params.get("max_leaf_nodes", 7),
                min_samples_leaf=params.get("min_samples_leaf", 40),
                l2_regularization=params.get("l2", 1.0),
                early_stopping=False, random_state=42)
        else:
            model = ExtraTreesRegressor(
                n_estimators=params.get("n_estimators", 200),
                max_depth=params.get("max_depth", 8),
                min_samples_leaf=params.get("min_samples_leaf", 30),
                max_features=params.get("max_features", .5),
                n_jobs=1, random_state=42)
    steps.append(("model", model))
    return Pipeline(steps)


def build_selected_estimator(task: str, kind: str, params: dict, k: int) -> Pipeline:
    """Wrap the estimator in an in-fold univariate screen.

    Order matters: the imputer runs first because ``SelectKBest`` cannot consume
    NaN, and the screen is fitted on training rows only, so validation labels
    never influence which columns are kept.

    The screen uses the ANOVA F statistic rather than mutual information. Both
    rank features without peeking at validation data, but ``mutual_info_*``
    re-estimates a k-nearest-neighbour density per feature and is roughly two
    orders of magnitude slower on a bank this wide, which is what previously made
    the search impractical. F is deterministic, so reruns still reproduce.
    """
    base = build_estimator(task, kind, params)
    if not k or k < 0:
        return base
    # SelectKBest calls score_func(X, y) and needs (scores, pvalues), which is
    # exactly what f_classif/f_regression already return.
    score_func = f_classif if task == "direction" else f_regression
    imputer = SimpleImputer(strategy="median", keep_empty_features=True,
                            add_indicator=True)
    scaler = StandardScaler()
    return Pipeline([("impute", imputer), ("scale", scaler),
                     ("select", SelectKBest(score_func=score_func, k=k)),
                     ("model", base.named_steps["model"])])


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def direction_labels(frame: pd.DataFrame) -> np.ndarray:
    """1 for a positive realized return, 0 for negative. Zero returns are dropped.

    A zero return has no direction, so including it would mislabel one class.
    """
    values = frame.target_return.to_numpy(dtype=float)
    return np.where(values > 0, 1, np.where(values < 0, 0, np.nan))


def score_fold(task: str, y_true: np.ndarray, prediction: np.ndarray) -> dict:
    """Score one validation block. ``higher_is_better`` drives selection."""
    if task == "return":
        error = y_true - prediction
        return {"rmse": float(np.sqrt(np.mean(error ** 2))),
                "mae": float(np.mean(np.abs(error))),
                "primary": -float(np.sqrt(np.mean(error ** 2)))}
    keep = ~np.isnan(y_true)
    truth, call = y_true[keep].astype(int), (prediction[keep] >= .5).astype(int)
    if not len(truth) or truth.min() == truth.max():
        return {"macro_f1": np.nan, "balanced_accuracy": np.nan,
                "accuracy": np.nan, "primary": -np.inf}
    up = (truth == 1)
    tp = int(((call == 1) & up).sum()); tn = int(((call == 0) & ~up).sum())
    fp = int(((call == 1) & ~up).sum()); fn = int(((call == 0) & up).sum())
    safe = lambda a, b: float(a / b) if b else 0.0
    up_p, up_r = safe(tp, tp + fp), safe(tp, tp + fn)
    dn_p, dn_r = safe(tn, tn + fn), safe(tn, tn + fp)
    macro_f1 = (safe(2 * tp, 2 * tp + fp + fn) + safe(2 * tn, 2 * tn + fp + fn)) / 2
    balanced = (up_r + dn_r) / 2
    return {"accuracy": safe(tp + tn, len(truth)), "balanced_accuracy": balanced,
            "macro_f1": macro_f1, "up_precision": up_p, "up_recall": up_r,
            "down_precision": dn_p, "down_recall": dn_r,
            # Macro F1 first, balanced accuracy as tie-break: plain accuracy
            # would reward a majority-class predictor on this series.
            "primary": macro_f1 + 1e-6 * balanced}


class Fold:
    """One expanding-window split, stored as arrays rather than DataFrames.

    Worker processes receive these by value, so keeping them as plain numpy
    arrays avoids shipping a wide DataFrame (with its index and dtypes) to every
    one of thousands of tasks. The frame is 36 MB, and pickling it per task is
    what exhausted memory and killed the worker pool.
    """

    __slots__ = ("x_train", "y_train", "x_valid", "y_valid")

    def __init__(self, x_train, y_train, x_valid, y_valid):
        self.x_train = x_train
        self.y_train = y_train
        self.x_valid = x_valid
        self.y_valid = y_valid


def make_folds(frame: pd.DataFrame, columns: list[str], folds, task: str) -> list[Fold]:
    """Materialize each split once, reusing one column ordering across folds."""
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ValueError(f"Feature columns absent from the frame: {missing[:5]}")
    values = frame[columns].to_numpy(dtype=np.float64)
    globals()["frame_column_index"] = {c: i for i, c in enumerate(columns)}
    out = []
    for train, valid in folds:
        rows_train = train.index.to_numpy()
        rows_valid = valid.index.to_numpy()
        if task == "return":
            target = frame.target_return.to_numpy(dtype=float)
        else:
            target = direction_labels(frame)
        out.append(Fold(values[rows_train], target[rows_train],
                        values[rows_valid], target[rows_valid]))
    return out


def evaluate(folds: list[Fold], task: str, columns: list[str], kind: str,
             params: dict, k: int) -> dict:
    """Fit on train, score on valid. All fitting happens inside this call."""
    if not columns or not folds:
        return {"primary": -np.inf}
    positions = [frame_column_index[c] for c in columns]
    model = build_selected_estimator(task, kind, params, k)
    scores = []
    for fold in folds:
        if task == "return":
            y_train = fold.y_train
            ok = ~np.isnan(y_train)
            if ok.sum() < 50:
                return {"primary": -np.inf}
            model.fit(fold.x_train[ok][:, positions], y_train[ok])
            scores.append(score_fold(task, fold.y_valid,
                                     model.predict(fold.x_valid[:, positions])))
        else:
            y_train = fold.y_train
            ok = ~np.isnan(y_train)
            if ok.sum() < 50 or len(np.unique(y_train[ok])) < 2:
                return {"primary": -np.inf}
            model.fit(fold.x_train[ok][:, positions], y_train[ok].astype(int))
            scores.append(score_fold(task, fold.y_valid,
                                     model.predict_proba(fold.x_valid[:, positions])[:, 1]))
    usable = [s for s in scores if np.isfinite(s.get("primary", np.nan))]
    if not usable:
        return {"primary": -np.inf}
    out = {"primary": float(np.mean([s["primary"] for s in usable])),
           "folds_scored": len(usable)}
    for key in ("rmse", "macro_f1", "balanced_accuracy", "accuracy"):
        values = [s[key] for s in usable if key in s and np.isfinite(s[key])]
        if values:
            out[key] = float(np.mean(values))
    return out


# --------------------------------------------------------------------------- #
# Selection strategies
# --------------------------------------------------------------------------- #
GRID: dict[str, list[tuple[str, dict]]] = {
    "direction": [
        ("logistic", {"C": .01}), ("logistic", {"C": .1}),
        ("logistic", {"C": 1.0}),
        ("hist", {"learning_rate": .03, "max_leaf_nodes": 7, "min_samples_leaf": 40, "l2": 1.}),
        ("hist", {"learning_rate": .05, "max_leaf_nodes": 15, "min_samples_leaf": 20, "l2": .5}),
        ("hist", {"learning_rate": .02, "max_leaf_nodes": 31, "min_samples_leaf": 60, "l2": 5.}),
        ("forest", {"max_depth": 5, "min_samples_leaf": 40, "max_features": .4}),
        ("forest", {"max_depth": 8, "min_samples_leaf": 20, "max_features": .6}),
        ("forest", {"max_depth": None, "min_samples_leaf": 60, "max_features": .3}),
    ],
    "return": [
        ("ridge", {"alpha": 100.}), ("ridge", {"alpha": 1000.}),
        ("ridge", {"alpha": 10000.}),
        ("hist", {"learning_rate": .03, "max_leaf_nodes": 7, "min_samples_leaf": 40, "l2": 1.}),
        ("hist", {"learning_rate": .05, "max_leaf_nodes": 15, "min_samples_leaf": 20, "l2": .5}),
        ("forest", {"max_depth": 5, "min_samples_leaf": 40, "max_features": .4}),
        ("forest", {"max_depth": 8, "min_samples_leaf": 20, "max_features": .6}),
        ("forest", {"max_depth": None, "min_samples_leaf": 60, "max_features": .3}),
    ],
}
SCREEN_CHOICES = [0, 30, 60, 120]


def _candidate_specs(task: str, columns: list[str]) -> list[tuple[str, dict, int]]:
    """Every (kind, params, screen) combination to try for one column set."""
    specs = []
    for kind, params in GRID[task]:
        for k in SCREEN_CHOICES:
            if k and k > len(columns):
                continue
            specs.append((kind, params, k))
    return specs


def best_candidate(task: str, folds: list[Fold], columns: list[str], log: list,
                   label: str) -> dict:
    """Best (kind, params, screen) for one column set, searched on validation only.

    Candidates are independent, so they are scored in parallel. Every estimator is
    seeded and BLAS is pinned to one thread per worker, so the winner does not
    depend on the worker count.
    """
    specs = _candidate_specs(task, columns)
    if not specs:
        return {"primary": -np.inf, "kind": None, "params": {}, "k": 0, "label": None}
    jobs = _POOL(
        delayed(evaluate)(folds, task, columns, kind, params, k)
        for kind, params, k in specs)
    best = {"primary": -np.inf, "kind": None, "params": {}, "k": 0, "label": None}
    for (kind, params, k), score in zip(specs, jobs):
        primary = score.get("primary", -np.inf)
        log.append({"stage": "candidate", "set": label, "kind": kind,
                    "params": params, "k": k, "score": primary})
        if primary > best["primary"]:
            best.update({"primary": primary, "kind": kind, "params": params, "k": k,
                         "label": f"{kind}{params}k{k}", **score})
    return best


def select_grouped(task: str, folds: list[Fold], families: dict[str, list[str]],
                   log: list) -> dict:
    """Score each family alone, then greedily add the best remaining family."""
    singles = {}
    for name, columns in families.items():
        best = best_candidate(task, folds, columns, log, label=f"group:{name}")
        singles[name] = best
        log.append({"stage": "group_alone", "family": name,
                    "score": best["primary"], "spec": best["label"]})
    chosen, order, score = [], [], -np.inf
    while len(chosen) < len(families):
        gains = []
        for name, columns in families.items():
            if name in chosen:
                continue
            columns = columns + sum((families[c] for c in chosen), [])
            best = best_candidate(task, folds, columns, log, label=f"+{name}")
            gains.append((best["primary"], name, best))
        if not gains:
            break
        gains.sort(key=lambda item: (-item[0], item[1]))
        top, name, best = gains[0]
        if top <= score + 1e-9 and chosen:
            log.append({"stage": "group_stop", "added": None, "score": score})
            break
        score, spec = top, best
        chosen.append(name)
        order.append(name)
        log.append({"stage": "group_add", "family": name, "score": score,
                    "columns": len(sum((families[c] for c in chosen), []))})
    columns = sum((families[c] for c in chosen), [])
    return {"families": order, "columns": columns, "score": score, "spec": spec,
            "group_alone": {k: v["primary"] for k, v in singles.items()}}


def select_forward(task: str, frame: pd.DataFrame, folds: list[Fold],
                   families: dict[str, list[str]], pool_size: int,
                   max_steps: int, log: list) -> dict:
    """Greedy forward selection from a screened candidate pool.

    ``pool_size`` is how many *features* to consider, not a column label. The
    candidate pool is every predictor in the frame; it is ranked once by a cheap
    univariate statistic computed on the **earliest training fold only**. Using
    one fixed fold to build the pool keeps later validation folds from
    influencing which features are even considered.

    Missing values are median-imputed for this ranking only, using medians from
    the same training rows. The F statistics cannot consume NaN, and a median
    fill here cannot leak: no validation row and no future row contributes.
    """
    candidates = sorted(sum(families.values(), []))
    earliest = folds[0]
    y_train = earliest.y_train
    ok = ~np.isnan(y_train)
    target = y_train[ok]
    if task == "direction" and len(np.unique(target)) < 2:
        return {"columns": [], "score": -np.inf, "families": [], "spec": None}
    screen = SimpleImputer(strategy="median", keep_empty_features=True)
    positions = [frame_column_index[c] for c in candidates]
    filled = screen.fit_transform(earliest.x_train[ok][:, positions])
    if task == "direction":
        # f_classif returns (F, p); rank on the F statistic alone.
        scored = f_classif(filled, target.astype(int))[0]
    else:
        scored = f_regression(filled, target)[0]
    ranked = pd.Series(np.nan_to_num(scored, nan=-np.inf),
                       index=candidates).sort_values(ascending=False)
    pool = list(ranked.index[:pool_size])
    chosen, best_score, spec = [], -np.inf, None
    while len(chosen) < max_steps:
        gains = []
        for column in pool:
            if column in chosen:
                continue
            candidate = chosen + [column]
            result = best_candidate(task, folds, candidate, log,
                                    label=f"fwd+{column[:28]}")
            gains.append((result["primary"], column, result))
        if not gains:
            break
        gains.sort(key=lambda item: (-item[0], item[1]))
        top, column, result = gains[0]
        if top <= best_score + 1e-9 and chosen:
            log.append({"stage": "forward_stop", "score": best_score})
            break
        best_score, spec = top, result
        chosen.append(column)
        log.append({"stage": "forward_add", "feature": column, "score": best_score,
                    "steps": len(chosen)})
    return {"families": sorted({next((f for f, cols in families.items() if column in cols),
                                     "other") for column in chosen}),
            "columns": chosen, "score": best_score, "spec": spec}


# --------------------------------------------------------------------------- #
# Nested validation and reporting
# --------------------------------------------------------------------------- #
def nested_score(task: str, frame: pd.DataFrame, families: dict, outer_folds: int,
                 test_size: int, log: list) -> dict:
    """Run grouped selection inside each outer fold and score on held-out rows.

    This is the only procedure that measures whether *selection* generalizes. The
    inner search sees data up to the outer validation start; the outer block is
    scored by a model whose feature set and hyperparameters the inner search never
    observed.
    """
    outer = expanding_folds(frame, HOLDOUT_START, splits=outer_folds, test_size=test_size)
    all_columns = sorted(sum(families.values(), []))
    picks, scores = [], []
    for index, (inner_train, inner_valid) in enumerate(outer):
        inner_frame = frame.loc[frame.Date.le(inner_valid.Date.max())].reset_index(drop=True)
        inner = expanding_folds(inner_frame, HOLDOUT_START + pd.Timedelta(days=1),
                                splits=3, test_size=max(120, test_size // 3))
        if len(inner) < 2:
            continue
        # The inner search and the outer scoring share one column ordering, so the
        # fold bundles stay comparable across the two stages.
        inner_bundles = make_folds(inner_frame, all_columns, inner, task)
        picked = select_grouped(task, inner_bundles, families, log)
        if not picked["columns"]:
            continue
        outer_bundles = make_folds(frame.loc[frame.Date.ge(inner_train.Date.min())],
                                  all_columns,
                                  [(inner_train, inner_valid)], task)
        spec = picked["spec"]
        result = evaluate(outer_bundles, task, picked["columns"],
                          spec["kind"], spec["params"], spec["k"])
        if np.isfinite(result.get("primary", np.nan)):
            scores.append(result)
            picks.append({"outer_fold": index, "families": picked["families"],
                          "columns": len(picked["columns"]),
                          "spec": spec["label"], **result})
        print(f"  nested outer fold {index + 1}/{len(outer)}: "
              f"{picked['families']} -> {spec['label']}", flush=True)
    if not scores:
        return {"outer_folds": 0}
    summary = {"outer_folds": len(scores),
               "selected_families": [p["families"] for p in picks]}
    for key in ("macro_f1", "balanced_accuracy", "accuracy", "rmse"):
        values = [s[key] for s in scores if key in s and np.isfinite(s[key])]
        if values:
            summary[f"mean_{key}"] = float(np.mean(values))
            summary[f"std_{key}"] = float(np.std(values))
    return summary


def holdout_report(task: str, frame: pd.DataFrame, result: dict) -> dict:
    """Fit the frozen choice on all pre-2022 data and score the 2022+ holdout once.

    Reported for completeness only. This period has been inspected in earlier
    work, so it is not an untouched test and cannot select between approaches.
    """
    if not result.get("columns") or not result.get("spec"):
        return {}
    train = mature_training_rows(frame, HOLDOUT_START)
    hold = frame.loc[frame.Date.ge(HOLDOUT_START) & frame.target_return.notna()]
    if not len(train) or not len(hold):
        return {}
    bundles = make_folds(frame, result["columns"], [(train, hold)], task)
    spec = result["spec"]
    scored = evaluate(bundles, task, result["columns"], spec["kind"],
                      spec["params"], spec["k"])
    return {"holdout_rows": len(hold), "train_rows": len(train),
            "families": result.get("families"), "columns": len(result["columns"]),
            "spec": spec["label"], **scored}


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", choices=["direction", "return", "both"], default="both")
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--splits", type=int, default=4)
    parser.add_argument("--test-size", type=int, default=504)
    parser.add_argument("--outer-folds", type=int, default=4)
    parser.add_argument("--forward-pool", type=int, default=25)
    parser.add_argument("--forward-steps", type=int, default=6)
    parser.add_argument("--jobs", type=int, default=N_JOBS,
                        help="worker processes for the independent candidate fits")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "MonthFu/artifacts/grid_search")
    args = parser.parse_args()
    if args.jobs != N_JOBS:
        # Rebuild the shared pool at the requested width.
        globals()["N_JOBS"] = max(1, args.jobs)
        globals()["_POOL"] = Parallel(n_jobs=max(1, args.jobs), backend="loky",
                                       batch_size=1, max_nbytes=None,
                                       timeout=None, verbose=0)

    started = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    base, _, _ = build_feature_frame(ROOT)
    frame = add_targets(base, args.horizon, "calendar")
    families = feature_families(frame)
    print(f"families: { {k: len(v) for k, v in families.items()} }", flush=True)

    tasks = ["direction", "return"] if args.task == "both" else [args.task]
    summary = {"horizon_days": args.horizon, "splits": args.splits,
               "test_size": args.test_size, "outer_folds": args.outer_folds,
               "family_sizes": {k: len(v) for k, v in families.items()},
               "selection_policy": (
                   "Grouped, forward and nested strategies are ranked by nested "
                   "outer-fold score. The 2022+ holdout is reported once and never "
                   "used to choose between approaches."),
               "tasks": {}}

    for task in tasks:
        print(f"\n=== task: {task} ===", flush=True)
        raw_folds = expanding_folds(frame, HOLDOUT_START, splits=args.splits,
                                    test_size=args.test_size)
        all_columns = sorted(sum(families.values(), []))
        folds = make_folds(frame, all_columns, raw_folds, task)
        log: list = []
        record: dict = {"folds": len(folds)}

        print("-- grouped --", flush=True)
        grouped = select_grouped(task, folds, families, log)
        spec = grouped["spec"]
        print(f"   families={grouped['families']} spec={spec['label']}", flush=True)
        record["grouped"] = {
            "families": grouped["families"], "columns": len(grouped["columns"]),
            "spec": spec["label"],
            "cv": evaluate(folds, task, grouped["columns"], spec["kind"],
                           spec["params"], spec["k"]),
            "group_alone": grouped.get("group_alone")}

        print("-- forward --", flush=True)
        forward = select_forward(task, frame, folds, families,
                                 args.forward_pool, args.forward_steps, log)
        fspec = forward["spec"]
        record["forward"] = {
            "families": forward["families"], "columns": len(forward["columns"]),
            "spec": fspec["label"] if fspec else None,
            "chosen": forward["columns"][:40],
            "cv": (evaluate(folds, task, forward["columns"], fspec["kind"],
                            fspec["params"], fspec["k"]) if fspec else {})}

        print("-- nested (ranking basis) --", flush=True)
        record["nested_grouped"] = nested_score(task, frame, families,
                                                args.outer_folds, args.test_size, log)
        record["holdout"] = {"grouped": holdout_report(task, frame, grouped),
                             "forward": holdout_report(task, frame, forward)}

        nested = record["nested_grouped"]
        record["rank_basis"] = ("nested outer-fold macro F1" if task == "direction"
                                else "nested outer-fold negative RMSE")
        summary["tasks"][task] = record
        if nested.get("outer_folds"):
            print(f"   nested mean: "
                  f"{ {k: round(v, 4) for k, v in nested.items() if k.startswith('mean_')} }",
                  flush=True)

    summary["elapsed_seconds"] = time.perf_counter() - started
    (args.output_dir / "grid_search_summary.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n")
    print(f"\nwrote {args.output_dir/'grid_search_summary.json'} "
          f"in {summary['elapsed_seconds']:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())






