"""Separate UP/DOWN direction models over a (lookback, forward) grid.

This is a follow-up to ``train_direction.py``.  Where that script fits one
``P(up) vs P(down)`` head per horizon, this one fits **two independent heads**,
one for each side of the direction, over the expanded
``(lookback, forward)`` grid in ``window_dataset``.

Staged source ablation
----------------------
The feature sources are added in a fixed order and each stage is scored before
the next one runs, so the contribution of COT and weather is measured rather
than assumed:

======  ==========================================  =====================
Stage   Sources                                      Question it answers
======  ==========================================  =====================
S0      price + candlestick                          baseline, no COT/weather
S1      S0 + full COT bank                           does positioning help?
S2      S0 + full weather bank                       does weather help?
S3      S0 + COT + weather                           combined effect
======  ==========================================  =====================

COT and weather are used with their published availability rules unchanged: a COT
report enters on the first observed session strictly after its publication date,
and weather is delayed by five calendar days with a seven-day carry limit.  The
stages therefore differ *only* in which columns are admitted, never in timing.

Parallelism
-----------
Grid cells, stages, folds and candidate models are all independent fits, so the
work is farmed out to a process pool.  Each worker imports its own copy of the
feature frame once via an ``initializer`` and then receives only lightweight task
tuples.  BLAS threading is pinned to one thread per worker and every estimator
is built with ``n_jobs=1``, so the pool scales by process rather than
oversubscribing cores.

Protocol
--------
Expanding-window cross-validation runs entirely before 2022.  Each polarity's
candidate and threshold are selected on those out-of-fold rows and written to
``selection.json`` **before** any holdout fit.  Quarterly walk-forward refits
then use only labels that have matured.  Uncertainty uses a moving-block
bootstrap whose block length covers the cell's forward offset.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
import time

# Pin BLAS threading before numpy is imported anywhere in the worker.
for _key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "LOKY_MAX_CPU_COUNT", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
    os.environ[_key] = "1"

ROOT = Path(__file__).resolve().parents[3]
EXPERIMENTS = ROOT / "MonthFu" / "experiments"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXPERIMENTS / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from candlestick_patterns import (  # noqa: E402
    KNOWN_NAME_FORMULA_MISMATCH, PATTERN_NAMES, add_candlestick_features,
)
from direction_metrics import binary_labels_from_returns  # noqa: E402
from polarity_metrics import (  # noqa: E402
    block_bootstrap_difference, decide, evaluate_calls, polarity_report,
    select_side_threshold, select_threshold_pair,
)
from polarity_models import (  # noqa: E402
    DirectionSpec, candidates, fit_direction, polarity_banks, predict_direction,
)
from window_dataset import (  # noqa: E402
    DEFAULT_FORWARDS, DEFAULT_LOOKBACKS, Cell, cell_frame,
)

HOLDOUT_START = pd.Timestamp("2022-01-01")
CANDLE_WINDOWS = DEFAULT_LOOKBACKS

# Ordered ablation.  ``sources`` lists the prefixes admitted at that stage.
STAGES: dict[str, dict] = {
    "S0_price_candles": {"sources": ("price",), "label": "price + candlestick"},
    "S1_plus_cot": {"sources": ("price", "cot"), "label": "price + candlestick + COT"},
    "S2_plus_weather": {"sources": ("price", "weather"), "label": "price + candlestick + weather"},
    "S3_plus_all": {"sources": ("price", "cot", "weather"), "label": "price + candlestick + COT + weather"},
}
STAGE_ORDER = ["S0_price_candles", "S1_plus_cot", "S2_plus_weather", "S3_plus_all"]

GROUPS = ["basic", "candles", "engineered"]


def ready(value):
    """JSON-safe conversion matching the existing experiment scripts."""
    if isinstance(value, dict):
        return {str(k): ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [ready(v) for v in value]
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (pd.Timestamp, Path)):
        return str(value)
    return value


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ready(value), indent=2, allow_nan=False) + "\n")


def output_root(quick: bool) -> Path:
    return EXPERIMENTS / "artifacts" / ("polarity_quick" if quick else "polarity")


def base_frame_path() -> Path:
    return EXPERIMENTS / "artifacts" / "polarity_cache" / "base_features.pkl"


def build_base_frame(repo_root: Path, refresh: bool = False) -> tuple[pd.DataFrame, dict]:
    """Build the shared predictor frame once and cache it.

    The MonthFu pipeline (price + COT + weather) takes minutes to build and does
    not depend on the ablation stage, so every stage reuses this one frame and
    differs only by which columns it admits.
    """
    from MonthFu.src.features import build_feature_frame
    from MonthFu.src.modeling import mature_training_rows  # noqa: F401

    cache = base_frame_path()
    if cache.exists() and not refresh:
        payload = pickle.loads(cache.read_bytes())
        print(f"[frame] cache hit: {payload['frame'].shape[0]} rows, "
              f"{payload['frame'].shape[1]} columns", flush=True)
        return payload["frame"], payload["metadata"]
    started = time.perf_counter()
    frame, manifest, audits = build_feature_frame(repo_root)
    frame, candle_names = add_candlestick_features(frame, windows=CANDLE_WINDOWS)
    metadata = {
        "built_seconds": time.perf_counter() - started,
        "price": audits["metadata"].get("price", {}),
        "cot": audits["metadata"].get("cot", {}),
        "external": audits["metadata"].get("external", {}),
        "feature_counts": audits["metadata"].get("feature_counts", {}),
        "candle_columns": candle_names,
        "candle_windows": list(CANDLE_WINDOWS),
        "candle_patterns": list(PATTERN_NAMES),
        "known_name_formula_mismatch": KNOWN_NAME_FORMULA_MISMATCH,
        "signal_time": "After the observed session close; a feature on date t uses only bars through t.",
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(pickle.dumps({"frame": frame, "metadata": metadata}))
    print(f"[frame] built {frame.shape[0]} rows x {frame.shape[1]} columns "
          f"in {metadata['built_seconds']:.0f}s", flush=True)
    return frame, metadata


def shared_columns(frame: pd.DataFrame, stage: str) -> list[str]:
    """Non-candle predictors admitted at this stage, by source prefix."""
    if stage not in STAGES:
        raise ValueError(f"Unknown stage {stage!r}; choose from {sorted(STAGES)}.")
    sources = STAGES[stage]["sources"]
    numeric = pd.api.types.is_numeric_dtype
    columns = []
    for column in frame.columns:
        if column.startswith("candle_") or not numeric(frame[column]):
            continue
        if any(column.startswith(f"{source}_") for source in sources):
            columns.append(column)
    return columns


def expanding_validation_folds(frame: pd.DataFrame, holdout_start: pd.Timestamp,
                               splits: int = 4, test_size: int = 504,
                               min_train: int = 1000) -> list[dict]:
    """Purged expanding folds; every training label matures before its fold.

    ``test_size`` counts origins, not sessions of label span, so the purge is
    applied explicitly through ``target_end_date < cutoff``.
    """
    development = frame.loc[frame.target_return.notna()
                            & frame.target_end_date.lt(holdout_start)].reset_index(drop=True)
    first = len(development) - splits * test_size
    if first < min_train:
        raise ValueError("Not enough development history for the requested folds.")
    folds = []
    for number, start in enumerate(range(first, len(development), test_size), start=1):
        valid = development.iloc[start:start + test_size]
        if not len(valid):
            continue
        cutoff = valid.Date.iloc[0]
        train = frame.loc[frame.target_return.notna()
                          & frame.target_end_date.lt(cutoff)
                          & frame.Date.lt(cutoff)]
        if len(train) < min_train:
            raise ValueError(f"Fold {number} leaves only {len(train)} mature training rows.")
        folds.append({"fold": number, "train": train, "valid": valid})
    return folds


# --------------------------------------------------------------------------
# Worker side.  The frame is loaded once per process through the pool
# initializer, so every task carries only a small tuple and no DataFrame is
# pickled per fit.
# --------------------------------------------------------------------------
_WORKER: dict = {}


def cell_name(key) -> str:
    """Stable string label for a (lookback, forward) cell."""
    return f"L{key[0]}_k{key[1]}"


def _init_worker(cache_path: str, lookbacks: list[int], forwards: list[int],
                 quick: bool, splits: int, test_size: int, min_train: int) -> None:
    payload = pickle.loads(Path(cache_path).read_bytes())
    _WORKER["base"] = payload["frame"]
    _WORKER["candles"] = payload["metadata"]["candle_columns"]
    _WORKER["cells"] = {(l, k): Cell(l, k) for l in lookbacks for k in forwards}
    _WORKER["quick"] = quick
    _WORKER["folds"] = {}
    _WORKER["bank_cache"] = {}
    _WORKER["fold_args"] = (splits, test_size, min_train)


def _cell_frame(key) -> pd.DataFrame:
    """Per-process cache of one cell's labelled frame.

    The cell frame is a deterministic function of the base frame and the cell,
    so each worker builds it once and reuses it across every fold and candidate
    assigned to that worker.
    """
    cache = _WORKER.setdefault("cell_cache", {})
    if key not in cache:
        cache[key] = cell_frame(_WORKER["base"], _WORKER["cells"][key])
    return cache[key]


def _cell_folds(key) -> dict:
    if key not in _WORKER["folds"]:
        frame = _cell_frame(key)
        folds = {}
        splits, test_size, min_train = _WORKER["fold_args"]
        for fold in expanding_validation_folds(frame, HOLDOUT_START, splits, test_size, min_train):
            folds[fold["fold"]] = (fold["train"].index.to_numpy(),
                                   fold["valid"].index.to_numpy())
        _WORKER["folds"][key] = folds
    return _WORKER["folds"][key]


def _candle_subset(key) -> list[str]:
    """Candle columns for one cell: untagged bars plus this cell's own window."""
    tag = f"@{_WORKER['cells'][key].lookback}"
    return [c for c in _WORKER["candles"] if "@" not in c or c.endswith(tag)]


def _banks(stage: str, key, polarity: str) -> dict[str, list[str]]:
    cache_key = (stage, key, polarity)
    if cache_key not in _WORKER["bank_cache"]:
        shared = [c for c in shared_columns(_WORKER["base"], stage) if c != "Close"]
        _WORKER["bank_cache"][cache_key] = polarity_banks(_candle_subset(key), shared, polarity)
    return _WORKER["bank_cache"][cache_key]


def _cell_holdout(key, holdout_start: pd.Timestamp) -> dict:
    """Quarterly walk-forward blocks over the holdout period.

    A quarter is refitted on every row whose label has *matured* strictly before
    the quarter opens, so a training set never contains an outcome from inside
    the block it predicts.  This is walk-forward evaluation, not a fixed
    train/test split.
    """
    cache_key = ("holdout", key)
    if cache_key in _WORKER.get("holdout_cache", {}):
        return _WORKER["holdout_cache"][cache_key]
    frame = _cell_frame(key).copy()
    frame["Date"] = pd.to_datetime(frame.Date)
    horizon = _WORKER["cells"][key].forward
    blocks = {}
    hold = frame[frame.Date >= holdout_start].sort_values("Date")
    for quarter, group in hold.groupby(hold.Date.dt.to_period("Q"), sort=True):
        valid_index = group.index.to_numpy()
        cutoff = frame.Date.iloc[valid_index[0]]
        train_index = frame.index[(frame.target_return.notna()
                                   & frame.target_end_date.lt(cutoff)
                                   & frame.Date.lt(cutoff))].to_numpy()
        # Every label must end at least one full horizon before the block opens,
        # so no training outcome overlaps the first predicted window.
        if len(train_index) < _WORKER["fold_args"][2]:
            continue
        blocks[str(quarter)] = (train_index, valid_index, int(horizon))
    _WORKER.setdefault("holdout_cache", {})[cache_key] = blocks
    return blocks


def fit_holdout(task: tuple) -> dict:
    """Refit one frozen pair on matured labels and score one holdout quarter.

    The candidate and both thresholds come from the pre-holdout decision and are
    never revisited here; only the weights are re-estimated as labels mature.
    """
    stage, key, quarter, up_name, down_name = task
    try:
        frame = _cell_frame(key)
        blocks = _cell_holdout(key, HOLDOUT_START)
        if quarter not in blocks:
            return {"stage": stage, "cell": cell_name(key), "quarter": quarter,
                    "skipped": "no matured training rows"}
        train_index, valid_index, _ = blocks[quarter]
        up = DirectionSpec(*_parse_spec(up_name))
        down = DirectionSpec(*_parse_spec(down_name))
        banks_up = _banks(stage, key, "up")
        banks_down = _banks(stage, key, "down")
        train, valid = frame.loc[train_index], frame.loc[valid_index]
        member_up = fit_direction(up, train, banks_up[up.group], quick=_WORKER["quick"])
        member_down = fit_direction(down, train, banks_down[down.group], quick=_WORKER["quick"])
        up_scores = predict_direction(member_up, valid)
        down_scores = predict_direction(member_down, valid)
        if len(up_scores) != len(valid) or len(down_scores) != len(valid):
            raise ValueError("Holdout score length does not match the predicted origins.")
        return {"stage": stage, "cell": cell_name(key), "cell_key": key, "quarter": quarter,
                "valid_index": valid_index, "up": up_scores, "down": down_scores,
                "train_rows": int(len(train_index)),
                "audit": {"stage": stage, "cell": cell_name(key), "quarter": quarter,
                          "up_candidate": up_name, "down_candidate": down_name,
                          "up_features": len(member_up["features"]),
                          "down_features": len(member_down["features"]),
                          "train_rows": int(len(train_index))}}
    except Exception as error:  # noqa: BLE001 - reported, never hidden
        return {"stage": stage, "cell": cell_name(key), "cell_key": key, "quarter": quarter,
                "error": f"{type(error).__name__}: {error}"}


def run_holdout(pool, stages, cells, frozen, batch_size: int = 32) -> list[dict]:
    """Refit every frozen (stage, cell) pair across the holdout quarters."""
    tasks = []
    for stage in stages:
        for cell in cells:
            name = cell_name(cell)
            decision = frozen.get(f"{stage}|{name}")
            if not decision:
                print(f"[holdout] no frozen decision for {stage}|{name}; skipped", flush=True)
                continue
            up = decision["selection"]["up"]["candidate"]
            down = decision["selection"]["down"]["candidate"]
            for quarter in decision["quarters"]:
                tasks.append((stage, cell, quarter, up, down))
    print(f"[holdout] {len(tasks)} quarterly refits "
          f"(2 frozen models each) across {len(cells)} cells", flush=True)
    if not tasks:
        return []
    batches = list(_chunks(tasks, max(1, batch_size)))
    started, results, done = time.perf_counter(), [], 0
    for number, batch in enumerate(batches, start=1):
        for future in as_completed([pool.submit(fit_holdout, task) for task in batch]):
            results.append(future.result())
        done += len(batch)
        print(f"[holdout] batch {number}/{len(batches)}  {done}/{len(tasks)} refits "
              f"({done / max(time.perf_counter() - started, 1e-9):.1f}/s)", flush=True)
    return results


def collect_holdout(results) -> pd.DataFrame:
    """One row per predicted origin, with both frozen scores attached.

    Labels and scores are accumulated per quarter and only then joined, because
    an inner join against a single quarter's label frame silently discards every
    later quarter - which is exactly what an earlier version of this function
    did, leaving 3 of 19 quarters in the artifact while still looking non-empty.
    """
    labels: dict[tuple, list] = {}
    scores: dict[tuple, list] = {}
    audits, failures = [], []
    for result in results:
        if "error" in result:
            failures.append({k: result[k] for k in ("stage", "cell", "quarter", "error")})
            continue
        if "skipped" in result:
            continue
        audits.append(result["audit"])
        key = (result["stage"], result["cell"])
        frame = _cell_frame(result["cell_key"]).loc[result["valid_index"]]
        # Both frames carry the cell frame's own index labels so the join below
        # is on the origin, not on position.
        labels.setdefault(key, []).append(pd.DataFrame({
            "stage": result["stage"], "cell": result["cell"], "quarter": result["quarter"],
            "Date": frame.Date.to_numpy(), "target_end_date": frame.target_end_date.to_numpy(),
            "target_return": frame.target_return.to_numpy(),
        }, index=frame.index))
        scores.setdefault(key, []).append(pd.DataFrame(
            {"p_up": result["up"], "p_down": result["down"]}, index=frame.index))
    if not labels:
        raise ValueError("Every holdout refit failed; there is nothing to score.")
    parts = []
    for key, blocks in labels.items():
        joined = (pd.concat(blocks).sort_index()
                  .join(pd.concat(scores[key]).sort_index(), how="inner"))
        if joined.empty:
            raise ValueError(f"{key} produced no matched holdout rows; "
                             "the label and score frames must share an index.")
        parts.append(joined)
    holdout = pd.concat(parts, ignore_index=True)
    holdout = holdout[holdout.target_return.notna()].sort_values("Date").reset_index(drop=True)
    duplicated = holdout.duplicated(subset=["stage", "cell", "Date"]).sum()
    if duplicated:
        raise ValueError(f"{duplicated} duplicated holdout origins survived the pivot.")
    if not len(holdout):
        raise ValueError("The holdout join produced zero rows; refusing to report an empty result.")
    quarters_seen = holdout.quarter.nunique()
    expected = max((len(b) for b in labels.values()), default=0)
    if quarters_seen < expected:
        raise ValueError(f"only {quarters_seen} of {expected} holdout quarters survived "
                         "the join; the label frames are incomplete.")
    return holdout, pd.DataFrame(audits), pd.DataFrame(failures)


def score_holdout(holdout: pd.DataFrame, stage: str, cell: str, chosen: dict,
                  block: int, draws: int) -> dict:
    """Score one frozen rule on the holdout, against the always-UP reference."""
    rows = holdout[(holdout.stage == stage) & (holdout.cell == cell)].sort_values("Date")
    if not len(rows):
        raise ValueError(f"No holdout rows for {stage}/{cell}.")
    up_cut = chosen["up"]["threshold"]
    down_cut = chosen["down"]["threshold"]
    report = polarity_report(rows, rows.p_up.to_numpy(), rows.p_down.to_numpy(), up_cut, down_cut)
    reference = evaluate_calls(
        binary_labels_from_returns(rows.target_return)[np.isfinite(
            binary_labels_from_returns(rows.target_return))],
        np.ones(int(np.isfinite(binary_labels_from_returns(rows.target_return)).sum())))
    report.update({"stage": stage, "cell": cell, "split": "holdout_2022_plus",
                   "up_candidate": chosen["up"]["candidate"],
                   "down_candidate": chosen["down"]["candidate"],
                   "always_up_accuracy": reference["accuracy"],
                   "always_up_macro_f1": reference["macro_f1"],
                   "accuracy_vs_always_up": report["accuracy"] - reference["accuracy"]})
    labels = binary_labels_from_returns(rows.target_return)
    keep = np.isfinite(labels)
    selected = decide(rows.p_up.to_numpy()[keep], rows.p_down.to_numpy()[keep], up_cut, down_cut)
    baseline = np.ones(int(keep.sum()))
    report["bootstrap_vs_always_up"] = block_bootstrap_difference(
        labels[keep], selected, baseline, block=max(block, 20), draws=draws)
    return report


def _parse_spec(name: str) -> tuple:
    """``up__candles__xgb_1`` -> ``("up", "candles", "xgb", 1)``."""
    polarity, group, tail = name.split("__")
    family, variant = tail.rsplit("_", 1)
    return polarity, group, family, int(variant)


def fit_one(task: tuple) -> dict:
    """Fit one candidate on one fold and score that fold's validation origins.

    Failures are returned rather than raised so a single degenerate candidate
    cannot abort a long parallel run; the caller records them in the audit and
    excludes them from selection.
    """
    stage, key, fold_number, spec_name = task
    spec = DirectionSpec(*_parse_spec(spec_name))
    # A tuple would be read by pandas as array-like data, so cells travel as a name.
    name = cell_name(key)
    try:
        frame = _cell_frame(key)
        train_index, valid_index = _cell_folds(key)[fold_number]
        train, valid = frame.loc[train_index], frame.loc[valid_index]
        member = fit_direction(spec, train, _banks(stage, key, spec.polarity)[spec.group],
                               quick=_WORKER["quick"])
        return {"stage": stage, "cell": name, "cell_key": key, "fold": fold_number,
                "spec": spec_name, "valid_index": valid_index,
                "scores": predict_direction(member, valid),
                "audit": {"stage": stage, "cell": name, "fold": fold_number,
                          "spec": spec_name, "features": member["features"],
                          "feature_count": len(member["features"]),
                          "train_rows": member["train_rows"],
                          "train_last_date": member["train_last_date"],
                          "positive_rate": member["positive_rate"]}}
    except Exception as error:  # noqa: BLE001 - reported in the audit, never hidden
        return {"stage": stage, "cell": name, "cell_key": key, "fold": fold_number,
                "spec": spec_name, "error": f"{type(error).__name__}: {error}"}


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------
def _chunks(items, size):
    for start in range(0, len(items), size):
        yield items[start:start + size]


def run_cross_validation(pool, stages, cells, specs, fold_numbers,
                         batch_size: int = 64) -> list[dict]:
    """Every (stage, cell, fold, candidate) fit, fanned out across the pool.

    Work is submitted in **batches** rather than all at once.  Submitting the
    whole grid up front would materialise one Future and one queued task per fit
    - tens of thousands for the full grid - and hold every pending result in
    memory.  Batching bounds the queue, keeps peak memory flat as the grid grows,
    and gives per-batch progress that is meaningful for the slow COT and weather
    stages where a single fit carries hundreds of columns.

    Tasks are ordered stage-major then cell-major, so batches complete in
    ablation order and a batch boundary is a natural place to report progress for
    whichever source stage is running.
    """
    tasks = [(stage, cell, fold, spec.name)
             for stage in stages for cell in cells for fold in fold_numbers for spec in specs]
    batches = list(_chunks(tasks, max(1, batch_size)))
    print(f"[cv] {len(tasks)} fits across {len(cells)} cells x {len(stages)} stages "
          f"x {len(fold_numbers)} folds x {len(specs)} candidates", flush=True)
    print(f"[cv] submitting in {len(batches)} batches of {max(1, batch_size)}", flush=True)
    started = time.perf_counter()
    results, done, stage_started = [], 0, {}
    for number, batch in enumerate(batches, start=1):
        stage_started.setdefault(batch[0][0], time.perf_counter())
        for future in as_completed([pool.submit(fit_one, task) for task in batch]):
            results.append(future.result())
        done += len(batch)
        elapsed = time.perf_counter() - started
        rate = done / max(elapsed, 1e-9)
        stage_now = batch[0][0]
        stage_elapsed = time.perf_counter() - stage_started[stage_now]
        remaining = (len(tasks) - done) / max(rate, 1e-9)
        print(f"[cv] batch {number}/{len(batches)}  {done}/{len(tasks)} fits "
              f"({rate:.1f}/s)  stage={stage_now} "
              f"({len(batch) / max(stage_elapsed, 1e-9):.1f}/s in-stage)  "
              f"eta {remaining / 60:.0f}m", flush=True)
    return results


def collect_oof(results) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Pivot fold predictions into one row per origin with a column per candidate.

    Every candidate scores the same validation origins, so the label columns are
    built once per (stage, cell, fold) and each candidate's scores are attached
    as their own column.  Stacking the candidates as rows instead would repeat
    each origin once per candidate, which would corrupt every threshold search
    and every count downstream.
    """
    origins: dict[tuple, pd.DataFrame] = {}
    scores: dict[tuple, dict[str, np.ndarray]] = {}
    audits, failures = [], []
    for result in results:
        if "error" in result:
            failures.append({k: result[k] for k in ("stage", "cell", "fold", "spec", "error")})
            continue
        audits.append(result["audit"])
        key = (result["stage"], result["cell"], result["fold"])
        frame = _cell_frame(result["cell_key"]).loc[result["valid_index"]]
        values = np.asarray(result["scores"], dtype=float)
        if len(values) != len(frame):
            # A length mismatch would silently misalign every label below it.
            raise ValueError(f"{result['stage']}/{result['cell']}/{result['spec']} returned "
                             f"{len(values)} scores for {len(frame)} origins.")
        if key not in origins:
            origins[key] = pd.DataFrame({
                "stage": result["stage"], "cell": result["cell"], "fold": result["fold"],
                "Date": frame.Date.to_numpy(), "target_end_date": frame.target_end_date.to_numpy(),
                "target_return": frame.target_return.to_numpy(),
                "direction_up": frame.direction_up.to_numpy(),
                "direction_down": frame.direction_down.to_numpy(),
            })
        scores.setdefault(key, {})[result["spec"]] = values
    if not origins:
        raise ValueError("Every cross-validation fit failed; there is nothing to select on.")
    parts = [origins[key].assign(**columns) for key, columns in scores.items() if key in origins]
    oof = pd.concat(parts, ignore_index=True)
    oof = oof[oof.target_return.notna()].reset_index(drop=True)
    duplicated = oof.duplicated(subset=["stage", "cell", "fold", "Date"]).sum()
    if duplicated:
        raise ValueError(f"{duplicated} duplicated origins survived the out-of-fold pivot.")
    return oof, pd.DataFrame(audits), pd.DataFrame(failures)


def select_per_polarity(oof: pd.DataFrame, specs, stage: str, cell: str) -> tuple[dict, pd.DataFrame]:
    """Pick each side's candidate and threshold on pre-holdout out-of-fold rows.

    Each polarity is ranked by its own positive-class F1, so the two sides are
    selected independently and may end up on different families, groups or
    variants.  Ties fall back to balanced accuracy, then the candidate name, so
    the choice is deterministic.
    """
    rows = oof[(oof.stage == stage) & (oof.cell == cell)].sort_values("Date")
    if not len(rows):
        raise ValueError(f"No development rows for {stage}/{cell}.")
    ranking, chosen, decisions = [], {}, {}
    for polarity in ("up", "down"):
        side = [spec for spec in specs if spec.polarity == polarity]
        scored = []
        for spec in side:
            if spec.name not in rows:
                continue
            labels = rows[f"direction_{polarity}"].to_numpy(dtype=float)
            keep = np.isfinite(labels) & rows[spec.name].notna().to_numpy()
            if keep.sum() < 50 or len(np.unique(labels[keep])) < 2:
                continue
            threshold = select_side_threshold(
                labels[keep], rows[spec.name].to_numpy(dtype=float)[keep],
                dates=rows.Date.to_numpy()[keep],
                target_end_dates=rows.target_end_date.to_numpy()[keep],
                holdout_start=HOLDOUT_START, polarity=polarity)
            best = threshold["selection_metrics"]
            scored.append({"candidate": spec.name, "polarity": polarity,
                           "group": spec.group, "family": spec.family,
                           "variant": spec.variant,
                           "threshold": threshold["threshold"],
                           "f1": best[f"{polarity}_f1"],
                           "balanced_accuracy": best["balanced_accuracy"],
                           "macro_f1": best["macro_f1"], "rows": best["rows"]})
        if not scored:
            raise ValueError(f"No usable development predictions for the {polarity} side "
                             f"at {stage}/{cell}.")
        table = pd.DataFrame(scored).sort_values(
            ["f1", "balanced_accuracy", "candidate"], ascending=[False, False, True])
        ranking.append(table)
        winner = table.iloc[0]
        chosen[polarity] = {"candidate": winner.candidate, "group": winner.group,
                            "family": winner.family, "variant": int(winner.variant),
                            "side_threshold": float(winner.threshold)}
        decisions[polarity] = {"side_threshold": float(winner.threshold), "rows": int(winner.rows)}

    # The heads stay independent; the two decision cuts are chosen together
    # because they compose into a single rule.  See select_threshold_pair.
    labels = rows.direction_up.to_numpy(dtype=float)
    keep = (np.isfinite(labels) & rows[chosen["up"]["candidate"]].notna().to_numpy()
            & rows[chosen["down"]["candidate"]].notna().to_numpy())
    pair = select_threshold_pair(
        labels[keep],
        rows[chosen["up"]["candidate"]].to_numpy(dtype=float)[keep],
        rows[chosen["down"]["candidate"]].to_numpy(dtype=float)[keep],
        dates=rows.Date.to_numpy()[keep], target_end_dates=rows.target_end_date.to_numpy()[keep],
        holdout_start=HOLDOUT_START)
    for polarity in ("up", "down"):
        chosen[polarity]["threshold"] = pair[f"{polarity}_threshold"]
    decisions["joint"] = {k: v for k, v in pair.items() if k != "search"}
    return chosen, pd.concat(ranking, ignore_index=True), decisions


def score_oof(oof: pd.DataFrame, stage: str, cell: str, chosen: dict) -> dict:
    """Development-set score for one stage and cell, using the frozen thresholds.

    Reported before the holdout is touched so the reader can see what selection
    actually achieved out of sample.
    """
    rows = oof[(oof.stage == stage) & (oof.cell == cell)].sort_values("Date")
    up_name = chosen["up"]["candidate"]
    down_name = chosen["down"]["candidate"]
    subset = rows[rows[up_name].notna() & rows[down_name].notna()]
    if not len(subset):
        raise ValueError(f"No paired development predictions for {stage} {cell}.")
    report = polarity_report(subset, subset[up_name].to_numpy(), subset[down_name].to_numpy(),
                             chosen["up"]["threshold"], chosen["down"]["threshold"])
    report.update({"stage": stage, "cell": cell, "up_candidate": up_name,
                   "down_candidate": down_name, "split": "development_oof"})
    return report


def bootstrap_for(rows: pd.DataFrame, chosen: dict, block: int, draws: int) -> dict:
    """Block interval for the frozen rule against the simple score comparison."""
    labels = binary_labels_from_returns(rows.target_return)
    keep = np.isfinite(labels)
    up = rows[chosen["up"]["candidate"]].to_numpy()[keep]
    down = rows[chosen["down"]["candidate"]].to_numpy()[keep]
    selected = decide(up, down, chosen["up"]["threshold"], chosen["down"]["threshold"])
    baseline = np.where(up >= down, 1.0, -1.0)
    return block_bootstrap_difference(labels[keep], selected, baseline,
                                      block=block, draws=draws)


def stage_summary(reports: list[dict]) -> pd.DataFrame:
    """Aggregate cell-level development scores into one row per stage."""
    if not reports:
        return pd.DataFrame()
    table = pd.DataFrame(reports)
    keys = ["accuracy", "macro_f1", "balanced_accuracy", "coverage",
            "up_precision", "up_recall", "down_precision", "down_recall"]
    summary = table.groupby("stage")[keys].mean().reset_index()
    summary["cells"] = table.groupby("stage").size().to_numpy()
    summary["mean_rows"] = table.groupby("stage").rows.mean().to_numpy()
    summary["sources"] = [STAGES[s]["label"] for s in summary.stage]
    return summary


def print_stage_table(summary: pd.DataFrame, baseline: dict | None) -> None:
    """Human-readable accuracy summary printed after each stage."""
    if summary.empty:
        return
    header = (f"{'stage':<20}{'sources':<38}{'cells':>6}{'rows':>9}"
              f"{'acc':>8}{'macroF1':>9}{'balAcc':>8}{'cov':>7}{'upRec':>8}{'dnRec':>8}")
    print("\n" + "=" * len(header), flush=True)
    print(header, flush=True)
    print("-" * len(header), flush=True)
    for row in summary.itertuples():
        print(f"{row.stage:<20}{row.sources:<38}{row.cells:>6}{row.mean_rows:>9.0f}"
              f"{row.accuracy:>8.4f}{row.macro_f1:>9.4f}{row.balanced_accuracy:>8.4f}"
              f"{row.coverage:>7.3f}{row.up_recall:>8.4f}{row.down_recall:>8.4f}", flush=True)
    if baseline is not None:
        print("-" * len(header), flush=True)
        print(f"{'always-up reference':<20}{'(same origins, no features)':<38}{'':>6}"
              f"{baseline['rows']:>9.0f}{baseline['accuracy']:>8.4f}{baseline['macro_f1']:>9.4f}"
              f"{baseline['balanced_accuracy']:>8.4f}{'1.000':>7}"
              f"{baseline['up_recall']:>8.4f}{baseline['down_recall']:>8.4f}", flush=True)
    print("=" * len(header) + "\n", flush=True)


def source_hashes() -> dict:
    paths = sorted((EXPERIMENTS / "src").glob("*.py")) + [Path(__file__)]
    paths += [ROOT / "MonthFu/src/features.py", ROOT / "MonthFu/src/modeling.py",
              ROOT / "MonthFu/src/cot_release_features.py",
              ROOT / "MonthFu/src/external_return_features.py"]
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in paths if p.exists()}


def _available_quarters(cells) -> list[str]:
    """Holdout quarters present for every requested cell.

    Taking the intersection avoids a quarter where one cell has matured labels
    and another does not, which would make the stage comparison rest on
    different origins.
    """
    shared = None
    for cell in cells:
        quarters = set(_cell_holdout(cell, HOLDOUT_START))
        shared = quarters if shared is None else (shared & quarters)
    return sorted(shared or set())


def print_holdout_table(table: pd.DataFrame) -> None:
    """Accuracy on the untouched holdout, beside the always-UP reference."""
    if table.empty:
        return
    header = (f"{'stage':<20}{'cell':>10}{'rows':>7}{'acc':>8}{'vsUp':>8}{'macroF1':>9}"
              f"{'balAcc':>8}{'cov':>7}{'upRec':>8}{'dnRec':>8}")
    print("\n" + "=" * len(header) + "\nHOLDOUT 2022+ (frozen recipe and thresholds)\n"
          + "=" * len(header), flush=True)
    print(header, flush=True)
    print("-" * len(header), flush=True)
    for row in table.sort_values(["stage", "cell"]).itertuples():
        print(f"{row.stage:<20}{row.cell:>10}{row.rows:>7}{row.accuracy:>8.4f}"
              f"{row.accuracy_vs_always_up:>+8.4f}{row.macro_f1:>9.4f}"
              f"{row.balanced_accuracy:>8.4f}{row.coverage:>7.3f}"
              f"{row.up_recall:>8.4f}{row.down_recall:>8.4f}", flush=True)
    print("=" * len(header) + "\n", flush=True)


def _merge_csv(path: Path, frame: pd.DataFrame) -> pd.DataFrame:
    """Merge this run's rows into an existing artifact instead of replacing it.

    Stages are run separately to keep each batch tractable, so an earlier run's
    stage must survive a later one.  Overwriting here silently destroyed the
    S0 control when S2/S3 were run, which is exactly the comparison the ablation
    exists to produce.  Rows for stages present in the new frame replace the
    stored ones; every other stored stage is kept.
    """
    if not path.exists() or not len(frame):
        return frame
    try:
        previous = pd.read_csv(path)
    except Exception:  # noqa: BLE001 - an unreadable artifact is replaced, not merged
        return frame
    if "stage" not in previous.columns or "stage" not in frame.columns:
        return frame
    kept = previous[~previous.stage.isin(set(frame.stage))]
    return pd.concat([kept, frame], ignore_index=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lookbacks", nargs="+", type=int, default=list(DEFAULT_LOOKBACKS),
                        help="Trailing windows, in observed sessions.")
    parser.add_argument("--forwards", nargs="+", type=int, default=list(DEFAULT_FORWARDS),
                        help="Forward offsets, in observed sessions.")
    parser.add_argument("--stages", nargs="+", choices=STAGE_ORDER, default=STAGE_ORDER,
                        help="Ordered ablation stages to run.")
    parser.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1),
                        help="Worker processes; each pins BLAS to one thread.")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Fits submitted per pool batch; bounds the queue and peak memory.")
    parser.add_argument("--splits", type=int, default=4)
    parser.add_argument("--test-size", type=int, default=504)
    parser.add_argument("--min-train", type=int, default=1000)
    parser.add_argument("--draws", type=int, default=1000, help="Bootstrap draws per cell.")
    parser.add_argument("--quick", action="store_true", help="Small estimator budget for a smoke run.")
    parser.add_argument("--refresh-frame", action="store_true", help="Rebuild the cached feature frame.")
    parser.add_argument("--holdout", action="store_true",
                        help="After selection, refit the frozen pair on matured labels "
                             "across the 2022+ holdout quarters and score it.")
    parser.add_argument("--holdout-only", action="store_true",
                        help="Skip cross-validation and reuse the frozen decision already in "
                             "selection.json. Refits and scores only the holdout.")
    args = parser.parse_args(argv)
    # Validate the requested grid before building or caching anything.
    for lookback, forward in [(l, k) for l in args.lookbacks for k in args.forwards]:
        Cell(lookback, forward)
    return args


def always_up_reference(oof: pd.DataFrame) -> dict | None:
    """Accuracy of the trivial always-UP rule, averaged over the same rows.

    Computed per (stage, cell) on exactly the origins each report was scored on,
    then averaged.  Pooling every stage and cell together would report a far
    larger sample than any stage was actually evaluated on and make the printed
    row count incomparable with the stage rows above it.
    """
    rows = oof[oof.target_return.notna()]
    if not len(rows):
        return None
    scores, sizes = [], []
    for _, block in rows.groupby(["stage", "cell"], sort=True):
        labels = binary_labels_from_returns(block.sort_values("Date").target_return)
        keep = np.isfinite(labels)
        if not keep.any():
            continue
        scores.append(evaluate_calls(labels[keep], np.ones(int(keep.sum()))))
        sizes.append(int(keep.sum()))
    if not scores:
        return None
    keys = ["accuracy", "macro_f1", "balanced_accuracy", "up_recall", "down_recall"]
    return {"rows": float(np.mean(sizes)),
            **{key: float(np.mean([s[key] for s in scores])) for key in keys}}


def _run_holdout_stage(args, out: Path, cells, init_args, started: float) -> None:
    """Refit the frozen pair on matured labels and score the 2022+ holdout.

    The recipe and both thresholds are read back from ``selection.json`` and are
    never revisited.  Only the weights are re-estimated each quarter, on labels
    that matured before the quarter opened, which is walk-forward evaluation
    rather than a fresh fit.
    """
    frozen = json.loads((out / "selection.json").read_text())["cells"]
    quarters = _available_quarters(cells)
    for decision in frozen.values():
        decision["quarters"] = quarters
    if not quarters:
        raise ValueError("No holdout quarter has enough matured training rows.")
    print(f"[holdout] {len(quarters)} quarters, {quarters[0]} to {quarters[-1]}", flush=True)

    with ProcessPoolExecutor(max_workers=args.jobs, initializer=_init_worker,
                             initargs=init_args) as pool:
        holdout_results = run_holdout(pool, args.stages, cells, frozen,
                                      batch_size=max(8, args.batch_size // 2))
    holdout, holdout_audits, holdout_failures = collect_holdout(holdout_results)
    holdout.to_csv(out / "holdout_predictions.csv.gz", index=False, compression="gzip")
    holdout_audits.to_csv(out / "holdout_fit_audit.csv", index=False)
    if len(holdout_failures):
        holdout_failures.to_csv(out / "holdout_failures.csv", index=False)
    print(f"[holdout] {len(holdout)} predicted origins over {len(quarters)} quarters", flush=True)

    reports = []
    for stage in args.stages:
        for cell in cells:
            name = cell_name(cell)
            decision = frozen.get(f"{stage}|{name}")
            if not decision:
                print(f"[holdout] no frozen decision for {stage}|{name}; skipped", flush=True)
                continue
            reports.append(score_holdout(holdout, stage, name, decision["selection"],
                                         block=max(20, cell[1]), draws=args.draws))
    table = pd.DataFrame(reports)
    _merge_csv(out / "holdout_metrics.csv", table).to_csv(out / "holdout_metrics.csv", index=False)
    print_holdout_table(table)
    dump(out / "holdout_metrics.json", {
        "holdout_start": str(HOLDOUT_START.date()),
        "quarters": quarters,
        "predicted_origins": len(holdout),
        "failed_refits": len(holdout_failures),
        "protocol": ("Frozen pre-holdout candidate and threshold pair per stage and cell; "
                     "quarterly refit on matured labels only; no holdout tuning."),
        "results": [{k: v for k, v in row.items() if k != "bootstrap_vs_always_up"}
                    for row in reports]})
    print(f"[holdout done] {time.perf_counter() - started:.0f}s -> {out}", flush=True)


def main(argv=None) -> None:
    args = parse_args(argv)
    started = time.perf_counter()
    cells = [(l, k) for l in args.lookbacks for k in args.forwards]
    fold_numbers = list(range(1, args.splits + 1))

    frame, metadata = build_base_frame(ROOT, refresh=args.refresh_frame)
    out = output_root(args.quick)
    out.mkdir(parents=True, exist_ok=True)
    specs = candidates(GROUPS)
    dump(out / "candidate_parameters.json", [asdict(s) for s in specs])
    dump(out / "feature_metadata.json", {
        **metadata,
        "stages": STAGES,
        "shared_feature_counts": {stage: len(shared_columns(frame, stage)) for stage in STAGES},
        "lookbacks": args.lookbacks, "forwards": args.forwards,
        "holdout_start": str(HOLDOUT_START.date()),
        "protocol": ("Expanding purged folds wholly before the holdout; each polarity's candidate "
                     "and threshold are frozen from development out-of-fold rows only. COT enters "
                     "on the first observed session after publication; weather carries a "
                     "five-calendar-day availability lag and a seven-day maximum carry."),
    })

    # The parent reuses the worker helpers because collect_oof rebuilds the same
    # cell frames to align predictions with labels.
    init_args = (str(base_frame_path()), args.lookbacks, args.forwards, args.quick,
                 args.splits, args.test_size, args.min_train)
    _init_worker(*init_args)

    if args.holdout_only:
        # Reuse the decision already frozen in selection.json rather than
        # repeating the cross-validation search that produced it.
        print("[cv] skipped: --holdout-only reuses the frozen selection.json", flush=True)
        return _run_holdout_stage(args, out, cells, init_args, started)

    with ProcessPoolExecutor(max_workers=args.jobs, initializer=_init_worker,
                             initargs=init_args) as pool:
        results = run_cross_validation(pool, args.stages, cells, specs, fold_numbers,
                                       batch_size=args.batch_size)

    oof, audits, failures = collect_oof(results)
    oof.to_csv(out / "cv_predictions.csv.gz", index=False, compression="gzip")
    audits.to_csv(out / "fit_audit.csv", index=False)
    if len(failures):
        failures.to_csv(out / "fit_failures.csv", index=False)
        print(f"[cv] {len(failures)} fits failed and are excluded from selection", flush=True)

    # Selection is per (stage, cell): each cell has its own horizon and lookback,
    # so its development evidence is its own.  Nothing is selected on the holdout.
    reports, rankings, decisions_all, bootstraps = [], [], {}, {}
    for stage in args.stages:
        for cell in cells:
            name = cell_name(cell)
            chosen, ranking, decisions = select_per_polarity(oof, specs, stage, name)
            ranking.insert(0, "cell", name)
            rankings.append(ranking)
            decisions_all[f"{stage}|{name}"] = {"selection": chosen, "thresholds": decisions}
            reports.append(score_oof(oof, stage, name, chosen))
            rows = oof[(oof.stage == stage) & (oof.cell == name)].sort_values("Date")
            # The block must span the label overlap or the interval is too narrow.
            bootstraps[f"{stage}|{name}"] = bootstrap_for(
                rows, chosen, block=max(20, cell[1]), draws=args.draws)

    ranking_table = pd.concat(rankings, ignore_index=True)
    _merge_csv(out / "cv_ranking.csv", ranking_table).to_csv(out / "cv_ranking.csv", index=False)
    metrics_table = pd.DataFrame(reports)
    _merge_csv(out / "stage_cell_metrics.csv", metrics_table).to_csv(
        out / "stage_cell_metrics.csv", index=False)
    dump(out / "bootstrap_intervals.json", {**(
        json.loads((out / "bootstrap_intervals.json").read_text())
        if (out / "bootstrap_intervals.json").exists() else {}), **bootstraps})
    # Frozen before any holdout score exists, so it cannot have been tuned on one.
    existing_cells = json.loads((out / "selection.json").read_text())["cells"] \
        if (out / "selection.json").exists() else {}
    dump(out / "selection.json", {
        "holdout_start": str(HOLDOUT_START.date()),
        "decision_rule": ("UP when p_up >= up_threshold and p_down < down_threshold; DOWN when "
                          "the reverse holds; NEUTRAL otherwise."),
        "objective": "each side's own positive-class F1 on pre-holdout development rows",
        "grid": {f"L{l}_k{k}": {"lookback": l, "forward": k} for l, k in cells},
        "cells": {**existing_cells, **decisions_all}})

    summary = stage_summary(reports)
    summary.to_csv(out / "stage_summary.csv", index=False)
    baseline = always_up_reference(oof)
    print_stage_table(summary, baseline)

    elapsed = time.perf_counter() - started
    dump(out / "metrics.json", {
        "stages": list(args.stages), "cells": len(cells), "candidates": len(specs),
        "folds": len(fold_numbers), "oof_rows": len(oof), "failed_fits": len(failures),
        "stage_summary": summary.to_dict(orient="records"),
        "always_up_reference": baseline, "source_hashes": source_hashes(),
        "elapsed_seconds": elapsed})
    print(f"[done] {elapsed:.0f}s -> {out}", flush=True)

    if not args.holdout:
        return

    _run_holdout_stage(args, out, cells, init_args, started)


if __name__ == "__main__":
    main()