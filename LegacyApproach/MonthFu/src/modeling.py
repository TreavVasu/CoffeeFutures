"""Monthly return targets, bounded model search, and label-maturity evaluation."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from .research_models import ExtremeLearningMachineRegressor, fit_holt_winters
except ImportError:
    from research_models import ExtremeLearningMachineRegressor, fit_holt_winters


PREVIOUS_WEIGHTS = {"previous_rf": .4640262047149529,
                    "previous_hw": .41934745545223984,
                    "previous_ar": .11662633983280726}


def add_targets(frame: pd.DataFrame, horizon: int = 30, unit: str = "calendar") -> pd.DataFrame:
    if horizon < 1 or unit not in {"sessions", "calendar"}:
        raise ValueError("Use a positive horizon and sessions/calendar unit.")
    result = frame.copy()
    dates = pd.DatetimeIndex(result.Date)
    if not dates.is_monotonic_increasing or dates.has_duplicates or dates.hasnans:
        raise ValueError("Targets require unique increasing nonmissing observation dates.")
    end = (np.arange(len(result)) + horizon if unit == "sessions" else
           dates.searchsorted(dates + pd.Timedelta(days=horizon), side="left"))
    valid = end < len(result)
    result["target_end_date"] = pd.NaT
    result["target_close"] = np.nan
    result.loc[valid, "target_end_date"] = dates.take(end[valid]).to_numpy()
    result.loc[valid, "target_close"] = result.Close.to_numpy()[end[valid]]
    result["target_return"] = result.target_close / result.Close - 1
    result["target_sessions"] = np.where(valid, end - np.arange(len(result)), np.nan)
    result["forecast_steps"] = (float(horizon) if unit == "sessions" else np.array([
        max(1, np.busday_count(d.date(), (d + pd.Timedelta(days=horizon)).date())) for d in dates
    ], dtype=float))
    return result


@dataclass(frozen=True)
class Candidate:
    name: str
    group: str
    kind: str
    strength: float = 1.0
    leaf: int = 60
    train_years: int = 0
    normalized: bool = False
    half_life_years: float = 0
    require_all_cot: bool = False


def candidates(quick: bool = False,
               require_all_cot: bool = False) -> list[Candidate]:
    result = [Candidate("zero", "monthly_core", "zero"),
              Candidate("historical_mean", "monthly_core", "mean"),
              Candidate("compact_legacy", "compact_legacy", "hist", 10, 40),
              Candidate("previous_rf", "previous_technical", "previous_forest", 1, 10),
              Candidate("previous_ar", "previous_history", "ridge", 2),
              Candidate("previous_hw", "previous_technical", "holt_winters")]
    for group in ["price", "price_cot", "price_cot_weather", "monthly_core", "engineered"]:
        for alpha in ([1000] if quick else [100, 1000, 10000]):
            result.append(Candidate(f"{group}_ridge_{alpha}", group, "ridge", alpha))
        result.append(Candidate(f"{group}_hist", group, "hist", 1, 80))
        result.append(Candidate(f"{group}_extra", group, "extra", 1, 80))
    if not quick:
        result += [
            Candidate("monthly_core_hist_low_l2", "monthly_core", "hist", .01, 100),
            Candidate("monthly_core_ridge_recent10y", "monthly_core", "ridge", 1000, 60, 10),
            Candidate("monthly_core_ridge_recent5y", "monthly_core", "ridge", 1000, 60, 5),
            Candidate("engineered_ridge_recent10y", "engineered", "ridge", 10000, 60, 10),
            Candidate("monthly_core_ridge_decay", "monthly_core", "ridge", 1000, half_life_years=5),
            Candidate("engineered_elastic", "engineered", "elastic", .003),
            Candidate("monthly_core_vol_ridge", "monthly_core", "ridge", 1000, normalized=True),
            Candidate("engineered_vol_ridge", "engineered", "ridge", 10000, normalized=True),
            Candidate("monthly_core_vol_hist", "monthly_core", "hist", 10, 100, normalized=True),
            Candidate("price_cot_elm", "price_cot", "elm", 100),
            Candidate("monthly_core_elm", "monthly_core", "elm", 100),
            Candidate("engineered_elm", "engineered", "elm", 1000),
            Candidate("monthly_core_xgb", "monthly_core", "xgb", 10, 80),
            Candidate("engineered_xgb", "engineered", "xgb", 30, 100),
            Candidate("monthly_core_xgb_recent10y", "monthly_core", "xgb", 20, 100, 10),
        ]
    if require_all_cot:
        result = [replace(spec,
                    group="monthly_core_all_cot" if spec.group == "monthly_core" else spec.group,
                    require_all_cot=True)
                  if spec.group in {"price_cot", "price_cot_weather", "monthly_core", "engineered"}
                     and spec.kind not in {"zero", "mean"} else spec
                  for spec in result]
    return result


def make_estimator(spec: Candidate, quick: bool = False) -> Pipeline:
    count = 60 if quick else 160
    if spec.kind == "xgb":
        from xgboost import XGBRegressor
        model = XGBRegressor(n_estimators=count, max_depth=2, learning_rate=.025,
            min_child_weight=spec.leaf, subsample=.8, colsample_bytree=.7,
            reg_alpha=.03, reg_lambda=spec.strength, objective="reg:squarederror",
            tree_method="hist", n_jobs=1, random_state=42)
    else:
        estimators = {
            "zero": lambda: DummyRegressor(strategy="constant", constant=0),
            "mean": lambda: DummyRegressor(strategy="mean"),
            "ridge": lambda: Ridge(alpha=spec.strength),
            "elastic": lambda: ElasticNet(alpha=spec.strength, l1_ratio=.1, max_iter=30000,
                                          selection="random", random_state=42),
            "hist": lambda: HistGradientBoostingRegressor(max_iter=count, learning_rate=.035,
                max_leaf_nodes=7, min_samples_leaf=spec.leaf, l2_regularization=spec.strength,
                early_stopping=False, random_state=42),
            "extra": lambda: ExtraTreesRegressor(n_estimators=count, max_depth=7,
                min_samples_leaf=spec.leaf, max_features=.7, n_jobs=1, random_state=42),
            "previous_forest": lambda: RandomForestRegressor(n_estimators=80 if quick else 220,
                max_depth=10, min_samples_leaf=10, max_features=.75, n_jobs=1, random_state=42),
            "elm": lambda: ExtremeLearningMachineRegressor(n_hidden=128, alpha=spec.strength, random_state=42),
        }
        model = estimators[spec.kind]()
    return Pipeline([("impute", SimpleImputer(strategy="median", keep_empty_features=True, add_indicator=True)),
                     ("scale", StandardScaler()), ("model", model)])


def mature_training_rows(frame: pd.DataFrame, cutoff: pd.Timestamp, train_years: int = 0) -> pd.DataFrame:
    training = frame.loc[frame.target_return.notna() & frame.target_end_date.lt(cutoff) & frame.Date.lt(cutoff)]
    if train_years:
        training = training.loc[training.Date.ge(cutoff - pd.DateOffset(years=train_years))]
    return training


def expanding_folds(frame: pd.DataFrame, holdout_start: pd.Timestamp, splits: int = 4,
                    test_size: int = 504, min_train: int = 1000):
    if splits < 1 or test_size < 1:
        raise ValueError("Fold count and validation size must be positive.")
    development = mature_training_rows(frame, holdout_start).reset_index(drop=True)
    first = len(development) - splits * test_size
    if first < min_train:
        raise ValueError("Not enough development history for requested folds.")
    result = []
    for start in range(first, len(development), test_size):
        valid = development.iloc[start:start+test_size]
        train = mature_training_rows(frame, valid.Date.iloc[0])
        if len(train) < min_train:
            raise ValueError("Purging leaves too few training observations.")
        result.append((train, valid))
    return result


def volatility_scale(frame: pd.DataFrame, floor: float = .01) -> np.ndarray:
    # Every component is known at origin; never normalize with future realized volatility.
    return (frame.price_vol_60 * np.sqrt(frame.forecast_steps)).fillna(floor).clip(lower=floor).to_numpy()


@lru_cache(maxsize=48)
def _cached_holt_winters(prices: tuple):
    # The same origin-known price history appears across horizon experiments.
    # Prediction deep-copies the state, so sharing a fit cannot mutate history.
    return fit_holt_winters(prices,period=5)


def fit_candidate(spec: Candidate, train: pd.DataFrame, columns: list[str], quick: bool = False,
                  price_history: pd.DataFrame | None = None) -> dict:
    if spec.kind == "holt_winters":
        history = train if price_history is None else price_history
        history = history.sort_values("Date")
        return {"spec": asdict(spec), "features": ["Close"],
                "state": _cached_holt_winters(tuple(history.Close)),
                "state_as_of": history.Date.iloc[-1]}
    required_cot = [c for c in columns if c.startswith("cot_")] if spec.require_all_cot else []
    if spec.require_all_cot and not required_cot:
        raise ValueError("All-COT candidates require COT predictors")
    usable = [c for c in columns if c in required_cot or
              (train[c].notna().mean() >= .20 and train[c].nunique(dropna=True) > 1)]
    if not usable:
        raise ValueError(f"No train-time usable features for {spec.name}")
    target = train.target_return.to_numpy()
    if spec.normalized:
        target = target / volatility_scale(train)
    fit_args = {}
    if spec.half_life_years:
        age = (train.Date.max() - train.Date).dt.days / 365.25
        fit_args["model__sample_weight"] = np.power(.5, age / spec.half_life_years).to_numpy()
    estimator = make_estimator(spec, quick).fit(train[usable], target, **fit_args)
    return {"spec": asdict(spec), "features": usable, "estimator": estimator,
            "required_cot_features": required_cot,
            "cot_feature_audit": [{"feature": c, "observed_rows": int(train[c].notna().sum()),
                "training_rows": len(train), "unique_observed_values": int(train[c].nunique(dropna=True)),
                "retained": c in usable, "required": c in required_cot}
                for c in columns if c.startswith("cot_")]}


def predict_member(member: dict, frame: pd.DataFrame) -> np.ndarray:
    if member["spec"].get("require_all_cot"):
        required = set(member.get("required_cot_features", []))
        if not required or not required.issubset(member["features"]) or not required.issubset(frame.columns):
            raise ValueError("All-COT predictor schema is incomplete; rebuild features and retrain")
    if member["spec"]["kind"] == "holt_winters":
        if not frame.Date.is_monotonic_increasing or frame.Date.duplicated().any():
            raise ValueError("Stateful forecasts require unique sorted origin dates.")
        state = deepcopy(member["state"])
        result = []
        for row in frame.itertuples():
            if row.Date < member["state_as_of"]:
                raise ValueError("Cannot forecast before fitted price state.")
            if row.Date > member["state_as_of"]:
                state.update_price(row.Close)
            result.append(state.forecast_return(int(row.forecast_steps)))
        return np.asarray(result)
    result = member["estimator"].predict(frame[member["features"]])
    if member["spec"].get("normalized"):
        result = result * volatility_scale(frame)
    return result


def predict_bundle(bundle: dict, frame: pd.DataFrame) -> np.ndarray:
    return sum(member["weight"] * predict_member(member, frame) for member in bundle["members"])


def metrics(actual, predicted) -> dict:
    y, p = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    if len(y) == 0 or len(y) != len(p) or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("Metrics require nonempty paired finite predictions.")
    mse, denom = np.mean((y-p)**2), np.mean(y**2)
    corr = float(np.corrcoef(y, p)[0, 1]) if np.std(p) > 1e-12 and np.std(y) > 0 else None
    return {"rows": len(y), "rmse": float(np.sqrt(mse)), "mae": float(np.mean(np.abs(y-p))),
            "direction_accuracy": float(np.mean(np.sign(y) == np.sign(p))),
            "correlation": corr, "r2_vs_zero": float(1-mse/denom) if denom else None,
            "mean_prediction": float(np.mean(p)), "mean_actual": float(np.mean(y)),
            "direction_note": "Zero forecasts are neutral; direction is reported alongside a training-only majority-class benchmark."}


def select_recipe(oof: pd.DataFrame, specs: list[Candidate], eligible_names: set[str] | None = None):
    rows = [{"recipe": spec.name, "weights": {spec.name: 1.0},
             **metrics(oof.target_return, oof[spec.name])} for spec in specs]
    rows.append({"recipe": "previous_transferred_ensemble", "weights": PREVIOUS_WEIGHTS,
                 **metrics(oof.target_return, sum(oof[n]*w for n,w in PREVIOUS_WEIGHTS.items()))})
    nontrivial = sorted([r for r in rows if r["recipe"] not in {"zero", "historical_mean"}
                        and (eligible_names is None or r["recipe"] in eligible_names)],
                        key=lambda r: r["rmse"])
    # Mixtures contain only actual candidate columns, chosen from pre-holdout CV.
    leaders = [r["recipe"] for r in nontrivial if r["recipe"] in oof][:3]
    pools = [([name], np.ones(1)) for name in leaders]
    if len(leaders) > 1:
        matrix = oof[leaders].to_numpy()
        equal = np.ones(len(leaders))/len(leaders)
        solution = minimize(lambda w: np.mean((matrix@w-oof.target_return.to_numpy())**2)+1e-5*np.sum(w*w),
            equal, bounds=[(0,1)]*len(leaders), constraints=[{"type":"eq", "fun":lambda w:w.sum()-1}],
            method="SLSQP", options={"ftol":1e-12, "maxiter":200})
        pools += [(leaders, equal)]
        if solution.success:
            pools += [(leaders, solution.x/solution.x.sum())]
    for index, (names, base_weights) in enumerate(pools):
        for shrink in [.25,.5,.75,1.]:
            weights = {name: float(weight*shrink) for name,weight in zip(names,base_weights) if weight > 1e-6}
            p = sum(oof[name]*weight for name,weight in weights.items())
            rows.append({"recipe":f"blend_{index}_shrink_{shrink:g}", "weights":weights,
                         **metrics(oof.target_return,p)})
    def eligible(row):
        return eligible_names is None or set(row["weights"]).issubset(eligible_names)
    allowed = [row for row in rows if eligible(row)]
    if not allowed:
        raise ValueError("No eligible model recipe")
    best = min(allowed,key=lambda r:r["rmse"])
    recipe = {"name":best["recipe"], "weights":best["weights"], "cv_rmse":best["rmse"],
              "cv_skill_vs_zero":best["r2_vs_zero"]}
    return recipe, pd.DataFrame([{**row, "eligible_for_selection": eligible(row)} for row in rows]).sort_values("rmse").reset_index(drop=True)


def nonoverlap_positions(frame: pd.DataFrame, offset: int = 0) -> list[int]:
    if offset < 0:
        raise ValueError("Offset must be nonnegative.")
    positions, next_start = [], pd.Timestamp.min
    for idx in range(offset, len(frame)):
        row = frame.iloc[idx]
        if row.Date >= next_start:
            positions.append(idx)
            next_start = row.target_end_date
    return positions


def block_comparison(y, selected, baseline, block: int = 60, draws: int = 1000) -> dict:
    y, selected, baseline = map(lambda a: np.asarray(a,dtype=float), [y, selected, baseline])
    if not len(y) or not len(y)==len(selected)==len(baseline) or block < 1 or draws < 1:
        raise ValueError("Use nonempty paired arrays and positive block/draw counts.")
    if not all(np.isfinite(a).all() for a in [y,selected,baseline]):
        raise ValueError("Bootstrap arrays must be finite.")
    n, rng, differences = len(y), np.random.default_rng(42), []
    block = min(block,n)
    for _ in range(draws):
        starts = rng.integers(0,n-block+1,size=int(np.ceil(n/block)))
        idx = np.concatenate([np.arange(s,s+block) for s in starts])[:n]
        differences.append(np.sqrt(np.mean((y[idx]-selected[idx])**2))-np.sqrt(np.mean((y[idx]-baseline[idx])**2)))
    return {"rmse_difference":float(np.sqrt(np.mean((y-selected)**2))-np.sqrt(np.mean((y-baseline)**2))),
            "rmse_difference_95pct_block_interval":np.quantile(differences,[.025,.975]).tolist(),
            "block_sessions":block,"bootstrap_draws":draws,
            "interpretation":"Negative favors selected; exploratory dependence-aware interval."}
