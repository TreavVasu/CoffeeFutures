"""Bounded, causal paper-family parameter search using existing dependencies.

ARIMA is conditional-sum-of-squares (CSS), not statsmodels exact maximum
likelihood. The existing environment has scipy/sklearn but no statsmodels.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.signal import lfilter
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from MonthFu.src.research_models import ExtremeLearningMachineRegressor, fit_arima111, fit_holt_winters


@dataclass(frozen=True)
class PaperSpec:
    name: str
    family: str
    window: int = 1260
    order: tuple[int, int, int] = (1, 1, 1)
    drift: bool = True
    lag_count: int = 21
    alpha: float = 100.0
    n_hidden: int = 128
    activation: str = "tanh"
    seed: int = 42
    season: int = 0
    damping: float = 1.0


def paper_specs(quick: bool = False) -> list[PaperSpec]:
    orders = [(0, 1, 0), (1, 1, 0), (0, 1, 1), (1, 1, 1), (2, 1, 1), (1, 0, 0), (2, 0, 1)]
    if quick:
        orders = [(0, 1, 0), (1, 1, 1), (1, 0, 0)]
    windows = [252] if quick else [252, 1260]
    specs = [PaperSpec(f"arima_{p}{d}{q}_w{w}", "arima", window=w, order=(p, d, q))
             for w in windows for p, d, q in orders]
    specs += [PaperSpec("arima_111_no_drift_w1260", "arima", order=(1, 1, 1), drift=False),
              PaperSpec("legacy_fixed_arima_111", "legacy_arima", window=0),
              PaperSpec("legacy_fixed_holt_5", "legacy_holt", window=0, season=5)]
    for lags in ([21] if quick else [5, 21, 63]):
        for alpha in ([100.0] if quick else [2.0, 100.0, 1000.0]):
            specs.append(PaperSpec(f"ar_lags{lags}_ridge{int(alpha)}", "ar", window=0,
                                   lag_count=lags, alpha=alpha))
    for hidden in ([32] if quick else [32, 128]):
        for alpha in ([100.0] if quick else [10.0, 100.0, 1000.0]):
            for activation in (["tanh"] if quick else ["tanh", "relu"]):
                specs.append(PaperSpec(f"elm_h{hidden}_a{int(alpha)}_{activation}_s42", "elm",
                                       window=0, n_hidden=hidden, alpha=alpha, activation=activation))
    if not quick:
        specs += [PaperSpec("elm_h128_a100_tanh_s314", "elm", window=0, seed=314),
                  PaperSpec("elm_h128_a100_tanh_recent5y", "elm", window=1260)]
    for window in windows:
        for season in ([0, 5] if quick else [0, 5, 21]):
            for damping in ([0.98] if quick else [1.0, 0.98]):
                specs.append(PaperSpec(f"holt_s{season}_phi{str(damping).replace('.', '')}_w{window}",
                                       "holt", window=window, season=season, damping=damping))
        specs.append(PaperSpec(f"simple_smoothing_w{window}", "smoothing", window=window))
    return specs


def _positive_prices(prices: Iterable[float]) -> np.ndarray:
    values = np.asarray(list(prices), dtype=float)
    if values.ndim != 1 or len(values) < 30 or not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("At least 30 finite positive prices are required")
    return values


def _stationary_coefficients(reflection: np.ndarray) -> np.ndarray:
    """Map reflection coefficients inside (-1,1) to a stationary AR polynomial."""
    coefficients = np.array([], dtype=float)
    for value in reflection:
        coefficients = np.r_[coefficients - value * coefficients[::-1], value]
    return coefficients


@dataclass
class CssArimaState:
    order: tuple[int, int, int]
    center: float
    intercept: float
    ar: np.ndarray
    ma: np.ndarray
    observations: list[float]
    residuals: list[float]
    last_log_price: float
    optimizer_success: bool
    optimizer_message: str
    objective: float

    def update_price(self, price: float) -> None:
        log_price = float(np.log(price))
        observed = log_price - self.last_log_price if self.order[1] else log_price - self.center
        fitted = self.intercept
        for lag, coefficient in enumerate(self.ar, 1):
            fitted += coefficient * self.observations[-lag]
        for lag, coefficient in enumerate(self.ma, 1):
            fitted += coefficient * self.residuals[-lag]
        self.observations.append(observed)
        self.residuals.append(float(observed - fitted))
        self.observations = self.observations[-max(1, len(self.ar)):]
        self.residuals = self.residuals[-max(1, len(self.ma)):]
        self.last_log_price = log_price

    def forecast_return(self, steps: int) -> float:
        if steps < 1:
            raise ValueError("Positive forecast steps are required")
        observations, residuals = self.observations.copy(), self.residuals.copy()
        cumulative = 0.0
        value = 0.0
        for _ in range(steps):
            value = self.intercept
            for lag, coefficient in enumerate(self.ar, 1):
                value += coefficient * observations[-lag]
            for lag, coefficient in enumerate(self.ma, 1):
                value += coefficient * residuals[-lag]
            observations.append(float(value))
            residuals.append(0.0)
            cumulative += value
        difference = cumulative if self.order[1] else value + self.center - self.last_log_price
        return float(np.expm1(difference))


def fit_css_arima(prices: Iterable[float], order=(1, 1, 1), drift=True) -> CssArimaState:
    """Estimate stable ARIMA(p,d,q), d in {0,1}, with explicit optimizer status."""
    values = np.log(_positive_prices(prices))
    p, d, q = order
    if min(p, q) < 0 or d not in (0, 1):
        raise ValueError("Supported orders have nonnegative p/q and d equal to zero or one")
    center = float(values.mean()) if d == 0 else 0.0
    observations = np.diff(values) if d else values - center
    burn = max(5, p, q)
    def unpack(parameters):
        intercept = float(parameters[0]) if drift else 0.0
        offset = int(drift)
        ar = _stationary_coefficients(parameters[offset:offset+p])
        ma = -_stationary_coefficients(parameters[offset+p:offset+p+q])
        return intercept, ar, ma
    def errors(parameters):
        intercept, ar, ma = unpack(parameters)
        adjusted = lfilter(np.r_[1.0, -ar], [1.0], observations) - intercept
        return lfilter([1.0], np.r_[1.0, ma], adjusted)
    def objective(parameters):
        residuals = errors(parameters)[burn:]
        return float(np.mean(residuals ** 2))
    initial = np.r_[[float(observations.mean())] if drift else [], np.zeros(p+q)]
    bounds = ([(-0.10, 0.10)] if drift else []) + [(-0.98, 0.98)] * (p+q)
    # Scale CSS for stable finite-difference gradients; no data-dependent model selection.
    scale = max(float(np.var(observations)), 1e-6)
    if len(initial):
        result = minimize(lambda pars: objective(pars) / scale, initial, method="L-BFGS-B",
                          bounds=bounds, options={"maxiter": 250, "ftol": 1e-10})
        parameters, success, message = result.x, bool(result.success), str(result.message)
    else:
        parameters, success, message = initial, True, "No free parameters"
    intercept, ar, ma = unpack(parameters)
    residuals = errors(parameters)
    success = success and bool(np.isfinite(parameters).all()) and bool(np.isfinite(residuals).all())
    return CssArimaState(tuple(order), center, intercept, ar, ma,
                         observations[-max(1, p):].tolist(), residuals[-max(1, q):].tolist(),
                         float(values[-1]), success, message, objective(parameters))


@dataclass
class DampedHoltState:
    alpha: float
    beta: float
    gamma: float
    damping: float
    season: int
    level: float
    trend: float
    seasonal: np.ndarray
    time_index: int
    last_log_price: float
    optimizer_success: bool
    optimizer_message: str
    objective: float

    def update_price(self, price: float) -> None:
        value = float(np.log(price))
        position = self.time_index % self.season if self.season else 0
        seasonal = self.seasonal[position] if self.season else 0.0
        previous = self.level
        self.level = self.alpha * (value-seasonal) + (1-self.alpha) * (self.level+self.damping*self.trend)
        self.trend = self.beta * (self.level-previous) + (1-self.beta) * self.damping*self.trend
        if self.season:
            self.seasonal[position] = self.gamma * (value-self.level) + (1-self.gamma) * seasonal
        self.time_index += 1
        self.last_log_price = value

    def forecast_return(self, steps: int) -> float:
        if steps < 1:
            raise ValueError("Positive forecast steps are required")
        trend_steps = steps if self.damping == 1.0 else self.damping*(1-self.damping**steps)/(1-self.damping)
        seasonal = self.seasonal[(self.time_index+steps-1) % self.season] if self.season else 0.0
        return float(np.expm1(self.level+trend_steps*self.trend+seasonal-self.last_log_price))


def fit_damped_holt(prices: Iterable[float], season=0, damping=0.98, smoothing=False) -> DampedHoltState:
    values = np.log(_positive_prices(prices))
    if season and len(values) < 3*season:
        raise ValueError("Seasonal smoothing requires three full periods")
    def simulate(parameters):
        alpha, beta = float(parameters[0]), 0.0 if smoothing else float(parameters[1])
        gamma = float(parameters[2]) if season else 0.0
        start = season if season else 1
        level = float(values[:start].mean())
        trend = float((values[season:2*season].mean()-level)/season) if season else 0.0
        seasonal = values[:season]-level if season else np.zeros(1)
        state = DampedHoltState(alpha, beta, gamma, damping, season, level, trend,
                                seasonal.copy(), start, float(values[start-1]), True, "", 0.0)
        errors = []
        for value in values[start:]:
            seasonal_value = state.seasonal[state.time_index % season] if season else 0.0
            errors.append(value-(state.level+damping*state.trend+seasonal_value))
            state.update_price(float(np.exp(value)))
        return state, float(np.mean(np.square(errors)))
    if smoothing:
        result = minimize_scalar(lambda alpha: simulate([alpha])[1], bounds=(0.001, 0.999), method="bounded")
        parameters = [float(result.x)]
    else:
        initial = [0.3, 0.03] + ([0.03] if season else [])
        scale = max(float(np.var(np.diff(values))), 1e-6)
        result = minimize(lambda pars: simulate(pars)[1] / scale, initial, method="L-BFGS-B",
                          bounds=[(0.001, 0.999)]*len(initial), options={"maxiter": 160, "ftol": 1e-8})
        parameters = result.x
    state, objective = simulate(parameters)
    state.optimizer_success, state.optimizer_message, state.objective = bool(result.success), str(result.message), objective
    return state


def paper_features(prices: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Dense causal return lags and economical multiscale history for AR and ELM."""
    log_price = np.log(prices.Close)
    log_return = log_price.diff()
    features = {f"paper_log_return_lag_{lag}": log_return.shift(lag) for lag in range(63)}
    for window in [5, 21, 28, 30, 63, 126]:
        features[f"paper_momentum_{window}"] = log_price.diff(window)
        features[f"paper_volatility_{window}"] = log_return.rolling(window, min_periods=window).std()
        features[f"paper_bias_{window}"] = log_price-log_price.rolling(window, min_periods=window).mean()
    for lag in [0, 21, 42, 63]:
        features[f"paper_monthly_return_lag_{lag}"] = log_price.diff(21).shift(lag)
    dates = pd.to_datetime(prices.Date)
    features["paper_season_sin"] = np.sin(2*np.pi*dates.dt.dayofyear/365.25)
    features["paper_season_cos"] = np.cos(2*np.pi*dates.dt.dayofyear/365.25)
    frame = pd.concat([prices[["Date", "Close"]].reset_index(drop=True), pd.DataFrame(features)], axis=1)
    groups = {f"lags_{count}": [f"paper_log_return_lag_{lag}" for lag in range(count)]
              for count in [5, 21, 63]}
    groups["elm_history"] = groups["lags_21"] + [name for name in features if "log_return_lag" not in name]
    return frame, groups


def fit_paper(spec: PaperSpec, frame: pd.DataFrame, groups: dict, cutoff: pd.Timestamp) -> dict:
    """Supervised labels are strictly mature; price-only fits use prices before cutoff."""
    history = frame.loc[frame.Date < cutoff]
    if spec.window:
        history = history.tail(spec.window)
    diagnostic = {"model": spec.name, "family": spec.family, "cutoff": str(cutoff.date()),
                  "price_fit_start": str(history.Date.min().date()),
                  "price_fit_end": str(history.Date.max().date()), "price_rows": len(history)}
    if spec.family in {"ar", "elm"}:
        train = frame.loc[frame.target_end_date.lt(cutoff) & frame.target_return.notna()]
        if spec.window:
            train = train.loc[train.Date >= cutoff-pd.Timedelta(days=365.25*spec.window/252)]
        columns = groups[f"lags_{spec.lag_count}"] if spec.family == "ar" else groups["elm_history"]
        estimator = Ridge(alpha=spec.alpha) if spec.family == "ar" else ExtremeLearningMachineRegressor(
            n_hidden=spec.n_hidden, alpha=spec.alpha, activation=spec.activation, random_state=spec.seed)
        pipe = Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)),
                         ("scale", StandardScaler()), ("model", estimator)])
        pipe.fit(train[columns], train.target_return)
        diagnostic.update(train_rows=len(train), latest_training_label_end=str(train.target_end_date.max().date()),
                          optimizer_success=True, optimizer_message="Closed-form regularized least squares")
        return {"spec": asdict(spec), "pipeline": pipe, "features": columns, "diagnostic": diagnostic}
    if spec.family == "arima":
        state = fit_css_arima(history.Close, order=spec.order, drift=spec.drift)
    elif spec.family == "legacy_arima":
        state = fit_arima111(history.Close)
    elif spec.family == "legacy_holt":
        state = fit_holt_winters(history.Close, period=5)
    else:
        state = fit_damped_holt(history.Close, season=spec.season, damping=spec.damping,
                                smoothing=spec.family == "smoothing")
    diagnostic.update(optimizer_success=state.optimizer_success, optimizer_message=state.optimizer_message,
                      objective=getattr(state, "objective", None),
                      fitted_parameters={name: getattr(state, name) for name in ["intercept", "alpha", "beta", "gamma", "damping"] if hasattr(state, name)})
    if hasattr(state, "ar"):
        diagnostic["ar"] = np.asarray(state.ar).tolist()
    if hasattr(state, "ma"):
        diagnostic["ma"] = np.asarray(state.ma).tolist()
    return {"spec": asdict(spec), "state": state, "state_as_of": history.Date.iloc[-1], "diagnostic": diagnostic}


def predict_paper(member: dict, frame: pd.DataFrame, origins: pd.DataFrame) -> np.ndarray:
    if not member["diagnostic"]["optimizer_success"]:
        raise ValueError(f"Invalid optimizer fit for {member['spec']['name']}: {member['diagnostic']['optimizer_message']}")
    if "pipeline" in member:
        return np.asarray(member["pipeline"].predict(origins[member["features"]]), dtype=float)
    if origins.Date.duplicated().any() or not origins.Date.is_monotonic_increasing:
        raise ValueError("Stateful origins must be unique and increasing")
    state = deepcopy(member["state"])
    state_date = pd.Timestamp(member["state_as_of"])
    if len(origins) and origins.Date.min() < state_date:
        raise ValueError("Historical replay must use saved OOS predictions, not future fitted states")
    known = frame.loc[frame.Date.gt(state_date) & frame.Date.le(origins.Date.max()), ["Date", "Close"]]
    updates = iter(known.itertuples(index=False))
    next_update = next(updates, None)
    result = []
    for origin in origins.itertuples(index=False):
        while next_update is not None and next_update.Date <= origin.Date:
            state.update_price(next_update.Close)
            next_update = next(updates, None)
        result.append(state.forecast_return(int(origin.forecast_steps)))
    return np.asarray(result)


def direction_regression_metrics(actual, predicted) -> dict:
    y, p = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    if not len(y) or len(y) != len(p) or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("Metrics require nonempty finite paired predictions")
    # Binary metrics omit exact-zero realized returns. A predicted-zero tie is Up.
    nonzero = y != 0.0
    actual_up, predicted_up = y[nonzero] > 0.0, p[nonzero] >= 0.0
    precision, recall, f1, support = precision_recall_fscore_support(
        actual_up, predicted_up, labels=[False, True], zero_division=0)
    mse = np.mean(np.square(y-p))
    metrics = {"rows": len(y), "direction_rows": int(nonzero.sum()), "rmse": float(np.sqrt(mse)), "mae": float(np.mean(np.abs(y-p))),
               "direction_accuracy": float(accuracy_score(actual_up, predicted_up)),
               "strict_sign_accuracy": float(np.mean(np.sign(y) == np.sign(p))),
               "balanced_accuracy": float(balanced_accuracy_score(actual_up, predicted_up)),
               "macro_f1": float(f1.mean()), "predicted_up_fraction": float(predicted_up.mean()),
               "r2_vs_zero": float(1-mse/np.mean(y**2)) if np.mean(y**2) else None}
    for i, name in enumerate(["down", "up"]):
        metrics.update({f"{name}_precision": float(precision[i]), f"{name}_recall": float(recall[i]),
                        f"{name}_f1": float(f1[i]), f"{name}_support": int(support[i])})
    return metrics


def direction_block_comparison(actual, proposed, reference, block=60, draws=1000, seed=42) -> dict:
    """Paired moving-block intervals, preserving dependence in overlapping labels."""
    y, proposed, reference = [np.asarray(values, dtype=float) for values in [actual, proposed, reference]]
    if len(y) < block or not all(len(values) == len(y) and np.isfinite(values).all() for values in [proposed, reference]):
        raise ValueError("Block comparisons require matched finite rows and adequate history")
    def statistics(actual_values, prediction_values):
        nonzero = actual_values != 0
        true, pred = actual_values[nonzero] > 0, prediction_values[nonzero] >= 0
        tp, tn = np.sum(true & pred), np.sum(~true & ~pred)
        fp, fn = np.sum(~true & pred), np.sum(true & ~pred)
        divide = lambda numerator, denominator: float(numerator/denominator) if denominator else 0.0
        return np.array([divide(tp+tn, len(true)), (divide(2*tp, 2*tp+fp+fn)+divide(2*tn, 2*tn+fp+fn))/2,
                         divide(tp, tp+fp), divide(tp, tp+fn), divide(tn, tn+fn), divide(tn, tn+fp)])
    names = ["direction_accuracy", "macro_f1", "up_precision", "up_recall", "down_precision", "down_recall"]
    measured = statistics(y, proposed)-statistics(y, reference)
    rng = np.random.default_rng(seed)
    changes = np.empty((draws, len(names)))
    for draw in range(draws):
        starts = rng.integers(0, len(y)-block+1, size=int(np.ceil(len(y)/block)))
        positions = (starts[:, None]+np.arange(block)).ravel()[:len(y)]
        changes[draw] = statistics(y[positions], proposed[positions])-statistics(y[positions], reference[positions])
    return {"block_sessions": block, "draws": draws, "note": "Positive proposed-minus-reference differences favor proposed; overlapping historical labels and model selection remain limitations",
            "differences": {name: {"measured": float(measured[index]),
                                    "lower_95": float(np.quantile(changes[:, index], 0.025)),
                                    "upper_95": float(np.quantile(changes[:, index], 0.975))}
                            for index, name in enumerate(names)}}
