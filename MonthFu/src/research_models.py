from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Ridge


def _as_float_matrix(values) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("Expected a two-dimensional feature matrix.")
    if not np.isfinite(matrix).all():
        raise ValueError("Feature matrix contains non-finite values after preprocessing.")
    return matrix


def _as_positive_prices(values: Iterable[float]) -> np.ndarray:
    prices = np.asarray(list(values), dtype=float)
    if prices.ndim != 1 or len(prices) < 3:
        raise ValueError("At least three one-dimensional price observations are required.")
    if not np.isfinite(prices).all() or np.any(prices <= 0):
        raise ValueError("Prices must be finite and strictly positive.")
    return prices


class ExtremeLearningMachineRegressor(BaseEstimator, RegressorMixin):
    """Single-hidden-layer ELM with fixed random features and ridge output weights."""

    def __init__(
        self,
        n_hidden: int = 128,
        alpha: float = 1.0,
        activation: str = "tanh",
        random_state: int = 42,
    ) -> None:
        self.n_hidden = n_hidden
        self.alpha = alpha
        self.activation = activation
        self.random_state = random_state

    def _activate(self, values: np.ndarray) -> np.ndarray:
        if self.activation == "tanh":
            return np.tanh(values)
        if self.activation == "relu":
            return np.maximum(values, 0.0)
        raise ValueError(f"Unsupported activation: {self.activation}")

    def fit(self, X, y):
        matrix = _as_float_matrix(X)
        target = np.asarray(y, dtype=float).reshape(-1)
        if len(matrix) != len(target):
            raise ValueError("Feature and target row counts do not match.")
        if self.n_hidden <= 0 or self.alpha < 0:
            raise ValueError("n_hidden must be positive and alpha must be non-negative.")

        rng = np.random.default_rng(self.random_state)
        input_scale = 1.0 / np.sqrt(max(1, matrix.shape[1]))
        self.input_weights_ = rng.normal(
            loc=0.0,
            scale=input_scale,
            size=(matrix.shape[1], self.n_hidden),
        )
        self.hidden_bias_ = rng.normal(loc=0.0, scale=0.5, size=self.n_hidden)
        hidden = self._activate(matrix @ self.input_weights_ + self.hidden_bias_)
        design = np.column_stack([np.ones(len(hidden)), hidden])

        penalty = np.eye(design.shape[1]) * float(self.alpha)
        penalty[0, 0] = 0.0
        gram = design.T @ design + penalty
        rhs = design.T @ target
        try:
            self.output_weights_ = np.linalg.solve(gram, rhs)
        except np.linalg.LinAlgError:
            self.output_weights_ = np.linalg.pinv(gram) @ rhs
        self.n_features_in_ = matrix.shape[1]
        return self

    def predict(self, X) -> np.ndarray:
        matrix = _as_float_matrix(X)
        if matrix.shape[1] != self.n_features_in_:
            raise ValueError("Prediction feature count differs from fitted feature count.")
        hidden = self._activate(matrix @ self.input_weights_ + self.hidden_bias_)
        design = np.column_stack([np.ones(len(hidden)), hidden])
        return np.asarray(design @ self.output_weights_, dtype=float)


class ThresholdAutoregressiveRegressor(BaseEstimator, RegressorMixin):
    """Two-regime ridge AR model selected by an extreme-positioning feature."""

    def __init__(
        self,
        alpha: float = 1.0,
        min_regime_size: int = 120,
        threshold_quantiles: tuple[float, ...] = (0.60, 0.70, 0.80, 0.90),
    ) -> None:
        self.alpha = alpha
        self.min_regime_size = min_regime_size
        self.threshold_quantiles = threshold_quantiles

    def fit(self, X, y):
        matrix = _as_float_matrix(X)
        target = np.asarray(y, dtype=float).reshape(-1)
        if len(matrix) != len(target):
            raise ValueError("Feature and target row counts do not match.")
        if matrix.shape[1] < 2:
            raise ValueError("Threshold AR requires a regime feature and at least one predictor.")

        regime_score = np.abs(matrix[:, 0])
        best = None
        for quantile in self.threshold_quantiles:
            threshold = float(np.quantile(regime_score, quantile))
            high = regime_score > threshold
            low = ~high
            if min(int(high.sum()), int(low.sum())) < self.min_regime_size:
                continue
            low_model = Ridge(alpha=self.alpha).fit(matrix[low, 1:], target[low])
            high_model = Ridge(alpha=self.alpha).fit(matrix[high, 1:], target[high])
            predictions = np.empty_like(target)
            predictions[low] = low_model.predict(matrix[low, 1:])
            predictions[high] = high_model.predict(matrix[high, 1:])
            mse = float(np.mean(np.square(target - predictions)))
            if best is None or mse < best[0]:
                best = (mse, threshold, low_model, high_model)

        self.fallback_model_ = Ridge(alpha=self.alpha).fit(matrix[:, 1:], target)
        if best is None:
            self.threshold_ = np.inf
            self.low_model_ = self.fallback_model_
            self.high_model_ = self.fallback_model_
        else:
            _, self.threshold_, self.low_model_, self.high_model_ = best
        self.n_features_in_ = matrix.shape[1]
        return self

    def predict(self, X) -> np.ndarray:
        matrix = _as_float_matrix(X)
        if matrix.shape[1] != self.n_features_in_:
            raise ValueError("Prediction feature count differs from fitted feature count.")
        high = np.abs(matrix[:, 0]) > self.threshold_
        predictions = np.empty(len(matrix), dtype=float)
        predictions[~high] = self.low_model_.predict(matrix[~high, 1:])
        predictions[high] = self.high_model_.predict(matrix[high, 1:])
        return predictions


@dataclass
class Arima111State:
    drift: float
    ar: float
    ma: float
    last_log_price: float
    last_difference: float
    last_residual: float
    optimizer_success: bool
    optimizer_message: str

    def update_price(self, price: float) -> None:
        log_price = float(np.log(price))
        difference = log_price - self.last_log_price
        prediction = self.drift + self.ar * self.last_difference + self.ma * self.last_residual
        residual = difference - prediction
        self.last_log_price = log_price
        self.last_difference = difference
        self.last_residual = residual

    def forecast_return(self, horizon: int) -> float:
        if horizon <= 0:
            raise ValueError("Forecast horizon must be positive.")
        next_difference = self.drift + self.ar * self.last_difference + self.ma * self.last_residual
        total_difference = next_difference
        for _ in range(1, horizon):
            next_difference = self.drift + self.ar * next_difference
            total_difference += next_difference
        return float(np.expm1(total_difference))


def fit_arima111(prices: Iterable[float]) -> Arima111State:
    values = _as_positive_prices(prices)
    differences = np.diff(np.log(values))
    if len(differences) < 20:
        raise ValueError("ARIMA(1,1,1) requires at least 21 prices.")

    def residuals(params: np.ndarray) -> np.ndarray:
        drift, ar, ma = params
        errors = np.zeros_like(differences)
        errors[0] = differences[0] - drift
        for index in range(1, len(differences)):
            fitted = drift + ar * differences[index - 1] + ma * errors[index - 1]
            errors[index] = differences[index] - fitted
        return errors

    def objective(params: np.ndarray) -> float:
        errors = residuals(params)
        return float(np.mean(np.square(errors[1:])))

    initial = np.array([float(np.mean(differences)), 0.10, 0.10])
    result = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        bounds=[(-0.10, 0.10), (-0.98, 0.98), (-0.98, 0.98)],
        options={"maxiter": 300},
    )
    params = result.x if np.isfinite(result.fun) else initial
    errors = residuals(params)
    return Arima111State(
        drift=float(params[0]),
        ar=float(params[1]),
        ma=float(params[2]),
        last_log_price=float(np.log(values[-1])),
        last_difference=float(differences[-1]),
        last_residual=float(errors[-1]),
        optimizer_success=bool(result.success),
        optimizer_message=str(result.message),
    )


@dataclass
class SimpleExpSmoothingState:
    alpha: float
    level: float
    last_log_price: float
    optimizer_success: bool
    optimizer_message: str

    def update_price(self, price: float) -> None:
        log_price = float(np.log(price))
        self.level = self.alpha * log_price + (1.0 - self.alpha) * self.level
        self.last_log_price = log_price

    def forecast_return(self, horizon: int) -> float:
        if horizon <= 0:
            raise ValueError("Forecast horizon must be positive.")
        return float(np.expm1(self.level - self.last_log_price))


def fit_simple_exp_smoothing(prices: Iterable[float]) -> SimpleExpSmoothingState:
    values = np.log(_as_positive_prices(prices))

    def objective(alpha: float) -> float:
        level = float(values[0])
        errors = []
        for observed in values[1:]:
            errors.append(float(observed - level))
            level = alpha * float(observed) + (1.0 - alpha) * level
        return float(np.mean(np.square(errors)))

    result = minimize_scalar(objective, bounds=(0.001, 0.999), method="bounded")
    alpha = float(result.x) if result.success else 0.20
    level = float(values[0])
    for observed in values[1:]:
        level = alpha * float(observed) + (1.0 - alpha) * level
    return SimpleExpSmoothingState(
        alpha=alpha,
        level=level,
        last_log_price=float(values[-1]),
        optimizer_success=bool(result.success),
        optimizer_message=str(result.message),
    )


@dataclass
class HoltWintersState:
    alpha: float
    beta: float
    gamma: float
    period: int
    level: float
    trend: float
    seasonal: np.ndarray
    time_index: int
    last_log_price: float
    optimizer_success: bool
    optimizer_message: str

    def update_price(self, price: float) -> None:
        observed = float(np.log(price))
        season_index = self.time_index % self.period
        previous_level = self.level
        previous_season = float(self.seasonal[season_index])
        self.level = self.alpha * (observed - previous_season) + (1.0 - self.alpha) * (
            self.level + self.trend
        )
        self.trend = self.beta * (self.level - previous_level) + (1.0 - self.beta) * self.trend
        self.seasonal[season_index] = self.gamma * (observed - self.level) + (
            1.0 - self.gamma
        ) * previous_season
        self.time_index += 1
        self.last_log_price = observed

    def forecast_return(self, horizon: int) -> float:
        if horizon <= 0:
            raise ValueError("Forecast horizon must be positive.")
        season_index = (self.time_index + horizon - 1) % self.period
        forecast_log_price = self.level + horizon * self.trend + self.seasonal[season_index]
        return float(np.expm1(forecast_log_price - self.last_log_price))


def _initial_holt_winters_state(values: np.ndarray, period: int) -> tuple[float, float, np.ndarray]:
    first = values[:period]
    second = values[period : 2 * period]
    level = float(np.mean(first))
    trend = float((np.mean(second) - np.mean(first)) / period)
    seasonal = np.asarray(first - level, dtype=float)
    return level, trend, seasonal


def _simulate_holt_winters(
    values: np.ndarray,
    period: int,
    alpha: float,
    beta: float,
    gamma: float,
) -> tuple[float, float, np.ndarray, int, np.ndarray]:
    level, trend, seasonal = _initial_holt_winters_state(values, period)
    errors = []
    time_index = period
    for observed in values[period:]:
        season_index = time_index % period
        previous_level = level
        previous_season = float(seasonal[season_index])
        forecast = level + trend + previous_season
        errors.append(float(observed - forecast))
        level = alpha * (float(observed) - previous_season) + (1.0 - alpha) * (level + trend)
        trend = beta * (level - previous_level) + (1.0 - beta) * trend
        seasonal[season_index] = gamma * (float(observed) - level) + (1.0 - gamma) * previous_season
        time_index += 1
    return level, trend, seasonal, time_index, np.asarray(errors, dtype=float)


def fit_holt_winters(prices: Iterable[float], period: int = 5) -> HoltWintersState:
    values = np.log(_as_positive_prices(prices))
    if period < 2 or len(values) < period * 3:
        raise ValueError("Holt-Winters requires at least three complete seasonal periods.")

    def objective(params: np.ndarray) -> float:
        _, _, _, _, errors = _simulate_holt_winters(values, period, *params)
        return float(np.mean(np.square(errors)))

    initial = np.array([0.25, 0.05, 0.05])
    result = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        bounds=[(0.001, 0.999), (0.001, 0.999), (0.001, 0.999)],
        options={"maxiter": 250},
    )
    params = result.x if np.isfinite(result.fun) else initial
    level, trend, seasonal, time_index, _ = _simulate_holt_winters(values, period, *params)
    return HoltWintersState(
        alpha=float(params[0]),
        beta=float(params[1]),
        gamma=float(params[2]),
        period=int(period),
        level=float(level),
        trend=float(trend),
        seasonal=np.asarray(seasonal, dtype=float),
        time_index=int(time_index),
        last_log_price=float(values[-1]),
        optimizer_success=bool(result.success),
        optimizer_message=str(result.message),
    )


@dataclass
class Garch11State:
    omega: float
    alpha: float
    beta: float
    mean_return_pct: float
    last_residual_pct: float
    last_variance_pct2: float
    last_log_price: float
    optimizer_success: bool
    optimizer_message: str

    def update_price(self, price: float) -> None:
        log_price = float(np.log(price))
        observed_return_pct = (log_price - self.last_log_price) * 100.0
        variance = (
            self.omega
            + self.alpha * self.last_residual_pct**2
            + self.beta * self.last_variance_pct2
        )
        self.last_residual_pct = observed_return_pct - self.mean_return_pct
        self.last_variance_pct2 = max(float(variance), 1e-12)
        self.last_log_price = log_price

    def forecast_volatility(self, horizon: int) -> float:
        if horizon <= 0:
            raise ValueError("Forecast horizon must be positive.")
        variance = (
            self.omega
            + self.alpha * self.last_residual_pct**2
            + self.beta * self.last_variance_pct2
        )
        total_variance = max(float(variance), 1e-12)
        for _ in range(1, horizon):
            variance = self.omega + (self.alpha + self.beta) * variance
            total_variance += max(float(variance), 1e-12)
        return float(np.sqrt(total_variance) / 100.0)


def fit_garch11(prices: Iterable[float]) -> Garch11State:
    values = _as_positive_prices(prices)
    returns_pct = np.diff(np.log(values)) * 100.0
    if len(returns_pct) < 50:
        raise ValueError("GARCH(1,1) requires at least 51 prices.")
    mean_return = float(np.mean(returns_pct))
    residuals = returns_pct - mean_return
    unconditional_variance = max(float(np.var(residuals, ddof=1)), 1e-6)

    def conditional_variances(params: np.ndarray) -> np.ndarray:
        omega, alpha, beta = params
        variances = np.empty_like(residuals)
        variances[0] = unconditional_variance
        for index in range(1, len(residuals)):
            variances[index] = (
                omega + alpha * residuals[index - 1] ** 2 + beta * variances[index - 1]
            )
        return np.maximum(variances, 1e-12)

    def objective(params: np.ndarray) -> float:
        variances = conditional_variances(params)
        likelihood = np.log(variances) + np.square(residuals) / variances
        return float(0.5 * np.mean(likelihood))

    initial = np.array([unconditional_variance * 0.05, 0.08, 0.88])
    result = minimize(
        objective,
        initial,
        method="SLSQP",
        bounds=[
            (1e-10, max(unconditional_variance * 10.0, 1.0)),
            (1e-6, 0.998),
            (1e-6, 0.998),
        ],
        constraints=[{"type": "ineq", "fun": lambda params: 0.999 - params[1] - params[2]}],
        options={"maxiter": 500, "ftol": 1e-10},
    )
    params = result.x if np.isfinite(result.fun) else initial
    if params[1] + params[2] >= 0.999:
        params = initial
    variances = conditional_variances(params)
    return Garch11State(
        omega=float(params[0]),
        alpha=float(params[1]),
        beta=float(params[2]),
        mean_return_pct=mean_return,
        last_residual_pct=float(residuals[-1]),
        last_variance_pct2=float(variances[-1]),
        last_log_price=float(np.log(values[-1])),
        optimizer_success=bool(result.success),
        optimizer_message=str(result.message),
    )


def walk_forward_return_forecast(
    kind: str,
    fit_prices: Iterable[float],
    context_prices: Iterable[float],
    evaluation_prices: Iterable[float],
    horizon: int,
):
    if kind == "arima_111":
        state = fit_arima111(fit_prices)
    elif kind == "simple_exp_smoothing":
        state = fit_simple_exp_smoothing(fit_prices)
    elif kind == "holt_winters":
        state = fit_holt_winters(fit_prices, period=5)
    else:
        raise ValueError(f"Unsupported time-series model: {kind}")

    for price in context_prices:
        state.update_price(float(price))
    predictions = []
    for price in evaluation_prices:
        state.update_price(float(price))
        predictions.append(state.forecast_return(horizon))
    return state, np.asarray(predictions, dtype=float)


def rolling_polynomial_return_forecast(
    history_prices: Iterable[float],
    evaluation_prices: Iterable[float],
    horizon: int,
    degree: int,
    window: int = 252,
) -> np.ndarray:
    if degree not in {1, 2}:
        raise ValueError("Only linear and quadratic trend forecasts are supported.")
    history = list(_as_positive_prices(history_prices))
    predictions = []
    for current_price in evaluation_prices:
        current_price = float(current_price)
        if current_price <= 0 or not np.isfinite(current_price):
            raise ValueError("Evaluation prices must be finite and positive.")
        history.append(current_price)
        sample = np.log(np.asarray(history[-window:], dtype=float))
        x_values = np.arange(len(sample), dtype=float)
        coefficients = np.polyfit(x_values, sample, degree)
        forecast_log_price = float(np.polyval(coefficients, len(sample) - 1 + horizon))
        predictions.append(np.expm1(forecast_log_price - sample[-1]))
    return np.asarray(predictions, dtype=float)


def walk_forward_garch_forecast(
    fit_prices: Iterable[float],
    context_prices: Iterable[float],
    evaluation_prices: Iterable[float],
    horizon: int,
):
    state = fit_garch11(fit_prices)
    for price in context_prices:
        state.update_price(float(price))
    predictions = []
    for price in evaluation_prices:
        state.update_price(float(price))
        predictions.append(state.forecast_volatility(horizon))
    return state, np.asarray(predictions, dtype=float)
