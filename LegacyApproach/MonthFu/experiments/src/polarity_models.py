"""Two independent directional models: one for UP, one for DOWN.

The study keeps the two cases separate rather than fitting a single
``P(up) vs P(down)`` head.  Each side gets its own

* target definition -- ``1{return > 0}`` for the UP model, ``1{return < 0}``
  for the DOWN model, both on the *same* observed rows;
* feature bank -- the UP model may see bullish candle columns the DOWN model
  never sees, and the reverse for bearish ones;
* estimator family, hyper-parameters and class weighting;
* decision threshold, chosen on pre-holdout out-of-fold rows only.

Why this can differ from one joint head
---------------------------------------
On rows with a non-zero forward return the two labels are complements, so the
two targets carry the same information.  The value is therefore not extra data
but *separate capacity*: the UP and DOWN problems are not symmetric.  Up moves
are driven by breakouts and thrust while down moves are driven by stalls and
reversals, and a single shared head must compromise between them.  Fitting two
heads lets each choose its own representation and threshold.  This module does
not claim the separation is automatically better; ``polarity_metrics`` measures
whether it is.

Because the two heads are fitted independently they are **not** on a common
probability scale, so raw outputs must not be compared directly.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

POLARITIES = ("up", "down")


@dataclass(frozen=True)
class DirectionSpec:
    """One candidate model on one side of the direction.

    ``group`` names a feature bank; ``family`` and ``variant`` select the
    estimator.  The UP and DOWN searches run independently, so the same
    family/variant may be chosen on both sides without any coupling.
    """

    polarity: str
    group: str
    family: str
    variant: int

    def __post_init__(self) -> None:
        if self.polarity not in POLARITIES:
            raise ValueError(f"Polarity must be one of {POLARITIES}.")
        if self.family not in {"linear", "xgb", "forest"}:
            raise ValueError("Family must be linear, xgb or forest.")
        if self.variant not in (0, 1):
            raise ValueError("Only the two fixed variants 0 and 1 are defined.")

    @property
    def name(self) -> str:
        return f"{self.polarity}__{self.group}__{self.family}_{self.variant}"

    @property
    def target_column(self) -> str:
        return f"direction_{self.polarity}"


def candidates(groups: list[str]) -> list[DirectionSpec]:
    """Both polarities across every group, family and fixed variant.

    The two sides are enumerated independently so each can be selected on its
    own development rows; nothing here couples the two searches.
    """
    return [DirectionSpec(polarity, group, family, variant)
            for polarity in POLARITIES
            for group in groups
            for family in ("linear", "xgb", "forest")
            for variant in (0, 1)]


def training_rows(train: pd.DataFrame, polarity: str) -> pd.DataFrame:
    """Rows with an observed direction label for this side.

    An exactly-zero forward return has no direction, so it is excluded for both
    sides.  Both sides therefore see an identical row set, which is what makes
    their head-to-head comparison fair.
    """
    if polarity not in POLARITIES:
        raise ValueError(f"Polarity must be one of {POLARITIES}.")
    label = f"direction_{polarity}"
    if label not in train:
        raise ValueError(f"Frame is missing the {label} target column.")
    usable = train[label].notna() & np.isfinite(train.target_return.to_numpy(dtype=float))
    return train.loc[usable]


def usable_features(train: pd.DataFrame, columns: list[str], minimum_coverage: float = .2) -> list[str]:
    """Columns observable and varying inside this training fold only.

    Screening on the training rows keeps coverage and variance decisions inside
    the fold; a column's usefulness must not be judged with holdout labels.
    """
    keep = []
    for column in columns:
        if column not in train:
            continue
        series = train[column]
        if series.notna().mean() >= minimum_coverage and series.nunique(dropna=True) > 1:
            keep.append(column)
    return keep


def fit_direction(spec: DirectionSpec, train: pd.DataFrame, columns: list[str],
                  quick: bool = False, forest_trees: int = 100,
                  xgb_trees: int = 100) -> dict:
    """Fit one side's model on its own labels.

    Preprocessing (median imputation plus a missingness indicator) and scaling
    are fitted inside this fold, never on validation or holdout rows.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    rows = training_rows(train, spec.polarity)
    if not len(rows):
        raise ValueError("A direction model needs nonempty training rows with observed labels.")
    features = usable_features(rows, columns)
    if not features:
        raise ValueError(f"No usable features for {spec.name} in this training fold.")
    y = rows[spec.target_column].astype(int)
    if y.nunique() != 2:
        raise ValueError(f"Both {spec.polarity} outcomes are required to fit {spec.name}.")

    trees, boosting = (30, 30) if quick else (forest_trees, xgb_trees)
    if spec.family == "linear":
        model = LogisticRegression(C=[.01, .1][spec.variant], class_weight="balanced",
                                   max_iter=3000, solver="lbfgs", random_state=42)
    elif spec.family == "forest":
        model = RandomForestClassifier(
            n_estimators=trees, max_depth=[5, 8][spec.variant],
            min_samples_leaf=[60, 30][spec.variant], max_features=.7, n_jobs=1,
            random_state=42, class_weight="balanced_subsample")
    else:
        from xgboost import XGBClassifier
        model = XGBClassifier(
            n_estimators=boosting, max_depth=[2, 3][spec.variant], learning_rate=.035,
            min_child_weight=[60, 30][spec.variant], reg_lambda=[30., 10.][spec.variant],
            reg_alpha=.05, subsample=.8, colsample_bytree=.7, n_jobs=1,
            tree_method="hist", random_state=42, objective="binary:logistic",
            eval_metric="logloss")

    pipeline = Pipeline([("impute", SimpleImputer(strategy="median", keep_empty_features=True,
                                                  add_indicator=True)),
                         ("scale", StandardScaler()), ("model", model)])
    fit_kwargs = {}
    if spec.family == "xgb":
        # Balancing weights come from this fold's labels only.
        positive, negative = int(y.sum()), int((1 - y).sum())
        fit_kwargs["model__sample_weight"] = np.where(y, len(y) / (2 * positive), len(y) / (2 * negative))
    pipeline.fit(rows[features], y, **fit_kwargs)
    return {"spec": spec, "features": features, "estimator": pipeline,
            "train_rows": len(rows), "positive_rate": float(y.mean()),
            "train_last_date": str(pd.to_datetime(rows.Date).max().date())}


def predict_direction(member: dict, frame: pd.DataFrame) -> np.ndarray:
    """Probability that this side's outcome occurs, on the model's own scale."""
    return member["estimator"].predict_proba(frame[member["features"]])[:, 1]


def polarity_banks(candle_columns: list[str], shared: list[str], polarity: str) -> dict[str, list[str]]:
    """Feature groups for one side, so each side sees its own candle patterns.

    ``basic``      shared inputs only
    ``candles``    this side's own patterns plus shared inputs
    ``engineered`` shared inputs plus every current-bar candle column

    The two sides differ only through ``candles``: ``basic`` and ``engineered``
    are identical by construction so a comparison isolates the effect of the
    side-specific pattern bank.
    """
    from candlestick_patterns import candles_for_polarity

    if polarity not in POLARITIES:
        raise ValueError(f"Polarity must be one of {POLARITIES}.")
    own = candles_for_polarity(candle_columns, "bull" if polarity == "up" else "bear")
    current_bar = [c for c in candle_columns if "@" not in c]
    known = set(candle_columns) | set(shared)
    groups = {
        "basic": list(shared),
        "candles": list(dict.fromkeys(own + list(shared))),
        "engineered": list(dict.fromkeys(current_bar + list(shared))),
    }
    return {name: [c for c in columns if c in known] for name, columns in groups.items()}