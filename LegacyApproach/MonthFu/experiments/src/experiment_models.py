"""Small, deterministic direction/return families with train-only preprocessing."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class Spec:
    name: str
    group: str
    task: str
    family: str
    variant: int
    require_all_cot: bool = False


def candidates(groups, require_all_cot=False):
    return [Spec(f"{task}__{group}__{family}_{variant}", group, task, family, variant, require_all_cot)
            for task in ["class", "reg"] for group in groups
            for family in ["linear", "xgb", "forest"] for variant in [0, 1]]


def estimator(spec: Spec, quick=False):
    count = 30 if quick else 100
    classification = spec.task == "class"
    if spec.family == "linear":
        model = (LogisticRegression(C=[.01, .1][spec.variant], class_weight="balanced",
                    max_iter=3000, solver="lbfgs", random_state=42) if classification else
                 Ridge(alpha=[100., 1000.][spec.variant]))
    elif spec.family == "forest":
        cls = RandomForestClassifier if classification else RandomForestRegressor
        kwargs = dict(n_estimators=count, max_depth=[5, 8][spec.variant],
                      min_samples_leaf=[60, 30][spec.variant], max_features=.7,
                      n_jobs=1, random_state=42)
        if classification:
            kwargs["class_weight"] = "balanced_subsample"
        model = cls(**kwargs)
    elif spec.family == "xgb":
        from xgboost import XGBClassifier, XGBRegressor
        cls = XGBClassifier if classification else XGBRegressor
        model = cls(n_estimators=count, max_depth=[2, 3][spec.variant], learning_rate=.035,
                    min_child_weight=[60, 30][spec.variant], reg_lambda=[30., 10.][spec.variant],
                    reg_alpha=.05, subsample=.8, colsample_bytree=.7, n_jobs=1,
                    tree_method="hist", random_state=42,
                    objective="binary:logistic" if classification else "reg:squarederror")
    else:
        raise ValueError(f"Unknown family: {spec.family}")
    return Pipeline([("impute", SimpleImputer(strategy="median", keep_empty_features=True,
                                               add_indicator=True)),
                     ("scale", StandardScaler()), ("model", model)])


def fit_member(spec: Spec, train: pd.DataFrame, columns, quick=False):
    """Zero realized returns are excluded only from the binary training target."""
    if not len(train) or not np.isfinite(train.target_return.to_numpy(dtype=float)).all():
        raise ValueError("Training labels must be nonempty and finite")
    rows = train.loc[train.target_return.ne(0)] if spec.task == "class" else train
    cot_columns = [c for c in rows if c.startswith("cot_") and pd.api.types.is_numeric_dtype(rows[c])]
    if spec.require_all_cot:
        if not cot_columns:
            raise ValueError("All-COT fitting requires a nonempty COT schema")
        missing = set(cot_columns) - set(columns)
        if missing:
            raise ValueError(f"All-COT group omits required source fields: {sorted(missing)}")
    usable = [c for c in columns if (spec.require_all_cot and c in cot_columns)
              or (rows[c].notna().mean() >= .2 and rows[c].nunique(dropna=True) > 1)]
    if not usable or not len(rows):
        raise ValueError("No usable mature training rows/features")
    y = rows.target_return.gt(0).astype(int) if spec.task == "class" else rows.target_return
    if spec.task == "class" and y.nunique() != 2:
        raise ValueError("Both classes are required")
    kwargs = {}
    if spec.task == "class" and spec.family == "xgb":
        # Class balancing is fitted only on mature labels in this refit.
        positive, negative = int(y.sum()), int((1-y).sum())
        if not min(positive, negative):
            raise ValueError("Both classes are required")
        kwargs["model__sample_weight"] = np.where(y, len(y)/(2*positive), len(y)/(2*negative))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fitted = estimator(spec, quick).fit(rows[usable], y, **kwargs)
    return {"spec": asdict(spec), "features": usable, "estimator": fitted,
            "required_cot_features": cot_columns if spec.require_all_cot else [],
            "cot_inclusion_audit": [{"feature": c, "observed_rows": int(rows[c].notna().sum()),
                "observed_fraction": float(rows[c].notna().mean()),
                "unique_observed_values": int(rows[c].nunique(dropna=True)),
                "included": c in usable, "required": spec.require_all_cot,
                "empty_training_column": not bool(rows[c].notna().any())}
                for c in cot_columns],
            "train_rows": len(rows), "train_last_date": str(rows.Date.max().date()),
            "train_last_target_date": str(rows.target_end_date.max().date()),
            "warnings": [f"{type(w.message).__name__}: {w.message}" for w in caught]}


def predict_member(member, frame):
    if member["spec"].get("require_all_cot", False):
        required = set(member.get("required_cot_features", []))
        available = {c for c in frame if c.startswith("cot_") and pd.api.types.is_numeric_dtype(frame[c])}
        if not required or required != available or not required.issubset(member["features"]):
            raise ValueError("All-COT inference schema does not match the fitted source fields")
    x = frame[member["features"]]
    if member["spec"]["task"] == "class":
        return member["estimator"].predict_proba(x)[:, 1]
    return member["estimator"].predict(x)


def predict_recipe(recipe, fitted, frame):
    return sum(weight * predict_member(fitted[name], frame)
               for name, weight in recipe["weights"].items())


def regression_direction_score(predicted):
    """Monotonic bounded score for return-sign reporting, NOT calibrated P(up).

    The fixed .10 scale never uses holdout labels. At threshold .5 this exactly
    reproduces predicted-return >= 0. Proper probability scores should be
    interpreted only for classifiers; regression reports mark them as absent.
    """
    from scipy.special import expit
    return expit(np.asarray(predicted, dtype=float)/.10)
