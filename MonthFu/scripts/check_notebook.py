"""Execute the saved MonthFu notebook and check its artifact-backed views.

No model fitting or final-model historical inference occurs here. Comparisons
are recomputed independently from stored out-of-sample forecasts. Outputs stay
inside MonthFu, including the executed notebook and its dashboard preview.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MONTHFU = ROOT / "MonthFu"
ARTIFACTS = MONTHFU / "artifacts" / "all_cot"
EXPECTED_HORIZONS = {"30calendar": (30, "calendar"), "21sessions": (21, "sessions"),
                     "28calendar": (28, "calendar"), "21calendar": (21, "calendar"),
                     "30sessions": (30, "sessions")}


def require_close(actual, expected, label):
    # Float32 learner predictions change slightly after a CSV round trip;
    # squared-error skill amplifies that harmless serialization difference.
    # Match the independent auditor's absolute tolerance for derived skill.
    tolerance = 1e-8 if label.endswith("r2_vs_zero") else 1e-12
    if not np.isclose(float(actual), float(expected), rtol=1e-9, atol=tolerance):
        raise AssertionError(f"{label}: {actual} != saved value {expected}")


def return_metrics(rows, column):
    actual, predicted = rows.target_return.to_numpy(float), rows[column].to_numpy(float)
    if not (np.isfinite(actual).all() and np.isfinite(predicted).all()):
        raise AssertionError(f"Nonfinite evaluated returns for {column}")
    mse = float(np.mean((actual - predicted) ** 2))
    return {"rows": len(rows), "rmse": np.sqrt(mse), "mae": np.mean(np.abs(actual - predicted)),
            "direction_accuracy": np.mean(np.sign(actual) == np.sign(predicted)),
            "r2_vs_zero": 1 - mse / np.mean(actual ** 2)}


def verify_saved_artifacts():
    comparison = pd.read_csv(ARTIFACTS / "horizon_comparison.csv")
    forecasts = pd.read_csv(ARTIFACTS / "latest_forecasts.csv")
    if set(comparison.experiment) != set(EXPECTED_HORIZONS) or comparison.experiment.duplicated().any():
        raise AssertionError("Horizon comparison must contain exactly the five requested experiments")
    if set(forecasts.experiment) != set(EXPECTED_HORIZONS) or forecasts.experiment.duplicated().any():
        raise AssertionError("Latest forecasts must contain one row per requested experiment")
    prices = pd.read_csv(ROOT / "data/yahoo/arabica_coffee_futures_history.csv")
    dates = pd.to_datetime(prices.Date, errors="coerce")
    closes = pd.to_numeric(prices.Close, errors="coerce")
    cached_origin = dates.loc[dates.notna() & np.isfinite(closes) & closes.gt(0)].max().normalize()
    predictions = {}
    for experiment, (horizon, unit) in EXPECTED_HORIZONS.items():
        folder = ARTIFACTS / experiment
        summary = json.loads((folder / "metrics.json").read_text())
        if (summary["horizon"], summary["horizon_unit"]) != (horizon, unit):
            raise AssertionError(f"Wrong target definition in {experiment}")
        rows = pd.read_csv(folder / "holdout_predictions.csv", parse_dates=["Date", "target_end_date"])
        if rows.Date.duplicated().any() or not rows.Date.is_monotonic_increasing:
            raise AssertionError(f"Invalid saved replay calendar in {experiment}")
        if not rows.target_end_date.gt(rows.Date).all():
            raise AssertionError(f"Invalid target maturity in {experiment}")
        predictions[experiment] = rows
        metric_table = pd.read_csv(folder / "holdout_metrics.csv").set_index("model")
        for record in summary["holdout_metrics"]:
            calculated = return_metrics(rows, record["model"])
            for metric, value in calculated.items():
                require_close(value, record[metric], f"{experiment} JSON {record['model']} {metric}")
                require_close(value, metric_table.loc[record["model"], metric],
                              f"{experiment} CSV {record['model']} {metric}")
        cv = pd.read_csv(folder / "cv_predictions.csv")
        for metric, value in return_metrics(cv, "selected").items():
            require_close(value, summary["cv_selected"][metric], f"{experiment} CV {metric}")
        year_table = pd.read_csv(folder / "holdout_by_year.csv").set_index(["year", "model"])
        for year, subset in rows.groupby(rows.Date.dt.year):
            for column in metric_table.index:
                for metric, value in return_metrics(subset, column).items():
                    require_close(value, year_table.loc[(year, column), metric],
                                  f"{experiment} year {year} {column} {metric}")
        refits = pd.read_csv(folder / "holdout_refits.csv", parse_dates=["first_prediction_date", "train_last_target_date"])
        if not refits.train_last_target_date.lt(refits.first_prediction_date).all():
            raise AssertionError(f"Training labels cross an evaluation boundary in {experiment}")
        latest = pd.read_csv(folder / "latest_forecast.csv").iloc[0]
        combined = forecasts.loc[forecasts.experiment.eq(experiment)].iloc[0]
        if (int(latest.horizon), latest.horizon_unit) != (horizon, unit):
            raise AssertionError(f"{experiment} latest forecast target definition differs")
        if pd.Timestamp(latest.as_of_date).normalize() != cached_origin:
            raise AssertionError(f"{experiment} forecast origin does not match latest valid cached close")
        for column in ["predicted_return", "close", "cv_empirical_lower_80", "cv_empirical_upper_80"]:
            require_close(combined[column], latest[column], f"{experiment} latest {column}")
        if pd.Timestamp(combined.as_of_date) != pd.Timestamp(latest.as_of_date):
            raise AssertionError(f"{experiment} combined latest origin differs from saved forecast")
    common_dates = set.intersection(*(set(rows.Date) for rows in predictions.values()))
    if not common_dates:
        raise AssertionError("No common evaluated horizon origins")
    for experiment, rows in predictions.items():
        matched = rows.loc[rows.Date.isin(common_dates)]
        row = comparison.loc[comparison.experiment.eq(experiment)].iloc[0]
        selected = return_metrics(matched, "selected")
        previous = return_metrics(matched, "previous_transferred_ensemble")
        zero = return_metrics(matched, "zero")
        summary = json.loads((ARTIFACTS / experiment / "metrics.json").read_text())
        require_close(row.cv_skill_vs_zero, summary["cv_selected"]["r2_vs_zero"], f"{experiment} comparison CV skill")
        require_close(row.cv_rmse, summary["cv_selected"]["rmse"], f"{experiment} comparison CV RMSE")
        if "retuned_previous_rmse" in comparison:
            require_close(row.retuned_previous_rmse, return_metrics(matched, "previous_cv_ensemble")["rmse"],
                          f"{experiment} retuned previous RMSE")
        for value, expected, label in [
            (len(matched), row.common_holdout_rows, "common rows"),
            (selected["rmse"], row.holdout_rmse, "common selected RMSE"),
            (previous["rmse"], row.previous_rmse, "common previous RMSE"),
            (zero["rmse"], row.zero_rmse, "common zero RMSE"),
            (selected["r2_vs_zero"], row.holdout_skill_vs_zero, "common skill"),
            (selected["direction_accuracy"], row.direction_accuracy, "common direction"),
            (1-selected["rmse"]/previous["rmse"], row.holdout_rmse_improvement_vs_previous, "common improvement"),
        ]:
            require_close(value, expected, f"{experiment} {label}")
    return {"horizons": len(predictions), "common_holdout_origins": len(common_dates),
            "cached_origin": str(cached_origin.date())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-only", action="store_true", help="Verify saved CSV/JSON metrics without starting or executing a notebook kernel")
    parser.add_argument("--timeout", type=int, default=180, help="Maximum seconds per notebook cell")
    parser.add_argument("--output", type=Path, default=MONTHFU / "MonthFu.ipynb")
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(MONTHFU) or output.suffix != ".ipynb":
        raise ValueError("Write the executed .ipynb only inside MonthFu")
    saved_checks = verify_saved_artifacts()
    if args.artifacts_only:
        print(json.dumps({**saved_checks, "artifact_metrics_verified": True,
                          "notebook_execution": "not executed in artifacts-only mode"}, indent=2))
        return
    print(f"Verified all {saved_checks['horizons']} horizons and {saved_checks['common_holdout_origins']} common holdout origins. Launching the local Jupyter kernel.", flush=True)
    os.environ.setdefault("MPLCONFIGDIR", str(ARTIFACTS / ".matplotlib"))
    os.environ.setdefault("IPYTHONDIR", str(ARTIFACTS / ".ipython"))
    os.environ.setdefault("JUPYTER_RUNTIME_DIR", str(ARTIFACTS / ".jupyter_runtime"))
    import nbformat
    from nbclient import NotebookClient
    from jupyter_client import KernelManager
    from PIL import Image
    notebook_path = MONTHFU / "MonthFu.ipynb"
    notebook = nbformat.read(notebook_path, as_version=4)
    # This notebook's replay must be artifact-backed, never final-model inference.
    code = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")
    for prohibited in ["joblib.load", "predict_bundle(", "fit_candidate(", "subprocess.run("]:
        if prohibited in code:
            raise AssertionError(f"Notebook must not load/refit models or infer historical scores: {prohibited}")
    manager = KernelManager(kernel_name="python3")
    manager.kernel_spec.argv[0] = sys.executable
    client = NotebookClient(notebook, km=manager, timeout=args.timeout, kernel_name="python3",
                            resources={"metadata": {"path": str(MONTHFU)}}, allow_errors=False)
    executed = client.execute()
    marker = "MONTHFU_NOTEBOOK_CHECKS="
    notebook_checks = None
    for cell in executed.cells:
        for entry in cell.get("outputs", []):
            if entry.output_type == "stream":
                for line in entry.text.splitlines():
                    if line.startswith(marker):
                        notebook_checks = json.loads(line[len(marker):])
    if notebook_checks is None or not notebook_checks.get("saved_replay_verified"):
        raise AssertionError("Executed notebook did not confirm saved-score replay")
    for key in ["horizons", "cached_origin"]:
        if notebook_checks[key] != saved_checks[key]:
            raise AssertionError(f"Notebook artifact check disagrees for {key}")
    preview = ARTIFACTS / "monthfu_dashboard.png"
    if not preview.exists():
        raise AssertionError("Notebook did not render monthfu_dashboard.png")
    with Image.open(preview) as image:
        pixels = np.asarray(image.convert("RGB"))
        if min(image.size) < 600 or float(pixels.std()) < 5:
            raise AssertionError("Rendered dashboard is too small or blank")
    output.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(executed, output)
    print(json.dumps({**saved_checks, "saved_replay_verified": True,
                      "executed_notebook": str(output), "dashboard_preview": str(preview)}, indent=2))


if __name__ == "__main__":
    main()
