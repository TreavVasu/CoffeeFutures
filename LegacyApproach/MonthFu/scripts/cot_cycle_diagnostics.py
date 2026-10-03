"""Retrospective COT-cadence diagnostics; never creates model features or selections."""
from __future__ import annotations

import argparse
import hashlib
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd

HORIZONS = ["30calendar", "28calendar", "21calendar", "21sessions", "30sessions"]
MODELS = ["selected", "zero", "previous_transferred_ensemble", "previous_cv_ensemble"]


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
                     + ["| " + " | ".join(map(str, row)) + " |" for row in rows])


def diagnose(artifact_root: Path, horizons: list[str]) -> str:
    inputs, hashes, frames, audits = {}, [], {}, {}
    for horizon in horizons:
        loaded, digests = [], []
        for name in ("holdout_predictions.csv", "cot_release_audit.csv"):
            path = artifact_root / horizon / name
            raw = path.read_bytes()
            inputs[path] = hashlib.sha256(raw).hexdigest()
            digests.append(inputs[path])
            loaded.append(pd.read_csv(BytesIO(raw)))
        frame, audit = loaded
        frame["Date"] = pd.to_datetime(frame["Date"], errors="raise")
        frame["target_end_date"] = pd.to_datetime(frame["target_end_date"], errors="raise")
        if frame.Date.duplicated().any() or not (frame.target_end_date > frame.Date).all():
            raise ValueError(f"Invalid forecast dates in {horizon}")
        finite = np.isfinite(frame[["target_return"] + MODELS].to_numpy(dtype=float)).all(axis=1)
        frames[horizon] = frame.loc[finite].set_index("Date").sort_index()
        audits[horizon] = pd.to_datetime(audit.first_usable_session, errors="raise").dropna().drop_duplicates().sort_values().to_numpy(dtype="datetime64[ns]")
        hashes.append([horizon] + digests)
    common = sorted(set.intersection(*(set(frame.index) for frame in frames.values())))
    if not common:
        raise ValueError("No common finite holdout origins")
    scores, cycles = [], []
    for horizon in horizons:
        frame, events = frames[horizon].loc[common], audits[horizon]
        on_release = frame.index.isin(events)
        for group, mask in (("All origins", np.ones(len(frame), dtype=bool)), ("First usable COT session", on_release), ("Other origins", ~on_release)):
            sample = frame.loc[mask]
            if sample.empty:
                continue
            y = sample.target_return.to_numpy()
            rmses = [np.sqrt(np.mean((sample[model].to_numpy() - y) ** 2)) * 100 for model in MODELS]
            mae = np.mean(np.abs(sample.selected.to_numpy() - y)) * 100
            direction = np.mean(np.sign(sample.selected.to_numpy()) == np.sign(y)) * 100
            scores.append([horizon, group, len(sample)] + [f"{v:.3f}" for v in rmses] + [f"{mae:.3f}", f"{direction:.1f}%"])
        counts = np.searchsorted(events, frame.target_end_date.to_numpy(dtype="datetime64[ns]"), side="right") - np.searchsorted(events, frame.index.to_numpy(dtype="datetime64[ns]"), side="right")
        distribution = "; ".join(f"{int(k)}: {int(v)}" for k, v in pd.Series(counts).value_counts().sort_index().items())
        cycles.append([horizon, f"{np.mean(counts):.2f}", int(np.median(counts)), f"{np.quantile(counts, .1):.0f}–{np.quantile(counts, .9):.0f}", distribution])
    for path, digest in inputs.items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Artifact changed while reading: {path}; rerun after training completes")
    text = ["# COT release-cycle diagnostics", "",
            f"Retrospective diagnostics on {len(common):,} shared finite holdout origins, {common[0].date()}–{common[-1].date()}. Every displayed model is scored on the same rows within each subgroup and horizon. Return endpoints differ by horizon. These diagnostics do not select a horizon, tune the model, or add features.", "",
            "## Errors on newly usable COT sessions", "",
            "A release origin is an observed coffee session where at least one COT family first becomes usable under the saved audit. Legacy/disaggregated duplicates count once. It is usually the session after Friday publication; holidays, catch-up releases, and conservative correction bounds can change that. Selected recipes were frozen before holdout. This split is descriptive and does not isolate COT's causal contribution.", "",
            "RMSE and MAE are percentage points of return; lower is better. The previous-model columns retain the saved prediction names: transferred prior ensemble and its CV-retuned counterpart. Direction is shown for the selected model; zero forecasts are neutral.", "",
            table(["Horizon", "Origins", "N", "Selected RMSE", "Zero RMSE", "Previous transferred RMSE", "Previous CV RMSE", "Selected MAE", "Selected direction"], scores), "",
            "## Update events inside the realized return interval", "",
            "Count unique usable-session update events in (forecast origin, actual target-end date]. The origin's available report is excluded and an event on the endpoint is included. These are retrospective future counts, never model inputs. Delayed backlog releases need not follow a seven-day rhythm.", "",
            table(["Horizon", "Mean events", "Median", "10th–90th percentile", "Event count: number of origins"], cycles), "",
            "Twenty-eight calendar days cover four ordinary weekly cycles. Thirty calendar days usually contain four or five update events; twenty-one calendar days cover about three, while twenty-one observed sessions approximate a trading month. The measured distributions explain cadence; they do not demonstrate that any horizon forecasts best. [CFTC release schedule](https://www.cftc.gov/MarketReports/CommitmentsofTraders/ReleaseSchedule/index.htm).", "",
            "Daily monthly returns overlap heavily. Origins within and across the displayed subgroups are dependent, and weekly release timing is confounded with weekday/holiday effects. No confidence interval, significance claim, executable trading advantage, or holdout-based horizon choice follows from these tables. Historical release estimates and unavailable original COT vintages remain limitations; see [COT alignment policy](COT_ALIGNMENT.md).", "",
            "## Reproduce", "", "```sh", ".venv/bin/python MonthFu/scripts/cot_cycle_diagnostics.py", "```", "",
            "Run after all horizon exports finish. The script reads saved predictions/audits only and writes this document. SHA256 input snapshots:", "",
            table(["Horizon", "Holdout predictions SHA256", "COT audit SHA256"], hashes), ""]
    return "\n".join(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    monthfu = Path(__file__).resolve().parents[1]
    parser.add_argument("--artifact-root", type=Path, default=monthfu / "artifacts")
    parser.add_argument("--output", type=Path, default=monthfu / "docs/COT_CYCLE_RESULTS.md")
    parser.add_argument("--horizons", nargs="+", choices=HORIZONS, default=HORIZONS)
    args = parser.parse_args()
    result = diagnose(args.artifact_root, args.horizons)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(result)
    print(args.output)


if __name__ == "__main__":
    main()
