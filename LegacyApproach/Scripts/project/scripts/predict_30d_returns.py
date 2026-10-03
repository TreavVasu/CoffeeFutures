"""Rebuild causal features from local inputs and use the saved 30-day bundle."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import joblib

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "Scripts/project/src"))
from return_forecasting import build_frame
from train_30d_return_model import latest_forecast


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=ROOT / "Scripts/project/artifacts/returns_30d/model.joblib")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    bundle = joblib.load(args.bundle)
    frame, _, _, _ = build_frame(ROOT, bundle["horizon"], bundle["horizon_unit"])
    if frame.Date.max().strftime("%Y-%m-%d") < bundle["fitted_as_of"]:
        raise ValueError("Input cache predates the fitted model; historical replay needs historical weights.")
    latest = latest_forecast(bundle, frame)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        latest.to_csv(args.output, index=False)
    print(latest.to_string(index=False))


if __name__ == "__main__":
    main()
