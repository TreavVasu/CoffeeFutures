"""Generate comparisons from saved experiments; never select on holdout scores."""
from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/monthfu-experiments-matplotlib")
ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT/"MonthFu/experiments"
sys.path.insert(0, str(EXP/"src"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from experiment_models import Spec, estimator
from experiment_models import regression_direction_score
from direction_metrics import binary_labels_from_returns, paired_block_bootstrap


def json_parameters(values):
    # XGBoost's missing-value sentinel is NaN; represent it explicitly in JSON.
    return {k: ("NaN" if isinstance(v, (float, np.floating)) and np.isnan(v) else v)
            for k, v in values.items()}


def table(frame, columns, percent=()):
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|"*len(columns)]
    for _, row in frame.iterrows():
        values = []
        for c in columns:
            v = row[c]
            if pd.isna(v):
                values.append("—")
            elif isinstance(v, (float, np.floating)):
                values.append(f"{100*v:.2f}%" if c in percent else f"{v:.4f}")
            else:
                values.append(str(v))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def build():
    outputs, all_changes = [], []
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    prfig, praxes = plt.subplots(2, 1, figsize=(11, 8), constrained_layout=True)
    groups = ["basic", "direction_core", "price_direction", "cot_direction", "weather_direction", "engineered_direction"]
    for index, horizon in enumerate([30, 28]):
        out = EXP/"artifacts/direction"/f"{horizon}calendar"
        meta = json.loads((out/"metrics.json").read_text())
        selection = json.loads((out/"selection.json").read_text())
        c = pd.read_csv(out/"holdout_classification_metrics.csv")
        r = pd.read_csv(out/"holdout_regression_metrics.csv")
        cv = pd.read_csv(out/"cv_ranking.csv")
        changes = pd.read_csv(out/"changes_vs_monthly.csv")
        all_changes.append(changes.assign(horizon_calendar_days=horizon))
        paperout = EXP/"artifacts/paper_parameters"/f"{horizon}calendar"
        pc = pd.read_csv(paperout/"holdout_metrics.csv")
        ps = json.loads((paperout/"selection.json").read_text())
        policies = pd.read_csv(out/"precision_recall_policy_metrics.csv")
        holdout = pd.read_csv(out/"holdout_predictions.csv")
        labels = binary_labels_from_returns(holdout.target_return)
        valid = np.isfinite(labels)
        simple_intervals = {str(block): paired_block_bootstrap(labels[valid],
            holdout["class__basic__equal_ensemble"].to_numpy()[valid],
            regression_direction_score(holdout.baseline_monthly_return)[valid],
            threshold_selected=selection["recipes"]["class__basic__equal_ensemble"]["threshold"],
            block=block, draws=1000) for block in [30, 60, 90]}
        (out/"simple_ensemble_direction_bootstrap.json").write_text(json.dumps(simple_intervals, indent=2)+"\n")
        simple_f1_interval = simple_intervals["60"]["metrics"]["macro_f1"]["difference_95pct_block_interval"]
        chosen = c.loc[c.model.isin(["selected_direction", "class__basic__equal_ensemble", "selected_return", "reg__basic__equal_ensemble", "baseline_monthly_return", "baseline_paper_return", "baseline_majority"])].copy()
        chosen["horizon_days"] = horizon
        outputs.append(chosen)
        d = c.set_index("model").loc["selected_direction"]
        b = c.set_index("model").loc["baseline_monthly_return"]
        sc = meta["selective_policy"]
        delta = d.macro_f1-b.macro_f1
        verdict = "improved" if delta > 0 else "worsened" if delta < 0 else "matched"
        limits = meta["bootstrap_60"]["metrics"]["macro_f1"]["difference_95pct_block_interval"]
        direction_recipe = selection["recipes"]["selected_direction"]
        return_recipe = selection["recipes"]["selected_return"]
        both_metrics = ["model", "accuracy", "macro_f1", "balanced_accuracy", "up_precision", "up_recall", "down_precision", "down_recall"]
        report = [f"# {horizon}-calendar-day experiment", "",
            f"The direction recipe chosen before the 2022 holdout {verdict} holdout macro F1 by {100*delta:+.2f} percentage points versus the saved monthly model. Its accuracy is {100*d.accuracy:.2f}% versus {100*b.accuracy:.2f}%. These are exploratory reused-holdout results.", "",
            f"Binary metrics use {int(d.rows):,} common nonzero-return origins; return metrics retain {meta['holdout_rows']:,} origins. Up and down scores receive equal weight in macro F1. Pre-2022 selection used {meta['cv_rows']:,} expanding OOF origins and {meta['candidate_count']} classifier/regressor configurations.", "",
            "## Selected and simple model comparisons", "",
            table(chosen, both_metrics, both_metrics[1:]), "",
            "### Return forecasts", "",
            table(r.loc[r.model.isin(["selected_return", "reg__basic__equal_ensemble", "baseline_monthly_return", "baseline_paper_return", "baseline_mean_return", "baseline_zero_return"])],
                  ["model", "rmse", "mae", "direction_accuracy", "r2_vs_zero"], ["rmse", "mae", "direction_accuracy"]), "",
            "RMSE/MAE percentages are arithmetic-return percentage points. Original regression direction accuracy treats predicted zero as neutral; binary tables treat it as an Up tie.", "",
            "### Frozen recipes", "", f"Direction: `{json.dumps(direction_recipe, sort_keys=True)}`", "",
            f"Return: `{json.dumps(return_recipe, sort_keys=True)}`", "",
            "Each simple ensemble averages one CV-chosen linear model, one XGBoost model and one random forest with weights of 1/3. Its classifier threshold is also fixed from pre-2022 OOF. There is no learned stacker or holdout weight tuning.", "",
            "## Feature and model comparisons", "",
            table(c.loc[c.task.eq("class") & ~c.model.eq("selected_direction")], both_metrics, both_metrics[1:]), "",
            table(r.loc[~r.group.eq("previous")], ["model", "rmse", "mae", "direction_accuracy", "r2_vs_zero"], ["rmse", "mae", "direction_accuracy"]), "",
            "These ablations keep definitions fixed but choose each family's configuration on CV. Differences include that train-only parameter selection; they are not isolated causal feature effects. Larger feature banks can underperform smaller banks.", "",
            "## Optional uncertain indication", "",
            f"Frozen band: Down below {selection['selected_abstention']['lower']:.3f}, Up at/above {selection['selected_abstention']['upper']:.3f}, uncertain between. Holdout coverage is {100*sc['coverage']:.2f}% ({sc['accepted_rows']} accepted / {sc['rows']} binary origins).", "",
            f"Conditional Up precision/recall: {100*sc['up_precision']:.2f}% / {100*sc['up_recall']:.2f}%; Down: {100*sc['down_precision']:.2f}% / {100*sc['down_recall']:.2f}%. Unconditional recalls, counting withheld calls, are Up {100*sc['up_unconditional_recall']:.2f}% and Down {100*sc['down_unconditional_recall']:.2f}%.", "",
            "## Precision-first and balanced-recall policies", "",
            "These additional policies use the same 24 frozen classifier recipes and a fixed pre-2022 threshold grid. Precision-first maximizes the lower of Up/Down precision subject to both CV recalls being at least 40%; balanced-recall maximizes the lower class recall. Both make a call for every origin. CV gates do not guarantee holdout recall; this table records that tradeoff without replacing the main recipe.", "",
            table(policies, [col for col in ["policy", "recipe", "threshold", "accuracy", "macro_f1", "up_precision", "up_recall", "down_precision", "down_recall"] if col in policies], ["accuracy", "macro_f1", "up_precision", "up_recall", "down_precision", "down_recall"]), "",
            "## Dependence and changes", "",
            f"The selected-minus-monthly macro-F1 difference has a paired 60-session block-bootstrap 95% interval [{100*limits[0]:+.2f}, {100*limits[1]:+.2f}] percentage points. Thirty-/90-session sensitivity intervals and 30 offsets of nonoverlapping outcomes are saved. Overlapping daily monthly labels are not independent samples.", "",
            f"The predefined simple basic ensemble's macro-F1 difference interval is [{100*simple_f1_interval[0]:+.2f}, {100*simple_f1_interval[1]:+.2f}] percentage points with 60-session blocks. Point gains do not establish a reliable advantage when this interval spans zero.", "",
            table(changes, list(changes.columns), [col for col in changes if col.startswith("delta_") or col in ["rmse", "baseline_rmse", "rmse_relative_improvement"]]), "",
            "## Research-paper parameter sweep", "",
            f"54 configurations varied ARIMA order/window/drift, AR lag/Ridge penalty, ELM width/activation/penalty/seed/window, and Holt seasonality/damping/window. Direction choice: `{ps['macro_f1_model']}`; return choice: `{ps['rmse_model']}`.", "",
            table(pc.loc[pc.model.isin(["paper_selected_macro_f1", "paper_selected_rmse", "monthly_selected", "legacy_fixed_arima_111", "legacy_fixed_holt_5"])],
                  ["model", "rmse", "direction_accuracy", "macro_f1", "up_precision", "up_recall", "down_precision", "down_recall"],
                  ["rmse", "direction_accuracy", "macro_f1", "up_precision", "up_recall", "down_precision", "down_recall"]), "",
            "ARIMA uses the custom SciPy conditional-sum-of-squares extension of the existing implementation. It is not exact statsmodels maximum likelihood. Fits failing optimizer or stability checks cannot win. Details and official references are in [the paper search notes](../../../docs/PAPER_PARAMETER_SEARCH.md).", "",
            "## Practical interpretation", "",
            "The previous holdout has already influenced the wider research process. This run freezes choices before its holdout evaluation, but positive changes still need prospective confirmation. Direction and return selection serve different objectives; a direction gain can worsen return RMSE. COT/reanalysis revision caveats remain. Class-weighted classifier outputs are not independently calibrated market probabilities. News remains excluded.", "",
            "No deep learning is promoted. There are only a few hundred nonoverlapping monthly training outcomes, and about 50–60 separate outcomes in the holdout. The paper ELM sweep provides a bounded nonlinear neural comparison without adding a large sequence network.", ""]
        (out/"report.md").write_text("\n".join(report))
        plotc = c.set_index("model")
        plotr = r.set_index("model")
        ax = axes[index, 0]
        ax.barh(groups, [plotc.loc[f"class__{g}__equal_ensemble", "macro_f1"] for g in groups], color="#32779a")
        ax.axvline(b.macro_f1, color="#9c462d", linestyle="--", label="Previous monthly sign")
        ax.set(xlabel="Holdout macro F1", title=f"{horizon} calendar days: equal classifier ensemble", xlim=(0, .7))
        ax.legend(loc="lower right")
        prax = praxes[index]
        names = ["baseline_monthly_return", "selected_direction", "class__basic__equal_ensemble"]
        rate_columns = ["up_precision", "up_recall", "down_precision", "down_recall"]
        x = np.arange(4)
        for j, name in enumerate(names):
            prax.bar(x + (j-1)*.25, [100*plotc.loc[name, col] for col in rate_columns], width=.25,
                     label=["Previous monthly sign", "CV-selected classifier", "Simple basic ensemble"][j])
        prax.set(xticks=x, xticklabels=["Up precision", "Up recall", "Down precision", "Down recall"],
                  ylim=(0, 100), ylabel="Percent", title=f"{horizon} calendar days: precision and recall on all binary origins")
        prax.legend(loc="upper right", ncol=3, fontsize=8)
        ax = axes[index, 1]
        ax.barh(groups, [100*plotr.loc[f"reg__{g}__equal_ensemble", "rmse"] for g in groups], color="#758448")
        ax.axvline(100*plotr.loc["baseline_monthly_return", "rmse"], color="#9c462d", linestyle="--", label="Previous monthly")
        ax.set(xlabel="Return RMSE (percentage points; lower better)", title=f"{horizon} calendar days: equal regression ensemble")
        ax.legend(loc="lower right")
        expanded = []
        for spec_dict in json.loads((out/"candidate_parameters.json").read_text()):
            spec = Spec(**spec_dict)
            expanded.append({**asdict(spec), "parameters": json_parameters(estimator(spec).named_steps["model"].get_params())})
        (out/"expanded_model_parameters.json").write_text(json.dumps(expanded, indent=2, allow_nan=False)+"\n")
    fig.suptitle("Monthly Arabica feature experiments — frozen CV choices, exploratory 2022+ holdout", fontsize=13)
    fig.savefig(EXP/"feature_comparison.png", dpi=160)
    plt.close(fig)
    prfig.savefig(EXP/"precision_recall_comparison.png", dpi=160)
    plt.close(prfig)
    combined = pd.concat(outputs, ignore_index=True)
    combined.to_csv(EXP/"artifacts/direction/selected_comparison.csv", index=False)
    pd.concat(all_changes, ignore_index=True).to_csv(EXP/"artifacts/direction/changes_vs_monthly.csv", index=False)
    cols = ["horizon_days", "model", "accuracy", "macro_f1", "up_precision", "up_recall", "down_precision", "down_recall"]
    summary = ["# Direction and return experiment results", "",
        "Tested 72 feature/model/task configurations and 54 paper-family configurations at each of 30 and 28 calendar days: 252 horizon/configuration combinations, plus explicit equal-weight ensemble comparisons. Added 140 causal features. Direction thresholds and model choices use pre-2022 expanding validation; quarterly holdout fits use only matured labels.", "",
        "The simple 30-day classifier ensemble improved point macro F1 and Down recall while losing some Up recall. The paper AR(63)/Ridge model also improved point direction. Neither is established as a reliable gain by dependence-aware intervals. The overall CV-selected classifiers worsened holdout performance, and the new return regressions did not beat the existing monthly return models. Keep the original return model as the reference; treat the direction alternatives as challengers for prospective evaluation.", "",
        table(combined, cols, cols[2:]), "",
        "Read the full reports for feature/model matrices, return error, optional uncertain-band coverage, nonoverlapping samples and confidence intervals:", "",
        "- [30-day results](artifacts/direction/30calendar/report.md)",
        "- [28-day results](artifacts/direction/28calendar/report.md)",
        "- [Paper order/parameter search](docs/PAPER_PARAMETER_SEARCH.md)",
        "- [Feature definitions](docs/FEATURE_ENGINEERING.md)",
        "- [Validation and neural suitability](docs/VALIDATION.md)", "",
        "![Feature and ensemble comparisons](feature_comparison.png)", "",
        "![Precision and recall comparison](precision_recall_comparison.png)", "",
        "All gains and losses are recorded; original MonthFu models remain available. This is a reused historical holdout, so improved metrics are exploratory. A direction advantage does not automatically imply better return error. Large deep networks were judged unsuitable for the effective monthly sample count; the controlled ELM search was run instead.", ""]
    (EXP/"RESULTS.md").write_text("\n".join(summary))
    print(f"Wrote {EXP/'RESULTS.md'} and full per-horizon reports")


if __name__ == "__main__":
    build()
