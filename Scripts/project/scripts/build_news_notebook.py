"""Create the news-integrated companion, preserving the original notebook's stages."""
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[3]


def build():
    notebook = deepcopy(nbformat.from_dict(json.loads((ROOT / "ResearchModelTraining.ipynb").read_text())))
    sources = [
        ("markdown", """# Research Model Training / News Integrated

Arabica Coffee C **five-trading-day return prediction** using Yahoo price history, released COT positioning, coffee-region weather, and text-derived news features.

This companion retains the original workflow and prediction console. It uses the saved all-input model and local daily/weekly GDELT summaries. The evaluated news addition did not improve accuracy; the comparison below reports that result explicitly.
"""),
        ("code", """from pathlib import Path
import json
import os
import subprocess
import sys

ROOT = Path.cwd().resolve()
assert (ROOT / 'ResearchModelTraining.ipynb').exists(), 'Start the notebook from the repository root.'
os.environ['MPLCONFIGDIR'] = str(ROOT / 'Scripts/project/artifacts/.matplotlib')
sys.path.insert(0, str(ROOT / 'Scripts/project/scripts'))
import pandas as pd
from IPython.display import Image, display
from threadpoolctl import threadpool_limits
THREAD_LIMIT = threadpool_limits(limits=1)

OUTPUTS_DIR = ROOT / 'Scripts/project/artifacts/outputs'
PLOTS_DIR = ROOT / 'Scripts/project/artifacts/plots'
MODELS_DIR = ROOT / 'Scripts/project/artifacts/models'
DATA_PATH = ROOT / 'data/centralData/arabica_ml_model_ready.csv'
MODEL_PATH = MODELS_DIR / 'arabica_all_inputs_news_model.joblib'
print('Local workspace:', ROOT.name)
"""),
        ("markdown", "## 1. Data Check"),
        ("code", """ready = pd.read_csv(DATA_PATH, parse_dates=['Date'], low_memory=False)
required = [DATA_PATH,
    OUTPUTS_DIR / 'gdelt_coffee_events_2000_2026_daily_summary.csv',
    OUTPUTS_DIR / 'gdelt_coffee_events_2000_2026_weekly_summary.csv']
assert all(path.exists() for path in required), 'A required local input is missing.'
display(pd.DataFrame({'Input': [str(p.relative_to(ROOT)) for p in required],
                      'Available': [p.exists() for p in required]}))
print(f'{len(ready):,} market rows | {ready.Date.min():%Y-%m-%d} to {ready.Date.max():%Y-%m-%d}')
"""),
        ("markdown", """## 2. Train the All-Input Model

The saved model is loaded by default. Set `RETRAIN = True` to reproduce the paired training run using only local data. Learner selection uses validation RMSE; five-session purges separate training, validation and test labels. The final refit is stored separately from evaluation weights.
"""),
        ("code", """RETRAIN = False
if RETRAIN:
    subprocess.run([sys.executable, 'Scripts/project/scripts/train_all_inputs_news_model.py'], cwd=ROOT, check=True)
assert MODEL_PATH.exists(), 'Set RETRAIN = True to create the local model bundle.'
metrics = json.loads((OUTPUTS_DIR / 'arabica_all_inputs_news_metrics.json').read_text())
comparison = pd.read_csv(OUTPUTS_DIR / 'arabica_all_inputs_news_comparison.csv')
display(pd.DataFrame({'Source': list(metrics['source_feature_counts']),
                      'Feature count': list(metrics['source_feature_counts'].values())}))
print('Target:', metrics['target_formula'])
print('Model:', metrics['learner'])
"""),
        ("markdown", "## 3. Compare Metrics"),
        ("code", """display(comparison.style.format({'rmse': '{:.4%}', 'mae': '{:.4%}',
    'directional_accuracy': '{:.2%}', 'r2': '{:.4f}'}))
print('Selected feature variant on validation:', metrics['selected_variant_by_validation'])
"""),
        ("code", """delta = metrics['news_delta_vs_without_news']
display(pd.DataFrame([
    {'Measure': 'Direction accuracy change (percentage points)', 'Change': delta['directional_accuracy_delta_pp']},
    {'Measure': 'RMSE relative change (%)', 'Change': delta['rmse_relative_change_pct']},
    {'Measure': 'MAE change (return units)', 'Change': delta['mae_delta']},
]))
print(metrics['conclusion'])
"""),
        ("code", """yearly = pd.DataFrame([
    {'Year': year, 'Variant': variant, **values}
    for year, variants in metrics['yearly_test_metrics'].items()
    for variant, values in variants.items()
])
display(yearly[['Year', 'Variant', 'rows', 'rmse', 'directional_accuracy']].style.format(
    {'rmse': '{:.3%}', 'directional_accuracy': '{:.2%}'}))
"""),
        ("markdown", """## 4. News Impact Diagnostics

News impact here means the difference between separately fitted with-news and without-news models, not causal attribution. Structured event indicators and simple disruption/coffee text counts are delayed six price-calendar sessions because the historical event selection used subsequent five-session returns. Price-response and retrospective impact scores are excluded.
"""),
        ("code", """display(pd.DataFrame(metrics['features_used_by_fitted_model']))
display(pd.DataFrame({'News input': metrics['source_features']['news']}))
print('Paired block-bootstrap intervals:', metrics['news_bootstrap_vs_without_news'])
print('The earlier 61.35% direction result is withdrawn because it contained future-price information.')
"""),
        ("markdown", """## 5. Yahoo Real Price Comparison

The following table uses pre-test model weights and the same 1,003 held-out dates. Price-level forecasts are derived from predicted returns. These historical test results are exploratory, since this holdout has been inspected previously.
"""),
        ("code", """predictions = pd.read_csv(OUTPUTS_DIR / 'arabica_all_inputs_news_holdout_predictions.csv', parse_dates=['Date'])
predictions['actual_target_close_5d'] = predictions.Close * (1 + predictions.target_return_5d)
for variant in ['without_news', 'structured_news', 'all_inputs']:
    predictions[f'{variant}_predicted_close_5d'] = predictions.Close * (1 + predictions[f'{variant}_predicted_return_5d'])
predictions['news_model_delta'] = predictions.all_inputs_predicted_return_5d - predictions.without_news_predicted_return_5d
display(predictions.tail(10))
"""),
        ("code", """display(predictions[['Date', 'target_return_5d', 'without_news_predicted_return_5d',
    'all_inputs_predicted_return_5d', 'news_model_delta']].tail(15).style.format({
    'target_return_5d': '{:+.2%}', 'without_news_predicted_return_5d': '{:+.2%}',
    'all_inputs_predicted_return_5d': '{:+.2%}', 'news_model_delta': '{:+.3%}'}))
"""),
        ("code", """import matplotlib.pyplot as plt
fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
recent = predictions.tail(100)
axes[0].plot(recent.Date, recent.target_return_5d * 100, color='#626c72', label='Actual five-day return', alpha=0.7)
axes[0].plot(recent.Date, recent.all_inputs_predicted_return_5d * 100, color='#23826a', label='With news')
axes[0].plot(recent.Date, recent.without_news_predicted_return_5d * 100, color='#be8428', label='Without news')
axes[0].set_ylabel('Return (%)')
axes[0].legend(ncols=3)
axes[1].bar(recent.Date, recent.news_model_delta * 100, color='#398c9c')
axes[1].set_ylabel('News model delta (pp)')
axes[1].axhline(0, color='#626c72', linewidth=0.7)
fig.tight_layout()
fig.savefig(PLOTS_DIR / 'arabica_news_holdout_return_comparison.png', dpi=140)
display(fig)
plt.close(fig)
"""),
        ("markdown", "## 6. Visual Checks"),
        ("code", """display(Image(filename=str(PLOTS_DIR / 'arabica_all_inputs_news_evaluation.png')))
"""),
        ("markdown", "## 7. Artifact Check"),
        ("code", """artifacts = [MODEL_PATH, OUTPUTS_DIR / 'arabica_all_inputs_news_metrics.json',
    OUTPUTS_DIR / 'arabica_all_inputs_news_holdout_predictions.csv',
    OUTPUTS_DIR / 'arabica_all_inputs_news_latest_predictions.csv']
assert all(path.exists() for path in artifacts)
display(pd.DataFrame({'Artifact': [str(path.relative_to(ROOT)) for path in artifacts],
                      'Present': [path.exists() for path in artifacts]}))
"""),
        ("markdown", """## Prediction Console

The familiar Inputs, Latest Prediction, Dashboard and Context tabs now include a **News Impact** tab. Latest refit uses the final all-input model; Holdout replay uses the saved pre-test models. Both modes compare news against the price/COT/weather baseline. Future target dates beyond the observed calendar are weekday estimates.
"""),
        ("code", """from news_prediction_console import create_console, DASHBOARD_PATH
console = create_console()
display(console.tabs)
console.run()
assert console.last_error is None
"""),
        ("markdown", "## Saved News-Integrated Dashboard"),
        ("code", """display(Image(filename=str(DASHBOARD_PATH)))
print('Dashboard:', DASHBOARD_PATH.relative_to(ROOT))
print('Predictions:', 'Scripts/project/artifacts/outputs/latest_arabica_news_console_predictions.csv')
"""),
    ]
    notebook.cells = [nbformat.v4.new_markdown_cell(source.strip()) if kind == "markdown"
                      else nbformat.v4.new_code_cell(source.strip()) for kind, source in sources]
    notebook.metadata.pop("widgets", None)
    notebook.metadata.kernelspec = {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"}
    notebook.metadata["news_companion"] = {"source_notebook": "ResearchModelTraining.ipynb", "local_data_only": True}
    path = ROOT / "ResearchModelTrainingNews.ipynb"
    nbformat.validate(notebook)
    nbformat.write(notebook, path)
    return path


if __name__ == "__main__":
    print(build())
