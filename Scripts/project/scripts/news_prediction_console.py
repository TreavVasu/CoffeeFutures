"""Local prediction controls and dashboard for the all-input Arabica notebook."""
from __future__ import annotations

import html
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "Scripts/project/artifacts/.matplotlib"))

import ipywidgets as widgets
import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import HTML, display
from matplotlib.ticker import PercentFormatter
from threadpoolctl import threadpool_limits

from train_all_inputs_news_model import build_frame

OUTPUTS = ROOT / "Scripts/project/artifacts/outputs"
PLOTS = ROOT / "Scripts/project/artifacts/plots"
BUNDLE_PATH = ROOT / "Scripts/project/artifacts/models/arabica_all_inputs_news_model.joblib"
DASHBOARD_PATH = PLOTS / "latest_arabica_news_prediction_console.png"
LABELS = {"all_inputs": "Price + COT + weather + news", "without_news": "Price + COT + weather"}
BG, PANEL, FG, MUTED = "#101619", "#1a2428", "#edf2f3", "#acbcc2"
GREEN, RED, AMBER, CYAN = "#72d493", "#f08b80", "#edc370", "#66c7d6"


def forecast(bundle, frame, *, view="latest", variant="all_inputs", rows=30, threshold=0.005):
    if view not in {"latest", "holdout"} or variant not in LABELS:
        raise ValueError("Unknown prediction view or model.")
    if not 5 <= rows <= 180 or not 0 <= threshold <= 0.03:
        raise ValueError("Rows or threshold outside the console range.")
    history = frame.copy()
    if view == "holdout":
        span = bundle["metadata"]["split"]["test"]
        history = history.loc[history.Date.between(span["start"], span["end"])].copy()
    history = history.tail(rows).copy()
    if history.empty:
        raise ValueError("No rows are available for this view.")
    output = history[["Date", "Close"]].copy()
    calendar = pd.DatetimeIndex(frame.Date)
    positions = calendar.get_indexer(history.Date) + 5
    output["forecast_target_date"] = [
        calendar[index] if index < len(calendar) else date + pd.offsets.BDay(5)
        for index, date in zip(positions, history.Date)
    ]
    with threadpool_limits(limits=1):
        for name in LABELS:
            model = (bundle["evaluation_models"][name] if view == "holdout" else
                     bundle["model"] if name == "all_inputs" else bundle["without_news_model"])
            columns = bundle["feature_variants"][name]
            output[f"{name}_return_5d"] = model.predict(history[columns])
    output["predicted_return_5d"] = output[f"{variant}_return_5d"]
    output["predicted_close_5d"] = output.Close * (1 + output.predicted_return_5d)
    output["news_model_delta"] = output.all_inputs_return_5d - output.without_news_return_5d
    output["signal"] = np.where(output.predicted_return_5d > threshold, "positive",
                                np.where(output.predicted_return_5d < -threshold, "negative", "neutral"))
    holdout = pd.read_csv(OUTPUTS / "arabica_all_inputs_news_holdout_predictions.csv")
    residuals = holdout.target_return_5d - holdout[f"{variant}_predicted_return_5d"]
    lower, upper = residuals.quantile([0.05, 0.95])
    output["return_low_90"] = output.predicted_return_5d + lower
    output["return_high_90"] = output.predicted_return_5d + upper
    return history, output.reset_index(drop=True)


def panel(axis, title):
    axis.set_facecolor(PANEL)
    for spine in axis.spines.values():
        spine.set_color("#52656e")
    axis.tick_params(colors=MUTED, labelsize=9)
    axis.set_title(title, loc="left", color=FG, fontsize=11, fontweight="bold", pad=12)
    axis.xaxis.label.set_color(MUTED)
    axis.yaxis.label.set_color(MUTED)
    axis.grid(color="#405259", alpha=0.4, linewidth=0.6)
    axis.set_axisbelow(True)


def stat(axis, title, value, detail, color):
    panel(axis, title)
    axis.set_xticks([])
    axis.set_yticks([])
    axis.text(0.05, 0.48, value, transform=axis.transAxes, color=color, fontsize=27, fontweight="bold")
    axis.text(0.05, 0.16, detail, transform=axis.transAxes, color=MUTED, fontsize=9)


def date_axis(axis):
    axis.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=6))
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    axis.tick_params(axis="x", rotation=20)


def draw_dashboard(history, predictions, bundle, variant, threshold, view, path=DASHBOARD_PATH):
    metrics = pd.DataFrame(bundle["metadata"]["comparison"])
    test = metrics.loc[metrics.phase.eq("test")].set_index("variant")
    latest, context = predictions.iloc[-1], history.iloc[-1]
    signal_color = GREEN if latest.signal == "positive" else RED if latest.signal == "negative" else AMBER
    fig = plt.figure(figsize=(18, 13), facecolor=BG)
    grid = fig.add_gridspec(4, 12, left=0.065, right=0.975, top=0.865, bottom=0.085,
                           height_ratios=[1, 2.0, 2.9, 1.7], hspace=0.68, wspace=1.1)
    fig.suptitle("ARABICA COFFEE FUTURES / NEWS DASHBOARD", color=FG, fontsize=22, fontweight="bold", y=0.961)
    basis = "Pre-test model / holdout replay" if view == "holdout" else "Final refit / latest local data"
    fig.text(0.065, 0.914, f"{LABELS[variant]}  |  5-trading-day returns  |  as of {latest.Date:%Y-%m-%d}", color=MUTED, fontsize=11)
    fig.text(0.975, 0.89, basis, color=CYAN, ha="right", fontsize=10)
    stat(fig.add_subplot(grid[0, :4]), "PREDICTED FIVE-DAY RETURN", f"{latest.predicted_return_5d:+.2%}",
         f"{latest.signal.upper()}  |  implied close {latest.predicted_close_5d:.2f} cents/lb", signal_color)
    stat(fig.add_subplot(grid[0, 4:8]), "NEWS MODEL DIFFERENCE", f"{latest.news_model_delta * 100:+.3f} pp",
         "With-news minus separately fitted without-news model", CYAN)
    stat(fig.add_subplot(grid[0, 8:]), "HOLDOUT DIRECTION ACCURACY", f"{test.loc[variant, 'directional_accuracy']:.2%}",
         f"1,003 test rows  |  return RMSE {test.loc[variant, 'rmse']:.4%}", AMBER)

    axis = fig.add_subplot(grid[1, :4])
    panel(axis, "YAHOO FUTURES PRICE HISTORY")
    axis.fill_between(history.Date, history.Low, history.High, color=CYAN, alpha=0.15)
    axis.plot(history.Date, history.Close, color=FG, linewidth=1.8, label="Observed close")
    axis.plot(history.Date, history.Close.rolling(20, min_periods=1).mean(), color=AMBER, label="20-session mean")
    axis.set_ylabel("US cents/lb")
    date_axis(axis)
    axis.legend(frameon=False, labelcolor=FG, fontsize=8)

    axis = fig.add_subplot(grid[1, 4:8])
    panel(axis, "RELEASED COT POSITIONING")
    values = [float(context.get("commercial_net_pct_oi", 0)), float(context.get("noncommercial_net_pct_oi", 0))]
    bars = axis.barh(["Comm.", "Spec."], values, color=[RED, GREEN], height=0.45)
    axis.bar_label(bars, labels=[f"{value:+.1%}" for value in values], color=FG, padding=5)
    extent = max(abs(np.asarray(values)).max() * 1.4, 0.15)
    axis.set_xlim(-extent, extent)
    axis.axvline(0, color=MUTED, linewidth=1)
    axis.set_xlabel("Net positions / open interest")
    axis.xaxis.set_major_formatter(PercentFormatter(1))

    axis = fig.add_subplot(grid[1, 8:])
    panel(axis, "BRAZIL / MINAS GERAIS WEATHER")
    prefix = "weather_brazil_minas_gerais_"
    weather = [("Mean temperature", "temperature_2m_mean", "C"),
               ("Rainfall", "precipitation_sum", "mm"),
               ("Relative humidity", "relative_humidity_2m_mean", "%"),
               ("Soil moisture", "soil_moisture_0_to_100cm_mean", "m3/m3")]
    axis.set_xticks([])
    axis.set_yticks([])
    for index, (label, name, unit) in enumerate(weather):
        value = float(context.get(prefix + name, np.nan))
        text = f"{value:.2f} {unit}" if np.isfinite(value) else "Unavailable"
        y = 0.85 - index * 0.23
        axis.text(0.05, y, label, transform=axis.transAxes, color=MUTED, fontsize=10)
        axis.text(0.95, y, text, transform=axis.transAxes, color=FG, fontsize=11, ha="right")

    axis = fig.add_subplot(grid[2, :8])
    panel(axis, "FIVE-SESSION TARGET CLOSE / NEWS COMPARISON")
    axis.plot(history.Date, history.Close, color=FG, linewidth=1.5, label="Observed close")
    for name, color in [("all_inputs", CYAN), ("without_news", AMBER)]:
        axis.plot(predictions.forecast_target_date, predictions.Close * (1 + predictions[f"{name}_return_5d"]),
                  color=color, linewidth=1.5, label="With news" if name == "all_inputs" else "Without news")
    axis.fill_between(predictions.forecast_target_date, predictions.Close * (1 + predictions.return_low_90),
                      predictions.Close * (1 + predictions.return_high_90), color=CYAN, alpha=0.12,
                      label="Historical 90% residual range")
    axis.axvline(latest.Date, color=MUTED, linestyle="--", linewidth=1)
    axis.set_ylabel("US cents/lb")
    date_axis(axis)
    axis.legend(frameon=False, labelcolor=FG, fontsize=8, loc="upper left", ncols=2)

    axis = fig.add_subplot(grid[2, 8:])
    panel(axis, "AVAILABLE NEWS CONTEXT")
    specs = [("Balance", "news_weekly_bull_minus_bear_share"),
             ("Intensity", "news_weekly_mean_event_intensity"),
             ("Coffee %", "news_weekly_explicit_coffee_share"),
             ("Risk text", "news_weekly_text_disruption"),
             ("Coffee", "news_weekly_text_coffee")]
    values = [float(context.get(col, 0)) for _, col in specs]
    bars = axis.barh([label for label, _ in specs], values, height=0.5,
                     color=[CYAN if value >= 0 else RED for value in values])
    axis.bar_label(bars, fmt="%.3f", color=FG, padding=4, fontsize=9)
    axis.invert_yaxis()
    axis.set_xlim(min(min(values), 0) - 0.12, max(max(values), 0.15) * 1.45)
    axis.set_xlabel("Feature values; text = log(1 + matches)", fontsize=9)
    source_date = pd.Timestamp(context.weekly_latest_source_date)
    axis.text(0, -0.28, f"Weekly source through {source_date:%Y-%m-%d}\nSix-session availability delay", transform=axis.transAxes,
              color=MUTED, fontsize=9)

    axis = fig.add_subplot(grid[3, :8])
    panel(axis, "RETURN SIGNAL VS DECISION THRESHOLD")
    values = predictions.predicted_return_5d
    colors = [GREEN if value > threshold else RED if value < -threshold else AMBER for value in values]
    axis.bar(predictions.Date, values, color=colors, width=0.8)
    axis.axhspan(-threshold, threshold, color=MUTED, alpha=0.12)
    axis.axhline(threshold, color=GREEN, linestyle="--", linewidth=1)
    axis.axhline(-threshold, color=RED, linestyle="--", linewidth=1)
    axis.yaxis.set_major_formatter(PercentFormatter(1, decimals=1))
    date_axis(axis)
    axis.set_ylabel("Predicted return")

    axis = fig.add_subplot(grid[3, 8:])
    panel(axis, "TEST CHANGE FROM ADDING NEWS")
    delta = bundle["metadata"]["news_delta_vs_without_news"]
    axis.set_xticks([])
    axis.set_yticks([])
    axis.text(0.05, 0.77, f"Direction  {delta['directional_accuracy_delta_pp']:+.2f} pp", transform=axis.transAxes, color=RED, fontsize=14)
    axis.text(0.05, 0.49, f"RMSE  {delta['rmse_relative_change_pct']:+.3f}% relative", transform=axis.transAxes, color=AMBER, fontsize=13)
    axis.text(0.05, 0.15, "No demonstrated accuracy gain", transform=axis.transAxes, color=MUTED, fontsize=11)
    fig.text(0.065, 0.025, "Model difference is not causal news attribution. Historical ranges are descriptive. Future target dates use weekdays beyond the local calendar.",
             color=MUTED, fontsize=9)
    fig.savefig(path, dpi=140, facecolor=BG)
    plt.close(fig)
    return path


class NewsPredictionConsole:
    def __init__(self):
        self.bundle = joblib.load(BUNDLE_PATH)
        self.frame, _ = build_frame()
        self.last_error = None
        self.last_predictions = None
        self.last_dashboard_path = None
        control_layout = widgets.Layout(width="100%", max_width="440px")
        style = {"description_width": "100px"}
        self.model = widgets.Dropdown(options=[(label, name) for name, label in LABELS.items()], value="all_inputs",
                                      description="Model:", layout=control_layout, style=style)
        self.view = widgets.ToggleButtons(options=[("Latest refit", "latest"), ("Holdout replay", "holdout")], value="latest")
        self.rows = widgets.IntSlider(value=30, min=5, max=180, step=5, description="Rows:", continuous_update=False,
                                      layout=control_layout, style=style)
        self.threshold = widgets.FloatSlider(value=0.005, min=0, max=0.03, step=0.001, description="Threshold:",
                                             readout_format=".1%", continuous_update=False, layout=control_layout, style=style)
        self.run_button = widgets.Button(description="Run Prediction", button_style="primary", icon="play")
        self.status = widgets.HTML("<b>Ready</b>")
        self.prediction_out, self.news_out, self.context_out = widgets.Output(), widgets.Output(), widgets.Output()
        self.dashboard_image = widgets.Image(format="png", layout=widgets.Layout(width="100%", height="auto"))
        self.dashboard_box = widgets.VBox([self.dashboard_image])
        header = widgets.HTML("<h3 style='margin:8px 0'>Arabica Coffee C / Returns</h3>")
        inputs = widgets.VBox([header, self.model, self.view, self.rows, self.threshold,
                               widgets.HBox([self.run_button]), self.status])
        self.tabs = widgets.Tab(children=[inputs, self.prediction_out, self.dashboard_box, self.news_out, self.context_out],
                                layout=widgets.Layout(width="100%"))
        self.tabs.add_class("arabica-news-console")
        for index, title in enumerate(["Inputs", "Latest Prediction", "Dashboard", "News Impact", "Context"]):
            self.tabs.set_title(index, title)
        self.run_button.on_click(self.run)

    def run(self, _=None):
        self.run_button.disabled = True
        self.last_error = None
        self.status.value = "<b>Running...</b>"
        try:
            history, output = forecast(self.bundle, self.frame, view=self.view.value, variant=self.model.value,
                                       rows=self.rows.value, threshold=self.threshold.value)
            self.last_predictions = output
            if self.view.value == "latest":
                path = DASHBOARD_PATH
                output_path = OUTPUTS / "latest_arabica_news_console_predictions.csv"
            else:
                path = PLOTS / "arabica_news_holdout_console.png"
                output_path = OUTPUTS / "arabica_news_holdout_console_predictions.csv"
            self.last_dashboard_path = draw_dashboard(history, output, self.bundle, self.model.value,
                                                       self.threshold.value, self.view.value, path)
            output.to_csv(output_path, index=False)
            self.dashboard_image.value = path.read_bytes()
            latest = output.iloc[-1]
            self.prediction_out.clear_output(wait=True)
            with self.prediction_out:
                display(HTML(f"<h3>{'Holdout replay' if self.view.value == 'holdout' else 'Latest refit estimates'}</h3>"))
                display(output.tail(10).style.format({"Close": "{:.2f}", "predicted_return_5d": "{:+.2%}",
                    "predicted_close_5d": "{:.2f}", "news_model_delta": "{:+.3%}",
                    "all_inputs_return_5d": "{:+.2%}", "without_news_return_5d": "{:+.2%}",
                    "return_low_90": "{:+.2%}", "return_high_90": "{:+.2%}"}))
            self.news_out.clear_output(wait=True)
            with self.news_out:
                display(HTML(f"<h3>News model difference: {latest.news_model_delta * 100:+.3f} percentage points</h3>"
                             "<p>With-news minus separately fitted no-news model. This is not causal attribution.</p>"))
                features = self.bundle["metadata"]["source_features"]["news"]
                display(pd.DataFrame({"Feature": features, "Latest eligible value": history.iloc[-1][features].to_numpy()}))
                comparison = pd.DataFrame(self.bundle["metadata"]["comparison"])
                display(comparison.loc[comparison.phase.eq("test"), ["variant", "rmse", "mae", "directional_accuracy"]])
            self.context_out.clear_output(wait=True)
            with self.context_out:
                display(pd.DataFrame({"Source": list(self.bundle["metadata"]["source_feature_counts"]),
                                      "Supplied features": list(self.bundle["metadata"]["source_feature_counts"].values())}))
                display({"mode": self.view.value, "features_as_of": str(latest.Date.date()),
                         "daily_news_source": str(history.iloc[-1].daily_latest_source_date.date()),
                         "weekly_news_source": str(history.iloc[-1].weekly_latest_source_date.date()),
                         "news_delay_sessions": 6, "forecast_horizon_sessions": 5,
                         "evaluation_fit_end": self.bundle["metadata"]["evaluation_model_fit_end"],
                         "production_fit_end": self.bundle["metadata"]["production_model_fit_end"]})
            self.status.value = f"<b>Done.</b> {latest.Date:%Y-%m-%d}: {latest.signal}, {latest.predicted_return_5d:+.2%}"
            self.tabs.selected_index = 2
        except Exception as exc:
            self.last_error = exc
            self.status.value = f"<b>Error:</b> {html.escape(str(exc))}"
            raise
        finally:
            self.run_button.disabled = False


def create_console():
    return NewsPredictionConsole()
