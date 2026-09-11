from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns

warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "project" / "artifacts" / "outputs"
PLOT_DIR = ROOT / "project" / "artifacts" / "plots"
EVENT_DIR = ROOT / "data" / "events"
DEFAULT_CANDIDATES = OUTPUT_DIR / "gdelt_coffee_event_candidates_2000.csv"

NUMERIC_COLUMNS = [
    "GoldsteinScale",
    "NumMentions",
    "NumSources",
    "NumArticles",
    "AvgTone",
    "Close",
    "Volume",
    "return_1d",
    "future_return_1d",
    "future_return_5d",
    "future_return_10d",
    "future_return_20d",
    "realized_vol_20d",
    "Open_Interest_All",
    "managed_money_net",
    "commercial_net",
    "noncommercial_net",
    "managed_money_weekly_net_change",
    "commercial_weekly_net_change",
    "noncommercial_weekly_net_change",
    "open_interest_change_pct",
    "relevance_score_0_1",
    "event_intensity_score_0_1",
    "price_response_score_0_1",
    "rule_impact_score_0_1",
    "ollama_impact_score_0_1",
    "final_impact_score_0_1",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-score Coffee C event impact by daily or weekly periods using all extracted GDELT rows."
    )
    parser.add_argument("--candidates-path", default=str(DEFAULT_CANDIDATES))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--plots-dir", default=str(PLOT_DIR))
    parser.add_argument("--event-dir", default=str(EVENT_DIR))
    parser.add_argument("--period", choices=["daily", "weekly"], default="weekly")
    parser.add_argument("--date-column", choices=["event_date", "trading_date"], default="trading_date")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--top-events-per-period", type=int, default=5)
    parser.add_argument("--limit-periods", type=int, default=None)
    parser.add_argument("--use-ollama", action="store_true")
    parser.add_argument("--ollama-model", default="llama3.2:latest")
    parser.add_argument("--ollama-timeout", type=float, default=60.0)
    parser.add_argument("--ollama-num-predict", type=int, default=900)
    parser.add_argument("--ollama-sleep", type=float, default=0.2)
    parser.add_argument("--resume", action="store_true", help="Reuse existing period scores and only score missing periods.")
    parser.add_argument("--no-retry-failed", action="store_true", help="Do not retry parse-failed periods one at a time.")
    parser.add_argument("--no-row-dump", action="store_true", help="Skip writing the full row-level period dump.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = resolve_path(args.output_dir)
    plots_dir = resolve_path(args.plots_dir)
    event_dir = resolve_path(args.event_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    event_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    load_env_files([ROOT / ".env", ROOT / "data" / "centralData" / ".env"])

    candidates_path = resolve_path(args.candidates_path)
    year = infer_year(candidates_path)
    rows = load_candidates(candidates_path)
    rows = add_period_columns(rows, args.period, args.date_column)
    if not args.no_row_dump:
        row_dump_path = event_dir / f"gdelt_coffee_event_candidates_{year}_{args.period}_period_dump.csv"
        rows.to_csv(row_dump_path, index=False)
    else:
        row_dump_path = None

    summary = build_period_summary(rows, args.top_events_per_period)
    summary = add_deterministic_period_score(summary)
    if args.limit_periods:
        summary = summary.head(args.limit_periods).copy()

    aggregate_path = output_dir / f"gdelt_coffee_event_{args.period}_aggregate_{year}.csv"
    scores_path = output_dir / f"gdelt_coffee_event_{args.period}_ollama_scores_{year}.csv"
    scored_path = output_dir / f"gdelt_coffee_event_{args.period}_scored_{year}.csv"
    raw_batches_path = output_dir / f"gdelt_coffee_event_{args.period}_ollama_batches_{year}.jsonl"
    metadata_path = output_dir / f"gdelt_coffee_event_{args.period}_batch_metadata_{year}.json"

    summary.to_csv(aggregate_path, index=False)

    if args.use_ollama:
        ollama_scores = score_periods_with_ollama(summary, args, scores_path, raw_batches_path)
    else:
        ollama_scores = pd.DataFrame(columns=["period_id"])

    scored = merge_period_scores(summary, ollama_scores)
    scored.to_csv(scored_path, index=False)
    write_plots(scored, plots_dir, year, args.period)

    metadata = {
        "year": year,
        "source_candidates": str(candidates_path),
        "period": args.period,
        "date_column": args.date_column,
        "source_rows": int(len(rows)),
        "period_rows": int(len(summary)),
        "ollama_requested": bool(args.use_ollama),
        "ollama_model": args.ollama_model if args.use_ollama else None,
        "ollama_scored_periods": int(ollama_scores.loc[ollama_scores["ollama_status"].eq("scored"), "period_id"].nunique())
        if not ollama_scores.empty and "ollama_status" in ollama_scores
        else 0,
        "batch_size": args.batch_size,
        "top_events_per_period": args.top_events_per_period,
        "outputs": {
            "row_period_dump": str(row_dump_path) if row_dump_path else None,
            "aggregate": str(aggregate_path),
            "ollama_scores": str(scores_path),
            "scored_periods": str(scored_path),
            "raw_batches": str(raw_batches_path) if args.use_ollama else None,
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))

    print(f"Saved aggregate: {aggregate_path}")
    if row_dump_path:
        print(f"Saved row period dump: {row_dump_path}")
    print(f"Saved Ollama period scores: {scores_path}")
    print(f"Saved scored periods: {scored_path}")
    print(f"Saved metadata: {metadata_path}")
    display_cols = [
        "period_id",
        "event_count",
        "final_period_impact_score_0_1",
        "ollama_direction",
        "period_future_return_5d_mean",
        "close_period_return",
        "ollama_rationale",
    ]
    print(scored[display_cols].sort_values("final_period_impact_score_0_1", ascending=False).head(12).to_string(index=False))


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def infer_year(path: Path) -> int:
    match = re.search(r"(19|20)\d{2}", path.name)
    if match:
        return int(match.group(0))
    return 0


def load_env_files(paths: list[Path]) -> None:
    for path in paths:
        if not path.exists():
            continue
        for raw_line in path.read_text(errors="ignore").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
    if os.getenv("OLLAMA_KEY") and not os.getenv("OLLAMA_API_KEY"):
        os.environ["OLLAMA_API_KEY"] = os.getenv("OLLAMA_KEY", "")


def load_candidates(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    for col in ["event_date", "trading_date"]:
        if col in frame.columns:
            frame[col] = pd.to_datetime(frame[col], errors="coerce")
    for col in NUMERIC_COLUMNS:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    for col in ["keyword_relevant", "coffee_country_relevant", "coffee_country_name_relevant"]:
        if col in frame.columns:
            frame[col] = frame[col].fillna(False).astype(bool)
    if "final_impact_score_0_1" not in frame.columns:
        frame["final_impact_score_0_1"] = frame["rule_impact_score_0_1"]
    return frame


def add_period_columns(frame: pd.DataFrame, period: str, date_column: str) -> pd.DataFrame:
    out = frame.copy()
    score_date = pd.to_datetime(out[date_column], errors="coerce")
    out = out.loc[score_date.notna()].copy()
    score_date = score_date.loc[out.index]
    if period == "weekly":
        period_values = score_date.dt.to_period("W-SUN")
        out["score_period_start"] = period_values.dt.start_time.dt.normalize()
        out["score_period_end"] = period_values.dt.end_time.dt.normalize()
        out["period_id"] = out["score_period_start"].dt.strftime("%Y-%m-%d")
    else:
        out["score_period_start"] = score_date.dt.normalize()
        out["score_period_end"] = score_date.dt.normalize()
        out["period_id"] = out["score_period_start"].dt.strftime("%Y-%m-%d")
    return out


def build_period_summary(rows: pd.DataFrame, top_events_per_period: int) -> pd.DataFrame:
    summaries = []
    for period_id, group in rows.groupby("period_id", sort=True):
        group = group.copy()
        group = group.sort_values(["event_date", "rule_impact_score_0_1"], ascending=[True, False])
        top = group.sort_values("rule_impact_score_0_1", ascending=False).head(top_events_per_period)
        price_daily = (
            group.dropna(subset=["trading_date"])
            .sort_values("trading_date")
            .drop_duplicates("trading_date", keep="last")
        )
        direction_counts = group["impact_direction"].fillna("unknown").value_counts()
        event_count = len(group)
        summaries.append(
            {
                "period_id": period_id,
                "period_start": group["score_period_start"].iloc[0].date().isoformat(),
                "period_end": group["score_period_end"].iloc[0].date().isoformat(),
                "event_count": int(event_count),
                "unique_event_ids": int(group["GLOBALEVENTID"].nunique()) if "GLOBALEVENTID" in group else int(event_count),
                "event_day_count": int(group["event_date"].dt.date.nunique()),
                "explicit_coffee_event_count": int(group.get("keyword_relevant", pd.Series(False, index=group.index)).sum()),
                "producer_country_event_count": int(group.get("coffee_country_relevant", pd.Series(False, index=group.index)).sum()),
                "severe_rule_event_count": int(group["rule_impact_score_0_1"].ge(0.75).sum()),
                "max_rule_impact_score_0_1": safe_float(group["rule_impact_score_0_1"].max()),
                "mean_rule_impact_score_0_1": safe_float(group["rule_impact_score_0_1"].mean()),
                "max_row_final_impact_score_0_1": safe_float(group["final_impact_score_0_1"].max()),
                "mean_row_final_impact_score_0_1": safe_float(group["final_impact_score_0_1"].mean()),
                "mean_event_intensity_score_0_1": safe_float(group["event_intensity_score_0_1"].mean()),
                "mean_price_response_score_0_1": safe_float(group["price_response_score_0_1"].mean()),
                "bullish_event_count": int(direction_counts.get("bullish", 0)),
                "bearish_event_count": int(direction_counts.get("bearish", 0)),
                "neutral_event_count": int(direction_counts.get("neutral", 0)),
                "signed_rule_score_mean": signed_score_mean(group),
                "period_start_close": safe_float(price_daily["Close"].iloc[0]) if len(price_daily) else np.nan,
                "period_end_close": safe_float(price_daily["Close"].iloc[-1]) if len(price_daily) else np.nan,
                "close_period_return": close_period_return(price_daily),
                "period_future_return_5d_mean": safe_float(price_daily["future_return_5d"].mean()) if len(price_daily) else np.nan,
                "period_future_return_5d_abs_max": safe_float(price_daily["future_return_5d"].abs().max()) if len(price_daily) else np.nan,
                "period_future_return_20d_mean": safe_float(price_daily["future_return_20d"].mean()) if len(price_daily) else np.nan,
                "open_interest_last": last_numeric(price_daily, "Open_Interest_All"),
                "commercial_net_last": last_numeric(price_daily, "commercial_net"),
                "noncommercial_net_last": last_numeric(price_daily, "noncommercial_net"),
                "open_interest_change_pct_mean": safe_float(price_daily["open_interest_change_pct"].mean())
                if "open_interest_change_pct" in price_daily
                else np.nan,
                "top_countries": compact_counts(group.get("primary_country", pd.Series(index=group.index, dtype=object))),
                "top_event_roots": compact_counts(group.get("EventRootCode", pd.Series(index=group.index, dtype=object))),
                "top_event_digest": build_event_digest(top),
            }
        )
    return pd.DataFrame(summaries)


def safe_float(value) -> float:
    if pd.isna(value):
        return np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def signed_score_mean(group: pd.DataFrame) -> float:
    direction = group["impact_direction"].map({"bullish": 1, "bearish": -1, "neutral": 0}).fillna(0)
    signed = group["rule_impact_score_0_1"].fillna(0) * direction
    return safe_float(signed.mean())


def close_period_return(price_daily: pd.DataFrame) -> float:
    if len(price_daily) < 2:
        return np.nan
    start = price_daily["Close"].iloc[0]
    end = price_daily["Close"].iloc[-1]
    if pd.isna(start) or start == 0 or pd.isna(end):
        return np.nan
    return float(end / start - 1)


def last_numeric(frame: pd.DataFrame, column: str) -> float:
    if column not in frame:
        return np.nan
    values = frame[column].dropna()
    return safe_float(values.iloc[-1]) if len(values) else np.nan


def compact_counts(series: pd.Series, top_n: int = 6) -> str:
    counts = series.fillna("").astype(str)
    counts = counts.loc[counts.ne("")]
    if counts.empty:
        return ""
    return "; ".join(f"{key}:{value}" for key, value in counts.value_counts().head(top_n).items())


def build_event_digest(top: pd.DataFrame) -> str:
    parts = []
    for _, row in top.iterrows():
        date = pd.to_datetime(row.get("event_date"), errors="coerce")
        date_text = date.date().isoformat() if pd.notna(date) else ""
        summary = str(row.get("event_summary", ""))[:110].replace("|", "/")
        score = safe_float(row.get("rule_impact_score_0_1"))
        direction = str(row.get("impact_direction", ""))
        articles = safe_float(row.get("NumArticles"))
        parts.append(f"{date_text} score={score:.3f} dir={direction} articles={articles:.0f}: {summary}")
    return " | ".join(parts)


def add_deterministic_period_score(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    volume_max = max(float(out["event_count"].max()), 1.0)
    out["event_volume_score_0_1"] = np.log1p(out["event_count"]) / np.log1p(volume_max)
    future_abs = out["period_future_return_5d_abs_max"].fillna(0)
    response_scale = future_abs.quantile(0.9)
    if pd.isna(response_scale) or response_scale <= 0:
        response_scale = max(float(future_abs.max()), 0.05)
    out["period_price_response_score_0_1"] = (future_abs / response_scale).clip(0, 1)
    direction_total = out[["bullish_event_count", "bearish_event_count", "neutral_event_count"]].sum(axis=1).replace(0, np.nan)
    out["direction_concentration_score_0_1"] = (
        (out["bullish_event_count"] - out["bearish_event_count"]).abs() / direction_total
    ).fillna(0)
    out["deterministic_period_score_0_1"] = (
        0.36 * out["max_rule_impact_score_0_1"].fillna(0)
        + 0.18 * out["mean_rule_impact_score_0_1"].fillna(0)
        + 0.18 * out["event_volume_score_0_1"].fillna(0)
        + 0.18 * out["period_price_response_score_0_1"].fillna(0)
        + 0.10 * out["direction_concentration_score_0_1"].fillna(0)
    ).clip(0, 1)
    return out


def score_periods_with_ollama(
    summary: pd.DataFrame,
    args: argparse.Namespace,
    scores_path: Path,
    raw_batches_path: Path,
) -> pd.DataFrame:
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    api_key = os.getenv("OLLAMA_API_KEY") or os.getenv("OLLAMA_KEY") or ""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if not ollama_available(base_url, headers):
        print("Ollama endpoint was not reachable; writing deterministic period scores only.", file=sys.stderr)
        return pd.DataFrame(columns=["period_id"])

    existing = pd.DataFrame()
    scored_ids: set[str] = set()
    if args.resume and scores_path.exists():
        existing = pd.read_csv(scores_path)
        if "period_id" in existing:
            if "ollama_status" in existing:
                scored_ids = set(existing.loc[existing["ollama_status"].eq("scored"), "period_id"].astype(str))
            else:
                scored_ids = set(existing["period_id"].astype(str))

    todo = summary.loc[~summary["period_id"].astype(str).isin(scored_ids)].copy()
    if todo.empty:
        return existing

    all_scores = [] if existing.empty else existing.to_dict("records")
    raw_batches_path.write_text("" if not args.resume else raw_batches_path.read_text() if raw_batches_path.exists() else "")

    for batch_no, start in enumerate(range(0, len(todo), args.batch_size), start=1):
        batch = todo.iloc[start : start + args.batch_size].copy()
        prompt = build_batch_prompt(batch)
        parsed_scores = []
        raw_payload = {}
        try:
            response = requests.post(
                f"{base_url}/api/chat",
                headers=headers,
                json={
                    "model": args.ollama_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "format": "json",
                    "options": {"temperature": 0, "num_predict": args.ollama_num_predict},
                },
                timeout=args.ollama_timeout,
            )
            response.raise_for_status()
            raw_payload = response.json()
            content = raw_payload.get("message", {}).get("content") or raw_payload.get("response") or ""
            parsed_scores = parse_period_scores(content)
        except Exception as exc:
            parsed_scores = [
                {
                    "period_id": row["period_id"],
                    "ollama_status": "error",
                    "ollama_rationale": str(exc)[:500],
                }
                for _, row in batch.iterrows()
            ]

        normalized = normalize_period_scores(batch, parsed_scores)
        all_scores.extend(normalized)
        pd.DataFrame(all_scores).drop_duplicates("period_id", keep="last").to_csv(scores_path, index=False)
        with raw_batches_path.open("a") as handle:
            handle.write(
                json.dumps(
                    {
                        "batch_no": batch_no,
                        "period_ids": batch["period_id"].tolist(),
                        "parsed_count": len(parsed_scores),
                        "raw": raw_payload,
                    },
                    default=str,
                )
                + "\n"
            )
        print(f"Ollama batch {batch_no}: scored {len(normalized)} period(s)", flush=True)
        time.sleep(args.ollama_sleep)

    scores_df = pd.DataFrame(all_scores).drop_duplicates("period_id", keep="last")
    if not args.no_retry_failed:
        all_period_ids = set(summary["period_id"].astype(str))
        scored_period_ids = set(scores_df.loc[scores_df["ollama_status"].eq("scored"), "period_id"].astype(str))
        retry_ids = sorted(all_period_ids - scored_period_ids)
        for retry_no, period_id in enumerate(retry_ids, start=1):
            batch = summary.loc[summary["period_id"].astype(str).eq(period_id)].copy()
            if batch.empty:
                continue
            raw_payload = {}
            try:
                response = requests.post(
                    f"{base_url}/api/chat",
                    headers=headers,
                    json={
                        "model": args.ollama_model,
                        "messages": [{"role": "user", "content": build_batch_prompt(batch)}],
                        "stream": False,
                        "format": "json",
                        "options": {"temperature": 0, "num_predict": args.ollama_num_predict},
                    },
                    timeout=args.ollama_timeout,
                )
                response.raise_for_status()
                raw_payload = response.json()
                content = raw_payload.get("message", {}).get("content") or raw_payload.get("response") or ""
                parsed_scores = parse_period_scores(content)
            except Exception as exc:
                parsed_scores = [{"period_id": period_id, "ollama_status": "error", "ollama_rationale": str(exc)[:500]}]
            normalized = normalize_period_scores(batch, parsed_scores)
            all_scores = [row for row in all_scores if str(row.get("period_id")) != period_id]
            all_scores.extend(normalized)
            scores_df = pd.DataFrame(all_scores).drop_duplicates("period_id", keep="last")
            scores_df.to_csv(scores_path, index=False)
            with raw_batches_path.open("a") as handle:
                handle.write(
                    json.dumps(
                        {
                            "batch_no": f"retry_{retry_no}",
                            "period_ids": [period_id],
                            "parsed_count": len(parsed_scores),
                            "raw": raw_payload,
                        },
                        default=str,
                    )
                    + "\n"
                )
            print(f"Ollama retry {retry_no}: scored period {period_id}", flush=True)
            time.sleep(args.ollama_sleep)

    return pd.DataFrame(all_scores).drop_duplicates("period_id", keep="last")


def ollama_available(base_url: str, headers: dict) -> bool:
    try:
        response = requests.get(f"{base_url}/api/tags", headers=headers, timeout=5)
        return response.status_code < 500
    except Exception:
        return False


def build_batch_prompt(batch: pd.DataFrame) -> str:
    records = []
    prompt_cols = [
        "period_id",
        "period_start",
        "period_end",
        "event_count",
        "explicit_coffee_event_count",
        "producer_country_event_count",
        "severe_rule_event_count",
        "max_rule_impact_score_0_1",
        "mean_rule_impact_score_0_1",
        "bullish_event_count",
        "bearish_event_count",
        "neutral_event_count",
        "signed_rule_score_mean",
        "close_period_return",
        "period_future_return_5d_mean",
        "period_future_return_20d_mean",
        "open_interest_last",
        "commercial_net_last",
        "noncommercial_net_last",
        "top_countries",
        "top_event_roots",
        "top_event_digest",
    ]
    for _, row in batch.iterrows():
        record = {col: json_safe(row.get(col)) for col in prompt_cols}
        record["top_event_digest"] = compact_prompt_digest(record.get("top_event_digest", ""))
        records.append(record)
    return (
        "You are scoring aggregated GDELT event periods for likely impact on ICE Coffee C Arabica futures. "
        "Every period summary already includes all event rows for that period plus a digest of the highest-intensity rows. "
        "Return only valid JSON: {\"scores\":[...]} and include exactly one object for each input period_id. "
        "For each object use keys period_id, ollama_period_impact_score_0_1, direction, confidence_0_1, rationale, key_drivers. "
        "key_drivers must be a short array of 1 to 4 concise strings; do not copy event text into key_drivers. "
        "Score explicit coffee/crop/export/weather/logistics events highest. "
        "Do not treat a coffee-producing country name alone as direct coffee news. "
        "If explicit_coffee_event_count is 0 and the event digest does not mention coffee, crop, exports, weather, ports, transport, drought, frost, rain, harvest, or labor disruption, keep the score at or below 0.35. "
        "Country-only politics or conflict should be low unless the digest shows a plausible coffee supply, export, transport, labor, or demand mechanism. "
        "Never say explicit coffee events are present when explicit_coffee_event_count is 0. "
        "Use Coffee C price reaction and COT/open-interest context only as supporting evidence, not as the only reason. "
        "Input periods: "
        + json.dumps(records, separators=(",", ":"))
    )


def compact_prompt_digest(value: str, max_chars: int = 700) -> str:
    text = str(value or "")
    items = [item.strip() for item in text.split("|") if item.strip()]
    compact = " | ".join(items[:5])
    return compact[:max_chars]


def json_safe(value):
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if pd.isna(value):
        return None
    return value


def parse_period_scores(text: str) -> list[dict]:
    parsed = parse_first_json(text)
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict):
        for key in ["scores", "period_scores", "results", "periods"]:
            value = parsed.get(key)
            if isinstance(value, list):
                return value
        if "period_id" in parsed:
            return [parsed]
    return []


def parse_first_json(text: str):
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"(\{.*\}|\[.*\])", text, flags=re.S)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}


def normalize_period_scores(batch: pd.DataFrame, parsed_scores: list[dict]) -> list[dict]:
    by_id = {str(item.get("period_id")): item for item in parsed_scores if item.get("period_id") is not None}
    rows = []
    for _, source in batch.iterrows():
        period_id = str(source["period_id"])
        item = by_id.get(period_id, {})
        score = extract_score(item)
        confidence = clip_float(item.get("confidence_0_1", item.get("confidence")), 0, 1)
        rows.append(
            {
                "period_id": period_id,
                "ollama_period_impact_score_0_1": score,
                "ollama_direction": str(item.get("direction", "")).lower()[:40],
                "ollama_confidence_0_1": confidence,
                "ollama_rationale": str(item.get("rationale", ""))[:800],
                "ollama_key_drivers": stringify_drivers(item.get("key_drivers", "")),
                "ollama_status": "scored" if score is not None else "parse_failed",
            }
        )
    return rows


def extract_score(item: dict) -> float | None:
    for key in [
        "ollama_period_impact_score_0_1",
        "coffee_c_impact_score_0_1",
        "impact_score_0_1",
        "period_impact_score_0_1",
        "score",
        "severity",
    ]:
        if key in item:
            return clip_float(item.get(key), 0, 1)
    return None


def clip_float(value, low: float, high: float) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return float(min(high, max(low, number)))


def stringify_drivers(value) -> str:
    if isinstance(value, list):
        return "; ".join(str(item) for item in value)[:800]
    return str(value)[:800]


def merge_period_scores(summary: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    if scores.empty:
        out = summary.copy()
        out["ollama_period_impact_score_0_1"] = np.nan
        out["ollama_direction"] = ""
        out["ollama_confidence_0_1"] = np.nan
        out["ollama_rationale"] = ""
        out["ollama_key_drivers"] = ""
        out["ollama_status"] = "not_requested"
    else:
        out = summary.merge(scores, on="period_id", how="left")
        out["ollama_status"] = out["ollama_status"].fillna("not_scored")
    out["final_period_impact_score_0_1"] = out["deterministic_period_score_0_1"]
    scored_mask = out["ollama_period_impact_score_0_1"].notna()
    out.loc[scored_mask, "final_period_impact_score_0_1"] = (
        0.7 * out.loc[scored_mask, "ollama_period_impact_score_0_1"]
        + 0.3 * out.loc[scored_mask, "deterministic_period_score_0_1"]
    )
    return out.sort_values("period_start").reset_index(drop=True)


def write_plots(scored: pd.DataFrame, plots_dir: Path, year: int, period: str) -> None:
    if scored.empty:
        return
    frame = scored.copy()
    frame["period_start"] = pd.to_datetime(frame["period_start"], errors="coerce")
    frame = frame.sort_values("period_start")

    fig, ax1 = plt.subplots(figsize=(15, 6))
    ax1.bar(
        frame["period_start"],
        frame["final_period_impact_score_0_1"],
        width=5 if period == "weekly" else 0.8,
        color="#457b9d",
        alpha=0.68,
        label="Final period impact score",
    )
    ax1.set_ylabel("Impact score 0-1")
    ax2 = ax1.twinx()
    ax2.plot(frame["period_start"], frame["period_future_return_5d_mean"], color="#d1495b", marker="o", label="Mean next 5D return")
    ax2.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax2.set_ylabel("Mean next 5D Coffee C return")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    ax1.set_title(f"GDELT {period.title()} Coffee C Event Impact Scores, {year}")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate(rotation=35, ha="right")
    fig.tight_layout()
    fig.savefig(plots_dir / f"gdelt_coffee_event_{period}_impact_scores_{year}.png", dpi=180)
    plt.close(fig)

    plt.figure(figsize=(9, 6))
    sns.scatterplot(
        data=frame,
        x="final_period_impact_score_0_1",
        y="period_future_return_5d_mean",
        size="event_count",
        hue="ollama_status",
        sizes=(50, 240),
        alpha=0.78,
    )
    plt.axhline(0, color="gray", linestyle="--", linewidth=1)
    plt.title(f"{period.title()} Impact Score Vs Mean Next 5D Return, {year}")
    plt.xlabel("Final period impact score 0-1")
    plt.ylabel("Mean next 5D return")
    plt.tight_layout()
    plt.savefig(plots_dir / f"gdelt_coffee_event_{period}_score_vs_return_{year}.png", dpi=180)
    plt.close()


if __name__ == "__main__":
    main()
