from __future__ import annotations

import argparse
import html
import json
import os
import re
import sys
import time
from datetime import timedelta
from pathlib import Path
from urllib.parse import parse_qs, quote_plus, unquote, urlparse

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns

ROOT = Path(__file__).resolve().parents[3]


BULLISH_TERMS = {
    "frost": 5,
    "freeze": 5,
    "drought": 5,
    "dry weather": 4,
    "hot weather": 3,
    "heat": 3,
    "crop damage": 5,
    "crop losses": 5,
    "supply deficit": 5,
    "deficit": 3,
    "shortage": 5,
    "low inventories": 4,
    "stockpiles decline": 4,
    "exports fall": 4,
    "export decline": 4,
    "production drop": 4,
    "production cut": 4,
    "supply concern": 4,
    "supply concerns": 4,
    "supply strain": 4,
    "supply strains": 4,
    "harvest delay": 3,
    "logistics disruption": 3,
    "rally": 2,
    "surge": 2,
    "jump": 2,
    "soaring": 2,
    "surpass": 2,
    "highest": 2,
    "highs": 1,
    "record high": 3,
}

BEARISH_TERMS = {
    "rain": -3,
    "rains": -3,
    "favorable weather": -4,
    "crop recovery": -5,
    "bumper crop": -5,
    "surplus": -5,
    "oversupply": -5,
    "exports rise": -4,
    "export rise": -4,
    "production increase": -4,
    "production rises": -4,
    "inventories rise": -4,
    "stockpiles rise": -4,
    "harvest pressure": -3,
    "demand weakness": -4,
    "selloff": -2,
    "drop sharply": -4,
    "drop": -2,
    "falls": -2,
    "falling": -2,
    "declines": -2,
    "lower": -1,
    "ease": -1,
}

COFFEE_RELEVANCE_TERMS = [
    "coffee",
    "arabica",
    "robusta",
    "ice coffee",
    "coffee futures",
    "brazil",
    "colombia",
    "vietnam",
    "minas gerais",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Explain model drift/error dates with dated coffee news context.")
    parser.add_argument("--worst-dates", default="Scripts/project/artifacts/outputs/worst_error_dates_with_drift_context.csv")
    parser.add_argument("--output-dir", default="Scripts/project/artifacts/outputs")
    parser.add_argument("--plots-dir", default="Scripts/project/artifacts/plots")
    parser.add_argument("--top-n", type=int, default=25)
    parser.add_argument("--lookback-days", type=int, default=4)
    parser.add_argument("--lookahead-days", type=int, default=7)
    parser.add_argument("--max-articles", type=int, default=8)
    parser.add_argument("--sleep", type=float, default=0.4)
    parser.add_argument("--reuse-articles", action="store_true", help="Re-score existing news_articles_by_error_date.csv without searching.")
    parser.add_argument("--use-ollama", action="store_true", help="Optionally ask Ollama to classify article impact.")
    parser.add_argument("--ollama-model", default="llama3.1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = ROOT / args.output_dir
    plots_dir = ROOT / args.plots_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    load_dotenv(ROOT / "data" / "centralData" / ".env")
    worst = pd.read_csv(ROOT / args.worst_dates, parse_dates=["Date"]).head(args.top_n)
    article_rows = []
    summary_rows = []

    cached_articles = None
    cached_articles_path = output_dir / "news_articles_by_error_date.csv"
    if args.reuse_articles and cached_articles_path.exists():
        cached_articles = pd.read_csv(cached_articles_path)

    for _, row in worst.iterrows():
        event_date = row["Date"]
        start_date = event_date - timedelta(days=args.lookback_days)
        end_date = event_date + timedelta(days=args.lookahead_days)
        if cached_articles is not None:
            articles = cached_articles.loc[cached_articles["Date"].eq(event_date.date().isoformat())].to_dict("records")
        else:
            articles = search_news(event_date, start_date, end_date, max_articles=args.max_articles)
            time.sleep(args.sleep)
            if not articles:
                articles = search_duckduckgo(event_date, start_date, end_date, max_articles=args.max_articles)
                time.sleep(args.sleep)

        scored = [score_article(article, row) for article in articles]
        if args.use_ollama:
            scored = maybe_ollama_rescore(scored, row, args.ollama_model)

        for article in scored:
            article_rows.append(
                {
                    "Date": event_date.date().isoformat(),
                    "window_start": start_date.date().isoformat(),
                    "window_end": end_date.date().isoformat(),
                    **article,
                }
            )
        summary_rows.append(summarize_date(row, scored, start_date, end_date))

    articles_df = pd.DataFrame(article_rows)
    summary_df = pd.DataFrame(summary_rows)
    articles_path = output_dir / "news_articles_by_error_date.csv"
    summary_path = output_dir / "news_context_by_error_date.csv"
    articles_df.to_csv(articles_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    write_news_plots(summary_df, plots_dir)
    metadata = {
        "approach": "Public news search plus deterministic keyword impact scoring. Ollama is optional and disabled unless --use-ollama is passed.",
        "sources": ["GDELT 2.1 Doc API", "DuckDuckGo HTML fallback"],
        "lookback_days": args.lookback_days,
        "lookahead_days": args.lookahead_days,
        "top_n_error_dates": args.top_n,
        "ollama_used": args.use_ollama,
        "outputs": {
            "summary": str(summary_path),
            "articles": str(articles_path),
            "plots": str(plots_dir),
        },
    }
    (output_dir / "news_context_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Saved news summary: {summary_path}")
    print(f"Saved news article detail: {articles_path}")
    print(f"Saved news plots: {plots_dir}")


def search_news(event_date: pd.Timestamp, start_date: pd.Timestamp, end_date: pd.Timestamp, max_articles: int) -> list[dict]:
    query = '(arabica OR "coffee futures" OR "coffee prices")'
    params = {
        "query": query,
        "mode": "ArtList",
        "format": "json",
        "sort": "HybridRel",
        "maxrecords": max_articles,
        "startdatetime": start_date.strftime("%Y%m%d000000"),
        "enddatetime": end_date.strftime("%Y%m%d235959"),
    }
    try:
        response = requests.get("https://api.gdeltproject.org/api/v2/doc/doc", params=params, timeout=30)
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        print(f"GDELT search failed for {event_date.date()}: {exc}", file=sys.stderr)
        return []

    articles = []
    for item in payload.get("articles", []):
        title = html.unescape(item.get("title") or "")
        url = item.get("url") or ""
        if not is_relevant(title, url):
            continue
        articles.append(
            {
                "source_engine": "gdelt",
                "published": item.get("seendate") or "",
                "source": item.get("domain") or "",
                "title": title,
                "snippet": "",
                "url": url,
            }
        )
    return dedupe_articles(articles)[:max_articles]


def search_duckduckgo(
    event_date: pd.Timestamp,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    max_articles: int,
) -> list[dict]:
    query = (
        f'arabica coffee futures news {event_date:%Y-%m-%d} '
        f'coffee price after:{start_date:%Y-%m-%d} before:{end_date:%Y-%m-%d}'
    )
    headers = {"User-Agent": "Mozilla/5.0 ArabicaFuturesResearch/1.0"}
    try:
        response = requests.get(
            "https://html.duckduckgo.com/html/",
            params={"q": query},
            headers=headers,
            timeout=30,
        )
        response.raise_for_status()
    except Exception as exc:
        print(f"DuckDuckGo search failed for {event_date.date()}: {exc}", file=sys.stderr)
        return []

    return parse_duckduckgo_html(response.text, max_articles=max_articles)


def parse_duckduckgo_html(markup: str, max_articles: int) -> list[dict]:
    results = []
    blocks = re.findall(r'<div class="result results_links.*?</div>\s*</div>', markup, flags=re.S)
    for block in blocks:
        title_match = re.search(r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', block, flags=re.S)
        snippet_match = re.search(r'class="result__snippet"[^>]*>(.*?)</a>', block, flags=re.S)
        if not title_match:
            continue
        raw_url = html.unescape(title_match.group(1))
        title = clean_html(title_match.group(2))
        snippet = clean_html(snippet_match.group(1)) if snippet_match else ""
        url = clean_duckduckgo_url(raw_url)
        if not is_relevant(title, snippet, url):
            continue
        results.append(
            {
                "source_engine": "duckduckgo",
                "published": "",
                "source": urlparse(url).netloc,
                "title": title,
                "snippet": snippet,
                "url": url,
            }
        )
    return dedupe_articles(results)[:max_articles]


def score_article(article: dict, drift_row: pd.Series) -> dict:
    text = f"{article.get('title', '')} {article.get('snippet', '')}".lower()
    score = 0
    matched_terms = []
    for term, weight in BULLISH_TERMS.items():
        if term_matches(text, term):
            score += weight
            matched_terms.append(term)
    for term, weight in BEARISH_TERMS.items():
        if term_matches(text, term):
            score += weight
            matched_terms.append(term)

    relevance = sum(1 for term in COFFEE_RELEVANCE_TERMS if term in text)
    if relevance == 0:
        score *= 0.35

    actual_return = float(drift_row["target_return_5d"])
    direction_alignment = 0
    if abs(score) > 0:
        direction_alignment = int(np.sign(score) == np.sign(actual_return))
    weekly_strength = min(5.0, abs(score) * 0.75 + relevance * 0.25)
    article.update(
        {
            "impact_score": float(score),
            "weekly_strength_score": float(weekly_strength),
            "direction_alignment": direction_alignment,
            "matched_terms": "; ".join(matched_terms[:12]),
        }
    )
    return article


def term_matches(text: str, term: str) -> bool:
    escaped = re.escape(term.lower())
    return re.search(rf"(?<![a-z0-9]){escaped}(?![a-z0-9])", text) is not None


def summarize_date(row: pd.Series, articles: list[dict], start_date: pd.Timestamp, end_date: pd.Timestamp) -> dict:
    scores = np.array([article.get("impact_score", 0.0) for article in articles], dtype=float)
    strengths = np.array([article.get("weekly_strength_score", 0.0) for article in articles], dtype=float)
    actual_return = float(row["target_return_5d"])
    weighted_news_score = float(scores.sum()) if len(scores) else 0.0
    max_strength = float(strengths.max()) if len(strengths) else 0.0
    aligned_articles = int(sum(article.get("direction_alignment", 0) for article in articles))
    top_articles = articles[:3]
    explanation = build_news_explanation(actual_return, weighted_news_score, max_strength, top_articles)
    return {
        "Date": row["Date"].date().isoformat(),
        "window_start": start_date.date().isoformat(),
        "window_end": end_date.date().isoformat(),
        "actual_return_5d": actual_return,
        "predicted_return_5d": float(row["predicted_return_5d"]),
        "abs_return_error": float(row["abs_return_error"]),
        "absolute_price_error": float(row["absolute_price_error"]),
        "article_count": len(articles),
        "weighted_news_score": weighted_news_score,
        "max_weekly_strength_score": max_strength,
        "aligned_article_count": aligned_articles,
        "news_direction": "bullish" if weighted_news_score > 0 else "bearish" if weighted_news_score < 0 else "mixed/none",
        "news_aligned_with_actual_move": bool(np.sign(weighted_news_score) == np.sign(actual_return)) if weighted_news_score else False,
        "strong_weekly_news_effect": bool(max_strength >= 4 or abs(weighted_news_score) >= 6),
        "dominant_shift_context": row.get("dominant_shift_context", ""),
        "top_news_titles": " | ".join(article.get("title", "") for article in top_articles),
        "news_explanation": explanation,
    }


def build_news_explanation(actual_return: float, weighted_score: float, max_strength: float, articles: list[dict]) -> str:
    if not articles:
        return "No relevant dated public-news result found in the search window."
    direction = "upward" if actual_return > 0 else "downward"
    news_direction = "bullish" if weighted_score > 0 else "bearish" if weighted_score < 0 else "mixed"
    strength = "strong" if max_strength >= 4 or abs(weighted_score) >= 6 else "moderate/weak"
    terms = sorted({term for article in articles for term in article.get("matched_terms", "").split("; ") if term})
    term_text = ", ".join(terms[:8]) if terms else "no high-impact keyword cluster"
    return f"{strength} {news_direction} news signal versus an actual {direction} weekly move; matched: {term_text}."


def maybe_ollama_rescore(articles: list[dict], drift_row: pd.Series, model: str) -> list[dict]:
    api_key = os.getenv("OLLAMA_API_KEY", "")
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    for article in articles:
        prompt = (
            "Classify the likely one-week impact of this coffee-market news on Arabica coffee futures. "
            "Return compact JSON with keys impact_score (-5 to 5), rationale. "
            f"Actual 5d return: {float(drift_row['target_return_5d']):.4f}. "
            f"Title: {article.get('title', '')}. Snippet: {article.get('snippet', '')}"
        )
        try:
            response = requests.post(
                f"{base_url}/api/generate",
                headers=headers,
                json={"model": model, "prompt": prompt, "stream": False},
                timeout=45,
            )
            response.raise_for_status()
            text = response.json().get("response", "")
            parsed = parse_first_json(text)
            if "impact_score" in parsed:
                article["ollama_impact_score"] = float(parsed["impact_score"])
                article["ollama_rationale"] = parsed.get("rationale", "")
        except Exception as exc:
            article["ollama_error"] = str(exc)
    return articles


def write_news_plots(summary: pd.DataFrame, plots_dir: Path) -> None:
    if summary.empty:
        return
    sns.set_theme(style="whitegrid")
    frame = summary.copy()
    frame["Date"] = pd.to_datetime(frame["Date"])
    frame = frame.sort_values("Date").reset_index(drop=True)
    for column in [
        "actual_return_5d",
        "predicted_return_5d",
        "abs_return_error",
        "absolute_price_error",
        "article_count",
        "weighted_news_score",
        "max_weekly_strength_score",
        "aligned_article_count",
    ]:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    plt.figure(figsize=(12, 6))
    plt.scatter(frame["weighted_news_score"], frame["actual_return_5d"], s=65, alpha=0.75)
    for _, row in frame.nlargest(6, "abs_return_error").iterrows():
        plt.annotate(row["Date"].strftime("%Y-%m-%d"), (row["weighted_news_score"], row["actual_return_5d"]), fontsize=8)
    plt.axhline(0, color="gray", linestyle="--", linewidth=1)
    plt.axvline(0, color="gray", linestyle="--", linewidth=1)
    plt.title("News Impact Score Vs Actual 5D Return")
    plt.xlabel("Weighted news impact score")
    plt.ylabel("Actual 5D return")
    plt.tight_layout()
    plt.savefig(plots_dir / "news_impact_vs_return.png", dpi=180)
    plt.close()

    plt.figure(figsize=(14, 6))
    plt.plot(frame["Date"], frame["abs_return_error"], label="Absolute return error", linewidth=1.8)
    plt.bar(frame["Date"], frame["max_weekly_strength_score"] / 20.0, width=3, alpha=0.35, label="News weekly strength / 20")
    plt.title("News Strength Around Worst Model Error Dates")
    plt.ylabel("Return error / scaled news strength")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "news_strength_vs_model_error.png", dpi=180)
    plt.close()

    plt.figure(figsize=(12, 6))
    colors = np.where(frame["news_aligned_with_actual_move"], "#2a9d8f", "#e76f51")
    plt.bar(frame["Date"].dt.strftime("%Y-%m-%d"), frame["article_count"], color=colors)
    plt.title("Relevant News Count By Model Error Date")
    plt.ylabel("Article count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(plots_dir / "news_article_count_by_error_date.png", dpi=180)
    plt.close()

    dated = frame.copy()
    dated["date_label"] = dated["Date"].dt.strftime("%Y-%m-%d")
    score_colors = np.where(dated["weighted_news_score"] >= 0, "#2a9d8f", "#d1495b")

    fig, ax1 = plt.subplots(figsize=(15, 6.5))
    ax1.bar(
        dated["Date"],
        dated["weighted_news_score"],
        width=3.0,
        color=score_colors,
        alpha=0.68,
        label="Weighted news score",
    )
    ax1.axhline(0, color="black", linewidth=0.9)
    ax1.set_ylabel("Weighted news score")
    ax2 = ax1.twinx()
    ax2.plot(
        dated["Date"],
        dated["actual_return_5d"],
        color="#264653",
        marker="o",
        linewidth=2.0,
        label="Actual 5D return",
    )
    ax2.plot(
        dated["Date"],
        dated["predicted_return_5d"],
        color="#f4a261",
        marker="s",
        linewidth=1.4,
        alpha=0.8,
        label="Predicted 5D return",
    )
    ax2.axhline(0, color="#777777", linestyle="--", linewidth=0.8)
    ax2.set_ylabel("5D return")
    for _, row in dated.nlargest(min(10, len(dated)), "abs_return_error").iterrows():
        ax2.annotate(
            row["date_label"],
            (row["Date"], row["actual_return_5d"]),
            textcoords="offset points",
            xytext=(0, 9),
            ha="center",
            fontsize=8,
            rotation=35,
        )
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    ax1.set_title("Dated News Signal Vs Actual/Predicted 5D Arabica Returns")
    ax1.set_xlabel("Event date")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate(rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(plots_dir / "news_dated_score_vs_5d_returns.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(15, 6.5))
    alignment_colors = np.where(dated["news_aligned_with_actual_move"].astype(bool), "#2a9d8f", "#e76f51")
    ax.scatter(
        dated["Date"],
        dated["abs_return_error"],
        s=(dated["article_count"].fillna(0) + 1) * 26,
        c=alignment_colors,
        alpha=0.75,
        edgecolor="white",
        linewidth=0.7,
        label="Error date",
    )
    ax.plot(dated["Date"], dated["abs_return_error"], color="#4b5563", linewidth=1.0, alpha=0.7)
    for _, row in dated.nlargest(min(12, len(dated)), "abs_return_error").iterrows():
        ax.annotate(
            row["date_label"],
            (row["Date"], row["abs_return_error"]),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8,
            rotation=35,
        )
    ax.set_title("Dated News Context Around Largest 5D Model Errors")
    ax.set_xlabel("Event date")
    ax.set_ylabel("Absolute 5D return error")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate(rotation=45, ha="right")
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#2a9d8f", markersize=9, label="News aligned"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#e76f51", markersize=9, label="News not aligned"),
    ]
    ax.legend(handles=handles, loc="upper right")
    fig.tight_layout()
    fig.savefig(plots_dir / "news_dated_error_alignment.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(15, 7))
    top_for_labels = set(dated.nlargest(min(10, len(dated)), "max_weekly_strength_score")["Date"])
    scatter = ax.scatter(
        dated["Date"],
        dated["weighted_news_score"],
        s=(dated["max_weekly_strength_score"].fillna(0) + 1) * 42,
        c=dated["actual_return_5d"],
        cmap="RdYlGn",
        alpha=0.78,
        edgecolor="white",
        linewidth=0.8,
    )
    ax.axhline(0, color="black", linewidth=0.9)
    for _, row in dated.iterrows():
        if row["Date"] in top_for_labels or bool(row.get("strong_weekly_news_effect", False)):
            ax.annotate(
                row["date_label"],
                (row["Date"], row["weighted_news_score"]),
                textcoords="offset points",
                xytext=(0, 9),
                ha="center",
                fontsize=8,
                rotation=35,
            )
    ax.set_title("Dated News Impact Score, Strength, And Actual 5D Return")
    ax.set_xlabel("Event date")
    ax.set_ylabel("Weighted news score")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label("Actual 5D return")
    fig.autofmt_xdate(rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(plots_dir / "news_dated_impact_strength_bubble.png", dpi=180)
    plt.close(fig)

    table_cols = [
        "Date",
        "window_start",
        "window_end",
        "actual_return_5d",
        "predicted_return_5d",
        "abs_return_error",
        "article_count",
        "weighted_news_score",
        "max_weekly_strength_score",
        "news_direction",
        "news_aligned_with_actual_move",
        "strong_weekly_news_effect",
        "top_news_titles",
        "news_explanation",
    ]
    available_cols = [col for col in table_cols if col in dated.columns]
    dated[available_cols].to_csv(plots_dir.parent / "outputs" / "news_dated_plot_points.csv", index=False)


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def is_relevant(*parts: str) -> bool:
    text = " ".join(parts).lower()
    return any(term in text for term in ["coffee", "arabica", "robusta"])


def dedupe_articles(articles: list[dict]) -> list[dict]:
    seen = set()
    deduped = []
    for article in articles:
        key = (article.get("title", "").strip().lower(), article.get("url", "").strip().lower())
        if key in seen or not key[0]:
            continue
        seen.add(key)
        deduped.append(article)
    return deduped


def clean_html(value: str) -> str:
    value = re.sub(r"<.*?>", " ", value, flags=re.S)
    value = html.unescape(value)
    return re.sub(r"\s+", " ", value).strip()


def clean_duckduckgo_url(url: str) -> str:
    parsed = urlparse(url)
    if "duckduckgo.com" in parsed.netloc and parsed.path.startswith("/l/"):
        query = parse_qs(parsed.query)
        if "uddg" in query:
            return unquote(query["uddg"][0])
    return url


def parse_first_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}


if __name__ == "__main__":
    main()
