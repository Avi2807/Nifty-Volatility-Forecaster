"""Train the classifier on PhraseBank, score business headlines, write the daily signal."""

from __future__ import annotations

import json
from collections import Counter

import numpy as np

from src.config import WINDOW_END, WINDOW_START, RunPaths
from src.data.features import DailySentiment, roll_to_trading_day
from src.data.loaders import load_business_headlines, load_phrasebank, write_table
from src.sentiment.model import (
    CLASSES,
    NEGATIVE,
    POSITIVE,
    SoftmaxRegression,
    classification_report,
    stratified_split,
    train,
)
from src.sentiment.text import TfidfVectorizer

HOLDOUT_SHARE = 0.2
SPLIT_SEED = 42
TRAIN_SEED = 42
CHUNK_SIZE = 10_000
EXTREMES_PER_DAY = 5
TOP_TERMS = 15


def deduplicate(sentences: list[str], labels: list[str]) -> tuple[list[str], list[str], int]:
    """Keep the first copy of each exact sentence, so a copy can't sit on both sides of the split."""
    seen: set[str] = set()
    unique_sentences, unique_labels = [], []
    for sentence, label in zip(sentences, labels):
        if sentence not in seen:
            seen.add(sentence)
            unique_sentences.append(sentence)
            unique_labels.append(label)
    return unique_sentences, unique_labels, len(sentences) - len(unique_sentences)


def score_in_chunks(
    model: SoftmaxRegression, vectorizer: TfidfVectorizer, texts: np.ndarray, chunk_size: int = CHUNK_SIZE
) -> tuple[np.ndarray, np.ndarray]:
    """(P(pos) - P(neg), whether negative is the top class) for each text, chunk by chunk."""
    scores = np.empty(texts.size)
    is_negative = np.empty(texts.size, dtype=bool)
    for start in range(0, texts.size, chunk_size):
        stop = start + chunk_size
        probabilities = model.predict_proba(vectorizer.transform(texts[start:stop]))
        scores[start:stop] = probabilities[:, POSITIVE] - probabilities[:, NEGATIVE]
        is_negative[start:stop] = probabilities.argmax(axis=1) == NEGATIVE
    return scores, is_negative


def aggregate_daily(
    headline_dates: np.ndarray, scores: np.ndarray, is_negative: np.ndarray, trading_days: np.ndarray
) -> DailySentiment:
    """Roll each headline forward to its trading day and summarise each day.

    Only days that received a headline are returned. Headlines after the last session are unused.
    """
    day = roll_to_trading_day(headline_dates, trading_days)
    used = day >= 0
    n_days = trading_days.size
    counts = np.bincount(day[used], minlength=n_days).astype(np.float64)
    score_sum = np.bincount(day[used], weights=scores[used], minlength=n_days)
    negative_sum = np.bincount(day[used], weights=is_negative[used].astype(np.float64), minlength=n_days)
    has_news = counts > 0
    return DailySentiment(
        trading_days[has_news],
        score_sum[has_news] / counts[has_news],
        negative_sum[has_news] / counts[has_news],
        np.log1p(counts[has_news]),
    )


def daily_extremes(
    headline_dates: np.ndarray, scores: np.ndarray, texts: np.ndarray, trading_days: np.ndarray, k: int = EXTREMES_PER_DAY
) -> dict[str, np.ndarray]:
    """The k most negative and k most positive distinct headlines of each trading day."""
    day = roll_to_trading_day(headline_dates, trading_days)
    used = np.flatnonzero(day >= 0)
    order = used[np.lexsort((scores[used], day[used]))]  # by day, then score ascending
    boundaries = np.flatnonzero(np.diff(day[order])) + 1
    columns: dict[str, list] = {"date": [], "side": [], "rank": [], "score": [], "headline": []}
    for rows in np.split(order, boundaries):
        for side, ranked in (("negative", rows), ("positive", rows[::-1])):
            seen: set[str] = set()
            for row in ranked:
                if texts[row] in seen:
                    continue
                seen.add(texts[row])
                columns["date"].append(trading_days[day[row]])
                columns["side"].append(side)
                columns["rank"].append(len(seen))
                columns["score"].append(scores[row])
                columns["headline"].append(texts[row])
                if len(seen) == k:
                    break
    return {
        "date": np.array(columns["date"], dtype="datetime64[D]"),
        "side": np.array(columns["side"]),
        "rank": np.array(columns["rank"]),
        "score": np.array(columns["score"]),
        "headline": np.array(columns["headline"], dtype=object),
    }


def top_terms(model: SoftmaxRegression, vectorizer: TfidfVectorizer, n: int = TOP_TERMS) -> dict[str, list[str]]:
    """Terms that push hardest toward each class, relative to the other two classes."""
    terms = np.array(vectorizer.terms, dtype=object)
    margin = model.W - model.W.mean(axis=1, keepdims=True)
    return {name: terms[np.argsort(-margin[:, c])[:n]].tolist() for c, name in enumerate(CLASSES)}


def run_sentiment(paths: RunPaths, trading_days: np.ndarray) -> dict:
    """Run the whole sentiment stage and return the metrics it writes."""
    print(f"\n=== Sentiment ({paths.data_source} data) ===")
    raw_sentences, raw_labels = load_phrasebank(paths.raw)
    sentences, label_names, n_duplicates = deduplicate(raw_sentences, raw_labels)
    labels = np.array([CLASSES.index(name) for name in label_names])
    text_array = np.array(sentences, dtype=object)
    print(f"PhraseBank: {len(raw_sentences)} sentences, {n_duplicates} exact duplicates removed")

    train_rows, holdout_rows = stratified_split(labels, HOLDOUT_SHARE, SPLIT_SEED)
    vectorizer = TfidfVectorizer().fit(text_array[train_rows])
    x_train = vectorizer.transform(text_array[train_rows])
    x_holdout = vectorizer.transform(text_array[holdout_rows])

    model = SoftmaxRegression(len(vectorizer.terms))
    result = train(model, x_train, labels[train_rows], x_holdout, labels[holdout_rows], TRAIN_SEED)
    report = classification_report(labels[holdout_rows], model.predict_proba(x_holdout).argmax(axis=1))
    majority_accuracy = float(np.bincount(labels[holdout_rows]).max() / holdout_rows.size)
    terms = top_terms(model, vectorizer)

    print(f"Vocabulary: {len(vectorizer.terms)} terms from {train_rows.size} training sentences")
    print(f"Stopped at epoch {result.epochs_run}, best epoch {result.best_epoch}, holdout loss {result.best_val_loss:.4f}")
    print(f"Holdout accuracy {report['accuracy']:.4f}, macro F1 {report['macro_f1']:.4f} (always-majority accuracy {majority_accuracy:.4f})")
    for name, f1 in zip(CLASSES, report["f1"]):
        print(f"  {name:9s} F1 {f1:.4f}")
    for name, words in terms.items():
        print(f"  top {name}: {', '.join(words)}")

    headlines = load_business_headlines(paths.raw, WINDOW_START, WINDOW_END)
    scores, is_negative = score_in_chunks(model, vectorizer, headlines.texts)
    daily = aggregate_daily(headlines.dates, scores, is_negative, trading_days)

    paths.processed.mkdir(parents=True, exist_ok=True)
    write_table(
        paths.processed / "sentiment_daily.csv",
        {"date": daily.dates, "mean_sentiment": daily.mean_sentiment,
         "negative_share": daily.negative_share, "log_count": daily.log_count},
    )
    write_table(paths.processed / "headline_extremes.csv", daily_extremes(headlines.dates, scores, headlines.texts, trading_days))

    years = headlines.dates.astype("datetime64[Y]").astype(int) + 1970
    per_year = {int(y): int(n) for y, n in zip(*np.unique(years, return_counts=True))}
    categories = Counter(headlines.categories.tolist())
    metrics = {
        "data_source": paths.data_source,
        "phrasebank_sentences": len(raw_sentences),
        "duplicates_removed": n_duplicates,
        "train_sentences": int(train_rows.size),
        "holdout_sentences": int(holdout_rows.size),
        "vocabulary_size": len(vectorizer.terms),
        "best_epoch": result.best_epoch,
        "epochs_run": result.epochs_run,
        "holdout": report,
        "holdout_note": "This split also chose the early-stopping epoch, so these figures are slightly optimistic.",
        "majority_class_accuracy": majority_accuracy,
        "classes": list(CLASSES),
        "top_terms": terms,
        "headlines_scored": int(headlines.texts.size),
        "headlines_per_year": per_year,
        "headline_categories": dict(categories.most_common()),
        "sessions_with_news": int(daily.dates.size),
    }
    (paths.processed / "sentiment_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Scored {headlines.texts.size} business headlines in chunks of {CHUNK_SIZE}")
    print(f"Headlines per year: {per_year}")
    print(f"Categories: {dict(categories.most_common(8))}{' ...' if len(categories) > 8 else ''}")
    print(f"Sessions with news: {daily.dates.size}")
    return metrics
