"""Sentiment stage: tokeniser, TF-IDF, classifier gradients, and the daily signal's dates."""

from __future__ import annotations

import numpy as np
import pytest

from src.nn.functional import stable_softmax
from src.nn.optim import AdamW
from src.sentiment.model import SoftmaxRegression, classification_report, stratified_split, train
from src.sentiment.pipeline import aggregate_daily, daily_extremes, deduplicate, score_in_chunks
from src.sentiment.text import TfidfVectorizer, tokenize


def days(*values: str) -> np.ndarray:
    return np.array(values, dtype="datetime64[D]")


def test_tokenizer_lowercases_strips_punctuation_and_replaces_digits():
    assert tokenize("Profit's UP, 10.5%!") == ["profits", "up", "NUM", "profits up", "up NUM"]


def test_numbers_with_separators_become_one_token():
    assert tokenize("Rs 4;940 crore") == ["rs", "NUM", "crore", "rs NUM", "NUM crore"]
    assert tokenize("1,020.5") == ["NUM"]


def test_tokenizer_handles_empty_text():
    assert tokenize("...") == []


def test_vocabulary_uses_min_count_and_ignores_unseen_terms():
    vectorizer = TfidfVectorizer(min_count=3).fit(["good good good", "bad bad"])
    assert "good" in vectorizer.terms
    assert "bad" not in vectorizer.terms  # only 2 occurrences
    assert vectorizer.transform(["never seen words"]).data.size == 0


def test_idf_formula_and_l2_normalisation():
    texts = ["a b", "a c", "a b"]
    vectorizer = TfidfVectorizer(min_count=1).fit(texts)
    index = vectorizer.term_index
    assert vectorizer.idf[index["a"]] == pytest.approx(np.log(4 / 4) + 1)
    assert vectorizer.idf[index["c"]] == pytest.approx(np.log(4 / 2) + 1)
    np.testing.assert_allclose(np.linalg.norm(vectorizer.transform(texts).to_dense(), axis=1), 1.0)


def test_sparse_operations_match_dense_algebra():
    x = TfidfVectorizer(min_count=1).fit(["a b c", "b c d", "a d"]).transform(["a b", "c", "zzz", "a b c d"])
    dense = x.to_dense()
    rng = np.random.default_rng(0)
    weights = rng.normal(size=(x.n_cols, 3))
    by_row = rng.normal(size=(x.n_rows, 3))
    np.testing.assert_allclose(x.dot(weights), dense @ weights)
    np.testing.assert_allclose(x.transpose_dot(by_row), dense.T @ by_row)
    rows = np.array([3, 0, 2])
    np.testing.assert_allclose(x.select(rows).to_dense(), dense[rows])


def test_softmax_is_stable_for_huge_inputs():
    out = stable_softmax(np.array([[1000.0, 1000.0, -1000.0]]))
    np.testing.assert_allclose(out, [[0.5, 0.5, 0.0]])


def test_softmax_regression_gradient_matches_central_differences():
    x = TfidfVectorizer(min_count=1).fit(["a b c", "b c d", "a d e"]).transform(["a b c", "b c d", "a d e", "e a"])
    y = np.array([0, 1, 2, 1])
    model = SoftmaxRegression(x.n_cols)
    rng = np.random.default_rng(1)
    model.W[...] = rng.normal(size=model.W.shape)
    model.b[...] = rng.normal(size=model.b.shape)
    model.loss_and_grad(x, y)
    step = 1e-6
    for value, grad in model.params():
        numeric = np.zeros_like(value)
        for index in np.ndindex(value.shape):
            original = value[index]
            value[index] = original + step
            above = model.loss(x, y)
            value[index] = original - step
            below = model.loss(x, y)
            value[index] = original
            numeric[index] = (above - below) / (2 * step)
        error = np.linalg.norm(numeric - grad) / (np.linalg.norm(numeric) + np.linalg.norm(grad))
        assert error < 1e-6


def test_adamw_decays_matrices_but_not_vectors():
    matrix, vector = np.ones((2, 2)), np.ones(2)
    optimiser = AdamW([(matrix, np.zeros((2, 2))), (vector, np.zeros(2))], lr=0.1, weight_decay=0.5)
    optimiser.step()
    np.testing.assert_allclose(matrix, 0.95)
    np.testing.assert_allclose(vector, 1.0)


def test_adamw_first_step_size_is_the_learning_rate():
    value, grad = np.array([1.0]), np.array([123.0])
    AdamW([(value, grad)], lr=0.01).step()
    np.testing.assert_allclose(value, 0.99)


def test_training_learns_a_separable_problem_and_restores_the_best_weights():
    texts = ["profit rises strongly"] * 30 + ["loss falls badly"] * 30 + ["meeting held today"] * 30
    y = np.repeat([2, 0, 1], 30)
    vectorizer = TfidfVectorizer(min_count=1).fit(texts)
    x = vectorizer.transform(texts)
    model = SoftmaxRegression(x.n_cols)
    result = train(model, x, y, x, y, seed=0, max_epochs=30)
    assert (model.predict_proba(x).argmax(axis=1) == y).all()
    assert model.loss(x, y) == pytest.approx(result.best_val_loss)


def test_stratified_split_keeps_class_shares_and_is_disjoint():
    labels = np.array([0] * 100 + [1] * 300 + [2] * 50)
    first, second = stratified_split(labels, 0.2, seed=3)
    assert np.bincount(labels[second]).tolist() == [20, 60, 10]
    assert not set(first) & set(second)
    assert first.size + second.size == labels.size
    np.testing.assert_array_equal(second, stratified_split(labels, 0.2, seed=3)[1])  # reproducible


def test_duplicates_are_removed_before_splitting():
    sentences, labels, removed = deduplicate(["a", "b", "a"], ["neutral", "positive", "neutral"])
    assert sentences == ["a", "b"] and labels == ["neutral", "positive"] and removed == 1


def test_classification_report_on_a_known_case():
    report = classification_report(np.array([0, 0, 1, 2]), np.array([0, 1, 1, 1]))
    assert report["accuracy"] == 0.5
    # F1 by class: negative 2/3, neutral 1/2, positive 0.
    assert report["macro_f1"] == pytest.approx((2 / 3 + 1 / 2 + 0) / 3)


def test_chunked_scoring_matches_scoring_everything_at_once():
    texts = np.array(["profit rises", "loss falls", "meeting held", "profit falls", "zzz"], dtype=object)
    vectorizer = TfidfVectorizer(min_count=1).fit(texts)
    model = SoftmaxRegression(len(vectorizer.terms))
    model.W[...] = np.random.default_rng(2).normal(size=model.W.shape)
    chunked = score_in_chunks(model, vectorizer, texts, chunk_size=2)
    whole = score_in_chunks(model, vectorizer, texts, chunk_size=100)
    np.testing.assert_allclose(chunked[0], whole[0])
    np.testing.assert_array_equal(chunked[1], whole[1])


def test_daily_signal_rolls_weekends_forward_and_ignores_later_headlines():
    trading_days = days("2021-01-08", "2021-01-11")  # Friday, Monday
    headline_dates = days("2021-01-08", "2021-01-09", "2021-01-10", "2021-01-11", "2021-01-12")
    scores = np.array([0.5, -1.0, 0.0, 1.0, 9.0])
    negative = np.array([False, True, False, False, True])
    daily = aggregate_daily(headline_dates, scores, negative, trading_days)
    np.testing.assert_array_equal(daily.dates, trading_days)
    np.testing.assert_allclose(daily.mean_sentiment, [0.5, 0.0])  # Monday: mean of -1, 0, 1
    np.testing.assert_allclose(daily.negative_share, [0.0, 1 / 3])
    np.testing.assert_allclose(daily.log_count, np.log1p([1, 3]))


def test_a_day_never_receives_a_headline_dated_after_it():
    """Changing later headlines must not change any earlier day's signal."""
    trading_days = days("2021-01-08", "2021-01-11", "2021-01-12")
    headline_dates = days("2021-01-08", "2021-01-09", "2021-01-11", "2021-01-12")
    base = np.array([0.1, 0.2, 0.3, 0.4])
    changed = np.array([0.1, 0.2, 0.3, 99.0])
    negative = np.zeros(4, dtype=bool)
    first = aggregate_daily(headline_dates, base, negative, trading_days)
    second = aggregate_daily(headline_dates, changed, negative, trading_days)
    np.testing.assert_array_equal(first.mean_sentiment[:2], second.mean_sentiment[:2])


def test_days_without_news_are_left_out_of_the_daily_signal():
    daily = aggregate_daily(days("2021-01-11"), np.array([0.5]), np.array([False]), days("2021-01-08", "2021-01-11"))
    assert daily.dates.astype(str).tolist() == ["2021-01-11"]


def test_extremes_are_ranked_distinct_and_on_the_rolled_day():
    trading_days = days("2021-01-08", "2021-01-11")
    dates = days("2021-01-08", "2021-01-09", "2021-01-09", "2021-01-10", "2021-01-11")
    scores = np.array([0.5, -0.9, -0.9, 0.8, 0.1])
    texts = np.array(["friday", "bad", "bad", "good", "plain"], dtype=object)
    out = daily_extremes(dates, scores, texts, trading_days, k=2)
    monday_negative = (out["date"] == trading_days[1]) & (out["side"] == "negative")
    assert out["headline"][monday_negative].tolist() == ["bad", "plain"]  # the duplicate appears once
    assert out["rank"][monday_negative].tolist() == [1, 2]
    monday_positive = (out["date"] == trading_days[1]) & (out["side"] == "positive")
    assert out["headline"][monday_positive].tolist() == ["good", "plain"]
    assert out["headline"][out["date"] == trading_days[0]].tolist() == ["friday", "friday"]
