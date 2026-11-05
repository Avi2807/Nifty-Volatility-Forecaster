"""Tokeniser and TF-IDF, written by hand, with a small sparse-row format."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

MIN_TERM_COUNT = 3
NUMBER_TOKEN = "NUM"

_APOSTROPHES = re.compile(r"['’`]")
# A whole number or a run of letters. The headline archive writes 4,940 as 4;940.
_TOKENS = re.compile(r"\d+(?:[.,;]\d+)*|[^\W\d_]+")


def tokenize(text: str) -> list[str]:
    """Lowercase, digits to NUM, punctuation removed, then unigrams plus bigrams.

    NUM is added after lowercasing, so it cannot collide with the English word "num".
    """
    words = [
        NUMBER_TOKEN if token[0].isdigit() else token
        for token in _TOKENS.findall(_APOSTROPHES.sub("", text.lower()))
    ]
    return words + [f"{first} {second}" for first, second in zip(words, words[1:])]


@dataclass(frozen=True)
class SparseRows:
    """A compressed sparse row matrix, so long texts never need a dense vocabulary array."""

    indptr: np.ndarray  # row i holds entries indptr[i]..indptr[i+1]
    indices: np.ndarray
    data: np.ndarray
    n_cols: int

    @property
    def n_rows(self) -> int:
        return self.indptr.size - 1

    def _row_numbers(self) -> np.ndarray:
        return np.repeat(np.arange(self.n_rows), np.diff(self.indptr))

    def select(self, rows: np.ndarray) -> "SparseRows":
        lengths = np.diff(self.indptr)[rows]
        indptr = np.concatenate([[0], np.cumsum(lengths)])
        picks = np.concatenate([np.arange(self.indptr[r], self.indptr[r + 1]) for r in rows]) if rows.size else np.empty(0, dtype=np.int64)
        return SparseRows(indptr, self.indices[picks.astype(np.int64)], self.data[picks.astype(np.int64)], self.n_cols)

    def dot(self, weights: np.ndarray) -> np.ndarray:
        """self @ weights, for a dense weights matrix of shape (n_cols, k)."""
        out = np.zeros((self.n_rows, weights.shape[1]))
        np.add.at(out, self._row_numbers(), self.data[:, None] * weights[self.indices])
        return out

    def transpose_dot(self, rows_by_k: np.ndarray) -> np.ndarray:
        """self.T @ rows_by_k, for a dense matrix of shape (n_rows, k)."""
        out = np.zeros((self.n_cols, rows_by_k.shape[1]))
        np.add.at(out, self.indices, self.data[:, None] * rows_by_k[self._row_numbers()])
        return out

    def to_dense(self) -> np.ndarray:
        out = np.zeros((self.n_rows, self.n_cols))
        np.add.at(out, (self._row_numbers(), self.indices), self.data)
        return out


class TfidfVectorizer:
    """TF-IDF with L2-normalised rows.

    idf(w) = log((1 + N) / (1 + df(w))) + 1, where N and df are from the training texts only.
    Terms outside the training vocabulary are ignored at transform time.
    """

    def __init__(self, min_count: int = MIN_TERM_COUNT) -> None:
        self.min_count = min_count
        self.term_index: dict[str, int] = {}
        self.terms: list[str] = []
        self.idf = np.empty(0)

    def fit(self, texts: Iterable[str]) -> "TfidfVectorizer":
        total_counts: Counter[str] = Counter()
        document_counts: Counter[str] = Counter()
        n_documents = 0
        for text in texts:
            tokens = tokenize(text)
            total_counts.update(tokens)
            document_counts.update(set(tokens))
            n_documents += 1
        self.terms = sorted(term for term, count in total_counts.items() if count >= self.min_count)
        self.term_index = {term: index for index, term in enumerate(self.terms)}
        df = np.array([document_counts[term] for term in self.terms], dtype=np.float64)
        self.idf = np.log((1.0 + n_documents) / (1.0 + df)) + 1.0
        return self

    def transform(self, texts: Sequence[str]) -> SparseRows:
        indptr = [0]
        indices: list[int] = []
        data: list[float] = []
        for text in texts:
            counts = Counter(self.term_index[t] for t in tokenize(text) if t in self.term_index)
            weights = np.array([tf * self.idf[i] for i, tf in counts.items()], dtype=np.float64)
            if weights.size:
                weights /= np.sqrt(np.sum(weights**2))
            indices.extend(counts)
            data.extend(weights.tolist())
            indptr.append(len(indices))
        return SparseRows(
            np.array(indptr, dtype=np.int64), np.array(indices, dtype=np.int64), np.array(data, dtype=np.float64), len(self.terms)
        )
