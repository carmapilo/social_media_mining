"""
Linguistic analysis module (Owner: Carlos).

Computes TF-IDF top terms, top bigrams/trigrams, and a keyword co-occurrence
matrix for the cleaned comments in ``data/processed/cleaned_dataset.csv``,
split by ``narration_type``. Figures are written to ``outputs/figures/``.
"""

from __future__ import annotations

import logging
from collections import Counter
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from config import AUTHENTICITY_KEYWORDS, FIGURES_DIR, PROCESSED_DIR, RANDOM_SEED

logger = logging.getLogger(__name__)

TOP_TFIDF_TERMS = 50
TOP_NGRAMS = 20


# ---------------------------------------------------------------------------
# TF-IDF
# ---------------------------------------------------------------------------
def top_tfidf_terms(corpus: list[str], top_n: int = TOP_TFIDF_TERMS) -> pd.DataFrame:
    """Return the ``top_n`` terms ranked by summed TF-IDF weight across ``corpus``."""
    if not corpus:
        return pd.DataFrame(columns=["term", "score"])
    vec = TfidfVectorizer(
        stop_words="english",
        max_df=0.9,
        min_df=3,
        ngram_range=(1, 1),
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z']+\b",
    )
    matrix = vec.fit_transform(corpus)
    scores = np.asarray(matrix.sum(axis=0)).ravel()
    terms = vec.get_feature_names_out()
    order = np.argsort(-scores)[:top_n]
    return pd.DataFrame({"term": terms[order], "score": scores[order]})


def _plot_top_terms(df: pd.DataFrame, title: str, out_path: Path) -> None:
    """Horizontal bar chart of top terms; saved (not shown) for headless runs."""
    fig, ax = plt.subplots(figsize=(8, max(6, 0.25 * len(df))))
    sns.barplot(data=df, x="score", y="term", ax=ax, color="steelblue")
    ax.set_title(title)
    ax.set_xlabel("Summed TF-IDF")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# N-grams
# ---------------------------------------------------------------------------
def top_ngrams(corpus: list[str], ngram_range: tuple[int, int], top_n: int = TOP_NGRAMS) -> pd.DataFrame:
    """Return the ``top_n`` raw-count n-grams for ``corpus`` in ``ngram_range``."""
    if not corpus:
        return pd.DataFrame(columns=["ngram", "count"])
    vec = CountVectorizer(
        stop_words="english",
        ngram_range=ngram_range,
        min_df=3,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z']+\b",
    )
    matrix = vec.fit_transform(corpus)
    counts = np.asarray(matrix.sum(axis=0)).ravel()
    grams = vec.get_feature_names_out()
    order = np.argsort(-counts)[:top_n]
    return pd.DataFrame({"ngram": grams[order], "count": counts[order]})


def _plot_ngrams(df: pd.DataFrame, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, max(5, 0.3 * len(df))))
    sns.barplot(data=df, x="count", y="ngram", ax=ax, color="darkorange")
    ax.set_title(title)
    ax.set_xlabel("Count")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Keyword co-occurrence (authenticity comments only)
# ---------------------------------------------------------------------------
def keyword_cooccurrence(
    texts: list[str], keywords: list[str]
) -> pd.DataFrame:
    """
    Build a symmetric keyword co-occurrence matrix over ``texts``.

    A keyword is "present" if it occurs as a whole-word substring in a document.
    The resulting DataFrame is indexed and columned by ``keywords``.
    """
    co = Counter()
    per_kw = Counter()
    for doc in texts:
        present = [k for k in keywords if k in doc]
        per_kw.update(present)
        for a, b in combinations(sorted(set(present)), 2):
            co[(a, b)] += 1

    matrix = pd.DataFrame(0, index=keywords, columns=keywords, dtype=int)
    for k, v in per_kw.items():
        matrix.loc[k, k] = v
    for (a, b), v in co.items():
        matrix.loc[a, b] = v
        matrix.loc[b, a] = v
    return matrix


def _plot_cooccurrence(matrix: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="viridis", ax=ax, cbar_kws={"label": "count"})
    ax.set_title("Authenticity keyword co-occurrence")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def run(cleaned_path: Path | str | None = None) -> dict[str, pd.DataFrame]:
    """
    Execute the full linguistic analysis and persist figures.

    Returns a dict with the computed tables for downstream inspection.
    """
    np.random.seed(RANDOM_SEED)
    cleaned_path = Path(cleaned_path or PROCESSED_DIR / "cleaned_dataset.csv")
    if not cleaned_path.exists():
        raise FileNotFoundError(f"{cleaned_path} not found. Run --preprocess first.")

    df = pd.read_csv(cleaned_path)
    df["text_clean"] = df["text_clean"].fillna("").astype(str)
    logger.info("Linguistic analysis on shape=%s", df.shape)
    print("[linguistic] input shape:", df.shape)

    results: dict[str, pd.DataFrame] = {}

    for group in ("human", "ai"):
        corpus = df.loc[df["narration_type"] == group, "text_clean"].tolist()
        logger.info("Group=%s  n_comments=%d", group, len(corpus))
        if not corpus:
            continue

        tfidf_df = top_tfidf_terms(corpus)
        _plot_top_terms(
            tfidf_df,
            f"Top {TOP_TFIDF_TERMS} TF-IDF terms ({group})",
            FIGURES_DIR / f"tfidf_top_{group}.png",
        )
        results[f"tfidf_{group}"] = tfidf_df
        print(f"[linguistic] top TF-IDF ({group}):")
        print(tfidf_df.head(15).to_string(index=False))

        bigrams = top_ngrams(corpus, (2, 2))
        trigrams = top_ngrams(corpus, (3, 3))
        _plot_ngrams(bigrams, f"Top bigrams ({group})", FIGURES_DIR / f"bigrams_{group}.png")
        _plot_ngrams(trigrams, f"Top trigrams ({group})", FIGURES_DIR / f"trigrams_{group}.png")
        results[f"bigrams_{group}"] = bigrams
        results[f"trigrams_{group}"] = trigrams

    # ------------------------------------------------------------------
    # Co-occurrence over authenticity-flagged comments
    # ------------------------------------------------------------------
    auth_df = df[df["authenticity_flag"] == True]  # noqa: E712
    if not auth_df.empty:
        cooc = keyword_cooccurrence(auth_df["text_clean"].tolist(), AUTHENTICITY_KEYWORDS)
        cooc.to_csv(FIGURES_DIR.parent / "authenticity_cooccurrence.csv")
        _plot_cooccurrence(cooc, FIGURES_DIR / "authenticity_cooccurrence.png")
        results["cooccurrence"] = cooc
        print("[linguistic] authenticity co-occurrence (diag = marginal counts):")
        print(cooc.to_string())
    else:
        logger.info("No authenticity-flagged comments; skipping co-occurrence.")

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run()
