"""
Sentiment & embedding analysis module (Owner: Henrique).

Produces two complementary views:

1. Dense embeddings from ``sentence-transformers`` clustered with KMeans,
   visualized in 2D via UMAP.
2. Lexicon-based VADER compound scores aggregated to the video level.

The merged per-video sentiment table is written to
``data/processed/sentiment_dataset.csv`` and figures to ``outputs/figures/``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score

from config import (
    FIGURES_DIR,
    KMEANS_K_DEFAULT,
    KMEANS_K_RANGE,
    PROCESSED_DIR,
    RANDOM_SEED,
    SENTENCE_MODEL,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# NLTK / VADER bootstrap
# ---------------------------------------------------------------------------
def _ensure_vader():
    """Download the VADER lexicon on first use; return a SentimentIntensityAnalyzer."""
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer

    try:
        SentimentIntensityAnalyzer()
    except LookupError:
        nltk.download("vader_lexicon", quiet=True)
    return SentimentIntensityAnalyzer()


# ---------------------------------------------------------------------------
# Embeddings + clustering
# ---------------------------------------------------------------------------
def embed_comments(texts: list[str], model_name: str = SENTENCE_MODEL) -> np.ndarray:
    """Encode ``texts`` with a sentence-transformers model and return an (N, D) array."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    return np.asarray(
        model.encode(texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True)
    )


def tune_kmeans(
    embeddings: np.ndarray,
    k_range: tuple[int, int] = KMEANS_K_RANGE,
    sample_size: int = 5000,
) -> tuple[int, dict[int, float]]:
    """
    Pick the ``k`` with the highest silhouette score on a random sub-sample.

    Returns (best_k, {k: score}).
    """
    rng = np.random.default_rng(RANDOM_SEED)
    if len(embeddings) > sample_size:
        idx = rng.choice(len(embeddings), size=sample_size, replace=False)
        sample = embeddings[idx]
    else:
        sample = embeddings

    scores: dict[int, float] = {}
    lo, hi = k_range
    for k in range(lo, hi + 1):
        km = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
        labels = km.fit_predict(sample)
        try:
            scores[k] = float(silhouette_score(sample, labels))
        except ValueError:
            scores[k] = float("nan")
    best_k = max(scores, key=lambda k: scores[k])
    return best_k, scores


def label_clusters_by_keywords(
    texts: list[str], labels: np.ndarray, top_k: int = 5
) -> dict[int, list[str]]:
    """Derive per-cluster keyword labels via TF-IDF over each cluster's documents."""
    vec = TfidfVectorizer(stop_words="english", max_df=0.9, min_df=2)
    matrix = vec.fit_transform(texts)
    vocab = np.asarray(vec.get_feature_names_out())
    keywords: dict[int, list[str]] = {}
    for cid in np.unique(labels):
        rows = matrix[labels == cid]
        if rows.shape[0] == 0:
            keywords[int(cid)] = []
            continue
        means = np.asarray(rows.mean(axis=0)).ravel()
        top = np.argsort(-means)[:top_k]
        keywords[int(cid)] = vocab[top].tolist()
    return keywords


def _plot_umap_clusters(
    embeddings: np.ndarray, labels: np.ndarray, narration_type: pd.Series, out_path: Path
) -> None:
    """2D UMAP projection colored by cluster with markers split by narration_type."""
    import umap

    reducer = umap.UMAP(n_components=2, random_state=RANDOM_SEED, n_neighbors=15, min_dist=0.1)
    coords = reducer.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(9, 7))
    plot_df = pd.DataFrame(
        {"x": coords[:, 0], "y": coords[:, 1], "cluster": labels, "narration": narration_type.values}
    )
    sns.scatterplot(
        data=plot_df,
        x="x", y="y",
        hue="cluster",
        style="narration",
        palette="tab10",
        s=10, alpha=0.6, ax=ax,
    )
    ax.set_title("UMAP projection of comment embeddings")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.legend(bbox_to_anchor=(1.02, 1.0), loc="upper left", borderaxespad=0.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# VADER
# ---------------------------------------------------------------------------
def score_vader(texts: list[str]) -> np.ndarray:
    """Return the VADER ``compound`` score for each text in ``texts``."""
    sia = _ensure_vader()
    return np.array([sia.polarity_scores(t or "")["compound"] for t in texts], dtype=float)


def aggregate_per_video(df: pd.DataFrame) -> pd.DataFrame:
    """Reduce comment-level scores to per-video summary statistics."""
    grouped = df.groupby("video_id").agg(
        mean_sentiment=("vader_compound", "mean"),
        std_sentiment=("vader_compound", "std"),
        pct_negative=("vader_compound", lambda s: float((s < -0.05).mean())),
        pct_positive=("vader_compound", lambda s: float((s > 0.05).mean())),
        authenticity_flag_rate=("authenticity_flag", "mean"),
        n_comments=("vader_compound", "size"),
    )
    return grouped.reset_index()


def _plot_sentiment_distribution(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.kdeplot(
        data=df, x="vader_compound", hue="narration_type",
        common_norm=False, fill=True, alpha=0.35, ax=ax,
    )
    ax.set_title("VADER compound sentiment by narration type")
    ax.set_xlabel("VADER compound")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_per_video_sentiment(video_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=video_df, x="narration_type", y="mean_sentiment", ax=ax)
    sns.stripplot(
        data=video_df, x="narration_type", y="mean_sentiment",
        color="black", alpha=0.4, size=3, ax=ax,
    )
    ax.set_title("Per-video mean sentiment by narration type")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def run(
    cleaned_path: Path | str | None = None,
    output_path: Path | str | None = None,
    *,
    tune_k: bool = True,
) -> pd.DataFrame:
    """
    Run embedding clustering + VADER scoring and persist artefacts.

    Returns the per-video sentiment DataFrame (also written to
    ``sentiment_dataset.csv``). The comment-level frame with VADER scores and
    cluster IDs is written to ``comments_scored.csv``.
    """
    np.random.seed(RANDOM_SEED)
    cleaned_path = Path(cleaned_path or PROCESSED_DIR / "cleaned_dataset.csv")
    output_path = Path(output_path or PROCESSED_DIR / "sentiment_dataset.csv")
    if not cleaned_path.exists():
        raise FileNotFoundError(f"{cleaned_path} not found. Run --preprocess first.")

    df = pd.read_csv(cleaned_path)
    df["text_clean"] = df["text_clean"].fillna("").astype(str)
    logger.info("Sentiment input shape=%s", df.shape)
    print("[sentiment] input shape:", df.shape)

    # ------------------------------------------------------------------
    # Embeddings + KMeans
    # ------------------------------------------------------------------
    texts = df["text_clean"].tolist()
    embeddings = embed_comments(texts)
    logger.info("Embeddings shape=%s", embeddings.shape)

    if tune_k:
        best_k, scores = tune_kmeans(embeddings)
        logger.info("KMeans silhouette scores: %s -> best k=%d", scores, best_k)
    else:
        best_k = KMEANS_K_DEFAULT

    km = KMeans(n_clusters=best_k, random_state=RANDOM_SEED, n_init=10)
    cluster_labels = km.fit_predict(embeddings)
    df["cluster"] = cluster_labels

    cluster_keywords = label_clusters_by_keywords(texts, cluster_labels)
    print("[sentiment] cluster keywords:")
    for cid, kws in cluster_keywords.items():
        print(f"  cluster {cid}: {', '.join(kws)}")

    _plot_umap_clusters(
        embeddings, cluster_labels, df["narration_type"],
        FIGURES_DIR / "umap_clusters.png",
    )

    # ------------------------------------------------------------------
    # VADER
    # ------------------------------------------------------------------
    df["vader_compound"] = score_vader(texts)
    _plot_sentiment_distribution(df, FIGURES_DIR / "sentiment_distribution.png")

    # ------------------------------------------------------------------
    # Per-video aggregation + merge
    # ------------------------------------------------------------------
    per_video = aggregate_per_video(df)

    video_meta_cols = [
        "video_id", "channel_id", "narration_type", "genre", "title",
        "viewCount", "likeCount", "commentCount",
        "like_to_view_ratio", "comment_rate",
    ]
    video_meta = (
        df[video_meta_cols]
        .drop_duplicates(subset=["video_id"])
        .reset_index(drop=True)
    )
    per_video = per_video.merge(video_meta, on="video_id", how="left")

    per_video.to_csv(output_path, index=False, encoding="utf-8")
    df.to_csv(PROCESSED_DIR / "comments_scored.csv", index=False, encoding="utf-8")
    _plot_per_video_sentiment(per_video, FIGURES_DIR / "per_video_sentiment.png")

    logger.info("Wrote per-video sentiment: %s (shape=%s)", output_path, per_video.shape)
    print("[sentiment] per-video shape:", per_video.shape)
    print(per_video.head(3).to_string())
    return per_video


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run()
