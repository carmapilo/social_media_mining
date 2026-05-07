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
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split

from config import (
    FIGURES_DIR,
    KMEANS_K_DEFAULT,
    KMEANS_K_RANGE,
    OUTPUTS_DIR,
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
        except (ValueError, MemoryError, np.core._exceptions._ArrayMemoryError):
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


def _plot_sentiment_classifier_metrics(metrics_df: pd.DataFrame, out_path: Path) -> None:
    """Bar chart of F1 and ROC-AUC for supervised TF-IDF sentiment classifiers."""
    df = metrics_df.copy()
    df["label"] = df["model"].str.replace("_", " ").str.title()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    df_f1 = df.sort_values("f1", ascending=True)
    sns.barplot(data=df_f1, y="label", x="f1", ax=axes[0], color="#4C72B0")
    axes[0].set_title("F1 (positive vs negative, weak VADER labels)")
    axes[0].set_xlabel("F1")
    axes[0].set_xlim(0, 1.02)
    axes[0].set_ylabel("")
    df_auc = df.sort_values("roc_auc", ascending=True)
    sns.barplot(data=df_auc, y="label", x="roc_auc", ax=axes[1], color="#55A868")
    axes[1].set_title("ROC-AUC")
    axes[1].set_xlabel("ROC-AUC")
    axes[1].set_xlim(0, 1.02)
    axes[1].set_ylabel("")
    fig.suptitle("Supervised sentiment models (TF-IDF)", y=1.02, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Supervised sentiment models (feature importance)
# ---------------------------------------------------------------------------
def _build_weak_sentiment_labels(vader_scores: pd.Series) -> pd.Series:
    """
    Convert VADER compound scores into weak binary labels.

    Positive: compound > 0.05
    Negative: compound < -0.05
    Neutral comments are dropped for supervised training.
    """
    labels = pd.Series(index=vader_scores.index, dtype="object")
    labels.loc[vader_scores > 0.05] = "positive"
    labels.loc[vader_scores < -0.05] = "negative"
    return labels


def _extract_signed_terms(
    model_name: str,
    feature_names: np.ndarray,
    importances: np.ndarray,
    top_k: int = 20,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    order_desc = np.argsort(-importances)[:top_k]
    for rank, idx in enumerate(order_desc, start=1):
        rows.append(
            {
                "model": model_name,
                "term": str(feature_names[idx]),
                "score": float(importances[idx]),
                "direction": "positive",
                "rank": rank,
            }
        )
    order_asc = np.argsort(importances)[:top_k]
    for rank, idx in enumerate(order_asc, start=1):
        rows.append(
            {
                "model": model_name,
                "term": str(feature_names[idx]),
                "score": float(importances[idx]),
                "direction": "negative",
                "rank": rank,
            }
        )
    return rows


def _extract_unsigned_terms(
    model_name: str,
    feature_names: np.ndarray,
    importances: np.ndarray,
    top_k: int = 20,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    order_desc = np.argsort(-importances)[:top_k]
    for rank, idx in enumerate(order_desc, start=1):
        rows.append(
            {
                "model": model_name,
                "term": str(feature_names[idx]),
                "score": float(importances[idx]),
                "direction": "global",
                "rank": rank,
            }
        )
    return rows


def train_sentiment_models(
    text_series: pd.Series,
    vader_scores: pd.Series,
    *,
    top_k_terms: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Train TF-IDF-based sentiment models and return:
      1) model-level metrics
      2) top important terms per model
    """
    weak_labels = _build_weak_sentiment_labels(vader_scores)
    train_df = pd.DataFrame({"text": text_series, "label": weak_labels}).dropna()
    if train_df.shape[0] < 200:
        raise ValueError(
            "Not enough non-neutral comments for supervised sentiment training. "
            f"Need >=200, got {train_df.shape[0]}."
        )
    if train_df["label"].nunique() < 2:
        raise ValueError("Weak labels produced a single class; cannot train classifiers.")

    y = (train_df["label"] == "positive").astype(int).to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(
        train_df["text"],
        y,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        max_features=20000,
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    feature_names = vectorizer.get_feature_names_out()

    models: dict[str, object] = {
        "logistic_regression": LogisticRegression(
            max_iter=1500,
            random_state=RANDOM_SEED,
            class_weight="balanced",
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=400,
            random_state=RANDOM_SEED,
            n_jobs=-1,
            class_weight="balanced_subsample",
        ),
    }

    try:
        from xgboost import XGBClassifier

        models["xgboost"] = XGBClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )
    except ImportError:
        logger.warning("xgboost is not installed; skipping XGBoost model.")

    metrics_rows: list[dict[str, float | str]] = []
    terms_rows: list[dict[str, object]] = []

    for model_name, model in models.items():
        model.fit(X_train_tfidf, y_train)
        y_pred = model.predict(X_test_tfidf)
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test_tfidf)[:, 1]
            roc_auc = float(roc_auc_score(y_test, y_prob))
        else:
            roc_auc = float("nan")

        metrics_rows.append(
            {
                "model": model_name,
                "n_train": int(X_train_tfidf.shape[0]),
                "n_test": int(X_test_tfidf.shape[0]),
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "precision": float(precision_score(y_test, y_pred, zero_division=0)),
                "recall": float(recall_score(y_test, y_pred, zero_division=0)),
                "f1": float(f1_score(y_test, y_pred, zero_division=0)),
                "roc_auc": roc_auc,
            }
        )

        if model_name == "logistic_regression":
            importances = model.coef_[0]
            terms_rows.extend(
                _extract_signed_terms(model_name, feature_names, importances, top_k=top_k_terms)
            )
        else:
            importances = np.asarray(model.feature_importances_)
            terms_rows.extend(
                _extract_unsigned_terms(model_name, feature_names, importances, top_k=top_k_terms)
            )

    metrics_df = pd.DataFrame(metrics_rows).sort_values("f1", ascending=False).reset_index(drop=True)
    terms_df = pd.DataFrame(terms_rows)
    return metrics_df, terms_df


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
        try:
            best_k, scores = tune_kmeans(embeddings)
            logger.info("KMeans silhouette scores: %s -> best k=%d", scores, best_k)
        except (MemoryError, np.core._exceptions._ArrayMemoryError):
            best_k = KMEANS_K_DEFAULT
            logger.warning(
                "KMeans tuning ran out of memory; using default k=%d.",
                best_k,
            )
            print(f"[sentiment] k tuning OOM, fallback to default k={best_k}")
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
    try:
        model_metrics, top_terms = train_sentiment_models(df["text_clean"], df["vader_compound"])
        model_metrics.to_csv(OUTPUTS_DIR / "sentiment_model_comparison.csv", index=False, encoding="utf-8")
        top_terms.to_csv(OUTPUTS_DIR / "sentiment_top_terms.csv", index=False, encoding="utf-8")
        _plot_sentiment_classifier_metrics(
            model_metrics, FIGURES_DIR / "sentiment_classifier_metrics.png"
        )
        print("[sentiment] model comparison:")
        print(model_metrics.to_string(index=False))
    except ValueError as exc:
        logger.warning("Skipping supervised sentiment model training: %s", exc)
        print(f"[sentiment] skipped supervised model training: {exc}")

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
