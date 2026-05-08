"""
Statistical analysis module (Owner: Natacha).

Compares the human- vs. AI-narrated groups using Mann-Whitney U tests on
engagement and sentiment metrics, reports rank-biserial effect sizes, and
computes within-group Spearman correlations between sentiment and engagement.

Reads ``data/processed/sentiment_dataset.csv`` and writes a summary table to
``outputs/statistical_summary.csv``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from config import OUTPUTS_DIR, PROCESSED_DIR

logger = logging.getLogger(__name__)


METRICS = [
    "like_to_view_ratio",
    "comment_rate",
    "mean_sentiment",
    "pct_negative",
    "authenticity_flag_rate",
]


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------
def mann_whitney(
    human: pd.Series, ai: pd.Series
) -> dict[str, float]:
    """
    Two-sided Mann-Whitney U test with a rank-biserial effect size.

    Effect size ``r`` = 1 - 2U / (n1 * n2); positive means the ``human`` group
    tends to rank higher than the ``ai`` group.
    """
    human = human.dropna().astype(float)
    ai = ai.dropna().astype(float)
    n1, n2 = len(human), len(ai)
    if n1 == 0 or n2 == 0:
        return {"U": np.nan, "p_value": np.nan, "effect_size_r": np.nan, "n_human": n1, "n_ai": n2}

    res = stats.mannwhitneyu(human, ai, alternative="two-sided")
    r = 1.0 - (2.0 * res.statistic) / (n1 * n2)
    return {
        "U": float(res.statistic),
        "p_value": float(res.pvalue),
        "effect_size_r": float(r),
        "n_human": n1,
        "n_ai": n2,
        "median_human": float(human.median()),
        "median_ai": float(ai.median()),
    }


def spearman(x: pd.Series, y: pd.Series) -> dict[str, float]:
    """Spearman rank correlation with sample-size reporting; NaN-safe."""
    mask = x.notna() & y.notna()
    if mask.sum() < 3:
        return {"rho": np.nan, "p_value": np.nan, "n": int(mask.sum())}
    res = stats.spearmanr(x[mask], y[mask])
    return {"rho": float(res.statistic), "p_value": float(res.pvalue), "n": int(mask.sum())}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def run(
    sentiment_path: Path | str | None = None,
    output_path: Path | str | None = None,
) -> pd.DataFrame:
    """
    Produce the full significance table and return it as a DataFrame.
    """
    sentiment_path = Path(sentiment_path or PROCESSED_DIR / "sentiment_dataset.csv")
    output_path = Path(output_path or OUTPUTS_DIR / "statistical_summary.csv")
    if not sentiment_path.exists():
        raise FileNotFoundError(
            f"{sentiment_path} not found. Run --sentiment first."
        )

    df = pd.read_csv(sentiment_path)
    logger.info("Statistical input shape=%s", df.shape)
    print("[statistics] input shape:", df.shape)

    human = df[df["narration_type"] == "human"]
    ai = df[df["narration_type"] == "ai"]

    rows: list[dict] = []
    for metric in METRICS:
        if metric not in df.columns:
            logger.warning("Metric %s missing; skipping.", metric)
            continue
        row = {"metric": metric, **mann_whitney(human[metric], ai[metric])}
        row["significant_0.05"] = bool(row["p_value"] < 0.05) if not np.isnan(row["p_value"]) else False
        row["significant_bonferroni"] = bool(row["p_value"] < (0.05 / len(METRICS))) if not np.isnan(row["p_value"]) else False
        rows.append(row)

    stats_df = pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Spearman within each group: mean_sentiment vs. like_to_view_ratio
    # ------------------------------------------------------------------
    corr_rows = []
    for grp_name, grp in (("human", human), ("ai", ai)):
        r = spearman(grp["mean_sentiment"], grp["like_to_view_ratio"])
        corr_rows.append({"group": grp_name, "pair": "mean_sentiment~like_to_view_ratio", **r})
    corr_df = pd.DataFrame(corr_rows)
    if "authenticity_flag_rate" in df.columns:
        mean_flag_rate = df["authenticity_flag_rate"].mean()
        if mean_flag_rate > 0.30:
            logger.warning(
                "Mean authenticity_flag_rate=%.3f is unusually high; verify keyword matching in preprocessing.",
                mean_flag_rate,
            )

    # ------------------------------------------------------------------
    # Persist + pretty-print
    # ------------------------------------------------------------------
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats_df.to_csv(output_path, index=False)
    corr_df.to_csv(OUTPUTS_DIR / "spearman_summary.csv", index=False)

    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    print("\n=== Mann-Whitney U: human vs. AI ===")
    print(stats_df.to_string(index=False))
    print("\n=== Spearman: mean_sentiment ~ like_to_view_ratio ===")
    print(corr_df.to_string(index=False))
    pd.reset_option("display.float_format")

    return stats_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run()
