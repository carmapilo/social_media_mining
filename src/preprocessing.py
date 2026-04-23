"""
Preprocessing module.

Consumes ``data/processed/master_dataset.csv`` (produced by ``acquisition.py``),
cleans comment text, engineers per-video engagement ratios, filters short
comments, and flags authenticity-related comments. Writes the result to
``data/processed/cleaned_dataset.csv``.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from pathlib import Path

import emoji
import numpy as np
import pandas as pd

from config import AUTHENTICITY_KEYWORDS, MIN_COMMENT_WORDS, PROCESSED_DIR

logger = logging.getLogger(__name__)


URL_RE = re.compile(r"https?://\S+|www\.\S+")
HTML_TAG_RE = re.compile(r"<[^>]+>")
MULTI_SPACE_RE = re.compile(r"\s+")
NON_TEXT_RE = re.compile(r"[^a-z0-9\s']")


def _normalize_unicode(text: str) -> str:
    """NFKC-normalize and strip combining marks that break tokenization."""
    return unicodedata.normalize("NFKC", text)


def clean_text(text: str, *, keep_emojis: bool = True) -> str:
    """
    Lowercase, strip URLs/HTML, and normalize whitespace in ``text``.

    When ``keep_emojis`` is True, emojis are replaced with their textual alias
    (``:thumbs_up:``) so they survive downstream tokenization; otherwise they
    are dropped.
    """
    if not isinstance(text, str) or not text:
        return ""

    text = _normalize_unicode(text).lower()
    text = HTML_TAG_RE.sub(" ", text)
    text = URL_RE.sub(" ", text)

    if keep_emojis:
        # demojize -> ":red_heart:" tokens we can keep.
        text = emoji.demojize(text, delimiters=(" :", ": "))
    else:
        text = emoji.replace_emoji(text, replace="")

    # Preserve alphanum, whitespace, apostrophes, and the ':alias:' tokens.
    text = re.sub(r"[^a-z0-9\s':_]", " ", text)
    text = MULTI_SPACE_RE.sub(" ", text).strip()
    return text


def _word_count(text: str) -> int:
    """Whitespace-tokenized word count."""
    return len(text.split()) if isinstance(text, str) else 0


def _flag_authenticity(text: str, keywords: list[str]) -> bool:
    """True if any authenticity keyword appears as a whole-word match."""
    if not isinstance(text, str) or not text:
        return False
    # Word-boundary matching, but allow "sounds like" style phrases.
    for kw in keywords:
        if " " in kw:
            if kw in text:
                return True
        else:
            if re.search(rf"\b{re.escape(kw)}\b", text):
                return True
    return False


def preprocess(
    master_path: Path | str | None = None,
    output_path: Path | str | None = None,
    *,
    keep_emojis: bool = True,
) -> pd.DataFrame:
    """
    Run the full cleaning pipeline and return the cleaned DataFrame.

    Parameters
    ----------
    master_path
        Path to the master CSV produced by ``acquisition.build_dataset``.
    output_path
        Where to write the cleaned CSV. Defaults to
        ``data/processed/cleaned_dataset.csv``.
    keep_emojis
        Forwarded to ``clean_text``.
    """
    master_path = Path(master_path or PROCESSED_DIR / "master_dataset.csv")
    output_path = Path(output_path or PROCESSED_DIR / "cleaned_dataset.csv")

    if not master_path.exists():
        raise FileNotFoundError(
            f"{master_path} not found. Run the acquisition step first (--acquire)."
        )

    df = pd.read_csv(master_path)
    logger.info("Loaded master dataset: shape=%s", df.shape)
    print("[preprocessing] input shape:", df.shape)

    # ------------------------------------------------------------------
    # Text cleanup
    # ------------------------------------------------------------------
    df["text_raw"] = df["text"].astype(str)
    df["text_clean"] = df["text_raw"].map(lambda t: clean_text(t, keep_emojis=keep_emojis))
    df["word_count"] = df["text_clean"].map(_word_count)

    # ------------------------------------------------------------------
    # Engagement ratios (per-video; broadcast through the comment rows)
    # ------------------------------------------------------------------
    for col in ("viewCount", "likeCount", "commentCount"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df["like_to_view_ratio"] = np.where(
        df["viewCount"] > 0, df["likeCount"] / df["viewCount"], np.nan
    )
    df["comment_rate"] = np.where(
        df["viewCount"] > 0, df["commentCount"] / df["viewCount"], np.nan
    )

    # ------------------------------------------------------------------
    # Filter short comments + flag authenticity keywords
    # ------------------------------------------------------------------
    before = len(df)
    df = df[df["word_count"] >= MIN_COMMENT_WORDS].copy()
    logger.info("Dropped %d comments with <%d words.", before - len(df), MIN_COMMENT_WORDS)

    df["authenticity_flag"] = df["text_clean"].map(
        lambda t: _flag_authenticity(t, AUTHENTICITY_KEYWORDS)
    )

    # ------------------------------------------------------------------
    # Persist
    # ------------------------------------------------------------------
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8")
    logger.info("Wrote cleaned dataset: %s (shape=%s)", output_path, df.shape)

    print("[preprocessing] output shape:", df.shape)
    print(df[["video_id", "narration_type", "word_count", "authenticity_flag"]].head(3).to_string())
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    preprocess()
