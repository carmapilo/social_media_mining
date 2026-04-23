"""
CLI entry point for the YouTube narration study pipeline.

Usage
-----
    python main.py --acquire      # only pull from the YouTube Data API
    python main.py --preprocess   # clean + feature-engineer
    python main.py --linguistic   # TF-IDF / n-grams / co-occurrence
    python main.py --sentiment    # embeddings + clustering + VADER
    python main.py --stats        # Mann-Whitney U + Spearman tables
    python main.py --all          # run everything end-to-end
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Ensure the project root is on sys.path when run as a script.
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Windows' default console codec (cp1252) explodes on emojis in video titles /
# comments; force UTF-8 with a replacement fallback so print(...) never kills
# the pipeline over a rendering issue.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger("pipeline")


def _timed(name: str, fn, *args, **kwargs):
    """Run ``fn`` while logging wall-clock duration under the label ``name``."""
    logger.info("=== %s: START ===", name)
    t0 = time.perf_counter()
    try:
        result = fn(*args, **kwargs)
    except Exception:
        logger.exception("%s: FAILED", name)
        raise
    logger.info("=== %s: DONE in %.1fs ===", name, time.perf_counter() - t0)
    return result


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="YouTube AI vs. human narration study pipeline")
    p.add_argument("--acquire", action="store_true", help="Pull videos + comments from the YouTube API")
    p.add_argument("--preprocess", action="store_true", help="Clean text + engineer features")
    p.add_argument("--linguistic", action="store_true", help="TF-IDF + n-grams + co-occurrence")
    p.add_argument("--sentiment", action="store_true", help="Embeddings + KMeans + VADER")
    p.add_argument("--stats", action="store_true", help="Mann-Whitney U + Spearman")
    p.add_argument("--all", action="store_true", help="Run every step in order")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if not any([args.acquire, args.preprocess, args.linguistic, args.sentiment, args.stats, args.all]):
        logger.error("No pipeline stage selected. See --help.")
        sys.exit(2)

    if args.acquire or args.all:
        from src.acquisition import build_dataset
        _timed("acquire", build_dataset)

    if args.preprocess or args.all:
        from src.preprocessing import preprocess
        _timed("preprocess", preprocess)

    if args.linguistic or args.all:
        from src.linguistic_analysis import run as run_linguistic
        _timed("linguistic", run_linguistic)

    if args.sentiment or args.all:
        from src.sentiment_analysis import run as run_sentiment
        _timed("sentiment", run_sentiment)

    if args.stats or args.all:
        from src.statistical_analysis import run as run_stats
        _timed("stats", run_stats)

    logger.info("Pipeline finished successfully.")


if __name__ == "__main__":
    main()
