"""
Data acquisition module (Owner: Tiago).

Thin wrapper around the YouTube Data API v3 (`googleapiclient.discovery`) used
to pull video metadata and comment threads for the channels defined in
``config.py``. Raw API payloads are persisted to ``data/raw/`` so that the
rest of the pipeline can run offline; the public entry point
``build_dataset()`` returns a tidy merged DataFrame and writes it to
``data/processed/master_dataset.csv``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from tqdm import tqdm

from config import (
    AI_NARRATED_CHANNELS,
    CHANNEL_LOOKUP,
    HUMAN_NARRATED_CHANNELS,
    MAX_COMMENTS_PER_VIDEO,
    MAX_RESULTS_PER_CHANNEL,
    PROCESSED_DIR,
    RANDOM_SEED,
    RAW_DIR,
    TARGET_YEAR,
    require_api_key,
)

logger = logging.getLogger(__name__)
random.seed(RANDOM_SEED)


# ---------------------------------------------------------------------------
# Client / retry helpers
# ---------------------------------------------------------------------------
def _build_client():
    """Instantiate a YouTube Data API v3 client using the configured key."""
    return build("youtube", "v3", developerKey=require_api_key(), cache_discovery=False)


def _with_backoff(callable_, *, max_retries: int = 6, base_delay: float = 1.5):
    """
    Execute ``callable_`` with exponential backoff on transient API errors.

    Retries on HTTP 403 (quota / rate limit) and 5xx; re-raises everything else.
    """
    for attempt in range(max_retries):
        try:
            return callable_()
        except HttpError as exc:
            status = getattr(exc.resp, "status", None)
            reason = str(exc)
            transient = status in (403, 500, 502, 503, 504) or "quota" in reason.lower()
            if not transient or attempt == max_retries - 1:
                raise
            sleep_s = base_delay * (2 ** attempt) + random.random()
            logger.warning(
                "YouTube API error (status=%s, attempt=%d/%d). Sleeping %.1fs. %s",
                status, attempt + 1, max_retries, sleep_s, reason,
            )
            time.sleep(sleep_s)


def _dump_raw(payload: Any, name: str) -> Path:
    """Persist a raw JSON response under ``data/raw/`` and return its path.

    Windows limits single path components to 255 chars, and YouTube's
    ``nextPageToken`` can easily be >1000 chars, so we hash anything too long
    and keep a short readable prefix for debuggability.
    """
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
    if len(safe) > 120:
        digest = hashlib.sha1(safe.encode("utf-8")).hexdigest()[:12]
        safe = f"{safe[:100]}_{digest}"
    path = RAW_DIR / f"{safe}.json"
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    return path


# ---------------------------------------------------------------------------
# ISO 8601 duration parsing (avoids adding a dependency on isodate)
# ---------------------------------------------------------------------------
_DURATION_RE = re.compile(
    r"^P(?:(?P<days>\d+)D)?T?(?:(?P<hours>\d+)H)?(?:(?P<minutes>\d+)M)?(?:(?P<seconds>\d+)S)?$"
)


def _iso_duration_to_seconds(iso: str | None) -> int | None:
    """Convert ``PT#H#M#S`` style durations to integer seconds."""
    if not iso:
        return None
    m = _DURATION_RE.match(iso)
    if not m:
        return None
    parts = {k: int(v) if v else 0 for k, v in m.groupdict().items()}
    return parts["days"] * 86400 + parts["hours"] * 3600 + parts["minutes"] * 60 + parts["seconds"]


# ---------------------------------------------------------------------------
# Channel / video metadata
# ---------------------------------------------------------------------------
def _get_uploads_playlist_id(youtube, channel_id: str) -> str | None:
    """Look up a channel's uploads playlist ID (needed to page through videos)."""
    resp = _with_backoff(
        lambda: youtube.channels().list(part="contentDetails", id=channel_id).execute()
    )
    items = resp.get("items", [])
    if not items:
        logger.warning("Channel %s returned no items; skipping.", channel_id)
        return None
    return items[0]["contentDetails"]["relatedPlaylists"]["uploads"]


def _iter_playlist_video_ids(youtube, playlist_id: str, max_results: int) -> Iterable[str]:
    """Yield up to ``max_results`` video IDs from ``playlist_id``."""
    collected, page_token = 0, None
    while collected < max_results:
        page_size = min(50, max_results - collected)
        resp = _with_backoff(
            lambda: youtube.playlistItems()
            .list(
                part="contentDetails",
                playlistId=playlist_id,
                maxResults=page_size,
                pageToken=page_token,
            )
            .execute()
        )
        for item in resp.get("items", []):
            vid = item["contentDetails"].get("videoId")
            published = item["contentDetails"].get("videoPublishedAt")
            if vid:
                yield vid, published
                collected += 1
                if collected >= max_results:
                    break
        page_token = resp.get("nextPageToken")
        if not page_token:
            break


def get_channel_videos(channel_id: str, max_results: int = MAX_RESULTS_PER_CHANNEL) -> pd.DataFrame:
    """
    Fetch up to ``max_results`` recent videos for ``channel_id``.

    Returns a DataFrame with: video_id, channel_id, title, publishedAt, viewCount,
    likeCount, commentCount, duration_seconds. Videos not published in
    ``TARGET_YEAR`` are filtered out.
    """
    youtube = _build_client()
    uploads = _get_uploads_playlist_id(youtube, channel_id)
    if uploads is None:
        return pd.DataFrame()

    candidate_ids = list(_iter_playlist_video_ids(youtube, uploads, max_results))
    if not candidate_ids:
        return pd.DataFrame()

    # Pre-filter by published year using playlistItems metadata to save quota.
    filtered_ids = [vid for vid, pub in candidate_ids if pub and pub.startswith(str(TARGET_YEAR))]
    if not filtered_ids:
        logger.info("Channel %s: no videos in %d.", channel_id, TARGET_YEAR)
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    # videos.list supports up to 50 IDs per call.
    for i in range(0, len(filtered_ids), 50):
        chunk = filtered_ids[i : i + 50]
        resp = _with_backoff(
            lambda: youtube.videos()
            .list(part="snippet,statistics,contentDetails", id=",".join(chunk))
            .execute()
        )
        _dump_raw(resp, f"videos_{channel_id}_{i}")
        for item in resp.get("items", []):
            snippet = item.get("snippet", {})
            stats = item.get("statistics", {})
            content = item.get("contentDetails", {})
            published_at = snippet.get("publishedAt", "")
            if not published_at.startswith(str(TARGET_YEAR)):
                continue
            rows.append(
                {
                    "video_id": item["id"],
                    "channel_id": channel_id,
                    "title": snippet.get("title", ""),
                    "publishedAt": published_at,
                    "viewCount": int(stats.get("viewCount", 0) or 0),
                    "likeCount": int(stats.get("likeCount", 0) or 0),
                    "commentCount": int(stats.get("commentCount", 0) or 0),
                    "duration_seconds": _iso_duration_to_seconds(content.get("duration")),
                }
            )

    df = pd.DataFrame(rows)
    logger.info("Channel %s: %d videos from %d.", channel_id, len(df), TARGET_YEAR)
    return df


# ---------------------------------------------------------------------------
# Comments
# ---------------------------------------------------------------------------
def get_video_comments(video_id: str, max_comments: int = MAX_COMMENTS_PER_VIDEO) -> pd.DataFrame:
    """
    Fetch up to ``max_comments`` top-level comments for ``video_id``.

    Returns a DataFrame with: video_id, comment_id, text, likeCount, publishedAt,
    author. Silently returns an empty DataFrame if comments are disabled.
    """
    youtube = _build_client()
    rows: list[dict[str, Any]] = []
    page_token = None

    try:
        while len(rows) < max_comments:
            page_size = min(100, max_comments - len(rows))
            resp = _with_backoff(
                lambda: youtube.commentThreads()
                .list(
                    part="snippet",
                    videoId=video_id,
                    maxResults=page_size,
                    pageToken=page_token,
                    textFormat="plainText",
                    order="relevance",
                )
                .execute()
            )
            _dump_raw(resp, f"comments_{video_id}_{page_token or 'p0'}")
            for item in resp.get("items", []):
                top = item["snippet"]["topLevelComment"]["snippet"]
                rows.append(
                    {
                        "video_id": video_id,
                        "comment_id": item["id"],
                        "text": top.get("textDisplay", ""),
                        "likeCount": int(top.get("likeCount", 0) or 0),
                        "publishedAt": top.get("publishedAt", ""),
                        "author": top.get("authorDisplayName", ""),
                    }
                )
            page_token = resp.get("nextPageToken")
            if not page_token:
                break
    except HttpError as exc:
        # Comments disabled / not found -> skip quietly.
        if getattr(exc.resp, "status", None) in (403, 404):
            logger.info("Comments unavailable for %s: %s", video_id, exc)
            return pd.DataFrame()
        raise

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def build_dataset() -> pd.DataFrame:
    """
    Iterate every channel in ``config`` and assemble the master DataFrame.

    Each row is a single comment, joined with its parent video's metadata and
    annotated with ``narration_type`` ("human" | "ai") and ``genre``. The
    resulting frame is written to ``data/processed/master_dataset.csv``.
    """
    all_channels = {**HUMAN_NARRATED_CHANNELS, **AI_NARRATED_CHANNELS}
    video_frames: list[pd.DataFrame] = []
    comment_frames: list[pd.DataFrame] = []

    logger.info("Starting acquisition for %d channels.", len(all_channels))
    for channel_id in tqdm(list(all_channels.keys()), desc="channels"):
        try:
            videos = get_channel_videos(channel_id, MAX_RESULTS_PER_CHANNEL)
        except HttpError as exc:
            logger.error("Failed to fetch videos for %s: %s", channel_id, exc)
            continue

        if videos.empty:
            continue

        narration_type, genre = CHANNEL_LOOKUP[channel_id]
        videos["narration_type"] = narration_type
        videos["genre"] = genre
        video_frames.append(videos)

        for video_id in tqdm(videos["video_id"].tolist(), desc=f"  {channel_id[:10]}", leave=False):
            try:
                comments = get_video_comments(video_id, MAX_COMMENTS_PER_VIDEO)
            except HttpError as exc:
                logger.error("Failed to fetch comments for %s: %s", video_id, exc)
                continue
            if not comments.empty:
                comment_frames.append(comments)

    if not video_frames:
        logger.warning("No videos collected; returning empty DataFrame.")
        return pd.DataFrame()

    videos_df = pd.concat(video_frames, ignore_index=True)
    comments_df = (
        pd.concat(comment_frames, ignore_index=True)
        if comment_frames
        else pd.DataFrame(columns=["video_id", "comment_id", "text", "likeCount", "publishedAt", "author"])
    )

    # Rename overlapping columns so the join is unambiguous.
    comments_df = comments_df.rename(
        columns={"likeCount": "comment_likeCount", "publishedAt": "comment_publishedAt"}
    )
    merged = comments_df.merge(videos_df, on="video_id", how="left", suffixes=("", "_video"))

    out_path = PROCESSED_DIR / "master_dataset.csv"
    merged.to_csv(out_path, index=False, encoding="utf-8")
    logger.info("Wrote master dataset: %s (shape=%s)", out_path, merged.shape)

    # Also snapshot a videos-only table for downstream per-video analytics.
    videos_df.to_csv(PROCESSED_DIR / "videos_dataset.csv", index=False, encoding="utf-8")

    print("[acquisition] master_dataset shape:", merged.shape)
    if not merged.empty:
        print(merged.head(3).to_string())
    return merged


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    build_dataset()
