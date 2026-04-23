"""
Configuration module for the YouTube narration study.

Loads the YouTube Data API v3 key from the environment (.env) and defines
the channel universe (human- vs. AI-narrated) along with pipeline parameters.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"
OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"
FIGURES_DIR: Path = OUTPUTS_DIR / "figures"

for _d in (RAW_DIR, PROCESSED_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Secrets
# ---------------------------------------------------------------------------
load_dotenv(PROJECT_ROOT / ".env")

YOUTUBE_API_KEY: str | None = os.getenv("YOUTUBE_API_KEY")


def require_api_key() -> str:
    """Return the API key or raise a helpful error if it is missing."""
    if not YOUTUBE_API_KEY:
        raise RuntimeError(
            "YOUTUBE_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    return YOUTUBE_API_KEY


# ---------------------------------------------------------------------------
# Pipeline parameters
# ---------------------------------------------------------------------------
MAX_RESULTS_PER_CHANNEL: int = 50
MAX_COMMENTS_PER_VIDEO: int = 100
TARGET_YEAR: int = 2026
RANDOM_SEED: int = 42


# ---------------------------------------------------------------------------
# Channel universe
# ---------------------------------------------------------------------------
# Map YouTube channel_id -> genre label. These IDs are placeholders intended
# to be edited by the research team; any `UC...` channel ID is valid.
HUMAN_NARRATED_CHANNELS: dict[str, str] = {
    # History / documentary
    "UCX6b17PVsYBQ0ip5gyeme-Q": "science",            # CrashCourse
    "UCsooa4yRKGN_zEE8iknghZA": "education",          # TED-Ed
    "UC9-y-6csu5WGm29I7JiwpnA": "tech",               # Computerphile
    "UCYO_jab_esuFRV4b17AJtAw": "education",          # 3Blue1Brown
    "UCHnyfMqiRRG1u-2MsSQLbXA": "science",            # Veritasium
    # Vlogs / lifestyle
    "UC-lHJZR3Gqxm24_Vd_AJ5Yw": "entertainment",      # PewDiePie
    "UCBJycsmduvYEL83R_U4JriQ": "tech-review",        # Marques Brownlee
    "UCsTcErHg8oDvUnTzoqsYeNw": "entertainment",      # Unbox Therapy
    # News / commentary
    "UCXIJgqnII2ZOINSWNOGFThA": "news",               # Fox News
    "UCeY0bbntWzzVIaj2z3QigXg": "news",               # NBC News
    "UC16niRr50-MSBwiO3YDb3RA": "news",               # BBC News
    # Culture / true crime
    "UC0intLFzLaudFG-xAvUEO-A": "tech-explainer",     # Not Enough Nelsons (example)
    "UCOmHUn--16B90oW2L6FRR3A": "music",              # BLACKPINK
}

AI_NARRATED_CHANNELS: dict[str, str] = {
    # Common AI-narrated formats: top-10 lists, "facts", auto-generated compilations
    "UCpFFItkfZz1qz5PpHpqzYBw": "top-lists",          # Top 10 (placeholder)
    "UC6107grRI4m0o2-emgoDnAA": "science",            # SmarterEveryDay-like placeholder
    "UC4QZ_LsYcvcq7qOsOhpAX4A": "history",            # ColdFusion-like placeholder
    "UCb_MAhL8Thb3HJ_wPkH3gcw": "facts",              # Facts channel placeholder
    "UC9RM-iSvTu1uPJb8X5yp3EQ": "documentary",        # Wendover-like placeholder
    "UCZaT_X_mc0BI-djXOlfhqWQ": "history",            # Vox-like placeholder
    "UClWCQNaggkMW7SDtS3BkEBg": "crypto",             # Crypto AI placeholder
    "UCqVEHtQoXHmUCfJ-9smpTSg": "gaming",             # Gaming compilation placeholder
    "UC295-Dw_tDNtZXFeAPAW6Aw": "top-lists",          # 5-Minute Crafts (often AI narrated)
    "UCYzPXprvl5Y-Sf0g4vX-m6g": "how-to",             # jacksepticeye-like placeholder
    "UCNvzD7Z-g64bPXxGzaQaa4g": "finance",            # Finance AI placeholder
    "UCA19mAJURyYHbJzhfpqhpCA": "mystery",            # Mystery facts placeholder
    "UCYfdidRxbB8Qhf0Nx7ioOYw": "space",              # Space facts placeholder
}


# Convenience: a flat lookup from channel_id -> (narration_type, genre)
CHANNEL_LOOKUP: dict[str, tuple[str, str]] = {
    **{cid: ("human", g) for cid, g in HUMAN_NARRATED_CHANNELS.items()},
    **{cid: ("ai", g) for cid, g in AI_NARRATED_CHANNELS.items()},
}


# ---------------------------------------------------------------------------
# Analysis knobs
# ---------------------------------------------------------------------------
AUTHENTICITY_KEYWORDS: list[str] = [
    "ai",
    "robot",
    "fake",
    "real",
    "voice",
    "generated",
    "synthetic",
    "human",
    "sounds like",
]

MIN_COMMENT_WORDS: int = 5
KMEANS_K_DEFAULT: int = 6
KMEANS_K_RANGE: tuple[int, int] = (3, 10)  # for silhouette tuning
SENTENCE_MODEL: str = "all-MiniLM-L6-v2"
