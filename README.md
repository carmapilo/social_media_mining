# YouTube Narration Study

Pipeline for analyzing YouTube audience reception of **AI-narrated** vs.
**human-narrated** videos across multiple genres, using the YouTube Data API
v3.

## Layout

```
yt_narration_study/
├── config.py                 # API key loading + channel lists + parameters
├── main.py                   # argparse CLI orchestrating every stage
├── requirements.txt
├── .env.example              # copy to .env and fill in YOUTUBE_API_KEY
├── data/
│   ├── raw/                  # raw JSON responses from the API
│   └── processed/            # tidy CSVs (master / cleaned / sentiment)
├── src/
│   ├── acquisition.py        # Tiago  — YouTube Data API v3 pulls
│   ├── preprocessing.py      # text cleaning + engagement ratios
│   ├── linguistic_analysis.py# Carlos — TF-IDF + n-grams + co-occurrence
│   ├── sentiment_analysis.py # Henrique — embeddings + KMeans + VADER
│   └── statistical_analysis.py# Natacha — Mann-Whitney U + Spearman
├── notebooks/
│   └── exploratory.ipynb
└── outputs/
    └── figures/
```

## Setup (Windows / PowerShell)

```powershell
cd yt_narration_study
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env    # then edit .env with your real API key
```

On macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
```

Get a YouTube Data API v3 key from
<https://console.cloud.google.com/apis/credentials>. The pipeline will refuse
to start if `YOUTUBE_API_KEY` is missing (no hardcoded fallbacks).

## Running the pipeline

```bash
python main.py --all          # end-to-end
python main.py --acquire      # just pull from the API
python main.py --preprocess   # clean CSVs produced by --acquire
python main.py --linguistic
python main.py --sentiment
python main.py --stats
```

Each stage prints the shape of its output DataFrame and a short sample to
stdout for debugging.

## Notes

- Random seed is `42` for all stochastic operations (KMeans, UMAP, sampling).
- Only videos published in `TARGET_YEAR` (default **2026**) are kept.
- The channel lists in `config.py` are placeholders — edit
  `HUMAN_NARRATED_CHANNELS` and `AI_NARRATED_CHANNELS` with the actual
  channel IDs you want to study before running `--acquire`.
- The first run of `--sentiment` downloads the
  `sentence-transformers/all-MiniLM-L6-v2` model (~90 MB) and the NLTK
  `vader_lexicon`.
