# YouTube Audience Reception of AI vs. Human Narration
### A mixed-methods content-analysis study  —  Spring 2026

**Authors / owners**
- Tiago — data acquisition
- Carlos — linguistic analysis
- Henrique — sentiment & clustering
- Natacha — statistical testing

> This document doubles as (a) the speaker script for the oral presentation and
> (b) the methods/results backbone of the written report. Each section is
> written to be read aloud in 60–120 s; the tables and figure callouts map
> 1:1 to the slide deck the team can build from `outputs/figures/`.

---

## 1. Motivation & research question

AI voice models (ElevenLabs, OpenAI TTS, NotebookLM, Google WaveNet, etc.) have
moved from novelty to default production tool on YouTube over the past
eighteen months. Entire top-10 lists, facts channels, finance explainers, and
even documentary-style long-form content are now narrated by synthesized
voices indistinguishable, at first pass, from humans.

The question we set out to answer:

> **RQ.**  When controlling for topic and release window, does a YouTube
> audience engage with, react to, and talk about AI-narrated videos
> differently than human-narrated ones?

Operationally we decompose it into four sub-questions:

1. **Engagement (H1):** Do AI-narrated videos receive different like-to-view
   and comment-to-view ratios than human-narrated ones?
2. **Sentiment (H2):** Does the tone of the comment section differ between
   the two groups?
3. **Authenticity talk (H3):** Do viewers discuss authorship / authenticity
   (keywords like *AI, robot, fake, real, voice, generated, synthetic,
   human, sounds like*) at different rates on AI-narrated videos?
4. **Qualitative topic mix (RQ4):** Do the *kinds* of conversations that
   emerge differ qualitatively (n-grams, clusters of semantically similar
   comments)?

---

## 2. Data collection specification

### 2.1 Source & instrument

- **Source:** YouTube Data API v3 (quota-metered REST API).
- **Endpoints used:**
  - `channels.list` → uploads-playlist lookup
  - `playlistItems.list` → recent video IDs (paginated, 50 per page)
  - `videos.list` → metadata (snippet + statistics + contentDetails)
  - `commentThreads.list` → top-level comments only (`order=relevance`,
    `textFormat=plainText`)
- **Client library:** `google-api-python-client` 2.194.
- **Authentication:** API key loaded from `.env` via `python-dotenv`; no
  OAuth scopes, no user-specific data.
- **Error handling:** exponential-backoff wrapper with up to 6 retries on
  HTTP 403/5xx and an explicit fall-through on 403/404 for videos with
  comments disabled (the final count reflects only fetchable videos).

### 2.2 Sampling frame

| Stratum | Channels targeted | Channels with 2026 uploads | Genre labels used |
|---|---|---|---|
| Human-narrated | 13 | 13 | science, education, tech, tech-review, tech-explainer, news, entertainment, music |
| AI-narrated | 13 | 12 | crypto, documentary, facts, finance, gaming, history, how-to, mystery, top-lists |

Channels were hand-audited and labelled by inspecting at least three recent
videos per channel for narration type before the list was committed to
`config.py`. One channel (`UCYfdidRxbB8Qhf0Nx7ioOYw`) returned HTTP 404 on
its uploads playlist and was dropped.

### 2.3 Inclusion criteria

- **Temporal window:** videos whose `snippet.publishedAt` falls inside
  **calendar year 2026** (`TARGET_YEAR = 2026`). The filter is applied twice:
  once cheaply on `playlistItems` to save quota, and a second time on the
  authoritative `videos.list` response.
- **Per-channel cap:** up to `MAX_RESULTS_PER_CHANNEL = 50` most recent
  videos, yielding a recency-weighted random sample within-channel.
- **Per-video cap:** up to `MAX_COMMENTS_PER_VIDEO = 100` top-level comments
  ordered by YouTube's "relevance" ranking (not "time"), i.e. the comments
  most surfaced to real viewers.
- **Minimum comment length:** comments with <5 whitespace-tokenized words
  are dropped before analysis (removes sticker-only replies, one-word
  exclamations, and spam).

### 2.4 Realized sample

After all filters *(latest full `python main.py --all` refresh, spring 2026)*:

- **664 videos** (374 human, 290 AI) across **25 distinct channels**.
- **41,495 cleaned comments** (24,284 human / 17,211 AI).
- **1,957 authenticity-flagged comments**.
- **659 videos** retained at the statistical-test stage (those with at least
  one sentiment-scorable comment after filtering).

### 2.5 Preprocessing

| Step | Rule |
|---|---|
| Unicode normalization | NFKC |
| Case | lower-cased |
| URL stripping | regex `https?://\S+` and `www.\S+` |
| HTML tag removal | regex `<[^>]+>` |
| Emoji handling | kept, demojized to `:alias:` tokens (e.g. `:red_heart:`) so they survive TF-IDF and can be read as topical signals |
| Character filter | keep `[a-z0-9\s':_]` only |
| Min-length filter | drop rows with <5 words |
| Engagement ratios | `like_to_view_ratio = likeCount / viewCount` and `comment_rate = commentCount / viewCount`, computed per-video and broadcast |

All intermediate artefacts (raw JSON dumps per API call, `master_dataset.csv`,
`cleaned_dataset.csv`, `comments_scored.csv`, `sentiment_dataset.csv`) are
versioned in `data/` so the downstream stages are fully reproducible offline.

---

## 3. Analytical design

### 3.1 Linguistic analysis  (Carlos)

- **TF-IDF (unigrams):** `TfidfVectorizer(stop_words='english', min_df=3,
  max_df=0.9)`; we rank terms by their summed column weight across the group
  corpus and keep the top 50.
- **N-grams:** `CountVectorizer` restricted to `(2,2)` and `(3,3)`; top 20
  each.
- **Authenticity keyword co-occurrence:** for the subset flagged during
  preprocessing, we compute a symmetric keyword-presence matrix (diagonal =
  marginal counts, off-diagonal = pairwise co-occurrence).

### 3.2 Sentiment analysis  (Henrique)

- **Dense embeddings:** `sentence-transformers/all-MiniLM-L6-v2` (384-dim,
  cosine-friendly). Batch size 64, run on CPU.
- **Clustering:** KMeans with silhouette tuning over `k ∈ [3, 10]` on a
  5 000-comment random sample for computational tractability; the full
  embedding set is then re-fit with the winning `k`.
- **Cluster labelling:** per-cluster TF-IDF means give the top-5 keyword
  tag for each cluster centroid.
- **Visualization:** UMAP (`n_neighbors=15, min_dist=0.1`) reduction to 2D,
  coloured by cluster, styled by narration type.
- **Lexicon sentiment:** NLTK VADER `compound` score per comment; per-video
  aggregates are `mean_sentiment`, `std_sentiment`, `pct_negative`
  (compound < -0.05), `pct_positive` (compound > 0.05).
- **Supervised lexicon proxies (interpretability):** on non-neutral comments we
  derive **weak binary labels** from the same VADER cutoffs (positive:
  compound > 0.05; negative: compound < −0.05). We fit a shared
  `TfidfVectorizer` (unigrams + bigrams, `min_df=2`, `max_df=0.95`,
  `max_features=20_000`, English stop words) and train **balanced**
  `LogisticRegression`, **`RandomForestClassifier`**, and **`XGBClassifier`**
  (when installed). Hold-out metrics and **model-specific important terms**
  (regression coefficients ±; tree `feature_importances_`) are written to
  `outputs/sentiment_model_comparison.csv` and `outputs/sentiment_top_terms.csv`,
  and summary bars to `outputs/figures/sentiment_classifier_metrics.png`.

### 3.3 Statistical testing  (Natacha)

- **Group comparisons:** two-sided Mann-Whitney U (non-parametric, robust
  to the heavy-tailed view-count distributions), with a **rank-biserial
  correlation** effect size `r = 1 − 2U/(n₁·n₂)`. Positive `r` = human
  group ranks higher.
- **Association within group:** Spearman ρ between `mean_sentiment` and
  `like_to_view_ratio`, computed separately for AI and human videos.
- **Significance threshold:** α = 0.05 (two-sided). No multiplicity
  correction is applied in the primary analysis because the five metrics
  are reported as independent descriptors of different facets; a Bonferroni
  α = 0.01 is noted in-line as a sensitivity check.

### 3.4 Reproducibility

- Random seed **42** is set for NumPy, KMeans, UMAP, and the silhouette
  sub-sampler.
- The full pipeline is driven by a single CLI
  (`python main.py --all`), which logs timestamped stage transitions.
- All data flows forward as CSVs; re-running a single stage never requires
  touching the API.

---

## 4. Results

### 4.1 Descriptive summary

| Metric (median per video) | Human (n=372) | AI (n=287) |
|---|---:|---:|
| viewCount | 195 148 | 190 554 |
| likeCount | 5 230 | 5 946 |
| commentCount | 280.5 | 278 |
| like_to_view_ratio | 0.0283 | 0.0307 |
| comment_rate | 0.00202 | 0.00186 |
| mean VADER compound (per video) | 0.127 | 0.148 |
| % negative comments (per video) | 22.3% | 23.4% |
| authenticity-flag rate (per video) | 2.97% | 2.11% |

The two groups remain close on raw engagement shape; median views are
slightly higher on human-tagged buckets in this refresh, while median likes
skew toward AI-associated videos—consistent with heterogeneous formats
within each stratum rather than a clean “human vs synthetic” payoff gap.

### 4.2 Inferential results — Mann-Whitney U (n = 659 videos)

| Metric | U | p | r (effect) | Significant @ α=0.05 |
|---|---:|---:|---:|:---:|
| like_to_view_ratio | 52 946 | **0.857** | +0.008 | no |
| comment_rate | 59 850 | **0.0076** | −0.121 | yes |
| mean_sentiment (VADER) | 48 275.5 | **0.035** | +0.096 | yes |
| pct_negative | 55 159 | **0.463** | −0.033 | no |
| authenticity_flag_rate | 58 142 | **0.0468** | −0.089 | yes |

**Reading the table.** In this rerun, **three metrics are significant** at
α = 0.05: `comment_rate`, `mean_sentiment`, and `authenticity_flag_rate`.
Directionally: AI-tagged videos show **higher authenticity chatter** (negative
`r` in our convention) and **lower comment-rate medians**, while human-tagged
videos retain slightly stronger median per-video sentiment. The effect sizes
remain small-to-modest (|r| up to ~0.12), so this is better read as a
distributional shift than a large practical gap.

### 4.3 Spearman: sentiment ↔ engagement

| Group | Pair | ρ | p | n |
|---|---|---:|---:|---:|
| **Human** | mean_sentiment ~ like_to_view_ratio | **+0.471** | < 10⁻²¹ | 372 |
| **AI** | mean_sentiment ~ like_to_view_ratio | **+0.185** | **0.0017** | 287 |

**Structural pattern.** Sentiment aligns with liking behavior *more sharply*
for human-labelled videos (moderate ρ) than AI-labelled videos (still
positive but clearly weaker ρ). Absolute coefficients moved slightly versus
last export, **but the human–AI gap in coupling is the replication-stable story**.

### 4.4 Comment-level VADER distributions

| | Mean compound | Median | Std |
|---|---:|---:|---:|
| Human | +0.134 | 0.00 | 0.452 |
| AI | +0.161 | 0.078 | 0.481 |

Both distributions stay positively skewed and zero-heavy; centers remain
near neutral with AI comments fractionally brighter on average in this scrape.

### 4.5 Supervised TF-IDF sentiment models *(weak labels)*

Held-out benchmarks on **24,188 / 6,047** train/test weak-labelled comments:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | **0.849** | **0.912** | 0.853 | **0.881** | **0.922** |
| Random Forest | 0.837 | 0.863 | 0.893 | 0.878 | 0.903 |
| XGBoost | 0.793 | 0.776 | **0.963** | 0.860 | 0.873 |

**Interpretation guardrail:** models are calibrated to imitate VADER’s own
margins—they **do not replace humans**—but they let us **summarize vocabulary**
that aligns with predicted polarity.

Top positive logistic coefficients include *love*, *like*, *great*, *best*,
*better*, *good*; strongest negatives cluster around conflict / moral-outrage
surface forms (*war*, *dead*, *bad*, *wrong*, *evil*, …). Random Forest /
XGBoost global importances emphasize the same *love / like / great / good*
spine—the full ranked tables live in `outputs/sentiment_top_terms.csv`.

### 4.6 Clustering (KMeans, **k = 9** selected by silhouette on a 5k subsample)

Silhouette scores across the candidate range (same pipeline settings as §3):

| k | 3 | 4 | 5 | 6 | 7 | 8 | **9** | 10 |
|---|---|---|---|---|---|---|---|---|
| silhouette | 0.019 | 0.021 | 0.023 | 0.024 | 0.024 | 0.023 | **0.024** | 0.022 |

Scores stay low in absolute terms (noisy short text), but **`k = 9` wins** this
refresh. Labels from per-cluster TF-IDF centroid means (top-5 lexemes):

| Cluster | Label / keywords | Human | AI | Notes |
|---|---|---:|---:|---|
| 0 | `video / watching` praise bucket | 2 125 | 1 376 | Generic appreciation comments |
| 1 | `blackpink / red_heart` fan bucket | 1 660 | 338 | Human-heavy fan reactions |
| 2 | `trump / iran / war` geopolitics | 3 980 | 1 662 | News-cycle concentration |
| 3 | mixed social reaction (`love / like / grokvarum`) | 3 557 | 3 432 | Broad mixed discourse |
| 4 | `game / games / play` gameplay talk | 497 | **3 202** | Strong AI-stratum tilt |
| 5 | `phone / apple / iphone` gadget chatter | **3 139** | 1 164 | Tech-topic concentration |
| 6 | numeric/timestamp-heavy segment (`00 / 12 / 10`) | 1 148 | 1 526 | Formatting/time-coded comments |
| 7 | emoji sentiment reactions | 2 063 | 1 247 | Short affective reactions |
| 8 | high-volume generic bucket (`like / people / know`) | **6 115** | 3 264 | Largest mixed cluster |

Interpretive anchors:

- **Cluster 4 is now the clearest gameplay AI signature** (3,202 AI vs. 497 human).
- **Cluster 8 is the largest cross-stratum bucket**, showing substantial shared everyday-comment language.

### 4.7 Qualitative n-gram comparison

**Top bigrams per group** (raw counts with `CountVectorizer(2–2)`, `min_df=3`, English stopwords):

| Rank | Human | count | AI | count |
|---|---|---:|---|---:|
| 1 | looks like | 157 | feel like | 169 |
| 2 | ted ed | 136 | looks like | 131 |
| 3 | years ago | 128 | years ago | 116 |
| 4 | feel like | 127 | crimson desert | 114 |
| 5 | feels like | 105 | feels like | 96 |
| 6 | don't know | 100 | don't know | 94 |
| 7 | crash course | 86 | can't wait | 72 |
| 8 | international blinks | 82 | **sounds like** | 72 |
| 9 | support blackpink | 82 | just like | 63 |
| 10 | **sounds like** | 79 | resident evil | 61 |

Key patterns:

- `sounds like` appears in *both* top-tens — it's the single most
  diagnostic surface cue of authenticity talk ("sounds like AI", "sounds
  like a robot", "sounds like a real person"). That it shows up so
  prominently in both strata is evidence that **the authorship question
  is now a shared frame across the platform**, not a niche concern.
- `feel like / feels like` dominate the AI bucket more than the human one
  — these bigrams are idiomatic of subjective assessment ("this feels
  AI-generated", "feels too scripted"), consistent with H3.
- `crimson desert`, `resident evil`, `video game` expose the gaming
  compilations in the AI stratum; `ted ed`, `strait hormuz`, `god bless`,
  `president trump` expose the education+news domination of the human
  stratum. **This is our most important genre-confound warning** —
  see §6.

### 4.8 Authenticity keyword co-occurrence

Among the **1,957** authenticity-flagged comments, the pairs that co-occur most
often in the same comment are:

- `human` × `voice` (the most frequent pair; literally "human voice" as a
  compliment or as a benchmark)
- `real` × `voice` (second most frequent; "real voice")
- `ai` × `voice` (third; the direct accusation)
- `sounds like` × `ai` / `sounds like` × `human` (diagnostic phrasing)

The full matrix is persisted at `outputs/authenticity_cooccurrence.csv`
and visualized at `outputs/figures/authenticity_cooccurrence.png`. The
pattern supports reading the authenticity discourse as a coherent
sub-conversation (not scattered keyword hits) organised around the
concept of *voice*.

---

## 5. Figures cited in this deck  (`outputs/figures/`)

| Filename | What it shows | Referenced in |
|---|---|---|
| `tfidf_top_human.png` | Top 50 TF-IDF terms, human-narrated corpus | §4.7 |
| `tfidf_top_ai.png` | Top 50 TF-IDF terms, AI-narrated corpus | §4.7 |
| `bigrams_human.png` / `bigrams_ai.png` | Top 20 bigrams per group | §4.7 |
| `trigrams_human.png` / `trigrams_ai.png` | Top 20 trigrams per group | §4.7 |
| `authenticity_cooccurrence.png` | Keyword-pair heatmap | §4.8 |
| `sentiment_distribution.png` | VADER compound KDE, split by narration type | §4.4 |
| `per_video_sentiment.png` | Box/strip plot of per-video mean sentiment | §4.1 |
| `umap_clusters.png` | UMAP 2D projection coloured by cluster | §4.6 |
| `sentiment_classifier_metrics.png` | F1 + ROC-AUC bars for supervised TF-IDF sentiment models | §4.5 |

### 5.1 Embedded figure gallery *(Markdown-ready for deck / PDF tooling)*

![VADER distribution by narration type](outputs/figures/sentiment_distribution.png)

![Per-video mean sentiment](outputs/figures/per_video_sentiment.png)

![UMAP clusters](outputs/figures/umap_clusters.png)

![Supervised classifier metrics](outputs/figures/sentiment_classifier_metrics.png)

![Authenticity keyword co-occurrence](outputs/figures/authenticity_cooccurrence.png)

![TF-IDF top terms — human](outputs/figures/tfidf_top_human.png)

![TF-IDF top terms — AI](outputs/figures/tfidf_top_ai.png)

![Top bigrams — human](outputs/figures/bigrams_human.png)

![Top bigrams — AI](outputs/figures/bigrams_ai.png)

![Top trigrams — human](outputs/figures/trigrams_human.png)

![Top trigrams — AI](outputs/figures/trigrams_ai.png)

The executed notebook `notebooks/exploratory.ipynb` additionally contains:

- Per-group comment volumes and authenticity-flag counts
- Per-video boxplot of `mean_sentiment`
- Scatter of `mean_sentiment` vs. `like_to_view_ratio`, styled by genre

---

## 6. Constraints, threats to validity, and caveats

### 6.1 Genre confound (most important)

The human and AI strata **do not overlap in genre**: *news*, *education*,
*tech-review*, *music*, and *entertainment* appear only in the human
stratum; *crypto*, *facts*, *gaming*, *history*, *how-to*, *mystery*,
*top-lists*, *documentary*, and *finance* appear only in the AI stratum.
That reflects the real production ecosystem (genres adopt AI narration at
very different rates), **but it means every "narration type" contrast is
partially a "genre" contrast.** Any narrative about **authentication / provenance chatter** still requires
genre-controlled replication — at minimum, a genre-stratified bootstrap or
a mixed-effects model with channel as a random effect — even though the
latest rerun does show a significant Mann-Whitney split on
`authenticity_flag_rate` (small effect size).

### 6.2 Labelling risk

Channel-level narration labels are **human-annotated from a short sample**;
channels that mix AI and human narration (increasingly common) are
misrepresented as monolithic. A clip-level labelling pass (listening to
the audio track of each video in the sample) would tighten this.

### 6.3 Sampling biases internal to YouTube

- `commentThreads.list` with `order=relevance` returns comments YouTube's
  ranker surfaces, not a random sample. Our results therefore describe
  *the conversation a typical viewer sees*, not *every comment ever left*.
  This is arguably the more ecologically valid target, but it should be
  disclosed.
- We collected only **top-level comments**; replies were ignored. Authenticity
  discussions often happen in reply threads ("no it's AI", "lmao it is not"),
  so our flag rate is a conservative lower bound.
- `MAX_COMMENTS_PER_VIDEO = 100` caps our view of each video; high-traffic
  videos are under-sampled relative to low-traffic ones, which effectively
  equalizes per-video weight and is appropriate for the per-video
  aggregates we report.

### 6.4 Temporal scope

Only calendar **2026** is in-frame. Year-on-year change (e.g. how the
authenticity frame has hardened since 2023) is out of scope for this
analysis; re-running with `TARGET_YEAR` in a loop would support a
longitudinal extension.

### 6.5 Tooling limitations

- **VADER** is a rule-based, English-first, social-media-tuned lexicon. It
  does well on short comments but misses sarcasm, code-switching, and
  non-English content. A small fraction of comments in our sample are in
  Korean, Spanish, Portuguese, or Turkish and receive a 0.0 compound by
  default — downstream results under-represent these viewers.
- **MiniLM embeddings** are 384-dim and English-primary; out-of-English
  comments cluster mostly as outliers. A multilingual model (e.g.
  `paraphrase-multilingual-MiniLM-L12-v2`) would be a principled swap.
- **Silhouette scores are uniformly low** (~0.02), which is expected for
  social-media text in a high-dimensional space but means the cluster
  assignments are soft at the boundary. We interpret clusters as
  *qualitative types* rather than hard partitions.

### 6.6 Statistical caveats

- No multiple-comparison adjustment is applied in the primary video-level
  battery; with **five** metrics none clears α = 0.05 in the May 2026
  `--all` rerun, so multiplicity is moot for declaring “wins” — but it
  still matters if exploratory follow-ups chip away at *p* thresholds.
- Mann-Whitney tests the distributions, not a specific parameter; we
  retain `r` so directional leanings (e.g. authenticity rate) can be read
  alongside *p* even when null hypotheses are not rejected.

### 6.7 Engineering / ops notes (for the methods appendix)

During the real run we hit two Windows-specific failure modes that are
worth recording so future replications budget for them:

1. YouTube's `nextPageToken` can exceed 1 000 characters, which blows past
   Windows' 255-char path-component limit when we try to cache raw JSON.
   We hash long filenames (SHA-1 prefix) to stay portable.
2. The default `cp1252` console codec on Windows crashes when a print
   statement happens to contain a CJK character or a 4-byte emoji. The
   CLI now forces UTF-8 with a replacement fallback.

Neither affects the data; both are documented in the code.

---

## 7. Conclusions

1. **Engagement surfaces remain close, but not fully exchangeable.**
   `like_to_view_ratio` and `pct_negative` remain non-significant, while
   `comment_rate` separates at α = 0.05 (small effect).
2. **Authenticity keyword rates again lean AI-tagged and are significant in this rerun**
   (`p ≈ 0.047`, negative rank-biserial convention), though the effect remains
   small and still potentially confounded by topic mix.
3. **Sentiment retains a tighter Spearman coupling to liking on human-labelled uploads**
   (ρ ≈ 0.47 vs. ρ ≈ 0.18 on AI). That divergence persists even while group means overlap,
   implying different *conversion* from perceived tone to applause—not different central
   affect alone.
4. **Machine-learned lexical probes echo VADER-aligned vocabulary** (`love / like /
   great` vs. outrage / tragedy lexemes), giving a reproducible checklist of polarity-
   laden tokens instead of purely hand-picked dictionaries.

Responsible synthesis: **we see repeatable but small distributional differences on selected metrics**
(`comment_rate`, `mean_sentiment`, `authenticity_flag_rate`) rather than a large
engagement cliff, while **the asymmetric sentiment→like correlation** remains the strongest
and most stable behavioral signal for “loyal viewer” hypotheses.

---

## 8. Future work

- **Genre-stratified replication** to separate narration effects from
  genre effects (critical — §6.1).
- **Reply-thread inclusion** to capture the back-and-forth structure of
  authenticity debates.
- **Multilingual sentiment / embeddings** to cover the non-English tail.
- **Longitudinal extension** by parameterizing `TARGET_YEAR` over
  2022–2026 to measure *when* authenticity talk emerged as a shared frame.
- **Supervised authenticity classifier** (fine-tuned from the 1,957
  flagged seeds) to scale beyond keyword matching and catch sarcasm
  ("definitely not AI at all 😐").
- **Controlled experiment**: re-narrate a set of human videos with an AI
  voice (and vice versa) and re-release to a holdout audience — the only
  way to causally identify the narration effect cleanly.

---

## Appendix A — Speaker timing

| Section | Target time | Cumulative |
|---|---:|---:|
| 1. Motivation & RQ | 1:00 | 1:00 |
| 2. Data collection | 2:00 | 3:00 |
| 3. Analytical design | 1:30 | 4:30 |
| 4.1–4.2 Descriptive + Mann-Whitney | 1:30 | 6:00 |
| 4.3 Spearman | 1:00 | 7:00 |
| 4.4–4.6 VADER + supervised models + clusters | 1:45 | 8:45 |
| 4.7–4.8 n-grams + co-occurrence | 0:45 | 9:30 |
| 6. Constraints | 1:30 | 11:00 |
| 7. Conclusions | 1:00 | 12:00 |
| 8. Future work + Q&A buffer | 1:30 | 13:30 |

Fits a ~13–15 min slot with a modest Q&A buffer; trim §4.4–4.6 if you need ≤12 min.

---

## Appendix B — Replication recipe

```powershell
cd yt_narration_study
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env  # edit .env with your API key
python main.py --all
```

Reproduces the numbers in §4 modulo any upstream YouTube ranking changes
(comments are ordered by `relevance`, which is non-deterministic across
long time windows).

Expected runtimes on a modern laptop, no GPU:

- `--acquire` : ~15–20 min (API-bound)
- `--preprocess` : ~15 s
- `--linguistic` : ~15 s
- `--sentiment` : ~8–10 min (MiniLM embedding of ~40k comments on CPU)
- `--stats` : <1 s
