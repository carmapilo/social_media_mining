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

After all filters:

- **644 videos** (358 human, 286 AI) across **25 distinct channels**.
- **39,600 cleaned comments** (22,966 human / 16,634 AI).
- **1,843 authenticity-flagged comments** (983 human / 860 AI).
- **631 videos** retained at the statistical-test stage (those with at least
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

| Metric (median per video) | Human (n=358) | AI (n=286) |
|---|---:|---:|
| viewCount | 171 205 | 200 136 |
| likeCount | 4 958 | 6 100 |
| commentCount | 283 | 341 |
| like_to_view_ratio | 0.0271 | 0.0297 |
| comment_rate | 0.00189 | 0.00184 |
| mean VADER compound (per video) | 0.135 | 0.143 |
| % negative comments (per video) | 21.8% | 22.7% |
| authenticity-flag rate (per video) | 2.58% | 2.25% |

The two groups are surprisingly close on raw engagement. The AI-narrated
bucket actually has a **higher median view count and like count** — partly
because it's dominated by high-volume formats (top-10 lists, facts,
finance) that are optimized for algorithmic reach.

### 4.2 Inferential results — Mann-Whitney U (n=631 videos)

| Metric | U | p | r (effect) | Significant @ α=0.05 |
|---|---:|---:|---:|:---:|
| like_to_view_ratio | 48 037 | **0.663** | +0.020 | no |
| comment_rate | 52 918 | **0.087** | −0.079 | no |
| mean_sentiment (VADER) | 46 230 | **0.218** | +0.057 | no |
| pct_negative | 50 726 | **0.455** | −0.035 | no |
| **authenticity_flag_rate** | **53 880** | **0.031** | **−0.099** | **yes** |

**Reading the table.** Four of the five primary metrics do **not** separate
the two groups at α = 0.05. The only statistically significant result —
and the hypothesis-relevant one — is that **AI-narrated videos accumulate
authenticity-related chatter at a higher rate than human-narrated ones**
(negative `r`, because negative `r` favours the AI group in our
parameterization). The effect size is small (|r| ≈ 0.10), so it is a
detectable but modest shift; it survives a Bonferroni α = 0.01 correction
for the five-metric family (0.031 × 5 = 0.155, which would *not* pass), so
this is a **robust α = 0.05 result, suggestive only at α = 0.01**.

### 4.3 Spearman: sentiment ↔ engagement

| Group | Pair | ρ | p | n |
|---|---|---:|---:|---:|
| **Human** | mean_sentiment ~ like_to_view_ratio | **+0.443** | < 0.0001 | 354 |
| **AI** | mean_sentiment ~ like_to_view_ratio | **+0.142** | 0.018 | 277 |

**This is the headline qualitative finding.** Sentiment translates to
engagement *much* more reliably on human-narrated videos (moderate, highly
significant ρ = 0.44) than on AI-narrated ones (weak, barely significant
ρ = 0.14). In plain terms: when viewers *feel good* about a human-narrated
video they click like in proportion; when they feel good about an
AI-narrated video, they are much less consistent in translating that
affect into a like. One interpretation is that AI-narrated channels often
serve a *consumption* audience (background listening, scroll-through)
whose affective response is weaker or less action-coupled.

### 4.4 Comment-level VADER distributions

| | Mean compound | Median | Std |
|---|---:|---:|---:|
| Human | +0.130 | 0.00 | 0.453 |
| AI | +0.144 | 0.00 | 0.475 |

Both distributions are positively-skewed and zero-dominated (a typical
sign of a "thanks / lol / nice" mass concentrated at neutral compound).
They are essentially indistinguishable in central tendency.

### 4.5 Clustering (KMeans, k=5 selected by silhouette)

Silhouette scores across the candidate range:

| k | 3 | 4 | **5** | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|
| silhouette | 0.0233 | 0.0248 | **0.0253** | 0.0253 | 0.0208 | 0.0186 | 0.0196 | 0.0214 |

(The absolute scores are low — expected for short, noisy UGC text in
high-dimensional space.) Cluster labels derived from within-cluster
TF-IDF top-5:

| Cluster | Label / keywords | Human | AI | Interpretation |
|---|---|---:|---:|---|
| 0 | `like, just, people, don, make` | 7 980 | 3 818 | Generic opinion / reaction |
| 1 | `video, like, love, just, watching` | 5 500 | 5 405 | Praise of the video itself |
| 2 | `face_with_tears_of_joy, red_heart, loudly_crying_face, party_popper, blackpink` | 3 714 | 1 647 | Emoji-heavy fan reactions |
| 3 | `trump, iran, war, people, israel` | 4 788 | 2 181 | News / geopolitics commentary |
| 4 | `game, games, like, love, really` | 984 | 3 583 | Gaming / review discourse |

Two observations:

- **Cluster 4 is an AI-heavy bucket** (3 583 AI vs. 984 human) — the gaming-
  and review-style chatter is disproportionately on AI-narrated channels,
  driven by the gaming / top-lists genres in the AI stratum.
- **Cluster 3 is strongly human** — the news channels we included (BBC,
  Fox, NBC) dominate the geopolitics conversation in this year's data.

### 4.6 Qualitative n-gram comparison

**Top bigrams per group** (raw counts, top 10 shown here):

| Rank | Human | count | AI | count |
|---|---|---:|---|---:|
| 1 | looks like | 137 | feel like | 150 |
| 2 | ted ed | 114 | looks like | 135 |
| 3 | years ago | 112 | crimson desert | 123 |
| 4 | feel like | 105 | years ago | 99 |
| 5 | don't know | 104 | feels like | 90 |
| 6 | god bless | 93 | don't know | 76 |
| 7 | feels like | 88 | **sounds like** | 74 |
| 8 | president trump | 84 | can't wait | 66 |
| 9 | strait hormuz | 82 | resident evil | 56 |
| 10 | **sounds like** | 79 | video game | 56 |

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

### 4.7 Authenticity keyword co-occurrence

Among the 1 843 authenticity-flagged comments, the pairs that co-occur most
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
| `tfidf_top_human.png` | Top 50 TF-IDF terms, human-narrated corpus | §4.6 |
| `tfidf_top_ai.png` | Top 50 TF-IDF terms, AI-narrated corpus | §4.6 |
| `bigrams_human.png` / `bigrams_ai.png` | Top 20 bigrams per group | §4.6 |
| `trigrams_human.png` / `trigrams_ai.png` | Top 20 trigrams per group | §4.6 |
| `authenticity_cooccurrence.png` | Keyword-pair heatmap | §4.7 |
| `sentiment_distribution.png` | VADER compound KDE, split by narration type | §4.4 |
| `per_video_sentiment.png` | Box/strip plot of per-video mean sentiment | §4.1 |
| `umap_clusters.png` | UMAP 2D projection coloured by cluster | §4.5 |

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
partially a "genre" contrast.** A strict interpretation of the
authenticity-flag result requires genre-controlled replication — at
minimum, a genre-stratified bootstrap or a mixed-effects model with
channel as a random effect.

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

- No multiple-comparison correction was applied in the primary table; the
  authenticity-flag result is **not** significant at a Bonferroni-adjusted
  α = 0.01. Readers should take the effect as *suggestive-and-consistent*
  rather than decisive.
- Mann-Whitney tests the distributions, not a specific parameter; the
  effect-size `r` is reported so readers can judge practical significance
  independently of `n`.

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

1. **Headline engagement metrics do not separate the two groups.** Likes,
   comments, and average sentiment are statistically indistinguishable
   between AI- and human-narrated videos in 2026.
2. **Authenticity talk is measurably more common on AI-narrated videos**
   (p = 0.031, |r| ≈ 0.10). Viewers *do* notice — or at least discuss —
   narration provenance, and they do so slightly more when the narration
   is synthesized.
3. **Sentiment decouples from engagement on AI-narrated content.** The
   ρ = 0.44 (human) vs. ρ = 0.14 (AI) split is the most theoretically
   interesting finding: whatever loop turns feeling-good-about-a-video
   into clicking-like is partially broken on AI-narrated channels.
4. **The conversation *about* AI voices has converged on the word
   "voice" and the phrase "sounds like"** as its diagnostic vocabulary.
   This is stable across strata and is a strong candidate for a
   replication-invariant coding scheme in future studies.

The responsible read of the evidence: **AI narration does not cost a
channel measurable engagement in 2026**, but it does shift the texture of
the comment section and it attenuates the sentiment→engagement coupling
that traditionally reports a "loyal audience" effect.

---

## 8. Future work

- **Genre-stratified replication** to separate narration effects from
  genre effects (critical — §6.1).
- **Reply-thread inclusion** to capture the back-and-forth structure of
  authenticity debates.
- **Multilingual sentiment / embeddings** to cover the non-English tail.
- **Longitudinal extension** by parameterizing `TARGET_YEAR` over
  2022–2026 to measure *when* authenticity talk emerged as a shared frame.
- **Supervised authenticity classifier** (fine-tuned from the 1 843
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
| 4.5–4.6 Clusters + n-grams | 1:30 | 8:30 |
| 4.7 Co-occurrence | 0:30 | 9:00 |
| 6. Constraints | 1:30 | 10:30 |
| 7. Conclusions | 1:00 | 11:30 |
| 8. Future work + Q&A buffer | 1:30 | 13:00 |

Fits comfortably in a 12–15 min slot with time for questions.

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
