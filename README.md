# Sentiment Toward AI in Women's Media

An analysis of long-form YouTube videos and comments centred around women, across the timeline of the AI boom.

## Introduction

Studies show that women adopt AI at rates 25% lower than men. Call it the "ick," but some claim the difference is rooted in how AI systems are designed with feminized personas that perform submissiveness (Awomosu, 2026). This project investigates how women's media frames artificial intelligence throughout the AI boom.

**Research question:** What is the sentiment of artificial intelligence in long-form YouTube videos and comments centered around women, from 2022–2026? And how does this sentiment vary across domains?

## Data Sources

1. **YouTube Data API for Video Metadata**
   Used to produce a primary dataset of 400+ long-form YouTube videos. Videos were retrieved via search.list queries by keyword and year, then enriched with full metadata (view count, duration, etc.) via batched videos.list calls.

2. **YouTube Data API for Comments**
   Used to collect the top 100 most relevant comments from each video in the primary dataset. Retrieved via commentThreads.list called per video, sorted by relevance and capped at 100 comments per video.

3. **yt-dlp for Transcripts**
   Used as a fallback method to capture raw transcript text from videos in the primary dataset. Called via Python's subprocess.run(), fetching auto-generated or manual English subtitles in json3 format, then tokenized into sentences using NLTK's sent_tokenize().

## Analysis

This project applies VADER (Valence Aware Dictionary and sEntiment Reasoner) sentiment scoring to both the comment and transcript datasets. Each sentence or comment receives a compound sentiment score ranging from -1 (most negative) to +1 (most positive). Results are analyzed across ten thematic domains (health, finance, wellbeing, safety, politics, creativity, education, career, community, and spirituality) using a custom use-case lexicon to tag and filter relevant content. Keyword frequency visualization is used to surface dominant AI framings over the 2022–2026 period.

To investigate the framing shift observed around 2024–2025, LDA (Latent Dirichlet Allocation) topic modeling was also applied to identify the top 3 topics per year. This revealed a notable shift from relational to institutional language in how AI is discussed in women's media after 2024.

> *Note: Sections labeled "AI generated code" in the source files were produced with assistance from Claude (Anthropic).*

## Summary of Results

Surprisingly, sentiment across both transcripts and comments remained above neutral throughout the full study period. However, several notable trends emerged:

- **Comments:** Average sentiment declines steadily over time.
- **Transcripts:** Average sentiment increases leading up to 2024, then drops in 2025.
- **Across all domains:** A significant decrease in average sentiment score occurs after 2024.
- **Relationships domain:** Conversations around relationships experience the greatest drop in sentiment from 2024 to 2025.

## How to Run

### Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

You will also need:
- A **YouTube Data API v3** key — store it as YOUTUBE_API_KEY in a .env file inside src/. See .env.example for the template.
- **yt-dlp** installed: pip install yt-dlp
- A cookies.txt file in the project root for yt-dlp authentication

> Do **not** commit your .env file or cookies.txt to the repository. Both are listed in .gitignore.

### Running the Main Pipeline

From the src/ directory:

bash
cd src
python main.py

This will:
1. Filter and tag comments and transcripts with thematic categories
2. Run VADER sentiment scoring on both datasets
3. Save filtered_comments.csv and filtered_transcripts.csv to data/
4. Generate all visualizations

### Optional: Exploratory Analysis Notebook

Open and run results_test.ipynb in Jupyter for full exploratory analysis and additional visualizations.