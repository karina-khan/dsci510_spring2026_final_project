import os
import csv
import time
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
from nltk.tokenize import sent_tokenize
import nltk
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

# ── Load API key from .env ──────────────────────────────────────────
load_dotenv()

# ── Instantiate the API (new syntax for v1.0+) ──────────────────────
ytt_api = YouTubeTranscriptApi()

# ── Read video IDs, dates, and queries from Dataset 1 ──────────────
videos = []
with open("dataset1_youtube_videos.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        videos.append({
            "video_id":     row["video_id"],
            "publish_date": row["publish_date"],
            "query":        row["query"]
        })

# ── Output CSV setup ────────────────────────────────────────────────
output_file = "dataset3_transcripts.csv"
fieldnames  = ["video_id", "publish_date", "query", "sentence"]

with open(output_file, "w", newline="", encoding="utf-8") as out_f:
    writer = csv.DictWriter(out_f, fieldnames=fieldnames)
    writer.writeheader()

    for video in videos:
        video_id     = video["video_id"]
        publish_date = video["publish_date"]
        query        = video["query"]

        try:
            # Step 1: Fetch transcript chunks (new syntax)
            transcript = ytt_api.fetch(video_id)

            # Step 2: Join all chunks into one big string
            full_text = " ".join([chunk.text for chunk in transcript])

            # Step 3: Split into sentences
            sentences = sent_tokenize(full_text)

            # Step 4: Write one row per sentence
            for sentence in sentences:
                writer.writerow({
                    "video_id":     video_id,
                    "publish_date": publish_date,
                    "query":        query,
                    "sentence":     sentence.strip()
                })

            print(f"✓ {video_id} — {len(sentences)} sentences")

        except Exception as e:
            print(f"✗ Error for {video_id}: {e}")

        time.sleep(0.5)

print(f"\nDone! Transcript sentences saved to {output_file}")