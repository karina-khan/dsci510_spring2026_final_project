# Load libraries to collect data from API sources

import os
import time
import csv
import sys
import json
import subprocess

from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

csv.field_size_limit(sys.maxsize)
from dotenv import load_dotenv
from nltk.tokenize import sent_tokenize
import nltk
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

# Load YouTube Data API key for retrieving videos from YouTube 
load_dotenv()
API_KEY = os.getenv("YOUTUBE_API_KEY")
youtube = build("youtube", "v3", developerKey=API_KEY)

# Define search parameters
QUERIES = [
    "women AI podcast",
    "women AI health podcast",
    "women AI education podcast",
    "women AI art podcast",
    "women AI safety podcast",
    "women AI relationships podcast",
    "women AI work podcast",
    "women AI finance podcast",
    "women AI wellness podcast",
    "women AI spirituality podcast",
    "women AI career podcast",
    "women AI feminism podcast",
    "women AI equity podcast",
    "women AI technology podcast",
    "women AI creative podcast"
]

YEARS = [2022, 2023, 2024, 2025, 2026]

# DATASET 1: Search for videos 
def search_videos(query, year):
    # Returns up to 25 video results for a given query and year.
    results = youtube.search().list(
        q=query,
        type="video",
        order="viewCount",
        publishedAfter=f"{year}-01-01T00:00:00Z",
        publishedBefore=f"{year}-12-31T23:59:59Z",
        maxResults=25,
        part="snippet",
        relevanceLanguage="en", # filter to English-language results
        videoDuration="long", # "long" = over 20 min (API's closest option to 30 min)
        regionCode="US" # geo-restrict to US (single country code only available)
    ).execute()

    ENGLISH = {"en", "en-US", "en-GB", "en-CA", "en-AU"}

    videos = []
    for item in results.get("items", []):
        snippet = item["snippet"]
        lang = snippet.get("defaultAudioLanguage") or snippet.get("defaultLanguage")
        # keep the video if it's tagged as English, or if no language tag is present
        if lang is not None and lang not in ENGLISH:
            continue
        videos.append({
            "video_id": item["id"]["videoId"],
            "title": snippet["title"],
            "publish_date": snippet["publishedAt"],
            "query": query,
            "year": year
        })
    return videos

# Incorporate view counts
def get_view_counts(video_ids):
    # Takes a list of video IDs, returns a dictionary {video_id: view_count}
    view_counts = {}

    # Process in batches of 50
    for i in range(0, len(video_ids), 50):
        batch = video_ids[i:i+50]
        response = youtube.videos().list(
            part="statistics",
            id=",".join(batch)
        ).execute()

        for item in response.get("items", []):
            vid_id = item["id"]
            view_counts[vid_id] = item["statistics"].get("viewCount", 0)

        time.sleep(0.5) # Slow down the API call

    return view_counts

# # AI-Generated code below. I originally only downloaded 2022-2025 and wanted to append 2026 results:
# # Load existing video IDs to avoid duplicates on append
# output_file = os.path.join(os.path.dirname(__file__), "..", "data", "dataset1_youtube_videos.csv")
# fieldnames  = ["video_id", "title", "publish_date", "query", "year", "view_count"]

# existing_ids = set()
# if os.path.exists(output_file):
#     with open(output_file, newline="", encoding="utf-8") as f:
#         reader = csv.DictReader(f)
#         for row in reader:
#             existing_ids.add(row["video_id"])
# print(f"Found {len(existing_ids)} existing video ID(s) in CSV — will skip these.")


output_file = os.path.join(os.path.dirname(__file__), "..", "data", "dataset1_youtube_videos.csv")
fieldnames  = ["video_id", "title", "publish_date", "query", "year", "view_count"]

all_videos = []

for query in QUERIES:
    for year in YEARS:
        print(f"Searching: '{query}' | {year}")
        results = search_videos(query, year)
        all_videos.extend(results)
        time.sleep(0.5)

# Collect all video IDs
all_ids = [v["video_id"] for v in all_videos]
view_counts = get_view_counts(all_ids)

# Add view count to each video record
for video in all_videos:
    video["view_count"] = view_counts.get(video["video_id"], 0)

# Deduplicate by video_id, keeping first occurrence
seen = set()
deduped_videos = []
for video in all_videos:
    if video["video_id"] not in seen:
        seen.add(video["video_id"])
        deduped_videos.append(video)

print(f"{len(all_videos) - len(deduped_videos)} duplicate(s) removed. "
      f"{len(deduped_videos)} unique video(s) to write.")

with open(output_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(deduped_videos)

print(f"\nDone! {len(deduped_videos)} video(s) saved to {output_file}")

# DATASET 2: Retrieving comments from YouTube Data API

# Read video IDs from dataset1 CSV 
dataset1_path = os.path.join(os.path.dirname(__file__), "..", "data", "dataset1_youtube_videos.csv")
video_ids = []
with open(dataset1_path, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        video_ids.append(row["video_id"])

output_file = os.path.join(os.path.dirname(__file__), "..", "data", "dataset2_yt_comments.csv")
fieldnames = ["comment_id", "video_id", "comment_text", "author_name", "like_count", "publish_date"]

total_comments = 0
total_videos = 0

with open(output_file, "w", newline="", encoding="utf-8") as out_f:
    writer = csv.DictWriter(out_f, fieldnames=fieldnames)
    writer.writeheader()

    for video_id in video_ids:
        try:
            response = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=100,
                order="relevance",
                textFormat="plainText",
            ).execute()

            video_comment_count = 0
            for item in response.get("items", []):
                snippet = item["snippet"]["topLevelComment"]["snippet"]
                writer.writerow({
                    "comment_id": item["id"],
                    "video_id": video_id,
                    "comment_text": snippet.get("textDisplay", ""),
                    "author_name": snippet.get("authorDisplayName", ""),
                    "like_count": snippet.get("likeCount", 0),
                    "publish_date": snippet.get("publishedAt", ""),
                })
                video_comment_count += 1

            if video_comment_count > 0:
                total_comments += video_comment_count
                total_videos += 1

        # AI-Generated code to handle errors for diasbled comments
        except HttpError as e:
            if e.resp.status == 403 and b"commentsDisabled" in e.content:
                print(f"Comments disabled for video {video_id}, skipping.")
            elif e.resp.status == 404:
                print(f"Video {video_id} not found, skipping.")
            else:
                print(f"HTTP error for video {video_id}: {e}")

print(f"Done. Added {total_comments} comments across {total_videos} videos. Saved to {output_file}")

# Use yt-dlp to load auto-generated or manual subtitles as plain text to use as transcript data (my YouTubeTranscipts API didn't work)
# AI-Generated this section to support with the yt-dlp API:

load_dotenv()
cookies_path = os.path.join(os.path.dirname(__file__), "..", "cookies.txt")

# Read video IDs, dates, and queries from Dataset 1
videos = []
with open(os.path.join(os.path.dirname(__file__), "..", "data", "dataset1_youtube_videos.csv"), "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        videos.append({
            "video_id": row["video_id"],
            "publish_date": row["publish_date"],
            "query": row["query"]
        })

# Output CSV
output_file = os.path.join(os.path.dirname(__file__), "..", "data", "dataset3_transcripts.csv")
fieldnames  = ["video_id", "publish_date", "query", "sentence"]

def fetch_transcript(video_id):
    url = f"https://www.youtube.com/watch?v={video_id}"
    cmd = [
        "yt-dlp",
        "--cookies", cookies_path,
        "--write-auto-subs",
        "--write-subs",
        "--sub-langs", "en",
        "--sub-format", "json3",
        "--skip-download",
        "--js-runtimes", "node",
        "--quiet",
        "-o", "/tmp/yt_transcript_%(id)s",
        url,
    ]
    # Run yt-dlp
    subprocess.run(cmd, capture_output=True)

    # yt-dlp writes subtitle file to /tmp/yt_transcript_<id>.en.json3
    sub_file = f"/tmp/yt_transcript_{video_id}.en.json3"
    if not os.path.exists(sub_file):
        return None

    with open(sub_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    os.remove(sub_file)

    # Extract text from json3 format
    words = []
    for event in data.get("events", []):
        for seg in event.get("segs", []):
            text = seg.get("utf8", "").strip()
            if text and text != "\n":
                words.append(text)

    return " ".join(words).strip() if words else None


new_sentences = 0
new_videos = 0

with open(output_file, "w", newline="", encoding="utf-8") as out_f:
    writer = csv.DictWriter(out_f, fieldnames=fieldnames)
    writer.writeheader()

    for video in videos:
        video_id     = video["video_id"]
        publish_date = video["publish_date"]
        query        = video["query"]

        try:
            full_text = fetch_transcript(video_id)
            if not full_text:
                print(f"✗ {video_id}: no transcript found")
                continue

            sentences = sent_tokenize(full_text)

            for sentence in sentences:
                writer.writerow({
                    "video_id":     video_id,
                    "publish_date": publish_date,
                    "query":        query,
                    "sentence":     sentence.strip()
                })

            new_sentences += len(sentences)
            new_videos += 1
            print(f"{video_id} — {len(sentences)} sentences")

        except Exception as e:
            print(f"X {video_id}: {e}")

        time.sleep(5)

print(f"\nDone! Added {new_sentences} new sentences from {new_videos} new videos to {output_file}")
