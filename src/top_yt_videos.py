import os
import time
import csv
from dotenv import load_dotenv
from googleapiclient.discovery import build

# ── Load API key from .env ──────────────────────────────────────────
load_dotenv()
API_KEY = os.getenv("YOUTUBE_API_KEY")
youtube = build("youtube", "v3", developerKey=API_KEY)

# ── Define your search parameters ──────────────────────────────────
QUERIES = [
    "AI women health",
    "AI women careers",
    "AI women culture"
]

YEARS = [2022, 2023, 2024, 2025]

# ── Step 1: Search for videos ───────────────────────────────────────
def search_videos(query, year):
    """Returns up to 25 video results for a given query and year."""
    results = youtube.search().list(
        q=query,
        type="video",
        order="viewCount",
        publishedAfter=f"{year}-01-01T00:00:00Z",
        publishedBefore=f"{year}-12-31T23:59:59Z",
        maxResults=25,
        part="snippet"
    ).execute()

    videos = []
    for item in results.get("items", []):
        videos.append({
            "video_id":     item["id"]["videoId"],
            "title":        item["snippet"]["title"],
            "publish_date": item["snippet"]["publishedAt"],
            "query":        query,
            "year":         year
        })
    return videos

# ── Step 2: Enrich with view counts ────────────────────────────────
def get_view_counts(video_ids):
    """Takes a list of video IDs, returns a dict of {video_id: view_count}."""
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
            views  = item["statistics"].get("viewCount", 0)
            view_counts[vid_id] = views
        
        time.sleep(0.5)  # Be polite to the API
    
    return view_counts

# ── Run everything ──────────────────────────────────────────────────
all_videos = []

for query in QUERIES:
    for year in YEARS:
        print(f"Searching: '{query}' | {year}")
        results = search_videos(query, year)
        all_videos.extend(results)
        time.sleep(0.5)

# Collect all video IDs and enrich with view counts
all_ids    = [v["video_id"] for v in all_videos]
view_counts = get_view_counts(all_ids)

# Add view count to each video record
for video in all_videos:
    video["view_count"] = view_counts.get(video["video_id"], 0)

# ── Save to CSV ─────────────────────────────────────────────────────
output_file = "dataset1_youtube_videos.csv"
fieldnames  = ["video_id", "title", "publish_date", "query", "year", "view_count"]

with open(output_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(all_videos)

print(f"\nDone! {len(all_videos)} videos saved to {output_file}")