import os
import csv
from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# Load API key from .env
load_dotenv()
API_KEY = os.getenv("YOUTUBE_API_KEY")
youtube = build("youtube", "v3", developerKey=API_KEY)

# Read video IDs from dataset1 CSV
video_ids = []
with open("dataset1_youtube_videos.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        video_ids.append(row["video_id"])

# Write comments to a new CSV

output_file = "dataset2_yt_comments.csv"
fieldnames = ["comment_id", "video_id", "comment_text", "author_name", "like_count", "publish_date"]

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

        # Error handling
        except HttpError as e:
            if e.resp.status == 403 and b"commentsDisabled" in e.content:
                print(f"Comments disabled for video {video_id}, skipping.")
            elif e.resp.status == 404:
                print(f"Video {video_id} not found, skipping.")
            else:
                print(f"HTTP error for video {video_id}: {e}")
        
print(f"Done. Comments saved to {output_file}")
