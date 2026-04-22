import os
from dotenv import load_dotenv
from googleapiclient.discovery import build

load_dotenv()

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "..", "data")

# API
API_KEY = os.getenv("YOUTUBE_API_KEY")
youtube = build("youtube", "v3", developerKey=API_KEY)

# Input datasets (load.py)
dataset1_path = os.path.join(DATA_DIR, "dataset1_youtube_videos.csv")
dataset2_path = os.path.join(DATA_DIR, "dataset2_yt_comments.csv")
dataset3_path = os.path.join(DATA_DIR, "dataset3_transcripts.csv")

# Output datasets (main.py)
filtered_comments_path    = os.path.join(DATA_DIR, "filtered_comments.csv")
filtered_transcripts_path = os.path.join(DATA_DIR, "filtered_transcripts.csv")

# yt-dlp
cookies_path         = os.path.join(BASE_DIR, "..", "cookies.txt")
transcripts_tmp_path = "/tmp/yt_transcript_%(id)s"