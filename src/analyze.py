# Pipelines for Dataset 2 (Comments) and Dataset 3 (Transcripts):
# Comments:
#   1. Filter to comments that contain AI keywords
#   2. Tag each comment with thematic categories based on keyword matching
#   3. Run VADER sentiment scoring on tagged comments
#   4. Save results to filtered_comments.csv
#
# Transcripts:
#   1. Re-tokenize any rows that weren't properly split into sentences
#   2. Filter to sentences that contain AI keywords
#   3. Tag each sentence with thematic categories based on keyword matching
#   4. Run VADER sentiment scoring on tagged sentences
#   5. Save results to filtered_transcripts.csv

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from process import contains_ai_keyword, get_categories, retokenize_transcripts
from config import filtered_comments_path, filtered_transcripts_path, dataset2_path, dataset3_path


def run_vader(df, text_col):
    # Add VADER sentiment columns to df in place.
    analyzer = SentimentIntensityAnalyzer()
    scores = [analyzer.polarity_scores(str(t)) for t in df[text_col]]
    df["vader_negative"] = [s["neg"] for s in scores]
    df["vader_neutral"]  = [s["neu"] for s in scores]
    df["vader_positive"] = [s["pos"] for s in scores]
    df["vader_score"]    = [s["compound"] for s in scores]


# Dataset 2: Comments

def process_comments():
    print("=" * 50)
    print("DATASET 2: Comments")
    print("=" * 50)

    print("Loading dataset2_yt_comments.csv...")
    df = pd.read_csv(dataset2_path)
    print(f"  Original rows: {len(df)}")

    # Step 1: Filter to comments containing AI keywords
    print("\nStep 1: Filtering for AI keywords...")
    df_ai = df[df["comment_text"].apply(contains_ai_keyword)].copy()
    print(f"  Comments mentioning AI: {len(df_ai)}")

    if df_ai.empty:
        print("  No AI-related comments found. Check your data or keywords.")
        return

    # Step 2: Tag with thematic categories
    print("\nStep 2: Tagging comments with thematic categories...")
    df_ai["categories"] = df_ai["comment_text"].apply(get_categories)
    print(f"  Comments with at least one category match: "
          f"{(df_ai['categories'] != 'uncategorized').sum()}")
    print(f"  Uncategorized: {(df_ai['categories'] == 'uncategorized').sum()}")

    # Step 3: VADER sentiment scoring
    print("\nStep 3: Running VADER sentiment scoring...")
    run_vader(df_ai, "comment_text")

    # Step 4: Save output
    output_cols = [
        "video_id", "comment_text", "author_name", "like_count", "publish_date",
        "categories", "vader_negative", "vader_neutral", "vader_positive", "vader_score"
    ]
    df_ai[output_cols].to_csv(filtered_comments_path, index=False)

    print(f"\n✓ Done! Saved to filtered_comments.csv")
    print(f"  Total AI-related comments: {len(df_ai)}")
    print(f"\nTop category matches:")
    print(df_ai["categories"].value_counts().head(10).to_string())
    print(f"\nVADER score distribution:")
    print(df_ai["vader_score"].describe().round(3).to_string())


# Dataset 3: Transcripts 

def process_transcripts():
    print("=" * 50)
    print("DATASET 3: Transcripts")
    print("=" * 50)

    print("Loading dataset3_transcripts.csv...")
    df = pd.read_csv(dataset3_path)
    print(f"  Original rows: {len(df)}")

    # Step 1: Re-tokenize oversized rows (only rows > 500 chars need splitting)
    print("\nStep 1: Re-tokenizing long transcript rows...")
    df_expanded = retokenize_transcripts(df)
    print(f"  Total rows after re-tokenization: {len(df_expanded)}")

    # Step 2: Filter to sentences containing AI keywords
    print("\nStep 2: Filtering for AI keywords...")
    df_ai = df_expanded[df_expanded["sentence"].apply(contains_ai_keyword)].copy()
    print(f"  Sentences mentioning AI: {len(df_ai)}")

    if df_ai.empty:
        print("  No AI-related sentences found. Check your data or keywords.")
        return

    # Step 3: Tag with thematic categories
    print("\nStep 3: Tagging sentences with thematic categories...")
    df_ai["categories"] = df_ai["sentence"].apply(get_categories)
    print(f"  Sentences with at least one category match: "
          f"{(df_ai['categories'] != 'uncategorized').sum()}")
    print(f"  Uncategorized: {(df_ai['categories'] == 'uncategorized').sum()}")

    # Step 4: VADER sentiment scoring
    print("\nStep 4: Running VADER sentiment scoring...")
    run_vader(df_ai, "sentence")

    # Step 5: Save output
    output_cols = [
        "video_id", "publish_date", "query", "sentence",
        "categories", "vader_negative", "vader_neutral", "vader_positive", "vader_score"
    ]
    df_ai[output_cols].to_csv(filtered_transcripts_path, index=False)

    print(f"\n✓ Done! Saved to filtered_transcripts.csv")
    print(f"  Total AI-related sentences: {len(df_ai)}")
    print(f"\nTop category matches:")
    print(df_ai["categories"].value_counts().head(10).to_string())
    print(f"\nVADER compound score distribution:")
    print(df_ai["vader_score"].describe().round(3).to_string())



def main():
    process_comments()
    print()
    process_transcripts()


if __name__ == "__main__":
    main()
