# AI-Generated code for visualizations after loading libraries
# Produces four charts for the final project:
#   1. Combined Data Frequency per Year (All Domains)
#   2. Sentiment per Year (All Domains Combined)
#   3. Average Sentiment per Domain per Year (Grouped Bar)
#   4. LDA Topic Clusters: Careers — 2024 & 2025 Transcripts Only

import pathlib
import numpy as np

RESULTS_DIR = pathlib.Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

from config import filtered_comments_path, filtered_transcripts_path

#constants 

DOMAINS = [
    "careers", "creativity", "education", "relationships", "safety",
    "politics", "health", "spirituality", "wellness", "finance"
]

PALETTE_10 = [
    "#0e0f44", "#1e2070", "#3a3ea0", "#6667c4", "#9899d8",
    "#cc99b8", "#f07070", "#ff9999", "#ffbbbb", "#ffe3e3",
]

YEAR_COLORS = ["#3a3ea0", "#6667c4", "#9899d8", "#ff9999", "#ff4444"]

_BG_DARK  = "#000000"
_BG_LIGHT = "#f4e7eb"
_TEXT     = "#ffe3e3"
_SPINE    = "#3a3ea0"
_NEUTRAL  = "#3a3ea0"

_N_TOPICS    = 3
_N_TOP_WORDS = 8
_TARGET_YEARS = [2024, 2025]
_TOPIC_PALETTE = ["#0e0f44", "#a0446e", "#ff9999"]

STOP_WORDS = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you",
    "your", "yours", "yourself", "yourselves", "he", "him", "his",
    "himself", "she", "her", "hers", "herself", "it", "its", "itself",
    "they", "them", "their", "theirs", "themselves", "what", "which",
    "who", "whom", "this", "that", "these", "those", "am", "is", "are",
    "was", "were", "be", "been", "being", "have", "has", "had", "having",
    "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if",
    "or", "because", "as", "until", "while", "of", "at", "by", "for",
    "with", "about", "against", "between", "into", "through", "during",
    "before", "after", "above", "below", "to", "from", "up", "down",
    "in", "out", "on", "off", "over", "under", "again", "further",
    "then", "once", "here", "there", "when", "where", "why", "how",
    "all", "both", "each", "few", "more", "most", "other", "some",
    "such", "no", "nor", "not", "only", "own", "same", "so", "than",
    "too", "very", "can", "will", "just", "should", "now", "any",
    "also", "well", "back", "even", "still", "way", "every", "never",
    "re", "ve", "ll", "isn", "aren", "wasn", "weren", "hasn", "hadn",
    "doesn", "don", "didn", "won", "wouldn", "couldn", "shouldn",
    "ai", "a.i", "artificial", "intelligence", "like", "think", "know",
    "really", "going", "want", "make", "use", "using", "used",
    "get", "got", "one", "would", "could", "right", "yeah", "okay",
    "thing", "things", "people", "much", "many", "lot", "go", "say",
    "said", "see", "time", "mean", "actually", "something", "everything",
    "anything", "need", "take", "look", "come", "new", "good", "great",
    "already", "around", "another", "always", "maybe", "might", "though",
    "across", "without", "within", "yes", "let", "put", "try", "help",
    "tell", "talk", "feel", "feels", "felt", "give", "gave", "given",
    "keep", "kept", "start", "started", "kind", "sort", "bit", "little",
    "big", "long", "real", "sure", "absolutely", "definitely", "probably",
    "basically",
    "um", "uh", "us", "oh", "ah", "hmm", "yeah", "yep", "nope",
}


#  Data loading 

def load_data():
    df_t = pd.read_csv(filtered_transcripts_path)
    df_c = pd.read_csv(filtered_comments_path)
    df_t = df_t.rename(columns={"sentence": "text"})
    df_c = df_c.rename(columns={"comment_text": "text"})
    df_t["year"] = df_t["publish_date"].astype(str).str[:4].astype(int)
    df_c["year"] = df_c["publish_date"].astype(str).str[:4].astype(int)
    print(f"Transcripts: {len(df_t):,} rows  |  Comments: {len(df_c):,} rows")
    return df_t, df_c


# Chart 1: Combined Data Frequency per Year 

def plot_data_frequency(df_t, df_c):
    df_t_tagged = df_t.copy(); df_t_tagged["source"] = "transcript"
    df_c_tagged = df_c.copy(); df_c_tagged["source"] = "comment"
    df_combined = pd.concat([df_t_tagged, df_c_tagged], ignore_index=True)

    rows = []
    for domain in DOMAINS:
        mask = df_combined["categories"].astype(str).str.contains(domain, na=False)
        for year, count in df_combined[mask].groupby("year").size().items():
            rows.append({"year": year, "domain": domain, "count": count})

    pivot = (
        pd.DataFrame(rows)
        .pivot(index="year", columns="domain", values="count")
        .fillna(0)
    )
    years_all = sorted(pivot.index)
    x_labels  = [str(y) for y in years_all]

    fig, ax = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor(_BG_DARK)
    ax.set_facecolor(_BG_DARK)

    bottom = np.zeros(len(years_all))
    for i, domain in enumerate(DOMAINS):
        vals = [pivot.loc[y, domain] if y in pivot.index else 0 for y in years_all]
        ax.bar(x_labels, vals, bottom=bottom, label=domain.capitalize(),
               color=PALETTE_10[i], edgecolor=_BG_DARK, linewidth=0.6)
        bottom += np.array(vals)

    totals = [pivot.loc[y].sum() if y in pivot.index else 0 for y in years_all]
    for xi, total in enumerate(totals):
        ax.text(xi, total + max(totals) * 0.01, f"{int(total):,}",
                ha="center", va="bottom", color=_TEXT, fontsize=8.5, fontweight="bold")

    ax.set_ylim(0, max(totals) * 1.08)
    ax.set_title("Combined AI Podcast Data Volume per Year\n(Transcripts + Comments — All Domains)",
                 fontweight="bold", color=_TEXT, fontsize=13, pad=14)
    ax.set_xlabel("Year", color=_TEXT, fontsize=11)
    ax.set_ylabel("Number of Domain Mentions", color=_TEXT, fontsize=11)
    ax.tick_params(axis="both", colors=_TEXT)
    for spine in ax.spines.values():
        spine.set_edgecolor(_SPINE)

    legend = ax.legend(title="Domains", bbox_to_anchor=(1.01, 1), loc="upper left",
                       fontsize=9, title_fontsize=10, facecolor="#0e0f44",
                       edgecolor=_SPINE, labelcolor=_TEXT)
    legend.get_title().set_color("#ff9999")

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "data_frequency.png", dpi=150, bbox_inches="tight")
    plt.show()


# Chart 2: Sentiment per Year (All Domains Combined)

def plot_sentiment_by_year(df_t, df_c):
    sent_t_year = df_t.groupby("year")["vader_score"].mean()
    sent_c_year = df_c.groupby("year")["vader_score"].mean()

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor(_BG_DARK)
    ax.set_facecolor(_BG_DARK)

    ax.plot(sent_t_year.index, sent_t_year.values, "o-", color="#ff9999",
            linewidth=2.5, markersize=8,
            markerfacecolor="#ff9999", markeredgecolor=_TEXT, markeredgewidth=1.2,
            label="Transcripts")
    ax.plot(sent_c_year.index, sent_c_year.values, "s-", color="#6667c4",
            linewidth=2.5, markersize=8,
            markerfacecolor="#6667c4", markeredgecolor="#9899d8", markeredgewidth=1.2,
            label="Comments")
    ax.axhline(0, color=_NEUTRAL, linewidth=1, linestyle="--", alpha=0.75, label="Neutral (0)")

    all_vals = pd.concat([sent_t_year, sent_c_year])
    y_pad = (all_vals.max() - all_vals.min()) * 0.2
    ax.set_ylim(all_vals.min() - y_pad, all_vals.max() + y_pad)

    for year, val in sent_t_year.items():
        ax.annotate(f"{val:.2f}", (year, val),
                    textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=7.5, color="#ffbbbb")
    for year, val in sent_c_year.items():
        ax.annotate(f"{val:.2f}", (year, val),
                    textcoords="offset points", xytext=(0, -14),
                    ha="center", fontsize=7.5, color="#9899d8")

    ax.set_title("Average VADER Sentiment per Year — Transcripts vs Comments\n(All Domains Combined)",
                 fontweight="bold", color=_TEXT, fontsize=13, pad=14)
    ax.set_xlabel("Year", color=_TEXT, fontsize=11)
    ax.set_ylabel("Average VADER Compound Score", color=_TEXT, fontsize=11)
    ax.tick_params(axis="both", colors=_TEXT)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    for spine in ax.spines.values():
        spine.set_edgecolor(_SPINE)

    ax.legend(facecolor="#0e0f44", edgecolor=_SPINE, labelcolor=_TEXT, fontsize=10)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "sentiment_by_year.png", dpi=150, bbox_inches="tight")
    plt.show()


# Chart 3: Average Sentiment per Domain per Year (Grouped Bar)

def plot_grouped_sentiment(df_t, df_c):
    rows = []
    for domain in DOMAINS:
        mask_t = df_t["categories"].astype(str).str.contains(domain, na=False)
        mask_c = df_c["categories"].astype(str).str.contains(domain, na=False)
        combined = pd.concat([
            df_t.loc[mask_t, ["year", "vader_score"]],
            df_c.loc[mask_c, ["year", "vader_score"]],
        ])
        for year, grp in combined.groupby("year"):
            rows.append({"domain": domain.capitalize(), "year": int(year),
                         "avg_sentiment": grp["vader_score"].mean()})

    df_grouped = pd.DataFrame(rows)
    years_g    = sorted(df_grouped["year"].unique())
    domains_g  = [d.capitalize() for d in DOMAINS]
    n_years    = len(years_g)
    n_domains  = len(domains_g)
    colors     = YEAR_COLORS[:n_years]

    fig, ax = plt.subplots(figsize=(16, 6))
    fig.patch.set_facecolor(_BG_DARK)
    ax.set_facecolor(_BG_DARK)

    bar_width = 0.8 / n_years
    x = np.arange(n_domains)

    for i, (year, color) in enumerate(zip(years_g, colors)):
        subset = df_grouped[df_grouped["year"] == year].set_index("domain")
        vals   = [subset.loc[d, "avg_sentiment"] if d in subset.index else np.nan
                  for d in domains_g]
        offset = (i - n_years / 2 + 0.5) * bar_width
        ax.bar(x + offset, vals, width=bar_width, color=color, label=str(year),
               edgecolor=_BG_DARK, linewidth=0.5, zorder=3)

    ax.axhline(0, color=_TEXT, linewidth=1, linestyle="--", alpha=0.6,
               label="Neutral (0)", zorder=2)

    ax.set_xticks(x)
    ax.set_xticklabels(domains_g, color=_TEXT, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks(np.arange(0, 1.1, 0.2))
    ax.tick_params(axis="y", colors=_TEXT, labelsize=9)
    ax.tick_params(axis="x", colors=_TEXT)
    ax.set_title("Average Sentiment Score per Domain per Year",
                 color=_TEXT, fontsize=14, fontweight="bold", pad=14)
    ax.set_xlabel("Domain", color=_TEXT, fontsize=11, labelpad=8)
    ax.set_ylabel("Avg VADER Compound Score", color=_TEXT, fontsize=11, labelpad=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(_SPINE)
    ax.yaxis.grid(True, color="#1e2070", linewidth=0.5, linestyle=":", zorder=0)
    ax.set_axisbelow(True)

    legend = ax.legend(title="Year", bbox_to_anchor=(1.01, 1), loc="upper left",
                       facecolor="#0e0f44", edgecolor=_SPINE,
                       labelcolor=_TEXT, fontsize=9, title_fontsize=10)
    legend.get_title().set_color("#ff9999")

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "grouped_sentiment.png", dpi=150, bbox_inches="tight")
    plt.show()


# Chart 4: LDA Topic Clusters — Careers 2024 & 2025 Transcripts

def _run_lda_year(docs):
    if len(docs) < _N_TOPICS:
        return None
    vec = CountVectorizer(stop_words=list(STOP_WORDS), max_features=500, min_df=2)
    try:
        dtm = vec.fit_transform(docs)
    except ValueError:
        return None
    lda = LatentDirichletAllocation(n_components=_N_TOPICS, random_state=42)
    lda.fit(dtm)
    vocab = vec.get_feature_names_out()
    topics = []
    for comp in lda.components_:
        top_indices = comp.argsort()[: -(_N_TOP_WORDS + 1): -1]
        topics.append([vocab[i] for i in top_indices])
    return topics


def plot_lda_careers(df_t):
    mask_careers = df_t["categories"].astype(str).str.contains("careers", na=False)
    df_careers   = df_t.loc[mask_careers & df_t["year"].isin(_TARGET_YEARS), ["year", "text"]]

    docs_by_year = {
        yr: df_careers.loc[df_careers["year"] == yr, "text"].astype(str).tolist()
        for yr in _TARGET_YEARS
    }

    fig, axes = plt.subplots(1, len(_TARGET_YEARS),
                              figsize=(6.5 * len(_TARGET_YEARS), 11),
                              constrained_layout=True)
    fig.patch.set_facecolor(_BG_LIGHT)
    fig.suptitle(
        f"LDA Topic Clusters — Careers  |  2024 & 2025 Transcripts\n"
        f"(top {_N_TOPICS} topics, {_N_TOP_WORDS} words each)",
        fontsize=14, fontweight="bold", color="#0e0f44",
    )

    for col, year in enumerate(_TARGET_YEARS):
        ax   = axes[col]
        ax.set_facecolor(_BG_LIGHT)
        docs = docs_by_year.get(year, [])
        topics = _run_lda_year(docs)

        ax.set_title(f"{year}  ({len(docs)} docs)",
                     fontsize=11, fontweight="bold", color="#0e0f44")

        if topics is None:
            ax.text(0.5, 0.5, "Not enough data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=10, color="#0e0f44")
            ax.axis("off")
            continue

        y_base  = 0
        y_ticks = []
        y_lbls  = []
        gap     = 1.4

        for t_idx, words in enumerate(topics):
            color  = _TOPIC_PALETTE[t_idx % len(_TOPIC_PALETTE)]
            y_pos  = [y_base + i for i in range(len(words))]
            widths = [len(words) - i for i in range(len(words))]
            ax.barh(y_pos, widths, color=color, alpha=0.88,
                    edgecolor=_BG_LIGHT, linewidth=0.8)
            for y, word in zip(y_pos, words):
                ax.text(0.2, y, word, va="center", fontsize=8.5,
                        color="white" if t_idx == 0 else "#0e0f44",
                        fontweight="bold")
            mid = y_base + len(words) / 2 - 0.5
            y_ticks.append(mid)
            y_lbls.append(f"Topic {t_idx + 1}")
            y_base += len(words) + gap

        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_lbls, fontsize=10, fontweight="bold", color="#0e0f44")
        ax.set_xlabel("Relative word rank", fontsize=9, color="#0e0f44")
        ax.tick_params(axis="x", labelsize=8, colors="#0e0f44")
        ax.tick_params(axis="y", colors="#0e0f44")
        ax.set_xlim(0, _N_TOP_WORDS + 1)
        for spine in ax.spines.values():
            spine.set_edgecolor("#d4b8c0")

    plt.savefig(RESULTS_DIR / "lda_careers.png", dpi=150, bbox_inches="tight")
    plt.show()


def main():
    df_t, df_c = load_data()
    plot_data_frequency(df_t, df_c)
    plot_sentiment_by_year(df_t, df_c)
    plot_grouped_sentiment(df_t, df_c)
    plot_lda_careers(df_t)


if __name__ == "__main__":
    main()