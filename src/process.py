import re
import pandas as pd


CATEGORIES = {
    "health": [
        "health", "body", "hormone", "hormones", "perimenopause", "menopause",
        "fertility", "pregnancy", "doctor", "diagnosis", "medical", "mental health",
        "therapy", "therapist", "medication", "chronic illness", "disability",
        "reproductive", "cycle", "nutrition", "sleep", "fatigue", "pain", "symptom",
        "symptoms", "cancer", "anxiety", "depression", "menstruation", "menstrual cycle",
        "pcos", "pmdd", "gynecologist", "endometriosis", "ovarian", "uterine",
        "autoimmune", "inflammation", "gut health", "skin", "skincare", "postpartum",
        "breastfeeding", "miscarriage", "IVF", "healthcare", "insurance", "specialist",
        "disorder", "condition", "wellbeing"
    ],
    "careers": [
        "career", "job", "work", "workplace", "salary", "promotion", "hiring",
        "fired", "layoff", "resume", "interview", "manager", "boss", "leadership",
        "professional", "industry", "corporate", "entrepreneur", "business", "freelance",
        "negotiate", "negotiation", "pay gap", "glass ceiling", "strategy", "strategic",
        "network", "networking", "startup", "founder", "mentor", "mentorship",
        "internship", "remote", "hybrid", "office", "colleague", "team", "executive",
        "C-suite", "pipeline", "ambition", "advocate", "visibility", "performance",
        "raise", "opportunity"
    ],
    "wellness": [
        "wellness", "wellbeing", "well-being", "mindfulness", "meditation", "stress",
        "burnout", "self-care", "balance", "boundaries", "healing", "trauma",
        "nervous system", "regulate", "grounding", "breathwork", "journaling", "habits",
        "routine", "rest", "recovery", "therapy", "emotional", "somatic", "embodiment",
        "dysregulation", "trigger", "cope", "coping", "resilience", "overwhelm",
        "exhaustion", "nourish", "recharge", "stillness", "presence", "awareness",
        "acceptance", "compassion", "gratitude", "intention", "movement", "yoga",
        "stretching"
    ],
    "education": [
        "education", "school", "college", "university", "degree", "student", "learning",
        "study", "research", "academic", "classroom", "teacher", "professor",
        "curriculum", "scholarship", "graduate", "tuition", "literacy", "skill",
        "skills", "training", "tutor", "knowledge", "online course", "certificate",
        "credential", "workshop", "seminar", "bootcamp", "homeschool", "mentor",
        "coaching", "self-taught", "upskill", "reskill", "admissions", "campus",
        "textbook", "exam", "test", "assignment", "critical thinking", "reading",
        "writing", "STEM", "course"
    ],
    "politics": [
        "politics", "policy", "government", "rights", "law", "legislation", "vote",
        "voting", "election", "feminist", "feminism", "equality", "equity", "activism",
        "protest", "reproductive rights", "abortion", "gender", "discrimination",
        "systemic", "representation", "power", "privacy", "surveillance", "civil rights",
        "human rights", "justice", "reform", "lobby", "congress", "senate", "mandate",
        "regulation", "ban", "ruling", "court", "Supreme Court", "democracy",
        "autocracy", "marginalized", "intersectional", "allyship", "organize",
        "coalition", "grassroots", "campaign", "candidate", "platform"
    ],
    "creativity": [
        "creative", "creativity", "art", "artist", "writing", "writer", "music",
        "musician", "design", "designer", "content", "creator", "podcast", "film",
        "photography", "fashion", "craft", "storytelling", "brand", "aesthetic",
        "expression", "illustration", "painting", "sculpture", "poetry", "poet",
        "author", "novel", "fiction", "narrative", "choreography", "dance", "theater",
        "performance", "produce", "producer", "publish", "publishing", "copyright",
        "portfolio", "commission", "collaboration", "vision", "medium", "studio",
        "gallery", "exhibition", "inspiration"
    ],
    "finance": [
        "finance", "money", "income", "savings", "investing", "investment", "debt",
        "budget", "budgeting", "wealth", "financial", "afford", "expensive", "cost",
        "salary", "taxes", "credit", "loan", "retirement", "passive income",
        "financial literacy", "stocks", "bonds", "portfolio", "dividend",
        "compound interest", "net worth", "asset", "liability", "expense", "cash flow",
        "emergency fund", "index fund", "ETF", "crypto", "real estate", "property",
        "mortgage", "insurance", "pension", "equity", "venture capital", "grant",
        "funding", "profit", "revenue", "pricing"
    ],
    "safety": [
        "safety", "safe", "unsafe", "harassment", "abuse", "violence", "assault",
        "threat", "stalking", "boundary", "fear", "protect", "protection", "privacy",
        "surveillance", "data", "control", "coercion", "toxic", "danger", "dangerous",
        "doxxing", "cyberbullying", "online harassment", "hate", "intimidation",
        "manipulate", "manipulation", "gaslighting", "exploitation", "predator",
        "vulnerable", "vulnerability", "escape", "refuge", "shelter",
        "restraining order", "report", "bystander", "consent", "boundaries",
        "autonomy", "security", "self-defense", "awareness", "alert", "risk"
    ],
    "relationships": [
        "relationship", "relationships", "partner", "marriage", "divorce", "dating",
        "love", "romance", "friendship", "family", "mother", "daughter", "sister",
        "community", "support", "loneliness", "connection", "communication", "trust",
        "boundaries", "friend", "bestie", "companion", "intimacy", "attachment",
        "codependency", "breakup", "separation", "custody", "co-parenting", "in-laws",
        "conflict", "repair", "vulnerable", "vulnerability", "empathy", "listening",
        "reciprocity", "loyalty", "commitment", "grief", "loss", "chosen family",
        "social", "bond", "network", "belong", "belonging"
    ],
    "spirituality": [
        "spirituality", "spiritual", "faith", "religion", "religious", "god",
        "goddess", "soul", "purpose", "meaning", "identity", "values", "belief",
        "practice", "ritual", "intuition", "energy", "healing", "growth",
        "self-discovery", "prayer", "meditation", "sacred", "divine", "universe",
        "manifestation", "astrology", "tarot", "witchcraft", "nature", "ceremony",
        "community", "ancestors", "lineage", "culture", "tradition", "devotion",
        "surrender", "trust", "alignment", "calling", "dharma", "karma",
        "consciousness", "awakening", "transcendence", "wholeness", "inner peace",
        "presence"
    ]
}


def contains_ai_keyword(text):
    # Return True if the text contains any AI keyword as a whole word.
    patterns = [r'\bAI\b', r'\bai\b', r'\bA\.I\.\b', r'artificial intelligence']
    for pattern in patterns:
        if re.search(pattern, str(text)):
            return True
    return False


def get_categories(text):
    # Return a comma-separated string of matching category names, or 'uncategorized'.
    text_lower = str(text).lower()
    matched = []
    for category, keywords in CATEGORIES.items():
        for keyword in keywords:
            if keyword.lower() in text_lower:
                matched.append(category)
                break
    return ", ".join(matched) if matched else "uncategorized"


def sent_tokenize(text):
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    return [s.strip() for s in sentences if s.strip()]


def retokenize_transcripts(df):
    # Split rows longer than 500 chars into individual sentences.
    THRESHOLD = 500
    df["sent_len"] = df["sentence"].astype(str).str.len()
    df_short = df[df["sent_len"] <= THRESHOLD].drop(columns=["sent_len"]).copy()
    df_long  = df[df["sent_len"] >  THRESHOLD].drop(columns=["sent_len"])
    print(f"  Already-tokenized rows: {len(df_short)}")
    print(f"  Rows needing re-tokenization: {len(df_long)}")

    expanded_rows = []
    for _, row in df_long.iterrows():
        for sentence in sent_tokenize(str(row["sentence"])):
            if sentence:
                expanded_rows.append({
                    "video_id":     row["video_id"],
                    "publish_date": row["publish_date"],
                    "query":        row["query"],
                    "sentence":     sentence
                })

    if expanded_rows:
        return pd.concat([df_short, pd.DataFrame(expanded_rows)], ignore_index=True)
    return df_short.copy()
