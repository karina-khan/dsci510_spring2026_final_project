# AI Sentiment in Women's Media
### An analysis of long-form YouTube videos and comments centred around women, across the timeline of the AI boom.

Studies show that women adopt AI at rates 25% lower than men. Call it the 
"ick," but some claim the difference is rooted in how AI systems are designed 
with feminized personas that perform submissiveness (Awomosu, 2026). This 
project investigates how women's media frames artificial intelligence 
throughout the AI boom.

Research question: What is the sentiment of artificial intelligence in 
long-form YouTube videos and comments centered around women, from 2022–2026? 
And how does this sentiment vary across domains?

Requirements

Install dependencies:
bash
pip install -r requirements.txt

You will also need:
- A YouTube Data API v3 key stored in a .env file in src/
- yt-dlp installed (pip install yt-dlp)
- A cookies.txt file in the project root for yt-dlp authentication
- Use .env.example as a template.

How to Run the main pipeline

From the src/ directory:
bash
cd src
python main.py

This will filter and tag comments and transcripts with thematic categories,
run VADER sentiment scoring on both datasets, save filtered_comments.csv
and filtered_transcripts.csv to data/, and generate all visualizations.

Optional: Run the analysis notebook

Open and run results_test.ipynb in Jupyter for full exploratory 
analysis and additional visualizations.
