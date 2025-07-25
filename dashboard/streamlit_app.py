
import streamlit as st
import os
import json
import subprocess

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOPICS_FILE = os.path.join(BASE_DIR, "config", "topics.json")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# Load topics
def load_topics():
    with open(TOPICS_FILE, "r") as f:
        return json.load(f)

st.set_page_config(page_title="Python Niche Automation", layout="wide")

st.title("🧠 Python Niche Content Automation System")

niches = list(load_topics().keys())
selected_niche = st.selectbox("Select a Niche", niches)

topics = load_topics().get(selected_niche, [])
selected_topic = st.selectbox("Select a Topic", topics)

article_count = st.slider("How many articles to generate?", 1, 10, 1)

if st.button("⚙️ Generate Article(s)"):
    with st.spinner("Generating article(s)..."):
        cmd = f"python {os.path.join(BASE_DIR, 'scripts/generate_articles.py')} --niche {selected_niche} --count {article_count}"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        st.code(result.stdout + result.stderr)

st.markdown("---")

if st.button("🚀 Publish Articles to WordPress"):
    with st.spinner("Publishing..."):
        cmd = f"python {os.path.join(BASE_DIR, 'scripts/wordpress_publisher.py')} --niche {selected_niche}"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        st.code(result.stdout + result.stderr)

# Show recent articles
st.markdown("### 📂 Latest Articles")
niche_dir = os.path.join(OUTPUT_DIR, selected_niche)
if os.path.exists(niche_dir):
    for topic_dir in sorted(os.listdir(niche_dir)):
        article_path = os.path.join(niche_dir, topic_dir, "article.md")
        if os.path.exists(article_path):
            with open(article_path) as f:
                preview = f.read(500)
                st.markdown(f"#### {topic_dir.replace('-', ' ').title()}")
                st.code(preview)
else:
    st.warning("No articles found for this niche.")
