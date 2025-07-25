
import os
import json
import openai
from image_generator import fetch_unsplash_image
from seo_optimizer import generate_meta_description, extract_keywords
from utils import slugify, ensure_directory
import requests
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOPIC_FILE = os.path.join(BASE_DIR, "config", "topics.json")
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
MEDIA_DIR = os.path.join(BASE_DIR, "media")

openai.api_key = os.getenv("OPENAI_API_KEY")
UNSPLASH_ACCESS_KEY = os.getenv("UNSPLASH_ACCESS_KEY")

os.makedirs(MEDIA_DIR, exist_ok=True)

SECTIONS = [
    "Introduction",
    "Step-by-step Guide",
    "Python Code Examples",
    "Common Mistakes",
    "Best Practices",
    "FAQs",
    "Conclusion",
    "Meta Description and Keywords"
]

def load_topics(niche):
    with open(TOPIC_FILE, "r") as f:
        topics = json.load(f)
    return topics.get(niche, [])

def build_section_prompt(topic, section):
    return f"Write the '{section}' section for a technical Python blog post on the topic: '{topic}'. Format it in Markdown and include relevant details, examples, and clear explanation."

def generate_section_content(topic, section):
    try:
        prompt = build_section_prompt(topic, section)
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a Python expert and technical content writer."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"Error generating section '{section}':", e)
        return f"## {section}\nContent unavailable."


    try:
        url = "https://api.unsplash.com/photos/random"
        params = {
            "query": query,
            "orientation": "landscape",
            "client_id": UNSPLASH_ACCESS_KEY
        }
        response = requests.get(url, params=params)
        data = response.json()
        image_url = data.get("urls", {}).get("regular")
        if image_url:
            img_data = requests.get(image_url).content
            with open(save_path, "wb") as f:
                f.write(img_data)
            return True
    except Exception as e:
        print("Unsplash Error:", e)
    return False


    return False

def save_article(niche, title, sections):

    import shutil

    def copy_fallback_image(niche, image_path):
        fallback_path = os.path.join(BASE_DIR, "fallback_images", f"{niche}.jpg")
        if os.path.exists(fallback_path):
            shutil.copy(fallback_path, image_path)
        else:
            print("⚠ No fallback image found for:", niche)
    slug = slugify(title)
    folder_path = os.path.join(OUTPUT_DIR, niche, slug)
    ensure_directory(folder_path)

    # Save image
    image_path = os.path.join(folder_path, "featured.jpg")
    if not fetch_unsplash_image(title, image_path):
        copy_fallback_image(niche, image_path)

    alt_text = f"{title} - Python article on {niche.replace('_', ' ')}"
    with open(os.path.join(folder_path, "image_alt.txt"), "w") as f:
        f.write(alt_text)

    content = f"![{alt_text}](featured.jpg)\n\n# {title}\n\n"
    content += "\n\n".join(sections)

    file_path = os.path.join(folder_path, "article.md")
    with open(file_path, "w") as f:
        f.write(content)

    # Save SEO metadata
    meta_desc = generate_meta_description(content)
    keywords = extract_keywords(content)
    with open(os.path.join(folder_path, "meta.json"), "w") as f:
        json.dump({"description": meta_desc, "keywords": keywords}, f, indent=2)

    # Copy image to global media folder
    shutil.copy(image_path, os.path.join(MEDIA_DIR, f"{niche}_{slug}.jpg"))

    print(f"✅ Article saved: {file_path}")
    return file_path

def main(niche, count=1, title=None):
    topics = load_topics(niche)
    for topic in topics:
        if title and topic != title:
            continue
        print(f"\n📝 Generating article for: {topic}")
        sections = []
        for section in SECTIONS:
            section_content = generate_section_content(topic, section)
            sections.append(section_content)
        save_article(niche, topic, sections)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--niche", required=True)
    parser.add_argument("--count", type=int, default=1)
    args = parser.parse_args()
    main(args.niche, args.count)
