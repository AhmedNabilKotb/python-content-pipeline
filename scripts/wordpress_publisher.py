
import os
from utils import slugify, ensure_directory
import json
import requests
from pathlib import Path
from base64 import b64encode

# Config
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# Replace with your own WordPress credentials and site URL
WP_URL = os.getenv("WP_URL", "https://your-wordpress-site.com/wp-json/wp/v2")
WP_USER = os.getenv("WP_USER", "your_username")
WP_APP_PASS = os.getenv("WP_APP_PASS", "your_application_password")  # Use Application Passwords plugin

HEADERS = {
    "Authorization": "Basic " + b64encode(f"{WP_USER}:{WP_APP_PASS}".encode()).decode("utf-8")
}

def create_category(name):
    url = f"{WP_URL}/categories"
    response = requests.post(url, headers=HEADERS, json={"name": name})
    if response.status_code == 201:
        return response.json()["id"]
    elif response.status_code == 400 and "term_exists" in response.text:
        # Get existing category ID
        cats = requests.get(url, headers=HEADERS).json()
        for c in cats:
            if c["name"].lower() == name.lower():
                return c["id"]
    return None

def upload_image(image_path):
    with open(image_path, "rb") as img:
        filename = os.path.basename(image_path)
        headers = HEADERS.copy()
        headers["Content-Disposition"] = f"attachment; filename={filename}"
        headers["Content-Type"] = "image/jpeg"
        response = requests.post(f"{WP_URL}/media", headers=headers, data=img)
        if response.status_code == 201:
            return response.json()["id"]
    return None

def publish_article(niche, title, content, image_id=None):
    category_id = create_category(niche.replace("_", " ").title())
    payload = {
        "title": title,
        "content": content,
        "status": "publish",
        "categories": [category_id],
    }
    if image_id:
        payload["featured_media"] = image_id

    response = requests.post(f"{WP_URL}/posts", headers=HEADERS, json=payload)
    if response.status_code == 201:
        print(f"✅ Published: {title}")
    else:
        print(f"❌ Failed to publish: {title}")
        print(response.text)

def main(niche):
    niche_dir = os.path.join(OUTPUT_DIR, niche)
    if not os.path.exists(niche_dir):
        print(f"No content found for niche '{niche}'")
        return

    for topic_dir in Path(niche_dir).iterdir():
        article_file = topic_dir / "article.md"
        if article_file.exists():
            with open(article_file, "r") as f:
                content = f.read()
            title = topic_dir.name.replace("-", " ").title()
            image_path = topic_dir / "featured.jpg"
            image_id = upload_image(image_path) if image_path.exists() else None
            publish_article(niche, title, content, image_id)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--niche", required=True, help="Niche to publish (e.g., web_dev)")
    args = parser.parse_args()
    main(args.niche)
