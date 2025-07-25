import os
import json
from datetime import datetime
from scripts.generate_articles import main as generate_article

# Path to the rotated 60-day publishing plan
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCHEDULE_PATH = os.path.join(BASE_DIR, "../rotated_publish_schedule_60_days.json")

def run_today_articles():
    today = datetime.today().strftime("%Y-%m-%d")
    if not os.path.exists(SCHEDULE_PATH):
        print("❌ Publishing schedule not found.")
        return

    with open(SCHEDULE_PATH, "r") as f:
        schedule = json.load(f)

    todays_articles = schedule.get(today, [])
    if not todays_articles:
        print("📭 No articles scheduled for today.")
        return

    print(f"📅 Generating {len(todays_articles)} articles for {today}")
    for article in todays_articles:
        generate_article(article["niche"], title=article["title"])

if __name__ == "__main__":
    run_today_articles()