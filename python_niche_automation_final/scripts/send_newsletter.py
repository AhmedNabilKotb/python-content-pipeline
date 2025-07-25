
import os
import smtplib
from email.mime.text import MIMEText
from datetime import datetime
import json

# Load today's published articles from rotated schedule
def load_todays_articles(schedule_path="config/rotated_publish_schedule_60_days.json"):
    today = datetime.now().strftime("%Y-%m-%d")
    with open(schedule_path, "r") as f:
        schedule = json.load(f)
    return [a for a in schedule if a["publish_date"] == today]

def build_email_content(articles):
    if not articles:
        return "No new articles published today."

    content = "🚀 Today's Python Niche Articles:

"
    for article in articles:
        content += f"📝 [{article['title']}] in {article['niche'].replace('_', ' ').title()}
"
    return content

def send_newsletter(subject, content, sender_email, recipient_email, smtp_server, smtp_port, smtp_password):
    msg = MIMEText(content)
    msg['Subject'] = subject
    msg['From'] = sender_email
    msg['To'] = recipient_email

    try:
        with smtplib.SMTP_SSL(smtp_server, smtp_port) as server:
            server.login(sender_email, smtp_password)
            server.sendmail(sender_email, recipient_email, msg.as_string())
        print("✅ Newsletter sent successfully.")
    except Exception as e:
        print("❌ Failed to send newsletter:", e)

if __name__ == "__main__":
    articles = load_todays_articles()
    content = build_email_content(articles)

    # These should be loaded from secure environment vars or config in production
    send_newsletter(
        subject="📰 Today's Python Articles Digest",
        content=content,
        sender_email=os.getenv("SENDER_EMAIL"),
        recipient_email=os.getenv("RECIPIENT_EMAIL"),
        smtp_server=os.getenv("SMTP_SERVER"),
        smtp_port=int(os.getenv("SMTP_PORT", 465)),
        smtp_password=os.getenv("SMTP_PASSWORD")
    )
