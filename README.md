# 🐍 Python Niche Content Automation Pipeline

This project is a full-featured, automated pipeline designed to generate, optimize, and publish SEO-friendly articles for Python-related niches on WordPress.

---

## 📁 Project Structure

```
final_restructured_pipeline_complete/
├── config/
│   ├── config.yaml                       # Core generation and SEO settings
│   └── rotated_publish_schedule_60_days.json  # Balanced 60-day publish plan
│
├── dashboard/
│   └── streamlit_app.py                  # Control dashboard GUI
│
├── prompts/
│   ├── prompt_template_*.txt             # Prompt templates for 8 Python niches
│
├── scripts/
│   ├── generate_articles.py              # Core article generator
│   ├── daily_publish.py                  # Scheduled batch generator (3/day)
│   ├── wordpress_publisher.py            # WordPress auto-publish integration
│   ├── utils.py                          # Helpers (slugify, dir creation)
│   ├── seo_optimizer.py                  # Meta + keyword generator
│   ├── image_generator.py                # Fetches Unsplash featured images
│   ├── send_newsletter.py               # Weekly digest from recent articles
│   ├── code_tester.py                   # Validates Python code blocks
│   └── gsc_report.py                    # Google Search Console analytics
```

---

## ✅ Features

- ✍️ Generates long-form articles (1200–2000+ words)
- 🧠 SEO-optimized with meta + keyword extraction
- 🖼 Featured image from Unsplash (auto)
- 🔁 Rotates across 8 Python-related niches
- ⏰ Publishes 3 articles/day (CRON-ready)
- 📤 Auto-posts to WordPress
- 📊 Search Console performance script
- 📨 Newsletter generator
- 🧪 Code snippet tester
- 📊 Streamlit dashboard to manage topics + preview

---

## 🚀 Quick Start

### 1. 🔧 Install Requirements

```bash
pip install -r requirements.txt
```

Also set environment variables:
```bash
export OPENAI_API_KEY=your-key
export UNSPLASH_ACCESS_KEY=your-key
export WORDPRESS_URL=https://your-site.com
export WORDPRESS_USERNAME=admin
export WORDPRESS_APP_PASSWORD=xyz
```

### 2. 📅 Run Daily Publisher

```bash
python scripts/daily_publish.py
```

To test manually:
```bash
python scripts/generate_articles.py --niche web_development --count 1
```

### 3. 🖥 Launch Streamlit UI

```bash
streamlit run dashboard/streamlit_app.py
```

---

## 🛠 CRON Job Example

Run daily at 6am:

```bash
0 6 * * * cd /path/to/final_restructured_pipeline_complete && /usr/bin/python3 scripts/daily_publish.py >> logs/daily.log 2>&1
```

---

## 🧠 Supported Niches

- Web Development
- Data Science & Analytics
- Machine Learning & AI
- Automation & Scripting
- Cybersecurity & Ethical Hacking
- Python for Finance
- Educational Python
- Web Scraping & Data Extraction

---

## 📬 Weekly Newsletter (Optional)

```bash
python scripts/send_newsletter.py
```

---

## 🔍 SEO Reports

```bash
python scripts/gsc_report.py
```

---

## 🧪 Test Code in Articles

```bash
python scripts/code_tester.py --niche data_science
```

---

## 📦 Deployment Tips

- Keep `config.yaml` synced with your editorial goals
- Rotate prompt tones/styles every 60–90 days
- Monitor indexing with Google Search Console
- Use `send_newsletter.py` with Brevo/Mailchimp

---

## 📄 License

MIT License

---

Built with ❤️ for automated Python content publishing.