
import os
import re
import unicodedata

def slugify(text, max_length=60):
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("utf-8")
    text = re.sub(r"[^\w\s-]", "", text).strip().lower()
    slug = re.sub(r"[\s_-]+", "-", text)
    return slug[:max_length]

def ensure_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)
