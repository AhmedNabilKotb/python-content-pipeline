
def generate_meta_description(article_text, max_length=155):
    lines = article_text.strip().split("\n")
    for line in lines:
        if line and len(line) <= max_length:
            return line.strip()
    return article_text[:max_length].strip() + "..."

def extract_keywords(article_text, num_keywords=10):
    from collections import Counter
    import re
    words = re.findall(r'\b[a-zA-Z]{4,}\b', article_text.lower())
    stopwords = set([
        'python', 'this', 'that', 'with', 'from', 'your', 'have', 'just',
        'like', 'will', 'what', 'when', 'where', 'about', 'should'
    ])
    filtered = [word for word in words if word not in stopwords]
    counts = Counter(filtered).most_common(num_keywords)
    return ', '.join([word for word, _ in counts])
