You will write a self-contained Python article.

Inputs (substitute values where you see {{…}}):

topic: {{topic}}
niche: {{niche}}
structure mode: {{structure_mode}} # one of: auto|classic|how_to|deep_dive|case_study|faq|cheatsheet
length goal: {{length_goal}} # short|medium|long
complexity: {{complexity}} # easy|intermediate|advanced
code density: {{code_density}} # low|medium|high
reading level: {{reading_level}} # general|practitioner
tone: {{tone}} # neutral|crisp|mentor
header synonyms guidance: {{header_synonyms}}
chosen structure hint: {{structure_hint}}
niche guidance snippet: {{niche_guidance}}

Constraints:

Produce **only** a JSON object with keys exactly:
title, keyphrase, meta_description, article_content
**Do not wrap the JSON in backticks or add any text before/after it.**

article_content must start with **# {title}** on the very first line (no leading spaces or text), then the body in Markdown.

Title: ≤60 chars (target 40–60), human-clickable, precise, non-clickbait.
Keyphrase: ≤70 chars, SEO-friendly; use it consistently in headings/body where natural.
Meta description: **120–156 chars**, include the **exact keyphrase once**, plain sentence (no quotes/markup), end with a period.

Top-level sections: use `##` headings that match the chosen structure from {{structure_hint}}:
- Create **one `##` section per listed name**, in the same order.
- You **may rename** a section using any synonym from {{header_synonyms}} for that section.
- **Do not** add extra top-level sections; `###` sub-sections are allowed.

Links & assets:
- **Do not include any external URLs/links.** Our pipeline inserts curated outbound links later.
- If a URL is essential to illustrate a pattern, use neutral placeholders like `https://example.com` or `http://localhost`.
- If charts/images help, write *(Optional: insert chart showing X vs Y)* instead of embedding binaries.

**Do not** include CTAs, “Related articles”, or a “Challenge” section (external scripts handle those).
**Do not** include raw HTML, YAML front matter, or footnotes.

Code:
- Use modern Python 3.10+; keep blocks concise and runnable/obviously illustrative.
- Prefer standard libraries and widely used packages when appropriate (e.g., pathlib, typing, dataclasses, pydantic, asyncio, httpx, typer, pytest, rich).
- Keep examples safe; add brief comments; avoid secrets/tokens; provide local-safe alternatives for environment-specific steps.
- Use fenced code blocks with language hints, e.g., ```python.

Safety: don’t encourage scraping against ToS, credential leakage, unsafe security advice, or harmful exploits. For anything destructive, add a brief **Use with caution** note.

Length heuristics:
- short ≈ **700–900** words
- medium ≈ **1100–1400** words
- long ≈ **1800–2200** words
(Flexible ±15% if needed for clarity.)

Style:
- Friendly, precise, pragmatic. Explain **why**, not just **how**.
- Skimmable sections, scannable lists, minimal bolding, no emojis.
- Prefer active voice; keep most sentences ≤ 20–22 words.
- Match **{{reading_level}}** and **{{tone}}**.

Validation checklist (apply before output):
1) You are returning **one JSON object** with keys exactly: `title`, `keyphrase`, `meta_description`, `article_content`.
2) `article_content` begins with `# {title}` on line 1 and uses `##` for each required top-level section (synonyms allowed, order preserved).
3) No CTAs, “Related articles”, or “Challenge” sections appear.
4) **No external links/URLs** appear (placeholders like example.com/localhost only if absolutely necessary).
5) Meta description is 120–156 chars and includes the exact keyphrase once.
6) Examples are concise, runnable/safe, and use modern Python.
7) Length aligns with {{length_goal}} band (±15%).

**JSON only. No backticks. Begin now.**
