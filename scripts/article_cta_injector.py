# article_cta_injector.py

import logging
import re
from typing import Optional, List, Dict, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

# Ensure logs dir exists and avoid duplicate root handlers
Path("logs").mkdir(parents=True, exist_ok=True)
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("logs/article_cta_injector.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
logger = logging.getLogger(__name__)


class CTAPlacement(Enum):
    FOOTER = "FOOTER"
    AFTER_FIRST_SECTION = "AFTER_FIRST_SECTION"
    BEFORE_CONCLUSION = "BEFORE_CONCLUSION"


class CTAStyle(Enum):
    MINIMAL = "MINIMAL"
    STANDARD = "STANDARD"
    PREMIUM = "PREMIUM"


@dataclass
class ArticleCTAConfig:
    templates: Dict[str, str] = field(default_factory=dict)
    style: str = "STANDARD"
    placement: str = "FOOTER"
    # Primary threshold used by the injector; we map from settings with alias support
    min_content_length_words: int = 800
    exclude_sections_keywords: List[str] = field(default_factory=list)
    # Idempotency markers
    marker_start: str = "<!-- CTA:BOOKMARK:START -->"
    marker_end: str = "<!-- CTA:BOOKMARK:END -->"
    # If a CTA with marker exists: "skip" (default), "replace"
    on_existing: str = "skip"
    # Extra safety/placement tuning
    conclusion_synonyms: List[str] = field(default_factory=lambda: [
        "conclusion", "summary", "final thoughts", "wrapping up", "wrap-up",
        "takeaways", "next steps"
    ])
    require_min_words_in_intro: int = 50  # used by AFTER_FIRST_SECTION fallback
    respect_yaml_front_matter: bool = True
    # NEW: avoid placing CTA inside these sections (case-insensitive)
    blocklist_sections: List[str] = field(
        default_factory=lambda: ["further reading", "summary cheat sheet", "performance benchmarks"]
    )
    # NEW: replace these placeholders with CTA before heuristic placement (optional)
    placeholders: List[str] = field(default_factory=list)


class ArticleCTAInjector:
    """
    Injects a bookmark/share CTA into long-form markdown articles.

    Highlights:
      • Idempotent: uses HTML comment markers to avoid duplicates (and replace if configured).
      • Safer word-count: counts VISIBLE words only (ignores front matter, code, tables, HTML tags/comments, <script>/<style>).
      • YAML front-matter aware; never inserts inside protected spans.
      • Configurable templates/styles/placements, conclusion synonyms, blocklisted sections, and placeholders.
    """

    # ------------------------ Protected-region regexes ------------------------

    _WORD_RE = re.compile(r"\b\w+\b", re.UNICODE)
    _TABLE_LINE_RE = re.compile(r"^\s*\|.*\|\s*$", re.MULTILINE)

    _HTML_TAG_RE = re.compile(r"<[^>]+>")
    _HTML_COMMENT_RE = re.compile(r"<!--[\s\S]*?-->", re.IGNORECASE)
    _SCRIPT_BLOCK_RE = re.compile(r"<script\b[^>]*>[\s\S]*?</script\s*>", re.IGNORECASE)
    _STYLE_BLOCK_RE = re.compile(r"<style\b[^>]*>[\s\S]*?</style\s*>", re.IGNORECASE)

    def __init__(self, app_settings: Dict):
        self.config = ArticleCTAConfig()
        cta_settings = (app_settings or {}).get("article_cta_injector", {}) or {}

        # Load templates
        self.config.templates = cta_settings.get("templates", {}) or {}
        if not self.config.templates:
            logger.warning("No CTA templates found in settings.json. Using defaults.")
            self.config.templates = {
                "MINIMAL": """
{marker_start}
---
💾 **Bookmark “{article_title}”** for later reference.
{marker_end}
""".strip(),
                "STANDARD": """
{marker_start}
---
<div class="bookmark-cta">
  <h3>🔖 Save This Guide: {article_title}</h3>
  <p>Want to revisit this later? Bookmark it now or add to your reading list.</p>
  <ul>
    <li><a href="#save-to-pocket">Save to Pocket</a></li>
    <li><a href="#add-to-notion">Add to Notion</a></li>
    <li><a href="#bookmark-browser">Browser Bookmark</a></li>
  </ul>
</div>
{marker_end}
""".strip(),
                "PREMIUM": """
{marker_start}
---
<div class="premium-cta">
  <div class="cta-header" style="display:flex;gap:.5rem;align-items:center">
    <svg width="24" height="24" viewBox="0 0 24 24" aria-hidden="true">
      <path d="M19 21L12 16L5 21V5a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2v16z" fill="currentColor"/>
    </svg>
    <h3 style="margin:0">Save “{article_title}”</h3>
  </div>
  <p>This guide took 15+ hours to create. Bookmark it to:</p>
  <ul>
    <li>Access the code samples anytime</li>
    <li>Reference the cheatsheets</li>
    <li>Support our work</li>
  </ul>
  <div class="cta-buttons" style="display:flex;gap:.5rem;flex-wrap:wrap">
    <button class="save-button">💾 Save to Bookmarks</button>
    <button class="share-button">📤 Share with Team</button>
  </div>
</div>
{marker_end}
""".strip(),
            }

        # Basic fields
        self.config.style = (cta_settings.get("style", self.config.style) or "STANDARD").upper()
        self.config.placement = (cta_settings.get("placement", self.config.placement) or "FOOTER").upper()
        self.config.marker_start = cta_settings.get("marker_start", self.config.marker_start)
        self.config.marker_end = cta_settings.get("marker_end", self.config.marker_end)
        self.config.on_existing = (cta_settings.get("on_existing", self.config.on_existing) or "skip").lower()

        # Threshold with alias support
        self.config.min_content_length_words = int(
            cta_settings.get(
                "min_word_count_for_injection",
                cta_settings.get("min_content_length_words", self.config.min_content_length_words),
            )
        )
        self.config.exclude_sections_keywords = [k.lower() for k in cta_settings.get("exclude_sections_keywords", [])]
        if "conclusion_synonyms" in cta_settings and isinstance(cta_settings["conclusion_synonyms"], list):
            self.config.conclusion_synonyms = [str(x).lower() for x in cta_settings["conclusion_synonyms"]]

        self.config.require_min_words_in_intro = int(
            cta_settings.get("require_min_words_in_intro", self.config.require_min_words_in_intro)
        )
        self.config.respect_yaml_front_matter = bool(
            cta_settings.get("respect_yaml_front_matter", self.config.respect_yaml_front_matter)
        )

        # NEW configs
        self.config.blocklist_sections = [s.lower().strip() for s in cta_settings.get("blocklist_sections", [])] or \
                                         self.config.blocklist_sections
        self.config.placeholders = [p for p in cta_settings.get("placeholders", []) if (p or "").strip()]

        self._validate_config()

    # ----------------------------- Validation ---------------------------

    def _validate_config(self) -> None:
        try:
            CTAPlacement[self.config.placement]
            CTAStyle[self.config.style]
        except KeyError as e:
            raise ValueError(f"Invalid CTA configuration: {e}. Check 'style' or 'placement' in settings.")
        if self.config.min_content_length_words < 100:
            raise ValueError("Minimum content length (words) must be at least 100.")
        if self.config.on_existing not in {"skip", "replace"}:
            raise ValueError("on_existing must be 'skip' or 'replace'.")

    # ----------------------------- Public API ---------------------------

    def inject_cta(self, markdown_text: str, context: Optional[Dict[str, str]] = None) -> str:
        """
        Inject CTA into the given markdown.

        context (optional):
          Arbitrary placeholders usable in templates, e.g. {"article_url": "..."}.
          Always supports {article_title}, {marker_start}, {marker_end}.
        """
        if not markdown_text:
            return markdown_text

        # If a CTA already exists
        if self._has_existing_cta(markdown_text):
            if self.config.on_existing == "skip":
                logger.info("CTA already present; skipping (on_existing=skip).")
                return markdown_text
            logger.info("CTA already present; replacing (on_existing=replace).")
            markdown_text = self._remove_existing_cta(markdown_text)

        if not self._should_inject_cta(markdown_text):
            logger.info("Skipping CTA injection based on content criteria.")
            return markdown_text

        try:
            # Extract article title for dynamic CTA content
            title_match = re.search(r'^\s*#\s+(.*)\s*$', markdown_text, re.MULTILINE)
            article_title = title_match.group(1).strip() if title_match else "This Guide"

            placeholders: Dict[str, str] = {"article_title": article_title}
            if context:
                placeholders.update({k: str(v) for k, v in context.items()})
            placeholders["marker_start"] = self.config.marker_start
            placeholders["marker_end"] = self.config.marker_end

            cta_content = self._generate_cta_content(placeholders)

            # Phase 0: replace placeholders if provided
            if self.config.placeholders:
                for ph in self.config.placeholders:
                    idx = markdown_text.find(ph)
                    if idx != -1:
                        logger.info("CTA placeholder found; replacing.")
                        return markdown_text[:idx] + self._normalize_pad_around(cta_content) + markdown_text[idx + len(ph):]

            return self._insert_cta(markdown_text, cta_content)
        except Exception as e:
            logger.error(f"Failed to inject CTA: {e}", exc_info=True)
            return markdown_text

    # ----------------------------- Internals ----------------------------

    # ---------- Protected spans / visibility ----------

    @staticmethod
    def _yaml_front_matter_span(text: str) -> Optional[Tuple[int, int]]:
        """Return (start, end) of YAML front matter at file start if present."""
        m = re.match(r"^---\s*\n[\s\S]*?\n---\s*(\n|$)", text)
        return (m.start(), m.end()) if m else None

    @staticmethod
    def _code_spans(text: str) -> List[Tuple[int, int]]:
        """
        Return (start,end) byte offsets of fenced code blocks to avoid inserting within them.
        Supports ``` and ~~~, with language tags, and unclosed fences to EOF.
        """
        spans: List[Tuple[int, int]] = []
        offset = 0
        lines = text.splitlines(keepends=True)

        fence_pat = re.compile(r"^([`~]{3,})(.*)$")  # line with ```... or ~~~...
        in_fence = False
        fence_char = ""
        fence_len = 0
        start_idx = 0

        for ln in lines:
            if not in_fence:
                m = fence_pat.match(ln.strip())
                if m:
                    in_fence = True
                    fence_char = m.group(1)[0]
                    fence_len = len(m.group(1))
                    start_idx = offset
            else:
                # A closing fence must start the (stripped) line and match fence char & length
                stripped = ln.lstrip()
                if stripped.startswith(fence_char * fence_len):
                    in_fence = False
                    spans.append((start_idx, offset + len(ln)))
            offset += len(ln)

        if in_fence:
            spans.append((start_idx, len(text)))
        return spans

    @staticmethod
    def _html_comment_spans(text: str) -> List[Tuple[int, int]]:
        return [m.span() for m in re.finditer(r"<!--[\s\S]*?-->", text, re.IGNORECASE)]

    @staticmethod
    def _script_style_spans(text: str) -> List[Tuple[int, int]]:
        spans: List[Tuple[int, int]] = []
        spans.extend(m.span() for m in re.finditer(r"<script\b[^>]*>[\s\S]*?</script\s*>", text, re.IGNORECASE))
        spans.extend(m.span() for m in re.finditer(r"<style\b[^>]*>[\s\S]*?</style\s*>", text, re.IGNORECASE))
        return spans

    @staticmethod
    def _html_tag_spans(text: str) -> List[Tuple[int, int]]:
        return [m.span() for m in re.finditer(r"<[^>]+>", text or "")]

    def _table_spans(self, text: str) -> List[Tuple[int, int]]:
        """
        Heuristic: contiguous blocks of >=2 lines that look like markdown tables (contain '|').
        """
        spans: List[Tuple[int, int]] = []
        if not text:
            return spans

        lines = text.splitlines(keepends=True)
        offsets = []
        pos = 0
        for ln in lines:
            offsets.append(pos)
            pos += len(ln)
        offsets.append(pos)

        def is_table_line(s: str) -> bool:
            return "|" in s and bool(self._TABLE_LINE_RE.match(s))

        i = 0
        n = len(lines)
        while i < n:
            if is_table_line(lines[i]):
                j = i + 1
                while j < n and is_table_line(lines[j]):
                    j += 1
                if j - i >= 2:
                    spans.append((offsets[i], offsets[j]))
                i = j
            else:
                i += 1
        return spans

    @staticmethod
    def _merge_spans(spans: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        if not spans:
            return []
        spans = sorted(spans)
        merged = [spans[0]]
        for a, b in spans[1:]:
            la, lb = merged[-1]
            if a <= lb:
                merged[-1] = (la, max(lb, b))
            else:
                merged.append((a, b))
        return merged

    def _protected_spans(self, text: str) -> List[Tuple[int, int]]:
        spans: List[Tuple[int, int]] = []
        fm = self._yaml_front_matter_span(text)
        if fm:
            spans.append(fm)
        spans.extend(self._code_spans(text))
        spans.extend(self._table_spans(text))
        spans.extend(self._html_comment_spans(text))
        spans.extend(self._script_style_spans(text))
        spans.extend(self._html_tag_spans(text))
        return self._merge_spans(spans)

    @staticmethod
    def _pos_in_spans(pos: int, spans: List[Tuple[int, int]]) -> bool:
        return any(a <= pos < b for a, b in spans)

    def _visible_word_count(self, text: str) -> int:
        """Count words outside protected spans."""
        spans = self._protected_spans(text)
        return sum(1 for m in self._WORD_RE.finditer(text or "") if not self._pos_in_spans(m.start(), spans))

    # ----------------------------- Public helpers ---------------------------

    def _should_inject_cta(self, content: str) -> bool:
        content_word_count = self._visible_word_count(content)

        lower_content = content.lower()
        if any(keyword in lower_content for keyword in self.config.exclude_sections_keywords):
            logger.info("CTA injection skipped: content contains excluded keywords.")
            return False

        if content_word_count < self.config.min_content_length_words:
            logger.info(
                "CTA injection skipped: %d < %d visible words.",
                content_word_count, self.config.min_content_length_words
            )
            return False

        return True

    class _SafeDict(dict):
        def __missing__(self, key):
            return ""

    def _generate_cta_content(self, placeholders: Dict[str, str]) -> str:
        template = self.config.templates.get(self.config.style, self.config.templates.get("STANDARD", ""))
        # Safe format (missing keys -> empty)
        return template.format_map(self._SafeDict(placeholders)).strip()

    # ----------------------------- Insertion -------------------------------

    @staticmethod
    def _normalize_pad_around(snippet: str) -> str:
        s = snippet.strip("\n")
        return f"\n\n{s}\n\n"

    def _insert_cta(self, content: str, cta: str) -> str:
        placement = CTAPlacement[self.config.placement]
        spans = self._protected_spans(content)  # avoid inserting inside protected regions

        # Respect YAML front matter for insertion calculations
        fm_span = self._yaml_front_matter_span(content) if self.config.respect_yaml_front_matter else None
        fm_end = (fm_span[1] if fm_span else 0)

        if placement == CTAPlacement.AFTER_FIRST_SECTION:
            # Insert after the first non-blocklisted H2 line that is not inside a protected span or front matter
            blocked = {s.lower().strip() for s in self.config.blocklist_sections}
            for m in re.finditer(r'(?m)^##\s+(.+)$', content):
                if m.start() < fm_end or self._pos_in_spans(m.start(), spans):
                    continue
                title = re.sub(r"\s+", " ", m.group(1)).strip().lower()
                if title in blocked:
                    continue
                # Insert after the first blank line following this H2
                insert_pos = m.end()
                after = content[insert_pos:]
                gap = re.search(r"\n\s*\n", after)
                if gap:
                    insert_pos += gap.end()
                # If we landed in a protected span, hop to end of that span and then to the next blank line
                while insert_pos < len(content) and self._pos_in_spans(insert_pos - 1, spans):
                    for a, b in spans:
                        if a <= insert_pos - 1 < b:
                            insert_pos = b
                            break
                    nxt = re.search(r"\n\s*\n", content[insert_pos:])
                    insert_pos = (insert_pos + nxt.end()) if nxt else len(content)
                logger.info("CTA inserted AFTER_FIRST_SECTION after first non-blocklisted H2.")
                return content[:insert_pos] + self._normalize_pad_around(cta) + content[insert_pos:]

            # Fallback: after first substantial paragraph (≥ require_min_words_in_intro) outside protected regions/front matter
            parts = [p for p in re.split(r'(\n\s*\n)', content)]
            acc_len = 0
            for i in range(0, len(parts), 2):  # step over blocks
                block = parts[i]
                sep = parts[i + 1] if i + 1 < len(parts) else ""
                start_pos = acc_len
                end_pos = acc_len + len(block)
                if block.strip() and end_pos > fm_end:
                    if not self._pos_in_spans(start_pos, spans) and not self._pos_in_spans(end_pos - 1, spans):
                        # visible words in this block
                        block_spans = self._protected_spans(block)
                        wc = sum(1 for m in self._WORD_RE.finditer(block) if not self._pos_in_spans(m.start(), block_spans))
                        if wc >= self.config.require_min_words_in_intro:
                            logger.info("CTA inserted AFTER_FIRST_SECTION after first substantial paragraph.")
                            return content[:end_pos] + self._normalize_pad_around(cta) + content[end_pos:]
                acc_len += len(block) + len(sep)

            logger.warning("AFTER_FIRST_SECTION spot not found. Appending to FOOTER.")
            return self._append_footer(content, cta)

        if placement == CTAPlacement.BEFORE_CONCLUSION:
            # Insert before a conclusion-like heading (case-insensitive), not in protected regions
            concl_re = r'(?im)^\s*#{1,6}\s*(' + "|".join(map(re.escape, self.config.conclusion_synonyms)) + r')\b.*$'
            for m in re.finditer(concl_re, content):
                if (m.start() >= fm_end) and not self._pos_in_spans(m.start(), spans):
                    insert_pos = m.start()
                    logger.info("CTA inserted BEFORE_CONCLUSION at detected conclusion heading.")
                    return content[:insert_pos] + self._normalize_pad_around(cta) + content[insert_pos:]

            # Fallback: before the last paragraph outside protected regions/front matter
            paras = [p for p in re.split(r'(\n\s*\n)', content)]
            acc = 0
            last_block_start = None
            for i in range(0, len(paras), 2):
                block = paras[i]
                sep = paras[i + 1] if i + 1 < len(paras) else ""
                start_pos = acc
                if block.strip() and start_pos >= fm_end and not self._pos_in_spans(start_pos, spans):
                    last_block_start = start_pos
                acc += len(block) + len(sep)
            if last_block_start is not None:
                logger.info("CTA inserted BEFORE_CONCLUSION at last paragraph.")
                return content[:last_block_start] + self._normalize_pad_around(cta) + content[last_block_start:]

            logger.warning("BEFORE_CONCLUSION spot not found. Appending to FOOTER.")
            return self._append_footer(content, cta)

        # FOOTER
        logger.info("CTA inserted at FOOTER.")
        return self._append_footer(content, cta)

    def _append_footer(self, content: str, cta: str) -> str:
        if not content.endswith("\n"):
            content += "\n"
        return content + self._normalize_pad_around(cta)

    # ----------------------------- Idempotency ---------------------------

    def _has_existing_cta(self, content: str) -> bool:
        # Marker-based detection
        start = re.escape(self.config.marker_start)
        end = re.escape(self.config.marker_end)
        pat = re.compile(start + r"[\s\S]*?" + end, flags=re.DOTALL)
        if pat.search(content):
            return True
        # Fallback: common heading/class if markers were changed externally
        loose = re.compile(r'^\s*#{2,3}\s*(save|bookmark)\b.*$|class="(?:bookmark-cta|premium-cta)"',
                           re.IGNORECASE | re.MULTILINE)
        return bool(loose.search(content))

    def _remove_existing_cta(self, content: str) -> str:
        # Remove by markers first
        start = re.escape(self.config.marker_start)
        end = re.escape(self.config.marker_end)
        pat = re.compile(rf"\s*{start}[\s\S]*?{end}\s*", flags=re.DOTALL)
        new_text, n = pat.subn("", content)
        if n:
            logger.info("Removed %d existing CTA block(s) by marker.", n)
            return new_text
        # Fallback: remove by loose heading/class if markers missing
        loose_block = re.compile(
            r"(?ms)^\s*#{2,3}\s*(save|bookmark)\b[\s\S]*?(?=^\s*#{1,6}\s|\Z)|"
            r'<div[^>]+class="(?:bookmark-cta|premium-cta)".*?</div>\s*',
            re.IGNORECASE,
        )
        new_text2, n2 = loose_block.subn("", content)
        if n2:
            logger.info("Removed %d existing CTA block(s) by loose match.", n2)
            return new_text2
        return content


# ----------------------------- Legacy helper -------------------------------

def inject_bookmark_cta(markdown_text: str) -> str:
    dummy_app_settings = {
        "article_cta_injector": {
            "style": "STANDARD",
            "placement": "FOOTER",
            "min_word_count_for_injection": 500,
            "exclude_sections_keywords": [],
        }
    }
    injector = ArticleCTAInjector(dummy_app_settings)
    return injector.inject_cta(markdown_text)


# ----------------------------- Demo ----------------------------------------

if __name__ == "__main__":
    dummy_app_settings = {
        "article_cta_injector": {
            "style": "PREMIUM",
            "placement": "AFTER_FIRST_SECTION",
            "min_word_count_for_injection": 100,
            "exclude_sections_keywords": ["faq", "appendix"],
            "on_existing": "replace",
            "conclusion_synonyms": ["conclusion", "summary", "final thoughts", "wrap-up"],
            "blocklist_sections": ["further reading", "summary cheat sheet", "performance benchmarks"],
            "placeholders": ["<!-- CTA-HOOK -->"],
        }
    }
    injector = ArticleCTAInjector(dummy_app_settings)

    sample = (
        "---\n"
        "title: My Post\n"
        "date: 2025-08-15\n"
        "---\n\n"
        "# Title\n\n"
        "Intro para with enough words to pass the threshold. " * 10 + "\n\n"
        "```python\n# code block with ## Heading\nprint('do not insert CTA here')\n```\n\n"
        "## First H2\n\n"
        "Body content continues here...\n\n"
        "## Further Reading\n\n"
        "- Link A\n"
        "- Link B\n\n"
        "## Conclusion\n\n"
        "Some final words.\n"
        "<!-- CTA-HOOK -->\n"
    )

    out = injector.inject_cta(sample, context={"article_url": "https://example.com/title"})
    print(out)
