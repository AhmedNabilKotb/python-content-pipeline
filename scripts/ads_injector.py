# ads_injector.py

import logging
import re
from typing import Optional, List, Dict, Tuple

logger = logging.getLogger(__name__)

# ---------------------------- regex & constants ----------------------------

_WORD_RE = re.compile(r"\b\w+\b", re.UNICODE)
_H2_RE = re.compile(r"(?m)^(##\s+.+)$")

# HTML/Markdown protections
_HTML_TAG_RE = re.compile(r"<[^>]+>")  # single tags
_HTML_COMMENT_RE = re.compile(r"<!--[\s\S]*?-->", re.IGNORECASE)
_SCRIPT_BLOCK_RE = re.compile(r"<script\b[^>]*>[\s\S]*?</script\s*>", re.IGNORECASE)
_STYLE_BLOCK_RE = re.compile(r"<style\b[^>]*>[\s\S]*?</style\s*>", re.IGNORECASE)

# Markdown table heuristic: contiguous lines that contain '|'
_TABLE_LINE_RE = re.compile(r"^\s*\|.*\|\s*$", re.MULTILINE)

# Default sections where we prefer NOT to insert ads inside
_DEFAULT_BLOCKLIST_SECTIONS = {
    "further reading",
    "summary cheat sheet",
    "performance benchmarks",
}

# ---------------------------- span helpers ---------------------------------

def _yaml_front_matter_span(text: str) -> Optional[Tuple[int, int]]:
    """Return (start,end) of leading YAML front matter if present."""
    m = re.match(r"^---\s*\n[\s\S]*?\n---\s*(\n|$)", text)
    return (m.start(), m.end()) if m else None

def _code_spans(text: str) -> List[Tuple[int, int]]:
    """
    Return list of (start,end) spans for fenced code blocks.
    Supports ``` and ~~~ with optional language, and unclosed fences to EOF.
    """
    spans: List[Tuple[int, int]] = []
    offset = 0
    lines = text.splitlines(keepends=True)

    fence_pat = re.compile(r"^([`~]{3,})(.*)$")  # ```lang or ~~~lang
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
            stripped = ln.lstrip()
            if stripped.startswith(fence_char * fence_len):
                in_fence = False
                spans.append((start_idx, offset + len(ln)))
        offset += len(ln)

    if in_fence:
        spans.append((start_idx, len(text)))
    return spans

def _html_tag_spans(text: str) -> List[Tuple[int, int]]:
    return [m.span() for m in _HTML_TAG_RE.finditer(text or "")]

def _html_comment_spans(text: str) -> List[Tuple[int, int]]:
    return [m.span() for m in _HTML_COMMENT_RE.finditer(text or "")]

def _script_style_spans(text: str) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    spans.extend(m.span() for m in _SCRIPT_BLOCK_RE.finditer(text or ""))
    spans.extend(m.span() for m in _STYLE_BLOCK_RE.finditer(text or ""))
    return spans

def _table_spans(text: str) -> List[Tuple[int, int]]:
    """
    Heuristic: treat any contiguous block of >=2 lines that contain '|' as a table.
    Avoid placing inside or counting words from tables.
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

    def _line_has_table(l: str) -> bool:
        return '|' in l and _TABLE_LINE_RE.match(l) is not None

    i = 0
    n = len(lines)
    while i < n:
        if _line_has_table(lines[i]):
            j = i + 1
            while j < n and _line_has_table(lines[j]):
                j += 1
            if j - i >= 2:
                spans.append((offsets[i], offsets[j]))
            i = j
        else:
            i += 1
    return spans

def _merge_spans(spans: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Merge overlapping/adjacent spans."""
    if not spans:
        return []
    spans = sorted(spans)
    merged = [spans[0]]
    for a, b in spans[1:]:
        la, lb = merged[-1]
        if a <= lb:  # overlap/adjacent
            merged[-1] = (la, max(lb, b))
        else:
            merged.append((a, b))
    return merged

def _protected_spans(text: str) -> List[Tuple[int, int]]:
    """All regions to avoid: front matter, code fences, tables, HTML comments/tags, script/style."""
    spans: List[Tuple[int, int]] = []
    fm = _yaml_front_matter_span(text)
    if fm:
        spans.append(fm)
    spans.extend(_code_spans(text))
    spans.extend(_table_spans(text))
    spans.extend(_html_comment_spans(text))
    spans.extend(_script_style_spans(text))
    spans.extend(_html_tag_spans(text))
    return _merge_spans(spans)

def _in_spans(pos: int, spans: List[Tuple[int, int]]) -> bool:
    return any(a <= pos < b for a, b in spans)

# ---------------------------- section helpers ------------------------------

def _section_spans(text: str) -> List[Tuple[int, int, str]]:
    """
    Return list of (start, end, title_lower) for H2/H3 sections.
    """
    sections: List[Tuple[int, int, str]] = []
    pattern = re.compile(r"(?m)^(#{2,3})\s+(.+)$")
    matches = list(pattern.finditer(text or ""))
    if not matches:
        return sections

    for i, m in enumerate(matches):
        start = m.start()
        title = re.sub(r"\s+", " ", m.group(2)).strip().lower()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections.append((start, end, title))
    return sections

def _index_outside_blocked_section(text: str, idx: int, blocked: Optional[set]) -> int:
    """
    If idx falls inside a blocked section, move idx to the section's end (safe).
    """
    if not blocked:
        return idx
    for a, b, title in _section_spans(text):
        if a <= idx < b and title in blocked:
            return b  # after the blocked section
    return idx

# ---------------------------- word utilities -------------------------------

def _visible_word_positions(text: str) -> List[int]:
    """Start indices of words NOT inside protected spans (code, html, front matter, tables, script/style)."""
    spans = _protected_spans(text)
    positions: List[int] = []
    for m in _WORD_RE.finditer(text or ""):
        if not _in_spans(m.start(), spans):
            positions.append(m.start())
    return positions

def _visible_word_count(text: str) -> int:
    return len(_visible_word_positions(text))

# ---------------------------- placement helpers ----------------------------

def _find_para_end(text: str, start_at: int) -> int:
    """Find end-of-paragraph (first blank line) after start_at; else len(text)."""
    m = re.search(r"\n\s*\n", text[start_at:])
    return (start_at + m.end()) if m else len(text)

def _normalize_pad_around(snippet: str) -> str:
    """Ensure snippet is wrapped with single blank lines on both sides."""
    s = snippet.strip("\n")
    return f"\n\n{s}\n\n"

def _safe_insert(text: str, snippet: str, at_idx: int) -> str:
    """Insert snippet at char index, avoiding duplicate snippet bodies nearby."""
    body = snippet.strip()
    if body and body in text:
        logger.info("Ad snippet already present; skipping duplicate insertion.")
        return text
    pad = _normalize_pad_around(snippet)
    return text[:at_idx] + pad + text[at_idx:]

def _safe_after_n_words(text: str, n_words: int, blocked_sections: Optional[set]) -> Optional[int]:
    """
    Compute a safe insertion point AFTER the paragraph that contains the Nth visible word.
    Avoids placing inside protected spans and blocked sections (by moving to section end).
    """
    if n_words <= 0:
        return None
    positions = _visible_word_positions(text)
    if not positions:
        return None
    idx = positions[min(n_words, len(positions)) - 1]
    spans = _protected_spans(text)

    # If Nth word lands inside a protected span (shouldn't, but be safe), move forward.
    while idx < len(text) and _in_spans(idx, spans):
        idx += 1

    insert_at = _find_para_end(text, idx)
    # If end lands inside a span, hop past it and find next paragraph end.
    while _in_spans(insert_at - 1, spans):
        for a, b in spans:
            if a <= insert_at - 1 < b:
                insert_at = b
                break
        insert_at = _find_para_end(text, insert_at)

    # Respect blocked sections (move to section end if needed)
    insert_at = _index_outside_blocked_section(text, insert_at, blocked_sections)
    return min(insert_at, len(text))

def _safe_before_last_h2(text: str) -> int:
    """
    Safe insertion index BEFORE the last H2 section (outside protected spans).
    Falls back to end-of-document if no suitable H2 exists.
    """
    spans = _protected_spans(text)
    last_h2 = None
    for m in _H2_RE.finditer(text or ""):
        if not _in_spans(m.start(), spans):
            last_h2 = m
    if not last_h2:
        return len(text)

    # Insert a blank line before the H2 block; try to land on a paragraph boundary.
    start = last_h2.start()
    back = text.rfind("\n\n", 0, start)
    insert_at = (back + 2) if back != -1 else max(0, start)

    # Ensure not inside a protected span
    if _in_spans(insert_at, spans):
        for a, b in spans:
            if a <= insert_at < b:
                insert_at = max(0, a)
                break
    return insert_at

# ------------------------- legacy wrappers (kept) -------------------------

def _insert_after_words(text: str, snippet: str, word_index: int) -> str:
    """
    Backward-compatible wrapper that now:
      • Counts visible words only (ignores code/html/front matter/tables/scripts).
      • Inserts after the paragraph that contains the Nth word.
      • Preserves paragraph structure and avoids code fences.
    """
    pos = _safe_after_n_words(text, word_index, blocked_sections=_DEFAULT_BLOCKLIST_SECTIONS)
    if pos is None:
        return f"{text}\n\n{snippet}"
    return _safe_insert(text, snippet, pos)

def _insert_before_last_h2(text: str, snippet: str) -> str:
    """
    Backward-compatible wrapper that:
      • Inserts before the last H2.
      • Avoids protected spans and keeps spacing tidy.
    """
    pos = _safe_before_last_h2(text)
    return _safe_insert(text, snippet, pos)

# ------------------------------ main API ------------------------------------

def inject_ads(content: str, settings: Dict) -> Tuple[str, int]:
    """
    UX-safe AdSense placement (idempotent, code/html/table/front-matter aware, paragraph aligned):
      Placeholder replacement phase (optional), then up to 3 passes:
        1) After ~first_words (default 150) words (end of that paragraph)
        2) Around the midpoint (~50% visible words, min 200 words)
        3) Before the last H2 (or end if none)

    Respects:
      settings["adsense"]["placement_markers"] : List[str]
      settings["adsense"]["max_ads_per_article"] : int
      Optional tuning:
      settings["adsense"]["first_words"] : int (default 150)
      settings["adsense"]["min_words_for_second_ad"] : int (default 200)
      settings["adsense"]["min_article_words"] : int (default 0 → no skip)
      settings["adsense"]["placeholders"] : List[str]  # placeholders to replace first
      settings["adsense"]["blocklist_sections"] : List[str]  # section titles to avoid
    """
    ads_cfg = settings.get("adsense", {}) or {}
    markers: List[str] = [m for m in (ads_cfg.get("placement_markers") or []) if (m or "").strip()]
    max_ads = int(ads_cfg.get("max_ads_per_article", 3) or 0)

    if not content or not markers or max_ads <= 0:
        logger.info("Ad injection skipped: no content, markers, or max_ads.")
        return content, 0

    # Tunables / heuristics
    first_words = int(ads_cfg.get("first_words", 150))
    min_words_second = int(ads_cfg.get("min_words_for_second_ad", 200))
    min_article_words = int(ads_cfg.get("min_article_words", 0))

    # Blocklist sections (normalized)
    blist_cfg = {re.sub(r"\s+", " ", s).strip().lower()
                 for s in (ads_cfg.get("blocklist_sections") or []) if (s or "").strip()}
    blocked_sections = blist_cfg or _DEFAULT_BLOCKLIST_SECTIONS

    total_words = _visible_word_count(content)
    if min_article_words and total_words < min_article_words:
        logger.info("Article below min_article_words threshold; skipping ad injection.")
        return content, 0

    # Light throttle on short pieces (by visible words)
    if total_words < 600:
        max_ads = min(max_ads, 2)

    # Avoid adding the same marker twice if already present verbatim
    remaining_markers: List[str] = []
    for m in markers:
        if (m or "").strip() and (m.strip() in content):
            logger.info("Marker already present; skipping duplicate.")
        else:
            remaining_markers.append(m)
    if not remaining_markers:
        return content, 0

    placed = 0
    out = content

    # Phase 0: Replace placeholders (if provided)
    placeholders: List[str] = [p for p in (ads_cfg.get("placeholders") or []) if (p or "").strip()]
    if placeholders:
        i_marker = 0
        for ph in placeholders:
            if placed >= max_ads or i_marker >= len(remaining_markers):
                break
            # Replace first occurrence of placeholder with the marker
            idx = out.find(ph)
            if idx != -1:
                out = out[:idx] + _normalize_pad_around(remaining_markers[i_marker]) + out[idx + len(ph):]
                placed += 1
                i_marker += 1
        # Remove the markers we've already used from the front of the list
        remaining_markers = remaining_markers[placed:]

    # 1) After ~first_words
    if placed < max_ads and placed < len(remaining_markers):
        pos = _safe_after_n_words(out, first_words, blocked_sections=blocked_sections)
        if pos is not None:
            out = _safe_insert(out, remaining_markers[placed], pos)
            placed += 1

    # 2) At ~50% (min guard)
    if placed < max_ads and placed < len(remaining_markers):
        # Recompute total_words after first insertion, but word positions exclude protected regions anyway
        total_words = _visible_word_count(out)
        mid_idx_words = max(min_words_second, int(total_words * 0.5))
        pos2 = _safe_after_n_words(out, mid_idx_words, blocked_sections=blocked_sections)
        if pos2 is not None:
            out = _safe_insert(out, remaining_markers[placed], pos2)
            placed += 1

    # 3) Before last H2 (or end)
    if placed < max_ads and placed < len(remaining_markers):
        pos3 = _safe_before_last_h2(out)
        # Make sure we are not inside a blocked section (we're inserting BEFORE a section)
        out = _safe_insert(out, remaining_markers[placed], pos3)
        placed += 1

    return out, placed
