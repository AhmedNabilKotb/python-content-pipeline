# article_challenge_injector.py

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, List, Tuple, Set

# --- Logging ---------------------------------------------------------------
LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Avoid duplicating root handlers
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(LOG_DIR / "article_challenge_injector.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
logger = logging.getLogger(__name__)

_WORD_RE = re.compile(r"\b\w+\b", re.UNICODE)

# --- Config ----------------------------------------------------------------
@dataclass
class ArticleChallengeConfig:
    min_word_count_for_injection: int = 1000
    difficulty_level: str = "INTERMEDIATE"  # EASY | INTERMEDIATE | ADVANCED
    position: str = "FOOTER"  # "FOOTER" | "AFTER_INTRO"
    # Idempotency markers (prevents duplicate injections; enables replacement)
    marker_start: str = "<!-- CHALLENGE:START -->"
    marker_end: str = "<!-- CHALLENGE:END -->"
    # If a challenge already exists: "skip" (default) or "replace"
    on_existing: str = "skip"
    # Optional exclusions (lowercased substring match against whole article)
    exclude_sections_keywords: List[str] = field(default_factory=list)
    # Template supports {difficulty}, {topic}, {marker_start}, {marker_end}
    template: str = (
        "{marker_start}\n"
        "## Your Challenge!\n\n"
        "💡 Ready to test your understanding?\n\n"
        "**Difficulty:** {difficulty}\n\n"
        "Build a small, self-contained solution related to **{topic}**:\n\n"
        "1) Define the problem and constraints.\n"
        "2) Write clean, idiomatic Python with tests.\n"
        "3) Consider performance and edge cases.\n\n"
        "_Optional_: share your approach, benchmarks, and trade-offs.\n"
        "{marker_end}"
    )
    # Optional: if no topic passed, try to infer from first H1
    infer_topic_from_h1: bool = True
    # Optional: treat tildes (~~~) fences as code, too
    support_tilde_fences: bool = True
    # Avoid inserting in these H2/H3 sections (case-insensitive)
    blocklist_sections: List[str] = field(
        default_factory=lambda: ["further reading", "summary cheat sheet", "performance benchmarks"]
    )


# --- Injector --------------------------------------------------------------
class ArticleChallengeInjector:
    """
    Adds a single 'Your Challenge!' section to a markdown article.

    Pipeline-safe features:
      • Protected-region aware (front matter, code fences, tables, HTML tags/comments, <script>/<style>).
      • Idempotent via HTML comment markers; optional replacement.
      • Minimum word-count uses visible words only.
      • Never inserts inside protected spans.
      • AFTER_INTRO = after first non-blocklisted H2 (or first substantial paragraph).
      • Exclusion keywords (substring, case-insensitive).
      • Optional topic inference from first H1 if not provided.
      • ~~~ fences support is configurable.
    """

    def __init__(self, app_settings: Dict, openai_client: Optional[object] = None):
        cfg = (app_settings or {}).get("article_challenge_injector", {}) or {}

        self.config = ArticleChallengeConfig(
            min_word_count_for_injection=int(cfg.get("min_word_count_for_injection", 1000)),
            difficulty_level=str(cfg.get("difficulty_level", "INTERMEDIATE")).upper(),
            position=str(cfg.get("position", "FOOTER")).upper(),
            marker_start=str(cfg.get("marker_start", ArticleChallengeConfig.marker_start)),
            marker_end=str(cfg.get("marker_end", ArticleChallengeConfig.marker_end)),
            on_existing=str(cfg.get("on_existing", "skip")).lower(),
            exclude_sections_keywords=[k.lower() for k in cfg.get("exclude_sections_keywords", [])] or [],
            template=str(cfg.get("template", ArticleChallengeConfig.template)),
            infer_topic_from_h1=bool(cfg.get("infer_topic_from_h1", True)),
            support_tilde_fences=bool(cfg.get("support_tilde_fences", True)),
            blocklist_sections=[s.lower().strip() for s in cfg.get("blocklist_sections", [])] or
                               ArticleChallengeConfig().blocklist_sections,
        )

        # Reserved for future LLM-generated challenges
        self.openai_client = openai_client

        # Validate knobs
        if self.config.on_existing not in {"skip", "replace"}:
            raise ValueError("on_existing must be 'skip' or 'replace'.")
        if self.config.position not in {"FOOTER", "AFTER_INTRO"}:
            raise ValueError("position must be 'FOOTER' or 'AFTER_INTRO'.")
        if self.config.difficulty_level not in {"EASY", "INTERMEDIATE", "ADVANCED"}:
            logger.warning("Unknown difficulty_level; defaulting to INTERMEDIATE.")
            self.config.difficulty_level = "INTERMEDIATE"

    # ---------- public API ----------

    def inject_challenge(self, content: str, topic: Optional[str]) -> str:
        """
        Insert a 'Your Challenge!' section if not already present (idempotent).
        Honors config.on_existing = 'skip' | 'replace'.
        """
        if not content:
            return content

        # Respect exclusions
        low = content.lower()
        if any(k in low for k in self.config.exclude_sections_keywords):
            logger.info("Challenge injection skipped: content contains excluded keywords.")
            return content

        has_existing = self._has_challenge(content)
        if has_existing and self.config.on_existing == "skip":
            logger.info("Challenge section already present. Skipping injection.")
            return content
        if has_existing and self.config.on_existing == "replace":
            content = self._remove_existing(content)

        # Safety: enforce minimum length based on visible words only
        if self._visible_word_count(content) < self.config.min_word_count_for_injection:
            logger.info(
                "Content length below challenge threshold (%d visible words). Skipping.",
                self.config.min_word_count_for_injection,
            )
            return content

        # Topic fallback
        if (not topic) and self.config.infer_topic_from_h1:
            topic = self._infer_topic_from_h1(content) or "this article"

        section = self._render_section(topic or "this article")

        if self.config.position == "AFTER_INTRO":
            updated = self._insert_after_intro(content, section)
        else:  # FOOTER
            updated = self._append_footer(content, section)

        logger.info("Inserted challenge section (%s).", self.config.position)
        return updated

    # ---------- formatting ----------

    class _SafeDict(dict):
        def __missing__(self, key):
            return ""

    def _render_section(self, topic: str) -> str:
        txt = self.config.template.format_map(
            self._SafeDict(
                topic=topic,
                difficulty=self.config.difficulty_level,
                marker_start=self.config.marker_start,
                marker_end=self.config.marker_end,
            )
        ).strip()
        # Keep a blank line padding before section when inserting
        return "\n\n" + txt + "\n"

    # ---------- protected spans / visibility ----------

    @staticmethod
    def _yaml_front_matter_span(text: str) -> Optional[Tuple[int, int]]:
        m = re.match(r"^---\s*\n[\s\S]*?\n---\s*(\n|$)", text)
        return (m.start(), m.end()) if m else None

    def _code_spans(self, text: str) -> List[Tuple[int, int]]:
        """
        Return [ (start_index, end_index) ] for fenced code blocks.
        Supports ``` and optionally ~~~ fences with any language tag; requires matching fence char.
        Respects config.support_tilde_fences.
        """
        allow_tilde = self.config.support_tilde_fences
        spans: List[Tuple[int, int]] = []
        offset = 0
        lines = text.splitlines(keepends=True)

        fence_pat = re.compile(r"^([`~]{3,})(.*)$")  # start of fence line
        in_fence = False
        fence_char = ""
        fence_len = 0
        start_idx = 0

        for ln in lines:
            if not in_fence:
                m = fence_pat.match(ln.strip())
                if m:
                    ch = m.group(1)[0]
                    if ch == "~" and not allow_tilde:
                        # treat ~~~ as normal text if tilde fences disabled
                        pass
                    else:
                        in_fence = True
                        fence_char = ch
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

    @staticmethod
    def _html_comment_spans(text: str) -> List[Tuple[int, int]]:
        return [m.span() for m in re.finditer(r"<!--[\s\S]*?-->", text, re.IGNORECASE)]

    @staticmethod
    def _script_style_spans(text: str) -> List[Tuple[int, int]]:
        spans: List[Tuple[int, int]] = []
        for m in re.finditer(r"<script\b[^>]*>[\s\S]*?</script\s*>", text, re.IGNORECASE):
            spans.append(m.span())
        for m in re.finditer(r"<style\b[^>]*>[\s\S]*?</style\s*>", text, re.IGNORECASE):
            spans.append(m.span())
        return spans

    @staticmethod
    def _html_tag_spans(text: str) -> List[Tuple[int, int]]:
        return [m.span() for m in re.finditer(r"<[^>]+>", text or "")]

    @staticmethod
    def _table_spans(text: str) -> List[Tuple[int, int]]:
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

        line_re = re.compile(r"^\s*\|.*\|\s*$")
        def is_table_line(s: str) -> bool:
            return "|" in s and bool(line_re.match(s))

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
        return sum(1 for m in _WORD_RE.finditer(text or "") if not self._pos_in_spans(m.start(), spans))

    # --- challenge detection/removal ---

    def _has_challenge(self, md: str) -> bool:
        # Prefer markers; fallback to heading detection
        start = re.escape(self.config.marker_start)
        end = re.escape(self.config.marker_end)
        pat = re.compile(start + r".*?" + end, flags=re.DOTALL)
        if pat.search(md):
            return True
        return bool(re.search(r"^\s{0,3}##\s+your\s+challenge!?\s*$", md, flags=re.IGNORECASE | re.MULTILINE))

    def _remove_existing(self, md: str) -> str:
        start = re.escape(self.config.marker_start)
        end = re.escape(self.config.marker_end)
        pat = re.compile(rf"\s*{start}[\s\S]*?{end}\s*", flags=re.DOTALL)
        new_md, n = pat.subn("", md)
        if n:
            logger.info("Removed %d existing challenge block(s) by marker.", n)
            return new_md
        head_pat = re.compile(
            r"(?ms)^\s{0,3}##\s+your\s+challenge!?[\s\S]*?(?=^\s{0,3}#{1,6}\s|\Z)",
            flags=re.IGNORECASE,
        )
        new_md2, n2 = head_pat.subn("", md)
        if n2:
            logger.info("Removed %d existing challenge block(s) by heading.", n2)
            return new_md2
        return md

    # --- insertion helpers ---

    @staticmethod
    def _append_footer(md: str, section: str) -> str:
        return md.rstrip() + section

    def _h2_h3_sections(self, md: str) -> List[Tuple[int, int, str]]:
        """
        Return list of (start, end, title_lower) for H2/H3 sections.
        """
        sections: List[Tuple[int, int, str]] = []
        pattern = re.compile(r"(?m)^(#{2,3})\s+(.+)$")
        matches = list(pattern.finditer(md or ""))
        for i, m in enumerate(matches):
            start = m.start()
            title = re.sub(r"\s+", " ", m.group(2)).strip().lower()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(md)
            sections.append((start, end, title))
        return sections

    def _find_after_h2_block(self, md: str) -> Optional[int]:
        """
        Find insertion index right after the first non-blocklisted H2 block.
        Skip protected regions for the final cursor.
        """
        spans = self._protected_spans(md)
        blocked: Set[str] = set(self.config.blocklist_sections)

        for m in re.finditer(r"(?m)^##\s+(.+)$", md):
            title = re.sub(r"\s+", " ", m.group(1)).strip().lower()
            if title in blocked:
                continue
            if self._pos_in_spans(m.start(), spans):
                continue  # shouldn't happen, but safe

            # Move to end of the paragraph after this H2
            insert_pos = m.end()
            after = md[insert_pos:]
            gap = re.search(r"\n\s*\n", after)
            if gap:
                insert_pos += gap.end()
            # If we landed in a protected span, hop to the end of that span and find next paragraph break
            while self._pos_in_spans(insert_pos - 1, spans) and insert_pos < len(md):
                for a, b in spans:
                    if a <= insert_pos - 1 < b:
                        insert_pos = b
                        break
                nxt = re.search(r"\n\s*\n", md[insert_pos:])
                insert_pos = (insert_pos + nxt.end()) if nxt else len(md)
            return min(insert_pos, len(md))

        return None

    def _insert_after_intro(self, md: str, section: str) -> str:
        """
        Insert after the first non-blocklisted H2 block if present and outside protected regions.
        Fallbacks:
          1) After first substantial paragraph (≥ 50 visible words) outside code.
          2) Append to footer.
        """
        # 1) After H2 block
        pos = self._find_after_h2_block(md)
        if pos is not None:
            return md[:pos] + section + md[pos:]

        # 2) After first substantial paragraph (≥ 50 words visible in that block)
        spans = self._protected_spans(md)
        parts = [p for p in re.split(r"(\n\s*\n)", md)]  # keep separators
        acc = 0
        for i in range(0, len(parts), 2):
            block = parts[i]
            sep = parts[i + 1] if i + 1 < len(parts) else ""
            if not block.strip():
                acc += len(block) + len(sep)
                continue
            # Skip if block begins inside a protected span
            if self._pos_in_spans(acc, spans):
                acc += len(block) + len(sep)
                continue
            # Count visible words just within this block (excluding inner code/html/tables)
            block_spans = self._protected_spans(block)
            wc = sum(1 for m in _WORD_RE.finditer(block) if not self._pos_in_spans(m.start(), block_spans))
            if wc >= 50:
                pos2 = acc + len(block)
                return md[:pos2] + "\n\n" + section + "\n\n" + md[pos2:]
            acc += len(block) + len(sep)

        # 3) Footer
        return self._append_footer(md, section)

    # --- topic inference ---
    @staticmethod
    def _infer_topic_from_h1(md: str) -> Optional[str]:
        m = re.search(r"(?m)^\s*#\s+(.+?)\s*$", md)
        if m:
            # Strip trailing markdown link or anchor clutter if present
            topic = re.sub(r"\s*<a[^>]*>.*?</a>\s*$", "", m.group(1)).strip()
            topic = re.sub(r"\s*{#.*}\s*$", "", topic).strip()
            return topic or None
        return None


__all__ = ["ArticleChallengeInjector", "ArticleChallengeConfig"]
