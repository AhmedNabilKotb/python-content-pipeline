# chart_generator.py

import os
import io
import json
import re
import base64
import hashlib
from pathlib import Path
from typing import Dict, Optional, Tuple, List

# Ensure logs directory exists before configuring logging
Path("logs").mkdir(parents=True, exist_ok=True)

# Headless backend (PythonAnywhere / servers)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from openai import OpenAI
import logging
from dotenv import load_dotenv, find_dotenv

# --- Logging ---------------------------------------------------------------
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("logs/chart_generation.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
logger = logging.getLogger(__name__)


class ChartGenerator:
    """
    Heuristic-first chart extractor to avoid LLM calls.
    Falls back to a small, JSON-only LLM extraction if needed.
    Caches results per-article to avoid repeat API usage.

    Safeguards & features:
      • Never inserts inside protected regions (front matter, code fences, tables, HTML, comments, script/style).
      • Idempotent markers allow replacement.
      • Detects markdown tables, CSV-ish blocks, bullets/ordered lists, quarters, inline percentages.
      • Robust numeric parsing (thousands separators, %, k/m/b suffixes).
      • Chooses pie/line/bar sensibly; trims long labels; aggregates tail to 'Other'.
      • Prefers insertion right after the data source block if found.
      • Optional placeholders supported: <!-- CHART-HOOK -->, <!-- CHARTGEN-PLACEHOLDER -->
    """

    MONTHS = ("jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec")
    MARKER_START = "<!-- CHARTGEN:START -->"
    MARKER_END = "<!-- CHARTGEN:END -->"

    # Recognized placeholders (first match wins)
    PLACEHOLDERS = ["<!-- CHART-HOOK -->", "<!-- CHARTGEN-PLACEHOLDER -->"]

    # Blocklisted sections to avoid dropping into (case-insensitive)
    BLOCKLIST_SECTIONS = {"further reading", "summary cheat sheet", "performance benchmarks", "references"}

    def __init__(self, openai_client: Optional[OpenAI] = None, *, enable_llm_fallback: Optional[bool] = None):
        self.logger = logging.getLogger(__name__)
        # Enable/disable LLM fallback via env; default True but we try to never call it
        if enable_llm_fallback is None:
            val = os.getenv("CHARTGEN_ENABLE_LLM", "1").strip().lower()
            enable_llm_fallback = val not in ("0", "false", "no")
        self.enable_llm_fallback = enable_llm_fallback

        # Cache dir
        self.cache_dir = Path("cache/charts")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        if openai_client:
            self.client = openai_client
            self.logger.info("ChartGenerator initialized with provided OpenAI client.")
        else:
            load_dotenv(find_dotenv(usecwd=True))
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                # No client; we'll simply skip LLM fallback
                self.client = None
                if self.enable_llm_fallback:
                    self.logger.warning("OPENAI_API_KEY missing. LLM fallback will be disabled.")
                self.enable_llm_fallback = False
            else:
                self.client = OpenAI(api_key=api_key)
                self.logger.info("ChartGenerator initialized its own OpenAI client.")

    # ------------------- Protected regions helpers -------------------

    _WORD_RE = re.compile(r"\b\w+\b", re.UNICODE)
    _H2H3_RE = re.compile(r"(?m)^(#{2,3})\s+(.+)$")
    _TABLE_LINE_RE = re.compile(r"^\s*\|.*\|\s*$", re.MULTILINE)
    _HTML_TAG_RE = re.compile(r"<[^>]+>")
    _HTML_COMMENT_RE = re.compile(r"<!--[\s\S]*?-->", re.IGNORECASE)
    _SCRIPT_BLOCK_RE = re.compile(r"<script\b[^>]*>[\s\S]*?</script\s*>", re.IGNORECASE)
    _STYLE_BLOCK_RE = re.compile(r"<style\b[^>]*>[\s\S]*?</style\s*>", re.IGNORECASE)

    @staticmethod
    def _yaml_front_matter_span(text: str) -> Optional[Tuple[int, int]]:
        m = re.match(r"^---\s*\n[\s\S]*?\n---\s*(\n|$)", text)
        return (m.start(), m.end()) if m else None

    @staticmethod
    def _strip_code_blocks(text: str) -> str:
        # Remove fenced code blocks of ``` and ~~~
        text = re.sub(r"```[\s\S]*?```", "", text)
        text = re.sub(r"~~~[\s\S]*?~~~", "", text)
        return text

    @staticmethod
    def _code_spans(text: str) -> List[Tuple[int, int]]:
        """
        Return spans for ``` and ~~~ fenced code blocks.
        """
        spans: List[Tuple[int, int]] = []
        offset = 0
        lines = text.splitlines(keepends=True)

        fence_pat = re.compile(r"^([`~]{3,})(.*)$")  # fence line
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

    def _html_comment_spans(self, text: str) -> List[Tuple[int, int]]:
        return [m.span() for m in self._HTML_COMMENT_RE.finditer(text or "")]

    def _script_style_spans(self, text: str) -> List[Tuple[int, int]]:
        spans: List[Tuple[int, int]] = []
        spans.extend(m.span() for m in self._SCRIPT_BLOCK_RE.finditer(text or ""))
        spans.extend(m.span() for m in self._STYLE_BLOCK_RE.finditer(text or ""))
        return spans

    def _html_tag_spans(self, text: str) -> List[Tuple[int, int]]:
        return [m.span() for m in self._HTML_TAG_RE.finditer(text or "")]

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

    # ---------------- Heuristic extraction (no LLM) -----------------

    @staticmethod
    def _lines_with_numbers(text: str) -> List[str]:
        # keep short lines with digits or %; drop code blocks first
        text = ChartGenerator._strip_code_blocks(text)
        lines = [ln.rstrip() for ln in text.splitlines()]
        return [ln.strip() for ln in lines if any(ch.isdigit() for ch in ln) and len(ln) <= 180]

    @staticmethod
    def _parse_number(raw: str) -> Optional[float]:
        """
        Robust numeric parser that handles:
          - thousands separators: 1,234 or 1 234
          - percentages (ignored for numeric value): '45%'
          - suffixes: 1.2k, 3.4M, 5b
          - time units ignored (ms, s) — value stays numeric
        """
        if raw is None:
            return None
        s = str(raw).strip().lower()
        neg = s.startswith("-")
        s = s.lstrip("+-")
        multiplier = 1.0
        if s.endswith("k"):
            multiplier = 1e3
            s = s[:-1]
        elif s.endswith("m"):
            multiplier = 1e6
            s = s[:-1]
        elif s.endswith("b"):
            multiplier = 1e9
            s = s[:-1]
        s = s.rstrip("%")
        # remove thousands separators (comma or space)
        s = s.replace(",", "").replace(" ", "")
        try:
            val = float(s) * multiplier
            return -val if neg else val
        except Exception:
            return None

    @staticmethod
    def _is_time_axis(labels: List[str]) -> bool:
        if not labels:
            return False
        low = [l.strip().lower() for l in labels]
        # months (3+ unique monthish tokens or majority)
        month_hits = sum(any(m == l[:3] for m in ChartGenerator.MONTHS) for l in low)
        if month_hits >= max(3, len(labels) // 2):
            return True
        # quarters
        if sum(bool(re.fullmatch(r"q[1-4]", l)) for l in low) >= max(3, len(labels) // 2):
            return True
        # years (YYYY)
        if sum(bool(re.fullmatch(r"\d{4}", l)) for l in low) >= max(3, len(labels) // 2):
            return True
        return False

    def _decide_chart_type(self, labels: List[str], values: List[float], *, percent_hint: bool = False) -> str:
        if self._is_time_axis(labels):
            return "line"
        if values and all(0.0 <= v <= 100.0 for v in values) and 95 <= sum(values) <= 105:
            return "pie"
        if percent_hint:
            return "bar"
        return "bar"

    # ---- extractors now return optional 'source_span' for placement ----

    def _extract_kv_pairs(self, text: str, full: str) -> Optional[Dict]:
        """
        Extract lines like: Q1: 15.2 seconds ; Django - 45% ; Item — 1,200
        """
        pattern = re.compile(
            r"^\s*([A-Za-z][\w\s\/&+\-().]+?)\s*[:\-–—]\s*([+-]?\d+(?:[\s,]?\d{3})*(?:\.\d+)?%?[kmb]?)\b.*$",
            re.IGNORECASE | re.MULTILINE,
        )
        matches = list(pattern.finditer(text))
        if len(matches) < 3:
            return None

        dedup: Dict[str, float] = {}
        any_percent = False
        # For span, cover from first to last match in FULL content
        cover_start, cover_end = None, None

        for m in matches:
            label = re.sub(r"\s+", " ", m.group(1)).strip(" .,:;")
            rawnum = m.group(2)
            if not label or label.lower() in {"total", "sum"}:
                continue
            val = self._parse_number(rawnum)
            if val is None:
                continue
            any_percent = any_percent or rawnum.strip().endswith("%")
            key = label.lower()
            dedup[key] = dedup.get(key, 0.0) + val
            # find this matched line in the full text for span coverage
            seg = m.group(0)
            idx = full.find(seg)  # first occurrence
            if idx != -1:
                s, e = idx, idx + len(seg)
                cover_start = min(cover_start, s) if cover_start is not None else s
                cover_end = max(cover_end, e) if cover_end is not None else e

        if len(dedup) < 3:
            return None

        labels = list(dedup.keys())
        values = [dedup[k] for k in labels]
        ctype = self._decide_chart_type(labels, values, percent_hint=any_percent)

        return {
            "chart_type": ctype,
            "title": "Extracted Metrics",
            "x_label": "Category" if ctype != "pie" else "",
            "y_label": "Value" if ctype != "pie" else "",
            "labels": labels,
            "data": values,
            "source_span": (cover_start, cover_end) if (cover_start is not None and cover_end is not None) else None,
        }

    def _extract_quarters(self, text: str, full: str) -> Optional[Dict]:
        pattern = re.compile(r"\b(Q[1-4])\s*[:\-]\s*([+-]?\d+(?:[\s,]?\d{3})*(?:\.\d+)?%?[kmb]?)\b", re.I)
        pairs = list(pattern.finditer(text))
        vals = [self._parse_number(p.group(2)) for p in pairs]
        if pairs and all(v is not None for v in vals) and len(pairs) >= 3:
            labels = [p.group(1).upper() for p in pairs]
            data = [float(v) for v in vals]  # type: ignore
            ctype = "line" if self._is_time_axis(labels) else "bar"
            # span coverage in full
            cover_start = cover_end = None
            for m in pairs:
                seg = m.group(0)
                idx = full.find(seg)
                if idx != -1:
                    s, e = idx, idx + len(seg)
                    cover_start = min(cover_start, s) if cover_start is not None else s
                    cover_end = max(cover_end, e) if cover_end is not None else e
            return {
                "chart_type": ctype,
                "title": "Quarterly Values",
                "x_label": "Quarter",
                "y_label": "Value",
                "labels": labels,
                "data": data,
                "source_span": (cover_start, cover_end) if (cover_start is not None and cover_end is not None) else None,
            }
        return None

    def _extract_inline_percentages(self, text: str, full: str) -> Optional[Dict]:
        """
        Matches simple phrases like:
          Flask is used by 35%, Django by 45%, FastAPI by 15%, others by 5%.
        """
        pattern = re.compile(
            r"\b([A-Z][A-Za-z0-9+/&\-\s]{2,}?)\s+(?:is|are)?\s*(?:used\s+by|at|with)?\s*(\d+(?:\.\d+)?)\s*%",
            re.M,
        )
        pairs = list(pattern.finditer(text))
        seen, labels, data = set(), [], []
        cover_start = cover_end = None
        for m in pairs:
            label = re.sub(r"\s+", " ", m.group(1)).strip(" ,.")
            pct = m.group(2)
            key = label.lower()
            if label and key not in seen:
                seen.add(key)
                labels.append(label)
                data.append(float(pct))
            seg = m.group(0)
            idx = full.find(seg)
            if idx != -1:
                s, e = idx, idx + len(seg)
                cover_start = min(cover_start, s) if cover_start is not None else s
                cover_end = max(cover_end, e) if cover_end is not None else e
        if len(labels) >= 3 and 95 <= sum(data) <= 105:
            return {
                "chart_type": "pie",
                "title": "Distribution",
                "x_label": "",
                "y_label": "",
                "labels": labels,
                "data": data,
                "source_span": (cover_start, cover_end) if (cover_start is not None and cover_end is not None) else None,
            }
        return None

    def _extract_markdown_table(self, text: str, full: str) -> Optional[Dict]:
        """
        Parse simple markdown tables; if multiple tables exist, pick the one
        with the strongest numeric column signal.
        """
        text_nocode = self._strip_code_blocks(text)
        table_re = re.compile(
            r"(^\s*\|.+?\|\s*$\n^\s*\|[\s:\-\|]+\|\s*$\n(?:^\s*\|.+?\|\s*$\n?){2,})",
            re.MULTILINE,
        )
        matches = table_re.findall(text_nocode)
        if not matches:
            return None

        best = None
        best_score = -1

        def _num_val(s: str) -> Optional[float]:
            return self._parse_number(s)

        for tbl in matches:
            rows = [r.rstrip() for r in tbl.strip().splitlines()]
            header = [c.strip() for c in rows[0].strip("|").split("|")]
            body = rows[2:]  # skip header + divider
            cells = [[c.strip() for c in r.strip("|").split("|")] for r in body]
            if not cells or len(cells[0]) < 2:
                continue

            col_scores = []
            for j in range(len(cells[0])):
                nums = sum(_num_val(row[j]) is not None for row in cells)
                col_scores.append(nums)

            data_col = max(range(len(col_scores)), key=lambda j: col_scores[j])
            label_col = next((j for j in range(len(cells[0])) if j != data_col), None)
            if label_col is None:
                continue

            labels, values = [], []
            for row in cells:
                lbl = re.sub(r"\s+", " ", row[label_col]).strip(" .,:;")
                val = _num_val(row[data_col])
                if lbl and (val is not None):
                    labels.append(lbl)
                    values.append(val)

            if len(labels) < 3:
                continue

            score = col_scores[data_col]
            if score > best_score:
                percentish = all(0.0 <= v <= 100.0 for v in values) and 95 <= sum(values) <= 105
                ctype = "pie" if percentish else ("line" if self._is_time_axis(labels) else "bar")
                title = "Table Summary"
                if 0 <= data_col < len(header) and 0 <= label_col < len(header):
                    title = f"{header[data_col]} by {header[label_col]}".strip()

                # find this table block in the full text to get span
                start_idx = full.find(tbl)
                span = (start_idx, start_idx + len(tbl)) if start_idx != -1 else None

                best = {
                    "chart_type": ctype,
                    "title": title,
                    "x_label": header[label_col] if ctype != "pie" else "",
                    "y_label": header[data_col] if ctype != "pie" else "",
                    "labels": labels,
                    "data": values,
                    "source_span": span,
                }
                best_score = score

        return best

    def _extract_bulleted_or_ordered_list(self, text: str, full: str) -> Optional[Dict]:
        """
        Extract from bullets and ordered lists like:
          - Django: 45%
          * Flask - 35
          • FastAPI — 15
          1. Pandas — 120
        """
        pat = re.compile(
            r"^\s*(?:[-*•]|\d+\.)\s*([A-Za-z][\w\s/&+\-().]+?)\s*[:\-–—]\s*([+-]?\d+(?:[\s,]?\d{3})*(?:\.\d+)?%?[kmb]?)\s*$",
            re.MULTILINE,
        )
        matches = list(pat.finditer(text))
        if len(matches) < 3:
            return None

        dedup: Dict[str, float] = {}
        percent_hint = False
        cover_start = cover_end = None

        for m in matches:
            label = re.sub(r"\s+", " ", m.group(1)).strip(" .,:;")
            rawnum = m.group(2)
            if not label:
                continue
            v = self._parse_number(rawnum)
            if v is None:
                continue
            percent_hint = percent_hint or rawnum.strip().endswith("%")
            k = label.lower()
            dedup[k] = dedup.get(k, 0.0) + v

            seg = m.group(0)
            idx = full.find(seg)
            if idx != -1:
                s, e = idx, idx + len(seg)
                cover_start = min(cover_start, s) if cover_start is not None else s
                cover_end = max(cover_end, e) if cover_end is not None else e

        if len(dedup) < 3:
            return None

        labels = list(dedup.keys())
        values = [dedup[k] for k in labels]
        ctype = self._decide_chart_type(labels, values, percent_hint=percent_hint)
        return {
            "chart_type": ctype,
            "title": "List Metrics",
            "x_label": "Category" if ctype != "pie" else "",
            "y_label": "Value" if ctype != "pie" else "",
            "labels": labels,
            "data": values,
            "source_span": (cover_start, cover_end) if (cover_start is not None and cover_end is not None) else None,
        }

    def _extract_csvish_block(self, text: str, full: str) -> Optional[Dict]:
        """
        Very simple CSV-ish detector: lines with commas and stable column count (>=2),
        use most-numeric column as data and a sensible label column.
        """
        lines = [ln.strip() for ln in self._strip_code_blocks(text).splitlines() if "," in ln]
        if len(lines) < 3:
            return None
        # Heuristic: use the largest set of consecutive lines with same comma count
        groups: List[List[str]] = []
        current: List[str] = []
        last_cols = None
        for ln in lines:
            cols = ln.count(",") + 1
            if last_cols is None or cols == last_cols:
                current.append(ln)
            else:
                if len(current) >= 3:
                    groups.append(current)
                current = [ln]
            last_cols = cols
        if len(current) >= 3:
            groups.append(current)
        if not groups:
            return None

        best = None
        best_score = -1
        for g in groups:
            grid = [[c.strip() for c in row.split(",")] for row in g]
            ncols = len(grid[0])
            # numeric score per column
            scores = []
            for j in range(ncols):
                scores.append(sum(self._parse_number(r[j]) is not None for r in grid))
            data_col = max(range(ncols), key=lambda j: scores[j])
            label_col = next((j for j in range(ncols) if j != data_col), None)
            if label_col is None:
                continue
            labels, values = [], []
            for r in grid:
                lbl = re.sub(r"\s+", " ", r[label_col]).strip(" .,:;")
                val = self._parse_number(r[data_col])
                if lbl and (val is not None):
                    labels.append(lbl)
                    values.append(val)
            if len(labels) < 3:
                continue
            score = scores[data_col]
            if score > best_score:
                ctype = "pie" if (all(0.0 <= v <= 100.0 for v in values) and 95 <= sum(values) <= 105) \
                        else ("line" if self._is_time_axis(labels) else "bar")
                src_text = "\n".join(g)
                start_idx = full.find(src_text)
                span = (start_idx, start_idx + len(src_text)) if start_idx != -1 else None
                best = {
                    "chart_type": ctype,
                    "title": "CSV Summary",
                    "x_label": "Category" if ctype != "pie" else "",
                    "y_label": "Value" if ctype != "pie" else "",
                    "labels": labels,
                    "data": values,
                    "source_span": span,
                }
                best_score = score
        return best

    def _try_heuristics(self, content: str) -> Optional[Dict]:
        """Run extractors on the full content; each will internally skip code blocks."""
        if not any(ch.isdigit() for ch in content):
            return None
        # Try more specific -> generic
        for extractor in (
            self._extract_markdown_table,
            self._extract_csvish_block,
            self._extract_quarters,
            self._extract_inline_percentages,
            self._extract_bulleted_or_ordered_list,
            self._extract_kv_pairs,
        ):
            data = extractor(content, content)
            if data:
                return self._limit_categories(data)
        return None

    # ---------------- Presentation helpers -----------------

    @staticmethod
    def _shorten_labels(labels: List[str], max_len: int = 32) -> List[str]:
        out = []
        for l in labels:
            l = re.sub(r"\s+", " ", l).strip()
            out.append((l[: max_len - 1] + "…") if len(l) > max_len else l)
        return out

    @staticmethod
    def _limit_categories(data: Dict, max_points: int = 12) -> Dict:
        labels = data.get("labels", [])
        values = data.get("data", [])
        if len(labels) <= max_points:
            return data

        # For pie/bar: keep top N and aggregate the rest as "Other"
        pairs = list(zip(labels, values))
        pairs.sort(key=lambda x: x[1], reverse=True)
        head = pairs[: max_points - 1]
        tail = pairs[max_points - 1 :]
        other_sum = sum(v for _, v in tail)

        data["labels"] = [l for l, _ in head] + ["Other"]
        data["data"] = [v for _, v in head] + [other_sum]
        return data

    # ---------------- Cache helpers -----------------

    def _cache_path(self, slug: str, content: str) -> Path:
        # hash only the numeric context to keep cache stable across small edits
        ctx = "\n".join(self._lines_with_numbers(content))
        key = hashlib.sha1((slug + "\n" + ctx).encode("utf-8")).hexdigest()
        return self.cache_dir / f"{slug}-{key}.json"

    def _load_cache(self, slug: str, content: str) -> Optional[Dict]:
        p = self._cache_path(slug, content)
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                return None
        return None

    def _save_cache(self, slug: str, content: str, data: Optional[Dict]):
        p = self._cache_path(slug, content)
        try:
            p.write_text(json.dumps(data or {"NO_DATA": True}), encoding="utf-8")
        except Exception:
            pass

    # ---------------- LLM fallback (cheap & strict) -----------------

    def _extract_data_llm(self, content: str) -> Optional[Dict]:
        if not (self.enable_llm_fallback and self.client):
            return None

        numeric_context = "\n".join(self._lines_with_numbers(content))[:4000]  # hard cap
        if not numeric_context:
            return None

        system_msg = (
            "You are a strict data extractor for charts. "
            "Return ONLY a JSON object with keys: chart_type (bar|line|pie), "
            "title, x_label, y_label (omit for pie), labels (list[str]), data (list[number])."
        )
        user_prompt = (
            "Analyze the following lines containing numbers and extract a single chart-ready dataset.\n\n"
            f"{numeric_context}\n\n"
            "If there isn't a coherent dataset with >= 3 points, respond with the literal string: NO_DATA."
        )

        try:
            resp = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": system_msg},
                          {"role": "user", "content": user_prompt}],
                response_format={"type": "json_object"},
                max_tokens=250,
                temperature=0.1,
            )
            text = (resp.choices[0].message.content or "").strip()
            if text == "NO_DATA":
                return None
            data = json.loads(text)
            # minimal validation
            if data.get("chart_type") not in ("bar", "line", "pie"):
                return None
            if not isinstance(data.get("labels"), list) or not isinstance(data.get("data"), list):
                return None
            if len(data["labels"]) < 3 or len(data["labels"]) != len(data["data"]):
                return None
            if data["chart_type"] in ("bar", "line"):
                if not data.get("x_label") or not data.get("y_label"):
                    return None
            return self._limit_categories(data)
        except Exception as e:
            self.logger.warning(f"LLM fallback failed: {e}")
            return None

    # ---------------- Insertion helpers -----------------

    @staticmethod
    def _normalize_pad(snippet: str) -> str:
        s = snippet.strip("\n")
        return f"\n\n{s}\n\n"

    def _find_para_end(self, text: str, start_at: int) -> int:
        m = re.search(r"\n\s*\n", text[start_at:])
        return (start_at + m.end()) if m else len(text)

    def _index_outside_blocked_section(self, text: str, idx: int) -> int:
        if not self.BLOCKLIST_SECTIONS:
            return idx
        sections = list(self._H2H3_RE.finditer(text))
        for i, m in enumerate(sections):
            start = m.start()
            end = sections[i+1].start() if i+1 < len(sections) else len(text)
            title = re.sub(r"\s+", " ", m.group(2)).strip().lower()
            if start <= idx < end and title in self.BLOCKLIST_SECTIONS:
                return end
        return idx

    def _safe_insert_after_span_or_para(self, content: str, preferred_span: Optional[Tuple[int, int]]) -> int:
        """
        If a source span is known, insert after the paragraph containing its end.
        Otherwise, use fallback: after first substantial paragraph (>50 visible words),
        then after first heading, else end-of-document.
        All while avoiding protected spans and blocked sections.
        """
        spans = self._protected_spans(content)
        fm = self._yaml_front_matter_span(content)
        fm_end = fm[1] if fm else 0

        def outside(pos: int) -> bool:
            return not self._pos_in_spans(pos, spans)

        # Preferred: after the data source span's paragraph
        if preferred_span:
            end = max(preferred_span[1], fm_end)
            # skip if inside protected; hop to end of that span
            while end < len(content) and not outside(end - 1):
                for a, b in spans:
                    if a <= end - 1 < b:
                        end = b
                        break
            insert_at = self._find_para_end(content, end)
            while not outside(insert_at - 1) and insert_at < len(content):
                for a, b in spans:
                    if a <= insert_at - 1 < b:
                        insert_at = b
                        break
                insert_at = self._find_para_end(content, insert_at)
            insert_at = self._index_outside_blocked_section(content, insert_at)
            return min(insert_at, len(content))

        # Fallback 1: after first substantial paragraph (>50 visible words)
        parts = re.split(r"(\n\s*\n)", content)
        idx = 0
        for i in range(0, len(parts), 2):
            block = parts[i]
            sep = parts[i+1] if i+1 < len(parts) else ""
            block_end = idx + len(block)
            if block_end > fm_end and outside(idx) and outside(block_end - 1):
                block_spans = self._protected_spans(block)
                wc = sum(1 for m in self._WORD_RE.finditer(block) if not self._pos_in_spans(m.start(), block_spans))
                if wc >= 50:
                    insert_at = block_end + len(sep)
                    insert_at = self._index_outside_blocked_section(content, insert_at)
                    return min(insert_at, len(content))
            idx = block_end + len(sep)

        # Fallback 2: after first heading
        for m in re.finditer(r"(?m)^\s*#{1,6}\s+.+$", content):
            if m.start() >= fm_end and outside(m.start()):
                insert_at = self._find_para_end(content, m.end())
                insert_at = self._index_outside_blocked_section(content, insert_at)
                return min(insert_at, len(content))

        # Fallback 3: end
        return len(content.rstrip())

    # ---------------- Public API -----------------

    def generate_and_embed_chart(self, article_content: str, article_slug: str) -> Tuple[str, bool]:
        """
        Try cache -> heuristics -> (optional) LLM fallback.
        On success, embed a base64 PNG chart image into the article markdown.
        Idempotent: existing CHARTGEN block will be replaced.
        """
        # If a chart block already exists, strip it (we will regenerate)
        if self.MARKER_START in article_content and self.MARKER_END in article_content:
            article_content = re.sub(
                re.escape(self.MARKER_START) + r"[\s\S]*?" + re.escape(self.MARKER_END),
                "",
                article_content,
            ).rstrip()

        # Placeholder replacement phase (if any)
        placeholder_idx = None
        for ph in self.PLACEHOLDERS:
            idx = article_content.find(ph)
            if idx != -1:
                placeholder_idx = (idx, idx + len(ph))
                break

        # Cache check
        cached = self._load_cache(article_slug, article_content)
        if isinstance(cached, dict) and not cached.get("NO_DATA"):
            data = cached
        else:
            # Heuristics first (no tokens)
            data = self._try_heuristics(article_content)
            # Optional LLM fallback
            if not data:
                data = self._extract_data_llm(article_content)
            # Save result (including NO_DATA)
            self._save_cache(article_slug, article_content, data)

        if not data:
            self.logger.info("No chartable data found.")
            return article_content, False

        self.logger.info(f"Generating {data.get('chart_type')} chart: {data.get('title', 'Chart')}")
        labels = self._shorten_labels(data.get("labels", []))
        series = data.get("data", [])
        if not labels or not series or len(labels) != len(series) or len(labels) < 3:
            self.logger.info("Insufficient data after sanitization; skipping chart.")
            return article_content, False
        data["labels"] = labels
        data["data"] = series

        try:
            # Optional style (safe if available; ignored otherwise)
            try:
                plt.style.use("seaborn-v0_8-whitegrid")
            except Exception:
                pass
            fig, ax = plt.subplots(figsize=(10, 6), dpi=110)

            ctype = data["chart_type"]

            if ctype == "bar":
                ax.bar(labels, series)
                ax.set_xlabel(data.get("x_label", ""))
                ax.set_ylabel(data.get("y_label", ""))
                plt.xticks(rotation=45, ha="right")
            elif ctype == "line":
                ax.plot(labels, series, marker="o", linestyle="-")
                ax.set_xlabel(data.get("x_label", ""))
                ax.set_ylabel(data.get("y_label", ""))
                plt.xticks(rotation=45, ha="right")
            else:  # pie
                ax.pie(series, labels=labels, autopct="%1.1f%%", startangle=90)
                ax.axis("equal")

            ax.set_title(data.get("title", "Chart"), fontsize=14, fontweight="bold")
            plt.tight_layout()

            buf = io.BytesIO()
            plt.savefig(buf, format="png", bbox_inches="tight")
            plt.close(fig)
            buf.seek(0)
            image_b64 = base64.b64encode(buf.read()).decode("utf-8")

            # Concise alt text (first few points)
            preview = ", ".join(f"{l}: {v:g}" for l, v in list(zip(labels, series))[:4])
            alt = data.get("title", "Generated Chart")
            if preview:
                alt = f"{alt} — {preview}"

            chart_md = (
                f"\n\n{self.MARKER_START}\n"
                f'![{alt}](data:image/png;base64,{image_b64} "{data.get("title", "Generated Chart")}")\n'
                f"{self.MARKER_END}\n\n"
            )

            # If a placeholder exists, replace it directly
            if placeholder_idx is not None:
                s, e = placeholder_idx
                updated = article_content[:s] + self._normalize_pad(chart_md) + article_content[e:]
                return updated, True

            # Otherwise insert smartly
            insert_at = self._safe_insert_after_span_or_para(article_content, data.get("source_span"))
            updated = article_content[:insert_at] + chart_md + article_content[insert_at:]
            return updated, True

        except Exception as e:
            self.logger.error(f"Error generating/embedding chart for slug {article_slug}: {e}", exc_info=True)
            return article_content, False


# --- Manual test harness ----------------------------------------------------
if __name__ == "__main__":
    load_dotenv(find_dotenv(usecwd=True))
    gen = ChartGenerator(openai_client=None)  # uses env key if present, else disables LLM fallback

    sample_content_with_data = """
## Understanding Python Performance Trends

| Quarter | Avg Time (s) |
|---------|--------------:|
| Q1      | 15.2          |
| Q2      | 12.8          |
| Q3      | 10.5          |
| Q4      |  9.1          |

## Different Framework Adoption
Flask is used by 35% of developers, Django by 45%, FastAPI by 15%, and others by 5%.

```python
# ensure we never inject inside code fences
data = {"Q1": 1, "Q2": 2}
"""
sample_content_no_data = """

This article discusses async programming in Python and general concepts without concrete numeric series.
"""
print("--- Testing with data ---")
content_with_chart, added = gen.generate_and_embed_chart(sample_content_with_data, "python-performance-report")
print("Chart added:", added)

print("\n--- Testing without data ---")
content_no_chart, added2 = gen.generate_and_embed_chart(sample_content_no_data, "async-python")
print("Chart added:", added2)
