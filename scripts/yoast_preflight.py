# yoast_preflight.py

from __future__ import annotations

import re
import logging
from dataclasses import dataclass
from typing import Dict, Tuple, List, Any, Optional, Callable
from urllib.parse import urlparse, urlsplit, urlunsplit, parse_qsl, urlencode

try:
    from .internal_linker import ArticleLinker
except Exception:  # pragma: no cover
    from scripts.internal_linker import ArticleLinker  # type: ignore

# Niche-aware outbound selector (graceful fallback to static list)
try:
    from scripts.helpers.outbound_links import select_outbound_links  # type: ignore
except Exception:  # pragma: no cover
    def select_outbound_links(niche: str, json_path: str = "config/niche_outbound_links.json", k: int = 2) -> Dict[str, str]:
        return {
            "Python Docs": "https://docs.python.org/3/",
            "MDN Web Docs – HTTP": "https://developer.mozilla.org/en-US/docs/Web/HTTP/Overview",
        }

logger = logging.getLogger(__name__)

# --- Heuristics and regexes -------------------------------------------------
_TRANSITIONS = {
    "therefore", "however", "in addition", "consequently", "as a result",
    "meanwhile", "crucially", "similarly", "in contrast", "specifically",
    "moving on", "ultimately", "to illustrate", "in essence", "furthermore",
    "moreover", "on the other hand", "additionally", "for example"
}
_BE_FORMS = r"(?:am|is|are|was|were|be|been|being)"
# naive passive: "was created", "is handled", etc.
_PASSIVE_PAT = re.compile(rf"\b{_BE_FORMS}\s+\w+ed\b", re.IGNORECASE)
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_WORD_SPLIT = re.compile(r"\w+(?:'\w+)?")
_H1_RE = re.compile(r"(?m)^\s*#\s+([^\n]+)\s*$")
_EXTRA_H1_RE = re.compile(r"(?m)^(?!\A)\s*#\s+([^\n]+)\s*$")
_H2_RE = re.compile(r"(?m)^##\s+([^\n]+)")
_URL_MD_RE = re.compile(r"\[(?P<txt>[^\]]+)\]\((?P<href>[^)]+)\)")
_URL_BARE_RE = re.compile(r"\bhttps?://[^\s)]+")
_WS_MULTI = re.compile(r"\s+")
_CODE_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
_IMG_MD_RE = re.compile(r"!\[[^\]]*\]\([^)]+\)")

# light AI-tell patterns to strip
_AI_TELL_PATTERNS = [
    r"\bas an ai\b.*?(?=[.!?])",
    r"\bas a language model\b.*?(?=[.!?])",
    r"\bin this article\b",
    r"\blet'?s dive in\b",
    r"\bthis post will\b",
    r"\bwe will (?:explore|cover|show)\b",
]

# Tracking params to remove when enabled
_TRACKING_KEYS = {"fbclid", "gclid", "yclid", "mc_eid", "mc_cid"}
_TRACKING_PREFIXES = ("utm_", "ref_")

# Accept transition words at sentence start, optionally bold/italic, with comma/colon/dash or space.
_TRANSITION_TERMS_PATTERN = "|".join(
    sorted([re.escape(t).replace(r"\ ", r"\s+") for t in _TRANSITIONS], key=len, reverse=True)
)
_TRANSITION_START_RE = re.compile(
    rf"""
    ^\s*
    (?:\*\*|__|\*|_)?                  # optional opening emphasis
    (?P<tw>(?:{_TRANSITION_TERMS_PATTERN}))
    (?:\*\*|__|\*|_)?                  # optional closing emphasis
    \s*
    (?:,|:|—|–|-)?                     # comma OR colon OR dash (optional)
    \s*
    """,
    re.IGNORECASE | re.VERBOSE,
)


@dataclass
class YoastReport:
    changed: bool
    notes: List[str]
    seo: float
    readability: float
    metrics: Dict[str, float]
    warnings: int
    pre_metrics: Optional[Dict[str, float]] = None  # added for visibility


class YoastPreflight:
    """
    Lightweight, deterministic content fixer to lift Yoast-like metrics.
    Adds iterative early-stop with targets and patience.
    Skips modifications inside fenced code blocks (```...```).
    """

    def __init__(self, app_settings: Dict[str, Any], article_linker: Optional[ArticleLinker] = None):
        self.cfg = app_settings.get("yoast_compliance", {}) or {}
        self.article_linker = article_linker

        # Defaults (can be overridden per-call via **knobs)
        self.target_seo = float(self.cfg.get("target_seo_score", 95))
        self.target_read = float(self.cfg.get("target_readability_score", 95))
        self.max_warnings_default = int(self.cfg.get("max_warnings", 1))
        self.max_iters_default = int(self.cfg.get("max_enhancement_iterations", 4))
        self.patience_default = int(self.cfg.get("early_stop_patience", 2))

        self.min_h2 = int(self.cfg.get("min_subheadings", 3))
        self.min_internal = int(self.cfg.get("min_internal_links", 2))
        self.min_outbound = int(self.cfg.get("min_outbound_links", 2))

        il = app_settings.get("internal_linking", {}) or {}
        self.max_outbound = int(il.get("max_outbound_links", 3))
        self.avoid_outbound_domains: List[str] = il.get("avoid_outbound_domains", []) or []
        self.strip_query_tracking = bool(il.get("strip_query_tracking", True))
        self.use_html_for_outbound = bool(il.get("use_html_for_outbound", False))
        self.outbound_rel_default = str(il.get("outbound_rel_default", "noopener noreferrer"))
        self.outbound_target_blank = bool(il.get("outbound_target_blank", True))

        self.key_min = float(self.cfg.get("min_keyphrase_density", 0.5))
        self.key_max = float(self.cfg.get("max_keyphrase_density", 2.5))

        self.trans_min_pct = float(self.cfg.get("min_transition_word_percent", 30))
        self.long_sent_max_pct = float(self.cfg.get("max_long_sentence_percent", 20))
        self.passive_max_pct = float(self.cfg.get("passive_voice_max_percent", 7))
        self.paragraph_words_max = int(self.cfg.get("max_paragraph_words", 200))
        self.max_title_len = int(self.cfg.get("max_title_length", 60))
        self.min_title_len = int(self.cfg.get("min_title_length", 40))
        self.max_meta_len = int(self.cfg.get("max_meta_length", 156))
        self.min_meta_len = int(self.cfg.get("min_meta_length", 120))

        # NEW — flexible readability knobs (unified scorer)
        self.readability_mode = str(self.cfg.get("readability_mode", "weighted")).lower()  # "weighted"|"binary"
        self.readability_weights = dict(self.cfg.get("readability_weights", {
            "transitions": 0.35,
            "passive":     0.30,
            "long_sentence": 0.25,
            "paragraph":   0.10,
        }))
        # normalize weights
        _w_sum = sum(float(v) for v in self.readability_weights.values()) or 1.0
        for k in list(self.readability_weights.keys()):
            self.readability_weights[k] = float(self.readability_weights[k]) / _w_sum

        self.adaptive_readability = bool(self.cfg.get("adaptive_readability", True))

        # Site domain for internal/outbound detection
        wp = app_settings.get("wordpress", {}) or {}
        base = (wp.get("base_url") or wp.get("api_base_url") or "").strip()
        try:
            self.site_domain = urlparse(base).netloc.lower().replace("www.", "")
        except Exception:
            self.site_domain = ""

    # ----------------- Code-block aware helpers ------------------------------

    def _split_code_blocks(self, text: str) -> List[Tuple[bool, str]]:
        """
        Returns a list of (is_code, chunk). Non-code chunks preserve order.
        """
        chunks: List[Tuple[bool, str]] = []
        last = 0
        for m in _CODE_FENCE_RE.finditer(text):
            if m.start() > last:
                chunks.append((False, text[last:m.start()]))
            chunks.append((True, m.group(0)))
            last = m.end()
        if last < len(text):
            chunks.append((False, text[last:]))
        return chunks or [(False, text)]

    def _join_chunks(self, chunks: List[Tuple[bool, str]]) -> str:
        return "".join(seg for _, seg in chunks)

    def _apply_noncode(self, text: str, fn: Callable[[str], str]) -> Tuple[str, bool]:
        chunks = self._split_code_blocks(text)
        changed = False
        new_chunks: List[Tuple[bool, str]] = []
        for is_code, seg in chunks:
            if is_code:
                new_chunks.append((is_code, seg))
            else:
                new_seg = fn(seg)
                changed = changed or (new_seg != seg)
                new_chunks.append((False, new_seg))
        return self._join_chunks(new_chunks), changed

    def _noncode_text(self, text: str) -> str:
        return "".join(seg for is_code, seg in self._split_code_blocks(text) if not is_code)

    # ----------------- Low-level helpers ------------------------------------

    def _split_sentences(self, text: str) -> List[str]:
        text = _WS_MULTI.sub(" ", text.strip())
        return [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]

    def _transition_pct(self, sents: List[str]) -> float:
        """
        Count sentences that *start* with a transition word. Robust to bold/italic
        and punctuation variants: ',', ':', '—', '–', '-'.
        """
        if not sents:
            return 0.0
        hits = 0
        effective = 0
        for s in sents:
            if s.lstrip().startswith("#"):
                continue
            effective += 1
            if _TRANSITION_START_RE.match(s.strip()):
                hits += 1
        effective = max(1, effective)
        return 100.0 * hits / effective

    def _passive_pct_text(self, text: str) -> float:
        sents = self._split_sentences(text)
        if not sents:
            return 0.0
        passive = 0
        total = 0
        for s in sents:
            if s.lstrip().startswith("#"):
                continue
            total += 1
            if _PASSIVE_PAT.search(s):
                passive += 1
        total = max(1, total)
        return 100.0 * passive / total

    def _long_sentence_pct_text(self, text: str) -> float:
        sents = self._split_sentences(text)
        if not sents:
            return 0.0
        long = 0
        total = 0
        for s in sents:
            if s.lstrip().startswith("#"):
                continue
            total += 1
            wc = len(_WORD_SPLIT.findall(s))
            if wc > 20:  # simple threshold; Yoast guideline
                long += 1
        total = max(1, total)
        return 100.0 * long / total

    def _paragraphs(self, text: str) -> List[str]:
        return [p.strip() for p in text.split("\n\n") if p.strip()]

    def _keyphrase_density_pct(self, text: str, keyphrase: str) -> float:
        if not keyphrase:
            return 0.0
        noncode = self._noncode_text(text).lower()
        tokens = _WORD_SPLIT.findall(noncode)
        if not tokens:
            return 0.0
        blob = " ".join(tokens)
        kp = re.escape(keyphrase.lower().strip())
        hits = len(re.findall(rf"(?<!\w){kp}(?!\w)", blob))
        return 100.0 * hits / max(1, len(tokens))

    def _same_site(self, href: str) -> bool:
        if not self.site_domain:
            return False
        try:
            host = urlparse(href).netloc.lower().replace("www.", "")
            return host == self.site_domain
        except Exception:
            return False

    def _link_counts(self, text: str) -> Tuple[int, int]:
        """
        Absolute inside same domain => internal; relative => internal; other absolute => outbound.
        Duplicate hrefs counted once.
        """
        noncode = self._noncode_text(text)
        links = list(_URL_MD_RE.finditer(noncode))
        links += [m for m in _URL_BARE_RE.finditer(noncode)]
        internal = 0
        outbound = 0
        seen: set[str] = set()

        def _href(m) -> Optional[str]:
            if hasattr(m, "group"):
                return m.group("href") if isinstance(m, re.Match) and "href" in m.re.groupindex else m.group(0)  # type: ignore[arg-type]
            return None

        for m in links:
            href = _href(m)
            if not href or href in seen:
                continue
            seen.add(href)
            href = href.strip()
            if href.startswith("#") or href.startswith("mailto:"):
                continue
            if href.startswith(("http://", "https://")):
                if self._same_site(href):
                    internal += 1
                else:
                    outbound += 1
            else:
                internal += 1
        return internal, outbound

    # ----------------- URL hygiene ------------------------------------------

    @staticmethod
    def _strip_tracking(url: str) -> str:
        try:
            parts = urlsplit(url)
            kept = []
            for k, v in parse_qsl(parts.query, keep_blank_values=True):
                kl = k.lower()
                if any(kl.startswith(p) for p in _TRACKING_PREFIXES) or kl in _TRACKING_KEYS:
                    continue
                kept.append((k, v))
            return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(kept, doseq=True), parts.fragment))
        except Exception:
            return url

    def _strip_tracking_in_md(self, md: str) -> str:
        """Strip tracking params in markdown links and bare URLs (outside code)."""
        def _fn(seg: str) -> str:
            seg = _URL_MD_RE.sub(lambda m: f"[{m.group('txt')}]({self._strip_tracking(m.group('href'))})", seg)
            seg = _URL_BARE_RE.sub(lambda m: self._strip_tracking(m.group(0)), seg)
            return seg
        new_md, _ = self._apply_noncode(md, _fn)
        return new_md

    # ----------------- H1 helpers -------------------------------------------

    def _ensure_h1(self, content: str, title: str) -> Tuple[str, str, bool]:
        """
        Ensure a single H1 exists and matches the provided title; demote extra H1s to H2.
        Returns (new_content, new_title, changed)
        """
        changed = False
        m = _H1_RE.search(content or "")
        if not m:
            content = f"# {title or 'Untitled Article'}\n\n{content.strip()}\n"
            changed = True
        else:
            current_h1 = m.group(1).strip()
            use_title = title or current_h1 or "Untitled Article"
            if current_h1 != use_title:
                content = _H1_RE.sub(f"# {use_title}", content, count=1)
                changed = True
            title = use_title

        # Demote extra H1s to H2
        if _EXTRA_H1_RE.search(content or ""):
            first = _H1_RE.search(content or "")
            if first:
                head = content[: first.end()]
                tail = content[first.end():]
                tail = _EXTRA_H1_RE.sub(lambda mm: f"## {mm.group(1)}", tail)
                content = head + tail
                changed = True
        return content, title, changed

    # ----------------- Fixers ------------------------------------------------

    def _sanitize_ai_tells(self, text: str) -> Tuple[str, bool]:
        def _fn(seg: str) -> str:
            out = seg
            for pat in _AI_TELL_PATTERNS:
                out = re.sub(pat, "", out, flags=re.IGNORECASE)
            return re.sub(r"\s{2,}", " ", out).strip()
        return self._apply_noncode(text, _fn)

    def _balance_keyphrase_density(self, text: str, keyphrase: str) -> Tuple[str, bool]:
        """
        If density is too high, replace later occurrences with 'this approach'.
        Keep first 3 intact; from the 4th, every second match is thinned.
        """
        if not keyphrase:
            return text, False

        kp_pat = re.compile(rf"(?<!\w){re.escape(keyphrase)}(?!\w)", re.IGNORECASE)

        def _fn(seg: str) -> str:
            idx = 0
            def repl(m: re.Match) -> str:
                nonlocal idx
                idx += 1
                if idx <= 3 or idx % 2 == 1:
                    return m.group(0)
                return "this approach"
            return kp_pat.sub(repl, seg)

        new_text, changed = self._apply_noncode(text, _fn)
        return new_text, changed

    def _ensure_early_keyphrase(self, text: str, keyphrase: str) -> Tuple[str, bool]:
        if not keyphrase:
            return text, False

        def _fn(seg: str) -> str:
            # respect leading H1 if present
            if seg.lstrip().startswith("# "):
                head, _, rest = seg.partition("\n")
                window = " ".join(rest.split()[:50]).lower()
                if keyphrase.lower() in window:
                    return seg
                return f"{head}\n\n**{keyphrase}** sets the direction here. {rest}".strip()
            window = " ".join(seg.split()[:50]).lower()
            if keyphrase.lower() in window:
                return seg
            lead = f"**{keyphrase}** sets the direction here. "
            return f"{lead}{seg}" if seg.strip() else seg

        return self._apply_noncode(text, _fn)

    def _ensure_min_keyphrase_density(self, text: str, keyphrase: str) -> Tuple[str, bool]:
        """
        If density is below the configured minimum, deterministically insert
        gentle mentions until the minimum is reached:
         • add to the first paragraph intro
         • augment up to 3 H2s with " — {keyphrase}"
         • sprinkle short inline clarifications at paragraph starts
        """
        if not keyphrase:
            return text, False

        noncode = self._noncode_text(text)
        tokens = _WORD_SPLIT.findall(noncode)
        if not tokens:
            return text, False

        kp = re.escape(keyphrase.lower().strip())
        current_hits = len(re.findall(rf"(?<!\w){kp}(?!\w)", noncode.lower()))
        required_hits = int((self.key_min / 100.0) * len(tokens) + 0.999)  # ceil
        required_hits = max(required_hits, 1)

        if current_hits >= required_hits:
            return text, False

        need = min(required_hits - current_hits, 6)  # keep it sane

        def _fn(seg: str) -> str:
            nonlocal need
            if need <= 0:
                return seg
            parts = seg.split("\n")
            out_lines: List[str] = []
            injected_intro = False

            # First paragraph intro boost
            for idx, line in enumerate(parts):
                if need <= 0:
                    out_lines.extend(parts[idx:])
                    break

                if not injected_intro and line.strip() and not line.lstrip().startswith("#"):
                    out_lines.append(f"**{keyphrase}** in practice: {line.strip()}")
                    injected_intro = True
                    need -= 1
                else:
                    out_lines.append(line)

            seg = "\n".join(out_lines)

            # H2 augmentations
            if need > 0:
                lines = seg.splitlines()
                for i, ln in enumerate(lines):
                    if need <= 0:
                        break
                    m = _H2_RE.match(ln)
                    if not m:
                        continue
                    if keyphrase.lower() in ln.lower():
                        continue
                    candidate = f"{ln} — {keyphrase}"
                    lines[i] = candidate if len(candidate) <= 90 else f"{ln}: {keyphrase}"
                    need -= 1
                seg = "\n".join(lines)

            # Paragraph sprinkle (short prefix sentence)
            if need > 0:
                paras = self._paragraphs(seg)
                for pi, p in enumerate(paras):
                    if need <= 0:
                        break
                    if p.lstrip().startswith("#") or p.lstrip().startswith("```"):
                        continue
                    if keyphrase.lower() in p.lower():
                        continue
                    paras[pi] = f"In the context of **{keyphrase}**, {p}"
                    need -= 1
                seg = "\n\n".join(paras)

            return seg

        new_text, changed = self._apply_noncode(text, _fn)
        return new_text, changed

    def _ensure_h2s_min(self, text: str, keyphrase: str = "", niche: str = "") -> Tuple[str, bool]:
        def _fn(seg: str) -> str:
            count = len(_H2_RE.findall(seg))
            if count >= self.min_h2:
                return seg
            needed = self.min_h2 - count
            appendix = []
            bases = [
                "Field Notes",
                "Pitfalls & Trade-offs",
                "Implementation Tips",
                "Performance Considerations",
                "Testing & Validation",
            ]
            for i in range(needed):
                base = bases[i % len(bases)]
                suffix = f": {keyphrase}" if keyphrase and len(f"{base}: {keyphrase}") <= 80 else ""
                appendix.append(
                    f"## {base}{suffix}\n\n"
                    "Practical notes that clarify edge cases, trade-offs, and deployment realities."
                )
            return (seg.rstrip() + "\n\n" + "\n\n".join(appendix))
        return self._apply_noncode(text, _fn)

    def _add_keyphrase_to_h2s(self, text: str, keyphrase: str, need_at_least: int = 3) -> Tuple[str, bool]:
        if not keyphrase:
            return text, False

        def _fn(seg: str) -> str:
            lines = seg.splitlines()
            hit = 0
            for i, ln in enumerate(lines):
                m = _H2_RE.match(ln)
                if not m:
                    continue
                if keyphrase.lower() in ln.lower():
                    hit += 1
                    continue
                if hit < need_at_least:
                    candidate = f"{ln} — {keyphrase}"
                    lines[i] = candidate if len(candidate) <= 90 else f"{ln}: {keyphrase}"
                    hit += 1
            return "\n".join(lines)
        return self._apply_noncode(text, _fn)

    def _boost_transitions(self, text: str) -> Tuple[str, bool]:
        """
        Aggressively (but deterministically) add transition-word starts to reach the
        configured percentage. Skips headings, code, lists, blockquotes, images.
        """
        def _fn(seg: str) -> str:
            paras = self._paragraphs(seg)
            changed_local = False
            out_paras: List[str] = []
            # Deterministic rotation over transition terms
            tw_cycle = sorted(_TRANSITIONS)
            pick_i = 0

            for p in paras:
                stripped = p.lstrip()
                if stripped.startswith("#") or stripped.startswith(("```", "-", "*", ">")) or _IMG_MD_RE.match(stripped):
                    out_paras.append(p)
                    continue

                sents = self._split_sentences(p)
                if not sents:
                    out_paras.append(p)
                    continue

                have = sum(1 for s in sents if _TRANSITION_START_RE.match(s.strip()))
                target = (int(self.trans_min_pct * len(sents) + 99) // 100)  # ceil
                need = max(0, target - have)

                if need <= 0:
                    out_paras.append(p)
                    continue

                for j, s in enumerate(sents):
                    if need <= 0:
                        break
                    if _TRANSITION_START_RE.match(s.strip()):
                        continue
                    tw = tw_cycle[pick_i % len(tw_cycle)]
                    pick_i += 1
                    core = s[0].lower() + s[1:] if len(s) > 1 else s
                    sents[j] = f"{tw.capitalize()}: {core}"
                    need -= 1
                    changed_local = True

                out_paras.append(" ".join(sents))

            return "\n\n".join(out_paras)

        return self._apply_noncode(text, _fn)

    def _reduce_passive(self, text: str) -> Tuple[str, bool]:
        def _fn(seg: str) -> str:
            sents = self._split_sentences(seg)
            if not sents or self._passive_pct_text(seg) <= self.passive_max_pct:
                return seg

            def _fix_sent(s: str) -> str:
                # naive: collapse "was handled" → "we handled"
                return re.sub(rf"\b{_BE_FORMS}\s+(?P<v>\w+ed)\b", r"we \g<v>", s, flags=re.IGNORECASE)

            for i, s in enumerate(sents):
                if s.lstrip().startswith("#"):
                    continue
                if _PASSIVE_PAT.search(s):
                    sents[i] = _fix_sent(s)
            return " ".join(sents)
        return self._apply_noncode(text, _fn)

    def _wrap_paragraphs(self, text: str) -> Tuple[str, bool]:
        def _fn(seg: str) -> str:
            paras = self._paragraphs(seg)
            new_paras: List[str] = []
            changed = False
            for p in paras:
                if p.lstrip().startswith("#") or p.lstrip().startswith("```"):
                    new_paras.append(p)
                    continue
                words = _WORD_SPLIT.findall(p)
                if len(words) > self.paragraph_words_max:
                    sents = self._split_sentences(p)
                    half = max(1, len(sents) // 2)
                    new_paras.append(" ".join(sents[:half]).strip())
                    new_paras.append(" ".join(sents[half:]).strip())
                    changed = True
                else:
                    new_paras.append(p)
            return ("\n\n".join(new_paras))
        return self._apply_noncode(text, _fn)

    def _ensure_links_minimum(self, text: str, niche: str) -> Tuple[str, bool]:
        """
        Adds just enough niche-aware outbound links to meet the minimum, without exceeding max_outbound.
        Respects avoid_outbound_domains and can emit HTML anchors with rel/target if configured.
        If internal links are below minimum and an ArticleLinker is wired, add a soft placeholder for it to resolve (once).
        """
        PLACEHOLDER = "_See related guides in this series._"

        def _fn(seg: str) -> str:
            internal, outbound = self._link_counts(seg)
            changed_local = False

            # Determine existing hrefs to avoid duplicates
            existing_urls = set()
            for m in _URL_MD_RE.finditer(seg):
                existing_urls.add(m.group("href"))
            for m in _URL_BARE_RE.finditer(seg):
                existing_urls.add(m.group(0))

            # OUTBOUND
            needed_out = max(0, self.min_outbound - outbound)
            room_out = max(0, self.max_outbound - outbound)
            add_count = min(needed_out, room_out)

            if add_count > 0:
                picks = select_outbound_links(niche, k=add_count) or {}
                filtered = []
                for t, u in picks.items():
                    if u in existing_urls:
                        continue
                    if any(d and d.lower() in u.lower() for d in self.avoid_outbound_domains):
                        continue
                    filtered.append((t, u))

                if filtered:
                    if self.use_html_for_outbound:
                        attrs = f' rel="{self.outbound_rel_default}"' if self.outbound_rel_default else ""
                        if self.outbound_target_blank:
                            attrs += ' target="_blank"'
                        entries = " ".join(f'<a href="{u}"{attrs}>{t}</a>' for t, u in filtered)
                        seg = seg.rstrip() + f"\n\n<p><strong>Resources:</strong> {entries}</p>\n"
                    else:
                        bullets = " · ".join([f"[{t}]({u})" for t, u in filtered])
                        seg = seg.rstrip() + f"\n\n**Resources:** {bullets}\n"
                    changed_local = True

            # INTERNAL
            if internal < self.min_internal and self.article_linker and PLACEHOLDER not in seg:
                seg = seg.rstrip() + f"\n\n{PLACEHOLDER}\n"
                changed_local = True

            return seg

        return self._apply_noncode(text, _fn)

    def _adjust_meta_bounds(self, meta: str) -> Tuple[str, bool]:
        m = meta.strip()
        changed = False
        if len(m) > self.max_meta_len:
            m = m[: self.max_meta_len].rsplit(" ", 1)[0].rstrip(".!,; ") + "."
            changed = True
        elif len(m) < self.min_meta_len and m:
            m = (m + " Learn more inside.").strip()
            changed = True
        return m, changed

    def _adjust_title_bounds(self, title: str, keyphrase: str) -> Tuple[str, bool]:
        t = title.strip()
        changed = False
        if len(t) > self.max_title_len:
            t = t[: self.max_title_len].rsplit(" ", 1)[0].rstrip(" -–—:·,")
            changed = True
        elif len(t) < self.min_title_len and keyphrase and keyphrase.lower() not in t.lower():
            aug = f"{t} — {keyphrase}"
            if len(aug) <= self.max_title_len:
                t = aug
                changed = True
        return t, changed

    # ----------------- Adaptive thresholds ----------------------------------

    def _adaptive_thresholds(self, text: str) -> tuple[float, float, float, int]:
        """
        Return (trans_min, passive_max, long_sent_max, paragraph_words_max),
        optionally adjusted for long/code-heavy docs.
        """
        trans_min = float(self.trans_min_pct)
        passive_max = float(self.passive_max_pct)
        long_max = float(self.long_sent_max_pct)
        para_max = int(self.paragraph_words_max)

        if not self.adaptive_readability:
            return trans_min, passive_max, long_max, para_max

        code_blocks = len(_CODE_FENCE_RE.findall(text or ""))
        tokens = _WORD_SPLIT.findall(self._noncode_text(text))
        word_count = len(tokens)

        # Code-heavy → allow fewer transitions, slightly longer sentences/paras
        if code_blocks >= 3:
            trans_min = max(10.0, trans_min - 10.0)
            long_max = min(40.0, long_max + 5.0)
            para_max = min(300,  para_max + 40)

        # Very long articles → ease transitions + paragraph cap a bit
        if word_count > 1800:
            trans_min = max(15.0, trans_min - 5.0)
            para_max = min(300,  para_max + 20)

        return trans_min, passive_max, long_max, para_max

    # ----------------- Scoring + warnings -----------------------------------

    def _score(self, text: str, keyphrase: str, title: str = "", meta: str = "") -> Tuple[float, float, Dict[str, float], int]:
        noncode = self._noncode_text(text)

        dens = self._keyphrase_density_pct(text, keyphrase)
        sents = self._split_sentences(noncode)
        trans = self._transition_pct(sents)
        passive = self._passive_pct_text(noncode)
        long_s = self._long_sentence_pct_text(noncode)
        internal, outbound = self._link_counts(text)
        h2c = len(_H2_RE.findall(noncode))

        # Extras for SEO score
        title_ok = bool(title) and (self.min_title_len <= len(title.strip()) <= self.max_title_len)
        title_has_kp = bool(keyphrase) and keyphrase.lower() in (title or "").lower()
        meta_ok = bool(meta) and (self.min_meta_len <= len(meta.strip()) <= self.max_meta_len)
        meta_has_kp = bool(keyphrase) and keyphrase.lower() in (meta or "").lower()
        early_hit = False
        if noncode:
            first_50 = " ".join(noncode.split()[:50]).lower()
            early_hit = bool(keyphrase and keyphrase.lower() in first_50)

        # SEO scoring (100 max), designed so meeting mins reaches ~60–70
        seo = 0.0
        if self.key_min <= dens <= self.key_max:
            seo += 35
        if h2c >= self.min_h2:
            seo += 15
        if outbound >= self.min_outbound:
            seo += 15
        if internal >= self.min_internal:
            seo += 10
        if title_ok and title_has_kp:
            seo += 7
        if meta_ok and meta_has_kp:
            seo += 8
        if early_hit:
            seo += 10
        seo = min(100.0, seo)

        # --- readability (flexible) ---
        t_min, p_max, ls_max, para_max = self._adaptive_thresholds(text)

        if self.readability_mode == "weighted":
            # smooth partial credit
            trans_score = min(1.0, trans / max(1e-9, t_min)) if t_min > 0 else 1.0
            passive_score = 1.0 if passive <= p_max else max(0.0, 1.0 - (passive - p_max) / max(1.0, 2.0 * p_max))
            long_score = 1.0 if long_s <= ls_max else max(0.0, 1.0 - (long_s - ls_max) / max(1.0, 2.0 * ls_max))
            para_ok = all(len(_WORD_SPLIT.findall(p)) <= para_max for p in noncode.split("\n\n") if p.strip())
            para_score = 1.0 if para_ok else 0.6  # give some credit if only a few paras are long

            w = self.readability_weights
            readability = round(100.0 * (
                w.get("transitions", .35)   * trans_score +
                w.get("passive", .30)       * passive_score +
                w.get("long_sentence", .25) * long_score +
                w.get("paragraph", .10)     * para_score
            ), 1)
        else:
            # legacy binary points
            readability = 0.0
            if trans >= t_min: readability += 30
            if passive <= p_max: readability += 30
            if long_s <= ls_max: readability += 25
            if all(len(_WORD_SPLIT.findall(p)) <= para_max for p in noncode.split("\n\n") if p.strip()):
                readability += 15

        # Warnings respect the adapted thresholds
        warnings = 0
        warnings += 0 if self.key_min <= dens <= self.key_max else 1
        warnings += 0 if trans   >= t_min   else 1
        warnings += 0 if passive <= p_max   else 1
        warnings += 0 if long_s  <= ls_max  else 1
        warnings += 0 if h2c     >= self.min_h2 else 1
        warnings += 0 if outbound >= self.min_outbound else 1
        warnings += 0 if internal >= self.min_internal else 1

        metrics = {
            "keyphrase_density_pct": dens,
            "transition_pct": trans,
            "passive_pct": passive,
            "long_sentence_pct": long_s,
            "h2_count": h2c,
            "internal_links": internal,
            "outbound_links": outbound,
            "seo_score": seo,
            "readability_score": readability,
            "thresholds_used": {
                "trans_min_pct": t_min,
                "passive_max_pct": p_max,
                "long_sentence_max_pct": ls_max,
                "paragraph_words_max": para_max,
            },
        }
        return seo, readability, metrics, warnings

    # ----------------- Public API (iterative) --------------------------------

    def fix(
        self,
        content: str,
        title: str,
        meta: str,
        keyphrase: str,
        niche: str,
        **knobs: Any,
    ) -> Tuple[str, str, str, Dict[str, Any]]:
        """
        Iteratively apply low-cost fixes until targets are met or patience expires.
        Accepts optional knobs to override defaults:
          target_seo_score, target_readability_score, max_warnings,
          max_enhancement_iterations, early_stop_patience, etc.
        """
        target_seo = float(knobs.get("target_seo_score", self.target_seo))
        target_read = float(knobs.get("target_readability_score", self.target_read))
        max_warn = int(knobs.get("max_warnings", self.max_warnings_default))
        max_iters = int(knobs.get("max_enhancement_iterations", self.max_iters_default))
        patience = int(knobs.get("early_stop_patience", self.patience_default))

        # NEW: allow per-call overrides for flexible readability
        if "readability_mode" in knobs:
            self.readability_mode = str(knobs["readability_mode"]).lower()
        if isinstance(knobs.get("readability_weights"), dict):
            self.readability_weights.update(knobs["readability_weights"])
            s = sum(self.readability_weights.values()) or 1.0
            for k in list(self.readability_weights.keys()):
                self.readability_weights[k] = float(self.readability_weights[k]) / s
        if "adaptive_readability" in knobs:
            self.adaptive_readability = bool(knobs["adaptive_readability"])

        notes: List[str] = []
        changed_any = False

        # H1 hygiene first (synchronizes title)
        content, title, c_h1 = self._ensure_h1(content, title)
        if c_h1:
            changed_any = True
            notes.append("Normalized H1 (single, aligned with title).")

        # Initial cheap normalizations (bounds + optional URL cleanup)
        if self.strip_query_tracking:
            new_content = self._strip_tracking_in_md(content)
            if new_content != content:
                content = new_content
                changed_any = True
                notes.append("Stripped tracking parameters from URLs.")
        meta, c_meta = self._adjust_meta_bounds(meta)
        if c_meta:
            changed_any = True
            notes.append("Adjusted meta length bounds.")
        title, c_title = self._adjust_title_bounds(title, keyphrase)
        if c_title:
            changed_any = True
            notes.append("Adjusted title length / keyphrase.")

        # Evaluate baseline
        best_content = content
        best_title = title
        best_meta = meta
        best_seo, best_read, best_metrics, best_warn = self._score(best_content, keyphrase, title, meta)
        pre_metrics = dict(best_metrics)

        no_gain = 0

        def _improves(prev: Tuple[float, float, int], cur: Tuple[float, float, int]) -> bool:
            p_seo, p_read, p_w = prev
            c_seo, c_read, c_w = cur
            if c_w < p_w:
                return True
            if c_w == p_w and (c_seo + c_read) > (p_seo + p_read) + 0.5:
                return True
            return False

        # Early stop if already good
        if best_seo >= target_seo and best_read >= target_read and best_warn <= max_warn:
            report = {
                "changed": changed_any,
                "notes": notes,
                **best_metrics,
                "seo": best_seo,
                "readability": best_read,
                "warnings": best_warn,
                "pre_metrics": pre_metrics,
            }
            return best_content, best_title, best_meta, report

        # Iterative passes
        for _ in range(max_iters):
            trial = best_content
            local_notes = []

            # Ordered, low-cost passes (all skip code blocks)
            trial, c = self._sanitize_ai_tells(trial)
            if c: local_notes.append("Removed AI-tell phrases.")
            trial, c = self._ensure_early_keyphrase(trial, keyphrase)
            if c: local_notes.append("Injected early keyphrase lead.")
            trial, c = self._ensure_h2s_min(trial, keyphrase=keyphrase, niche=niche)
            if c: local_notes.append("Added H2 sections to meet minimum.")
            trial, c = self._add_keyphrase_to_h2s(trial, keyphrase, need_at_least=3)
            if c: local_notes.append("Added keyphrase to H2 headings.")
            trial, c = self._ensure_min_keyphrase_density(trial, keyphrase)
            if c: local_notes.append("Raised keyphrase density to minimum.")
            trial, c = self._wrap_paragraphs(trial)
            if c: local_notes.append("Split long paragraphs.")
            trial, c = self._boost_transitions(trial)
            if c: local_notes.append("Increased transition words.")
            trial, c = self._reduce_passive(trial)
            if c: local_notes.append("Reduced passive voice (heuristic).")
            trial, c = self._ensure_links_minimum(trial, niche)
            if c: local_notes.append("Ensured minimum outbound/internal links.")
            # Thin keyphrase only if still high after other edits
            if self._keyphrase_density_pct(trial, keyphrase) > self.key_max:
                trial, c = self._balance_keyphrase_density(trial, keyphrase)
                if c: local_notes.append("Reduced keyphrase density to avoid stuffing.")

            seo, read, metrics, warn = self._score(trial, keyphrase, best_title, best_meta)

            if _improves((best_seo, best_read, best_warn), (seo, read, warn)):
                best_content = trial
                best_seo, best_read, best_warn = seo, read, warn
                best_metrics = metrics
                notes.extend(local_notes)
                changed_any = True
                no_gain = 0
            else:
                no_gain += 1

            if best_seo >= target_seo and best_read >= target_read and best_warn <= max_warn:
                break
            if no_gain >= patience:
                break

        report = {
            "changed": changed_any,
            "notes": notes,
            **best_metrics,
            "seo": best_seo,
            "readability": best_read,
            "warnings": best_warn,
            "pre_metrics": pre_metrics,
        }
        return best_content, best_title, best_meta, report
