# internal_linker.py

import os
import json
import random
import re
import hashlib
import tempfile
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode
from pathlib import Path
from typing import Dict, List, Optional, Iterable, Set, Union, Tuple, Any
import logging
from dataclasses import dataclass
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Logging (avoid duplicate handlers if the app already configured logging)
# ---------------------------------------------------------------------------
LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(LOG_DIR / 'article_links.log', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
logger = logging.getLogger(__name__)


@dataclass
class ArticleLinkConfig:
    output_dir: Path = Path("output")
    # Internal links (footer section)
    max_links: int = 3
    min_articles_for_linking: int = 2
    # Outbound links (inline injection)
    max_outbound_links: int = 2
    min_paragraph_words_for_outbound: int = 12
    avoid_outbound_domains: Optional[Set[str]] = None  # e.g. {"example.com"}
    # This exact header will be written; detection below is tolerant/regex-based
    link_section_header: str = "\n\n---\n\n### 🔗 Related Articles\n"
    keep_backups: bool = False  # set True if you want to keep timestamped .bak files
    # Placeholder resolution (inserted by yoast_preflight)
    inline_placeholder_regex: str = r"(?mi)^\s*_?see related guides in this series\._?\s*$|^\s*\[\[RELATED\]\]\s*$"
    inline_intro_label: str = "**See also:**"
    inline_links_separator: str = " · "
    inline_links_count: int = 2
    # Don’t link to future-dated posts unless allowed
    link_future_posts: bool = True
    # UTM/trackers stripping for canonical internal/extern URLs
    strip_query_tracking: bool = True
    # NEW: explicit idempotency markers + behavior
    marker_start: str = "<!-- RELATED:START -->"
    marker_end: str = "<!-- RELATED:END -->"
    on_existing: str = "replace"  # "replace" | "skip" | "append"

    # -------------------- Outbound rendering controls ------------------------
    # Use HTML for outbound anchors (lets us attach rel/target).
    use_html_for_outbound: bool = True
    # Default rel when rendering HTML (added unless a link already provides rel).
    outbound_rel_default: str = "noopener noreferrer"
    # Add target="_blank" for outbound links by default (overridden per-link if provided).
    outbound_target_blank: bool = True
    # When linker_map is grouped by type, pick in this priority order:
    outbound_group_order: Tuple[str, ...] = (
        "official_docs", "official", "library", "guide", "course", "tutorial",
        "platform", "book", "interactive", "community", "blog", "newsletter",
        "fallback"
    )


@dataclass
class ArticleInfo:
    title: str
    url: str
    path: Path
    niche: str
    published_at: Optional[datetime] = None  # parsed from metadata["date_published"] when present


@dataclass
class _OutboundAnchor:
    text: str
    url: str
    rel: Optional[str] = None
    nofollow: Optional[bool] = None
    target_blank: Optional[bool] = None  # True/False to force, None to use config default


class ArticleLinker:
    """
    Internal & outbound linker with:
      • Stable, deterministic selection (seeded by SHA-1 of URL)
      • Idempotent related-section updates (markers; replace/skip/append)
      • Inline placeholder resolution (e.g., 'See related guides...' or [[RELATED]])
      • Outbound injection that supports dict/list inputs and avoids duplicates
      • Safe file writes with optional timestamped backups
      • Skips future-dated posts if configured
      • Understands select_outbound_links(..., return_format="linker_map")
      • Tolerant metadata reader (JSONC comments & trailing commas)
      • Detects already-present URLs in Markdown **and** HTML (href="...")
    """

    def __init__(self, config: Optional[ArticleLinkConfig] = None):
        self.config = config if config else ArticleLinkConfig()
        # Normalize avoid list
        if self.config.avoid_outbound_domains:
            self.config.avoid_outbound_domains = {d.lower().lstrip("www.") for d in self.config.avoid_outbound_domains}
        self._validate_config()
        # Compile once
        self._placeholder_re = re.compile(self.config.inline_placeholder_regex)

    # ------------------------- Validation / Loading -------------------------

    def _validate_config(self):
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        if self.config.max_links < 1:
            raise ValueError("max_links must be at least 1")
        if self.config.max_outbound_links < 0:
            raise ValueError("max_outbound_links cannot be negative")
        if self.config.inline_links_count < 0:
            raise ValueError("inline_links_count cannot be negative")
        if self.config.on_existing not in {"replace", "skip", "append"}:
            raise ValueError("on_existing must be 'replace', 'skip', or 'append'")

    # --- tolerant JSON loader (JSONC) --------------------------------------

    @staticmethod
    def _load_jsonc(path: Path) -> Optional[dict]:
        if not path.exists():
            return None
        try:
            raw = path.read_text(encoding="utf-8")
        except Exception as e:
            logger.error(f"Failed to read {path}: {e}")
            return None
        # Remove /* ... */ block comments
        raw = re.sub(r"/\*.*?\*/", "", raw, flags=re.S)
        # Remove // line comments
        raw = re.sub(r"(^|\s)//.*?$", r"\1", raw, flags=re.M)
        # Remove trailing commas (objects & arrays)
        raw = re.sub(r",\s*([}\]])", r"\1", raw)
        raw = raw.strip()
        if not raw:
            return None
        try:
            data = json.loads(raw)
            return data if isinstance(data, dict) else None
        except Exception as e:
            logger.warning(f"JSONC parse issue in {path}: {e}")
            return None

    def get_articles_by_niche(self, base_path: Path) -> Dict[str, List[ArticleInfo]]:
        """Scan /output for article folders and organize by niche."""
        niche_map: Dict[str, List[ArticleInfo]] = {}

        if not base_path.exists():
            raise FileNotFoundError(f"Base path not found: {base_path}")

        for article_dir in sorted(base_path.iterdir(), key=lambda p: p.name):
            if not article_dir.is_dir():
                continue

            try:
                metadata = self._load_article_metadata(article_dir)
                if not metadata:
                    continue

                published_at = None
                dt_str = metadata.get("date_published") or metadata.get("date") or metadata.get("date_gmt")
                if isinstance(dt_str, str):
                    try:
                        published_at = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
                    except Exception:
                        published_at = None

                info = ArticleInfo(
                    title=str(metadata.get("title") or "").strip(),
                    url=self._canonical_url(str(metadata.get("url") or "").strip()),
                    path=article_dir / "article.md",
                    niche=str(metadata.get("niche") or "").strip().lower() or "general",
                    published_at=published_at
                )

                if info.title and info.url and info.niche and info.path.exists():
                    niche_map.setdefault(info.niche, []).append(info)

            except Exception as e:
                logger.error(f"Failed to process '{article_dir.name}': {e}", exc_info=False)

        return niche_map

    def _load_article_metadata(self, article_dir: Path) -> Optional[dict]:
        meta_path = article_dir / "metadata.json"
        data = self._load_jsonc(meta_path)
        if not data:
            return None
        if not isinstance(data, dict):
            logger.error(f"{meta_path} was not a JSON object.")
            return None
        return data

    # ------------------------- Internal Linking -----------------------------

    def inject_internal_links(self, niche_map: Dict[str, List[ArticleInfo]]) -> None:
        """
        For each niche with enough articles, resolve inline placeholders
        and/or inject/replace a 'Related Articles' footer section.
        """
        now = datetime.now(timezone.utc)
        for niche, articles in niche_map.items():
            # Only proceed if we have enough articles to cross-link
            if len(articles) < self.config.min_articles_for_linking:
                logger.info(f"Skipping niche '{niche}': not enough articles to link ({len(articles)} found).")
                continue

            # Sort by recency (desc)
            articles_sorted = sorted(
                articles,
                key=lambda a: (a.published_at is not None, a.published_at or datetime.min.replace(tzinfo=timezone.utc)),
                reverse=True
            )

            for current in articles_sorted:
                try:
                    # Candidate pool for current article
                    pool = self._filter_candidates(current, articles_sorted, now)
                    if not pool:
                        logger.info(f"No eligible related posts for: {current.path}")
                        continue

                    # 1) Try resolving inline placeholders first
                    resolved_inline = self._resolve_inline_placeholders(current, pool)

                    # 2) If no inline placeholder existed, ensure footer 'Related Articles' section
                    if not resolved_inline:
                        related_for_footer = self._select_related_links(current, pool, self.config.max_links)
                        if related_for_footer:
                            links_md = self._format_links_markdown(related_for_footer)
                            self._update_article_with_links(current.path, links_md)
                            logger.info(f"Updated footer internal links in: {current.path}")

                except Exception as e:
                    logger.error(f"Failed linking for {current.path.name}: {e}", exc_info=False)

    def _filter_candidates(self, current_article: ArticleInfo, articles: List[ArticleInfo], now_utc: datetime) -> List[ArticleInfo]:
        """Remove self, optionally future-dated posts, keep same-niche list."""
        def is_future(a: ArticleInfo) -> bool:
            return a.published_at and a.published_at.tzinfo and a.published_at > now_utc

        pool = []
        for a in articles:
            if not a.url or self._canonical_url(a.url) == self._canonical_url(current_article.url):
                continue
            if not self.config.link_future_posts and is_future(a):
                continue
            pool.append(a)
        return pool

    def _stable_seed_from(self, s: str) -> int:
        """Return a stable 64-bit seed from an arbitrary string."""
        digest = hashlib.sha1((s or "").encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "big", signed=False)

    def _select_related_links(self, current_article: ArticleInfo, candidates: List[ArticleInfo], limit: int) -> List[ArticleInfo]:
        """
        Deterministic selection: same inputs → same outputs.
        Prefer recency then stable shuffle seeded by the current article URL.
        """
        if not candidates:
            return []

        # Prefer recency first
        cands = sorted(
            candidates,
            key=lambda a: (a.published_at is not None, a.published_at or datetime.min.replace(tzinfo=timezone.utc)),
            reverse=True
        )

        rnd = random.Random(self._stable_seed_from(current_article.url or str(current_article.path)))
        rnd.shuffle(cands)

        dedup: Dict[str, ArticleInfo] = {}
        for c in cands:
            url_key = self._canonical_url(c.url)
            if c.title.strip() and url_key not in dedup:
                dedup[url_key] = c

        count = min(max(0, limit), len(dedup))
        return list(dedup.values())[:count]

    # ---------- Inline placeholder resolution --------------------------------

    def _resolve_inline_placeholders(self, current: ArticleInfo, pool: List[ArticleInfo]) -> bool:
        """
        Replace placeholder lines with a compact inline 'See also' list.
        Returns True if any replacement occurred.
        """
        path = current.path
        if not path.exists():
            logger.warning(f"Article path not found for inline resolution: {path}")
            return False

        text = path.read_text(encoding="utf-8")
        matches = list(self._placeholder_re.finditer(text))
        if not matches:
            return False

        # Build inline set (deterministic)
        inline_links = self._select_related_links(current, pool, max(1, self.config.inline_links_count))
        if not inline_links:
            return False

        inline_md = self._format_inline_links(inline_links)

        # Replace all placeholders with the same inline block
        updated = self._placeholder_re.sub(inline_md, text)
        if updated == text:
            return False

        self._write_with_backup(path, text, updated)
        logger.info(f"Resolved inline related placeholders in: {path}")
        return True

    # ------------------------- Markdown formatting ---------------------------

    def _escape_md_text(self, s: str) -> str:
        """Escape brackets/parentheses to avoid breaking Markdown link syntax."""
        return (s or "").replace("[", r"\[").replace("]", r"\]").replace("(", r"\(").replace(")", r"\)")

    def _escape_html(self, s: str) -> str:
        return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")

    def _format_inline_links(self, links: List[ArticleInfo]) -> str:
        items = self.config.inline_links_separator.join(
            f"[{self._escape_md_text(link.title)}]({self._canonical_url(link.url)})" for link in links
        )
        return f"{self.config.inline_intro_label} {items}"

    def _format_links_markdown(self, links: List[ArticleInfo]) -> str:
        items = "\n".join(f"- [{self._escape_md_text(link.title)}]({self._canonical_url(link.url)})" for link in links)
        # Wrap in explicit markers for idempotent replacements
        return (
            f"\n\n{self.config.marker_start}\n"
            f"{self.config.link_section_header}{items}\n"
            f"{self.config.marker_end}\n"
        )

    # ------------------------- File Update Helpers --------------------------

    def _extract_heading_text_from_header(self) -> Optional[str]:
        """
        Try to extract the heading line text from link_section_header. Example:
          "\n\n---\n\n### 🔗 Related Articles\n" -> "Related Articles"
        """
        for line in self.config.link_section_header.splitlines():
            if line.strip().startswith("#"):
                # Drop leading #'s and emoji/symbols
                raw = re.sub(r"^#{2,6}\s*", "", line.strip())
                # Remove emoji/symbols at the beginning
                raw = re.sub(r"^[^\w]+", "", raw).strip()
                return raw or None
        return None

    def _related_section_pattern(self) -> re.Pattern:
        """
        Build a tolerant regex that matches an existing Related section:
          - optional '---' HR line
          - a heading line containing either the configured heading text or the phrase 'Related Articles'/'Related Reading'
          - capture through to the next heading or EOF
        """
        configured = self._extract_heading_text_from_header()
        if configured:
            # Allow minor variations around the configured text
            heading_pat = re.escape(configured).replace(r"\ ", r"\s+")
        else:
            heading_pat = r"(?:Related\s+Articles|Related\s+Reading)"

        return re.compile(
            rf"""(?msx)
            ^\s*(?:---\s*\n\s*)?                 # optional horizontal rule line
            #{2,6}\s*[^\n]*?(?:{heading_pat})[^\n]*\n   # heading line
            .*?                                  # section body
            (?=^\s*#{1,6}\s|\Z)                  # stop at next heading or EOF
            """,
            re.IGNORECASE,
        )

    def _marked_section_span(self, text: str) -> Optional[Tuple[int, int]]:
        """Return (start,end) if a marker-based block exists."""
        s, e = re.escape(self.config.marker_start), re.escape(self.config.marker_end)
        m = re.search(s + r"[\s\S]*?" + e, text, flags=re.MULTILINE)
        if m:
            return m.start(), m.end()
        return None

    def _existing_related_urls(self, block_text: str) -> Set[str]:
        """Extract set of URLs from a related section block (Markdown and HTML)."""
        urls_md = re.findall(r"\((https?://[^)]+)\)", block_text)              # Markdown
        urls_html = re.findall(r'href="(https?://[^"]+)"', block_text)         # HTML anchors
        urls = set(urls_md) | set(urls_html)
        return {self._canonical_url(u) for u in urls}

    def _update_article_with_links(self, article_path: Path, new_links_section: str) -> None:
        if not article_path.exists():
            logger.error(f"Article path does not exist, cannot update links: {article_path}")
            return

        original = article_path.read_text(encoding="utf-8")

        # Prefer marker-based replacement
        span = self._marked_section_span(original)
        if span:
            start, end = span
            old_block = original[start:end]
            # No-op if same URLs already present
            if self._existing_related_urls(old_block) == self._existing_related_urls(new_links_section):
                logger.info(f"Related section already up-to-date (markers) for {article_path.name}.")
                return

            if self.config.on_existing == "skip":
                logger.info(f"Existing related section present; skipping (on_existing=skip): {article_path.name}")
                return
            if self.config.on_existing == "append":
                updated = original[:end] + "\n" + new_links_section + original[end:]
            else:  # replace
                updated = original[:start] + new_links_section + original[end:]
            self._write_with_backup(article_path, original, updated)
            return

        # Fallback: tolerant section match (no markers)
        pattern = self._related_section_pattern()
        m = pattern.search(original)

        if m:
            old_block = m.group(0)
            if self._existing_related_urls(old_block) == self._existing_related_urls(new_links_section):
                logger.info(f"Related section already up-to-date for {article_path.name}.")
                return

            if self.config.on_existing == "skip":
                logger.info(f"Existing related section present; skipping (on_existing=skip): {article_path.name}")
                return
            if self.config.on_existing == "append":
                updated = original[:m.end()] + "\n" + new_links_section + original[m.end():]
            else:
                updated = original[:m.start()] + new_links_section + original[m.end():]
            self._write_with_backup(article_path, original, updated)
            return

        # If no section exists, append at the end (with ensured spacing)
        base = original if original.endswith("\n") else (original + "\n")
        updated = base.rstrip() + "\n" + new_links_section
        if updated == original:
            logger.info(f"No change detected (append) for {article_path.name}.")
            return
        self._write_with_backup(article_path, original, updated)

    def _write_with_backup(self, path: Path, original: str, updated: str) -> None:
        """
        Safer write:
          • timestamped backup (optional keep)
          • atomic replace with fsync via temp file in the same directory
        """
        ts = datetime.now().strftime("%Y%m%d%H%M%S")
        backup_path = path.with_suffix(f".{ts}.bak")
        tmp_fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
        ok = False
        try:
            backup_path.write_text(original, encoding="utf-8")
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                f.write(updated)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_name, path)
            ok = True
            if not self.config.keep_backups:
                backup_path.unlink(missing_ok=True)
            logger.info(f"Wrote related links to: {path}")
        except Exception as e:
            logger.error(f"Error writing file {path.name}: {e}. Restoring from backup.", exc_info=False)
            try:
                if ok and backup_path.exists():
                    # File replaced successfully; restore if needed
                    path.write_text(backup_path.read_text(encoding="utf-8"), encoding="utf-8")
            except Exception:
                pass
            raise
        finally:
            try:
                if os.path.exists(tmp_name):
                    os.remove(tmp_name)
            except Exception:
                pass

    # ------------------------- Outbound Links (enhanced) --------------------

    def _domain_label(self, url: str) -> str:
        try:
            netloc = urlparse(url).netloc.lower()
            netloc = netloc[4:] if netloc.startswith("www.") else netloc
            return netloc
        except Exception:
            return url

    def _canonical_url(self, url: str) -> str:
        """Optionally strip common tracking params and trailing slashes (except root) to canonicalize URLs."""
        if not url:
            return url
        try:
            parsed = urlparse(url)
            if not parsed.scheme.startswith("http"):
                return url
            # strip tracking params
            if self.config.strip_query_tracking and parsed.query:
                qs = [(k, v) for k, v in parse_qsl(parsed.query, keep_blank_values=True)
                      if not k.lower().startswith(("utm_", "gclid", "fbclid", "mc_eid", "mc_cid"))]
            else:
                qs = parse_qsl(parsed.query, keep_blank_values=True)

            # normalize trailing slash (keep root '/')
            path = parsed.path or "/"
            if len(path) > 1 and path.endswith("/"):
                path = path.rstrip("/")

            new = parsed._replace(query=urlencode(qs), path=path)
            return urlunparse(new)
        except Exception:
            return url

    # -------- normalize multiple outbound input shapes into anchors ----------

    def _normalize_outbound_input(
        self,
        outbound_links: Union[Dict[str, Union[str, List[str], List[dict], dict]], Iterable[Union[str, dict]]]
    ) -> List[_OutboundAnchor]:
        """
        Accepts:
          • linker_map dicts (e.g., {"fallback":[...]} or {"official_docs":[...], "tutorial":[...]})
          • dict[label -> url] or dict[label -> [urls...]]
          • iterable of raw urls or dicts with {text,url,rel,nofollow,target}
        Returns a list of standardized _OutboundAnchor, de-duped & canonicalized,
        filtered by avoid_outbound_domains and basic scheme checks.
        """
        anchors: List[_OutboundAnchor] = []

        def _as_anchor(obj: Union[str, dict], fallback_label: Optional[str] = None) -> Optional[_OutboundAnchor]:
            if isinstance(obj, str):
                u = self._canonical_url(obj.strip())
                if not u.startswith("http"):
                    return None
                label = fallback_label or self._domain_label(u)
                return _OutboundAnchor(text=label, url=u)
            if isinstance(obj, dict):
                u = self._canonical_url(str(obj.get("url") or "").strip())
                if not u.startswith("http"):
                    return None
                text = str(obj.get("text") or fallback_label or self._domain_label(u)).strip()
                rel = (str(obj.get("rel")).strip() or None) if obj.get("rel") is not None else None
                nf = bool(obj.get("nofollow")) if obj.get("nofollow") is not None else None
                tgt_field = obj.get("target")
                tgt_blank = None
                if isinstance(tgt_field, str):
                    tgt_blank = (tgt_field.lower() == "_blank")
                elif isinstance(tgt_field, bool):
                    tgt_blank = bool(tgt_field)
                return _OutboundAnchor(text=text, url=u, rel=rel, nofollow=nf, target_blank=tgt_blank)
            return None

        def _add_anchor(a: _OutboundAnchor):
            # domain bans
            if self.config.avoid_outbound_domains:
                if self._domain_label(a.url) in self.config.avoid_outbound_domains:
                    return
            anchors.append(a)

        # 1) dict cases
        if isinstance(outbound_links, dict):
            # Detect linker_map (has "fallback" OR keys matching known groups)
            keys = list(outbound_links.keys())
            is_linker_map = ("fallback" in outbound_links) or any(
                k in self.config.outbound_group_order for k in keys
            )

            if is_linker_map:
                # Flatten by group priority
                ordered_groups = [k for k in self.config.outbound_group_order if k in outbound_links]
                # Add any remaining custom groups while preserving input order
                for k in keys:
                    if k not in ordered_groups:
                        ordered_groups.append(k)

                for grp in ordered_groups:
                    val = outbound_links.get(grp)
                    if isinstance(val, (list, tuple, set)):
                        for item in val:
                            a = _as_anchor(item)
                            if a:
                                _add_anchor(a)
                    elif isinstance(val, dict):
                        a = _as_anchor(val)
                        if a:
                            _add_anchor(a)
                    elif isinstance(val, str):
                        a = _as_anchor(val)
                        if a:
                            _add_anchor(a)

            else:
                # Legacy dict[label -> url or [urls...]]
                for label, val in outbound_links.items():
                    if isinstance(val, (list, tuple, set)):
                        for u in val:
                            a = _as_anchor(u, fallback_label=str(label))
                            if a:
                                _add_anchor(a)
                    else:
                        a = _as_anchor(val, fallback_label=str(label))
                        if a:
                            _add_anchor(a)

        # 2) iterable cases (list/tuple/set) of strings or dicts
        elif isinstance(outbound_links, (list, tuple, set)):
            for item in outbound_links:
                a = _as_anchor(item)
                if a:
                    _add_anchor(a)
        else:
            # Unknown shape: try string
            if isinstance(outbound_links, str):
                a = _as_anchor(outbound_links)
                if a:
                    _add_anchor(a)

        # Canonicalize & dedupe by URL; further de-dupe against doc happens at injection time
        seen_urls: Set[str] = set()
        final: List[_OutboundAnchor] = []
        for a in anchors:
            if a.url in seen_urls:
                continue
            seen_urls.add(a.url)
            final.append(a)
        return final

    def _as_markdown_anchor(self, a: _OutboundAnchor) -> str:
        return f"[{self._escape_md_text(a.text)}]({a.url})"

    def _as_html_anchor(self, a: _OutboundAnchor) -> str:
        rel_parts: List[str] = []
        if a.rel:
            rel_parts.append(a.rel.strip())
        if a.nofollow:
            if not any(part.lower() == "nofollow" for part in rel_parts):
                rel_parts.append("nofollow")
        if not rel_parts and self.config.outbound_rel_default:
            rel_parts.append(self.config.outbound_rel_default.strip())

        attrs = [f'href="{self._escape_html(a.url)}"']
        rel_value = " ".join(part for part in rel_parts if part)
        if rel_value:
            attrs.append(f'rel="{self._escape_html(rel_value)}"')

        use_blank = self.config.outbound_target_blank
        if a.target_blank is not None:
            use_blank = bool(a.target_blank)
        if use_blank:
            attrs.append('target="_blank"')

        return f"<a {' '.join(attrs)}>{self._escape_html(a.text)}</a>"

    def inject_outbound_links(
        self,
        article_text: str,
        outbound_links: Union[
            Dict[str, Union[str, List[str], List[dict], dict]],
            Iterable[Union[str, dict]]
        ]
    ) -> str:
        """
        Inject up to max_outbound_links into the first suitable paragraph (not a heading,
        not code, decent length, and doesn’t already contain links).
        - Supports:
            * linker_map dicts (e.g., {'fallback':[...]} or grouped types)
            * dict[label->url] OR dict[label->[urls...]]
            * iterable of url strings OR dicts {text,url,rel,nofollow,target}
        - Skips URLs already present anywhere in the article (Markdown & HTML).
        - Renders HTML anchors when rel/target/nofollow is present (or when
          config.use_html_for_outbound=True), otherwise Markdown.
        """
        if not outbound_links:
            return article_text

        anchors = self._normalize_outbound_input(outbound_links)
        if not anchors:
            return article_text

        # Detect already-present URLs in both Markdown and HTML
        already_md = set(re.findall(r"\((https?://[^)]+)\)", article_text))
        already_html = set(re.findall(r'href="(https?://[^"]+)"', article_text))
        already_raw = set(re.findall(r"\bhttps?://\S+", article_text))
        already = {self._canonical_url(u.strip("()\"'")) for u in (already_md | already_html | already_raw)}

        anchors = [a for a in anchors if self._canonical_url(a.url) not in already]
        if not anchors:
            logger.info("All outbound links already present; skipping injection.")
            return article_text

        anchors = anchors[: max(0, self.config.max_outbound_links)]

        # Split by paragraphs while preserving newlines for rejoin
        paragraphs = article_text.split("\n\n")

        def is_code_block(p: str) -> bool:
            s = p.lstrip()
            return s.startswith("```") or s.startswith("~~~") or "\n```" in p or "\n~~~" in p

        injected = False
        for i, p in enumerate(paragraphs):
            line = p.strip()
            if not line:
                continue
            # Skip headings, lists, quotes, code blocks, or paragraphs containing existing markdown links
            if line.startswith("#") or line.startswith(("-", "*", ">")) or is_code_block(p) or re.search(r"\[[^\]]+\]\([^)]+\)", p):
                continue
            if len(re.findall(r"\w+(?:'\w+)?", line)) < self.config.min_paragraph_words_for_outbound:
                continue

            # Compose anchors
            parts: List[str] = []
            for a in anchors:
                if self.config.use_html_for_outbound or a.rel or a.nofollow is not None or a.target_blank is not None:
                    parts.append(self._as_html_anchor(a))
                else:
                    parts.append(self._as_markdown_anchor(a))

            links_str = " ".join(parts)
            # Keep sentence separation tidy: add a period if paragraph doesn't end with punctuation
            joiner = "" if re.search(r"[.!?]\s*$", p) else "."
            paragraphs[i] = (p.rstrip() + joiner + " " + links_str).strip()
            injected = True
            break

        if injected:
            logger.info(f"Injected {len(anchors)} outbound link(s).")
        else:
            logger.warning("No suitable paragraph found for outbound link injection.")
        return "\n\n".join(paragraphs)


# ------------------------- Legacy Helpers -----------------------------------

def get_articles_by_niche(base_path):
    linker = ArticleLinker()
    niche_map = linker.get_articles_by_niche(Path(base_path))
    return {k: [a.__dict__ for a in v] for k, v in niche_map.items()}

def inject_internal_links(niche_map):
    converted_map = {}
    for niche, articles in niche_map.items():
        converted_map[niche] = [
            ArticleInfo(
                path=Path(a['path']),
                **{k: v for k, v in a.items() if k != 'path'}
            )
            for a in articles
        ]
    linker = ArticleLinker()
    linker.inject_internal_links(converted_map)

def inject_outbound_links(article_text, outbound_links):
    linker = ArticleLinker()
    return linker.inject_outbound_links(article_text, outbound_links)
