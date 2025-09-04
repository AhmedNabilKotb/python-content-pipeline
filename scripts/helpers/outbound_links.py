# helpers/outbound_links.py
#
# Select high-quality outbound links from a v2 JSON config and (optionally)
# generate SEO-friendly anchor text. Compatible with your existing callers:
# - return_format='dict'  -> {title: url}
# - return_format='list'  -> [(title, url)]
# - return_format='markdown' -> "- [Title](url)" (or HTML <a> with rel)
# - return_format='linker_map'
#     * when linker_rich_objects=True (default):
#           {"fallback": [{"text","url","rel","nofollow"} , ...]}
#           or grouped by type if linker_group_by_type=True
#     * when linker_rich_objects=False:
#           {"fallback": ["https://...", ...]}  (legacy behavior)
#
# This module enforces:
# - HTTPS and allowed TLDs (config.quality_metrics)
# - min trust_score, maintained flag, and recency threshold (last_checked)
# - banned/blacklist domains; optional allowlist
# - de-duplication / domain caps; optional “avoid domains already in article”
# - optional recent-history rotation via JSONL file
#
# Anchor text generation:
# - Prefers each link’s `preferred_anchors` (if present)
# - Otherwise uses defaults.anchor_templates with {keyphrase}, {topic}, {brand}, {library}
# - If niche guidelines require including keyphrase and it’s missing, we append it softly.
#
# NOTE: Safe best-effort; all failures are non-fatal and degrade gracefully.
#

from __future__ import annotations

import json
import logging
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Iterable, Optional, Any, Union
from urllib.parse import urlparse, urlsplit, urlunsplit, parse_qsl, urlencode
from datetime import datetime, date

# -----------------------------------------------------------------------------
# Logging (keeps existing root handlers if configured elsewhere)
# -----------------------------------------------------------------------------
logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.setLevel(logging.INFO)
    try:
        Path("logs").mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler("logs/outbound_links.log", encoding="utf-8")
        sh = logging.StreamHandler()
        fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        fh.setFormatter(fmt)
        sh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(sh)
    except Exception:
        logging.basicConfig(level=logging.INFO)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
_norm_ws = re.compile(r"\s+")
_JSON_CACHE: Dict[str, Tuple[float, Any]] = {}  # path -> (mtime, parsed json)
_MD_LINK_RE = re.compile(r"\]\((https?://[^)\s]+)\)")
_RAW_URL_RE = re.compile(r"(https?://[^\s)]+)")

def _norm(s: str) -> str:
    s = (s or "").strip().lower()
    s = "".join(ch if (ch.isalnum() or ch.isspace()) else " " for ch in s)
    return _norm_ws.sub(" ", s).strip()

def _norm_domain(d: str) -> str:
    d = (d or "").lower().strip()
    if d.startswith("www."):
        d = d[4:]
    return d

def _domain(url: str) -> str:
    try:
        return _norm_domain(urlparse(url).netloc)
    except Exception:
        return ""

def _is_http_url(url: str) -> bool:
    try:
        scheme = urlparse(url).scheme.lower()
        return scheme in {"http", "https"}
    except Exception:
        return False

def _strip_tracking_params(url: str) -> str:
    try:
        parts = urlsplit(url)
        if not parts.query:
            return url
        allowed = []
        for k, v in parse_qsl(parts.query, keep_blank_values=True):
            lk = k.lower()
            if lk.startswith("utm_"):
                continue
            if lk in {"fbclid", "gclid", "yclid"}:
                continue
            if lk == "ref" or lk.startswith("ref_"):
                continue
            allowed.append((k, v))
        newq = urlencode(allowed, doseq=True)
        return urlunsplit((parts.scheme, parts.netloc, parts.path, newq, parts.fragment))
    except Exception:
        return url

def _load_json(path: Path) -> Optional[dict]:
    try:
        mtime = path.stat().st_mtime
    except FileNotFoundError:
        logger.warning("Outbound links file not found: %s", path)
        return None
    except Exception as e:
        logger.error("Failed to stat outbound links json %s: %s", path, e)
        return None

    cache = _JSON_CACHE.get(str(path))
    if cache and cache[0] == mtime:
        return cache[1]

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        _JSON_CACHE[str(path)] = (mtime, data)
        return data
    except Exception as e:
        logger.error("Failed to read outbound links json %s: %s", path, e)
        return None

def _match_niche_key(niches: Dict[str, dict], niche: str) -> Optional[str]:
    if not niche:
        return None
    n = _norm(niche)

    for k in niches.keys():
        if _norm(k) == n:
            return k

    for k, cfg in niches.items():
        aliases = (cfg or {}).get("aliases") or []
        for a in aliases:
            if _norm(a) == n:
                return k

    for k in niches.keys():
        nk = _norm(k)
        if n in nk or nk in n:
            return k

    n_tokens = set(n.split())
    best = None
    best_score = 0
    for k in niches.keys():
        tokens = set(_norm(k).split())
        score = len(n_tokens & tokens)
        if score > best_score:
            best = k
            best_score = score

    if best_score == 0:
        for k, cfg in niches.items():
            aliases = (cfg or {}).get("aliases") or []
            for a in aliases:
                tokens = set(_norm(a).split())
                score = len(n_tokens & tokens)
                if score > best_score:
                    best = k
                    best_score = score

    return best if best_score > 0 else None

def _parse_date_yyyy_mm_dd(s: str) -> Optional[date]:
    if not s:
        return None
    try:
        # Accept "YYYY-MM-DD" or "YYYY-MM-DDTHH:MM:SSZ"
        core = s.split("T", 1)[0]
        return datetime.strptime(core, "%Y-%m-%d").date()
    except Exception:
        return None

def _tld_allowed(domain: str, allowed_tlds: Iterable[str]) -> bool:
    if not allowed_tlds:
        return True
    d = domain or ""
    for tld in allowed_tlds:
        tld = (tld or "").strip().lower()
        if not tld:
            continue
        if not tld.startswith("."):
            tld = "." + tld
        if d.endswith(tld):
            return True
    return False

@dataclass
class _LinkItem:
    priority: int
    sort_bonus: float  # larger is better; negative values allowed
    title: str
    url: str
    type: str
    domain: str
    trust_score: float = 0.0
    last_checked: str = ""
    maintained: Optional[bool] = None
    nofollow: Optional[bool] = None
    # NEW: Used for anchor generation
    preferred_anchors: Optional[List[str]] = None
    brand: Optional[str] = None
    library: Optional[str] = None

# -----------------------------------------------------------------------------
# Anchor text generation
# -----------------------------------------------------------------------------
_ANCHOR_SAFE_WS = re.compile(r"\s+")

def _normalize_for_match(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower())

def _choose_anchor_for_link(
    link: _LinkItem,
    defaults: Dict[str, Any],
    niche_cfg: Dict[str, Any],
    *,
    topic: Optional[str],
    keyphrase: Optional[str],
    used_texts: set,
) -> str:
    """
    Select a single anchor text for a link:
    1) Use link.preferred_anchors if available and unused
    2) Else render defaults.anchor_templates with {keyphrase, topic, brand, library}
    3) Else fall back to brand/title patterns
    If niche guidelines request including keyphrase, append it softly if missing.
    """
    # 1) Preferred
    if link.preferred_anchors:
        for a in link.preferred_anchors:
            cand = (a or "").strip()
            if cand and cand not in used_texts:
                used_texts.add(cand)
                return cand

    # 2) Templates
    templates = list(defaults.get("anchor_templates") or [])
    brand = (link.brand or link.title or "").strip()
    library = (link.library or brand).strip()

    def render(t: str) -> str:
        return (t or "").format(
            keyphrase=(keyphrase or "").strip(),
            topic=(topic or brand or "").strip(),
            brand=brand,
            library=library,
        ).strip()

    include_keyphrase = bool((niche_cfg.get("anchor_guidelines") or {}).get("include_keyphrase")) and bool(keyphrase)
    kp_norm = _normalize_for_match(keyphrase) if keyphrase else ""

    for t in templates:
        cand = render(t)
        if not cand:
            continue
        if include_keyphrase and kp_norm and kp_norm not in _normalize_for_match(cand):
            cand = f"{cand} — {keyphrase}"
        if cand not in used_texts:
            used_texts.add(cand)
            return cand

    # 3) Fallbacks
    fallbacks = [
        f"{brand} documentation",
        f"{brand} guide",
        f"{brand} for {topic}" if topic else f"{brand} reference",
        link.title,
    ]
    for fb in fallbacks:
        c = (fb or "").strip()
        if not c:
            continue
        if include_keyphrase and kp_norm and kp_norm not in _normalize_for_match(c):
            c = f"{c} — {keyphrase}"
        if c not in used_texts:
            used_texts.add(c)
            return c

    # Last resort
    res = (link.title or brand or "External resource").strip() or "External resource"
    used_texts.add(res)
    return res

# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
def select_outbound_links(
    niche: str,
    json_path: str = "config/niche_outbound_links.json",
    k: int = 2,
    *,
    include_fallback: bool = True,
    fallback_keys: Iterable[str] = ("general", "general_python", "python"),
    exclude_domains: Iterable[str] | None = None,
    types_whitelist: Iterable[str] | None = None,
    https_only: bool = False,
    dedupe_by_domain: bool = True,
    max_per_domain: Optional[int] = None,
    seed: Optional[int] = None,
    site_domain: Optional[str] = None,
    strip_utm: bool = True,
    enforce_type_diversity: bool = False,
    recent_history_path: Optional[str] = None,
    avoid_recent_days: int = 14,
    min_score: float = 0.0,
    exclude_url_patterns: Iterable[str] | None = None,
    allowlist_domains: Iterable[str] | None = None,
    existing_text: Optional[str] = None,
    avoid_domains_in_text: bool = False,
    markdown_rel: Optional[str] = None,
    return_format: str = "dict",  # 'dict' | 'list' | 'markdown' | 'linker_map'
    # v2 quality overrides:
    min_trust_score_override: Optional[float] = None,
    linker_group_by_type: bool = False,
    # NEW: anchor/rel/nofollow controls for linker_map
    topic: Optional[str] = None,
    keyphrase: Optional[str] = None,
    linker_rich_objects: bool = True,
) -> Union[
    Dict[str, str],
    List[Tuple[str, str]],
    str,
    Dict[str, List[Union[str, Dict[str, Any]]]],
]:
    """
    Select up to `k` high-quality outbound links for a given niche.

    - Understands niche_outbound_links.json v1/v2 (trust_score, last_checked, maintained, banned/allowed domains, allowed TLDs).
    - Respects niche aliases and fallbacks.
    - Filters by type, protocol, blacklist/banned/allowlist, TLD, trust score, recent-history, and optional URL patterns.
    - Optional per-domain cap (without breaking old dedupe behavior).
    - Optional exclusion of domains already linked in the existing article text.
    - Strips tracking params by default.
    - Stable, lightly randomized ordering inside priority buckets.
    - When `markdown_rel` is provided and return_format='markdown', outputs HTML anchors with rel attributes.
    - `return_format='linker_map'` builds a mapping suitable for ArticleLinker.inject_outbound_links:
        - if `linker_group_by_type` is False →
              {"fallback": [ {"text","url","rel","nofollow"} ] }  (or plain URLs if linker_rich_objects=False)
        - if True →
              {"official_docs": [ ... ], "tutorial": [ ... ], ...}
    """
    path = Path(json_path)
    data = _load_json(path)
    if not isinstance(data, dict):
        return _empty_for_format(return_format)

    version = int(data.get("version", 1) or 1)
    defaults = data.get("defaults") or {}
    niches = data.get("niches") or {}
    link_types = data.get("link_types") or {}
    qm = data.get("quality_metrics") or {}

    required = set(qm.get("required_attributes", ["url", "type"]))
    boost_domains: Dict[str, float] = { _norm_domain(k): float(v) for k, v in (data.get("boost_domains") or {}).items() }

    # Back-compat: combine old `blacklist_domains` and new `banned_domains`
    blacklist_domains_cfg = { _norm_domain(d) for d in (data.get("blacklist_domains") or []) }
    banned_domains_cfg = { _norm_domain(d) for d in (qm.get("banned_domains") or []) }
    blacklist_domains_cfg |= banned_domains_cfg

    # v2 quality settings
    q_min_trust = float(qm.get("min_trust_score", 0) or 0.0)
    if min_trust_score_override is not None:
        q_min_trust = float(min_trust_score_override)
    q_require_https = bool(qm.get("require_https", False))
    allowed_tlds = [str(t).lower().strip() for t in (qm.get("allowed_tlds") or [])]
    recent_threshold_date = _parse_date_yyyy_mm_dd(qm.get("recent_threshold", ""))  # None if not provided
    require_maintained = bool(qm.get("maintained", False))

    if not isinstance(niches, dict) or not niches:
        logger.info("No niches found in %s.", path)
        return _empty_for_format(return_format)

    niche_key = _match_niche_key(niches, niche)
    if not niche_key and not include_fallback:
        logger.info("No niche match for '%s' and fallback disabled.", niche)
        return _empty_for_format(return_format)

    # Build candidate pools (primary niche, then fallbacks)
    niche_keys: List[str] = []
    if niche_key:
        niche_keys.append(niche_key)
    if include_fallback:
        # Respect legacy fallback keys if present
        for fb in fallback_keys:
            if fb in niches and fb not in niche_keys:
                niche_keys.append(fb)

    if not niche_keys:
        logger.info("No matching niches (including fallbacks) for '%s'.", niche)
        return _empty_for_format(return_format)

    # If k not explicitly set, use per-niche max_links_per_article or global default
    if not isinstance(k, int) or k <= 0:
        k = int((niches.get(niche_key) or {}).get("max_links_per_article", (defaults.get("max_links_per_article", 2))))

    # Collect and score links
    items: List[_LinkItem] = []
    for nk in niche_keys:
        links = (niches.get(nk) or {}).get("links") or {}
        if isinstance(links, dict):
            for title, meta in links.items():
                _maybe_add_item(items, title, meta, link_types, required, boost_domains, min_score)
        elif isinstance(links, list):
            for entry in links:
                if not isinstance(entry, dict):
                    continue
                title = str(entry.get("title") or "").strip()
                _maybe_add_item(items, title, entry, link_types, required, boost_domains, min_score)

    if not items:
        return _empty_for_format(return_format)

    # v2: trust / maintained / recency / TLD checks (filter AFTER collection)
    if q_min_trust > 0:
        before = len(items)
        items = [it for it in items if (it.trust_score or 0.0) >= q_min_trust]
        if len(items) != before:
            logger.info("Trust-score filter removed %d item(s).", before - len(items))

    if require_maintained:
        before = len(items)
        items = [it for it in items if it.maintained is None or bool(it.maintained)]
        if len(items) != before:
            logger.info("Maintained filter removed %d item(s).", before - len(items))

    if recent_threshold_date:
        before = len(items)
        kept: List[_LinkItem] = []
        for it in items:
            d = _parse_date_yyyy_mm_dd(it.last_checked)
            if d is None or d >= recent_threshold_date:
                kept.append(it)
        if len(kept) != before:
            logger.info("Recent-threshold filter removed %d item(s).", before - len(kept))
        items = kept

    # Filter: whitelist/blacklist/https/patterns/site/exclude_domains/TLDs
    exdomains = { _norm_domain(d) for d in (exclude_domains or []) if d }
    exdomains |= blacklist_domains_cfg
    if site_domain:
        exdomains.add(_norm_domain(site_domain))

    # v2 can imply https requirement globally
    if q_require_https:
        https_only = True

    if types_whitelist:
        allow = {t.lower() for t in types_whitelist}
        before = len(items)
        items = [it for it in items if it.type.lower() in allow]
        if not items:
            logger.info("All candidates filtered by types_whitelist (kept 0 of %d).", before)

    if https_only:
        items = [it for it in items if it.url.lower().startswith("https://")]

    if allowlist_domains:
        allowset = { _norm_domain(d) for d in allowlist_domains if d }
        items = [it for it in items if it.domain in allowset]

    if exdomains:
        items = [it for it in items if it.domain and it.domain not in exdomains]

    if allowed_tlds:
        items = [it for it in items if _tld_allowed(it.domain, allowed_tlds)]

    if exclude_url_patterns:
        pats = [re.compile(p) for p in exclude_url_patterns if p]
        if pats:
            items = [it for it in items if not any(p.search(it.url) for p in pats)]

    # Avoid domains already in the existing article (optional)
    if avoid_domains_in_text and existing_text:
        found_domains = _extract_domains_from_text(existing_text)
        if found_domains:
            before = len(items)
            items = [it for it in items if it.domain not in found_domains]
            if before != len(items):
                logger.info("Avoided %d item(s) already linked in article.", before - len(items))

    if not items:
        return _empty_for_format(return_format)

    # Recent-history avoidance
    if recent_history_path:
        cutoff_ts = time.time() - (avoid_recent_days * 86400)
        try:
            used_domains = set()
            used_urls = set()
            p = Path(recent_history_path)
            if p.exists():
                with p.open("r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            rec = json.loads(line)
                            ts = float(rec.get("ts", 0))
                            if ts >= cutoff_ts:
                                u = str(rec.get("url", "")).strip()
                                d = _norm_domain(rec.get("domain", "") or _domain(u))
                                if u:
                                    used_urls.add(u)
                                if d:
                                    used_domains.add(d)
                        except Exception:
                            continue
            before = len(items)
            items = [it for it in items if it.url not in used_urls and it.domain not in used_domains]
            if before != len(items):
                logger.info("Recent-history filter pruned %d candidate(s).", before - len(items))
        except Exception as e:
            logger.warning("Failed recent-history filtering: %s", e)

    # Stable tie-breaking/randomization inside priority groups
    rng = random.Random(seed if seed is not None else _stable_seed(niche))
    items.sort(key=lambda it: (it.priority, -it.sort_bonus, it.domain, it.title.lower()))
    grouped: List[_LinkItem] = []
    i = 0
    while i < len(items):
        j = i + 1
        while j < len(items) and items[j].priority == items[i].priority:
            j += 1
        block = items[i:j]
        rng.shuffle(block)
        grouped.extend(block)
        i = j
    items = grouped

    # Type diversity (greedy, then fill)
    if enforce_type_diversity and k > 1:
        picked: List[_LinkItem] = []
        seen_types: set = set()
        for it in items:
            if len(picked) >= k:
                break
            if it.type not in seen_types:
                picked.append(it)
                seen_types.add(it.type)
        if len(picked) < k:
            for it in items:
                if len(picked) >= k:
                    break
                if it not in picked:
                    picked.append(it)
        items = picked

    # Domain caps (backwards compatible with dedupe_by_domain)
    if dedupe_by_domain:
        effective_cap = 1
    else:
        effective_cap = (max_per_domain if isinstance(max_per_domain, int) and max_per_domain > 0 else 10**9)

    if effective_cap >= 1:
        domain_counts: Dict[str, int] = {}
        filtered: List[_LinkItem] = []
        for it in items:
            if len(filtered) >= k:
                break
            c = domain_counts.get(it.domain, 0)
            if c >= effective_cap:
                continue
            domain_counts[it.domain] = c + 1
            filtered.append(it)
        items = filtered
    else:
        items = items[: max(0, int(k))]

    # Strip tracking params if requested (after final selection)
    if strip_utm:
        for idx, it in enumerate(items):
            items[idx] = _LinkItem(
                priority=it.priority,
                sort_bonus=it.sort_bonus,
                title=it.title,
                url=_strip_tracking_params(it.url),
                type=it.type,
                domain=it.domain,
                trust_score=it.trust_score,
                last_checked=it.last_checked,
                maintained=it.maintained,
                nofollow=it.nofollow,
                preferred_anchors=it.preferred_anchors,
                brand=it.brand,
                library=it.library,
            )

    # Persist recent picks (fire-and-forget best-effort)
    if recent_history_path and items:
        try:
            p = Path(recent_history_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            with p.open("a", encoding="utf-8") as f:
                now = time.time()
                for it in items[:k]:
                    f.write(json.dumps({"ts": now, "url": it.url, "domain": it.domain}) + "\n")
        except Exception as e:
            logger.debug("Failed to write recent-history file: %s", e)

    # Build result (final trim to k)
    items = items[: max(0, int(k))]

    if return_format == "list":
        return [(it.title, it.url) for it in items]

    if return_format == "markdown":
        if markdown_rel:
            return "\n".join(f'- <a href="{it.url}" rel="{markdown_rel}">{_escape_html(it.title)}</a>' for it in items)
        return "\n".join(f"- [{it.title}]({it.url})" for it in items)

    if return_format == "linker_map":
        if linker_group_by_type:
            if linker_rich_objects:
                groups: Dict[str, List[Dict[str, Any]]] = {}
                anchor_objs = _build_anchor_objects(items, defaults, niches.get(niche_key) or {}, topic, keyphrase)
                for it, obj in zip(items, anchor_objs):
                    groups.setdefault(it.type, []).append(obj)
                return groups
            else:
                out: Dict[str, List[str]] = {}
                for it in items:
                    out.setdefault(it.type, []).append(it.url)
                return out
        else:
            if linker_rich_objects:
                anchor_objs = _build_anchor_objects(items, defaults, niches.get(niche_key) or {}, topic, keyphrase)
                return {"fallback": anchor_objs}
            else:
                return {"fallback": [it.url for it in items]}

    # default 'dict'
    result = {it.title: it.url for it in items}
    logger.info("Selected %d outbound link(s) for niche '%s'.", len(result), niche)
    return result

# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------
def _empty_for_format(return_format: str):
    if return_format == "dict":
        return {}
    if return_format == "list":
        return []
    if return_format == "markdown":
        return ""
    if return_format == "linker_map":
        return {"fallback": []}
    return {}

def _maybe_add_item(
    items: List[_LinkItem],
    title: str,
    meta: dict,
    link_types: dict,
    required: set,
    boost_domains: Dict[str, float],
    min_score: float,
) -> None:
    if not isinstance(meta, dict):
        return
    if not title or not isinstance(title, str):
        return
    if not required.issubset(meta.keys()):
        return

    url = str(meta.get("url") or "").strip()
    ltype = str(meta.get("type") or "").strip() or "other"
    if not url or not _is_http_url(url):
        return

    # Priority from type; unknown types sink to the bottom (higher number = worse)
    type_cfg = (link_types.get(ltype) or {})
    prio = int(type_cfg.get("priority", 99))

    # Optional quality signals: higher is better (defaults to 0)
    try:
        base_score = float(meta.get("score", 0.0))
    except Exception:
        base_score = 0.0

    # Extra bonus per type (optional) + domain boost
    type_bonus = float(type_cfg.get("bonus", 0.0))
    d = _domain(url)
    bonus = boost_domains.get(d, 0.0)

    total_bonus = base_score + type_bonus + bonus
    if base_score < min_score:
        return

    # v2 extras (trust/recency/maintained/nofollow/anchors/brand/library)
    try:
        trust = float(meta.get("trust_score", 0.0))
    except Exception:
        trust = 0.0
    last_checked = str(meta.get("last_checked", "") or "")
    maintained = meta.get("maintained", None)
    nofollow = meta.get("nofollow", None)
    preferred_anchors = list(meta.get("preferred_anchors") or []) or None
    brand = meta.get("brand") or meta.get("author") or title  # reasonable default
    library = meta.get("library")

    it = _LinkItem(
        priority=prio,
        sort_bonus=total_bonus,
        title=title.strip(),
        url=url,
        type=ltype,
        domain=d,
        trust_score=trust,
        last_checked=last_checked,
        maintained=maintained if isinstance(maintained, bool) else None,
        nofollow=bool(nofollow) if isinstance(nofollow, bool) else None,
        preferred_anchors=preferred_anchors,
        brand=str(brand) if brand else None,
        library=str(library) if library else None,
    )
    items.append(it)

def _stable_seed(niche: str) -> int:
    n = _norm(niche)
    acc = 0
    for ch in n:
        acc = (acc * 131 + ord(ch)) & 0xFFFFFFFF
    return acc or 1

def _extract_domains_from_text(text: str) -> set:
    """Find domains already linked in the given markdown/plain text."""
    found = set()
    for m in _MD_LINK_RE.finditer(text or ""):
        found.add(_domain(m.group(1)))
    for m in _RAW_URL_RE.finditer(text or ""):
        found.add(_domain(m.group(1)))
    return {d for d in found if d}

def _escape_html(s: str) -> str:
    return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")

def _build_anchor_objects(
    items: List[_LinkItem],
    defaults: Dict[str, Any],
    niche_cfg: Dict[str, Any],
    topic: Optional[str],
    keyphrase: Optional[str],
) -> List[Dict[str, Any]]:
    """
    Convert selected _LinkItem objects into rich link dicts:
    { "text": "...", "url": "...", "rel": "noopener noreferrer[ nofollow]", "nofollow": bool }
    """
    used_texts: set = set()
    rel_base = " ".join((defaults.get("rel") or ["noopener", "noreferrer"]))
    nofollow_threshold = int(defaults.get("nofollow_low_trust_below", 3))
    open_in_new_tab = bool(defaults.get("open_in_new_tab", True))

    anchor_objs: List[Dict[str, Any]] = []
    for it in items:
        text = _choose_anchor_for_link(
            it, defaults, niche_cfg, topic=topic, keyphrase=keyphrase, used_texts=used_texts
        )
        # Determine nofollow
        if it.nofollow is not None:
            nofollow = bool(it.nofollow)
        else:
            nofollow = (it.trust_score or 0) < nofollow_threshold

        rel = f"{rel_base}{' nofollow' if nofollow else ''}".strip()
        obj: Dict[str, Any] = {
            "text": text,
            "url": it.url,
            "rel": rel,
            "nofollow": nofollow,
        }
        # If your linker supports target, uncomment:
        # if open_in_new_tab:
        #     obj["target"] = "_blank"

        anchor_objs.append(obj)
    return anchor_objs
