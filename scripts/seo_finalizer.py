import re
import textwrap
import random
import hashlib
from typing import Dict, Any, List, Tuple, Optional
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode

# ---------------------------------------------------------------------------
# Constants & regexes
# ---------------------------------------------------------------------------

# Bare transition terms (aligns with YoastPreflight detector)
TRANSITIONS_BARE = [
    "moreover", "however", "therefore", "in addition", "meanwhile", "consequently",
    "specifically", "for example", "on the other hand", "in practice", "as a result",
    "ultimately", "similarly", "in contrast", "crucially", "to illustrate", "in essence",
    "additionally", "moving on"
]

# Punctuated versions used when injecting
TRANSITIONS_INJECT = [t.capitalize() + ":" for t in TRANSITIONS_BARE]

DEFAULT_OUTBOUND = [
    "https://docs.python.org/3/",
    "https://pypi.org/",
]

API_GW_LINKS = [
    "https://docs.konghq.com/",
    "https://tyk.io/docs/",
    "https://docs.aws.amazon.com/apigateway/latest/developerguide/welcome.html",
]

H2_SCAFFOLD = [
    "Why {KEYPHRASE} Matters",
    "Core Concepts and Architecture",
    "Hands-On: Minimal Example",
    "Authentication, Rate Limiting, and Quotas",
    "Observability: Logs, Metrics, Traces",
    "Deployment and Production Tips",
    "Common Pitfalls and How to Avoid Them",
    "FAQs",
]

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_WORD = re.compile(r"\b[\w']+\b", flags=re.UNICODE)
_H1 = re.compile(r"(?m)^\s*#\s+(.+?)\s*$")
_EXTRA_H1 = re.compile(r"(?m)^(?!\A)\s*#\s+(.+?)\s*$")  # any H1 that's not the first
_H2 = re.compile(r"(?m)^\s*##\s+.+?$")
_CODE = re.compile(r"```.*?```", re.DOTALL)
_IMG = re.compile(r"!\[[^\]]*\]\([^)]+\)")
PASSIVE = re.compile(r"\b(?:is|are|was|were|be|been|being)\s+\w+ed\b", re.IGNORECASE)
_LINK_PAREN = re.compile(r"\((https?://[^)]+)\)")  # markdown link target
_BARE_URL = re.compile(r"(?<!\()(?P<url>https?://[^\s)]+)")  # bare URL not immediately after '('

TRACKING_KEYS = {"fbclid", "gclid", "yclid", "mc_eid", "mc_cid"}
TRACKING_PREFIXES = ("utm_", "ref_")

# Sentence-start pattern used for measuring transitions (matches YoastPreflight)
_TRANSITION_START_RE = re.compile(
    r"""
    ^\s*(?:\*\*|__|\*|_)?                      # optional emphasis
    (?:
        """ + "|".join(sorted([re.escape(t).replace(r"\ ", r"\s+") for t in TRANSITIONS_BARE], key=len, reverse=True)) + r"""
    )
    (?:\*\*|__|\*|_)?\s*(?:,|:|—|–|-)?\s*      # optional punctuation
    """,
    re.IGNORECASE | re.VERBOSE,
)

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _stable_rng(seed_text: str) -> random.Random:
    """Return a deterministic RNG seeded from arbitrary text."""
    h = hashlib.sha256(seed_text.encode("utf-8")).hexdigest()
    return random.Random(int(h[:16], 16))

def _count_words(text: str) -> int:
    return len(_WORD.findall(text or ""))

def _paragraphs(md: str) -> List[str]:
    return [p for p in (md or "").split("\n\n") if p.strip()]

def _join_paras(paras: List[str]) -> str:
    return ("\n\n".join(paras).strip() + "\n") if paras else ""

def _find_code_spans(text: str) -> List[Tuple[int, int]]:
    """Return list of (start, end) spans for fenced code blocks."""
    return [(m.start(), m.end()) for m in _CODE.finditer(text or "")]

def _in_spans(pos: int, spans: List[Tuple[int, int]]) -> bool:
    return any(a <= pos < b for a, b in spans)

def _sub_outside_code(text: str, pattern: re.Pattern, repl: str, count: int = 0, flags: int = 0) -> Tuple[str, int]:
    """
    Perform regex substitution only outside fenced code blocks.
    Returns (new_text, replacements_made).
    """
    spans = _find_code_spans(text)
    out_parts: List[str] = []
    idx = 0
    replaced_total = 0

    for (a, b) in spans:
        non = text[idx:a]
        if count and replaced_total >= count:
            out_parts.append(non)
        else:
            non, n = re.subn(pattern, repl, non, 0 if count == 0 else count - replaced_total, flags)
            replaced_total += n
            out_parts.append(non)
        out_parts.append(text[a:b])  # keep code fence
        idx = b

    tail = text[idx:]
    if count and replaced_total >= count:
        out_parts.append(tail)
    else:
        tail, n = re.subn(pattern, repl, tail, 0 if count == 0 else count - replaced_total, flags)
        replaced_total += n
        out_parts.append(tail)

    return "".join(out_parts), replaced_total

def _normalize_site(site_domain: Optional[str]) -> Optional[str]:
    if not site_domain:
        return None
    s = site_domain.strip()
    if not s:
        return None
    if not s.startswith("http"):
        s = "https://" + s
    return s.rstrip("/")

def _same_site(url: str, site_root: Optional[str]) -> bool:
    if not site_root:
        return False
    try:
        return urlsplit(url).netloc.lower().strip("www.") == urlsplit(site_root).netloc.lower().strip("www.")
    except Exception:
        return False

def _strip_tracking(u: str) -> str:
    try:
        parts = urlsplit(u)
        kept = []
        for k, v in parse_qsl(parts.query, keep_blank_values=True):
            kl = k.lower()
            if any(kl.startswith(p) for p in TRACKING_PREFIXES) or kl in TRACKING_KEYS:
                continue
            kept.append((k, v))
        return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(kept, doseq=True), parts.fragment))
    except Exception:
        return u

def _strip_tracking_in_md(md: str) -> str:
    """Strip tracking params in markdown links and bare URLs (outside code)."""
    spans = _find_code_spans(md)
    out = []
    i = 0
    for (a, b) in spans:
        non = md[i:a]
        def _paren_repl(m):
            return "(" + _strip_tracking(m.group(1)) + ")"
        non = _LINK_PAREN.sub(_paren_repl, non)
        def _bare_repl(m):
            return _strip_tracking(m.group("url"))
        non = _BARE_URL.sub(_bare_repl, non)
        out.append(non)
        out.append(md[a:b])  # keep code as-is
        i = b
    tail = md[i:]
    tail = _LINK_PAREN.sub(lambda m: "(" + _strip_tracking(m.group(1)) + ")", tail)
    tail = _BARE_URL.sub(lambda m: _strip_tracking(m.group("url")), tail)
    out.append(tail)
    return "".join(out)

# ---------------------------------------------------------------------------
# Structure enforcement
# ---------------------------------------------------------------------------

def _demote_extra_h1s(md: str) -> str:
    """Ensure only one H1: demote subsequent '# ' lines to '## '."""
    first = _H1.search(md or "")
    if not first:
        return md
    start_1st, end_1st = first.span()
    head = md[:end_1st]
    tail = md[end_1st:]
    tail = _EXTRA_H1.sub(lambda m: "## " + m.group(1), tail)
    return head + tail

def _ensure_h1(md: str, title: str) -> Tuple[str, str]:
    """Ensure a single H1 exists and matches the provided title (with fallbacks)."""
    detected = _H1.search(md or "")
    if not title:
        if detected:
            title = detected.group(1).strip()
        else:
            title = "Untitled Article"

    if detected:
        md = _H1.sub(f"# {title}", md, count=1)
    else:
        md = f"# {title}\n\n{md.strip()}\n"
    md = _demote_extra_h1s(md)
    return md, title

def _ensure_lead_has_keyphrase(md: str, keyphrase: str) -> str:
    if not keyphrase:
        return md
    m = _H1.search(md or "")
    insert_at = m.end() if m else 0
    lead_span = md[insert_at: insert_at + 600]
    if re.search(rf"\b{re.escape(keyphrase)}\b", lead_span, re.IGNORECASE):
        return md

    lead = f"**{keyphrase}** is the focus of this guide and the examples below."
    if m:
        after = md[insert_at:]
        if after.startswith("\n\n"):
            insert_at += 2
        elif after.startswith("\n"):
            insert_at += 1
    return md[:insert_at] + f"\n\n{lead}\n\n" + md[insert_at:]

def _ensure_min_h2(md: str, keyphrase: str, min_h2: int, rng: random.Random) -> str:
    """Guarantee at least `min_h2` subheadings using a deterministic scaffold."""
    current_h2s = list(_H2.finditer(md or ""))
    if len(current_h2s) >= min_h2:
        return md

    paras = _paragraphs(md)
    if not paras:
        return md

    scaffold_titles: List[str] = []
    for title in H2_SCAFFOLD:
        t = title.replace("{KEYPHRASE}", keyphrase)
        need_prefix = (keyphrase and keyphrase.lower() not in t.lower() and (rng.random() < 0.4))
        if need_prefix:
            t = f"{keyphrase}: {t}"
        scaffold_titles.append(f"## {t}")

    needed = max(0, min_h2 - len(current_h2s))
    out: List[str] = []
    h2_count = len(current_h2s)
    for p in paras:
        if p.strip().startswith("##"):
            out.append(p)
            continue
        if h2_count < min_h2 and _count_words(p) > 80:
            out.append(scaffold_titles[h2_count % len(scaffold_titles)])
            h2_count += 1
        out.append(p)

    while h2_count < min_h2:
        out.append(scaffold_titles[h2_count % len(scaffold_titles)])
        out.append("This section provides additional context and actionable guidance.")
        h2_count += 1

    return _join_paras(out)

def _limit_paragraph_length(md: str, max_words: int) -> str:
    out = []
    for p in _paragraphs(md):
        words = _WORD.findall(p)
        if len(words) <= max_words or p.strip().startswith("```"):
            out.append(p)
            continue

        sentences = _SENT_SPLIT.split(p.strip())
        bucket, acc = [], 0
        for s in sentences:
            w = _count_words(s)
            if acc + w > max_words and bucket:
                out.append(" ".join(bucket).strip())
                bucket, acc = [s], w
            else:
                bucket.append(s)
                acc += w
        if bucket:
            out.append(" ".join(bucket).strip())
    return _join_paras(out)

def _reduce_long_sentences(md: str, max_words: int) -> str:
    def _fix_para(p: str) -> str:
        if p.strip().startswith("```"):
            return p
        parts = _SENT_SPLIT.split(p.strip())
        fixed = []
        for s in parts:
            words = _WORD.findall(s)
            if len(words) <= max_words:
                fixed.append(s)
                continue
            # heuristic splits
            chunks = re.split(r"—|;|:|\)|\(", s)
            buff, acc = [], 0
            for c in chunks:
                w = _count_words(c)
                if acc + w > max_words and buff:
                    fixed.append(" ".join(buff).strip() + ".")
                    buff, acc = [c], w
                else:
                    buff.append(c)
                    acc += w
            if buff:
                fixed.append(" ".join(buff).strip() + ".")
        return " ".join(fixed).strip()

    return _join_paras([_fix_para(p) for p in _paragraphs(md)])

# ---------------------------------------------------------------------------
# Transition boosting (sentence-level, Yoast-compatible)
# ---------------------------------------------------------------------------

def _sentences_in_noncode(md: str) -> List[Tuple[int, int, str]]:
    """Return list of (start_index, end_index, sentence_text) for sentences outside code blocks."""
    spans = _find_code_spans(md)
    sents: List[Tuple[int, int, str]] = []
    cursor = 0
    chunks: List[Tuple[int, str]] = []
    for (a, b) in spans:
        if a > cursor:
            chunks.append((cursor, md[cursor:a]))
        cursor = b
    if cursor < len(md):
        chunks.append((cursor, md[cursor:]))

    for base, chunk in chunks:
        idx = 0
        for m in _SENT_SPLIT.finditer(chunk):
            end = m.start() + 1  # include the punctuation
            text = chunk[idx:end].strip()
            if text:
                sents.append((base + idx, base + end, text))
            idx = m.end()
        tail = chunk[idx:].strip()
        if tail:
            sents.append((base + idx, base + idx + len(tail), tail))
    return sents

def _boost_transitions_sentences(md: str, min_pct: int, rng: random.Random) -> str:
    sents = _sentences_in_noncode(md)
    if not sents:
        return md

    # Filter out sentences that shouldn't be counted (headings, images, lists)
    eligible_idx = []
    for i, (a, b, s) in enumerate(sents):
        lead = s.lstrip()
        if lead.startswith("#") or lead.startswith(("```", "-", "*", ">")) or _IMG.match(lead):
            continue
        eligible_idx.append(i)

    if not eligible_idx:
        return md

    # Current hits
    hits = sum(1 for i in eligible_idx if _TRANSITION_START_RE.match(sents[i][2]))
    target = (min_pct * len(eligible_idx) + 99) // 100  # ceil
    need = max(0, target - hits)
    if need <= 0:
        return md

    pick_i = rng.randrange(len(TRANSITIONS_INJECT))
    md_list = list(md)

    # Insert transitions by rewriting the start of sentences
    for i in eligible_idx:
        if need <= 0:
            break
        a, b, s = sents[i]
        if _TRANSITION_START_RE.match(s):
            continue
        tw = TRANSITIONS_INJECT[pick_i % len(TRANSITIONS_INJECT)]
        pick_i += 1
        # lowercase the first character for flow
        core = (s[0].lower() + s[1:]) if s and s[0].isalpha() else s
        replacement = f"{tw} {core}"
        md_list[a:b] = replacement
        delta = len(replacement) - (b - a)

        # Shift following sentence ranges
        for j in range(i + 1, len(sents)):
            s_a, s_b, s_txt = sents[j]
            sents[j] = (s_a + delta, s_b + delta, s_txt)
        need -= 1

    return "".join(md_list)

# ---------------------------------------------------------------------------
# Links (outbound / internal)
# ---------------------------------------------------------------------------

def _collect_urls(md: str) -> List[str]:
    """Collect unique URLs from markdown link targets and bare URLs (outside code)."""
    spans = _find_code_spans(md)
    urls: List[str] = []
    i = 0
    for (a, b) in spans:
        non = md[i:a]
        urls += [m.group(1) for m in _LINK_PAREN.finditer(non)]
        urls += [m.group("url") for m in _BARE_URL.finditer(non)]
        i = b
    tail = md[i:]
    urls += [m.group(1) for m in _LINK_PAREN.finditer(tail)]
    urls += [m.group("url") for m in _BARE_URL.finditer(tail)]
    uniq = []
    seen = set()
    for u in urls:
        if u not in seen:
            uniq.append(u)
            seen.add(u)
    return uniq

def _ensure_outbound_links(
    md: str,
    links: List[str],
    min_outbound: int,
    *,
    site_root: Optional[str],
    avoid_domains: List[str],
    use_html: bool,
    rel_default: str,
    target_blank: bool,
    max_outbound_links: Optional[int] = None,
) -> str:
    """Ensure at least `min_outbound` *external* links exist (exclude internal site)."""
    if min_outbound <= 0:
        return md

    existing_urls = _collect_urls(md)
    outbound_existing = [
        u for u in existing_urls
        if (not _same_site(u, site_root))
        and (not any(d for d in avoid_domains if d and d.lower() in u.lower()))
    ]
    missing = max(0, min_outbound - len(outbound_existing))
    if missing == 0:
        return md

    candidate_pool = []
    for u in links:
        if any(u == e for e in existing_urls):
            continue
        if any(d and d.lower() in u.lower() for d in avoid_domains):
            continue
        if _same_site(u, site_root):
            continue
        candidate_pool.append(u)

    if not candidate_pool:
        return md

    to_add = candidate_pool[:missing]
    if max_outbound_links is not None:
        to_add = to_add[:max_outbound_links]

    if use_html:
        attrs = f' rel="{rel_default}"' if rel_default else ""
        if target_blank:
            attrs += ' target="_blank"'
        tail = "\n\n" + "\n".join(f'<p>See: <a href="{u}"{attrs}>{u}</a></p>' for u in to_add) + "\n"
    else:
        tail = "\n\n" + "\n".join(f"- See: <{u}>" for u in to_add) + "\n"

    return md.strip() + tail

def _ensure_internal_links(
    md: str,
    site_root: Optional[str],
    keyphrase: str,
    min_internal: int,
) -> str:
    """Ensure at least `min_internal` links to the same site; append simple tag link if needed."""
    if not site_root or min_internal <= 0:
        return md
    urls = _collect_urls(md)
    internal = [u for u in urls if _same_site(u, site_root)]
    missing = max(0, min_internal - len(internal))
    if missing == 0:
        return md
    kp_slug = re.sub(r"[^a-z0-9]+", "-", keyphrase.lower()).strip("-") if keyphrase else "blog"
    target = f"{site_root}/tag/{kp_slug}"
    if target in urls:
        return md
    tail = f'\n\nFurther reading: [Explore more about {keyphrase or "this topic"}]({target})\n'
    return md.strip() + tail

# ---------------------------------------------------------------------------
# Other helpers
# ---------------------------------------------------------------------------

def _ensure_image_alt(md: str, keyphrase: str) -> str:
    # Fix empty-alt images; otherwise ensure one relevant image exists
    if _IMG.search(md):
        md = re.sub(r"!\[\s*\]\(", f"![{keyphrase}](", md)
        return md
    alt = keyphrase or "illustration"
    placeholder = f'![{alt}](https://via.placeholder.com/1200x630.png?text={alt.replace(" ", "+")})'
    if placeholder in md:
        return md
    return md.strip() + f"\n\n{placeholder}\n"

def _ensure_faq(md: str, keyphrase: str) -> str:
    if re.search(r"(?mi)^##\s+FAQs?$", md):
        return md
    faq = textwrap.dedent(f"""
    ## FAQs

    **What is {keyphrase}?**
    It’s the focus of this guide, explained with examples and production tips.

    **How do I start?**
    Use the quick example above, then iterate with proper auth, quotas, and observability.

    **Is this production-ready?**
    Yes—follow the checklist and deployment tips to harden your setup.
    """).strip()
    return md.strip() + "\n\n" + faq + "\n"

def _ensure_conclusion(md: str) -> str:
    if re.search(r"(?mi)^##\s+Conclusion", md):
        return md
    concl = "## Conclusion\n\nYou now have a clear path from concepts to production. Iterate, measure, and refine."
    return md.strip() + "\n\n" + concl + "\n"

def _ensure_meta(title: str, meta: str, content: str, min_len: int, max_len: int) -> str:
    # Derive from first paragraph if meta missing/short; clamp length.
    base = (meta or "").strip()
    if not base:
        first_para = _paragraphs(content)[:1]
        base = (first_para[0] if first_para else "").strip()
    base = base.rstrip(".")
    if len(base) < min_len:
        candidate = (base + f". {title}").strip()
        base = candidate if len(candidate) <= max_len else base
    if len(base) > max_len:
        base = base[:max_len].rsplit(" ", 1)[0].rstrip(" ,;:.")
    return base

def _ensure_title(title: str, keyphrase: str, min_len: int, max_len: int) -> str:
    t = (title or "").strip()
    if keyphrase and keyphrase.lower() not in t.lower() and len(t) + len(keyphrase) + 2 <= max_len:
        t = f"{keyphrase}: {t}"
    t = re.sub(r"\b([Gg]uide)\s*-\s*\d+\b", r"\1", t)
    if len(t) > max_len:
        t = t[:max_len].rsplit(" ", 1)[0].rstrip(" -–—:,.;")
    if len(t) < min_len:
        sfx = " — a practical guide"
        if len(t) + len(sfx) <= max_len:
            t += sfx
    return t

def _inject_keyphrase_into_h2(md: str, keyphrase: str) -> str:
    """Prefix the first suitable H2 with the keyphrase if none contain it."""
    if not keyphrase:
        return md
    for m in _H2.finditer(md):
        line = m.group(0)
        if re.search(re.escape(keyphrase), line, re.IGNORECASE):
            return md
        if re.search(r"(?i)^##\s+(faqs?|conclusion)\b", line):
            continue
        new_line = re.sub(r"(?m)^\s*##\s+", f"## {keyphrase}: ", line, count=1)
        return md[:m.start()] + new_line + md[m.end():]
    return md

def _metrics(md: str, keyphrase: str) -> Dict[str, Any]:
    words = _count_words(md)
    h2 = len(list(_H2.finditer(md)))
    paras = _paragraphs(md)
    # Sentence-level transition metric (matches preflight)
    sent_texts = _SENT_SPLIT.split(re.sub(r"`[^`]+`", "", md))
    sent_texts = [s for s in sent_texts if s.strip()]
    starts = sum(1 for s in sent_texts if _TRANSITION_START_RE.match(s.strip()))
    trans_pct = round(100.0 * starts / max(1, len(sent_texts)), 2)
    passives = len(PASSIVE.findall(md))
    long_sents = sum(1 for s in sent_texts if _count_words(s) > 24)
    long_pct = round(100.0 * long_sents / max(1, len(sent_texts)), 2)
    kd_hits = len(re.findall(rf"\b{re.escape(keyphrase)}\b", md, flags=re.IGNORECASE)) if keyphrase else 0
    kd_pct = round(100.0 * kd_hits / max(1, words), 2)
    return {
        "word_count": words,
        "h2_count": h2,
        "transition_pct": trans_pct,
        "passive_hits": passives,
        "long_sentence_pct": long_pct,
        "keyphrase_density_pct": kd_pct,
    }

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def finalize(
    title: str,
    meta: str,
    content_md: str,
    keyphrase: str,
    knobs: Dict[str, Any],
):
    """
    Deterministically enforce structure so Yoast-style checks pass.
    knobs may include:
      - min_title_length, max_title_length, min_meta_length, max_meta_length
      - min_content_words, min_subheadings, min_transition_word_percent
      - max_paragraph_words, max_long_sentence_percent
      - min_keyphrase_density, max_keyphrase_density
      - min_outbound_links, min_internal_links
      - site_domain (e.g., "https://pythonprohub.com")
      - avoid_outbound_domains: List[str]
      - use_html_for_outbound (bool), outbound_rel_default (str), outbound_target_blank (bool)
      - strip_query_tracking (bool)
      - max_outbound_links (int)
    Returns (title, meta, content, report)
    """
    changed: List[str] = []

    # Knobs with sensible defaults
    t_min = int(knobs.get("min_title_length", 40))
    t_max = int(knobs.get("max_title_length", 60))
    m_min = int(knobs.get("min_meta_length", 120))
    m_max = int(knobs.get("max_meta_length", 156))
    min_words = int(knobs.get("min_content_words", 1100))
    min_h2 = int(knobs.get("min_subheadings", 3))
    min_trans = int(knobs.get("min_transition_word_percent", 30))
    max_para_words = int(knobs.get("max_paragraph_words", 160))
    max_long_sent_pct = int(knobs.get("max_long_sentence_percent", 20))
    min_kd = float(knobs.get("min_keyphrase_density", 0.5))   # percent
    max_kd = float(knobs.get("max_keyphrase_density", 2.5))   # percent
    min_outbound = int(knobs.get("min_outbound_links", 2))
    min_internal = int(knobs.get("min_internal_links", 0))
    site_root = _normalize_site(knobs.get("site_domain"))
    avoid_domains = knobs.get("avoid_outbound_domains") or []
    use_html = bool(knobs.get("use_html_for_outbound", False))
    rel_default = str(knobs.get("outbound_rel_default", "noopener noreferrer"))
    target_blank = bool(knobs.get("outbound_target_blank", True))
    strip_tracking = bool(knobs.get("strip_query_tracking", True))
    max_outbound_links = knobs.get("max_outbound_links")
    if isinstance(max_outbound_links, str) and max_outbound_links.isdigit():
        max_outbound_links = int(max_outbound_links)
    if not isinstance(max_outbound_links, (int, type(None))):
        max_outbound_links = None

    # Record baseline metrics
    pre_metrics = _metrics(content_md, keyphrase)
    pre_words = pre_metrics["word_count"]

    # Ensure H1 & normalize title
    content_md, title = _ensure_h1(content_md, title)
    new_title = _ensure_title(title, keyphrase, t_min, t_max)
    if new_title != title:
        changed.append("title")
        title = new_title
        content_md = _H1.sub(f"# {title}", content_md, count=1)
        content_md = _demote_extra_h1s(content_md)

    # Deterministic RNG for all heuristics below
    rng = _stable_rng(f"{title}|{keyphrase}|{len(content_md)}")

    # Optional link tracking cleanup
    if strip_tracking:
        before = content_md
        content_md = _strip_tracking_in_md(content_md)
        if content_md != before:
            changed.append("strip_tracking")

    # Keyphrase near lead + image/alt
    before = content_md
    content_md = _ensure_lead_has_keyphrase(content_md, keyphrase)
    if content_md != before:
        changed.append("lead")
    img_before = content_md
    content_md = _ensure_image_alt(content_md, keyphrase)
    if content_md != img_before:
        changed.append("image")

    # Subheadings and paragraph/sentence shaping
    before = content_md
    content_md = _ensure_min_h2(content_md, keyphrase, min_h2, rng)
    if content_md != before:
        changed.append("h2_scaffold")

    content_md = _limit_paragraph_length(content_md, max_para_words)
    content_md = _reduce_long_sentences(content_md, 24)

    # FAQ + Conclusion
    before = content_md
    content_md = _ensure_faq(content_md, keyphrase)
    content_md = _ensure_conclusion(content_md)
    if content_md != before:
        changed.append("faq_conclusion")

    # Ensure minimal length (deterministic filler) — short, readable, many transitions
    if _count_words(content_md) < min_words:
        missing = min_words - _count_words(content_md)
        seeds = [
            "Moreover: start with a minimal, working example and expand deliberately.",
            "However: keep interfaces small and validate inputs aggressively.",
            f"In addition: document assumptions and boundaries around **{keyphrase}**.",
            "Therefore: prefer boring solutions and predictable failure modes.",
            "For example: measure a baseline, then iterate in small steps.",
            "Consequently: monitor latency, error rates, and saturation from day one.",
        ]
        start = rng.randrange(len(seeds))
        filler_paras: List[str] = []
        i = 0
        # Keep each filler to one short sentence for readability
        while _count_words("\n\n".join(filler_paras)) < missing:
            filler_paras.append(seeds[(start + i) % len(seeds)])
            i += 1
        content_md = content_md.strip() + "\n\n" + "\n\n".join(filler_paras) + "\n"
        changed.append("length_expand")

    # Internal links (before outbound so counts are separate)
    content_md = _ensure_internal_links(content_md, site_root, keyphrase, min_internal)

    # Outbound links (contextual)
    suggested = API_GW_LINKS if re.search(r"\bapi\s*gateway\b", f"{keyphrase} {title}", re.IGNORECASE) else DEFAULT_OUTBOUND
    before = content_md
    content_md = _ensure_outbound_links(
        content_md,
        suggested,
        min_outbound,
        site_root=site_root,
        avoid_domains=avoid_domains if isinstance(avoid_domains, list) else [],
        use_html=use_html,
        rel_default=rel_default,
        target_blank=target_blank,
        max_outbound_links=max_outbound_links,
    )
    if content_md != before:
        changed.append("outbound_links")

    # Meta description
    new_meta = _ensure_meta(title, meta, content_md, m_min, m_max)
    if new_meta != meta:
        changed.append("meta")
        meta = new_meta

    # ---- Keyphrase density clamp (safe) ------------------------------------
    metrics = _metrics(content_md, keyphrase)
    kd = metrics["keyphrase_density_pct"]
    EPS = 0.2  # do not trim unless we're at least this much over the max

    if kd < min_kd and keyphrase:
        # First, add it to an H2 if missing
        before = content_md
        content_md = _inject_keyphrase_into_h2(content_md, keyphrase)
        if content_md != before:
            changed.append("h2_keyphrase")
        # If still low, add a gentle early mention
        metrics = _metrics(content_md, keyphrase)
        if metrics["keyphrase_density_pct"] < min_kd:
            lead_fix = f"In the context of **{keyphrase}**, "
            paras = _paragraphs(content_md)
            if paras:
                paras[0] = (lead_fix + paras[0]) if lead_fix not in paras[0] else paras[0]
                content_md = _join_paras(paras)
                changed.append("kd_raise")
    elif kd > (max_kd + EPS) and keyphrase:
        # Trim conservatively, then re-check that we didn't go below min
        target_remove = max(1, int(((kd - max_kd) * 0.01) * metrics["word_count"]))
        pattern = re.compile(rf"\b{re.escape(keyphrase)}\b", re.IGNORECASE)
        content_md, n = _sub_outside_code(content_md, pattern, "it", count=target_remove)
        if n > 0:
            changed.append("kd_trim")
        # Post-check: keep above minimum
        if _metrics(content_md, keyphrase)["keyphrase_density_pct"] < min_kd:
            content_md = _inject_keyphrase_into_h2(content_md, keyphrase)
            changed.append("kd_recover")

    # ---- Transition booster LAST (sentence-level) --------------------------
    before = content_md
    content_md = _boost_transitions_sentences(content_md, min_trans, rng)
    if content_md != before:
        changed.append("transitions")

    # If long sentence % still high, one more tightening pass
    metrics = _metrics(content_md, keyphrase)
    if metrics["long_sentence_pct"] > max_long_sent_pct:
        content_md = _reduce_long_sentences(content_md, 20)
        changed.append("sentence_tighten")

    post_metrics = _metrics(content_md, keyphrase)
    report = {
        "changed": changed,
        "pre_metrics": pre_metrics,
        "post_metrics": post_metrics,
        "length_delta": post_metrics["word_count"] - pre_words,
    }

    return title, meta, content_md, report
