import os
import re
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

from tenacity import retry, stop_after_attempt, wait_random_exponential
from dotenv import load_dotenv

# ---------------- Flexible imports (package or standalone) ----------------
def _fallback_slugify(text: str) -> str:
    import re as _re
    t = (text or "").lower()
    t = _re.sub(r"[^a-z0-9]+", "-", t).strip("-")
    t = _re.sub(r"-{2,}", "-", t)
    return t or "article"

try:
    # Package-relative
    from .seo_optimizer import SEOEnforcer, SEOEnhancementResult  # type: ignore
except Exception:
    from seo_optimizer import SEOEnforcer, SEOEnhancementResult  # type: ignore

try:
    from .utils.utils import load_config as load_app_config, slugify  # type: ignore
except Exception:
    try:
        from utils.utils import load_config as load_app_config, slugify  # type: ignore
    except Exception:
        def load_app_config(path: str) -> Dict[str, Any]:
            # Minimal loader fallback
            return json.loads(Path(path).read_text(encoding="utf-8"))
        slugify = _fallback_slugify  # type: ignore

# Title synthesizer: package import -> scripts import -> minimal fallback
def _fallback_generate_title(topic: str, keyphrase: str, niche: str, used_titles: Optional[List[str]] = None) -> str:
    base = (keyphrase or topic or "Python").strip()
    t = f"{base}: production patterns that actually scale"
    # clamp to 60 chars for Yoast window
    return t[:60].rstrip(" -–—:,")

try:
    from .title_synthesizer import generate_title  # type: ignore
except Exception:
    try:
        from scripts.title_synthesizer import generate_title  # type: ignore
    except Exception:
        generate_title = _fallback_generate_title  # type: ignore

# NEW: prepublish adapter and (optional) linker
try:
    from .prepublish import run_prepublish  # type: ignore
except Exception:
    from prepublish import run_prepublish  # type: ignore

try:
    from .internal_linker import ArticleLinker  # type: ignore
except Exception:
    ArticleLinker = None  # type: ignore

# ---------------- Env & logging ----------------
load_dotenv()

Path("logs").mkdir(parents=True, exist_ok=True)
# Avoid duplicate handlers if root logger is already configured by the app
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,  # switch to DEBUG when tuning
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("logs/article_generation.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
logger = logging.getLogger(__name__)

# ---------------- Draft normalizer ----------------
def normalize_draft(input_data: Any) -> Dict[str, Any]:
    """
    Normalize LLM output (expected: plain markdown OR dict) to a consistent dict.
    Preserves paragraph spacing; extracts H1 cleanly without collapsing newlines.
    Also normalizes CRLF -> LF and trims trailing spaces.
    """
    _log = logging.getLogger(__name__)

    title = ""
    meta_description = ""
    article_content = ""

    # Normalize newlines + strip trailing whitespace
    def _tidy(s: str) -> str:
        s = s.replace("\r\n", "\n").replace("\r", "\n")
        s = re.sub(r"[ \t]+$", "", s, flags=re.MULTILINE)
        return s

    if isinstance(input_data, str):
        s = _tidy(input_data.strip("\n"))
        # Prefer an explicit H1
        m = re.search(r"^\s*#\s+(.*)$", s, flags=re.MULTILINE)
        if m:
            title = m.group(1).strip()
            # Remove only the first H1 line (+ one following blank line if present)
            h1_pat = re.compile(r"^\s*#\s+.*\n(\s*\n)?", re.MULTILINE)
            article_content = h1_pat.sub("", s, count=1).lstrip("\n")
            _log.debug(f"normalize_draft: H1 extracted → '{title}'")
        else:
            # Heuristic: short first line ending in .!? becomes title
            lines = [ln for ln in s.splitlines()]
            first_nonempty_idx = next((i for i, ln in enumerate(lines) if ln.strip()), 0)
            first = lines[first_nonempty_idx].rstrip()
            if len(first.split()) < 20 and first[-1:] in ".!?":
                title = first.strip()
                # Drop that single line, keep spacing elsewhere
                article_content = "\n".join(lines[:first_nonempty_idx] + lines[first_nonempty_idx+1:]).lstrip("\n")
                _log.debug(f"normalize_draft: short-sentence title → '{title}'")
            else:
                title = (first[:100].strip() or "Untitled Content")
                article_content = s
                _log.warning("normalize_draft: no H1; using first line head as fallback.")
    elif isinstance(input_data, dict):
        _log.warning("normalize_draft: got dict; parsing as JSON fallback.")
        title = _tidy((input_data.get("title") or "").strip())
        meta_description = _tidy((input_data.get("meta") or input_data.get("meta_description") or input_data.get("meta description") or "").strip())
        article_content = _tidy((input_data.get("article_content") or input_data.get("article content") or input_data.get("article") or "").strip())

        if not article_content and (input_data.get("introduction") or input_data.get("sections")):
            parts = []
            if isinstance(input_data.get("introduction"), str):
                intro = _tidy(input_data["introduction"].strip())
                if intro:
                    parts.append(intro)
            for sec in input_data.get("sections", []):
                if isinstance(sec, dict):
                    st = (_tidy(sec.get("title") or "")).strip()
                    sc = (_tidy(sec.get("content") or "")).strip()
                    if sc:
                        parts.append((f"## {st}\n\n{sc}".strip()) if st else sc)
            if isinstance(input_data.get("conclusion"), str):
                concl = _tidy(input_data["conclusion"].strip())
                if concl:
                    # De-duplicate any existing conclusion heading
                    concl = re.sub(r"^\s*#+\s*conclusion\s*\n+", "", concl, flags=re.I)
                    parts.append(f"## Conclusion\n\n{concl}")
            article_content = "\n\n".join(parts).strip()

        if not title and article_content:
            s = article_content
            m = re.search(r"^\s*#\s+(.*)$", s, flags=re.MULTILINE)
            if m:
                title = m.group(1).strip()
                h1_pat = re.compile(r"^\s*#\s+.*\n(\s*\n)?", re.MULTILINE)
                article_content = h1_pat.sub("", s, count=1).lstrip("\n")
            else:
                # Keep spacing; don't collapse
                lines = s.splitlines()
                first_nonempty_idx = next((i for i, ln in enumerate(lines) if ln.strip()), 0)
                first = lines[first_nonempty_idx].rstrip()
                if len(first.split()) < 20 and first[-1:] in ".!?":
                    title = first.strip()
                    article_content = "\n".join(lines[:first_nonempty_idx] + lines[first_nonempty_idx+1:]).lstrip("\n")
                else:
                    title = (first[:100].strip() or "Untitled Content")
    else:
        _log.warning(f"normalize_draft: unsupported input type {type(input_data)}.")
        return {"title": "", "slug": "", "meta_description": "", "article_content": "", "keyphrase": "", "niche": ""}

    if not title:
        title = "Untitled Article"
        _log.warning("normalize_draft: title empty; generic fallback used.")

    if not article_content:
        article_content = "Content could not be generated or extracted."
        _log.warning("normalize_draft: content empty; generic fallback used.")

    if not meta_description:
        src = (article_content.split("\n\n")[0].strip() or title)
        meta_description = (src[:156].strip() + "...") if len(src) > 156 else src

    # Trim long meta on a word boundary
    if len(meta_description) > 156:
        tmp = meta_description[:156]
        tmp = re.sub(r"\s+\S*$", "", tmp).rstrip(" ,;:—-")
        meta_description = tmp or meta_description[:156]

    slug_val = slugify(title)[:75] if title else "untitled-article"

    return {
        "title": title,
        "slug": slug_val,
        "meta_description": meta_description,
        "article_content": article_content,
        "keyphrase": input_data.get("keyphrase", "") if isinstance(input_data, dict) else "",
        "niche": input_data.get("niche", "") if isinstance(input_data, dict) else ""
    }


# ---------------- Title helpers ----------------
def _titlecase(s: str) -> str:
    if not s:
        return s
    small = {"and","or","for","to","with","a","an","the","in","on","of"}
    words = [w.strip() for w in re.split(r"\s+", s.strip()) if w.strip()]
    out = []
    for i, w in enumerate(words):
        lw = w.lower()
        if i==0 or i==len(words)-1 or lw not in small:
            if re.search(r"[A-Z]", w):
                out.append(w)
            else:
                out.append(lw.capitalize())
        else:
            out.append(lw)
    return re.sub(r"\s+[-–—]\s+", " — ", " ".join(out)).strip(" -–—:")

def _sync_numbers(title: str, meta: str) -> Tuple[str, str]:
    # keep the year (YYYY) consistent between title and meta
    yr = re.search(r"\b(20\d{2})\b", title or "")
    if yr and yr.group(1) not in (meta or ""):
        meta = f"{meta.rstrip('. ')} ({yr.group(1)})."
    # never allow “-1/-2/...” fragments in TITLE (slug only)
    title = re.sub(r"\b([Gg]uide)[\s-]*\d+\b", r"\1", title or "")
    return title, meta


# ---------------- Small helpers ----------------
_H1_RE = re.compile(r"^\s*#\s+.*?$", flags=re.MULTILINE)

def _replace_h1(markdown_text: str, new_title: str) -> str:
    """Replace the first H1 in markdown with # new_title; insert if missing."""
    if _H1_RE.search(markdown_text or ""):
        return _H1_RE.sub(f"# {new_title}", markdown_text, count=1)
    return f"# {new_title}\n\n{markdown_text.strip()}"

def _collect_used_titles(output_dir: Path) -> List[str]:
    """Collect previously used titles from output/*/metadata.json to reduce duplicate-y headlines."""
    used: List[str] = []
    if not output_dir.exists():
        return used
    for d in output_dir.iterdir():
        if not d.is_dir():
            continue
        mp = d / "metadata.json"
        try:
            if mp.exists():
                data = json.loads(mp.read_text(encoding="utf-8"))
                t = (data.get("title") or "").strip()
                if t:
                    used.append(t)
        except Exception:
            continue
    return used[:500]  # cap for speed

def _enforce_title_bounds(title: str, keyphrase: str, min_len: int = 40, max_len: int = 60) -> str:
    """Keep title within Yoast sweet spot; trim on word boundaries, avoid trailing prepositions, and inject keyphrase if there's room."""
    t = (title or "").strip()
    if not t:
        return t

    # Ensure keyphrase presence when there's room
    if keyphrase and keyphrase.lower() not in t.lower():
        if len(t) + len(keyphrase) + 2 <= max_len:  # account for ": "
            t = f"{keyphrase}: {t}"

    # Hard cap, but backtrack to last full word (avoid mid-word truncation)
    if len(t) > max_len:
        t = t[:max_len]
        # backtrack to previous whitespace to end on a full word
        t = re.sub(r"\s+\S*$", "", t)

    # If still a bit over (e.g., no whitespace), tidy punctuation
    t = t.rstrip(" -–—:,.;")

    # If too short, add a tasteful suffix without breaking bounds
    if len(t) < min_len:
        suffixes = [
            " in practice",
            " for real-world projects",
            " with modern Python",
            " at scale",
            " — a practical guide",
        ]
        for sfx in suffixes:
            if len(t) + len(sfx) <= max_len:
                t = t + sfx
                break

    # Remove dangling prepositions/stop-words at the very end
    trailers = r"(?:in|with|for|to|of|by|at|on|and|or|a|an|the)"
    t = re.sub(rf"\b{trailers}\b[\s\-\–—:]*$", "", t, flags=re.IGNORECASE).rstrip(" -–—:,.;")

    # Normalize internal spaces
    t = re.sub(r"\s{2,}", " ", t).strip()

    return t

def _dedupe_slug(base_slug: str, output_dir: Path, max_len: int = 75) -> str:
    """Ensure slug uniqueness by checking output_dir and appending numeric suffixes if needed."""
    s = (base_slug or "article").strip("-")
    s = re.sub(r"-{2,}", "-", s)[:max_len].strip("-") or "article"
    candidate = s
    i = 2
    while (output_dir / candidate).exists():
        suffix = f"-{i}"
        trimmed = s[: max_len - len(suffix)].rstrip("-")
        candidate = f"{trimmed}{suffix}"
        i += 1
        if i > 99:
            break
    return candidate

# --- Code-fence & placement helpers (new) ---
_WORD_RE = re.compile(r"\b\w+\b", re.UNICODE)
_CODE_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
_H1_LINE_RE = re.compile(r"^\s*#\s+.*?$", re.MULTILINE)
_H2_LINE_RE = re.compile(r"(?m)^##\s+.+$")

def _code_spans(text: str) -> List[Tuple[int, int]]:
    return [(m.start(), m.end()) for m in _CODE_FENCE_RE.finditer(text or "")]

def _in_spans(pos: int, spans: List[Tuple[int, int]]) -> bool:
    return any(a <= pos < b for a, b in spans)

def _word_positions(text: str) -> List[int]:
    return [m.start() for m in _WORD_RE.finditer(text or "")]

def _find_para_end(text: str, start_at: int) -> int:
    m = re.search(r"\n\s*\n", text[start_at:])
    return (start_at + m.end()) if m else len(text)

def _normalize_pad(snippet: str) -> str:
    s = snippet.strip("\n")
    return f"\n\n{s}\n\n" if s else ""

def _safe_insert(text: str, snippet: str, at_idx: int) -> str:
    # Avoid duplicate insertion of identical snippet
    if snippet.strip() and snippet.strip() in text:
        return text
    pad = _normalize_pad(snippet)
    return text[:at_idx] + pad + text[at_idx:]

def _safe_after_n_words(text: str, n_words: int) -> Optional[int]:
    """Return a safe insertion index AFTER the paragraph that contains the Nth word."""
    if n_words <= 0:
        return None
    positions = _word_positions(text)
    if not positions:
        return None
    idx = positions[min(n_words, len(positions)) - 1]
    spans = _code_spans(text)

    # If inside code, move forward
    while idx < len(text) and _in_spans(idx, spans):
        idx += 1

    insert_at = _find_para_end(text, idx)
    # If end lands in code, hop to the end of that fence and skip to next paragraph break
    while _in_spans(insert_at - 1, spans):
        for a, b in spans:
            if a <= insert_at - 1 < b:
                insert_at = b
                break
        insert_at = _find_para_end(text, insert_at)
    return min(insert_at, len(text))

def _safe_before_last_h2(text: str) -> int:
    """Safe insertion point BEFORE the last H2, outside code fences; fallback to end."""
    spans = _code_spans(text)
    last_h2 = None
    for m in _H2_LINE_RE.finditer(text or ""):
        if not _in_spans(m.start(), spans):
            last_h2 = m
    if not last_h2:
        return len(text)
    start = last_h2.start()
    back = text.rfind("\n\n", 0, start)
    insert_at = (back + 2) if back != -1 else max(0, start)
    if _in_spans(insert_at, spans):
        for a, b in spans:
            if a <= insert_at < b:
                insert_at = max(0, a)
                break
    return insert_at

def _ensure_keyphrase_lead(markdown_text: str, keyphrase: str) -> str:
    """Ensure the keyphrase appears in the first ~50 words, inserting a short lead after the H1 (never inside code)."""
    if not keyphrase:
        return markdown_text

    words = markdown_text.split()
    if re.search(rf"\b{re.escape(keyphrase)}\b", " ".join(words[:50]), flags=re.IGNORECASE):
        return markdown_text

    # Add a short, neutral lead. Use period if the H1 block didn’t end with punctuation.
    lead = f"**{keyphrase}** drives the approach here."
    text = markdown_text

    # Find H1 line
    m = _H1_LINE_RE.search(text or "")
    if not m:
        # No H1 — just prefix a lead paragraph
        return _normalize_pad(lead) + text.lstrip("\n")

    insert_at = m.end()
    # Skip a single following blank line if present
    if insert_at < len(text):
        if text[insert_at:insert_at+2] == "\n\n":
            insert_at += 2
        elif text[insert_at:insert_at+1] == "\n":
            insert_at += 1

    spans = _code_spans(text)
    # If the spot is in a code fence, move to fence end
    if _in_spans(insert_at, spans):
        for a, b in spans:
            if a <= insert_at < b:
                insert_at = b
                break

    return _safe_insert(text, lead, insert_at)

# ---------------- Path helpers (support system.md / article_base.md) ----------------
def _first_existing_file(*candidates: str) -> Optional[Path]:
    for c in candidates:
        p = Path(c).expanduser().resolve()
        if p.is_file():
            return p
    return None

def _get_path(paths_cfg: Dict[str, Any], *keys: str, default: Optional[str] = None) -> Optional[str]:
    for k in keys:
        v = paths_cfg.get(k)
        if isinstance(v, str) and v.strip():
            return v
    return default

def _normalize_key(s: str) -> str:
    s = (s or "").lower()
    s = s.replace("&", " and ")
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s{2,}", " ", s).strip()
    return s

# ---------------- Configuration ----------------
class AppConfig:
    def __init__(self, settings_path: str = "config/settings.json"):
        self.settings = self._load_settings(Path(settings_path))
        paths_config = self.settings.get("paths", {}) or {}

        # dirs
        self.prompts_dir: Path = Path(paths_config.get("prompts_dir", "prompts")).expanduser().resolve()
        self.output_dir: Path = Path(self.settings.get("paths", {}).get("articles_output_dir", "output")).expanduser().resolve()

        # files (support both new and legacy keys; prefer explicit absolute paths)
        system_explicit = _get_path(
            paths_config,
            "system_prompt_path", "system_prompt_template_path", "system_prompt",
            default=""
        )
        article_base_explicit = _get_path(
            paths_config,
            "article_base_path", "default_prompt_template_path", "prompt_template_default",
            default=""
        )

        # fallbacks under prompts_dir
        self.system_prompt_path: Optional[Path] = _first_existing_file(
            system_explicit or "",
            str(self.prompts_dir / "system.md"),
            str(self.prompts_dir / "_system.md"),
        )
        self.article_base_path: Optional[Path] = _first_existing_file(
            article_base_explicit or "",
            str(self.prompts_dir / "article_base.md"),
        )

        if not self.system_prompt_path:
            logger.warning("system.md not found (checked settings + prompts dir). System prompt will be omitted.")
        else:
            logger.info(f"[Prompts] system: {self.system_prompt_path}")

        if not self.article_base_path:
            logger.error("article_base.md not found (checked settings + prompts dir). Generation cannot continue without it.")
        else:
            logger.info(f"[Prompts] base:   {self.article_base_path}")

    def _load_settings(self, path: Path) -> Dict[str, Any]:
        if not path.exists():
            logger.error(f"Settings file not found at {path}")
        return load_app_config(str(path))

    @property
    def openai_api_key(self) -> Optional[str]:
        return self.settings.get("api_keys", {}).get("openai") or os.getenv("OPENAI_API_KEY")

    @property
    def article_generation_model(self) -> str:
        # Allow an env override for quick switches
        return os.getenv("ARTICLE_MODEL") or self.settings.get("article_generation", {}).get("model", "gpt-4o")

    @property
    def article_temperature(self) -> float:
        return float(self.settings.get("article_generation", {}).get("temperature", 0.7))

    @property
    def max_article_tokens(self) -> int:
        return int(self.settings.get("article_generation", {}).get("max_tokens", 1300))

    @property
    def article_word_count_target(self) -> Tuple[int, int]:
        wc = self.settings.get("article_generation", {}).get("word_count_target", [1100, 1400])
        if isinstance(wc, (list, tuple)) and len(wc) == 2:
            return int(wc[0]), int(wc[1])
        return 1100, 1400

    @property
    def max_ads_per_article(self) -> int:
        return int(self.settings.get("adsense", {}).get("max_ads_per_article", 3))

    @property
    def adsense_placement_markers(self) -> List[str]:
        safe_defaults = ["", "", ""]
        v = self.settings.get("adsense", {}).get("placement_markers", safe_defaults)
        return v if isinstance(v, list) else safe_defaults

    @property
    def enable_ctr_title(self) -> bool:
        return bool(self.settings.get("titles", {}).get("enable_synthesizer", True))

    @property
    def structured_output(self) -> bool:
        # Toggle the strict JSON Schema step on/off if needed
        return bool(self.settings.get("article_generation", {}).get("structured_output", True))

# ---------------- Article Generator ----------------
_DEFAULT_GEN_SYSTEM = "You are an expert Python and software engineering writer. Produce accurate, actionable, original content with clean Markdown. Never include content outside of the requested JSON when asked for structured output."

class ArticleGenerator:
    def __init__(
        self,
        config: AppConfig,
        openai_client: Optional["OpenAI"] = None,  # type: ignore
        keyphrase_tracker_instance: Optional[Any] = None,
    ):
        self.config = config

        # Reuse the enforcer's OpenAI client & models for consistency/caching
        if openai_client is None and not self.config.openai_api_key:
            logger.critical("OpenAI API key not found for ArticleGenerator.")
            raise ValueError("OpenAI API key not found for ArticleGenerator.")
        self.seo_enforcer = SEOEnforcer(
            self.config.settings,
            openai_client=openai_client,  # may be None; enforcer can build its own
            keyphrase_tracker_instance=keyphrase_tracker_instance,
        )

        # ArticleLinker is optional; pass through if available
        self.article_linker = None
        try:
            if ArticleLinker:
                self.article_linker = ArticleLinker(self.config.settings)  # type: ignore
        except Exception:
            self.article_linker = None

        # Preload prompts
        self.system_prompt_text: str = ""
        if self.config.system_prompt_path and self.config.system_prompt_path.is_file():
            self.system_prompt_text = self.config.system_prompt_path.read_text(encoding="utf-8").strip()

        if not self.config.article_base_path:
            raise FileNotFoundError("article_base.md is required but was not found.")
        self.base_prompt_text: str = self.config.article_base_path.read_text(encoding="utf-8").strip()

    def _load_prompt_template(self, niche: str) -> str:
        """
        Choose the prompt for this niche:
        1) If settings.prompt_templates has an entry whose 'niche'/'topic'/'name'
           matches the given niche (normalized), use its inline 'template' or load from 'path'.
        2) Otherwise use the shared base prompt (article_base.md).
        """
        want_norm = _normalize_key(niche)
        for entry in self.config.settings.get("prompt_templates", []):
            key = (entry.get("niche") or entry.get("topic") or entry.get("name") or "")
            key_norm = _normalize_key(key)
            if key_norm and key_norm == want_norm:
                if entry.get("template"):
                    logger.info(f"Using topic-specific prompt template for '{niche}' from settings.json (inline).")
                    return entry["template"].strip()
                if entry.get("path"):
                    p = Path(entry["path"]).expanduser()
                    if p.is_file():
                        logger.info(f"Using topic-specific prompt template for '{niche}' from file: {p}.")
                        return p.read_text(encoding="utf-8").strip()
                    logger.warning(f"Template path not found for niche '{niche}': {p}")
                break  # matched but nothing usable; fall back

        # Fallback to article_base.md content
        if not self.base_prompt_text:
            logger.error("Base prompt (article_base.md) is empty or missing. Cannot generate article.")
            return ""
        return self.base_prompt_text

    def _cap_tokens(self, requested: int) -> int:
        return max(400, min(int(requested or 900), 1400))

    # ---------- JSON extraction helpers (for non-strict model behavior) ----------
    @staticmethod
    def _extract_json_block(text: str) -> str:
        """
        Extract a likely JSON object from a messy text: prefer ```json blocks,
        else take outermost {...}. Returns the raw JSON string or raises.
        """
        if not text:
            raise ValueError("Empty model output")

        # Try fenced ```json ... ```
        fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, flags=re.IGNORECASE)
        if fence:
            return fence.group(1).strip()

        # Fallback to outermost braces
        first = text.find("{")
        last = text.rfind("}")
        if first != -1 and last != -1 and last > first:
            return text[first:last+1].strip()

        raise ValueError("No JSON object found in output")

    # ---------- LLM call for JSON draft (uses enforcer's client, but custom system) ----------
    @retry(stop=stop_after_attempt(3), wait=wait_random_exponential(min=1, max=8), reraise=True)
    def _llm_json_call(self, final_prompt: str, json_schema: Dict[str, Any], max_tokens: int, temperature: float) -> str:
        """
        Try JSON Schema first; fallback to json_object; final fallback: plain text + extract JSON.
        Uses the same OpenAI client as SEOEnforcer to share HTTP pool and creds.
        """
        client = self.seo_enforcer.client
        system_msg = self.system_prompt_text or _DEFAULT_GEN_SYSTEM
        messages = [{"role": "system", "content": system_msg}, {"role": "user", "content": final_prompt}]

        # 1) json_schema (preferred) — can be disabled in settings
        if self.config.structured_output:
            try:
                resp = client.chat.completions.create(
                    model=self.config.article_generation_model,
                    messages=messages,
                    response_format={"type": "json_schema", "json_schema": json_schema},
                    max_tokens=self._cap_tokens(max_tokens),
                    temperature=temperature,
                )
                return (resp.choices[0].message.content or "").strip()
            except Exception as e:
                logger.warning(f"json_schema format failed; retrying with json_object. ({e})")

        # 2) json_object
        try:
            resp = client.chat.completions.create(
                model=self.config.article_generation_model,
                messages=messages,
                response_format={"type": "json_object"},
                max_tokens=self._cap_tokens(max_tokens),
                temperature=temperature,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            logger.warning(f"json_object format failed; retrying with free-form + extractor. ({e})")

        # 3) free-form + extraction
        resp = client.chat.completions.create(
            model=self.config.article_generation_model,
            messages=messages,
            max_tokens=self._cap_tokens(max_tokens),
            temperature=temperature,
        )
        raw = (resp.choices[0].message.content or "").strip()
        return self._extract_json_block(raw)

    @retry(stop=stop_after_attempt(3), wait=wait_random_exponential(min=1, max=8), reraise=True)
    def _generate_initial_article_draft(self, topic: str, keyphrase: str, niche: str) -> Optional[Dict[str, Any]]:
        prompt_template = self._load_prompt_template(niche)
        if not prompt_template:
            logger.error(f"No prompt template loaded for topic: {topic}")
            return None

        word_count_min, word_count_max = self.config.article_word_count_target
        author_name = self.config.settings.get("authors", {}).get("name", "Content Writer")
        site_name = self.config.settings.get("general", {}).get("site_name", "Our Blog")

        try:
            user_prompt = (
                prompt_template.replace("{ARTICLE_TOPIC}", topic)
                .replace("{KEYPHRASE}", keyphrase)
                .replace("{ARTICLE_NICHE}", niche)
                .replace("{WORD_COUNT_MIN}", str(word_count_min))
                .replace("{WORD_COUNT_MAX}", str(word_count_max))
                .replace("{AUTHOR_NAME}", author_name)
                .replace("{SITE_NAME}", site_name)
            ).strip()
        except Exception as e:
            logger.critical(f"Prompt render failed: {e}\n---\n{prompt_template}\n---", exc_info=True)
            return None

        if not user_prompt:
            logger.critical("Rendered prompt is empty.")
            return None

        # Compose final instruction. If system.md exists, we’ll use it as the system message in _llm_json_call.
        header = (
            "You will produce a complete Python article as JSON.\n"
            "Output ONLY valid JSON matching the schema. No markdown outside JSON.\n"
            "The `article_content` must start with '# {title}' on the first line, then the body.\n\n"
        )
        final_prompt = header + user_prompt

        # JSON schema (strict)
        json_schema = {
            "name": "ArticleDraft",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "keyphrase": {"type": "string"},
                    "meta_description": {"type": "string"},
                    "article_content": {"type": "string"},
                },
                "required": ["title", "keyphrase", "meta_description", "article_content"],
                "additionalProperties": False,
            },
        }

        try:
            mt = self._cap_tokens(self.config.max_article_tokens)
            raw = self._llm_json_call(
                final_prompt=final_prompt,
                json_schema=json_schema,
                max_tokens=mt,
                temperature=self.config.article_temperature,
            )
        except Exception as e:
            logger.error(f"OpenAI call failed for topic '{topic}': {e}", exc_info=True)
            # Return a minimal normalized draft so the pipeline can continue safely
            return normalize_draft(f"# {topic}\n\nContent generation failed for '{keyphrase}'. Please check logs.")

        # Parse and normalize, with robust fallback on malformed JSON
        try:
            data = json.loads(raw)
            draft = normalize_draft(data)
        except Exception as e:
            logger.error(f"JSON parse failed: {e}\nRaw (first 500 chars):\n{str(raw)[:500]}")
            # Try extraction once more in case we got a blob
            try:
                extracted = self._extract_json_block(raw)
                data = json.loads(extracted)
                draft = normalize_draft(data)
            except Exception:
                draft = normalize_draft(str(raw))

        # Ensure required fields; add disciplined title/slug/meta
        if not draft.get("title"):
            draft["title"] = topic or "Untitled Article"

        try:
            draft["title"] = _titlecase(draft.get("title", ""))
        except Exception:
            draft["title"] = draft.get("title", "").strip()

        # compute slug from cleaned title
        base_slug = slugify(draft["title"])[:75] or draft.get("slug") or slugify(topic)
        draft["slug"] = base_slug

        try:
            t_fixed, m_fixed = _sync_numbers(draft["title"], draft.get("meta_description", ""))
            draft["title"], draft["meta_description"] = t_fixed, m_fixed
        except Exception:
            md = (draft.get("meta_description") or "").strip()
            if md and len(md) > 156:
                draft["meta_description"] = md[:153] + "..."

        if not draft.get("article_content"):
            logger.error("normalize_draft produced empty content.")
            return None

        draft.setdefault("keyphrase", keyphrase)
        draft.setdefault("niche", niche)
        return draft

    def _insert_adsense_markers(self, content: str) -> str:
        """
        Insert up to N AdSense markers, avoiding code fences and duplicates.
        Positions: ~first 150 words, mid-article (~50%), and before last H2.
        """
        paragraphs_min_for_ads = 5
        markers = [m for m in self.config.adsense_placement_markers if isinstance(m, str)]
        num = min(len(markers), self.config.max_ads_per_article)

        if num == 0 or not content or len([p for p in content.split("\n\n") if p.strip()]) < paragraphs_min_for_ads:
            return content

        # De-duplicate markers (keep first instance of each marker text)
        seen = set()
        deduped = []
        for m in markers:
            if m.strip() and m.strip() not in seen:
                deduped.append(m.strip())
                seen.add(m.strip())
        markers = deduped
        num = min(len(markers), num)

        total_words = len(_WORD_RE.findall(content or ""))
        first_words = 150
        mid_words = max(200, int(total_words * 0.5))

        out = content
        placed = 0

        if placed < num and markers[placed].strip():
            pos = _safe_after_n_words(out, first_words)
            if pos is not None:
                out = _safe_insert(out, markers[placed], pos)
                placed += 1

        if placed < num and markers[placed].strip():
            pos2 = _safe_after_n_words(out, mid_words)
            if pos2 is not None:
                out = _safe_insert(out, markers[placed], pos2)
                placed += 1

        if placed < num and markers[placed].strip():
            pos3 = _safe_before_last_h2(out)
            out = _safe_insert(out, markers[placed], pos3)
            placed += 1

        if placed:
            logger.info(f"Inserted {placed} AdSense marker(s).")
        return out

    def generate_article(self, topic: str, keyphrase: str, niche: str) -> Optional["SEOEnhancementResult"]:
        logger.info(f"Generating article for topic: '{topic}'")
        draft = self._generate_initial_article_draft(topic, keyphrase, niche)
        if not draft:
            logger.error(f"Draft generation failed for '{topic}'.")
            return None

        # --- CTR-aware title synthesis (optional) + bounds + H1 sync + unique slug ---
        if self.config.enable_ctr_title:
            try:
                used_titles = _collect_used_titles(self.config.output_dir)
            except Exception:
                used_titles = []
            new_title = generate_title(topic=topic, keyphrase=keyphrase, niche=niche, used_titles=used_titles)
            if new_title and new_title.strip():
                bounded = _enforce_title_bounds(new_title.strip(), keyphrase, min_len=40, max_len=60)
                if bounded != draft["title"]:
                    logger.info(f"Title updated → '{bounded}' (from '{draft['title']}').")
                draft["title"] = bounded

        # Always dedupe slug after final title decision
        base_slug = slugify(draft["title"])[:75] or draft["slug"]
        draft["slug"] = _dedupe_slug(base_slug, self.config.output_dir, max_len=75)

        # Sync H1
        draft["article_content"] = _replace_h1(draft["article_content"], draft["title"])

        # Ensure early keyphrase mention
        draft["article_content"] = _ensure_keyphrase_lead(draft["article_content"], keyphrase)

        logger.info("Enhancing with SEOEnforcer…")
        try:
            seo_result = self.seo_enforcer.enhance_content(
                title=draft["title"],
                slug=draft["slug"],
                meta_desc=draft["meta_description"],
                content=draft["article_content"],
                keyphrase=keyphrase,
                niche=niche,
            )
        except Exception as e:
            logger.critical(f"SEO enhancement failed for '{topic}': {e}", exc_info=True)
            return None

        # ---------------- NEW: Prepublish adapter (YoastPreflight + optional Finalizer) ----------------
        # Pull knobs straight from settings.json and run a deterministic last-mile pass.
        try:
            content_pp, title_pp, meta_pp, pp_report = run_prepublish(
                settings=self.config.settings,
                content=seo_result.content,
                title=seo_result.title,
                meta=getattr(seo_result, "meta_description", getattr(seo_result, "meta", "")),
                keyphrase=keyphrase,
                niche=niche,
                article_linker=self.article_linker,
                logger=logger,
            )
            # Update the result with prepublish outcome
            seo_result.title = title_pp
            seo_result.content = content_pp
            if hasattr(seo_result, "meta_description"):
                seo_result.meta_description = meta_pp
            elif hasattr(seo_result, "meta"):
                seo_result.meta = meta_pp

            # attach metrics
            try:
                seo_result.yoast_seo_score = float(pp_report.get("finalizer", {}).get("metrics", {}).get("seo",
                                                pp_report.get("preflight", {}).get("seo")))
                seo_result.yoast_readability_score = float(pp_report.get("finalizer", {}).get("metrics", {}).get("readability",
                                                        pp_report.get("preflight", {}).get("readability")))
            except Exception:
                pass

            # optional notes/warnings snapshot
            seo_result.warnings = pp_report.get("preflight", {}).get("warnings")
            seo_result._prepublish_report = pp_report  # keep for debugging
        except Exception as e:
            logger.warning(f"Prepublish pass failed (continuing with enforcer output): {e}", exc_info=True)

        # Move AdSense placement to the very end, after all shaping
        seo_result.content = self._insert_adsense_markers(seo_result.content)

        logger.info(f"Completed article generation for '{topic}'")
        return seo_result

# ---------------- CLI / Standalone test ----------------
if __name__ == "__main__":
    from datetime import datetime
    try:
        load_dotenv()
        if "OPENAI_API_KEY" not in os.environ:
            print("ERROR: OPENAI_API_KEY is not set.")
            raise SystemExit(1)

        config_dir = Path("config")
        config_dir.mkdir(parents=True, exist_ok=True)
        config_path = config_dir / "settings.json"
        if not config_path.exists():
            default_settings_content = {
                "api_keys": {"openai": os.getenv("OPENAI_API_KEY")},
                "paths": {
                    "prompts_dir": "prompts",
                    "articles_output_dir": "output",
                    # make the default filenames explicit
                    "system_prompt_path": "prompts/system.md",
                    "article_base_path": "prompts/article_base.md"
                },
                "article_generation": {
                    "model": "gpt-4o",
                    "temperature": 0.7,
                    "max_tokens": 2000,
                    "word_count_target": [1200, 1500],
                    "structured_output": True
                },
                "yoast_compliance": {
                    "min_content_words": 1100,
                    "max_title_length": 60, "min_title_length": 40,
                    "max_meta_length": 156, "min_meta_length": 120,
                    "min_subheadings": 3,
                    "min_outbound_links": 2, "min_internal_links": 1,
                    "min_keyphrase_density": 0.5, "max_keyphrase_density": 2.5,
                    "flesch_reading_ease_good": 60,
                    "passive_voice_max_percent": 5,
                    "min_transition_word_percent": 30,
                    "max_long_sentence_percent": 20,
                    "max_paragraph_words": 160,
                    "max_consecutive_same_start": 0,
                    "target_seo_score": 95, "target_readability_score": 95,
                    "max_warnings": 1, "max_enhancement_iterations": 4,
                    "early_stop_patience": 2
                },
                "internal_linking": {
                    "max_outbound_links": 3,
                    "avoid_outbound_domains": [],
                    "strip_query_tracking": True,
                    "use_html_for_outbound": False,
                    "outbound_rel_default": "noopener noreferrer",
                    "outbound_target_blank": True
                },
                "finalizer": {
                    "enabled": True,
                    "run_if_within_points": 5,
                    "only_when_close": True,
                    "allow_seo_regression_points": 0,
                    "allow_read_regression_points": 0
                },
                "outbound_links_map": {
                    "general_python": ["https://www.python.org/doc/", "https://pypi.org/"],
                    "educational python": ["https://docs.python.org/3/tutorial/", "https://realpython.com/"],
                    "data science": ["https://pandas.pydata.org/docs/", "https://numpy.org/doc/"],
                    "web development": ["https://flask.palletsprojects.com/en/latest/", "https://docs.djangoproject.com/en/stable/"],
                    "machine learning": ["https://scikit-learn.org/stable/documentation.html", "https://pytorch.org/docs/stable/index.html"]
                },
                "authors": {"name": "PythonProHub Team"},
                "general": {"site_name": "PythonProHub"},
                "titles": {"enable_synthesizer": True},
                "adsense": {
                    "max_ads_per_article": 3,
                    "placement_markers": ["<!-- AD-1 -->", "<!-- AD-2 -->", "<!-- AD-3 -->"]
                },
                "wordpress": {
                    "base_url": "https://pythonprohub.com",
                    "api_base_url": "https://pythonprohub.com/wp-json"
                }
            }
            config_path.write_text(json.dumps(default_settings_content, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"Created default settings.json at {config_path}")

        prompts_dir = Path("prompts")
        prompts_dir.mkdir(parents=True, exist_ok=True)

        # Ensure defaults exist
        sys_path = prompts_dir / "system.md"
        if not sys_path.exists():
            sys_path.write_text("You are an expert technical writer. Keep outputs safe, accurate, and actionable.", encoding="utf-8")
            print(f"Created system.md at {sys_path}")

        base_path = prompts_dir / "article_base.md"
        if not base_path.exists():
            base_path.write_text(
                """Use the shared prompt at prompts/article_base.md with structure variants from config/structure_variants.json and style knobs from config/style_knobs.json.
Ensure output is MARKDOWN ONLY with this exact structure:
- First line: an H1 (# Title)
- Rich, narrative body (avoid rigid 'Introduction'/'Conclusion' headings)
- Practical code where helpful
Target {WORD_COUNT_MIN}–{WORD_COUNT_MAX} words.

Context:
- Topic: {ARTICLE_TOPIC}
- Keyphrase: {KEYPHRASE}
- Niche: {ARTICLE_NICHE}
- Author: {AUTHOR_NAME} @ {SITE_NAME}
""".strip(),
                encoding="utf-8"
            )
            print(f"Created article_base.md at {base_path}")

        # Optional local tracker for tests
        @dataclass
        class _DummyTracker:
            def get_unique_keyphrase(self, base_keyphrase: str, content_slug: str = "") -> str:
                return base_keyphrase

        cfg = AppConfig(settings_path=str(config_path))
        gen = ArticleGenerator(cfg, openai_client=None, keyphrase_tracker_instance=_DummyTracker())

        test_cases = [
            {"topic": "Building a Loan Default Prediction Model", "keyphrase": "Loan Default Prediction", "niche": "machine learning"},
            {"topic": "Advanced Python Decorators", "keyphrase": "Python Decorators", "niche": "Educational Python"},
            {"topic": "FastAPI Auth & Async Jobs", "keyphrase": "FastAPI Authentication", "niche": "web development"},
        ]

        for i, tc in enumerate(test_cases, 1):
            topic, keyphrase, niche = tc["topic"], tc["keyphrase"], tc["niche"]
            print(f"\n--- Test Case {i}: {topic} ---")
            res = gen.generate_article(topic, keyphrase, niche)
            if res:
                out_dir = cfg.output_dir / slugify(topic)
                out_dir.mkdir(parents=True, exist_ok=True)
                # Guard against double-H1: only add H1 if missing in res.content
                content = (res.content or "").lstrip()
                final_md = content if _H1_RE.match(content) else f"# {res.title}\n\n{content}"
                (out_dir / "article.md").write_text(final_md, encoding="utf-8")
                (out_dir / "metadata.json").write_text(json.dumps({
                    "title": res.title,
                    "meta_description": getattr(res, "meta_description", getattr(res, "meta", "")),
                    "slug": res.slug,
                    "keyphrase": getattr(res, "keyphrase", keyphrase),
                    "yoast_seo_score": getattr(res, "yoast_seo_score", None),
                    "yoast_readability_score": getattr(res, "yoast_readability_score", None),
                    "niche": niche,
                    "date_published": datetime.now().isoformat(),
                    "warnings": getattr(res, "warnings", None),
                    "prepublish_report": getattr(res, "_prepublish_report", None),
                }, indent=2, ensure_ascii=False), encoding="utf-8")
                logger.info(f"Saved article to {out_dir}")
            else:
                logger.error(f"Generation failed for: {topic}")

    except Exception as e:
        logger.critical(f"Standalone run error: {e}", exc_info=True)
        sys.exit(1)
