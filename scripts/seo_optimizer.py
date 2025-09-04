import re
import time
import os
import json
import random
from collections import Counter
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional, Dict, Any, Set
from dotenv import load_dotenv
import logging
from datetime import datetime
from pathlib import Path
import hashlib

import httpx
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type

# OpenAI SDK imports (v1+)
try:
    from openai import OpenAI, RateLimitError, APIError
except Exception:  # pragma: no cover
    from openai import OpenAI  # type: ignore
    RateLimitError = Exception  # type: ignore
    APIError = Exception  # type: ignore

# --- slugify: prefer project util, fallback to local ---
def _fallback_slugify(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    text = re.sub(r"-{2,}", "-", text)
    return text or "article"

try:
    from .utils.utils import slugify  # type: ignore
except Exception:
    try:
        from utils.utils import slugify  # type: ignore
    except Exception:
        slugify = _fallback_slugify  # type: ignore

# --- Logging setup -----------------------------------------------------------
LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,  # INFO in prod; switch to DEBUG for deep tracing
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(LOG_DIR / "seo_optimizer.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
logger = logging.getLogger(__name__)

load_dotenv()

# --- Global constants & helpers ---------------------------------------------
DEFAULT_IMAGE_URL = "https://pythonprohub.com/wp-content/uploads/default-python.png"
DEFAULT_SITE_DOMAIN = "https://pythonprohub.com/"

# Stable defaults for enhancement prompts; can be overridden by settings["enhancement_prompts"]
ENHANCEMENT_PROMPTS = {
    "readability_global": (
        "Improve readability of the following article while preserving ALL information and markdown.\n"
        "Rules:\n"
        "• Prefer active voice; split sentences >20 words; break long paragraphs.\n"
        "• Keep existing headings; do NOT add generic headers (Introduction/Conclusion/FAQs).\n"
        "• Keep ALL code fences/inline code intact.\n"
        "• Maintain keyphrase density at ~{kp_density:.2f}% (±0.20%), "
        "and keep the keyphrase in the first 300 characters.\n"
        "Return ONLY the rewritten article text.\n\n{content}"
    ),
    "length_expand": (
        "Expand the article to land in {target_low}-{target_high} words without padding or fluff.\n"
        "Rules:\n"
        "• Elaborate on existing points with concrete detail/examples.\n"
        "• Keep current headings; do NOT add generic headers (Introduction/Conclusion/FAQs).\n"
        "• Preserve markdown and ALL code blocks.\n"
        "• Maintain keyphrase density near {kp_density:.2f}% (±0.20%).\n"
        "Return ONLY the expanded article.\n\n{content}"
    ),
    "subheads_keyphrase": (
        "The article is about the exact keyphrase: '{keyphrase}'. If needed, add up to {need_h2} new H2 sections "
        "(2–3 sentences each) and/or lightly rewrite some H2/H3 so at least {target_kp} heading(s) contain the exact keyphrase.\n"
        "Rules:\n"
        "• Keep content and code; only add small, natural sections.\n"
        "• Do NOT add generic headers (Introduction/Conclusion/FAQs).\n"
        "• Maintain keyphrase density near {kp_density:.2f}% (±0.20%).\n"
        "Return the full updated article ONLY.\n\n{content}"
    ),
    # Defaults for configurable small prompts
    "title_opt": (
        "Rewrite this title to include '{KEYPHRASE}' naturally; keep it engaging and within "
        "{MIN_TITLE}-{MAX_TITLE} characters. Return ONLY the title.\n\n{TITLE}"
    ),
    "meta_gen": (
        "Meta description {MIN_META}-{MAX_META} chars for an article titled '{TITLE}'. "
        "Include the exact keyphrase '{KEYPHRASE}' naturally. No markup, only the sentence. "
        "Here is the opening context:\n\n{CONTENT_SNIPPET}"
    ),
}

AI_BOILERPLATE = (
    "as an ai", "as a language model", "i cannot", "i'm just a language model",
    "i am not able to", "cannot provide legal advice"
)
TRANSITION_WORDS = [
    "however", "moreover", "therefore", "in addition", "consequently", "thus",
    "for example", "in fact", "firstly", "secondly", "finally", "as a result",
    "in conclusion", "similarly", "conversely", "meanwhile", "furthermore",
    "accordingly", "in summary", "to conclude", "next", "then", "otherwise",
    "likewise", "although", "even though", "despite", "in contrast",
    "on the other hand", "specifically", "in particular", "for instance",
    "in short", "overall", "hence", "whereas", "in other words", "to illustrate",
    "in essence", "ultimately", "additionally", "simultaneously"
]

SYSTEM_BASE = (
    "You are an SEO/content optimization expert. Improve text readability and SEO while preserving meaning "
    "and markdown structure. Use clear, active voice; concise sentences; natural flow; and avoid filler. "
    "Return only the requested text, no extra commentary."
)

CACHE_DIR = Path("cache/seo_optimizer")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def _cache_key(model: str, prompt: str, max_tokens: int, temperature: float, tag: str) -> Path:
    raw = f"{model}|{max_tokens}|{temperature}|{tag}||{prompt}"
    key = hashlib.sha1(raw.encode("utf-8")).hexdigest()
    return CACHE_DIR / f"{key}.json"

def _maybe_load_cache(model: str, prompt: str, max_tokens: int, temperature: float, tag: str) -> Optional[str]:
    p = _cache_key(model, prompt, max_tokens, temperature, tag)
    if p.exists():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            return data.get("content", "")
        except Exception:
            return None
    return None

def _store_cache(model: str, prompt: str, max_tokens: int, temperature: float, tag: str, content: str, usage: Optional[Dict[str, Any]]) -> None:
    p = _cache_key(model, prompt, max_tokens, temperature, tag)
    try:
        p.write_text(json.dumps({"content": content, "usage": usage}, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass

@dataclass
class SEOEnhancementResult:
    content: str
    meta_description: str
    meta: str  # alias for pipelines expecting "meta"
    slug: str
    keyphrase: str
    keywords: str
    image_alt_texts: List[str]
    yoast_seo_score: int
    yoast_readability_score: int
    warnings: List[str]
    schema_markup: Optional[Dict] = None
    passed_all_checks: bool = False
    title: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SEOEnforcer:
    """
    Production-ready, token-efficient SEO optimizer.

    Now supports config-driven prompts via settings["enhancement_prompts"] with
    robust placeholder formatting (lower/UPPER case keys supported).
    """

    def __init__(
        self,
        app_config_settings: Dict[str, Any],
        openai_client: Optional[OpenAI] = None,
        keyphrase_tracker_instance: Optional[Any] = None,
    ):
        self.app_config_settings = app_config_settings or {}
        self.current_keyphrase: str = ""
        self._validate_config()

        # Site/domain & media (used for internal vs outbound checks & schema)
        self.site_domain: str = (
            self.app_config_settings.get("site", {}).get("base_url")
            or os.getenv("SITE_DOMAIN")
            or DEFAULT_SITE_DOMAIN
        ).rstrip("/") + "/"

        self.default_image_url: str = (
            self.app_config_settings.get("images", {}).get("default_article")
            or DEFAULT_IMAGE_URL
        )

        pub = self.app_config_settings.get("publisher", {}) or {}
        self.publisher_name: str = pub.get("name", "PythonProHub")
        self.publisher_logo_url: str = pub.get("logo_url", f"{self.site_domain.rstrip('/')}/wp-content/uploads/logo.png")

        # Word targets for expansion (fallback to 1200–1300)
        ag = (self.app_config_settings.get("article_generation") or {})
        wct = ag.get("word_count_target") or [1200, 1300]
        self.WORD_TARGET_LOW = int(wct[0]) if isinstance(wct, (list, tuple)) and len(wct) >= 1 else 1200
        self.WORD_TARGET_HIGH = int(wct[1]) if isinstance(wct, (list, tuple)) and len(wct) >= 2 else max(self.WORD_TARGET_LOW + 50, 1300)

        # Models (overridable from settings)
        models_cfg = self.app_config_settings.get("models", {})
        self.HEAVY_MODEL = models_cfg.get("heavy", "gpt-4o")
        self.LIGHT_MODEL = models_cfg.get("light", "gpt-4o-mini")

        # OpenAI client
        if openai_client:
            self.client = openai_client
            logger.info("SEOEnforcer initialized with provided OpenAI client.")
        else:
            api_key = self.app_config_settings.get("api_keys", {}).get("openai") or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key not found for SEOEnforcer.")
            http_client = httpx.Client(timeout=httpx.Timeout(30.0, connect=10.0, read=30.0, write=30.0))
            self.client = OpenAI(api_key=api_key, http_client=http_client)
            logger.warning("SEOEnforcer constructed its own OpenAI client.")

        # Keyphrase tracker (optional)
        if keyphrase_tracker_instance:
            self.keyphrase_tracker = keyphrase_tracker_instance
        else:
            @dataclass
            class _DummyKeyphraseConfig:
                storage_path: Path = Path("dummy_used_keyphrases.json")
            class _DummyKeyphraseTracker:
                def __init__(self, config: Optional[_DummyKeyphraseConfig] = None):
                    self.used_keyphrases: Dict[str, bool] = {}
                def get_unique_keyphrase(self, base_keyphrase: str, content_slug: str = "") -> str:
                    k = base_keyphrase.lower()
                    if k not in self.used_keyphrases:
                        self.used_keyphrases[k] = True
                        return base_keyphrase
                    variant = f"{base_keyphrase}-{datetime.now().strftime('%Y%m%d%H%M%S')}"
                    self.used_keyphrases[variant.lower()] = True
                    return variant
            self.keyphrase_tracker = _DummyKeyphraseTracker()

        # Regex utils
        self._passive_voice_regex = re.compile(r"\b(is|are|was|were|be|been|being)\s+(\w+ed)\b", re.IGNORECASE)
        self._sentence_splitter_regex = re.compile(r"(?<=[.!?])\s+(?=[A-Z]|\n|$)|(?<=\n\n)")
        self._word_tokenizer_regex = re.compile(r"\b\w+\b")

        # Usage tracking
        self.token_usage = {"prompt": 0, "completion": 0, "total": 0}

        # System prompt
        self.SYSTEM = SYSTEM_BASE

        # Load prompts (overridable via settings["enhancement_prompts"])
        self.prompts = self._load_prompts_from_settings(self.app_config_settings)

        # Derive a target density midpoint for config templates that reference it
        self.TARGET_DENSITY = round(
            (float(self.MIN_KEYPHRASE_DENSITY) + float(self.MAX_KEYPHRASE_DENSITY)) / 2.0, 2
        )

    # --- Config validation ----------------------------------------------------
    def _validate_config(self) -> None:
        required = [
            "min_content_words", "max_title_length", "min_title_length",
            "max_meta_length", "min_meta_length", "min_subheadings",
            "min_outbound_links", "min_internal_links",
            "min_keyphrase_density", "max_keyphrase_density",
            "flesch_reading_ease_good", "passive_voice_max_percent",
            "min_transition_word_percent", "max_long_sentence_percent",
            "max_paragraph_words", "max_consecutive_same_start",
            "target_seo_score", "target_readability_score",
            "max_warnings", "max_enhancement_iterations"
        ]
        yoast_settings = self.app_config_settings.get("yoast_compliance", {})
        for setting in required:
            if setting not in yoast_settings:
                raise ValueError(f"Missing required yoast_compliance setting: {setting}")
        for key, value in yoast_settings.items():
            setattr(self, key.upper(), value)

        # New: thresholds & guardrails (with sensible defaults)
        self.EARLY_STOP_PATIENCE = int(yoast_settings.get("early_stop_patience", 2))
        self.GOOD_ENOUGH_SEO = int(yoast_settings.get("good_enough_seo", 80))
        self.GOOD_ENOUGH_READ = int(yoast_settings.get("good_enough_read", 50))
        self.GOOD_ENOUGH_MAX_WARNINGS = int(yoast_settings.get("good_enough_max_warnings", 5))
        self.GUARDRAIL_ALLOWED_SEO_DROP = int(yoast_settings.get("guardrail_allowed_seo_drop", 5))

        # Finalizer defaults (optional block)
        fin = (self.app_config_settings.get("finalizer") or {})
        self.FINALIZER_ENABLED = bool(fin.get("enabled", True))
        self.FINALIZER_MARGIN = int(fin.get("run_if_within_points", 5))
        self.FINALIZER_ALLOW_SEO_REGRESS = float(fin.get("allow_seo_regression_points", 0))
        self.FINALIZER_ALLOW_READ_REGRESS = float(fin.get("allow_read_regression_points", 0))
        self.FINALIZER_ONLY_WHEN_CLOSE = bool(fin.get("only_when_close", True))

    # --- Prompt loading/formatting -------------------------------------------
    def _load_prompts_from_settings(self, settings: Dict[str, Any]) -> Dict[str, str]:
        """Merge defaults with settings['enhancement_prompts'] (if provided)."""
        merged = dict(ENHANCEMENT_PROMPTS)
        user = (settings or {}).get("enhancement_prompts") or {}
        if isinstance(user, dict):
            for k, v in user.items():
                if isinstance(v, str) and v.strip():
                    merged[k] = v
        return merged

    @staticmethod
    def _with_uppercase_aliases(d: Dict[str, Any]) -> Dict[str, Any]:
        """Provide both lower and UPPERCASE keys to satisfy mixed template styles."""
        out = dict(d)
        for k, v in list(d.items()):
            out[k.upper()] = v
        return out

    def _render_prompt(self, key: str, **kwargs) -> str:
        """Format a prompt template safely with flexible placeholder names."""
        tmpl = self.prompts.get(key, ENHANCEMENT_PROMPTS.get(key, ""))
        # Provide a rich default variable set; user kwargs can override
        base_vars: Dict[str, Any] = dict(
            MIN_TITLE=self.MIN_TITLE_LENGTH,
            MAX_TITLE=self.MAX_TITLE_LENGTH,
            MIN_META=self.MIN_META_LENGTH,
            MAX_META=self.MAX_META_LENGTH,
            MIN_WORDS=self.WORD_TARGET_LOW,
            MAX_WORDS=self.WORD_TARGET_HIGH,
            TARGET_LOW=self.WORD_TARGET_LOW,
            TARGET_HIGH=self.WORD_TARGET_HIGH,
            TARGET_DENSITY=self.TARGET_DENSITY,
            FLESCH_MIN=self.FLESCH_READING_EASE_GOOD,
        )
        base_vars.update(kwargs or {})
        # Ensure both lower and upper keys exist
        vars_all = self._with_uppercase_aliases(base_vars)

        class _SafeDict(dict):
            def __missing__(self, k):
                # leave the placeholder visible if not provided
                return "{" + k + "}"

        try:
            return tmpl.format_map(_SafeDict(vars_all))
        except Exception as e:
            logger.warning(f"Prompt render failed for '{key}': {e}")
            # last resort: return raw with best-effort substitutions
            try:
                return tmpl.format(**vars_all)
            except Exception:
                return tmpl

    # --- Hygiene --------------------------------------------------------------
    def _collapse_blank_lines(self, text: str) -> str:
        text = re.sub(r'[ \t]+$', '', text, flags=re.MULTILINE)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    def _normalize_code_fences(self, text: str) -> str:
        # Normalize any fence like ````python to ```
        text = re.sub(r'`{3,}([a-zA-Z0-9_+-]*)', r'```\1', text)
        text = re.sub(r'~{3,}([a-zA-Z0-9_+-]*)', r'```\1', text)
        # Remove "copy"/"bash copy" style labels
        text = re.sub(r'```(?:copy|copyable|clipboard|code)[^\n]*\n', '```\n', text, flags=re.IGNORECASE)
        return text

    def _remove_images_in_body(self, text: str) -> str:
        """Deprecated: keep images; only use for specialized pipelines."""
        return re.sub(r'!\[[^\]]*\]\([^)]*\)', '', text)

    def _strip_ai_boilerplate(self, text: str) -> str:
        if any(p in text.lower() for p in AI_BOILERPLATE):
            text = re.sub(r'(?i)as an ai.*?(?:\.|\n)', '', text)
            text = re.sub(r'(?i)as a language model.*?(?:\.|\n)', '', text)
            text = re.sub(r"(?i)i(?:\s|')?m just a language model.*?(?:\.|\n)", '', text)
            text = re.sub(r'(?i)i cannot.*?(?:\.|\n)', '', text)
        return text

    def _single_h1_and_deduplicate(self, text: str) -> Tuple[str, Optional[str]]:
        lines = text.strip().splitlines()
        clean = []
        seen_h1 = False
        title = None
        seen_headers: Set[str] = set()
        for ln in lines:
            if ln.startswith('# '):
                if not seen_h1:
                    seen_h1 = True
                    title = ln[2:].strip()
                    clean.append(ln)
                else:
                    h2 = '## ' + ln[2:].strip()
                    key = h2.lower()
                    if key not in seen_headers:
                        clean.append(h2)
                        seen_headers.add(key)
            elif re.match(r'#{2,6}\s', ln):
                hdr_text = re.sub(r'^#{2,6}\s+', '', ln).strip()
                key = (re.match(r'#{2,6}', ln).group(0) + ' ' + hdr_text).lower()
                if key not in seen_headers:
                    clean.append(ln)
                    seen_headers.add(key)
            else:
                clean.append(ln)
        return ("\n".join(clean).strip(), title)

    def _remove_generic_headers(self, text: str) -> str:
        pattern = re.compile(
            r'^(#{1,6})\s*(introduction|intro|final thoughts|summary|tl;dr|overview|step-by-step guide|what this article covers)\s*$',
            re.IGNORECASE | re.MULTILINE
        )
        return re.sub(pattern, '', text).strip()

    def _validate_and_clean_raw_text(self, text: str) -> str:
        if not text:
            return ""
        clean_text = text
        clean_text = self._strip_ai_boilerplate(clean_text)
        clean_text, _ = self._single_h1_and_deduplicate(clean_text)
        clean_text = self._remove_generic_headers(clean_text)
        clean_text = self._normalize_code_fences(clean_text)  # preserve languages
        clean_text = self._collapse_blank_lines(clean_text)
        return clean_text

    def _ensure_early_keyphrase_mention(self, content: str, keyphrase: str) -> str:
        if not keyphrase:
            return content
        words = content.split()
        first_50 = " ".join(words[:50]).lower()
        if keyphrase.lower() in first_50:
            return content
        lead = f"**{keyphrase}** frames the core challenge in this guide. "
        parts = content.split("\n\n", 1)
        if parts and parts[0].startswith("# "):
            head = parts[0]
            rest = parts[1] if len(parts) > 1 else ""
            return f"{head}\n\n{lead}{rest}"
        return f"{lead}{content}"

    # --- Metrics helpers ------------------------------------------------------
    def _kp_density(self, content: str, keyphrase: str) -> float:
        """Keyphrase density in percent based on whole-phrase matches."""
        words = len(self._word_tokenizer_regex.findall(content or ""))
        if words == 0 or not keyphrase.strip():
            return 0.0
        kp_pat = re.compile(rf"\b{re.escape(keyphrase.strip().lower())}\b", flags=re.IGNORECASE)
        hits = len(kp_pat.findall((content or "").lower()))
        return (hits / words) * 100.0

    # --- OpenAI call (centralized, cached, logged) ----------------------------
    @retry(
        wait=wait_random_exponential(min=1, max=12),
        stop=stop_after_attempt(2),
        reraise=True,
        retry=retry_if_exception_type((RateLimitError, APIError, httpx.HTTPError, TimeoutError)),
    )
    def _chat(
        self,
        prompt: str,
        *,
        model: str,
        max_tokens: int,
        temperature: float,
        tag: str,
        response_format: Optional[Dict[str, Any]] = None
    ) -> str:
        try:
            mt = int(max(32, min(int(max_tokens), 1600)))
        except Exception:
            mt = 400
        try:
            temp = float(temperature)
            if not 0.0 <= temp <= 1.0:
                temp = 0.7
        except Exception:
            temp = 0.7

        use_cache = tag in {"title_opt", "meta_gen", "subheads_keyphrase", "alt_enrich"} or mt <= 200
        if use_cache:
            cached = _maybe_load_cache(model, prompt, mt, temp, tag)
            if cached:
                return cached

        api_kwargs: Dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": self.SYSTEM},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": mt,
            "temperature": temp,
        }
        if response_format:
            api_kwargs["response_format"] = response_format

        t0 = time.time()
        resp = self.client.chat.completions.create(**api_kwargs)
        elapsed = time.time() - t0

        try:
            u = resp.usage
            self.token_usage["prompt"] += int(getattr(u, "prompt_tokens", 0) or 0)
            self.token_usage["completion"] += int(getattr(u, "completion_tokens", 0) or 0)
            self.token_usage["total"] += int(getattr(u, "total_tokens", 0) or 0)
            logger.info(f"[TOKENS] {tag} → prompt={getattr(u,'prompt_tokens',0)} completion={getattr(u,'completion_tokens',0)} total={getattr(u,'total_tokens',0)} ({elapsed:.2f}s)")
        except Exception:
            logger.info(f"[TOKENS] {tag} → (no usage) ({elapsed:.2f}s)")

        raw_text = (resp.choices[0].message.content or "").strip() if getattr(resp, "choices", None) else ""
        text = self._validate_and_clean_raw_text(raw_text)

        if use_cache and text:
            usage_dict = None
            try:
                usage_dict = {
                    "prompt": int(getattr(resp.usage, "prompt_tokens", 0) or 0),
                    "completion": int(getattr(resp.usage, "completion_tokens", 0) or 0),
                    "total": int(getattr(resp.usage, "total_tokens", 0) or 0),
                }
            except Exception:
                pass
            _store_cache(model, prompt, mt, temp, tag, text, usage_dict)
        return text

    # --- Readability & SEO calculations --------------------------------------
    def _count_syllables(self, word: str) -> int:
        word = word.lower()
        if not word:
            return 0
        count = 0
        vowels = "aeiouy"
        vg = False
        for ch in word:
            if ch in vowels:
                if not vg:
                    count += 1
                    vg = True
            else:
                vg = False
        if word.endswith('e'):
            count -= 1
        if word.endswith('le') and len(word) > 2 and word[-3] not in vowels:
            count += 1
        if word.endswith('ed') and sum(ch in vowels for ch in word) == 1:
            if len(word) > 2 and word[-3] not in vowels:
                count -= 1
        return max(1, count)

    def _calculate_flesch_reading_ease(self, text: str) -> float:
        sentences = [s.strip() for s in self._sentence_splitter_regex.split(text) if s.strip()]
        words = self._word_tokenizer_regex.findall(text)
        if not sentences or not words:
            return 0.0
        total_syllables = sum(self._count_syllables(w) for w in words)
        score = 206.835 - 1.015 * (len(words) / len(sentences)) - 84.6 * (total_syllables / len(words))
        return score

    def _count_outbound_links(self, text: str) -> int:
        site_domain_normalized = self.site_domain.replace("https://", "").replace("http://", "").strip("/")
        md_urls = re.findall(r'\((https?://[^)]+)\)', text)
        bare_urls = re.findall(r'\bhttps?://\S+', text)
        urls = set(md_urls + bare_urls)
        return sum(1 for u in urls if site_domain_normalized not in u)

    def _calculate_seo_score(self, data: Dict[str, Any]) -> Tuple[int, List[str]]:
        score = 0
        warnings: List[str] = []
        keyphrase_lower = (data.get("keyphrase") or "").lower()
        content_lower = (data.get("content") or "").lower()
        word_count = len(self._word_tokenizer_regex.findall(data.get("content") or ""))

        title = data.get("title") or ""
        meta_description = data.get("meta_description") or ""
        slug_val = data.get("slug") or ""

        if keyphrase_lower and keyphrase_lower in title.lower():
            score += 10
        else:
            warnings.append("SEO: Keyphrase missing in title.")

        paragraphs = [p.strip() for p in content_lower.split("\n\n") if p.strip()]
        intro_paragraph = paragraphs[0].lower() if paragraphs else ""
        if keyphrase_lower and keyphrase_lower in intro_paragraph[:300]:
            score += 10
        else:
            warnings.append("SEO: Keyphrase missing in introduction.")

        if keyphrase_lower and keyphrase_lower in meta_description.lower():
            score += 10
        else:
            warnings.append("SEO: Keyphrase missing in meta description.")

        if keyphrase_lower and slugify(keyphrase_lower) in slug_val:
            score += 10
        else:
            warnings.append("SEO: Keyphrase missing in slug.")

        # Subheading KPI
        subheads = re.findall(r"^(?:##|###)\s+(.*)$", data.get("content") or "", flags=re.MULTILINE)
        subheading_matches_kp = sum(1 for txt in subheads if keyphrase_lower and keyphrase_lower in (txt or "").lower())
        if subheading_matches_kp >= 1:
            score += 5
        else:
            warnings.append("SEO: Keyphrase not found in any subheading.")

        # Density (whole-phrase)
        if keyphrase_lower.strip():
            kp_pat = re.compile(rf"\b{re.escape(keyphrase_lower)}\b", flags=re.IGNORECASE)
            keyphrase_count = len(kp_pat.findall(content_lower))
        else:
            keyphrase_count = 0
        density = (keyphrase_count / word_count) * 100 if word_count > 0 else 0
        if self.MIN_KEYPHRASE_DENSITY <= density <= self.MAX_KEYPHRASE_DENSITY:
            score += 10
        else:
            warnings.append(
                f"SEO: Keyphrase density {density:.2f}% (target {self.MIN_KEYPHRASE_DENSITY}-{self.MAX_KEYPHRASE_DENSITY}%)"
            )

        if self.MIN_TITLE_LENGTH <= len(title) <= self.MAX_TITLE_LENGTH:
            score += 5
        else:
            warnings.append(
                f"SEO: Title length ({len(title)}) not optimal (target {self.MIN_TITLE_LENGTH}-{self.MAX_TITLE_LENGTH} chars)."
            )

        if self.MIN_META_LENGTH <= len(meta_description) <= self.MAX_META_LENGTH:
            score += 5
        else:
            warnings.append(
                f"SEO: Meta description length ({len(meta_description)}) not optimal (target {self.MIN_META_LENGTH}-{self.MAX_META_LENGTH} chars)."
            )

        if word_count >= self.MIN_CONTENT_WORDS:
            score += 10
        else:
            warnings.append(f"SEO: Content word count ({word_count}) too low (target {self.MIN_CONTENT_WORDS}+ words).")

        h2_count = (data.get("content") or "").count("\n## ")
        if h2_count >= self.MIN_SUBHEADINGS:
            score += 5
        else:
            warnings.append(f"SEO: Not enough H2 subheadings ({h2_count}, target {self.MIN_SUBHEADINGS}+).")

        # Link checks
        site_root = self.site_domain.rstrip("/")
        md_urls = set(re.findall(r'\((https?://[^)]+)\)', data.get("content") or ""))
        bare_urls = set(re.findall(r'\bhttps?://\S+', data.get("content") or ""))
        all_urls = md_urls | bare_urls

        outbound_link_count = sum(1 for u in all_urls if site_root not in u)
        if outbound_link_count >= self.MIN_OUTBOUND_LINKS:
            score += 5
        else:
            warnings.append(
                f"SEO: Not enough outbound links ({outbound_link_count}, target {self.MIN_OUTBOUND_LINKS}+)."
            )

        internal_link_count = sum(1 for u in all_urls if site_root in u)
        if internal_link_count >= self.MIN_INTERNAL_LINKS:
            score += 5
        else:
            warnings.append(
                f"SEO: Not enough internal links ({internal_link_count}, target {self.MIN_INTERNAL_LINKS}+)."
            )

        image_present = bool(re.search(r"!\[.*?\]\(https?://[^\s)]+\)", data.get("content") or ""))
        if image_present:
            score += 5
        else:
            warnings.append("SEO: Missing image with alt text.")

        image_alt_keyphrase_found = False
        for alt_text, _ in re.findall(r"!\[(.*?)\]\((.*?)\)", data.get("content") or ""):
            if keyphrase_lower and keyphrase_lower in (alt_text or "").lower():
                image_alt_keyphrase_found = True
                break
        if image_alt_keyphrase_found:
            score += 5
        else:
            warnings.append("SEO: Keyphrase not found in image alt text.")

        return min(100, score), warnings

    def _calculate_readability_score(self, content_text: str) -> Tuple[int, List[str]]:
        score = 0
        warnings: List[str] = []
        sentences = [s.strip() for s in self._sentence_splitter_regex.split(content_text) if s.strip()]
        total_sentences = len(sentences)
        words = self._word_tokenizer_regex.findall(content_text)
        total_words = len(words)

        if total_sentences == 0:
            return 0, ["READABILITY: Content is empty or has no sentences."]

        flesch_score = self._calculate_flesch_reading_ease(content_text)
        if flesch_score >= self.FLESCH_READING_EASE_GOOD:
            score += 20
        elif flesch_score >= 50:
            score += 10
            warnings.append(
                f"READABILITY: Flesch score {flesch_score:.2f} OK but could be better (target {self.FLESCH_READING_EASE_GOOD}+)."
            )
        else:
            warnings.append(
                f"READABILITY: Flesch score {flesch_score:.2f} is low (target {self.FLESCH_READING_EASE_GOOD}+)."
            )

        passive_sentences_count = sum(1 for s in sentences if self._passive_voice_regex.search(s))
        passive_percent = (passive_sentences_count / total_sentences) * 100
        if passive_percent <= self.PASSIVE_VOICE_MAX_PERCENT:
            score += 20
        else:
            warnings.append(
                f"READABILITY: Too much passive voice ({passive_percent:.2f}% of sentences). Target <{self.PASSIVE_VOICE_MAX_PERCENT}%."
            )

        transition_word_count = sum(
            1 for s in sentences
            if any(re.search(r"\b" + re.escape(tw.lower()) + r"\b", s.lower()) for tw in TRANSITION_WORDS)
        )
        transition_percent = (transition_word_count / total_sentences) * 100 if total_sentences > 0 else 0
        if transition_percent >= self.MIN_TRANSITION_WORD_PERCENT:
            score += 20
        else:
            warnings.append(
                f"READABILITY: Not enough transition words ({transition_percent:.2f}%). Target >{self.MIN_TRANSITION_WORD_PERCENT}%."
            )

        long_sentences = [s for s in sentences if len(self._word_tokenizer_regex.findall(s)) > 20]
        long_sentence_percent = (len(long_sentences) / total_sentences) * 100 if total_sentences > 0 else 0
        if long_sentence_percent <= self.MAX_LONG_SENTENCE_PERCENT:
            score += 20
        else:
            warnings.append(
                f"READABILITY: Too many long sentences ({long_sentence_percent:.2f}% >20 words). Target <{self.MAX_LONG_SENTENCE_PERCENT}%."
            )

        paragraphs = [p.strip() for p in content_text.split("\n\n") if p.strip()]
        long_paragraphs_count = sum(
            1 for p in paragraphs if len(self._word_tokenizer_regex.findall(p)) > self.MAX_PARAGRAPH_WORDS
        )
        if long_paragraphs_count == 0:
            score += 10
        else:
            warnings.append(
                f"READABILITY: {long_paragraphs_count} paragraph(s) too long (> {self.MAX_PARAGRAPH_WORDS} words)."
            )

        consecutive_same_start = 0
        for i in range(len(sentences) - 1):
            s1_words = self._word_tokenizer_regex.findall(sentences[i].lower())
            s2_words = self._word_tokenizer_regex.findall(sentences[i + 1].lower())
            if s1_words and s2_words and s1_words[0] == s2_words[0]:
                consecutive_same_start += 1
        if consecutive_same_start <= self.MAX_CONSECUTIVE_SAME_START:
            score += 10
        else:
            warnings.append(
                f"READABILITY: {consecutive_same_start} instance(s) of consecutive sentences starting with same word (target {self.MAX_CONSECUTIVE_SAME_START})."
            )

        return min(100, score), warnings

    # --- Global readability rewrite (uses config template) --------------------
    def _global_readability_rewrite(self, content: str, keyphrase: str) -> str:
        words = len(self._word_tokenizer_regex.findall(content))
        mt = min(max(int(words * 1.1), 400), 1400)
        kp_density = self._kp_density(content, keyphrase)
        prompt = self._render_prompt(
            "readability_global",
            content=content,
            CONTENT=content,
            keyphrase=keyphrase,
            kp_density=kp_density,
        )
        try:
            improved = self._chat(prompt, model=self.HEAVY_MODEL, max_tokens=mt, temperature=0.5, tag="readability_global")
            return improved or content
        except Exception as e:
            logger.warning(f"Global readability rewrite failed: {e}")
            return content

    # --- Deterministic simplifier (fallback) ----------------------------------
    def _baseline_simplify(self, text: str) -> str:
        replacements = {
            "utilize": "use", "approximately": "about", "subsequently": "then",
            "prior to": "before", "in order to": "to", "however": "but",
            "therefore": "so", "furthermore": "also", "moreover": "also",
            "consequently": "so", "demonstrate": "show", "illustrate": "show",
            "additionally": "also", "nevertheless": "but", "notwithstanding": "despite",
            "utilization": "use", "commence": "start", "terminate": "end",
            "endeavor": "try", "facilitate": "help", "implement": "do", "objective": "goal",
            "numerous": "many", "sufficient": "enough", "alternatively": "or",
            "delineate": "explain", "with respect to": "about", "a plethora of": "many",
            "is comprised of": "consists of", "due to the fact that": "because",
            "for the purpose of": "to", "in the event that": "if",
        }
        text = re.sub(r"([,;])\s+(and|but|or|however|therefore)\s+", r". \2 ", text, flags=re.IGNORECASE)
        text = re.sub(r"\.\s*(and|but|or)\b", r".\n\n\1", text, flags=re.IGNORECASE)
        for k, v in replacements.items():
            text = re.sub(rf"\b{re.escape(k)}\b", v, text, flags=re.IGNORECASE)
        text = re.sub(r"\s*\([^)]{25,}\)", "", text)
        text = re.sub(r"\[[^\]]]{30,}\]", "", text)
        text = re.sub(r'(?<!\.)\.\s*', '. ', text)
        text = re.sub(r'\.{2,}\s*', '... ', text)
        return text

    # --- Title / slug / meta / keywords / image-alts --------------------------
    def _optimize_title(self, title: str, keyphrase: str) -> str:
        title = (title or "").strip()
        kp = (keyphrase or "").strip()
        if not title and kp:
            return f"{kp}: A Practical Guide"
        elif not title:
            return "A Comprehensive Guide"

        # Prefer user template if provided
        tpl_prompt = self._render_prompt(
            "title_opt",
            KEYPHRASE=kp,
            MIN_TITLE=self.MIN_TITLE_LENGTH,
            MAX_TITLE=self.MAX_TITLE_LENGTH,
            TITLE=title,
        )
        try:
            llm = self._chat(tpl_prompt, model=self.LIGHT_MODEL, max_tokens=self.MAX_TITLE_LENGTH, temperature=0.4, tag="title_opt")
        except Exception:
            llm = ""

        if llm and kp and kp.lower() in llm.lower() and self.MIN_TITLE_LENGTH <= len(llm) <= self.MAX_TITLE_LENGTH:
            title = llm
        else:
            if kp and len(f"{kp}: {title}") <= self.MAX_TITLE_LENGTH:
                title = f"{kp}: {title}"
            elif kp and len(f"{title} - {kp}") <= self.MAX_TITLE_LENGTH:
                title = f"{title} - {kp}"
            else:
                truncated_len = max(self.MIN_TITLE_LENGTH, self.MAX_TITLE_LENGTH - len(kp) - 3)
                title = f"{title[:truncated_len].strip()}... {kp}" if kp else title

        if len(title) > self.MAX_TITLE_LENGTH:
            short_prompt = self._render_prompt(
                "title_opt",
                KEYPHRASE=kp, MIN_TITLE=self.MIN_TITLE_LENGTH, MAX_TITLE=self.MAX_TITLE_LENGTH, TITLE=title
            )
            short = self._chat(short_prompt, model=self.LIGHT_MODEL, max_tokens=self.MAX_TITLE_LENGTH, temperature=0.3, tag="title_opt")
            title = short if short and len(short) <= self.MAX_TITLE_LENGTH else title[: self.MAX_TITLE_LENGTH].rstrip()

        if len(title) < self.MIN_TITLE_LENGTH and kp:
            suffix = " - A Comprehensive Guide"
            if len(title) + len(suffix) <= self.MAX_TITLE_LENGTH:
                title += suffix

        if kp and kp.lower() not in title.lower():
            if len(f"{kp}: {title}") <= self.MAX_TITLE_LENGTH:
                title = f"{kp}: {title}"
            else:
                title = (kp + " " + title)[:self.MAX_TITLE_LENGTH].strip()
        return title.strip()

    def _enforce_slug_keyphrase(self, slug_text: str, keyphrase: str) -> str:
        base_slug = slugify(slug_text or "")
        kp_slug = slugify(keyphrase or "")
        if not kp_slug:
            return (base_slug or "article")[:80].strip("-")
        if kp_slug not in base_slug:
            new_slug = f"{kp_slug}-{base_slug}" if base_slug else kp_slug
            return new_slug[:80].strip("-")
        return base_slug[:80].strip("-")

    def _generate_meta_description(self, content: str, keyphrase: str, title: str) -> str:
        # Prefer user template if provided
        prompt = self._render_prompt(
            "meta_gen",
            KEYPHRASE=keyphrase,
            MIN_META=self.MIN_META_LENGTH,
            MAX_META=self.MAX_META_LENGTH,
            TITLE=title,
            CONTENT_SNIPPET=(content or "")[:400],
            CONTENT=content,
        )
        try:
            md = self._chat(prompt, model=self.LIGHT_MODEL, max_tokens=self.MAX_META_LENGTH, temperature=0.5, tag="meta_gen")
        except Exception:
            md = ""

        if md:
            if keyphrase and keyphrase.lower() not in md.lower():
                candidate = f"{keyphrase}: {md}"
                md = candidate if len(candidate) <= self.MAX_META_LENGTH else (md + f" - {keyphrase}")[:self.MAX_META_LENGTH]
            if len(md) > self.MAX_META_LENGTH:
                md = md[: self.MAX_META_LENGTH - 3].strip() + "..."
            if len(md) < self.MIN_META_LENGTH:
                pad = " Learn more inside."
                md = (md + pad)[: self.MAX_META_LENGTH]
            return md.strip()

        # fallback
        first_para = (content or "").split("\n\n")[0].strip()
        if first_para:
            fb = first_para if (keyphrase and keyphrase.lower() in first_para.lower()) else f"{keyphrase}: {first_para}"
            return (fb[: self.MAX_META_LENGTH - 3] + "...") if len(fb) > self.MAX_META_LENGTH else fb
        generic = f"Learn all about {keyphrase}. A concise, practical guide with tips and examples."
        return generic[: self.MAX_META_LENGTH]

    def _generate_image_alt_texts(self, content: str, keyphrase: str) -> List[str]:
        alts = [m[0].strip() for m in re.findall(r"!\[(.*?)\]\((.*?)\)", content or "")]
        alts = [a for a in alts if a]
        if keyphrase and not any(keyphrase.lower() in a.lower() for a in alts):
            alts.append(f"{keyphrase} – concepts and examples")
        seen, uniq = set(), []
        for a in alts:
            a = " ".join(a.split())
            if len(a.split()) > 15:
                a = " ".join(a.split()[:15]) + "..."
            if a.lower() not in seen:
                uniq.append(a); seen.add(a.lower())
        return uniq[:8]

    def _extract_keywords(self, content: str) -> str:
        words = self._word_tokenizer_regex.findall((content or "").lower())
        stop = {
            "a","an","the","and","or","but","is","are","was","were","in","on","at","for","with","as","by","of","to",
            "from","it","its","he","she","they","we","you","i","me","him","her","us","them","my","your","his","our",
            "their","this","that","these","those","can","will","would","should","could","has","have","had","do","does",
            "did","be","being","been","not","no","don","t","s","m","ll","ve","re","d","about","above","after","again",
            "against","all","am","any","aren","as","because","before","below","between","both","cannot","couldn",
            "didn","doing","down","during","each","few","further","hadn","hasn","haven","having","here","hers","herself",
            "himself","how","if","into","isn","itself","just","let","more","most","mustn","myself","nor","off","once",
            "only","other","ought","ours","ourselves","out","over","own","same","shan","she","shouldn","so","some",
            "such","than","their","theirs","themselves","then","there","these","they","ve","were","weren","what","when",
            "where","which","while","who","whom","why","won","wouldn","you","yours","yourself","yourselves",
        }
        counts = Counter(w for w in words if w not in stop and len(w) >= 3)
        for kw in self._word_tokenizer_regex.findall((self.current_keyphrase or "").lower()):
            counts[kw] = counts.get(kw, 0) + 10
        top = [w for w, _ in counts.most_common(8)]
        return ", ".join(top)

    # --- Structure helpers ----------------------------------------------------
    def _ensure_content_length(self, content: str, keyphrase: str) -> str:
        word_count = len(self._word_tokenizer_regex.findall(content))
        if word_count >= self.MIN_CONTENT_WORDS and self.WORD_TARGET_LOW <= word_count <= self.WORD_TARGET_HIGH:
            return content
        mt = min(max(int((self.WORD_TARGET_HIGH - word_count) * 1.2) + 300, 300), 1200)
        prompt = self._render_prompt(
            "length_expand",
            target_low=self.WORD_TARGET_LOW,
            target_high=self.WORD_TARGET_HIGH,
            kp_density=self._kp_density(content, keyphrase),
            keyphrase=keyphrase,
            content=content,
        )
        try:
            expanded = self._chat(prompt, model=self.HEAVY_MODEL, max_tokens=mt, temperature=0.6, tag="length_expand")
            if expanded:
                return expanded
        except Exception as e:
            logger.warning(f"Length expansion failed: {e}")
        return content

    def _normalize_outbound_candidate(self, s: str) -> Optional[str]:
        s = (s or "").strip()
        if not s:
            return None
        md = re.findall(r"\((https?://[^)]+)\)", s)
        if md:
            return md[0]
        if re.match(r"^https?://", s):
            return s
        return None

    def _ensure_content_structure(self, content: str, keyphrase: str, niche: str) -> str:
        modified = content

        # 1) Ensure at least one image with alt text
        if not re.search(r"!\[.*?\]\(https?://[^\s)]+\)", modified):
            alt = f"{keyphrase} illustration" if keyphrase else "illustration"
            img_block = f'\n\n![{alt}]({self.default_image_url})\n'
            parts = modified.split('\n\n')
            if len(parts) > 1:
                parts.insert(1, img_block)
                modified = "\n\n".join(parts)
            else:
                modified = modified.rstrip() + img_block
            logger.info("Added placeholder image with alt text.")

        # 2) Outbound links from mapping
        have_outbound = self._count_outbound_links(modified)
        need_outbound = self.MIN_OUTBOUND_LINKS - have_outbound
        if need_outbound > 0:
            outbound_map = self.app_config_settings.get("outbound_links_map", {})
            niche_lower = (niche or "").strip().lower()
            candidates: List[str] = []
            if niche_lower in outbound_map:
                v = outbound_map[niche_lower]
                candidates.extend(v if isinstance(v, list) else [v])
            if "general_python" in outbound_map:
                v = outbound_map["general_python"]
                candidates.extend(v if isinstance(v, list) else [v])
            urls = []
            for c in candidates:
                u = self._normalize_outbound_candidate(c)
                if u:
                    urls.append(u)
            site_norm = self.site_domain.replace("https://", "").replace("http://", "").strip("/")
            urls = [u for u in urls if site_norm not in u and u not in modified and f"({u})" not in modified]
            random.shuffle(urls)
            to_add = urls[:max(0, need_outbound)]
            if to_add:
                links_md = [f"[More on {u.split('//')[-1].split('/')[0].replace('www.','')}]({u})" for u in to_add]
                modified = modified.rstrip() + "\n\n" + "For more details: " + " | ".join(links_md) + "\n"

        # 3) Internal link
        site_norm = self.site_domain.rstrip("/")
        have_internal = len(re.findall(re.escape(site_norm) + r"/[^\s)\"]+", modified))
        need_internal = self.MIN_INTERNAL_LINKS - have_internal
        if need_internal > 0:
            internal_link = f"{site_norm}/tag/{slugify(keyphrase)}" if keyphrase else f"{site_norm}/blog"
            anchor = f"Explore more about {keyphrase}" if keyphrase else "Explore more articles"
            if internal_link not in modified and f"]({internal_link})" not in modified:
                modified = modified.rstrip() + f"\n\nFurther reading: [{anchor}]({internal_link})\n"

        return modified

    def _ensure_subheadings_distribution_and_keyphrase(self, content: str, keyphrase: str) -> str:
        h2_matches = re.findall(r"^##\s+(.+)$", content, flags=re.MULTILINE)
        need_h2 = max(0, self.MIN_SUBHEADINGS - len(h2_matches))

        h_tags = re.findall(r"^(#{2,3})\s+(.*)$", content, flags=re.MULTILINE)
        total_h = sum(1 for tag, _ in h_tags if tag in {"##", "###"})
        with_kp = sum(1 for tag, txt in h_tags if (tag in {"##", "###"} and keyphrase and keyphrase.lower() in txt.lower()))
        target_kp = min(3, total_h) if total_h >= 1 else 1

        if need_h2 == 0 and with_kp >= target_kp:
            return content

        words = len(self._word_tokenizer_regex.findall(content))
        mt = min(max(int(words * 0.25) + 200, 300), 900)
        prompt = self._render_prompt(
            "subheads_keyphrase",
            keyphrase=keyphrase,
            need_h2=need_h2,
            target_kp=target_kp,
            kp_density=self._kp_density(content, keyphrase),
            content=content,
        )
        try:
            updated = self._chat(prompt, model=self.HEAVY_MODEL, max_tokens=mt, temperature=0.6, tag="subheads_keyphrase")
            return updated or content
        except Exception as e:
            logger.warning(f"Subheading/keyphrase pass failed: {e}")
            return content

    # -------------------- Finalizer: guard + light polish ---------------------
    def _should_run_finalizer(self, pre_seo: float, pre_read: float) -> bool:
        if not self.FINALIZER_ENABLED:
            logger.info("Finalizer disabled in settings; skipping.")
            return False
        if self.FINALIZER_ONLY_WHEN_CLOSE:
            seo_target = float(self.TARGET_SEO_SCORE)
            read_target = float(self.TARGET_READABILITY_SCORE)
            margin = float(self.FINALIZER_MARGIN)
            close_to_seo = pre_seo >= (seo_target - margin)
            close_to_read = pre_read >= (read_target - margin)
            if not (close_to_seo and close_to_read):
                logger.info(
                    "Skipping finalizer — scores not within %.0f points of targets "
                    "(SEO %.1f/%.1f, Readability %.1f/%.1f).",
                    margin, pre_seo, seo_target, pre_read, read_target
                )
                return False
        return True

    def _run_finalizer_pass(
        self,
        *,
        content: str,
        title: str,
        slug: str,
        meta_desc: str,
        keyphrase: str,
        niche: str
    ) -> Tuple[str, str]:
        prompt = (
            "Lightly polish the following article for tone and micro-clarity ONLY:\n"
            "• Do not change structure or headings.\n"
            "• Do not add or remove links, images, or code.\n"
            "• Keep H1/H2/H3 text exactly; small punctuation/typo fixes allowed.\n"
            "• Prefer active voice; keep sentences tight.\n"
            "Return ONLY the polished article text.\n\n"
            f"{content}"
        )
        try:
            polished = self._chat(
                prompt, model=self.LIGHT_MODEL, max_tokens=min(600, max(300, int(len(content) * 0.15))),
                temperature=0.3, tag="finalizer_polish"
            )
            polished = self._validate_and_clean_raw_text(polished or content)
        except Exception as e:
            logger.info(f"Finalizer polish failed ({e}); keeping pre-finalizer content.")
            polished = content

        refreshed_meta = self._generate_meta_description(polished, keyphrase, title)
        return polished, refreshed_meta

    # --- Main entry point -----------------------------------------------------
    def enhance_content(
        self,
        title: str,
        slug: str,
        meta_desc: str,
        content: str,
        keyphrase: str,
        niche: str,
    ) -> SEOEnhancementResult:
        self.current_keyphrase = keyphrase

        # sanitize inputs
        title = (title or "").strip()
        slug = (slug or "").strip()
        meta_desc = (meta_desc or "").strip()
        content = (content or "").strip()

        # Fallbacks
        if not content:
            try:
                seed = self._chat(
                    f"Write a single engaging opening paragraph for an article about '{keyphrase}'. No headers. Only the paragraph.",
                    model=self.LIGHT_MODEL, max_tokens=220, temperature=0.7, tag="seed_intro"
                )
            except Exception:
                seed = ""
            content = seed or f"This article explores {keyphrase} in depth."
            logger.warning("Initial content was empty—seed intro created.")

        if not title:
            title = f"Mastering {keyphrase}" if keyphrase else "A Comprehensive Guide"
            logger.warning(f"Initial title empty—set to: '{title}'")

        if not meta_desc:
            meta_desc = self._generate_meta_description(content, keyphrase, title)

        # Ensure early keyphrase mention (Yoast)
        content = self._ensure_early_keyphrase_mention(content, keyphrase)

        # Initial scoring
        current = {"title": title, "slug": slug, "meta_description": meta_desc, "content": content, "keyphrase": keyphrase}
        seo_score, seo_warnings = self._calculate_seo_score(current)
        readability_score, readability_warnings = self._calculate_readability_score(content)
        all_warn = seo_warnings + readability_warnings
        logger.info(f"Initial → SEO: {seo_score}, Readability: {readability_score}, Warnings: {len(all_warn)}")

        # Track best state for patience-based early stop
        best = {
            "seo": seo_score,
            "read": readability_score,
            "warns": len(all_warn),
            "title": title,
            "slug": slug,
            "meta": meta_desc,
            "content": content,
        }
        no_gain = 0
        max_iter = int(self.MAX_ENHANCEMENT_ITERATIONS)

        for i in range(1, max_iter + 1):
            logger.info(f"--- Enhancement Iteration {i}/{max_iter} ---")
            # Snapshot BEFORE edits for guardrail rollback
            prev_snapshot = {
                "title": title, "slug": slug, "meta": meta_desc, "content": content,
                "seo": seo_score, "read": readability_score, "warns": len(all_warn)
            }

            # 1) Single global readability pass (1 call)
            content = self._global_readability_rewrite(content, keyphrase)

            # 2) Expand length if needed (1 call worst case)
            content = self._ensure_content_length(content, keyphrase)

            # 3) Subheadings + ensure KP in some H2/H3 (1 call worst case)
            content = self._ensure_subheadings_distribution_and_keyphrase(content, keyphrase)

            # 4) Structure (deterministic)
            content = self._ensure_content_structure(content, keyphrase, niche)

            # 5) Title/slug/meta refresh (mini model; cached)
            title = self._optimize_title(title, keyphrase)
            slug = self._enforce_slug_keyphrase(slug, keyphrase)
            meta_desc = self._generate_meta_description(content, keyphrase, title)

            # 6) Keywords & image alts (computed after content changes)
            image_alt_texts = self._generate_image_alt_texts(content, keyphrase)
            keywords = self._extract_keywords(content)

            # Re-score
            current.update({"title": title, "slug": slug, "meta_description": meta_desc, "content": content})
            new_seo, seo_warnings = self._calculate_seo_score(current)
            new_read, readability_warnings = self._calculate_readability_score(content)
            all_warn = seo_warnings + readability_warnings
            logger.info(f"Iter {i} → SEO: {new_seo}, Readability: {new_read}, Warnings: {len(all_warn)}")

            # --- Guardrail: revert iteration if SEO drops too much --------------
            if new_seo < (prev_snapshot["seo"] - self.GUARDRAIL_ALLOWED_SEO_DROP):
                logger.info(
                    "Guardrail triggered: SEO dropped from %d to %d (> %d). Reverting iteration.",
                    prev_snapshot["seo"], new_seo, self.GUARDRAIL_ALLOWED_SEO_DROP
                )
                title, slug, meta_desc, content = (
                    prev_snapshot["title"], prev_snapshot["slug"], prev_snapshot["meta"], prev_snapshot["content"]
                )
                # keep prior scores
                new_seo, new_read = prev_snapshot["seo"], prev_snapshot["read"]
                all_warn = []
                no_gain += 1
                if no_gain >= int(self.EARLY_STOP_PATIENCE):
                    logger.info("Early-stop after guardrail reversion—stopping.")
                    break
                continue

            # Update current scores
            seo_score, readability_score = new_seo, new_read

            # --- 'Good enough' early stopping ----------------------------------
            if (
                seo_score >= self.GOOD_ENOUGH_SEO
                and readability_score >= self.GOOD_ENOUGH_READ
                and len(all_warn) <= self.GOOD_ENOUGH_MAX_WARNINGS
            ):
                logger.info("Good-enough thresholds met—stopping early.")
                # also update best before break if improved
                if (seo_score, readability_score, -len(all_warn)) > (best["seo"], best["read"], -best["warns"]):
                    best.update({
                        "seo": seo_score, "read": readability_score, "warns": len(all_warn),
                        "title": title, "slug": slug, "meta": meta_desc, "content": content
                    })
                break

            # Check improvement vs best
            cur_tuple = (seo_score, readability_score, -len(all_warn))
            best_tuple = (best["seo"], best["read"], -best["warns"])
            if cur_tuple > best_tuple:
                best.update({
                    "seo": seo_score, "read": readability_score, "warns": len(all_warn),
                    "title": title, "slug": slug, "meta": meta_desc, "content": content
                })
                no_gain = 0
            else:
                no_gain += 1
                logger.info(f"No measurable gain this round (patience {no_gain}/{self.EARLY_STOP_PATIENCE}).")

            # stop conditions at targets
            if (
                seo_score >= self.TARGET_SEO_SCORE
                and readability_score >= self.TARGET_READABILITY_SCORE
                and len(all_warn) <= self.MAX_WARNINGS
            ):
                logger.info("Targets achieved—stopping.")
                break

            if content == prev_snapshot["content"] and i >= 2:
                simplified = self._baseline_simplify(content)
                if simplified != content:
                    content = simplified
                    logger.info("Applied deterministic simplifier.")
                    continue

            if no_gain >= int(self.EARLY_STOP_PATIENCE):
                logger.info("Early-stop patience reached—stopping.")
                break

        # Restore best-known state
        content = best["content"]
        title = best["title"]
        slug = best["slug"]
        meta_desc = best["meta"]
        seo_score = best["seo"]
        readability_score = best["read"]

        # ---------------------- Finalizer gate + safety -----------------------
        pre_seo, pre_read = float(seo_score), float(readability_score)
        if self._should_run_finalizer(pre_seo, pre_read):
            polished_content, refreshed_meta = self._run_finalizer_pass(
                content=content, title=title, slug=slug, meta_desc=meta_desc, keyphrase=keyphrase, niche=niche
            )
            # Re-score after finalizer
            cur = {"title": title, "slug": slug, "meta_description": refreshed_meta, "content": polished_content, "keyphrase": keyphrase}
            post_seo, post_warn_seo = self._calculate_seo_score(cur)
            post_read, post_warn_read = self._calculate_readability_score(polished_content)

            # Regression tolerance
            seo_ok = (post_seo + self.FINALIZER_ALLOW_SEO_REGRESS) >= pre_seo
            read_ok = (post_read + self.FINALIZER_ALLOW_READ_REGRESS) >= pre_read

            if not (seo_ok and read_ok):
                logger.info(
                    "Finalizer worsened metrics; reverting to pre-finalizer content "
                    "(SEO %.1f→%.1f, Readability %.1f→%.1f).",
                    pre_seo, float(post_seo), pre_read, float(post_read)
                )
                # keep pre-finalizer content/metrics
            else:
                # accept finalized
                content = polished_content
                meta_desc = refreshed_meta
                seo_score = int(post_seo)
                readability_score = int(post_read)
                all_warn = post_warn_seo + post_warn_read
        else:
            logger.info("Finalizer skipped.")

        # Final schema (use the finalized slug)
        schema_markup = self._generate_article_schema(
            title, meta_desc, keyphrase, content, slug=slug,
            author_name=self.app_config_settings.get("authors", {}).get("name", "PythonProHub Team"),
        )

        result = SEOEnhancementResult(
            content=content,
            meta_description=meta_desc,
            meta=meta_desc,
            slug=slug,
            keyphrase=keyphrase,
            keywords=self._extract_keywords(content),
            image_alt_texts=self._generate_image_alt_texts(content, keyphrase),
            yoast_seo_score=seo_score,
            yoast_readability_score=readability_score,
            warnings=all_warn,
            schema_markup=schema_markup,
            passed_all_checks=(
                seo_score >= self.TARGET_SEO_SCORE
                and readability_score >= self.TARGET_READABILITY_SCORE
                and len(all_warn) <= self.MAX_WARNINGS
            ),
            title=title,
        )

        logger.info(f"Total tokens this run: {self.token_usage}")
        return result

    # --- Schema ---------------------------------------------------------------
    def _generate_article_schema(
        self, title: str, meta_description: str, keyphrase: str, content: str, *, slug: str, author_name: str = "PythonProHub Team"
    ) -> Dict[str, Any]:
        logger.info("Generating Article Schema (JSON-LD).")
        now_iso = datetime.now().isoformat()
        first_image_url_match = re.search(r"!\[(.*?)\]\((.*?)\)", content or "")
        image_url = first_image_url_match.group(2) if first_image_url_match else self.default_image_url
        image_width = 1200
        image_height = 675
        short_desc = meta_description
        if len(short_desc) < 50 and len(content) > 100:
            first_paragraph = (content or "").split("\n\n")[0].strip()
            short_desc = (first_paragraph[:150].strip() + "...") if len(first_paragraph) > 150 else first_paragraph or (f"Guide about {keyphrase}." if keyphrase else "Guide.")

        site_root = self.site_domain.rstrip("/")
        return {
            "@context": "https://schema.org",
            "@type": "Article",
            "headline": title,
            "description": short_desc,
            "image": {"@type": "ImageObject", "url": image_url, "width": image_width, "height": image_height},
            "datePublished": now_iso,
            "dateModified": now_iso,
            "author": {"@type": "Organization", "name": author_name},
            "publisher": {
                "@type": "Organization",
                "name": self.publisher_name,
                "logo": {
                    "@type": "ImageObject",
                    "url": self.publisher_logo_url,
                    "width": 600,
                    "height": 60,
                },
            },
            "mainEntityOfPage": {"@type": "WebPage", "@id": f"{site_root}/{slug}"},
            "keywords": keyphrase,
        }


# ----------------------------- CLI / Demo -------------------------------------
if __name__ == "__main__":
    load_dotenv()

    if "OPENAI_API_KEY" not in os.environ or not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY is not set. Please set it in your environment or a .env file.")
        raise SystemExit(1)

    # Example dummy settings (adjust to your settings.json in pipeline)
    dummy_settings = {
        "api_keys": {"openai": os.getenv("OPENAI_API_KEY")},
        "site": {"base_url": os.getenv("SITE_DOMAIN", DEFAULT_SITE_DOMAIN)},
        "publisher": {
            "name": "PythonProHub",
            "logo_url": f"{(os.getenv('SITE_DOMAIN', DEFAULT_SITE_DOMAIN)).rstrip('/')}/wp-content/uploads/logo.png"
        },
        "images": {"default_article": DEFAULT_IMAGE_URL},
        "models": {"heavy": "gpt-4o", "light": "gpt-4o-mini"},
        "article_generation": {"word_count_target": [1300, 1600]},
        "yoast_compliance": {
            "min_content_words": 1100,
            "max_title_length": 60,
            "min_title_length": 40,
            "max_meta_length": 156,
            "min_meta_length": 120,
            "min_subheadings": 3,
            "min_outbound_links": 2,
            "min_internal_links": 1,
            "min_keyphrase_density": 0.5,
            "max_keyphrase_density": 2.5,
            "flesch_reading_ease_good": 60,
            "passive_voice_max_percent": 5,
            "min_transition_word_percent": 30,
            "max_long_sentence_percent": 20,
            "max_paragraph_words": 160,
            "max_consecutive_same_start": 0,
            "target_seo_score": 95,
            "target_readability_score": 95,
            "max_warnings": 1,
            "max_enhancement_iterations": 4,
            "early_stop_patience": 2,
            "good_enough_seo": 80,
            "good_enough_read": 50,
            "good_enough_max_warnings": 5,
            "guardrail_allowed_seo_drop": 5,
        },
        "finalizer": {
            "enabled": True,
            "run_if_within_points": 5,
            "allow_seo_regression_points": 0,
            "allow_read_regression_points": 0,
            "only_when_close": True
        },
        "outbound_links_map": {
            "general_python": [
                "https://www.python.org/doc/",
                "https://pypi.org/",
                "https://docs.python.org/3/tutorial/"
            ]
        },
        "authors": {"name": "PythonProHub Team"},
        # Include custom prompts to demonstrate wiring
        "enhancement_prompts": {
            "readability_global": "Improve clarity and flow. Keep meaning intact. DO NOT remove or add headings. Keep EXACT keyphrase '{KEYPHRASE}' density at {TARGET_DENSITY:.2f}%. Keep Flesch Reading Ease >= {FLESCH_MIN}. Keep sentences mostly <= 22 words. Return ONLY the improved Markdown.\n\n{CONTENT}",
            "length_expand": "Expand content to a total of {MIN_WORDS}-{MAX_WORDS} words. Preserve all existing headings and code blocks. Enrich with concrete, Python-focused examples. Keep EXACT keyphrase '{KEYPHRASE}' density at {TARGET_DENSITY:.2f}%. Avoid fluff. Return ONLY the full Markdown.\n\n{CONTENT}",
            "subheads_keyphrase": "Revise H2/H3 subheadings to be specific and to include the exact keyphrase '{KEYPHRASE}' in at least one H2. Do not change the article structure count. Return ONLY Markdown.\n\n{CONTENT}",
            "title_opt": "Suggest a concise SEO title (max {MAX_TITLE} chars, min {MIN_TITLE} chars) that includes the exact keyphrase '{KEYPHRASE}'. Return ONLY the title text.\n\n{TITLE}",
            "meta_gen": "Write a meta description ({MIN_META}-{MAX_META} chars) that naturally includes '{KEYPHRASE}'. No quotes or emojis. Return ONLY the meta text.\n\n{CONTENT_SNIPPET}"
        }
    }

    main_openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    seo = SEOEnforcer(dummy_settings, openai_client=main_openai_client)

    test_title = "Building a Loan Default Prediction Model"
    test_slug = slugify(test_title)
    test_meta_desc = ""
    test_keyphrase = "Loan Default Prediction"
    test_niche = "machine learning"
    test_content = """
Loan default prediction is a critical task in finance, enabling institutions to assess credit risk accurately.
By leveraging machine learning models, lenders can make more informed decisions, minimizing potential losses.
The process involves analyzing various financial and personal data points to determine the likelihood of a borrower defaulting on their loan.

Understanding the factors contributing to loan default is paramount. These can range from an individual's credit history and income stability to broader economic indicators.
Machine learning algorithms excel at identifying complex patterns within these datasets, providing insights that traditional statistical methods might miss.

Several models can be employed for this purpose, including logistic regression, decision trees, and more advanced neural networks.
Each model has its strengths and weaknesses, and the choice often depends on the specific dataset characteristics and the desired interpretability of the model.
""".strip()

    print("\n--- Starting SEO Enhancement Process ---")
    result = seo.enhance_content(
        title=test_title,
        slug=test_slug,
        meta_desc=test_meta_desc,
        content=test_content,
        keyphrase=test_keyphrase,
        niche=test_niche,
    )

    print("\n--- Final SEO Enhancement Result ---")
    print(f"Title: {result.title}")
    print(f"Slug: {result.slug}")
    print(f"Meta Description: {result.meta_description}")
    print(f"Final SEO Score: {result.yoast_seo_score}")
    print(f"Final Readability Score: {result.yoast_readability_score}")
    print(f"Passed All Checks: {result.passed_all_checks}")
    print(f"Keywords: {result.keywords}")
    print(f"Image Alt Texts: {result.image_alt_texts}")
    print(f"Warnings: {result.warnings[:10]}{'...' if len(result.warnings) > 10 else ''}")
    print(f"\n--- Enhanced Content Snippet (first 800 chars) ---\n{result.content[:800]}...")
    print(f"\n--- Schema Markup ---\n{json.dumps(result.schema_markup, indent=2)}")
