# live_demo_embedder.py

import hashlib
import logging
import os
import re
import textwrap
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import quote_plus, urlencode, urlparse, urlunparse, parse_qsl

# Ensure logs dir exists before adding FileHandler
Path("logs").mkdir(parents=True, exist_ok=True)

# Configure logging (avoid double-adding handlers if root already configured)
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("logs/demo_embedder.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
logger = logging.getLogger(__name__)


class DemoPlatform(Enum):
    REPLIT = "replit"
    COLAB = "colab"
    JUPYTER = "jupyter"
    VSCODE = "vscode"


@dataclass
class DemoEmbedderConfig:
    # Core behavior
    enabled: bool = True
    default_platform: str = "replit"
    max_embeds: int = 3
    position: str = "after"  # "before" or "after" the code block

    # What to show
    template: str = "\n[🔗 Try this code on {platform}]({url})\n"
    caption_template: Optional[str] = None  # e.g., "\n<sub>Opens in a new tab</sub>\n"

    # Filtering / safety / UX
    allowed_langs: List[str] = field(default_factory=lambda: ["python", "py"])
    min_code_chars: int = 16
    max_code_lines: int = 120
    max_code_chars: int = 4000
    url_max_length: int = 1900
    strip_prompts: bool = True
    dedent: bool = True
    # Optionally shrink code for URL by removing full-line comments & extra blank lines
    compress_for_url: bool = False
    skip_if_contains: List[str] = field(default_factory=lambda: [
        "input(", "getpass.getpass(", "tkinter", "subprocess", "os.remove(", "os.system(", "shutil.rmtree(",
    ])
    # Extra safety: skip if code *looks* like it contains tokens/secrets
    skip_if_regexes: List[str] = field(default_factory=lambda: [
        r"(?i)aws[_-]?secret[_-]?access[_-]?key\s*=",
        r"(?i)\bghp_[A-Za-z0-9]{30,}",
        r"(?i)\bsk_(live|test)_[A-Za-z0-9]{16,}",
    ])

    # URLs
    platform_urls: Dict[str, str] = field(
        default_factory=lambda: {
            # NOTE: These endpoints accept a `code` param; long code may hit URL limits.
            "replit": "https://replit.com/new/python3?code={code}",
            "colab": "https://colab.research.google.com/#create=true&language=python&code={code}",
            "jupyter": "https://jupyter.org/try?code={code}",
            "vscode": "vscode://ms-python.python/playground?code={code}",
        }
    )

    # Tracking
    utm_params: Dict[str, str] = field(default_factory=lambda: {
        "utm_source": "pythonprohub",
        "utm_medium": "live-demo",
        "utm_campaign": "article",
    })

    # Duplicate/marker control
    unique_marker_prefix: str = "<!-- live-demo:"
    unique_marker_suffix: str = " -->"
    # How many characters around a block to scan for our marker (idempotency)
    scan_context_chars: int = 320


class LiveDemoEmbedder:
    """
    Scans markdown for fenced Python code blocks and appends (or prepends)
    a 'Run this code' link for a chosen platform. Keeps state to limit
    how many embeds are added per article.

    Features:
      • Supports ``` and ~~~ fences and info strings like "python linenums".
      • Heuristic Python detection for unlabeled blocks.
      • Per-block override keywords: "replit|colab|jupyter|vscode" or "no-demo".
      • Env toggles: LIVE_DEMO_ENABLED, LIVE_DEMO_PLATFORM, LIVE_DEMO_MAX_EMBEDS.
      • Idempotent across re-runs via hidden markers scanned in surrounding text.
      • Safer URL limits and optional compression for long snippets.
    """

    def __init__(self, app_config_settings: Dict):
        self.config = DemoEmbedderConfig()
        demo_settings = (app_config_settings or {}).get("live_demos", {})

        # Merge simple fields
        self.config.default_platform = demo_settings.get("default_platform", self.config.default_platform)
        self.config.max_embeds = int(demo_settings.get("max_embeds", self.config.max_embeds))
        self.config.position = demo_settings.get("position", self.config.position)
        self.config.template = demo_settings.get("template", self.config.template)
        self.config.caption_template = demo_settings.get("caption_template", self.config.caption_template)
        self.config.min_code_chars = int(demo_settings.get("min_code_chars", self.config.min_code_chars))
        self.config.max_code_lines = int(demo_settings.get("max_code_lines", self.config.max_code_lines))
        self.config.max_code_chars = int(demo_settings.get("max_code_chars", self.config.max_code_chars))
        self.config.url_max_length = int(demo_settings.get("url_max_length", self.config.url_max_length))
        self.config.strip_prompts = bool(demo_settings.get("strip_prompts", self.config.strip_prompts))
        self.config.dedent = bool(demo_settings.get("dedent", self.config.dedent))
        self.config.compress_for_url = bool(demo_settings.get("compress_for_url", self.config.compress_for_url))
        self.config.unique_marker_prefix = demo_settings.get("unique_marker_prefix", self.config.unique_marker_prefix)
        self.config.unique_marker_suffix = demo_settings.get("unique_marker_suffix", self.config.unique_marker_suffix)
        self.config.scan_context_chars = int(demo_settings.get("scan_context_chars", self.config.scan_context_chars))

        # Merge lists/dicts
        if isinstance(demo_settings.get("allowed_langs"), list):
            self.config.allowed_langs = [str(x).lower() for x in demo_settings["allowed_langs"]]
        if isinstance(demo_settings.get("skip_if_contains"), list):
            self.config.skip_if_contains = list(demo_settings["skip_if_contains"])
        if isinstance(demo_settings.get("skip_if_regexes"), list):
            self.config.skip_if_regexes = list(demo_settings["skip_if_regexes"])
        if isinstance(demo_settings.get("platform_urls"), dict):
            self.config.platform_urls.update(demo_settings["platform_urls"])
        if isinstance(demo_settings.get("utm_params"), dict):
            self.config.utm_params.update(demo_settings["utm_params"])

        # Env overrides
        env_platform = os.getenv("LIVE_DEMO_PLATFORM")
        if env_platform:
            self.config.default_platform = env_platform.strip().lower()

        env_enabled = os.getenv("LIVE_DEMO_ENABLED")
        if isinstance(env_enabled, str) and env_enabled.lower() in {"0", "false", "off"}:
            self.config.enabled = False

        env_max = os.getenv("LIVE_DEMO_MAX_EMBEDS")
        if env_max and env_max.isdigit():
            self.config.max_embeds = int(env_max)

        self.embed_count = 0

        # Sanity: clamp position
        if self.config.position not in ("before", "after"):
            logger.warning("Invalid 'position' in live_demos config; defaulting to 'after'.")
            self.config.position = "after"

    # ---------------- Internal helpers ----------------

    def _prepare_code(self, code: str) -> str:
        """Normalize code for embedding."""
        c = (code or "").replace("\r\n", "\n").strip()
        if self.config.strip_prompts:
            # remove doctest prompts and leading ">>> " / "... "
            c = re.sub(r"^\s*(>>>|\.\.\.)\s?", "", c, flags=re.MULTILINE)
        if self.config.dedent:
            c = textwrap.dedent(c)
        return c.strip()

    def _compress_for_url_if_enabled(self, code: str) -> str:
        """Optionally remove full-line comments and collapse multiple blank lines."""
        if not self.config.compress_for_url:
            return code
        lines = []
        blank = 0
        for ln in code.splitlines():
            if ln.strip().startswith("#"):
                continue
            if not ln.strip():
                blank += 1
                if blank > 1:
                    continue
            else:
                blank = 0
            lines.append(ln.rstrip())
        out = "\n".join(lines).strip()
        # Avoid nuking docstrings or semantics; we keep them intact.
        return out

    def _hash_code(self, code: str) -> str:
        return hashlib.sha1(code.encode("utf-8")).hexdigest()[:10]

    def _marker_for_hash(self, h: str) -> str:
        return f"{self.config.unique_marker_prefix}{h}{self.config.unique_marker_suffix}"

    def _already_has_marker_nearby(self, surrounding_text: str, h: str) -> bool:
        marker = self._marker_for_hash(h)
        return marker in surrounding_text

    def _prepare_code_for_url(self, code: str) -> str:
        """Prepare code for URL embedding using robust encoding."""
        return quote_plus(code)

    def _append_utm(self, url: str, extra: Optional[Dict[str, str]] = None) -> str:
        """Append UTM params and optional extras, preserving existing query."""
        parsed = urlparse(url)
        q = dict(parse_qsl(parsed.query, keep_blank_values=True))
        q.update(self.config.utm_params)
        if extra:
            q.update({k: v for k, v in extra.items() if v})
        new_q = urlencode(q, doseq=True)
        return urlunparse(parsed._replace(query=new_q))

    def _generate_demo_link(self, code_content: str, platform_override: Optional[str] = None) -> str:
        """Generate platform-specific demo link for a given code string."""
        if self.embed_count >= self.config.max_embeds:
            return ""
        if not self.config.enabled:
            return ""

        platform_key = (platform_override or self.config.default_platform or "replit").strip().lower()
        try:
            platform_enum = DemoPlatform[platform_key.upper()]
            platform_name = platform_enum.value
        except KeyError:
            logger.warning(f"Invalid platform '{platform_key}' specified; skipping embed.")
            return ""

        url_template = self.config.platform_urls.get(platform_name, "")
        if not url_template:
            logger.warning(f"Unsupported or missing URL template for platform: {platform_name}")
            return ""

        # Some platforms require client-side tooling—warn but still proceed.
        if platform_name in {"jupyter", "vscode"}:
            logger.warning(f"The selected platform '{platform_name}' may require client-side setup.")

        try:
            encoded_code = self._prepare_code_for_url(code_content)
            demo_url = url_template.format(code=encoded_code)

            # Add UTM + per-snippet hash for analytics
            snippet_hash = self._hash_code(code_content)
            demo_url = self._append_utm(demo_url, {"utm_content": f"demo-{snippet_hash}"})

            # Guard: URL length
            if len(demo_url) > self.config.url_max_length:
                logger.warning(
                    f"Demo URL too long ({len(demo_url)} chars) — skipping embed. "
                    "Consider shortening the snippet or using a gist-based workflow."
                )
                return ""

            self.embed_count += 1
            link = self.config.template.format(platform=platform_name.capitalize(), url=demo_url)
            if self.config.caption_template:
                link += self.config.caption_template
            # Include an invisible marker so we don't embed twice on re-runs
            link += self._marker_for_hash(snippet_hash)
            return link
        except Exception as e:
            logger.error(f"Failed to generate demo link: {e}")
            return ""

    def _is_runnable(self, code: str) -> bool:
        """Heuristics to avoid embedding obviously non-clickable snippets."""
        if len(code) < self.config.min_code_chars:
            return False
        if code.count("\n") > self.config.max_code_lines:
            return False
        if len(code) > self.config.max_code_chars:
            return False
        lower = code.lower()
        for bad in self.config.skip_if_contains:
            if bad.lower() in lower:
                return False
        for pat in self.config.skip_if_regexes:
            try:
                if re.search(pat, code):
                    return False
            except re.error:
                # ignore invalid user-supplied patterns
                continue
        return True

    def _looks_like_python(self, code: str) -> bool:
        """Lightweight guess when language isn't labeled."""
        c = code.lower()
        hits = 0
        for needle in ("def ", "class ", "import ", "from ", "print(", "async def", "if __name__ == '__main__'"):
            if needle in c:
                hits += 1
        return hits >= 2

    def _has_existing_demo_link(self, chunk: str) -> bool:
        """Detect if a block already has our demo link nearby (inline scan)."""
        return "[🔗 Try this code on" in chunk or self.config.unique_marker_prefix in chunk

    def _parse_info_string(self, info: str) -> Dict[str, str]:
        """
        Parse the fence info string: first token is 'lang' (if alnum),
        any additional tokens can include platform keywords or 'no-demo'.
        Example: 'python replit linenums' -> {'lang':'python','platform':'replit'}
        """
        out = {"lang": "", "platform": "", "no_demo": "0"}
        tokens = [t for t in (info or "").strip().split() if t]
        if not tokens:
            return out
        # first token that's purely alnum/_ is language
        if re.match(r"^[A-Za-z0-9_]+$", tokens[0]):
            out["lang"] = tokens[0].lower()
            rest = [t.lower() for t in tokens[1:]]
        else:
            rest = [t.lower() for t in tokens]
        for t in rest:
            if t in {"replit", "colab", "jupyter", "vscode"}:
                out["platform"] = t
            if t in {"no-demo", "nodemo"}:
                out["no_demo"] = "1"
        return out

    # ---------------- Public API ----------------

    def embed_demos(self, markdown_text: str) -> str:
        """
        Embed live demo links for Python code blocks in markdown content.

        - Matches fenced code blocks (```lang ... ``` or ~~~lang ... ~~~).
        - Only adds links to blocks tagged in allowed_langs OR unlabeled blocks that look like Python.
        - Respects max_embeds, position, and per-block overrides.
        - Skips if link already present (uses invisible markers and text scan).
        """
        if not markdown_text:
            return markdown_text
        if not self.config.enabled:
            logger.info("Live demos disabled by configuration; skipping.")
            return markdown_text
        if "```" not in markdown_text and "~~~" not in markdown_text:
            return markdown_text

        self.embed_count = 0
        source_text = markdown_text  # capture for context scanning

        # Capture fenced blocks: ```info\ncode\n```  or  ~~~info\ncode\n~~~
        code_block_pattern = re.compile(
            r"(?P<fence>```|~~~)(?P<info>[^\n]*)\n(?P<code>.*?)(?P=fence)\s*",
            re.DOTALL,
        )

        def replacer(match: re.Match) -> str:
            full_block = match.group(0)
            info = (match.group("info") or "")
            code_content = match.group("code") or ""

            if self.embed_count >= self.config.max_embeds:
                return full_block

            # Build a small neighborhood around the match to detect prior markers
            start, end = match.span()
            ctx = source_text[max(0, start - self.config.scan_context_chars): min(len(source_text), end + self.config.scan_context_chars)]

            if self._has_existing_demo_link(ctx):
                return full_block

            # Parse info string
            meta = self._parse_info_string(info)
            if meta.get("no_demo") == "1":
                return full_block

            lang = meta.get("lang", "").lower()
            platform_override = meta.get("platform") or None

            prepared = self._prepare_code(code_content)
            prepared = self._compress_for_url_if_enabled(prepared)

            # Language / heuristics
            lang_ok = lang in self.config.allowed_langs if lang else False
            if not lang_ok and not lang:
                # unlabeled: guess Python
                lang_ok = self._looks_like_python(prepared)
            if not lang_ok:
                return full_block

            if not self._is_runnable(prepared):
                return full_block

            # Prevent double-embedding by scanning neighborhood for the specific hash marker
            snippet_hash = self._hash_code(prepared)
            if self._already_has_marker_nearby(ctx, snippet_hash):
                return full_block

            demo_link = self._generate_demo_link(prepared, platform_override=platform_override)
            if not demo_link:
                return full_block

            if self.config.position == "before":
                return f"{demo_link}{full_block}"
            return f"{full_block}{demo_link}"

        modified_markdown = code_block_pattern.sub(replacer, markdown_text)

        if self.embed_count:
            logger.info(f"Embedded {self.embed_count} live demo link(s).")
        else:
            logger.info("No eligible Python code blocks found for demo embedding.")
        return modified_markdown


# -------- Legacy shim (backward compatibility) --------

def embed_live_demos(markdown_text: str) -> str:
    """
    Back-compat single-function helper. Uses safe defaults.
    """
    dummy_settings = {
        "live_demos": {
            "default_platform": os.getenv("LIVE_DEMO_PLATFORM", "replit"),
            "max_embeds": int(os.getenv("LIVE_DEMO_MAX_EMBEDS", "3")),
            "position": "after",
            "template": "\n[🔗 Try this code on {platform}]({url})\n",
            "platform_urls": {
                "replit": "https://replit.com/new/python3?code={code}",
                "colab": "https://colab.research.google.com/#create=true&language=python&code={code}",
                "jupyter": "https://jupyter.org/try?code={code}",
                "vscode": "vscode://ms-python.python/playground?code={code}",
            },
            "utm_params": {
                "utm_source": "pythonprohub",
                "utm_medium": "live-demo",
                "utm_campaign": "article",
            },
        }
    }
    embedder = LiveDemoEmbedder(dummy_settings)
    # Respect LIVE_DEMO_ENABLED in legacy path too
    if os.getenv("LIVE_DEMO_ENABLED", "").lower() in {"0", "false", "off"}:
        embedder.config.enabled = False
    return embedder.embed_demos(markdown_text)


__all__ = [
    "LiveDemoEmbedder",
    "DemoEmbedderConfig",
    "DemoPlatform",
    "embed_live_demos",
]
