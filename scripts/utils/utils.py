import os
import json
import re
import unicodedata
import logging
from base64 import b64encode
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
from functools import lru_cache
from urllib.parse import urljoin, urlsplit, urlunsplit, parse_qsl, urlencode

# Optional advanced Unicode/emoji support
try:
    import regex as _uregex  # pip install regex
    _EMOJI_RE = _uregex.compile(r"\p{Extended_Pictographic}")
    _HAS_REGEX = True
except Exception:
    _uregex = None
    _EMOJI_RE = None
    _HAS_REGEX = False

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------
LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
logger = logging.getLogger("utils")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    try:
        fh = logging.FileHandler(LOG_DIR / "utils.log", encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    except Exception:
        pass
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------
_WINDOWS_RESERVED = {
    "con", "prn", "aux", "nul",
    *(f"com{i}" for i in range(1, 10)),
    *(f"lpt{i}" for i in range(1, 10)),
}
_TRACKING_KEYS = {"fbclid", "gclid", "yclid", "mc_eid", "mc_cid", "msclkid"}
_TRACKING_PREFIXES = ("utm_", "ref_", "vero_")

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")

# JSONC helpers: comments + trailing commas
_COMMENT_BLOCK_RE = re.compile(r"/\*.*?\*/", re.S)
_COMMENT_LINE_RE = re.compile(r"^\s*//.*$", re.M)
# remove a trailing comma just before } or ]
_TRAILING_COMMA_RE = re.compile(r",(\s*[}\]])")

# Zero-width characters to purge when desired
_ZERO_WIDTH_RE = re.compile(
    "[" +
    "\u200B\u200C\u200D\u200E\u200F" +  # ZWSP, ZWNJ, ZWJ, LRM, RLM
    "\u2060" +                          # WJ
    "\uFEFF" +                          # BOM
    "]"
)

# ---------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------
@dataclass
class UtilsConfig:
    base_url: str = "https://pythonprohub.com"
    max_slug_length: int = 75
    json_indent: int = 2
    json_ensure_ascii: bool = False
    # ${VAR:default} or $VAR — supports nested; '\$' escapes a literal dollar
    env_var_pattern: str = r'\$\{([^}:]+)(?::([^}]*))?\}|\$([^\W\d]\w*)'
    # Keep only characters NOT in the invalid set:
    valid_filename_pattern: str = r'[^<>:"/\\|?*\x00-\x1f]'
    # Optional: allow writes outside CWD (use with care)
    allow_outside_cwd: bool = False
    # Backup rotation for write_json(…, backup=True)
    backup_keep: int = 5

# ---------------------------------------------------------------------
# Core utilities class
# ---------------------------------------------------------------------
class EnhancedUtils:
    def __init__(self, config: Optional[UtilsConfig] = None):
        self.config = config or UtilsConfig()
        self._compiled_env_var_pattern = re.compile(self.config.env_var_pattern)
        self._compiled_filename_pattern = re.compile(self.config.valid_filename_pattern)
        self._emoji_warned_once = False

    # ----------------------------- Slugify ---------------------------------
    @staticmethod
    def _basic_replacements(text: str) -> str:
        """Small, readable substitutions before normalization."""
        if not text:
            return text
        # normalize common punctuation variants
        text = text.replace("’", "'").replace("“", '"').replace("”", '"').replace("–", "-").replace("—", "-")
        # friendly symbol words (helps slugs)
        text = (
            text.replace("&", " and ")
                .replace("@", " at ")
                .replace("+", " plus ")
                .replace("#", " number ")
        )
        return text

    @staticmethod
    @lru_cache(maxsize=4096)
    def _slugify_cached(text: str, max_length: int, keep_emojis_flag: bool) -> str:
        if not text:
            return "article"

        # Replace zero-width and small symbol words early
        text = _ZERO_WIDTH_RE.sub("", EnhancedUtils._basic_replacements(text))

        # Normalize/fold
        try:
            text = unicodedata.normalize("NFKD", text)
            if keep_emojis_flag:
                # Remove combining marks but keep emoji codepoints
                text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
            else:
                text = text.encode("ascii", "ignore").decode("ascii")
        except Exception:
            text = unicodedata.normalize("NFKC", text)

        allowed_basic = set("_-.~ ")
        out_chars: List[str] = []
        if keep_emojis_flag and _HAS_REGEX and _EMOJI_RE is not None:
            for ch in text:
                if ch.isalnum() or ch in allowed_basic or _EMOJI_RE.fullmatch(ch):
                    out_chars.append(ch)
        else:
            for ch in text:
                if ch.isalnum() or ch in allowed_basic:
                    out_chars.append(ch)

        lowered = "".join(out_chars).lower().strip()
        # collapse whitespace / separators to hyphens
        slug = re.sub(r"[\s\-\.~]+", "-", lowered).strip("-")
        # avoid leading/trailing dots (odd on some servers)
        slug = slug.strip(".")
        max_length = max(1, min(int(max_length or 1), 200))
        slug = slug[:max_length].strip("-")
        slug = re.sub(r"-{2,}", "-", slug).strip("-")

        return slug or "article"

    def slugify(self, text: str, *, keep_emojis: bool = False, max_length: Optional[int] = None) -> str:
        if keep_emojis and not _HAS_REGEX and not self._emoji_warned_once:
            self._emoji_warned_once = True
            logger.warning(
                "Emoji preservation requested, but 'regex' is not installed. Falling back to ASCII."
            )
        effective_len = int(max_length) if max_length is not None else self.config.max_slug_length
        return self._slugify_cached(text or "", effective_len, bool(keep_emojis))

    def cache_clear_slugify(self) -> None:
        """Clear slugify LRU cache (useful after changing config)."""
        try:
            self._slugify_cached.cache_clear()  # type: ignore[attr-defined]
        except Exception:
            pass

    # ----------------------------- Paths -----------------------------------
    @staticmethod
    def _is_under_cwd(path: Path) -> bool:
        try:
            path.resolve().relative_to(Path.cwd().resolve())
            return True
        except Exception:
            return False

    def ensure_directory(self, path: Union[str, Path], *, allow_outside_cwd: Optional[bool] = None) -> Path:
        try:
            resolved_path = Path(path).expanduser().resolve()
            allow = self.config.allow_outside_cwd if allow_outside_cwd is None else bool(allow_outside_cwd)
            if not allow and not self._is_under_cwd(resolved_path):
                raise ValueError(
                    f"Unsafe path: {resolved_path} is outside the working directory {Path.cwd().resolve()}."
                )
            resolved_path.mkdir(parents=True, exist_ok=True)
            return resolved_path
        except Exception as e:
            logger.error(f"Directory creation failed for path '{path}': {e}")
            raise

    @staticmethod
    def safe_join(base: Union[str, Path], *parts: str, allow_traversal: bool = False) -> Path:
        """
        Join parts to base safely. If allow_traversal=False, ensures the final
        path is inside base (prevents '../../' escapes).
        """
        basep = Path(base).expanduser().resolve()
        p = basep
        for part in parts:
            p = p.joinpath(part)
        p = p.resolve()
        if not allow_traversal:
            try:
                p.relative_to(basep)
            except Exception:
                raise ValueError(f"Unsafe join outside base: {p} not under {basep}")
        return p

    # ----------------------- Env var resolution ----------------------------
    def _coerce_env_value(self, value: str) -> Any:
        low = value.lower()
        if low == "true":
            return True
        if low == "false":
            return False
        if re.fullmatch(r"[+-]?\d+", value):
            try:
                return int(value)
            except Exception:
                pass
        if re.fullmatch(r"[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?", value):
            try:
                return float(value)
            except Exception:
                pass
        return value

    def resolve_env_vars(self, obj: Any) -> Any:
        """
        Replace $VAR or ${VAR:default} tokens recursively in strings/dicts/lists.
        Supports nested substitutions and '\$' to escape a literal dollar.
        If the whole string is a single token, coerce to a native type.
        """
        if isinstance(obj, dict):
            return {k: self.resolve_env_vars(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self.resolve_env_vars(item) for item in obj]
        if isinstance(obj, str):
            s = obj

            # Protect escaped '$' and '$$' first
            sentinel = "\uFFFFDOLLAR\uFFFF"
            s = s.replace("\\$", sentinel).replace("$$", sentinel)

            for _ in range(5):  # allow nested placeholders
                def replacer(match: re.Match) -> str:
                    var_name = match.group(1) or match.group(3)
                    default_value = match.group(2) if match.group(2) is not None else ""
                    return os.getenv(var_name, default_value)
                new_s = self._compiled_env_var_pattern.sub(replacer, s)
                if new_s == s:
                    break
                s = new_s

            # Restore literal dollars
            s = s.replace(sentinel, "$")

            # If whole string was just one placeholder, coerce to native type
            if self._compiled_env_var_pattern.fullmatch(obj):
                return self._coerce_env_value(s)
            return s
        return obj

    # ----------------------------- JSONC helpers ---------------------------
    @staticmethod
    def _strip_jsonc_text(text: str) -> str:
        """Strip BOM, /* */ and // comments & trailing commas so JSON can parse."""
        # remove BOM if present
        if text.startswith("\ufeff"):
            text = text.lstrip("\ufeff")
        # remove block comments
        text = _COMMENT_BLOCK_RE.sub("", text)
        # remove line comments
        text = _COMMENT_LINE_RE.sub("", text)
        # iteratively remove trailing commas before } or ]
        prev = None
        while prev != text:
            prev = text
            text = _TRAILING_COMMA_RE.sub(r"\1", text)
        return text

    @staticmethod
    def _json_error_excerpt(text: str, index: int, *, radius: int = 3) -> str:
        """Build a small, line-numbered excerpt with a caret at the error position."""
        lines = text.splitlines()
        # Compute line and column from absolute index
        upto = text[:index]
        line_no = upto.count("\n")
        col = len(upto) - (upto.rfind("\n") + 1 if "\n" in upto else 0)
        start = max(0, line_no - radius)
        end = min(len(lines), line_no + radius + 1)
        buf = []
        for i in range(start, end):
            ln = f"{i+1:4d}  {lines[i]}"
            buf.append(ln)
            if i == line_no:
                buf.append("      " + " " * (col) + "^")
        return "\n".join(buf)

    # ----------------------------- JSON ------------------------------------
    def load_config(self, config_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load configuration:
          1) Try strict JSON
          2) Fall back to relaxed JSONC mode (comments & trailing commas)
          3) On failure, raise with a helpful excerpt and caret
        Also resolves ${ENV} placeholders after parsing.
        """
        p = Path(config_path).resolve()
        if not p.is_file():
            raise FileNotFoundError(f"Config file not found: {p}")

        raw = p.read_text(encoding="utf-8")

        # 1) strict
        try:
            cfg: Any = json.loads(raw)
        except json.JSONDecodeError as e1:
            # 2) relaxed (JSONC)
            relaxed = self._strip_jsonc_text(raw)
            try:
                cfg = json.loads(relaxed)
                logger.warning("Config %s contained comments/trailing-commas; parsed in relaxed JSONC mode.", p)
            except json.JSONDecodeError as e2:
                excerpt = self._json_error_excerpt(raw, e2.pos)
                msg = (
                    f"Invalid JSON in config file {p}: {e2.msg} at line {e2.lineno} column {e2.colno}\n"
                    f"--- context ---\n{excerpt}\n--------------"
                )
                logger.error(msg)
                raise ValueError(msg) from e2

        if not isinstance(cfg, dict):
            raise TypeError("Config file must contain a JSON object.")
        return self.resolve_env_vars(cfg)

    def read_json(self, file_path: Union[str, Path]) -> Optional[Union[Dict, List]]:
        """
        Read a JSON/JSONC file. Returns None on failure.
        Accepts // and /* */ comments and trailing commas.
        """
        p = Path(file_path).resolve()
        try:
            text = p.read_text(encoding="utf-8")
        except FileNotFoundError:
            logger.warning(f"JSON file not found at: {file_path}")
            return None
        except Exception as e:
            logger.error(f"Failed to read JSON file {file_path}: {e}")
            return None

        # strict first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            try:
                return json.loads(self._strip_jsonc_text(text))
            except json.JSONDecodeError as e2:
                excerpt = self._json_error_excerpt(text, e2.pos)
                logger.error(
                    "Invalid JSON in %s: %s at line %d column %d\n--- context ---\n%s\n--------------",
                    file_path, e2.msg, e2.lineno, e2.colno, excerpt
                )
                return None
            except Exception as e:
                logger.error(f"Failed to decode JSONC in {file_path}: {e}")
                return None

    def _rotate_backups(self, path: Path) -> None:
        try:
            backups = sorted(path.parent.glob(path.name + ".bak.*"), key=lambda p: p.name, reverse=True)
            for old in backups[self.config.backup_keep:]:
                old.unlink(missing_ok=True)
        except Exception as e:
            logger.debug(f"Backup rotation failed for {path}: {e}")

    def write_json(
        self,
        data: Union[Dict, List],
        file_path: Union[str, Path],
        *,
        indent: Optional[int] = None,
        backup: bool = False,
        ensure_ascii: Optional[bool] = None,
        allow_nan: bool = False,  # kept for API compatibility (unused)
    ) -> bool:
        try:
            path = Path(file_path).expanduser().resolve()
            self.ensure_directory(path.parent)

            # optional timestamped backup
            if backup and path.exists():
                try:
                    ts = __import__("datetime").datetime.now().strftime("%Y%m%d%H%M%S")
                    bpath = path.with_suffix(path.suffix + f".bak.{ts}")
                    bpath.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
                    self._rotate_backups(path)
                except Exception as e:
                    logger.warning(f"Could not create backup for {path}: {e}")

            tmp = path.with_suffix(path.suffix + ".tmp")
            with tmp.open("w", encoding="utf-8") as f:
                json.dump(
                    data,
                    f,
                    indent=indent if indent is not None else self.config.json_indent,
                    ensure_ascii=self.config.json_ensure_ascii if ensure_ascii is None else bool(ensure_ascii),
                    allow_nan=False,
                )
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
            return True
        except Exception as e:
            logger.error(f"Failed to write JSON to {file_path}: {e}")
            return False

    # --------------------------- Text I/O (atomic) -------------------------
    @staticmethod
    def write_text_atomic(path: Union[str, Path], text: str, *, encoding: str = "utf-8") -> bool:
        try:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(p.suffix + ".tmp")
            with tmp.open("w", encoding=encoding) as f:
                f.write(text)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, p)
            return True
        except Exception as e:
            logger.error(f"Atomic write failed for {path}: {e}")
            return False

    @staticmethod
    def read_text(path: Union[str, Path], *, encoding: str = "utf-8") -> Optional[str]:
        try:
            return Path(path).read_text(encoding=encoding)
        except Exception as e:
            logger.error(f"Failed to read text from {path}: {e}")
            return None

    # --------------------------- Filenames ---------------------------------
    def sanitize_filename(self, filename: str, *, max_length: int = 255) -> str:
        """
        Make a safe filename while preserving the extension and avoiding
        Windows reserved names. Prevents blank/space-only names.
        """
        if not filename:
            return "untitled"
        filename = _ZERO_WIDTH_RE.sub("", unicodedata.normalize("NFKD", filename))
        # Separate extension
        p = Path(filename)
        stem, suffix = p.stem, p.suffix  # suffix includes the dot (or "")
        safe_stem = "".join(self._compiled_filename_pattern.findall(stem)).strip(". ")
        safe_ext = "".join(self._compiled_filename_pattern.findall(suffix))
        if not safe_stem:
            safe_stem = "untitled"
        # Avoid reserved device names
        if safe_stem.lower() in _WINDOWS_RESERVED:
            safe_stem = f"_{safe_stem}"
        # Recombine and clamp
        total_max = max(1, int(max_length) or 1)
        # Keep at least space for extension if present
        if safe_ext:
            stem_max = max(1, total_max - len(safe_ext))
            safe_stem = safe_stem[:stem_max]
            out = f"{safe_stem}{safe_ext}"
        else:
            out = safe_stem[:total_max]
        return out

    # --------------------------- URLs --------------------------------------
    def build_url(self, *parts: str, base_url: Optional[str] = None) -> str:
        base = (base_url or self.config.base_url or "").rstrip("/") + "/"
        out = base
        for p in parts:
            out = urljoin(out, (p or "").lstrip("/"))
        return out

    def canonical_url(self, slug_or_path: str, *, trailing_slash: bool = False, base_url: Optional[str] = None) -> str:
        url = self.build_url(slug_or_path, base_url=base_url)
        if trailing_slash and not url.endswith("/"):
            url += "/"
        return url

    @staticmethod
    def append_query_params(url: str, params: Dict[str, Any], *, merge: bool = True) -> str:
        parts = urlsplit(url)
        existing = dict(parse_qsl(parts.query, keep_blank_values=True)) if merge else {}
        existing.update({k: "" if v is None else str(v) for k, v in params.items()})
        new_q = urlencode(existing, doseq=True)
        return urlunsplit((parts.scheme, parts.netloc.lower(), parts.path, new_q, parts.fragment))

    @staticmethod
    def strip_tracking_params(url: str) -> str:
        try:
            parts = urlsplit(url)
            if not parts.query:
                return url
            kept: List[Tuple[str, str]] = []
            for k, v in parse_qsl(parts.query, keep_blank_values=True):
                lk = k.lower()
                if any(lk.startswith(pref) for pref in _TRACKING_PREFIXES):
                    continue
                if lk in _TRACKING_KEYS:
                    continue
                kept.append((k, v))
            new_q = urlencode(kept, doseq=True)
            return urlunsplit((parts.scheme, parts.netloc.lower(), parts.path.rstrip("/") or "/", new_q, parts.fragment))
        except Exception:
            return url

    # --------------------------- Slug helpers -------------------------------
    @staticmethod
    def _exists_case_insensitive(path: Path) -> bool:
        parent = path.parent
        if not parent.exists():
            return False
        name_lower = path.name.lower()
        try:
            return any(child.name.lower() == name_lower for child in parent.iterdir())
        except Exception:
            return path.exists()

    def dedupe_slug(self, slug: str, *, directory: Union[str, Path], max_len: Optional[int] = None) -> str:
        s = (slug or "article").strip("-")
        mlen = int(max_len) if max_len is not None else self.config.max_slug_length
        s = s[:mlen].strip("-")
        base = Path(directory)
        candidate = s
        if not self._exists_case_insensitive(base / candidate):
            return candidate
        for i in range(2, 100):
            suffix = f"-{i}"
            candidate = (s[: max(1, mlen - len(suffix))].rstrip("-")) + suffix
            if not self._exists_case_insensitive(base / candidate):
                return candidate
        return candidate

    # --------------------------- HTML helpers -------------------------------
    @staticmethod
    def strip_html(text_or_html: str) -> str:
        """Very light tag stripper and whitespace normalizer."""
        if not text_or_html:
            return ""
        txt = _HTML_TAG_RE.sub(" ", text_or_html)
        txt = txt.replace("&nbsp;", " ")
        return _WS_RE.sub(" ", txt).strip()

    @staticmethod
    def build_excerpt(text_or_html: str, max_len: int = 156) -> str:
        """Generate a Yoast-friendly excerpt from text or HTML, cutting on a word/punctuation boundary."""
        plain = EnhancedUtils.strip_html(text_or_html)
        if len(plain) <= max_len:
            return plain
        # Prefer to cut at punctuation within window
        window = plain[:max_len]
        for punct in [". ", "…", "; ", ": ", "! ", "? "]:
            idx = window.rfind(punct)
            if idx != -1 and idx > max_len * 0.6:  # avoid too early cuts
                return plain[:idx + len(punct.strip())].rstrip() + "…"
        # else fallback to last space
        cut = plain.rfind(" ", 0, max_len)
        if cut == -1:
            cut = max_len
        return plain[:cut].rstrip(" ,;:—-") + "…"

    # --------------------------- Text cleaners ------------------------------
    @staticmethod
    def remove_zero_width(s: str) -> str:
        """Strip zero-width characters commonly introduced by copy/paste."""
        if not s:
            return s
        return _ZERO_WIDTH_RE.sub("", s)

# ---------------------------------------------------------------------
# Global instance & shims
# ---------------------------------------------------------------------
_utils_instance = EnhancedUtils()

def slugify(text: str, max_length: int = 60, *, keep_emojis: bool = False) -> str:
    return _utils_instance.slugify(text, max_length=max_length, keep_emojis=keep_emojis)

def ensure_directory(path: Union[str, Path]) -> Path:
    return _utils_instance.ensure_directory(path)

def resolve_env_vars(obj: Any) -> Any:
    return _utils_instance.resolve_env_vars(obj)

def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    return _utils_instance.load_config(config_path)

def sanitize_filename(filename: str) -> str:
    return _utils_instance.sanitize_filename(filename)

def read_json(file_path: Union[str, Path]) -> Optional[Union[Dict, List]]:
    return _utils_instance.read_json(file_path)

def write_json(
    data: Union[Dict, List],
    file_path: Union[str, Path],
    *,
    indent: Optional[int] = None,
    backup: bool = False,
    ensure_ascii: Optional[bool] = None,
    allow_nan: bool = False,
) -> bool:
    return _utils_instance.write_json(
        data, file_path, indent=indent, backup=backup, ensure_ascii=ensure_ascii, allow_nan=allow_nan
    )

def write_text_atomic(path: Union[str, Path], text: str, *, encoding: str = "utf-8") -> bool:
    return EnhancedUtils.write_text_atomic(path, text, encoding=encoding)

def read_text(path: Union[str, Path], *, encoding: str = "utf-8") -> Optional[str]:
    return EnhancedUtils.read_text(path, encoding=encoding)

def build_url(*parts: str, base_url: Optional[str] = None) -> str:
    return _utils_instance.build_url(*parts, base_url=base_url)

def canonical_url(slug_or_path: str, *, trailing_slash: bool = False, base_url: Optional[str] = None) -> str:
    return _utils_instance.canonical_url(slug_or_path, trailing_slash=trailing_slash, base_url=base_url)

def append_query_params(url: str, params: Dict[str, Any], *, merge: bool = True) -> str:
    return EnhancedUtils.append_query_params(url, params, merge=merge)

def strip_tracking_params(url: str) -> str:
    return EnhancedUtils.strip_tracking_params(url)

def cache_clear_slugify() -> None:
    _utils_instance.cache_clear_slugify()

def safe_join(base: Union[str, Path], *parts: str, allow_traversal: bool = False) -> Path:
    return EnhancedUtils.safe_join(base, *parts, allow_traversal=allow_traversal)

def remove_zero_width(s: str) -> str:
    return EnhancedUtils.remove_zero_width(s)

# ---------------------------------------------------------------------
# Auth header builder
# ---------------------------------------------------------------------
def get_auth_header(config: Dict[str, Any]) -> Dict[str, str]:
    """
    Builds an Authorization header from config and environment.

    Supports:
      - Basic (default): needs 'username' and password via env var
        (config['password_env_var'] or WP_APP_PASSWORD) or direct 'password'.
      - Bearer: set config['auth_type']="bearer" and provide token via env var
        config['token_env_var'] or direct 'token'.
    """
    auth_type = (config or {}).get("auth_type", "basic").lower()

    if auth_type == "bearer":
        token_env_var = (config or {}).get("token_env_var", "WP_BEARER_TOKEN")
        token = os.getenv(token_env_var) or (config or {}).get("token")
        if not token:
            raise EnvironmentError(
                f"Missing bearer token. Set '{token_env_var}' env var or 'token' in config."
            )
        return {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "User-Agent": "PythonProHub-Automation/1.0",
        }

    username = (config or {}).get("username")
    password_env_var = (config or {}).get("password_env_var", "WP_APP_PASSWORD")
    password = (config or {}).get("password") or os.getenv(password_env_var)

    if not username or not password:
        raise EnvironmentError(
            f"Missing credentials. Provide 'username' and either 'password' or '{password_env_var}' env var."
        )

    token = f"{username}:{password}"
    encoded = b64encode(token.encode("utf-8")).decode("ascii")
    return {
        "Authorization": f"Basic {encoded}",
        "Content-Type": "application/json",
        "User-Agent": "PythonProHub-Automation/1.0",
    }

# ---------------------------------------------------------------------
# Draft normalizer (integrated)
# ---------------------------------------------------------------------
def normalize_draft(d: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalizes draft data from LLM response to a consistent dictionary shape.
    Ensures keys: title, slug, meta_description, article_content.
    Accepts aliases for backward compatibility and reconstructs content
    from outline structures when needed.
    """
    _slug = _utils_instance.slugify

    if not isinstance(d, dict):
        logger.warning(f"normalize_draft: Input is not a dict ({type(d)}).")
        return {}

    title = remove_zero_width((d.get("title") or "").strip())
    meta_description = remove_zero_width((d.get("meta_description") or d.get("meta description") or d.get("meta") or "").strip())
    slug_val = (d.get("slug") or "").strip()

    # Pull possible content forms
    raw_content = d.get("article_content") or d.get("article content")
    content = ""

    if isinstance(raw_content, str):
        content = remove_zero_width(raw_content.strip())
        logger.debug("normalize_draft: 'article_content' is a string.")
    elif isinstance(raw_content, list):
        # Join list-of-paragraphs
        content = "\n\n".join([remove_zero_width(p.strip()) for p in raw_content if isinstance(p, str) and p.strip()]).strip()
        logger.debug("normalize_draft: 'article_content' joined from list.")
    elif isinstance(raw_content, dict):
        # Reconstruct from dict sections
        parts: List[str] = []
        intro = raw_content.get("introduction")
        if isinstance(intro, str) and intro.strip():
            parts.append(remove_zero_width(intro.strip()))
        for sec in raw_content.get("sections", []):
            if isinstance(sec, dict):
                st = remove_zero_width((sec.get("title") or "").strip())
                sc = remove_zero_width((sec.get("content") or "").strip())
                if st and sc:
                    parts.append(f"## {st}\n\n{sc}".strip())
                elif sc:
                    parts.append(sc)
        concl = raw_content.get("conclusion")
        if isinstance(concl, str) and concl.strip():
            parts.append(f"## Conclusion\n\n{remove_zero_width(concl.strip())}")
        content = "\n\n".join(p for p in parts if p).strip()
        if not content:
            logger.warning("normalize_draft: dict-based 'article_content' reconstructed empty.")
    elif raw_content is not None:
        logger.warning(f"normalize_draft: unexpected 'article_content' type {type(raw_content)}; using empty.")

    # Fallback: look for top-level outline if content still empty
    if not content:
        parts_top: List[str] = []
        intro = d.get("introduction")
        if isinstance(intro, str) and intro.strip():
            parts_top.append(remove_zero_width(intro.strip()))
        for sec in d.get("sections", []):
            if isinstance(sec, dict):
                st = remove_zero_width((sec.get("title") or "").strip())
                sc = remove_zero_width((sec.get("content") or "").strip())
                if st and sc:
                    parts_top.append(f"## {st}\n\n{sc}".strip())
                elif sc:
                    parts_top.append(sc)
        concl = d.get("conclusion")
        if isinstance(concl, str) and concl.strip():
            parts_top.append(f"## Conclusion\n\n{remove_zero_width(concl.strip())}")
        if parts_top:
            content = "\n\n".join(p for p in parts_top if p).strip()
        else:
            logger.warning("normalize_draft: No content found in draft.")

    # Meta description fallback (try first paragraph or title)
    if not meta_description:
        src = (content.split("\n\n", 1)[0].strip() if content else title).strip()
        meta_description = EnhancedUtils.build_excerpt(src, max_len=156)

    # Slug fallback from title
    if not slug_val and title:
        slug_val = _slug(title, max_length=75)

    out = dict(d)  # preserve other fields
    out["title"] = title
    out["slug"] = slug_val
    out["meta_description"] = meta_description
    out["article_content"] = content

    # Clean consumed/aux keys
    for k in ("meta", "meta description", "article content", "introduction", "intro", "sections", "conclusion", "outline"):
        out.pop(k, None)

    return out
