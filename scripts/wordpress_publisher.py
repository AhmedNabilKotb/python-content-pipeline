# scripts/wordpress_publisher.py

import os
import re
import sys
import io
import json
import time
import mimetypes
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Dict, Tuple, Optional, Any, List
from datetime import datetime

from dotenv import load_dotenv
from markdown import markdown
import requests
from requests.auth import HTTPBasicAuth

from google.oauth2 import service_account
from google.auth.transport.requests import Request


# -----------------------------------------------------------------------------
# Optional notifier; provide a no-op fallback if unavailable
# -----------------------------------------------------------------------------
try:
    from scripts.error_emailer import ErrorNotifier, EmailPriority  # type: ignore
except Exception:  # pragma: no cover
    class EmailPriority:
        LOW = "LOW"
        NORMAL = "NORMAL"
        HIGH = "HIGH"
        CRITICAL = "CRITICAL"

    class ErrorNotifier:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def send_error_email(
            self, *, error_summary: str, priority: str = "LOW", additional_data: Optional[dict] = None
        ) -> None:
            logging.getLogger(__name__).warning(
                "[ErrorNotifier fallback] %s | priority=%s | extra=%s",
                error_summary, priority, (additional_data or {}),
            )


# -----------------------------------------------------------------------------
# Setup & Logging  (guard against duplicate handler attachment)
# -----------------------------------------------------------------------------
load_dotenv()

LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

_root = logging.getLogger()
if not _root.handlers:
    handler_file = RotatingFileHandler(
        LOG_DIR / "publisher.log", maxBytes=10_000_000, backupCount=5, encoding="utf-8"
    )
    handler_stream = logging.StreamHandler()
    logging.basicConfig(
        level=logging.INFO,  # switch to DEBUG when tuning
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[handler_file, handler_stream],
    )
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Utilities & tolerant JSON loader (JSONC + ${ENV} placeholders)
# -----------------------------------------------------------------------------
def _fallback_slugify(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    text = re.sub(r"-{2,}", "-", text)
    return text or "article"


try:
    # Prefer shared utils if available (may also be JSONC-aware in your project)
    from scripts.utils.utils import slugify as _slugify  # type: ignore
    from scripts.utils.utils import load_config as _load_config  # type: ignore
    from scripts.utils.utils import strip_html as _strip_html  # type: ignore
    HAVE_UTILS = True
except Exception:
    _slugify = _fallback_slugify  # type: ignore
    HAVE_UTILS = False

    def _strip_html(text_or_html: str) -> str:
        return re.sub(r"<[^>]+>", " ", text_or_html or "").strip()


# --- small per-process cache so settings are parsed once no matter who calls ---
_SETTINGS_CACHE: Optional[Dict[str, Any]] = None
_SETTINGS_PATH: Optional[str] = None


def _expand_env_placeholders_in_text(txt: str) -> str:
    """
    Replace ${VAR} inside JSON text with the ENV value (JSON-escaped).
    Only replaces exact tokens like "${VAR}" (doesn't try to parse bare $VAR).
    """
    def repl(m: re.Match) -> str:
        var = m.group(1)
        val = os.getenv(var, "")
        val = val.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{val}"'

    # Replace only when token is a JSON string containing a single ${VAR}
    # e.g., "api_key": "${BREVO_API_KEY}"
    return re.sub(r'"\$\{([^}]+)\}"', repl, txt)


def _resolve_env_placeholders_in_obj(obj: Any) -> Any:
    """
    After JSON parse, if a string equals "${VAR}" replace with env value
    (or keep original string if env missing).
    """
    if isinstance(obj, dict):
        return {k: _resolve_env_placeholders_in_obj(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve_env_placeholders_in_obj(v) for v in obj]
    if isinstance(obj, str):
        m = re.fullmatch(r"\$\{([^}]+)\}", obj.strip())
        if m:
            return os.getenv(m.group(1), obj)
        return obj
    return obj


def _load_jsonc_relaxed(path: Path) -> Dict[str, Any]:
    """
    Tolerant config loader:
      - If your project utils are available, uses them (handles comments/trailing commas/etc).
      - Else, strips // and /* */ comments, expands ${ENV}, then json.loads.
    Caches the parsed dict per process to avoid duplicate "parsed config" logs from other modules.
    """
    global _SETTINGS_CACHE, _SETTINGS_PATH
    if _SETTINGS_CACHE is not None and _SETTINGS_PATH == str(path):
        return _SETTINGS_CACHE

    if HAVE_UTILS:
        cfg = _load_config(str(path))
        _SETTINGS_CACHE, _SETTINGS_PATH = cfg, str(path)
        return cfg

    txt = path.read_text(encoding="utf-8")
    # remove /* ... */ comments
    txt = re.sub(r"/\*.*?\*/", "", txt, flags=re.S)
    # remove //...... line comments
    txt = re.sub(r"^\s*//.*$", "", txt, flags=re.M)
    # best-effort trailing commas in objects/arrays
    txt = re.sub(r",(\s*[}\]])", r"\1", txt)
    # expand ${ENV} placeholders
    txt = _expand_env_placeholders_in_text(txt)
    cfg = json.loads(txt)
    cfg = _resolve_env_placeholders_in_obj(cfg)
    _SETTINGS_CACHE, _SETTINGS_PATH = cfg, str(path)
    return cfg


def _read_json_relaxed(path: Path) -> Dict[str, Any]:
    """
    Relaxed JSON reader for metadata.json. Uses same JSONC logic as settings.
    """
    try:
        return _load_jsonc_relaxed(path)
    except FileNotFoundError:
        raise
    except Exception as e:
        logger.error(f"Failed to parse JSON at {path}: {e}")
        raise


def _is_int(s: Any) -> bool:
    try:
        int(s)
        return True
    except Exception:
        return False


def _guess_mimetype(filename: str) -> str:
    mt, _ = mimetypes.guess_type(filename)
    return mt or "application/octet-stream"


def _markdown_image(url: str, alt: str = "", title: Optional[str] = None) -> str:
    """
    Safe Markdown image builder—avoids nested f-strings with escapes.
    Produces: ![alt](url "title")  (title omitted if empty)
    """
    alt_safe = (alt or "").replace('"', "'").strip()
    if title:
        title_safe = title.replace('"', "'").strip()
        return f'![{alt_safe}]({url} "{title_safe}")'
    return f'![{alt_safe}]({url})'


# -----------------------------------------------------------------------------
# Main Class
# -----------------------------------------------------------------------------
class ArticlePublisher:
    """
    Publishes generated articles to WordPress using REST API (Application Password),
    with niceties:
      • Idempotency & conflict handling (check slug; update if configured)
      • Robust retries incl. 429/5xx with backoff and Retry-After
      • Markdown → HTML, auto-upload local images & rewrite to WP media URLs
      • Categories/Tags resolution (names or IDs), optional auto-create
      • Featured image from metadata OR first inline upload (fallback)
      • Optional scheduling via metadata.publish_at (ISO or 'YYYY-MM-DD HH:MM')
      • Optional Google Indexing API ping
      • Settings cache to avoid double “parsed config” logs
    """

    def __init__(self, config_path: str = "config/settings.json", settings: Optional[Dict[str, Any]] = None):
        cfg_path = Path(config_path)
        if settings is None and not cfg_path.exists():
            raise FileNotFoundError(f"Config not found: {cfg_path}")

        # >>> JSONC-TOLERANT LOAD with per-process cache (prevents duplicate parse logs)
        self.config: Dict[str, Any] = settings if settings is not None else _load_jsonc_relaxed(cfg_path)
        self._validate_config()

        paths_config = self.config.get("paths", {}) or {}
        self.output_dir = Path(paths_config.get("articles_output_dir", "output"))
        self.article_filename = paths_config.get("article_filename", "article.md")
        self.metadata_filename = paths_config.get("metadata_filename", "metadata.json")

        self.wp = self.config.get("wordpress", {}) or {}
        self.base_url = (self.wp.get("base_url") or "").rstrip("/")
        self.api_base_url = (self.wp.get("api_base_url") or f"{self.base_url}/wp-json").rstrip("/")
        self.post_status = self.wp.get("post_status", "publish")
        self.request_timeout = int(self.wp.get("request_timeout", 30))
        self.allow_taxonomy_create = bool(self.wp.get("auto_create_taxonomies", True))
        self.default_author_id = int(self.wp.get("default_author_id", 1))
        self.excerpt_max_len = int(self.wp.get("excerpt_max_len", 180))
        self.verify_ssl = bool(self.wp.get("verify_ssl", True))
        self.overwrite_if_exists = bool(self.wp.get("overwrite_if_exists", False))
        self.permalink_format = self.wp.get("permalink_format", "/{slug}")

        # Optional: HTTP/HTTPS proxies (same keys you use elsewhere)
        proxies_cfg = (self.config.get("proxies") or {})
        http_proxy = proxies_cfg.get("http") or os.getenv("HTTP_PROXY")
        https_proxy = proxies_cfg.get("https") or os.getenv("HTTPS_PROXY")

        self.published_urls: List[str] = []
        self.error_notifier = ErrorNotifier(self.config)
        self.google_creds = self._get_google_credentials()

        # Networking
        self.session = requests.Session()
        self.session.verify = self.verify_ssl
        self.session.headers.update({"User-Agent": "PythonProHub-Publisher/1.4"})
        if http_proxy or https_proxy:
            self.session.proxies.update({k: v for k, v in (("http", http_proxy), ("https", https_proxy)) if v})

        # Auth (Application Password)
        self._app_pw = os.getenv("WP_APP_PASSWORD") or self.wp.get("application_password")
        self._auth = HTTPBasicAuth(self.wp["username"], self._app_pw)

    # ------------------ Config & Credentials ---------------------------------
    def _validate_config(self) -> None:
        wp = self.config.get("wordpress", {}) or {}
        required = ["base_url", "username", "default_category"]
        missing = [k for k in required if k not in wp]
        if missing:
            raise KeyError(f"Missing required keys in 'wordpress' config: {missing}")

        app_pw = os.getenv("WP_APP_PASSWORD") or wp.get("application_password")
        if not app_pw:
            raise EnvironmentError(
                "WordPress application password not set. Define WP_APP_PASSWORD env var "
                "or add 'application_password' under 'wordpress' in settings.json."
            )

    def _get_google_credentials(self) -> Optional[service_account.Credentials]:
        google_config = self.config.get("google_indexing", {}) or {}
        service_account_file = google_config.get("service_account_file")
        if not service_account_file:
            logger.info("Google Indexing: service_account_file not set — indexing will be skipped.")
            return None

        path = Path(service_account_file)
        if not path.exists():
            logger.error(f"Google service account file not found: {path}")
            return None

        try:
            return service_account.Credentials.from_service_account_file(
                str(path),
                scopes=["https://www.googleapis.com/auth/indexing"]
            )
        except Exception as e:
            logger.error(f"Failed to load Google credentials: {e}")
            return None

    # ------------------ Endpoint mapping -------------------------------------
    @staticmethod
    def _posts_endpoint_for_type(post_type: str) -> str:
        """
        Map a post type to the correct REST collection endpoint.
        - Built-ins are pluralized (posts, pages)
        - Custom post types typically use their own slug unchanged
        """
        pt = (post_type or "").strip().lower()
        if pt in ("post", "posts", ""):
            return "wp/v2/posts"
        if pt in ("page", "pages"):
            return "wp/v2/pages"
        # Media/taxonomies are handled elsewhere; for CPTs use the CPT slug
        return f"wp/v2/{pt}"

    # ------------------ WP Request Helper ------------------------------------
    def _wp_request(
        self,
        method: str,
        endpoint: str,
        *,
        json_body: Any = None,
        data: Any = None,
        files: Any = None,
        params: Dict[str, Any] | None = None,
        retries: int = 2
    ) -> requests.Response:
        """
        Call WP REST with minimal retry on 429/5xx.
        endpoint: absolute or relative to api_base_url
        """
        url = endpoint if endpoint.startswith("http") else f"{self.api_base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        last_exc: Optional[Exception] = None
        for attempt in range(1, retries + 2):
            try:
                resp = self.session.request(
                    method=method.upper(),
                    url=url,
                    json=json_body,
                    data=data,
                    files=files,
                    params=params,
                    auth=self._auth,
                    timeout=self.request_timeout,
                )
                if resp.status_code == 404:
                    logger.error(f"WP {method} {url} -> 404 (rest_no_route?). Body: {resp.text[:300]}")
                # Retry on 429/5xx
                if resp.status_code == 429 and attempt <= retries:
                    delay = float(resp.headers.get("Retry-After", "1.5"))
                    logger.warning(f"WP {method} {url} -> 429; waiting {delay}s then retrying ({attempt}/{retries})...")
                    time.sleep(delay)
                    continue
                if 500 <= resp.status_code < 600 and attempt <= retries:
                    backoff = 1.5 * attempt
                    logger.warning(f"WP {method} {url} -> {resp.status_code}; retrying in {backoff:.1f}s ({attempt}/{retries})...")
                    time.sleep(backoff)
                    continue
                return resp
            except requests.exceptions.RequestException as e:
                last_exc = e
                if attempt <= retries:
                    backoff = 1.5 * attempt
                    logger.warning(f"WP {method} {url} request error: {e}; retrying in {backoff:.1f}s ({attempt}/{retries})...")
                    time.sleep(backoff)
                    continue
                raise
        assert last_exc is not None
        raise last_exc

    # ------------------ Taxonomy Resolution ----------------------------------
    def _resolve_category_id(self, category: Any) -> Optional[int]:
        """
        Accepts id (int), or name/slug (str). Returns an ID, optionally creating it.
        """
        try:
            if _is_int(category):
                return int(category)
            if not category:
                return None

            slug = _slugify(str(category))
            # Try slug first
            r = self._wp_request("GET", "wp/v2/categories", params={"slug": slug, "per_page": 1})
            if r.ok and r.json():
                return int(r.json()[0]["id"])

            # Fallback: search by name (not guaranteed exact)
            r = self._wp_request("GET", "wp/v2/categories", params={"search": str(category), "per_page": 5})
            if r.ok:
                items = r.json()
                for it in items:
                    if it.get("name", "").strip().lower() == str(category).strip().lower():
                        return int(it["id"])

            if self.allow_taxonomy_create:
                payload = {"name": str(category), "slug": slug}
                r = self._wp_request("POST", "wp/v2/categories", json_body=payload)
                if r.ok:
                    return int(r.json()["id"])
                else:
                    logger.error(f"Failed to create category '{category}': {r.status_code} {r.text[:300]}")
                    return None
            else:
                logger.warning(f"Category '{category}' not found and auto-create disabled.")
                return None
        except Exception as e:
            logger.error(f"Error resolving category '{category}': {e}")
            return None

    def _resolve_categories(self, categories: Any) -> List[int]:
        """
        Accepts a single id/name or list of ids/names. Returns a list of IDs.
        """
        if not categories:
            return []
        if not isinstance(categories, list):
            categories = [categories]
        out: List[int] = []
        for c in categories:
            cid = self._resolve_category_id(c)
            if cid:
                out.append(int(cid))
        # fallback to default category
        return out or [self._resolve_category_id(self.wp.get("default_category", 1)) or 1]

    def _resolve_tag_ids(self, tags: Any) -> List[int]:
        """
        Accepts list of ids or names. Returns a list of tag IDs, creating when allowed.
        """
        out: List[int] = []
        if not tags or not isinstance(tags, list):
            return out

        for t in tags:
            try:
                if _is_int(t):
                    out.append(int(t))
                    continue

                name = str(t).strip()
                slug = _slugify(name)

                r = self._wp_request("GET", "wp/v2/tags", params={"slug": slug, "per_page": 1})
                if r.ok and r.json():
                    out.append(int(r.json()[0]["id"]))
                    continue

                r = self._wp_request("GET", "wp/v2/tags", params={"search": name, "per_page": 5})
                if r.ok:
                    found = False
                    for it in r.json():
                        if it.get("name", "").strip().lower() == name.lower():
                            out.append(int(it["id"]))
                            found = True
                            break
                    if found:
                        continue

                if self.allow_taxonomy_create:
                    r = self._wp_request("POST", "wp/v2/tags", json_body={"name": name, "slug": slug})
                    if r.ok:
                        out.append(int(r.json()["id"]))
                    else:
                        logger.error(f"Failed to create tag '{name}': {r.status_code} {r.text[:300]}")
                else:
                    logger.warning(f"Tag '{name}' not found and auto-create disabled.")
            except Exception as e:
                logger.error(f"Error resolving tag '{t}': {e}")
        return out

    # ------------------ File I/O ---------------------------------------------
    def load_article(self, article_path: Path) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        try:
            content = article_path.read_text(encoding="utf-8")
            metadata_path = article_path.parent / self.metadata_filename
            metadata = _read_json_relaxed(metadata_path)
            return metadata, content
        except Exception as e:
            logger.error(f"Failed to load article from {article_path}: {e}")
            return None, None

    # ------------------ Images in Markdown -----------------------------------
    # Support title: ![alt](path "title"), allow URLs or local paths
    _IMG_MD_RE = re.compile(
        r"!\[(?P<alt>.*?)\]\((?P<src>[^)\s]+)(?:\s+\"(?P<title>.*?)\")?\)",
        flags=re.IGNORECASE,
    )

    def _upload_media(self, fname: str, file_bytes: bytes, title: str = "", alt_text: str = "") -> Tuple[Optional[int], Optional[str]]:
        """
        Upload a media file and return (media_id, source_url).
        """
        endpoint = "wp/v2/media"
        headers = {"Content-Disposition": f'attachment; filename="{Path(fname).name}"'}
        files = {
            "file": (Path(fname).name, io.BytesIO(file_bytes), _guess_mimetype(fname)),
        }
        data = {}
        if title:
            data["title"] = title
        if alt_text:
            data["alt_text"] = alt_text

        try:
            r = self._wp_request("POST", endpoint, files=files, data=data)
            if not r.ok:
                logger.error(f"Media upload failed: {r.status_code} {r.text[:300]}")
                return None, None
            data = r.json()
            return int(data.get("id")), (data.get("source_url") or data.get("guid", {}).get("rendered"))
        except Exception as e:
            logger.error(f"Media upload error: {e}")
            return None, None

    def _find_local_images(self, md: str) -> List[Tuple[str, str, Optional[str]]]:
        """
        Return list of (src, alt, title) for local (non-HTTP) images.
        """
        items: List[Tuple[str, str, Optional[str]]] = []
        for m in self._IMG_MD_RE.finditer(md or ""):
            src = (m.group("src") or "").strip()
            if not src or src.startswith(("http://", "https://")):
                continue
            items.append((src, m.group("alt") or "", m.group("title")))
        return items

    def _rewrite_markdown_images(self, md: str, article_dir: Path) -> Tuple[str, List[int]]:
        """
        Upload local images referenced in markdown and rewrite to their WP URLs.
        Returns (rewritten_markdown, uploaded_media_ids)
        """
        uploaded_ids: List[int] = []

        def _repl(m: re.Match) -> str:
            alt = m.group("alt") or ""
            src = (m.group("src") or "").strip()
            title = m.group("title") or ""
            if not src or src.startswith(("http://", "https://")):
                return m.group(0)
            # try local path
            p = (article_dir / src)
            if not p.exists():
                logger.warning(f"Inline image not found on disk: {p}")
                return m.group(0)
            try:
                mid, url = self._upload_media(p.name, p.read_bytes(), title=title or alt or p.stem, alt_text=alt)
                if mid and url:
                    uploaded_ids.append(mid)
                    return _markdown_image(url, alt, title if title else None)
            except Exception as e:
                logger.warning(f"Failed to upload inline image '{src}': {e}")
            return m.group(0)

        rewritten = self._IMG_MD_RE.sub(_repl, md or "")
        return rewritten, uploaded_ids

    # ------------------ Content Processing -----------------------------------
    def process_content(self, content: str, article_dir: Path) -> Tuple[str, List[int]]:
        """
        1) Upload local images referenced in markdown and rewrite to their WP URLs
        2) Remove a single leading H1 if the first non-blank line is '# ...'
        3) Convert Markdown → HTML
        Returns (html, uploaded_media_ids)
        """
        if not content:
            return "", []

        # Step 1: upload local images & rewrite markdown
        content_md, uploaded_ids = self._rewrite_markdown_images(content, article_dir)

        # Step 2: drop the first H1 (WordPress title will be used)
        lines = content_md.splitlines()
        i = 0
        while i < len(lines) and not lines[i].strip():
            i += 1
        if i < len(lines) and re.match(r'^#\s+.+$', lines[i]):
            del lines[i]
            while i < len(lines) and not lines[i].strip():
                del lines[i]
        content_no_h1 = "\n".join(lines)

        # Step 3: MD → HTML
        try:
            html = markdown(content_no_h1, extensions=["fenced_code", "tables"])
            return html, uploaded_ids
        except Exception as e:
            logger.error(f"Markdown conversion failed: {e}. Returning raw content as HTML-safe <pre>.")
            from html import escape
            return f"<pre>{escape(content_no_h1)}</pre>", uploaded_ids

    def generate_unique_slug(self, base_slug: str) -> str:
        base_slug = (base_slug or "article").strip("-")
        return f"{base_slug}-{datetime.now().strftime('%Y%m%d%H%M%S')}"

    # ------------------ Media (Featured Image) --------------------------------
    def _resolve_featured_media(
        self,
        metadata: Dict[str, Any],
        article_dir: Path,
        inline_media_ids: List[int]
    ) -> Optional[int]:
        """
        If metadata has 'featured_image' as local path or URL, upload and return media ID.
        Else, fallback to the first uploaded inline media (if any).
        """
        src = (metadata.get("featured_image") or "").strip()
        if src:
            try:
                if not src.startswith(("http://", "https://")):
                    p = (article_dir / src)
                    if p.exists():
                        mid, _ = self._upload_media(p.name, p.read_bytes(), title=metadata.get("title", ""))
                        if mid:
                            return mid
                else:
                    r = self.session.get(src, timeout=20)
                    r.raise_for_status()
                    mid, _ = self._upload_media(Path(src).name, r.content, title=metadata.get("title", ""))
                    if mid:
                        return mid
            except Exception as e:
                logger.warning(f"Featured image processing failed for '{src}': {e}")

        # Fallback: use first inline image upload (if any)
        return inline_media_ids[0] if inline_media_ids else None

    # ------------------ Post Data --------------------------------------------
    def _prepare_post_data(
        self,
        metadata: Dict[str, Any],
        html_content: str,
        slug_value: str,
        article_dir: Path,
        inline_media_ids: List[int]
    ) -> Dict[str, Any]:
        # Categories (accept single or list; names or ids)
        cats_meta = metadata.get("categories") or metadata.get("category_id") or self.wp.get("default_category", 1)
        category_ids = self._resolve_categories(cats_meta)

        yoast_meta_desc = (metadata.get("meta") or metadata.get("meta_description") or "").strip()
        yoast_focuskw = (metadata.get("keyphrase") or metadata.get("focus_keyphrase") or "").strip()

        # excerpt (prefer metadata.excerpt; else trim from content)
        excerpt = (metadata.get("excerpt") or yoast_meta_desc or "").strip()
        if not excerpt:
            plain = _strip_html(html_content)
            excerpt = (plain[: self.excerpt_max_len] + "…") if len(plain) > self.excerpt_max_len else plain

        post: Dict[str, Any] = {
            "title": (metadata.get("title") or "Untitled Post").strip(),
            "content": html_content,
            "status": metadata.get("status", self.post_status),
            "slug": slug_value,
            "categories": category_ids,
            "author": int(metadata.get("author_id", self.default_author_id)),
        }

        # Scheduling support via metadata.publish_at (UTC ISO or local naive)
        publish_at = (metadata.get("publish_at") or "").strip()
        if publish_at:
            try:
                if publish_at.endswith("Z"):
                    post["date_gmt"] = publish_at
                    post["status"] = "future"
                else:
                    post["date"] = publish_at
                    post["status"] = "future"
            except Exception:
                logger.warning(f"Invalid publish_at value ignored: {publish_at}")

        # Tags: accept IDs or names
        tag_ids = self._resolve_tag_ids(metadata.get("tags", []))
        if tag_ids:
            post["tags"] = tag_ids

        if excerpt:
            post["excerpt"] = excerpt

        # Featured image
        media_id = self._resolve_featured_media(metadata, article_dir, inline_media_ids)
        if media_id:
            post["featured_media"] = int(media_id)

        # Yoast meta (site must expose these via REST)
        post_meta = {
            "_yoast_wpseo_metadesc": yoast_meta_desc,
            "_yoast_wpseo_focuskw": yoast_focuskw,
        }
        user_meta = metadata.get("custom_meta") or metadata.get("meta_fields") or {}
        if isinstance(user_meta, dict):
            post_meta.update(user_meta)
        post["meta"] = post_meta

        # Custom post type support (optional) — map to correct collection endpoint
        cpt = (self.wp.get("post_type") or metadata.get("post_type") or "post").strip()
        post["__endpoint__"] = self._posts_endpoint_for_type(cpt)

        return post

    # ------------------ WordPress Publishing ----------------------------------
    def _build_canonical_url(self, slug: str) -> Optional[str]:
        root = (self.base_url or self.api_base_url.split("/wp-json", 1)[0]).rstrip("/")
        if not root:
            return None
        try:
            url_path = (self.permalink_format or "/{slug}").format(slug=slug).lstrip("/")
        except Exception:
            url_path = slug
        return f"{root}/{url_path}"

    def _find_existing_post_by_slug(self, endpoint: str, slug_value: str) -> Optional[Dict[str, Any]]:
        r = self._wp_request("GET", f"{endpoint}", params={"slug": slug_value, "per_page": 1})
        if r.ok and isinstance(r.json(), list) and r.json():
            return r.json()[0]
        return None

    def publish_article(self, article_path: Path) -> Tuple[Optional[int], Optional[str]]:
        """
        Publishes a single article from its Markdown file path.
        Returns (post_id, published_url) on success, (None, None) on failure.
        """
        metadata, content = self.load_article(article_path)
        if not metadata or not content:
            logger.error(f"Missing metadata/content for {article_path}.")
            return None, None

        # Process markdown (upload local images, rewrite MD, convert to HTML)
        html_content, inline_media_ids = self.process_content(content, article_path.parent)

        desired_slug = (metadata.get("slug") or "").strip()
        if not desired_slug:
            logger.warning(f"{article_path.name}: metadata missing slug — generating from title.")
            desired_slug = _slugify(metadata.get("title", "untitled-post"))

        # Build post payload (endpoint stored inside)
        post_data = self._prepare_post_data(metadata, html_content, desired_slug, article_path.parent, inline_media_ids)
        endpoint = post_data.pop("__endpoint__", "wp/v2/posts")

        # Idempotency: optionally update if a post with same slug exists
        try_update = self.overwrite_if_exists or bool(metadata.get("overwrite_if_exists", False))
        if try_update:
            existing = self._find_existing_post_by_slug(endpoint, desired_slug)
            if existing:
                post_id = existing.get("id")
                logger.info(f"Post with slug '{desired_slug}' exists (id={post_id}); updating in-place.")
                r = self._wp_request("POST", f"{endpoint}/{post_id}", json_body=post_data)
                if not r.ok:
                    logger.error(f"WP update failed [{r.status_code}]: {r.text[:400]}")
                    r.raise_for_status()
                data = r.json()
                link = data.get("link") or data.get("guid", {}).get("rendered") or self._build_canonical_url(data.get("slug", desired_slug))
                self.published_urls.append(link or "")
                logger.info(f"Updated OK: id={post_id} url={link} slug={data.get('slug')}")
                return post_id, link

        # Attempt publish with single conflict recovery for slug
        attempts = 0
        while attempts < 3:
            attempts += 1
            try:
                resp = self._wp_request("POST", endpoint, json_body=post_data)
                if resp.status_code in (400, 409):
                    body = (resp.text or "").lower()
                    if "slug" in body or "already exists" in body:
                        new_slug = self.generate_unique_slug(post_data["slug"])
                        logger.warning(f"Slug conflict for '{post_data['slug']}', retrying with '{new_slug}'")
                        post_data["slug"] = new_slug
                        resp = self._wp_request("POST", endpoint, json_body=post_data)

                if resp.status_code >= 400:
                    logger.error(f"WP publish failed (try {attempts}) [{resp.status_code}] on {endpoint}: {resp.text[:400]}")
                    if 500 <= resp.status_code < 600 and attempts < 3:
                        time.sleep(2 * attempts)
                        continue
                    resp.raise_for_status()

                data = resp.json()
                post_id = data.get("id")
                link = data.get("link") or data.get("guid", {}).get("rendered") or self._build_canonical_url(
                    data.get("slug", post_data["slug"])
                )
                self.published_urls.append(link or "")
                logger.info(f"Published OK: id={post_id} url={link} slug={data.get('slug')}")
                return post_id, link

            except requests.exceptions.RequestException as e:
                resp_text = getattr(e.response, "text", "N/A") if hasattr(e, "response") and e.response is not None else "N/A"
                logger.error(
                    f"Request error (try {attempts}) while publishing '{desired_slug}' via {endpoint}: {e}. "
                    f"Response: {resp_text[:400]}"
                )
                if attempts < 3:
                    time.sleep(2 * attempts)
                    continue
                self.error_notifier.send_error_email(
                    error_summary=f"WordPress Publishing Error for slug: {desired_slug}",
                    priority=EmailPriority.HIGH,
                    additional_data={"error": str(e), "response_body": resp_text}
                )
                return None, None
            except Exception as e:
                logger.error(f"Unexpected error while publishing '{desired_slug}': {e}", exc_info=True)
                self.error_notifier.send_error_email(
                    error_summary=f"WordPress Publishing Unexpected Error for slug: {desired_slug}",
                    priority=EmailPriority.HIGH,
                    additional_data={"error": str(e)}
                )
                return None, None

    # ------------------ Google Indexing ---------------------------------------
    def request_google_indexing(self, url_to_index: str):
        if not self.google_creds:
            logger.info(f"Indexing skipped (no credentials): {url_to_index}")
            return

        endpoint = "https://indexing.googleapis.com/v3/urlNotifications:publish"
        payload = {"url": url_to_index, "type": "URL_UPDATED"}

        try:
            if not self.google_creds.valid:
                self.google_creds.refresh(Request())
        except Exception as e:
            logger.error(f"Failed to refresh Google token: {e}")
            return

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.google_creds.token}"
        }

        for attempt in range(1, 2 + 1):
            try:
                r = self.session.post(endpoint, headers=headers, json=payload, timeout=15)
                if r.status_code >= 400:
                    logger.error(f"Indexing API error (try {attempt}) [{r.status_code}]: {r.text[:300]}")
                    if 500 <= r.status_code < 600 and attempt < 2:
                        time.sleep(2)
                        continue
                    r.raise_for_status()
                logger.info(f"Indexing requested: {url_to_index}")
                return
            except Exception as e:
                logger.error(f"Indexing request failed (try {attempt}) for {url_to_index}: {e}")
                if attempt >= 2:
                    self.error_notifier.send_error_email(
                        error_summary=f"Google Indexing API Error for URL: {url_to_index}",
                        priority=EmailPriority.NORMAL,
                        additional_data={"error": str(e)}
                    )

    # ------------------ Connectivity ------------------------------------------
    def test_connection(self) -> bool:
        """
        Validate both the types endpoint (capabilities) and the real posts
        collection (route existence). If a custom post type is configured,
        verify its route as well.
        """
        try:
            url_types = f"{self.api_base_url}/wp/v2/types/post"
            r1 = self.session.get(url_types, auth=self._auth, timeout=self.request_timeout)
            r1.raise_for_status()

            url_posts = f"{self.api_base_url}/wp/v2/posts"
            r2 = self.session.get(url_posts, auth=self._auth, timeout=self.request_timeout, params={"per_page": 1})
            r2.raise_for_status()

            cpt = (self.wp.get("post_type") or "").strip().lower()
            if cpt and cpt not in ("post", "page", "posts", "pages"):
                ep = self._posts_endpoint_for_type(cpt)
                url_cpt = f"{self.api_base_url}/{ep}"
                r3 = self.session.get(url_cpt, auth=self._auth, timeout=self.request_timeout, params={"per_page": 1})
                r3.raise_for_status()

            logger.info("WordPress API connection OK.")
            return True
        except Exception as e:
            logger.error(f"WordPress connection test failed: {e}")
            self.error_notifier.send_error_email(
                error_summary="Critical: WordPress Connection Failure",
                priority=EmailPriority.HIGH,
                additional_data={"error": str(e)}
            )
            return False

    # ------------------ Runner -----------------------------------------------
    def run(self) -> None:
        logger.info("Starting publishing run...")
        if not self.test_connection():
            logger.critical("Aborting run due to connection failure.")
            return

        if not self.output_dir.exists():
            logger.error(f"Output directory not found: {self.output_dir}")
            return

        article_dirs = sorted([d for d in self.output_dir.iterdir() if d.is_dir()], key=lambda p: p.name)
        total = len(article_dirs)
        logger.info(f"Found {total} article folder(s) to publish.")

        success_count = 0
        for i, article_dir in enumerate(article_dirs, start=1):
            logger.info(f"[{i}/{total}] Processing '{article_dir.name}'...")
            article_path = article_dir / self.article_filename
            if not article_path.exists():
                logger.warning(f"Missing {self.article_filename} in: {article_dir}")
                continue

            post_id, url = self.publish_article(article_path)
            if post_id and url:
                success_count += 1
                # Optional indexing
                # self.request_google_indexing(url)

        logger.info(f"Publishing complete. {success_count}/{total} article(s) published.")
        if self.published_urls:
            logger.info("Published URLs:\n" + "\n".join(self.published_urls))


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    load_dotenv()
    test_config_path = Path("config/settings.json")
    if not test_config_path.exists():
        print(f"ERROR: {test_config_path} not found. Cannot run standalone publisher test.")
        sys.exit(1)

    try:
        # single parse + cached for all importers
        settings = _load_jsonc_relaxed(test_config_path)
        publisher = ArticlePublisher(config_path=str(test_config_path), settings=settings)
        publisher.run()
    except (KeyError, EnvironmentError, FileNotFoundError) as e:
        logger.critical(f"Configuration error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Unexpected error during publishing: {e}", exc_info=True)
        sys.exit(1)
