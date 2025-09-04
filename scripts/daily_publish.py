# scripts/daily_publish.py

import os
import json
import logging
import time
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
import traceback
import sys
import re
import signal
import threading
from tempfile import NamedTemporaryFile  # atomic writes
import socket  # for hostname in lock metadata

from dotenv import load_dotenv, find_dotenv
from openai import OpenAI  # SDK v1
import httpx

# --- LOGGING ---------------------------------------------------------------
LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

_root = logging.getLogger()
if not _root.handlers:  # prevent duplicate handlers if imported by another runner
    logging.basicConfig(
        level=logging.INFO,  # switch to DEBUG to tune locally
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(LOG_DIR / "daily_automation.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
for noisy in ("matplotlib", "httpx", "httpcore", "urllib3", "openai"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


# --- Helpers ---------------------------------------------------------------
def _parse_bool(val: Optional[str], *, default: bool = False) -> bool:
    if val is None:
        return default
    return str(val).strip().lower() in {"1", "true", "yes", "on", "y"}


def _fallback_slugify(text: str) -> str:
    t = (text or "").lower()
    t = re.sub(r"[^a-z0-9]+", "-", t).strip("-")
    t = re.sub(r"-{2,}", "-", t)
    return t or "article"


def _normalize_niche_name(name: str) -> str:
    """
    Normalize niche names so weights and pools can match even with punctuation differences.
    E.g., 'Scientific & Numerical Computing' == 'scientific and numerical computing'
    """
    s = (name or "").lower()
    s = s.replace("&", " and ")
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s{2,}", " ", s).strip()
    return s


def _parse_json_env_dict(env_var: str) -> Optional[Dict[str, Any]]:
    """Parse an env var as JSON dict, returning None on failure or if empty."""
    raw = os.getenv(env_var)
    if not raw:
        return None
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else None
    except Exception:
        logger.warning(f"Failed to parse {env_var} as JSON; ignoring.")
        return None


# --- IMPORTS & FALLBACKS ---------------------------------------------------
try:
    # Package-style imports (python -m scripts.daily_publish)
    from .keyphrase_tracker import KeyphraseTracker, KeyphraseConfig
    from .utils.utils import load_config as load_app_config, slugify
    from .utils.jsonc import load_jsonc
    from .generate_articles import ArticleGenerator, AppConfig as ArticleAppConfig
    from .chart_generator import ChartGenerator
    from .live_demo_embedder import LiveDemoEmbedder
    from .wordpress_publisher import ArticlePublisher
    from .internal_linker import ArticleLinker, ArticleLinkConfig
    from .send_newsletter import NewsletterSender
    from .error_emailer import ErrorNotifier, EmailPriority
    from .code_cleaner import CodeCleaner
    from .article_cta_injector import ArticleCTAInjector
    from .article_challenge_injector import ArticleChallengeInjector
    from .google_indexer import GoogleIndexer
    from .yoast_preflight import YoastPreflight
    from .helpers.outbound_links import select_outbound_links  # optional helper
    from .seo_finalizer import finalize as finalize_seo  # type: ignore

    # NEW: structure, technical, quality
    from .structure_enforcer import StructureEnforcer  # type: ignore
    from .technical_enhancer import TechnicalEnhancer  # type: ignore
    from .quality_gate import QualityGate  # type: ignore
except Exception:
    # Script-style fallback (python scripts/daily_publish.py)
    sys.path.append(str(Path(__file__).parent.parent.resolve()))

    try:
        from scripts.keyphrase_tracker import KeyphraseTracker, KeyphraseConfig  # type: ignore
    except Exception:
        class KeyphraseConfig:  # minimal fallback
            def __init__(self, storage_path: Path): self.storage_path = storage_path
        class KeyphraseTracker:  # minimal fallback
            def __init__(self, config: KeyphraseConfig): self.config = config
            def get_unique_keyphrase(self, base_keyphrase: str, content_slug: str = "") -> str:
                return base_keyphrase

    try:
        from scripts.utils.utils import load_config as load_app_config, slugify  # type: ignore
    except Exception:
        def load_app_config(path: Path) -> Dict[str, Any]:
            return json.loads(Path(path).read_text(encoding="utf-8"))
        slugify = _fallback_slugify  # type: ignore

    # JSONC loader (tolerates // and /* */ in JSON files)
    try:
        from scripts.utils.jsonc import load_jsonc  # type: ignore
    except Exception:
        # Minimal inline fallback: strip comments then json.loads
        def load_jsonc(path: Path):
            text = Path(path).read_text(encoding="utf-8")
            text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
            text = re.sub(r"^\s*//.*$", "", text, flags=re.M)
            return json.loads(text)

    try:
        from scripts.generate_articles import ArticleGenerator, AppConfig as ArticleAppConfig  # type: ignore
    except Exception:
        raise RuntimeError("generate_articles.py not found; cannot continue.")

    try:
        from scripts.chart_generator import ChartGenerator  # type: ignore
    except Exception:
        class ChartGenerator:  # noop fallback
            def __init__(self, openai_client=None): pass
            def generate_and_embed_chart(self, content: str, slug: str):
                return content, False

    try:
        from scripts.live_demo_embedder import LiveDemoEmbedder  # type: ignore
    except Exception:
        class LiveDemoEmbedder:  # noop fallback
            def __init__(self, settings: Dict[str, Any]): pass
            def embed_demos(self, content: str) -> str: return content

    try:
        from scripts.wordpress_publisher import ArticlePublisher  # type: ignore
    except Exception:
        raise RuntimeError("wordpress_publisher.py not found; cannot continue.")

    try:
        from scripts.internal_linker import ArticleLinker, ArticleLinkConfig  # type: ignore
    except Exception:
        class ArticleLinkConfig:
            def __init__(self, output_dir: Path, max_links=3, link_section_header="", min_articles_for_linking=2,
                         max_outbound_links=2, min_paragraph_words_for_outbound=12,
                         avoid_outbound_domains=None, link_future_posts=True, strip_query_tracking=True,
                         inline_placeholder_regex=r"(?mi)^\s*_?see related guides in this series\._?\s*$|^\s*\[\[RELATED\]\]\s*$",
                         inline_intro_label="**See also:**", inline_links_separator=" · ", inline_links_count=2,
                         marker_start="<!-- RELATED:START -->", marker_end="<!-- RELATED:END -->",
                         on_existing="replace", use_html_for_outbound=True, outbound_rel_default="noopener noreferrer",
                         outbound_target_blank=True, outbound_group_order=()):
                self.output_dir = output_dir
                self.max_links = max_links
                self.link_section_header = link_section_header
                self.min_articles_for_linking = min_articles_for_linking
                self.max_outbound_links = max_outbound_links
                self.min_paragraph_words_for_outbound = min_paragraph_words_for_outbound
                self.avoid_outbound_domains = avoid_outbound_domains
                self.link_future_posts = link_future_posts
                self.strip_query_tracking = strip_query_tracking
                self.inline_placeholder_regex = inline_placeholder_regex
                self.inline_intro_label = inline_intro_label
                self.inline_links_separator = inline_links_separator
                self.inline_links_count = inline_links_count
                self.marker_start = marker_start
                self.marker_end = marker_end
                self.on_existing = on_existing
                self.use_html_for_outbound = use_html_for_outbound
                self.outbound_rel_default = outbound_rel_default
                self.outbound_target_blank = outbound_target_blank
                self.outbound_group_order = outbound_group_order
        class ArticleLinker:
            def __init__(self, config: ArticleLinkConfig): self.config = config
            def get_articles_by_niche(self, base: Path) -> Dict[str, List[Path]]: return {}
            def inject_internal_links(self, m: Dict[str, List[Path]]) -> None: pass
            def inject_outbound_links(self, content: str, mapping: Dict[str, List[str]]) -> str: return content

    try:
        from scripts.send_newsletter import NewsletterSender  # type: ignore
    except Exception:
        class NewsletterSender:
            def __init__(self, settings: Dict[str, Any]): pass
            def send_digest_email(self, payload: List[Dict[str, Any]]): pass

    try:
        from scripts.error_emailer import ErrorNotifier, EmailPriority  # type: ignore
    except Exception:
        class EmailPriority:  # align names with real enum
            LOW="LOW"; NORMAL="NORMAL"; HIGH="HIGH"; CRITICAL="CRITICAL"
        class ErrorNotifier:
            def __init__(self, settings: Dict[str, Any]): self._is_initialized=False
            def send_error_email(self, **kwargs): pass

    try:
        from scripts.code_cleaner import CodeCleaner  # type: ignore
    except Exception:
        class CodeCleaner:
            def __init__(self, settings: Dict[str, Any], base_dir: Path): pass
            def clean_article_content(self, c: str, slug: str) -> str: return c

    try:
        from scripts.article_cta_injector import ArticleCTAInjector  # type: ignore
    except Exception:
        class ArticleCTAInjector:
            def __init__(self, settings: Dict[str, Any]): pass
            def inject_cta(self, c: str, context: Optional[Dict[str, str]] = None) -> str: return c

    try:
        from scripts.article_challenge_injector import ArticleChallengeInjector  # type: ignore
    except Exception:
        class ArticleChallengeInjector:
            def __init__(self, settings: Dict[str, Any], openai_client=None): pass
            def inject_challenge(self, c: str, topic: str) -> str: return c

    try:
        from scripts.google_indexer import GoogleIndexer  # type: ignore
    except Exception:
        class GoogleIndexer:
            def __init__(self, settings: Dict[str, Any]): self.enabled=False
            def publish_url(self, url: str): pass

    try:
        from scripts.yoast_preflight import YoastPreflight  # type: ignore
    except Exception:
        class YoastPreflight:
            def __init__(self, settings: Dict[str, Any], article_linker=None): pass
            def fix(self, **kwargs):
                # passthrough fallback
                return kwargs.get("content",""), kwargs.get("title",""), kwargs.get("meta",""), {"changed": {}}

    try:
        from scripts.helpers.outbound_links import select_outbound_links  # type: ignore
    except Exception:
        select_outbound_links = None  # type: ignore

    # FINALIZER fallback (no-op)
    try:
        from scripts.seo_finalizer import finalize as finalize_seo  # type: ignore
    except Exception:
        def finalize_seo(title: str, meta: str, content_md: str, keyphrase: str, knobs: Dict[str, Any]):
            return title, meta, content_md, {"changed": []}

    # NEW: structure, technical, quality fallbacks
    try:
        from scripts.structure_enforcer import StructureEnforcer  # type: ignore
    except Exception:
        class StructureEnforcer:
            def __init__(self, settings: Dict[str, Any]): pass
            def enforce(self, **kwargs):
                content = kwargs.get("content", "")
                return content, {"changed": False, "notes": []}

    try:
        from scripts.technical_enhancer import TechnicalEnhancer  # type: ignore
    except Exception:
        class TechnicalEnhancer:
            def __init__(self, settings: Dict[str, Any], openai_client=None): pass
            def enhance(self, **kwargs):
                content = kwargs.get("content", "")
                return content, {"touched": False}

    try:
        from scripts.quality_gate import QualityGate  # type: ignore
    except Exception:
        class QualityGate:
            def __init__(self, settings: Dict[str, Any]): pass
            def evaluate(self, **kwargs):
                # ok, reasons, extra_metrics
                return True, [], {"gate": "fallback"}


# --- Atomic writers --------------------------------------------------------
def _save_json_atomic(path: Path, data: Any) -> None:
    """Write JSON atomically to avoid partial/corrupt files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=str(path.parent)) as tmp:
        json.dump(data, tmp, ensure_ascii=False, indent=2)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_name = tmp.name
    Path(tmp_name).replace(path)

def _save_text_atomic(path: Path, text: str) -> None:
    """Write text atomically (mirrors _save_json_atomic)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=str(path.parent)) as tmp:
        tmp.write(text)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_name = tmp.name
    Path(tmp_name).replace(path)


# --- Run lock (prevent concurrent runs) ------------------------------------
class RunLock:
    """
    Robust single-run lock with:
      • Atomic creation (O_CREAT|O_EXCL) to avoid races
      • TTL auto-expiry (LOCK_TTL_MIN or ctor ttl_minutes)
      • PID liveness check (prevents zombie/abandoned locks)
      • Optional override (LOCK_FORCE=1)
    Lock file JSON schema:
      {"pid": 1234, "started": "2025-08-24T11:00:00+00:00", "hostname": "..."}
    """
    def __init__(self, lock_path: Path, ttl_minutes: int = 45, force: bool = False):
        self.lock_path = Path(lock_path)
        self.ttl_minutes = int(ttl_minutes)
        self.force = bool(force)
        self._acquired = False

    # ---- internals ----
    @staticmethod
    def _is_pid_alive(pid: int) -> bool:
        if not isinstance(pid, int) or pid <= 0:
            return False
        try:
            # Works on Unix & Windows: raises if not running
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            # Process exists but not ours
            return True
        except OSError:
            # Fallback probe for Unix-like /proc
            try:
                return Path(f"/proc/{pid}").exists()
            except Exception:
                return False

    def _read_state(self) -> Optional[Dict[str, Any]]:
        try:
            data = json.loads(self.lock_path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else None
        except Exception:
            return None

    def _is_stale(self, state: Optional[Dict[str, Any]]) -> Tuple[bool, str]:
        """Return (is_stale, reason)."""
        try:
            mtime_age = datetime.now(timezone.utc) - datetime.fromtimestamp(
                self.lock_path.stat().st_mtime, tz=timezone.utc
            )
        except Exception:
            mtime_age = timedelta(days=999)

        # TTL via file mtime
        if mtime_age > timedelta(minutes=self.ttl_minutes):
            return True, f"TTL expired (mtime age {mtime_age})"

        # PID liveness
        pid = 0
        if isinstance(state, dict):
            try:
                pid = int(state.get("pid", 0))
            except Exception:
                pid = 0
        if pid and not self._is_pid_alive(pid):
            return True, f"PID {pid} not alive"

        # Started timestamp sanity (optional)
        if isinstance(state, dict) and state.get("started"):
            try:
                started = datetime.fromisoformat(str(state["started"]))
                if started.tzinfo is None:
                    started = started.replace(tzinfo=timezone.utc)
                if datetime.now(timezone.utc) - started > timedelta(minutes=self.ttl_minutes):
                    return True, f"TTL expired (started age {datetime.now(timezone.utc) - started})"
            except Exception:
                # If unreadable, treat conservatively as stale only if TTL already said so
                pass

        return False, ""

    def _write_lockfile(self, fd: int) -> None:
        payload = {
            "pid": os.getpid(),
            "started": datetime.now(timezone.utc).isoformat(),
            "hostname": socket.gethostname(),
        }
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f)

    def acquire(self) -> None:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
        try:
            fd = os.open(self.lock_path, flags, 0o644)
            self._write_lockfile(fd)
            self._acquired = True
            logger.debug(f"RunLock acquired (fresh): {self.lock_path}")
            return
        except FileExistsError:
            # A lock exists: inspect it
            state = self._read_state()
            stale, reason = self._is_stale(state)
            if self.force:
                logger.warning("LOCK_FORCE=1 — stealing existing lock.")
                try:
                    self.lock_path.unlink(missing_ok=True)
                except Exception:
                    pass
            elif stale:
                logger.warning(f"Stale lock detected ({reason}); cleaning it up.")
                try:
                    self.lock_path.unlink(missing_ok=True)
                except Exception:
                    pass
            else:
                age = datetime.now() - datetime.fromtimestamp(self.lock_path.stat().st_mtime)
                raise SystemExit(
                    f"Another run appears active (lock age {age}). "
                    "Use LOCK_FORCE=1 or wait."
                )

            # Try to create again after cleanup
            fd = os.open(self.lock_path, flags, 0o644)
            self._write_lockfile(fd)
            self._acquired = True
            logger.debug(f"RunLock acquired (after cleanup): {self.lock_path}")

    def release(self) -> None:
        if self._acquired:
            try:
                self.lock_path.unlink(missing_ok=True)
            except Exception:
                pass
            self._acquired = False

    # ---- context manager ----
    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.release()


# --- Config ----------------------------------------------------------------
@dataclass
class DailyAutomationConfig:
    base_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent)
    log_dir: Path = Path("logs")
    settings_path: Path = Path("config/settings.json")
    publishing_schedule_path: Path = Path("config/publishing_schedule.json")
    published_log_path: Path = Path("logs/published_log.jsonl")
    output_dir: Path = Path("output")

    article_filename: str = "article.md"
    metadata_filename: str = "metadata.json"

    max_publish_retries: int = 3
    retry_delay_base_seconds: int = 60
    inter_article_delay_range_seconds: List[int] = field(default_factory=lambda: [120, 300])
    cpu_load_threshold: float = 2.0
    cpu_cooldown_period_seconds: int = 180
    min_topic_backlog_threshold: int = 10

    # NEW: optional publish window "HH:MM-HH:MM" (local time)
    publish_window: Optional[str] = None


# --- Main automation --------------------------------------------------------
class DailyAutomation:
    def __init__(self, config: Optional[DailyAutomationConfig] = None):
        self.config = config or DailyAutomationConfig()
        self.error_messages: List[str] = []
        self.successfully_published_articles: List[Dict] = []
        self.openai_client: Optional[OpenAI] = None

        # Orchestration flags (env feature toggles)
        self.dry_run: bool = _parse_bool(os.getenv("DRY_RUN")) or _parse_bool(os.getenv("SKIP_PUBLISH"))
        self.max_articles_override: Optional[int] = int(os.getenv("MAX_ARTICLES", "0") or "0") or None
        self.run_date_override: Optional[str] = os.getenv("RUN_DATE")  # YYYY-MM-DD, for backfilling
        self.skip_wp_test: bool = _parse_bool(os.getenv("SKIP_WP_TEST"))
        self.skip_yoast: bool = _parse_bool(os.getenv("SKIP_YOAST_PREFLIGHT"))
        self.skip_charts: bool = _parse_bool(os.getenv("SKIP_CHARTS"))
        self.skip_live_demos: bool = _parse_bool(os.getenv("SKIP_LIVE_DEMOS"))
        self.skip_cleanup: bool = _parse_bool(os.getenv("SKIP_CLEANUP"))
        self.skip_cta: bool = _parse_bool(os.getenv("SKIP_CTA"))
        self.skip_challenge: bool = _parse_bool(os.getenv("SKIP_CHALLENGE"))
        self.skip_internal_links: bool = _parse_bool(os.getenv("SKIP_INTERNAL_LINKS"))
        self.skip_newsletter: bool = _parse_bool(os.getenv("SKIP_NEWSLETTER"))

        # Filters (new)
        self.only_niches: Optional[List[str]] = [s.strip() for s in os.getenv("ONLY_NICHES", "").split(",") if s.strip()] or None
        self.exclude_niches: Optional[List[str]] = [s.strip() for s in os.getenv("EXCLUDE_NICHES", "").split(",") if s.strip()] or None

        # NEW toggles for the three modules
        self.skip_structure: bool = _parse_bool(os.getenv("SKIP_STRUCTURE_ENFORCER"))
        self.skip_technical: bool = _parse_bool(os.getenv("SKIP_TECHNICAL_ENHANCER"))
        self.skip_quality_gate: bool = _parse_bool(os.getenv("SKIP_QUALITY_GATE"))

        # Graceful stop
        self._stop_event = threading.Event()
        self._install_signal_handlers()

        self._load_app_settings()
        self._load_workflow_config()
        self._initialize_openai_client()
        self._setup_environment()

        self.keyphrase_tracker = KeyphraseTracker(
            config=KeyphraseConfig(storage_path=self.config.base_dir / "config" / "used_keyphrases.json")
        )

        self.article_generator = ArticleGenerator(
            ArticleAppConfig(settings_path=self.config.settings_path),
            openai_client=self.openai_client,
            keyphrase_tracker_instance=self.keyphrase_tracker,
        )
        self.chart_generator = ChartGenerator(openai_client=self.openai_client)
        self.live_demo_embedder = LiveDemoEmbedder(self.app_settings)

        # Lazily create the publisher (so dry-runs don't require WP creds)
        self._publisher: Optional[ArticlePublisher] = None

        self.google_indexer = GoogleIndexer(self.app_settings)

        linker_config_data = self.app_settings.get("internal_linking", {})
        # Build ArticleLinkConfig with extended knobs (backwards compatible)
        linker_config = ArticleLinkConfig(
            output_dir=self.config.base_dir / self.config.output_dir,
            max_links=linker_config_data.get("max_links", 3),
            link_section_header=linker_config_data.get("link_section_header", "\n\n---\n\n### 🔗 Related Articles\n"),
            min_articles_for_linking=linker_config_data.get("min_articles_for_linking", 2),
            # NEW/extended outbound controls:
            max_outbound_links=linker_config_data.get("max_outbound_links", 2),
            min_paragraph_words_for_outbound=linker_config_data.get("min_paragraph_words_for_outbound", 12),
            avoid_outbound_domains=set(linker_config_data.get("avoid_outbound_domains", [])) or None,
            link_future_posts=linker_config_data.get("link_future_posts", True),
            strip_query_tracking=linker_config_data.get("strip_query_tracking", True),
            inline_placeholder_regex=linker_config_data.get(
                "inline_placeholder_regex",
                r"(?mi)^\s*_?see related guides in this series\._?\s*$|^\s*\[\[RELATED\]\]\s*$",
            ),
            inline_intro_label=linker_config_data.get("inline_intro_label", "**See also:**"),
            inline_links_separator=linker_config_data.get("inline_links_separator", " · "),
            inline_links_count=linker_config_data.get("inline_links_count", 2),
            marker_start=linker_config_data.get("marker_start", "<!-- RELATED:START -->"),
            marker_end=linker_config_data.get("marker_end", "<!-- RELATED:END -->"),
            on_existing=linker_config_data.get("on_existing", "replace"),
            use_html_for_outbound=linker_config_data.get("use_html_for_outbound", True),
            outbound_rel_default=linker_config_data.get("outbound_rel_default", "noopener noreferrer"),
            outbound_target_blank=linker_config_data.get(
                "outbound_target_blank",
                _parse_bool(os.getenv("OUTBOUND_TARGET_BLANK"), default=True),
            ),
            outbound_group_order=tuple(
                linker_config_data.get(
                    "outbound_group_order",
                    (
                        "official_docs", "official", "library", "guide", "course", "tutorial",
                        "platform", "book", "interactive", "community", "blog", "newsletter",
                        "fallback",
                    ),
                )
            ),
        )
        self.article_linker = ArticleLinker(config=linker_config)
        # Expose for convenience
        self.linker_group_by_type: bool = bool(linker_config_data.get("linker_group_by_type", False))

        self.newsletter_sender = NewsletterSender(self.app_settings)
        self.error_notifier = ErrorNotifier(self.app_settings)
        self.code_cleaner = CodeCleaner(self.app_settings, self.config.base_dir)
        self.article_cta_injector = ArticleCTAInjector(self.app_settings)
        self.article_challenge_injector = ArticleChallengeInjector(self.app_settings, openai_client=self.openai_client)

        # NEW: structure/technical/quality components
        self.structure_enforcer = StructureEnforcer(self.app_settings)
        self.technical_enhancer = TechnicalEnhancer(self.app_settings, openai_client=self.openai_client)
        self.quality_gate = QualityGate(self.app_settings)

    # --- Bootstrapping ------------------------------------------------------
    def _install_signal_handlers(self):
        def _handler(signum, frame):
            logger.warning(f"Received signal {signum}; will stop after current step.")
            self._stop_event.set()
        try:
            signal.signal(signal.SIGINT, _handler)
            signal.signal(signal.SIGTERM, _handler)
        except Exception:
            pass

    def _safe_sleep(self, seconds: float):
        """Interruptible sleep that respects SIGINT/SIGTERM."""
        end = time.time() + max(0.0, seconds)
        while time.time() < end and not self._stop_event.is_set():
            time.sleep(min(0.5, end - time.time()))

    def _load_app_settings(self):
        try:
            self.app_settings = load_app_config(self.config.base_dir / self.config.settings_path)
            logger.info("Main application settings loaded successfully.")
        except Exception as e:
            self._log_error(f"FATAL: Error loading settings: {e}", critical=True)
            raise

    def _load_workflow_config(self):
        workflow_settings = self.app_settings.get("workflow", {})
        self.config.max_publish_retries = int(workflow_settings.get("max_publish_retries", self.config.max_publish_retries))
        self.config.retry_delay_base_seconds = int(workflow_settings.get(
            "retry_delay_base_seconds", self.config.retry_delay_base_seconds
        ))
        self.config.inter_article_delay_range_seconds = list(workflow_settings.get(
            "inter_article_delay_range_seconds", self.config.inter_article_delay_range_seconds
        ))
        self.config.cpu_load_threshold = float(workflow_settings.get("cpu_load_threshold", self.config.cpu_load_threshold))
        self.config.cpu_cooldown_period_seconds = int(workflow_settings.get(
            "cpu_cooldown_period_seconds", self.config.cpu_cooldown_period_seconds
        ))
        self.config.min_topic_backlog_threshold = int(workflow_settings.get(
            "min_topic_backlog_threshold", self.config.min_topic_backlog_threshold
        ))
        # NEW: optional publish window
        self.config.publish_window = (
            os.getenv("PUBLISH_WINDOW")
            or workflow_settings.get("publish_window")
            or None
        )
        if self.config.publish_window:
            logger.info(f"Publish window enforced: {self.config.publish_window}")
        logger.info("Workflow configuration loaded.")

    def _initialize_openai_client(self):
        load_dotenv(find_dotenv(usecwd=True))
        api_key = self.app_settings.get("api_keys", {}).get("openai") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.critical("OpenAI API key not found. Set OPENAI_API_KEY or put it in settings.json.")
            raise ValueError("Valid OpenAI API key not found.")

        proxies_config = self.app_settings.get("proxies", {})
        http_proxy = proxies_config.get("http") or os.getenv("HTTP_PROXY")
        https_proxy = proxies_config.get("https") or os.getenv("HTTPS_PROXY")
        # NOTE: httpx.Client has NO 'no_proxy' kwarg; honor NO_PROXY via environment.
        no_proxy_env = os.getenv("NO_PROXY") or os.getenv("no_proxy")
        if no_proxy_env:
            logger.info(f"NO_PROXY will be honored via environment: {no_proxy_env}")

        httpx_client_args: Dict[str, object] = {"timeout": 30.0}  # trust_env=True by default
        if http_proxy or https_proxy:
            proxies_dict: Dict[str, str] = {}
            if http_proxy:
                proxies_dict["http://"] = http_proxy
            if https_proxy:
                proxies_dict["https://"] = https_proxy
            httpx_client_args["proxies"] = proxies_dict
            logger.info(f"OpenAI client will use proxies: {proxies_dict}")

        base_url = os.getenv("OPENAI_BASE_URL") or self.app_settings.get("api_keys", {}).get("openai_base_url")

        try:
            custom_http_client = httpx.Client(**httpx_client_args)
            if base_url:
                self.openai_client = OpenAI(api_key=api_key, base_url=base_url, http_client=custom_http_client)
            else:
                self.openai_client = OpenAI(api_key=api_key, http_client=custom_http_client)
            logger.info("OpenAI client successfully initialized.")
        except Exception as e:
            self._log_error(f"Failed to initialize OpenAI client: {e}", critical=True)
            raise ConnectionError("OpenAI API client initialization failed.") from e

    def _setup_environment(self):
        (self.config.base_dir / self.config.log_dir).mkdir(exist_ok=True, parents=True)
        (self.config.base_dir / self.config.output_dir).mkdir(exist_ok=True, parents=True)
        published_log = self.config.base_dir / self.config.published_log_path
        published_log.parent.mkdir(parents=True, exist_ok=True)
        if not published_log.exists():
            published_log.touch()

        if not self._test_openai_connection():
            raise ConnectionError("OpenAI API connection failed during environment setup.")

    def _test_openai_connection(self) -> bool:
        if not self.openai_client:
            self._log_error("OpenAI client not initialized.")
            return False
        try:
            # Simple health-check
            self.openai_client.models.list()
            logger.info("OpenAI connection successful.")
            return True
        except Exception as e:
            self._log_error(f"OpenAI connection failed: {e}")
            return False

    # --- Utilities ----------------------------------------------------------
    def _log_error(self, message: str, critical: bool = False):
        prefix = "CRITICAL" if critical else "ERROR"
        full_message = f"[{datetime.now().isoformat()}] {prefix}: {message}"
        logger.error(full_message)
        self.error_messages.append(full_message)

    def _count_warnings(self, report: Dict[str, Any], max_warn: int) -> int:
        """Robustly count warnings regardless of the shape returned by YoastPreflight."""
        wf = report.get("warnings", [])
        if isinstance(wf, int):
            return wf
        if isinstance(wf, (list, tuple, set)):
            return len(wf)
        if wf in (None, {}, ""):
            return 0
        # Unknown/invalid shape → fail safe by exceeding max
        return max_warn + 1

    def _choose_best_version(
        self,
        base_pack: Tuple[str, str, str, Dict[str, Any]],
        final_pack: Tuple[str, str, str, Dict[str, Any]],
        knobs: Dict[str, Any],
    ) -> Tuple[str, str, str, Dict[str, Any], bool]:
        """
        Non-regression guard:
        Pick the version with >= SEO, >= Readability and <= warnings.
        If neither dominates, prefer higher SEO, then higher Readability, then fewer warnings.
        Returns (title, meta, content, report, used_final: bool)
        """
        (b_title, b_meta, b_content, b_rep) = base_pack
        (f_title, f_meta, f_content, f_rep) = final_pack

        target_seo = float(knobs.get("target_seo_score", 95))
        target_read = float(knobs.get("target_readability_score", 95))
        max_warn = int(knobs.get("max_warnings", 1))

        b_warn = self._count_warnings(b_rep, max_warn)
        f_warn = self._count_warnings(f_rep, max_warn)

        b_scores = (float(b_rep.get("seo", 0.0)), float(b_rep.get("readability", 0.0)), -b_warn)
        f_scores = (float(f_rep.get("seo", 0.0)), float(f_rep.get("readability", 0.0)), -f_warn)

        # Prefer the one that meets targets; if both/none, prefer lexicographically by (SEO, Readability, -warnings).
        b_meets = (b_scores[0] >= target_seo) and (b_scores[1] >= target_read) and (b_warn <= max_warn)
        f_meets = (f_scores[0] >= target_seo) and (f_scores[1] >= target_read) and (f_warn <= max_warn)

        pick_final = False
        if f_meets and not b_meets:
            pick_final = True
        elif b_meets and not f_meets:
            pick_final = False
        else:
            pick_final = f_scores > b_scores  # higher SEO, then readability, then fewer warnings

        if pick_final:
            return f_title, f_meta, f_content, f_rep, True
        return b_title, b_meta, b_content, b_rep, False

    # ---------- Weighted helpers for dynamic mode ---------------------------
    def _load_niche_weights(self) -> Dict[str, float]:
        """
        Loads config/niche_weights.json (JSONC allowed) and returns a mapping of
        *normalized niche name* -> weight (>0). Missing/failed file → weights {} (treated as 1.0 later).
        """
        weights_path = self.config.base_dir / "config" / "niche_weights.json"
        if not weights_path.exists():
            logger.info("niche_weights.json not found — defaulting all niche weights to 1.0.")
            return {}
        try:
            raw = load_jsonc(weights_path)
        except Exception as e:
            logger.warning(f"Failed to parse niche_weights.json ({e}) — defaulting weights to 1.0.")
            return {}
        out: Dict[str, float] = {}
        for k, v in (raw or {}).items():
            try:
                w = float(v)
                if w > 0:
                    out[_normalize_niche_name(str(k))] = w
            except Exception:
                continue
        return out

    def _fair_rounded_quotas(self, niches: List[str], weights: List[float], total: int) -> Dict[str, int]:
        """
        Given niche list and positive weights, compute integer quotas that:
        - Sum to 'total'
        - Are proportional to weights using largest remainders method
        """
        if total <= 0 or not niches:
            return {}

        s = sum(weights) or 1.0
        shares = [w / s * total for w in weights]
        floors = [int(x) for x in shares]
        remainder = total - sum(floors)
        # pair: (fractional_part, index)
        fracs = sorted([(shares[i] - floors[i], i) for i in range(len(niches))], reverse=True)
        quotas = {niches[i]: floors[i] for i in range(len(niches))}
        for _, i in fracs[:max(0, remainder)]:
            quotas[niches[i]] += 1
        return quotas

    # ---------- Dynamic candidates from topic_pools.json --------------------
    def _dynamic_candidates_from_topic_pools(self, day_str: str, count: int) -> List[Dict[str, Any]]:
        """
        Build N candidates for `day_str` by sampling config/topic_pools.json.
        Uses weights from config/niche_weights.json to bias selection.
        Interleaves picks across niches to avoid clumping while honoring proportions.
        Dedupes keyphrases via KeyphraseTracker.
        """
        pools_path = self.config.base_dir / "config" / "topic_pools.json"
        if not pools_path.exists():
            logger.warning(f"topic_pools.json not found at {pools_path}; dynamic fallback cannot run.")
            return []

        try:
            pools = load_jsonc(pools_path)
        except Exception as e:
            logger.error(f"Failed to read topic_pools.json: {e}")
            return []

        # Build: original_niche_name -> [topics...]
        by_niche_raw: Dict[str, List[str]] = {}
        for niche, topics in (pools or {}).items():
            bucket: List[str] = []
            for t in topics or []:
                if isinstance(t, str) and t.strip():
                    bucket.append(t.strip())
            if bucket:
                by_niche_raw[str(niche)] = bucket

        if not by_niche_raw:
            logger.warning("topic_pools.json had no usable topics.")
            return []

        # Map normalized niche → canonical/original niche key we will use
        norm_to_orig: Dict[str, str] = {}
        for orig in by_niche_raw.keys():
            norm = _normalize_niche_name(orig)
            # prefer first seen original for a given normalized key
            norm_to_orig.setdefault(norm, orig)

        # Load weights and align them to the *present* pools
        weights_norm = self._load_niche_weights()
        niches_order: List[str] = []   # canonical/original names present in pools
        weights_vec: List[float] = []  # aligned weights

        for norm, orig in norm_to_orig.items():
            w = weights_norm.get(norm, 1.0)
            niches_order.append(orig)
            weights_vec.append(max(0.0001, float(w)))

        # If all weights are 1.0, it's still fine — we’ll distribute evenly.
        quotas = self._fair_rounded_quotas(niches_order, weights_vec, max(1, int(count)))

        # Prepare mutable buckets per niche, copy to avoid mutating pools on disk
        shuffle_topics = _parse_bool(os.getenv("TOPIC_SHUFFLE"))
        by_niche: Dict[str, List[str]] = {}
        for orig, lst in by_niche_raw.items():
            lst_copy = list(lst)
            if shuffle_topics:
                random.shuffle(lst_copy)
            by_niche[orig] = lst_copy

        # Interleaved picking honoring quotas
        out: List[Dict[str, Any]] = []
        seen_topics: set[str] = set()

        # Round-robin over niches while quotas remain
        active_niches = [n for n in niches_order if quotas.get(n, 0) > 0 and by_niche.get(n)]
        idx = 0
        safety = 0
        while len(out) < count and active_niches and safety < 10000:
            safety += 1
            niche = active_niches[idx % len(active_niches)]
            if quotas.get(niche, 0) <= 0 or not by_niche.get(niche):
                # refresh active list and continue
                active_niches = [n for n in niches_order if quotas.get(n, 0) > 0 and by_niche.get(n)]
                idx += 1
                continue

            topic = None
            # pop first unused topic in this niche
            while by_niche[niche] and (by_niche[niche][0] in seen_topics):
                by_niche[niche].pop(0)
            if by_niche[niche]:
                topic = by_niche[niche].pop(0)

            if topic:
                seen_topics.add(topic)
                keyphrase = self.keyphrase_tracker.get_unique_keyphrase(
                    base_keyphrase=topic, content_slug=slugify(topic)
                )
                out.append({
                    "date": day_str,
                    "topic": topic,
                    "niche": niche,
                    "keyphrase": keyphrase,
                    "published": False
                })
                quotas[niche] = quotas.get(niche, 0) - 1

            # Recompute active niches after each pick to avoid starving others
            active_niches = [n for n in niches_order if quotas.get(n, 0) > 0 and by_niche.get(n)]
            idx += 1

        # If we still need more, drain any remaining topics across all niches
        if len(out) < count:
            for niche in niches_order:
                while len(out) < count and by_niche.get(niche):
                    t = by_niche[niche].pop(0)
                    if t in seen_topics:
                        continue
                    seen_topics.add(t)
                    k = self.keyphrase_tracker.get_unique_keyphrase(base_keyphrase=t, content_slug=slugify(t))
                    out.append({"date": day_str, "topic": t, "niche": niche, "keyphrase": k, "published": False})
                if len(out) >= count:
                    break

        logger.info(
            "Dynamic candidates generated: %d for %s (niches used: %s)",
            len(out), day_str, ", ".join(sorted({c['niche'] for c in out}))
        )
        return out[:max(1, int(count))]

    # ---------- Schedule loader with dynamic fallback -----------------------
    def _load_publishing_schedule(self) -> List[Dict]:
        schedule_path = self.config.base_dir / self.config.publishing_schedule_path
        articles_per_day = int(self.app_settings.get("content_strategy", {}).get("articles_per_day", 1))
        if self.max_articles_override:
            articles_per_day = min(articles_per_day, int(self.max_articles_override))
        day = (self.run_date_override or datetime.now().strftime("%Y-%m-%d")).strip()

        # Force dynamic mode via env
        if _parse_bool(os.getenv("SKIP_SCHEDULE")):
            logger.info("SKIP_SCHEDULE=1 — generating dynamic candidates from topic_pools.json (weighted).")
            return self._dynamic_candidates_from_topic_pools(day, max(1, int(articles_per_day)))

        logger.info(f"Attempting to load publishing schedule from: {schedule_path}")
        if not schedule_path.exists():
            logger.warning(f"Publishing schedule file not found at {schedule_path}. Using dynamic fallback for {day}.")
            return self._dynamic_candidates_from_topic_pools(day, max(1, int(articles_per_day)))

        try:
            raw = load_jsonc(schedule_path)
        except Exception as e:
            logger.error(f"Error decoding schedule: {e}. Using dynamic fallback.")
            return self._dynamic_candidates_from_topic_pools(day, max(1, int(articles_per_day)))

        # Normalize supported shapes:
        #  1) [ {date, topic, ...}, ... ]
        #  2) { "schedule": [ {...}, ... ] }
        #  3) { "YYYY-MM-DD": [ {...}, ... ] }  (daily buckets – optional)
        entries = None
        if isinstance(raw, list):
            entries = raw
        elif isinstance(raw, dict):
            if isinstance(raw.get("schedule"), list):
                entries = raw.get("schedule")
            elif isinstance(raw.get(day), list):
                # Daily-bucketed file; inject the date for today’s entries if missing
                e = raw.get(day) or []
                for it in e:
                    if isinstance(it, dict) and not it.get("date"):
                        it["date"] = day
                entries = e

        if not isinstance(entries, list):
            logger.error("Schedule must be a list OR {'schedule': [...]} OR a day-bucketed dict. Using dynamic fallback.")
            return self._dynamic_candidates_from_topic_pools(day, max(1, int(articles_per_day)))

        # Sanitize and de-rogue entries
        cleaned: List[Dict[str, Any]] = []
        seen_pairs = set()
        for idx, it in enumerate(entries, 1):
            if not isinstance(it, dict):
                logger.warning(f"Schedule item #{idx} is not an object; skipping.")
                continue
            d = str(it.get("date") or day).strip()
            topic = str(it.get("topic") or it.get("title") or "").strip()
            if not (d and topic):
                logger.warning(f"Schedule item #{idx} missing date/topic; skipping.")
                continue
            kp = str(it.get("keyphrase") or topic).strip()
            niche = str(it.get("niche") or it.get("category") or "General").strip()
            pub = bool(it.get("published", False))
            key = (d[:10], topic)
            if key in seen_pairs:
                logger.warning(f"Duplicate schedule entry for ({key[0]}, {key[1]}). Skipping duplicate.")
                continue
            seen_pairs.add(key)
            cleaned.append({
                "date": d[:10],
                "topic": topic,
                "keyphrase": kp,
                "niche": niche,
                "published": pub,
            })

        # Filter by ONLY/EXCLUDE niches if provided
        if self.only_niches:
            cleaned = [c for c in cleaned if c["niche"] in self.only_niches]
        if self.exclude_niches:
            cleaned = [c for c in cleaned if c["niche"] not in self.exclude_niches]

        todays = [a for a in cleaned if a["date"] == day and not a["published"]]
        logger.info(f"Found {len(todays)} articles scheduled for {day}. Will process up to {articles_per_day}.")
        if not todays:
            logger.info("Schedule empty for today — using dynamic fallback from topic_pools.json (weighted).")
            return self._dynamic_candidates_from_topic_pools(day, max(1, int(articles_per_day)))

        return todays[:max(1, int(articles_per_day))]

    def _update_publishing_status(self, article_data: Dict, published: bool):
        schedule_path = self.config.base_dir / self.config.publishing_schedule_path
        if not schedule_path.exists():
            return  # nothing to update when running dynamic

        try:
            raw = load_jsonc(schedule_path)
        except Exception as e:
            self._log_error(f"Failed to read schedule for status update: {e}")
            return

        # Determine shape and get a mutable list reference
        root_container = raw
        container_type = "list"
        if isinstance(raw, list):
            entries = raw
        elif isinstance(raw, dict) and isinstance(raw.get("schedule"), list):
            entries = raw["schedule"]
            container_type = "dict_schedule"
        elif isinstance(raw, dict):
            # day-bucketed shape: { "YYYY-MM-DD": [ ... ] }
            day = str(article_data.get("date", "")).strip()
            if day and isinstance(raw.get(day), list):
                entries = raw[day]
                container_type = "dict_daily"
            else:
                self._log_error("Unsupported schedule shape for status update; skipping write.")
                return
        else:
            self._log_error("Unsupported schedule type for status update; skipping write.")
            return

        found = False
        for it in entries:
            if not isinstance(it, dict):
                continue
            if str(it.get("topic")) == str(article_data.get("topic")) and str(it.get("date")) == str(article_data.get("date")):
                it["published"] = bool(published)
                found = True
                break

        if not found:
            logger.warning(
                f"Could not find article '{article_data.get('topic')}' for date '{article_data.get('date')}' in schedule to update status."
            )

        # Write back atomically, preserving original shape
        try:
            if container_type == "list":
                _save_json_atomic(schedule_path, entries)
            elif container_type == "dict_schedule":
                root_container["schedule"] = entries
                _save_json_atomic(schedule_path, root_container)
            else:  # dict_daily
                day = str(article_data.get("date", "")).strip()
                root_container[day] = entries
                _save_json_atomic(schedule_path, root_container)
        except Exception as e:
            self._log_error(f"Failed to update publishing status: {e}")

    def _log_publication(
        self,
        article_meta: Dict,
        success: bool,
        post_id: Optional[int],
        url: Optional[str],
        attempt: int,
        error_msg: Optional[str] = None,
        extra_metrics: Optional[Dict[str, Any]] = None,
        skipped_reason: Optional[str] = None,
    ):
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "topic": article_meta.get("topic"),
            "keyphrase": article_meta.get("keyphrase"),
            "niche": article_meta.get("niche"),
            "status": "SUCCESS" if success else "FAILURE",
            "postId": post_id,
            "url": url,
            "attempts_made": attempt,
            "error_message": error_msg,
        }
        if skipped_reason:
            log_entry["skipped_reason"] = skipped_reason
        if extra_metrics:
            log_entry.update({"metrics": extra_metrics})
        try:
            with open(self.config.base_dir / self.config.published_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry) + "\n")
        except Exception as e:
            self._log_error(f"Failed to write to publication log: {e}")

    def _check_cpu_and_wait(self):
        if hasattr(os, "getloadavg"):
            load_1min, _, _ = os.getloadavg()
            if load_1min > self.config.cpu_load_threshold:
                wait_s = self.config.cpu_cooldown_period_seconds
                logger.warning(f"CPU load high ({load_1min:.2f}). Waiting {wait_s}s...")
                self._safe_sleep(wait_s)

    def _build_canonical_url(self, slug: str) -> Optional[str]:
        wp_conf = self.app_settings.get("wordpress", {}) or {}
        root = (wp_conf.get("base_url") or "").split("/wp-json", 1)[0].rstrip("/")
        if not root:
            api_root = (wp_conf.get("api_base_url") or "").split("/wp-json", 1)[0].rstrip("/")
            root = api_root or ""
        if not root:
            return None
        fmt = wp_conf.get("permalink_format", "/{slug}")
        try:
            url_path = fmt.format(slug=slug).lstrip("/")
        except Exception:
            url_path = slug
        return f"{root}/{url_path}"

    # --- Time window helper -------------------------------------------------
    def _within_publish_window(self) -> bool:
        """
        If PUBLISH_WINDOW is set as 'HH:MM-HH:MM' (local time), enforce it.
        Supports windows that cross midnight (e.g., 22:00-05:00).
        """
        win = (self.config.publish_window or "").strip()
        if not win:
            return True
        m = re.match(r"^\s*(\d{2}):(\d{2})\s*-\s*(\d{2}):(\d{2})\s*$", win)
        if not m:
            logger.warning(f"Invalid PUBLISH_WINDOW format: '{win}' (expected 'HH:MM-HH:MM'). Ignoring.")
            return True
        sh, sm, eh, em = map(int, m.groups())
        now_t = datetime.now().time()
        start = now_t.replace(hour=sh, minute=sm, second=0, microsecond=0)
        end = now_t.replace(hour=eh, minute=em, second=0, microsecond=0)
        if start <= end:
            return start <= now_t <= end
        # window crosses midnight
        return now_t >= start or now_t <= end

    # --- Content helpers ----------------------------------------------------
    def _assemble_content_if_needed(self, seo_result) -> str:
        content = getattr(seo_result, "content", None)
        if content:
            return content

        sections = getattr(seo_result, "sections", None)
        conclusion = getattr(seo_result, "conclusion", "") or ""
        if not sections:
            return ""
        parts = []
        for s in sections:
            st = (s.get("title") or "").strip()
            sc = (s.get("content") or "").strip()
            if st and sc:
                parts.append(f"## {st}\n\n{sc}")
            elif sc:
                parts.append(sc)
        if conclusion.strip():
            parts.append(f"## Conclusion\n\n{conclusion.strip()}")
        return "\n\n".join(p for p in parts if p)

    def _validate_seo_result(self, seo_result) -> Tuple[str, str, str, str]:
        title = getattr(seo_result, "title", "") or ""
        slug = getattr(seo_result, "slug", "") or ""
        meta = getattr(seo_result, "meta", "") or getattr(seo_result, "meta_description", "") or ""
        content = self._assemble_content_if_needed(seo_result)

        missing = [k for k, v in [("title", title), ("slug", slug), ("meta", meta), ("content", content)] if not v]
        if missing:
            raise ValueError(f"Missing/empty required fields from SEO result: {missing}")

        return title, slug, meta, content

    def _get_yoast_knobs(self) -> Dict[str, Any]:
        yoast = self.app_settings.get("yoast_compliance", {}) or {}
        linking = self.app_settings.get("internal_linking", {}) or {}

        # NEW: flexible readability defaults + env overrides
        mode_env = os.getenv("READABILITY_MODE")
        weights_env = _parse_json_env_dict("READABILITY_WEIGHTS")
        adaptive_env = os.getenv("ADAPTIVE_READABILITY")

        readability_mode = (mode_env or yoast.get("readability_mode", "weighted")).lower()
        readability_weights = weights_env or yoast.get("readability_weights", {
            "transitions": 0.35,
            "passive": 0.30,
            "long_sentence": 0.25,
            "paragraph": 0.10,
        })
        adaptive_readability = _parse_bool(adaptive_env, default=bool(yoast.get("adaptive_readability", True)))

        return {
            "target_seo_score": yoast.get("target_seo_score", 95),
            "target_readability_score": yoast.get("target_readability_score", 95),
            "max_warnings": yoast.get("max_warnings", 1),
            "max_enhancement_iterations": yoast.get("max_enhancement_iterations", 6),
            "early_stop_patience": yoast.get("early_stop_patience", 4),

            # legacy/simple thresholds (still used internally by YoastPreflight)
            "flesch_reading_ease_good": yoast.get("flesch_reading_ease_good", 55),
            "passive_voice_max_percent": yoast.get("passive_voice_max_percent", 5),
            "min_transition_word_percent": yoast.get("min_transition_word_percent", 30),
            "max_long_sentence_percent": yoast.get("max_long_sentence_percent", 20),
            "max_paragraph_words": yoast.get("max_paragraph_words", 160),
            "max_consecutive_same_start": yoast.get("max_consecutive_same_start", 0),

            "min_internal_links": yoast.get("min_internal_links", 1),
            "min_outbound_links": yoast.get("min_outbound_links", 2),
            "max_outbound_links": linking.get("max_outbound_links", 2),

            "min_content_words": yoast.get("min_content_words", 1100),
            "min_subheadings": yoast.get("min_subheadings", 3),
            "max_title_length": yoast.get("max_title_length", 60),
            "min_title_length": yoast.get("min_title_length", 40),
            "max_meta_length": yoast.get("max_meta_length", 156),
            "min_meta_length": yoast.get("min_meta_length", 120),
            "min_keyphrase_density": yoast.get("min_keyphrase_density", 0.5),
            "max_keyphrase_density": yoast.get("max_keyphrase_density", 2.5),

            # NEW: pass through flexible readability controls
            "readability_mode": readability_mode,
            "readability_weights": readability_weights,
            "adaptive_readability": adaptive_readability,
        }

    def _get_gate_thresholds(self, yoast_knobs: Dict[str, Any]) -> Tuple[float, float, int, bool, float]:
        """
        Prefer settings.quality_gate if present; otherwise fall back to yoast_compliance targets.
        Returns: (min_seo, min_read, max_warn, allow_near_miss, near_miss_margin)
        """
        qg = self.app_settings.get("quality_gate", {}) or {}
        fin = self.app_settings.get("finalizer", {}) or {}
        min_seo = float(qg.get("min_seo", yoast_knobs.get("target_seo_score", 100)))
        min_read = float(qg.get("min_readability", yoast_knobs.get("target_readability_score", 100)))
        max_warn = int(qg.get("max_warnings", yoast_knobs.get("max_warnings", 0)))
        allow_nm = bool(qg.get("allow_near_miss", True))
        # default near-miss margin: finalizer.run_if_within_points or 5
        near_miss = float(qg.get("near_miss_margin", fin.get("run_if_within_points", 5)))
        return min_seo, min_read, max_warn, allow_nm, near_miss

    def _estimate_article_metrics(self, content: str, keyphrase: str) -> Dict[str, Any]:
        words = re.findall(r"\w+(?:'\w+)?", content)
        word_count = len(words)
        est_tokens = int(word_count * 1.3) + 150  # simple proxy
        price_out = float(os.getenv("OPENAI_OUT_PRICE_PER_1K", "0") or 0)
        est_cost_usd = round((est_tokens / 1000.0) * (price_out or 0), 4)
        density_hits = len(re.findall(re.escape(keyphrase.lower()), " ".join(w.lower() for w in words)))
        keyphrase_density_pct = round(100.0 * density_hits / max(1, len(words)), 3)
        reading_time_min = round(word_count / 230.0, 2)  # ~230 wpm default
        return {
            "word_count": word_count,
            "est_tokens": est_tokens,
            "est_cost_usd": est_cost_usd,
            "keyphrase_density_pct": keyphrase_density_pct,
            "reading_time_min": reading_time_min,
        }

    # --- Per-article workflow ----------------------------------------------
    def _get_article_publisher(self) -> Optional[ArticlePublisher]:
        """
        Lazy initializer so dry-runs don't require WP creds and to avoid crashing early.
        Also passes the already-parsed settings to avoid duplicate config parsing/log spam.
        """
        if self._publisher is not None:
            return self._publisher
        try:
            self._publisher = ArticlePublisher(config_path=self.config.settings_path, settings=self.app_settings)
            return self._publisher
        except Exception as e:
            logger.error(f"Failed to initialize ArticlePublisher: {e}")
            return None

    def _process_single_article(self, article_meta: Dict) -> bool:
        topic = article_meta.get("topic")
        keyphrase = article_meta.get("keyphrase", topic)
        niche = article_meta.get("niche", "General")

        try:
            unique_keyphrase = self.keyphrase_tracker.get_unique_keyphrase(
                base_keyphrase=keyphrase, content_slug=slugify(topic)
            )
            article_meta["keyphrase"] = unique_keyphrase
            logger.info(f"Assigned unique keyphrase '{unique_keyphrase}' to topic '{topic}'.")

            seo_result = self.article_generator.generate_article(topic, unique_keyphrase, niche)
            if not seo_result:
                raise RuntimeError("Article generation failed.")

            title, slug, meta, content = self._validate_seo_result(seo_result)

            # --- NEW: Structure enforcer (before any heavy additions)
            if not self.skip_structure:
                try:
                    content, struct_info = self.structure_enforcer.enforce(
                        content=content, title=title, keyphrase=unique_keyphrase, niche=niche, slug=slug
                    )
                    if isinstance(struct_info, dict) and struct_info.get("changed"):
                        logger.info("StructureEnforcer applied: %s", ", ".join(struct_info.get("notes", [])) or "adjusted")
                except TypeError:
                    # legacy signature
                    content, _ = self.structure_enforcer.enforce(content=content)
                except Exception as e:
                    logger.warning(f"StructureEnforcer failed (non-fatal): {e}")
            else:
                logger.info("StructureEnforcer skipped via env flag.")

            # --- NEW: Technical enhancer (adds depth/examples)
            if not self.skip_technical:
                try:
                    enh = self.technical_enhancer.enhance(
                        content=content, topic=topic, keyphrase=unique_keyphrase, niche=niche, slug=slug
                    )
                    if isinstance(enh, tuple) and len(enh) >= 1:
                        content = enh[0]
                    elif isinstance(enh, str):
                        content = enh
                    logger.info("TechnicalEnhancer completed.")
                except TypeError:
                    res = self.technical_enhancer.enhance(content=content)
                    content = res[0] if isinstance(res, tuple) else res
                except Exception as e:
                    logger.warning(f"TechnicalEnhancer failed (non-fatal): {e}")
            else:
                logger.info("TechnicalEnhancer skipped via env flag.")

            # Enhancements (charts, live demos, cleanup)
            if not self.skip_charts:
                try:
                    content, _ = self.chart_generator.generate_and_embed_chart(content, slug)
                except Exception as e:
                    logger.warning(f"Chart generation failed (non-fatal): {e}")
            if not self.skip_live_demos:
                try:
                    content = self.live_demo_embedder.embed_demos(content)
                except Exception as e:
                    logger.warning(f"Live demo embedder failed (non-fatal): {e}")
            if not self.skip_cleanup:
                try:
                    content = self.code_cleaner.clean_article_content(content, slug)
                except Exception as e:
                    logger.warning(f"Code cleaner failed (non-fatal): {e}")

            # ---------- Outbound links (first-pass) ----------
            if select_outbound_links:
                try:
                    k_links = int(self.app_settings.get("internal_linking", {}).get("max_outbound_links", 2))
                    selected = select_outbound_links(
                        niche,
                        json_path=str(self.config.base_dir / "config" / "niche_outbound_links.json"),
                        k=k_links,
                        https_only=True,
                        dedupe_by_domain=True,
                        existing_text=content,
                        avoid_domains_in_text=True,
                        recent_history_path=str(self.config.base_dir / "logs" / "outbound_recent.jsonl"),
                        enforce_type_diversity=True,
                        return_format="linker_map",           # << use linker_map directly
                        linker_group_by_type=self.linker_group_by_type,
                    ) or {}

                    outbound_map: Dict[str, Any]
                    if isinstance(selected, dict) and selected:
                        outbound_map = selected
                    else:
                        outbound_map = {}

                    # Fallback if helper produced no usable links
                    if not outbound_map or not any(v for v in outbound_map.values() if isinstance(v, (list, tuple)) and v):
                        outbound_map = {
                            "fallback": [
                                "https://docs.python.org/3/",
                                "https://pypi.org/",
                            ]
                        }

                    content = self.article_linker.inject_outbound_links(content, outbound_map)
                except Exception as e:
                    logger.warning(f"Outbound link helper failed: {e}")
            else:
                logger.info("select_outbound_links helper not available; skipping outbound injection.")

            # Get knobs upfront so we can use even if Yoast skipped
            yoast_knobs = self._get_yoast_knobs()

            # --- Yoast preflight (first pass) ---
            if not self.skip_yoast:
                pre = YoastPreflight(self.app_settings, article_linker=self.article_linker)
                try:
                    content, title, meta, report = pre.fix(
                        content=content,
                        title=title,
                        meta=meta,
                        keyphrase=unique_keyphrase,
                        niche=niche,
                        **yoast_knobs,
                    )
                except TypeError as e:
                    logger.warning(f"YoastPreflight.fix() incompatible knobs ({e}). Retrying with core args only.")
                    content, title, meta, report = pre.fix(
                        content=content, title=title, meta=meta, keyphrase=unique_keyphrase, niche=niche
                    )

                thr = report.get("thresholds_used") or {}
                logger.info(
                    "Yoast preflight: SEO %s, Readability %s, H2 %s, Passive %s%%, Transitions %s%% [mode=%s, thresholds=%s].",
                    report.get('seo'), report.get('readability'), report.get('h2_count'),
                    report.get('passive_pct'), report.get('transition_pct'),
                    yoast_knobs.get("readability_mode"),
                    thr if thr else "default"
                )

                # Capture baseline BEFORE finalizer (non-regression guard)
                base_pack = (title, meta, content, report)

                # --- FINALIZER (deterministic) then re-run Yoast ---
                used_final = False
                try:
                    ft, fm, fc, freport = finalize_seo(
                        title=title,
                        meta=meta,
                        content_md=content,
                        keyphrase=unique_keyphrase,
                        knobs=yoast_knobs,
                    )
                    changed = freport.get("changed")
                    if isinstance(changed, (list, tuple, set)) and changed:
                        logger.info("Final SEO fix touched: %s", ", ".join(changed))
                    elif changed:
                        logger.info("Final SEO fix touched.")
                    title, meta, content = ft, fm, fc

                    # Now that content is longer, insert CTA/Challenge if thresholds met
                    current_word_count = len(re.findall(r"\w+(?:'\w+)?", content))
                    cta_min = self.app_settings.get("cta_injection", {}).get("min_word_count", 800)
                    if not self.skip_cta and current_word_count >= cta_min:
                        content = self.article_cta_injector.inject_cta(content, context={"article_title": title})
                        logger.info(f"CTA inserted ({current_word_count}/{cta_min}).")
                    challenge_min = self.app_settings.get("challenge_injection", {}).get("min_word_count", 1000)
                    if not self.skip_challenge and current_word_count >= challenge_min:
                        content = self.article_challenge_injector.inject_challenge(content, topic)
                        logger.info(f"Challenge inserted ({current_word_count}/{challenge_min}).")

                    # Final Yoast check
                    pre2 = YoastPreflight(self.app_settings, article_linker=self.article_linker)
                    try:
                        content, title, meta, report2 = pre2.fix(
                            content=content, title=title, meta=meta,
                            keyphrase=unique_keyphrase, niche=niche, **yoast_knobs
                        )
                    except TypeError:
                        content, title, meta, report2 = pre2.fix(
                            content=content, title=title, meta=meta,
                            keyphrase=unique_keyphrase, niche=niche
                        )
                    thr2 = report2.get("thresholds_used") or {}
                    logger.info(
                        "[After finalizer] Yoast: SEO %s, Readability %s, H2 %s, Passive %s%%, Transitions %s%% [mode=%s, thresholds=%s].",
                        report2.get('seo'), report2.get('readability'), report2.get('h2_count'),
                        report2.get('passive_pct'), report2.get('transition_pct'),
                        yoast_knobs.get("readability_mode"),
                        thr2 if thr2 else "default"
                    )

                    # Non-regression guard: pick better of (pre, post)
                    chosen_title, chosen_meta, chosen_content, chosen_report, used_final = self._choose_best_version(
                        base_pack, (title, meta, content, report2), yoast_knobs
                    )
                    if not used_final:
                        logger.info("Finalizer worsened metrics; reverting to pre-finalizer content for stability.")
                    title, meta, content, report = chosen_title, chosen_meta, chosen_content, chosen_report

                except Exception as e:
                    logger.warning(f"Final SEO fix skipped due to error: {e}")

            else:
                logger.info("Yoast preflight skipped via env flag.")
                report = {"seo": 0, "readability": 0, "warnings": 0}  # placeholder if skipped

            # --- Quality gate: fail-closed (Yoast thresholds) ----------------
            if not self.skip_yoast:
                try:
                    # thresholds now draw from settings.quality_gate with near-miss support
                    min_seo, min_read, max_warn, allow_nm, near_miss = self._get_gate_thresholds(yoast_knobs)

                    warnings_count = self._count_warnings(report, max_warn)
                    seo_val = float(report.get("seo", 0))
                    read_val = float(report.get("readability", 0))

                    seo_ok = seo_val >= min_seo
                    read_ok = read_val >= min_read
                    warn_ok = int(warnings_count) <= max_warn

                    if seo_ok and read_ok and warn_ok:
                        pass  # normal pass
                    else:
                        # near-miss logic
                        if allow_nm:
                            seo_close = seo_val >= (min_seo - near_miss)
                            read_close = read_val >= (min_read - near_miss)
                            warn_close = warnings_count <= (max_warn + 1)  # allow one extra warning on near miss
                            if seo_close and read_close and warn_close:
                                logger.warning(
                                    "Quality gate near-miss accepted → "
                                    "SEO %.1f (>=%.1f-%.1f) | Readability %.1f (>=%.1f-%.1f) | Warnings %s (<=%d+1)",
                                    seo_val, min_seo, near_miss, read_val, min_read, near_miss, warnings_count, max_warn
                                )
                            else:
                                logger.error(
                                    "Quality gate failed → SEO %.1f (>=%.1f) | Readability %.1f (>=%.1f) | Warnings %s (<=%d)",
                                    seo_val, min_seo, read_val, min_read, warnings_count, max_warn,
                                )
                                metrics = {
                                    "yoast_seo_score": seo_val,
                                    "yoast_readability_score": read_val,
                                    "warnings": warnings_count,
                                    "readability_mode": yoast_knobs.get("readability_mode"),
                                    "thresholds_used": report.get("thresholds_used"),
                                    "near_miss_margin": near_miss,
                                }
                                self._log_publication(article_meta, False, None, None, 0, "Quality gate failed", extra_metrics=metrics)
                                self._update_publishing_status(article_meta, False)
                                return False
                        else:
                            logger.error(
                                "Quality gate failed → SEO %.1f (>=%.1f) | Readability %.1f (>=%.1f) | Warnings %s (<=%d)",
                                seo_val, min_seo, read_val, min_read, warnings_count, max_warn,
                            )
                            metrics = {
                                "yoast_seo_score": seo_val,
                                "yoast_readability_score": read_val,
                                "warnings": warnings_count,
                                "readability_mode": yoast_knobs.get("readability_mode"),
                                "thresholds_used": report.get("thresholds_used"),
                                "near_miss_allowed": False,
                            }
                            self._log_publication(article_meta, False, None, None, 0, "Quality gate failed", extra_metrics=metrics)
                            self._update_publishing_status(article_meta, False)
                            return False
                except Exception as qe:
                    logger.error(f"Quality gate crashed; blocking publish for safety: {qe}", exc_info=True)
                    self._log_publication(article_meta, False, None, None, 0, "Quality gate exception")
                    self._update_publishing_status(article_meta, False)
                    return False

            # --- NEW: QualityGate module (additional checks) -----------------
            if not self.skip_quality_gate:
                try:
                    ok, reasons, extra = self.quality_gate.evaluate(
                        title=title, meta=meta, content=content,
                        yoast_report=report, niche=niche, keyphrase=unique_keyphrase, topic=topic
                    )
                    if not ok:
                        logger.error("QualityGate module blocked publish: %s", "; ".join(reasons) or "no reasons provided")
                        self._log_publication(article_meta, False, None, None, 0, "QualityGate rejection", extra_metrics=extra or {})
                        self._update_publishing_status(article_meta, False)
                        return False
                    else:
                        logger.info("QualityGate module passed.")
                except Exception as e:
                    # Fail-closed for safety
                    logger.error(f"QualityGate crashed; blocking publish for safety: {e}", exc_info=True)
                    self._log_publication(article_meta, False, None, None, 0, "QualityGate exception")
                    self._update_publishing_status(article_meta, False)
                    return False
            else:
                logger.info("QualityGate module skipped via env flag.")

            # Save locally (atomic)
            article_dir = self.config.base_dir / self.config.output_dir / slug
            article_dir.mkdir(parents=True, exist_ok=True)
            article_filepath = article_dir / self.config.article_filename
            _save_text_atomic(article_filepath, f"# {title}\n\n{content}")

            # Metadata — keep in sync with final content (atomic)
            if hasattr(seo_result, "to_dict"):
                metadata = seo_result.to_dict()
            else:
                metadata = {
                    "title": title,
                    "slug": slug,
                    "meta_description": meta,
                    "keyphrase": unique_keyphrase
                }
            metadata["content_length"] = len(re.findall(r"\w+(?:'\w+)?", content))
            metadata["yoast"] = report
            metadata["yoast_mode"] = yoast_knobs.get("readability_mode")
            metadata["yoast_thresholds_used"] = report.get("thresholds_used")
            canonical = self._build_canonical_url(slug)
            if canonical:
                metadata["url"] = canonical

            _save_json_atomic(article_dir / self.config.metadata_filename, metadata)

            # Internal linking
            if not self.skip_internal_links:
                try:
                    all_articles_map = self.article_linker.get_articles_by_niche(self.config.base_dir / self.config.output_dir)
                    self.article_linker.inject_internal_links(all_articles_map)
                except Exception as e:
                    logger.warning(f"Internal linking step failed (non-fatal): {e}")
            else:
                logger.info("Internal linking skipped via env flag.")

            # Respect publish window; optionally store locally only
            if not self._within_publish_window() and not self.dry_run:
                logger.info("Outside publish window; saved locally and skipping WordPress publish for this article.")
                metrics = self._estimate_article_metrics(content, unique_keyphrase)
                self._log_publication(article_meta, True, None, metadata.get("url"), 0, None, extra_metrics=metrics, skipped_reason="outside_publish_window")
                # DO NOT mark as published in schedule; we’ll pick it up later if desired
                return True

            # Optional dry-run (skip publishing)
            if self.dry_run:
                logger.info("[DRY_RUN] Skipping publish; article saved locally.")
                metrics = self._estimate_article_metrics(content, unique_keyphrase)
                self._log_publication(article_meta, True, None, metadata.get("url"), 0, None, extra_metrics=metrics, skipped_reason="dry_run")
                return True

            # Publish with retries
            publisher = self._get_article_publisher()
            if not publisher:
                err = "Publisher not initialized (missing/invalid WP credentials?)"
                logger.error(err)
                self._log_publication(article_meta, False, None, None, 0, err)
                self._update_publishing_status(article_meta, False)
                return False

            for attempt in range(1, self.config.max_publish_retries + 1):
                try:
                    logger.info(f"Publishing '{title}' (Attempt {attempt})...")
                    post_id, url = publisher.publish_article(article_filepath)
                    if post_id and url:
                        self.successfully_published_articles.append({
                            "title": title,
                            "url": url,
                            "keyphrase": unique_keyphrase,
                            "slug": slug
                        })
                        metrics = self._estimate_article_metrics(content, unique_keyphrase)
                        self._log_publication(article_meta, True, post_id, url, attempt, None, extra_metrics=metrics)
                        self._update_publishing_status(article_meta, True)

                        try:
                            if getattr(self.google_indexer, "enabled", False) and url:
                                logger.info(f"Requesting Google indexing for URL: {url}")
                                self.google_indexer.publish_url(url)
                            else:
                                logger.info("Google Indexer disabled or not configured. Skipping indexing request.")
                        except Exception as e:
                            logger.warning(f"Google indexing request failed (non-fatal): {e}")
                        return True
                except Exception as e:
                    if attempt < self.config.max_publish_retries:
                        base = self.config.retry_delay_base_seconds * (2 ** (attempt - 1))
                        delay = base + random.uniform(0, base * 0.25)
                        logger.warning(f"Publish attempt {attempt} failed: {e}. Retrying in {delay:.1f}s...")
                        self._safe_sleep(delay)
                        if self._stop_event.is_set():
                            logger.warning("Stop requested. Aborting remaining retries for this article.")
                            break
                    else:
                        self._log_error(f"Publishing failed for '{title}': {e}", critical=True)
                        self._log_publication(article_meta, False, None, None, attempt, str(e))
                        self._update_publishing_status(article_meta, False)
                        return False

            return False

        except Exception as e:
            full_trace = traceback.format_exc()
            self._log_error(
                f"Critical error during article processing for topic '{topic}': {e}\nTraceback:\n{full_trace}",
                critical=True,
            )
            self._update_publishing_status(article_meta, False)
            self._log_publication(article_meta, False, None, None, 1, str(e))
            return False

    # --- Orchestration ------------------------------------------------------
    def execute_daily_workflow(self):
        logger.info(f"--- Starting Workflow for {(self.run_date_override or datetime.now().strftime('%Y-%m-%d'))} ---")

        # Optional: sanity check WordPress before we invest in generation
        try:
            if not self.dry_run and not self.skip_wp_test:
                publisher = self._get_article_publisher()
                if publisher and hasattr(publisher, "test_connection"):
                    publisher.test_connection()
        except Exception:
            logger.warning("WordPress connection pre-check failed; continuing (will retry on publish).")

        articles_to_process = self._load_publishing_schedule()
        if not articles_to_process:
            logger.info("No articles scheduled to process.")
            return

        for i, article_meta in enumerate(articles_to_process):
            if self._stop_event.is_set():
                logger.warning("Stop requested. Ending workflow early.")
                break
            self._check_cpu_and_wait()
            logger.info(f"Processing {i + 1}/{len(articles_to_process)}: '{article_meta.get('topic')}'")
            self._process_single_article(article_meta)
            if i < len(articles_to_process) - 1 and not self._stop_event.is_set():
                low, high = self.config.inter_article_delay_range_seconds
                delay = random.uniform(float(low), float(high))
                logger.info(f"Waiting {delay:.2f}s before next article...")
                self._safe_sleep(delay)

        # Newsletter
        if not self.skip_newsletter:
            min_needed = self.app_settings.get("newsletter", {}).get("min_articles_for_newsletter", 1)
            count = len(self.successfully_published_articles)
            if count >= int(min_needed):
                logger.info(f"Sending newsletter with {count} new article(s).")
                try:
                    self.newsletter_sender.send_digest_email(self.successfully_published_articles)
                except Exception as e:
                    self._log_error(f"Newsletter send failed: {e}")
            else:
                logger.info(f"Not enough articles published ({count}) for newsletter. Min required: {min_needed}")
        else:
            logger.info("Newsletter step skipped via env flag.")


if __name__ == "__main__":
    # Force CWD to project root when running as a script
    os.chdir(Path(__file__).parent.parent)
    logger.debug(f"DEBUG: CWD set to project root: {os.getcwd()}")

    # Optional deterministic randomness
    seed_env = os.getenv("SEED")
    if seed_env and seed_env.isdigit():
        random.seed(int(seed_env))

    # Lock config (env overrides supported)
    lock_path_env = os.getenv("LOCK_FILE")
    lock_path = Path(lock_path_env) if lock_path_env else (LOG_DIR / "daily_automation.lock")
    # accept either LOCK_TTL_MIN or legacy LOCK_TTL_MINUTES
    lock_ttl = int(os.getenv("LOCK_TTL_MIN", os.getenv("LOCK_TTL_MINUTES", "45")))
    lock_force = _parse_bool(os.getenv("LOCK_FORCE"))

    automation = None
    try:
        with RunLock(lock_path=lock_path, ttl_minutes=lock_ttl, force=lock_force):
            automation = DailyAutomation()
            if automation.dry_run:
                logger.info("[DRY_RUN] Running in dry-run mode (no WordPress publishing).")
            automation.execute_daily_workflow()
    except SystemExit as se:
        logger.error(str(se))
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Fatal error during automation run: {e}", exc_info=True)
        if automation and hasattr(automation, "error_notifier") and getattr(automation.error_notifier, "_is_initialized", False):
            error_subject = f"CRITICAL: Automation Fatal Error on {datetime.now().strftime('%Y-%m-%d')}"
            error_body = (
                "A fatal error occurred during the daily automation run.\n\n"
                f"Error: {e}\n\nTraceback:\n{traceback.format_exc()}"
            )
            try:
                automation.error_notifier.send_error_email(
                    error_summary=error_body,
                    subject=error_subject,
                    priority=EmailPriority.CRITICAL,
                )
            except Exception as email_e:
                logger.error(f"Failed to send critical error email: {email_e}", exc_info=True)
        sys.exit(1)
