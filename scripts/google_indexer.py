# google_indexer.py

import os
import sys
import json
import time
import random
import logging
from pathlib import Path
from typing import Dict, Optional, Iterable, Tuple, List, Any

import requests
from google.oauth2 import service_account
from google.auth.transport.requests import Request as GARequest
from urllib.parse import urlparse, urlencode

# Ensure logs/cache dirs exist before wiring handlers
Path("logs").mkdir(parents=True, exist_ok=True)
Path("cache").mkdir(parents=True, exist_ok=True)

# Configure logging (don’t duplicate handlers if a root config already exists)
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("logs/google_indexer.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
logger = logging.getLogger(__name__)


def safe_json(resp: requests.Response) -> Optional[dict]:
    try:
        return resp.json()
    except Exception:
        return None


class GoogleIndexer:
    """
    Lightweight Google Indexing workflow with:
      • Service-account auth (JSON key file or inline JSON via env).
      • URL ownership guard (same origin as property).
      • Optional robots.txt check (User-agent: *) with on-disk cache.
      • Dry-run mode (logs only).
      • Bulk publishing with polite jitter and min-interval throttle.
      • Backoff queue (cache/index_queue.jsonl) with attempts + next_ts.
      • Optional sitemap pings (Google & Bing) as a complementary signal.

    settings.json -> google_indexing:
    {
      "enabled": true,
      "service_account_file": "/abs/path/to/key.json",      // or "${GOOGLE_SERVICE_ACCOUNT_FILE}"
      "service_account_json": "${GOOGLE_SERVICE_ACCOUNT_JSON}", // raw JSON in env (alternative to file)
      "property_url": "https://example.com/",
      "dry_run": false,
      "jitter_ms": 250,
      "min_interval_s": 0.8,                // NEW: polite minimum interval per request
      "retries": 2,
      "timeout_s": 20,
      "queue_file": "cache/index_queue.jsonl",
      "queue_max_size": 20000,              // NEW: simple cap
      "dedupe_queue": true,                 // NEW
      "retry_schedule_s": [600, 3600, 14400, 86400],  // NEW: 10m, 1h, 4h, 24h
      "sitemaps": ["https://example.com/sitemap.xml"],
      "ping_sitemaps_after_publish": true,
      "check_robots_txt": true,             // NEW
      "robots_cache_ttl_s": 1800            // NEW: 30 minutes
    }

    Notes:
      • publish_url(url) sends type="URL_UPDATED" by default.
      • The Indexing API is officially intended for certain content types.
        We use it deliberately and reinforce with sitemap pings.
    """

    SCOPE = "https://www.googleapis.com/auth/indexing"
    ENDPOINT = "https://indexing.googleapis.com/v3/urlNotifications:publish"

    def __init__(self, app_settings: Dict):
        self.app_settings = app_settings or {}
        cfg = self.app_settings.get("google_indexing", {}) or {}

        # --- Config ---
        self.enabled: bool = bool(cfg.get("enabled", False))
        self.dry_run: bool = bool(cfg.get("dry_run", False))
        self.jitter_ms: float = float(cfg.get("jitter_ms", 250))
        self.min_interval_s: float = float(cfg.get("min_interval_s", 0.0))
        self.retries: int = int(cfg.get("retries", 2))
        self.timeout_s: int = int(cfg.get("timeout_s", 20))

        # queue/backoff
        self.queue_file: str = str(self._resolve_env_placeholder(cfg.get("queue_file", "cache/index_queue.jsonl")))
        self.queue_max_size: int = int(cfg.get("queue_max_size", 20000))
        self.dedupe_queue: bool = bool(cfg.get("dedupe_queue", True))
        self.retry_schedule_s: List[int] = list(cfg.get("retry_schedule_s", [600, 3600, 14400, 86400]))  # 10m,1h,4h,24h

        # sitemaps
        self.sitemaps: List[str] = list(cfg.get("sitemaps", []) or [])
        self.ping_sitemaps_after_publish: bool = bool(cfg.get("ping_sitemaps_after_publish", True))

        # robots
        self.check_robots_txt: bool = bool(cfg.get("check_robots_txt", True))
        self.robots_cache_ttl_s: int = int(cfg.get("robots_cache_ttl_s", 1800))
        self._robots_cache_dir = Path("cache/robots")
        self._robots_cache_dir.mkdir(parents=True, exist_ok=True)

        # property / credentials
        self.property_url_raw: Optional[str] = self._resolve_env_placeholder(cfg.get("property_url"))
        self.service_account_file: Optional[str] = self._resolve_env_placeholder(cfg.get("service_account_file"))
        self.service_account_json: Optional[str] = self._resolve_env_placeholder(cfg.get("service_account_json"))

        self._credentials: Optional[service_account.Credentials] = None
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "pythonprohub-indexer/1.1"})
        self._token_last_refresh_ts: float = 0.0
        self._last_call_ts: float = 0.0  # polite min-interval throttle

        # Validate/normalize property origin
        if not self.enabled:
            logger.info("Google Indexing is disabled in settings.")
            self.property_scheme = self.property_host = ""
            return

        try:
            self.property_scheme, self.property_host = self._normalize_origin(self.property_url_raw or "")
        except ValueError:
            logger.critical("Invalid or missing property_url in google_indexing settings. Disabling.")
            self.enabled = False
            self.property_scheme = self.property_host = ""
            return

        # Load credentials (unless dry-run)
        if self.dry_run:
            logger.info("Indexer is in DRY-RUN mode (no API calls will be sent).")
        else:
            if not self._load_credentials():
                self.enabled = False
                logger.critical("Credentials not loaded; indexer disabled.")

        if not self.sitemaps and self.property_url_raw:
            # Reasonable default
            self.sitemaps = [self.property_url_raw.rstrip("/") + "/sitemap.xml"]

        logger.info(
            "Google Indexer initialized. enabled=%s dry_run=%s property=%s",
            self.enabled, self.dry_run, self.property_url_raw
        )

    # ----------------------------- Public API ------------------------------

    def publish_url(self, url: str, *, type: str = "URL_UPDATED") -> bool:
        """
        Request indexing (or re-indexing) for a URL.
        Returns True on HTTP 200 success, False otherwise.
        """
        if not url or not isinstance(url, str):
            logger.error("publish_url called with an invalid URL.")
            return False

        if not self._is_within_property(url):
            logger.warning("URL outside property: %s (property=%s)", url, self.property_url_raw)
            return False

        if self.check_robots_txt and not self._robots_allows(url):
            logger.info("Robots.txt disallows indexing request for: %s (skipping)", url)
            return False

        if self.dry_run or not self.enabled:
            logger.info("[DRY] Would request indexing for: %s (%s)", url, type)
            self._maybe_queue(url, type=type)  # keep parity; you can flush later when enabled
            return True

        payload = {"url": url, "type": type}
        ok, status, text = self._request_with_retries(payload)
        if ok:
            notify_time = ""
            try:
                data = json.loads(text) if text else {}
                meta = (data or {}).get("urlNotificationMetadata", {})
                latest = meta.get("latestUpdate") or meta.get("latestNotify")
                notify_time = (latest or {}).get("notifyTime", "")
            except Exception:
                pass
            logger.info("Indexing request OK for %s%s", url, f" (notifyTime: {notify_time})" if notify_time else "")
            return True

        # Not ok → queue for later retry with backoff
        snippet = (text or "")[:500] if isinstance(text, str) else str(text)[:500]
        logger.error("Indexing request failed [%s] for %s: %s", status, url, snippet)
        self._maybe_queue(url, type=type, last_status=status, last_error=snippet)
        return False

    def publish_urls(
        self,
        urls: Iterable[str],
        *,
        type: str = "URL_UPDATED",
        sleep_range: Tuple[float, float] = (0.8, 1.6),
    ) -> Dict[str, bool]:
        """
        Convenience bulk publisher with polite jitter between calls.
        Returns a mapping: url -> success_bool.
        """
        results: Dict[str, bool] = {}
        for u in urls or []:
            results[u] = self.publish_url(u, type=type)
            time.sleep(random.uniform(*sleep_range))
        # Optionally ping sitemaps once after a batch
        if self.ping_sitemaps_after_publish and any(results.values()):
            self.ping_sitemaps()
        return results

    def flush_queue(self, max_items: int = 100) -> int:
        """
        Try to send queued URLs stored on disk (from earlier failures/disabled runs).
        Returns the number of successfully sent items.
        """
        qpath = Path(self.queue_file)
        if not qpath.exists():
            return 0

        lines = [ln for ln in qpath.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if not lines:
            return 0

        now = int(time.time())
        kept: List[str] = []
        ready_items: List[Dict[str, Any]] = []
        processed = 0
        sent_ok = 0

        for line in lines:
            try:
                item = json.loads(line)
            except Exception:
                # Keep unparsable lines to avoid data loss
                kept.append(line)
                continue

            next_ts = int(item.get("next_ts") or 0)
            if next_ts and next_ts > now:
                kept.append(line)
                continue

            ready_items.append(item)

        for item in ready_items[:max_items]:
            url = str(item.get("url", "")).strip()
            typ = str(item.get("type", "URL_UPDATED")).strip() or "URL_UPDATED"
            attempts = int(item.get("attempts", 0))

            ok = self.publish_url(url, type=typ)
            processed += 1

            if ok:
                sent_ok += 1
                continue

            # Re-queue with backoff
            attempts += 1
            backoff = self._backoff_for_attempt(attempts)
            item["attempts"] = attempts
            item["next_ts"] = int(time.time() + backoff)
            kept.append(json.dumps(item, ensure_ascii=False))

        # Preserve leftover unprocessed items and those not yet ready
        # Add tail of ready_items that exceeded max_items
        for item in ready_items[max_items:]:
            kept.append(json.dumps(item, ensure_ascii=False))

        # Rewrite queue atomically
        if kept:
            Path(self.queue_file).write_text("\n".join(kept) + "\n", encoding="utf-8")
        else:
            Path(self.queue_file).unlink(missing_ok=True)

        logger.info("Queue flush complete. sent_ok=%d kept=%d processed=%d", sent_ok, len(kept), processed)
        return sent_ok

    def ping_sitemaps(self) -> None:
        """
        Complementary hint to search engines by pinging sitemap URLs.
        Works even when Indexing API is disabled.
        """
        if not self.sitemaps:
            return
        for sm in self.sitemaps:
            self._ping_google_sitemap(sm)
            self._ping_bing_sitemap(sm)

    # ---------------------------- Internals --------------------------------

    def _respect_min_interval(self) -> None:
        if self.min_interval_s <= 0:
            return
        now = time.time()
        elapsed = now - self._last_call_ts
        remaining = self.min_interval_s - elapsed
        if remaining > 0:
            time.sleep(remaining)
        self._last_call_ts = time.time()

    def _request_with_retries(self, payload: Dict[str, str]) -> Tuple[bool, int, str]:
        """
        Do the POST with token refresh + retries on 429/5xx.
        Returns (ok, status_code, response_text).
        """
        self._ensure_bearer()

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._credentials.token}",  # type: ignore[union-attr]
        }

        attempts = max(1, int(self.retries) + 1)  # e.g., retries=2 -> attempts: 3
        for attempt in range(1, attempts + 1):
            try:
                self._respect_min_interval()
                resp = self._session.post(self.ENDPOINT, headers=headers, json=payload, timeout=self.timeout_s)

                if resp.status_code == 401:
                    # Token may be stale; refresh once immediately
                    logger.warning("401 from Indexing API; attempting token refresh.")
                    self._force_refresh_bearer()
                    headers["Authorization"] = f"Bearer {self._credentials.token}"  # type: ignore[union-attr]
                    self._respect_min_interval()
                    resp = self._session.post(self.ENDPOINT, headers=headers, json=payload, timeout=self.timeout_s)

                if resp.status_code == 200:
                    return True, resp.status_code, resp.text

                # Retry on 429 & 5xx
                if resp.status_code in (429, 500, 502, 503, 504) and attempt <= self.retries:
                    wait = (2 ** (attempt - 1)) + random.uniform(0.0, 0.5)
                    logger.warning("Indexing API %s; retrying in %.2fs (attempt %d/%d)",
                                   resp.status_code, wait, attempt, attempts)
                    time.sleep(wait)
                    continue

                # Definitive failure
                return False, resp.status_code, resp.text

            except requests.RequestException as e:
                if attempt <= self.retries:
                    wait = (2 ** (attempt - 1)) + random.uniform(0.0, 0.5)
                    logger.warning("Network error calling Indexing API: %s. Retrying in %.2fs", e, wait)
                    time.sleep(wait)
                    continue
                logger.error("Error calling Indexing API: %s", e, exc_info=True)
                return False, 0, str(e)
            finally:
                # Gentle jitter between attempts
                time.sleep(random.uniform(0, max(0.0, self.jitter_ms / 1000.0)))

        return False, 0, "Unknown error"

    def _ensure_bearer(self) -> None:
        if not self._credentials:
            raise RuntimeError("Credentials not initialized")
        # Refresh if invalid or token older than ~45 minutes (defensive)
        if (not self._credentials.valid) or (time.time() - self._token_last_refresh_ts > 45 * 60):
            self._force_refresh_bearer()

    def _force_refresh_bearer(self) -> None:
        assert self._credentials is not None
        try:
            self._credentials.refresh(GARequest())
            self._token_last_refresh_ts = time.time()
        except Exception as e:
            logger.error("Failed to refresh Google credentials: %s", e, exc_info=True)
            raise

    def _load_credentials(self) -> bool:
        """
        Load credentials from file; if not provided or missing, try inline JSON.
        """
        # Prefer file if present
        if self.service_account_file:
            if not os.path.exists(self.service_account_file):
                logger.critical("Service account file not found: %s", self.service_account_file)
                return False
            try:
                self._credentials = service_account.Credentials.from_service_account_file(
                    self.service_account_file,
                    scopes=[self.SCOPE],
                )
                return True
            except Exception as e:
                logger.critical("Failed to load Google credentials from file: %s", e, exc_info=True)
                return False

        # Else try inline JSON (string)
        sa_json = (self.service_account_json or "").strip()
        if sa_json:
            try:
                info = json.loads(sa_json)
                self._credentials = service_account.Credentials.from_service_account_info(
                    info,
                    scopes=[self.SCOPE],
                )
                return True
            except Exception as e:
                logger.critical("Failed to load Google credentials from inline JSON: %s", e, exc_info=True)
                return False

        logger.critical("No service_account_file or service_account_json provided.")
        return False

    # ---------------------------- Utilities ---------------------------------

    def _is_within_property(self, url: str) -> bool:
        if not self.property_scheme or not self.property_host:
            return False
        try:
            sch, host = self._normalize_origin(url)
        except ValueError:
            return False
        ph = self.property_host[4:] if self.property_host.startswith("www.") else self.property_host
        uh = host[4:] if host.startswith("www.") else host
        return (sch == self.property_scheme) and (ph == uh)

    @staticmethod
    def _normalize_origin(url: str) -> Tuple[str, str]:
        """
        Returns (scheme, netloc). Raises ValueError if invalid.
        """
        p = urlparse(url or "")
        if not p.scheme or not p.netloc:
            raise ValueError("Invalid URL for origin normalization.")
        return p.scheme.lower(), p.netloc.lower()

    @staticmethod
    def _resolve_env_placeholder(value: Optional[str]) -> Optional[str]:
        """
        Supports values like "${ENV_NAME}" -> os.getenv("ENV_NAME").
        """
        if value is None:
            return None
        if not isinstance(value, str):
            return value
        v = value.strip()
        if v.startswith("${") and v.endswith("}"):
            return os.getenv(v[2:-1])
        return v

    # ------------------------- Queue & Pings --------------------------------

    def _backoff_for_attempt(self, attempts: int) -> int:
        if attempts <= 0:
            return 0
        idx = min(attempts - 1, len(self.retry_schedule_s) - 1)
        return int(self.retry_schedule_s[idx])

    def _queue_load_map(self) -> Dict[Tuple[str, str], Dict[str, Any]]:
        """
        Load queue into a map keyed by (url, type) for easy de-duplication.
        """
        qpath = Path(self.queue_file)
        items: Dict[Tuple[str, str], Dict[str, Any]] = {}
        if not qpath.exists():
            return items
        for ln in qpath.read_text(encoding="utf-8").splitlines():
            if not ln.strip():
                continue
            try:
                item = json.loads(ln)
                url = str(item.get("url", "")).strip()
                typ = str(item.get("type", "URL_UPDATED")).strip() or "URL_UPDATED"
                if url:
                    items[(url, typ)] = item
            except Exception:
                # Skip bad lines
                continue
        return items

    def _queue_write_map(self, items: Dict[Tuple[str, str], Dict[str, Any]]) -> None:
        qpath = Path(self.queue_file)
        qpath.parent.mkdir(parents=True, exist_ok=True)
        lines = [json.dumps(v, ensure_ascii=False) for v in items.values()]
        qpath.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    def _maybe_queue(
        self,
        url: str,
        *,
        type: str = "URL_UPDATED",
        last_status: Optional[int] = None,
        last_error: Optional[str] = None,
    ) -> None:
        """
        Persist a JSON record we can retry later via flush_queue(). De-dupes by (url,type).
        """
        try:
            items = self._queue_load_map() if self.dedupe_queue else None
            key = (url, type)
            if items is not None:
                if len(items) >= self.queue_max_size and key not in items:
                    logger.warning("Queue at capacity (%d); dropping new entry.", self.queue_max_size)
                    return
                item = items.get(key, {
                    "url": url,
                    "type": type,
                    "ts": int(time.time()),
                    "attempts": 0,
                    "next_ts": 0
                })
                item["last_status"] = last_status
                if last_error:
                    item["last_error"] = str(last_error)[:300]
                items[key] = item
                self._queue_write_map(items)
            else:
                # Append mode (no dedupe)
                rec = {
                    "url": url,
                    "type": type,
                    "ts": int(time.time()),
                    "attempts": 0,
                    "next_ts": 0,
                    "last_status": last_status,
                }
                with open(self.queue_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning("Failed to queue URL for later retry: %s", e)

    def _ping_google_sitemap(self, sitemap_url: str) -> None:
        try:
            ping = "https://www.google.com/ping?" + urlencode({"sitemap": sitemap_url})
            r = self._session.get(ping, timeout=self.timeout_s)
            if 200 <= r.status_code < 400:
                logger.info("Pinged Google sitemap: %s", sitemap_url)
            else:
                logger.warning("Google sitemap ping failed (%s): %s", r.status_code, sitemap_url)
        except Exception as e:
            logger.warning("Google sitemap ping error: %s", e)

    def _ping_bing_sitemap(self, sitemap_url: str) -> None:
        try:
            ping = "https://www.bing.com/ping?" + urlencode({"sitemap": sitemap_url})
            r = self._session.get(ping, timeout=self.timeout_s)
            if 200 <= r.status_code < 400:
                logger.info("Pinged Bing sitemap: %s", sitemap_url)
            else:
                logger.warning("Bing sitemap ping failed (%s): %s", r.status_code, sitemap_url)
        except Exception as e:
            logger.warning("Bing sitemap ping error: %s", e)

    # ------------------------- robots.txt (naive) ---------------------------

    def _robots_allows(self, url: str) -> bool:
        """
        Very small robots.txt parser that considers only the User-agent: * group.
        Returns True if no disallow matches the URL path or an Allow is more specific.
        """
        try:
            sch, host = self._normalize_origin(url)
        except ValueError:
            return False
        path = urlparse(url).path or "/"
        rules = self._robots_rules_for_host(sch, host)
        if not rules:
            return True  # no robots available → permissive

        # Longest-match wins (Allow overrides Disallow if longer)
        allow = [r for r in rules if r[0] == "allow"]
        disallow = [r for r in rules if r[0] == "disallow"]

        best_allow = max((len(p) for _, p in allow if path.startswith(p)), default=-1)
        best_dis = max((len(p) for _, p in disallow if path.startswith(p)), default=-1)
        return best_allow >= best_dis

    def _robots_rules_for_host(self, scheme: str, host: str) -> List[Tuple[str, str]]:
        cache_key = f"{scheme}://{host}"
        cache_path = self._robots_cache_dir / (host.replace(":", "_") + ".json")
        now = time.time()

        # Load cache
        if cache_path.exists():
            try:
                obj = json.loads(cache_path.read_text(encoding="utf-8"))
                if now - float(obj.get("ts", 0)) <= self.robots_cache_ttl_s:
                    return obj.get("rules", [])
            except Exception:
                pass

        # Fetch robots.txt
        url = f"{scheme}://{host}/robots.txt"
        rules: List[Tuple[str, str]] = []
        try:
            r = self._session.get(url, timeout=min(self.timeout_s, 10))
            if r.status_code >= 400 or not r.text:
                # Cache negative result shortly to avoid hammering
                cache_path.write_text(json.dumps({"ts": now, "rules": []}), encoding="utf-8")
                return []
            ua_any = False
            active = False
            for raw in r.text.splitlines():
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split(":", 1)
                if len(parts) != 2:
                    continue
                k = parts[0].strip().lower()
                v = parts[1].strip()
                if k == "user-agent":
                    ua_any = (v == "*")
                    active = ua_any
                elif k == "allow" and active:
                    rules.append(("allow", v if v.startswith("/") else "/" + v))
                elif k == "disallow" and active:
                    # Empty Disallow means allow all
                    if v == "":
                        continue
                    rules.append(("disallow", v if v.startswith("/") else "/" + v))
                elif k == "sitemap":
                    # ignored here
                    pass
                # New user-agent section without '*' ends our active block
                if k == "user-agent" and v != "*":
                    active = False
            cache_path.write_text(json.dumps({"ts": now, "rules": rules}), encoding="utf-8")
        except Exception:
            # Cache empty on failure to avoid repeated calls
            try:
                cache_path.write_text(json.dumps({"ts": now, "rules": []}), encoding="utf-8")
            except Exception:
                pass
            return []
        return rules


# -------------------------- CLI --------------------------------------------

def _load_env_settings() -> Dict[str, Any]:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(usecwd=True))
    return {
        "google_indexing": {
            "enabled": os.getenv("INDEXER_ENABLED", "true").lower() in ("1", "true", "yes"),
            "dry_run": os.getenv("INDEXER_DRY_RUN", "false").lower() in ("1", "true", "yes"),
            "service_account_file": os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE", "") or None,
            "service_account_json": os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON", "") or None,
            "property_url": os.getenv("GOOGLE_PROPERTY_URL", "https://example.com/"),
            "sitemaps": [os.getenv("SITEMAP_URL", "")] if os.getenv("SITEMAP_URL") else [],
            "ping_sitemaps_after_publish": os.getenv("PING_SITEMAPS", "true").lower() in ("1", "true", "yes"),
            "jitter_ms": float(os.getenv("INDEXER_JITTER_MS", "250")),
            "min_interval_s": float(os.getenv("INDEXER_MIN_INTERVAL_S", "0.8")),
            "retries": int(os.getenv("INDEXER_RETRIES", "2")),
            "timeout_s": int(os.getenv("INDEXER_TIMEOUT_S", "20")),
            "queue_file": os.getenv("INDEXER_QUEUE_FILE", "cache/index_queue.jsonl"),
            "queue_max_size": int(os.getenv("INDEXER_QUEUE_MAX", "20000")),
            "dedupe_queue": os.getenv("INDEXER_DEDUPE_QUEUE", "1") not in ("0", "false", "no"),
            "retry_schedule_s": json.loads(os.getenv("INDEXER_RETRY_SCHEDULE", "[600,3600,14400,86400]")),
            "check_robots_txt": os.getenv("INDEXER_CHECK_ROBOTS", "1") not in ("0", "false", "no"),
            "robots_cache_ttl_s": int(os.getenv("INDEXER_ROBOTS_TTL_S", "1800")),
        }
    }


def _print_cli_help() -> None:
    print(
        "Usage:\n"
        "  python google_indexer.py publish <url> [URL_DELETED]\n"
        "  python google_indexer.py batch <path-to-file|-> [URL_DELETED]\n"
        "    (file should contain one URL per line; use '-' for stdin)\n"
        "  python google_indexer.py flush [max_items]\n"
        "  python google_indexer.py ping\n"
    )


def _cli() -> int:
    settings = _load_env_settings()
    indexer = GoogleIndexer(settings)

    if len(sys.argv) < 2:
        _print_cli_help()
        return 2

    cmd = sys.argv[1].lower()
    if cmd == "publish":
        if len(sys.argv) < 3:
            _print_cli_help()
            return 2
        url = sys.argv[2]
        typ = "URL_UPDATED"
        if len(sys.argv) >= 4 and sys.argv[3].upper() in ("URL_UPDATED", "URL_DELETED"):
            typ = sys.argv[3].upper()
        ok = indexer.publish_url(url, type=typ)
        logger.info("Publish result for %s: %s", url, ok)
        if indexer.ping_sitemaps_after_publish and ok:
            indexer.ping_sitemaps()
        return 0

    if cmd == "batch":
        if len(sys.argv) < 3:
            _print_cli_help()
            return 2
        src = sys.argv[2]
        typ = "URL_UPDATED"
        if len(sys.argv) >= 4 and sys.argv[3].upper() in ("URL_UPDATED", "URL_DELETED"):
            typ = sys.argv[3].upper()
        if src == "-":
            lines = [ln.strip() for ln in sys.stdin.read().splitlines() if ln.strip()]
        else:
            lines = [ln.strip() for ln in Path(src).read_text(encoding="utf-8").splitlines() if ln.strip()]
        res = indexer.publish_urls(lines, type=typ)
        ok_count = sum(1 for v in res.values() if v)
        logger.info("Batch done: %d/%d OK", ok_count, len(res))
        return 0

    if cmd == "flush":
        max_items = int(sys.argv[2]) if len(sys.argv) >= 3 else 100
        sent = indexer.flush_queue(max_items=max_items)
        logger.info("Flushed queue: sent_ok=%d", sent)
        return 0

    if cmd == "ping":
        indexer.ping_sitemaps()
        return 0

    _print_cli_help()
    return 2


# -------------------------- CLI smoke test ---------------------------------

if __name__ == "__main__":
    sys.exit(_cli())
