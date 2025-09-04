# error_emailer.py

import os
import re
import json
import html
import time
import base64
import hashlib
import traceback
import logging
import platform
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime
from pathlib import Path

import requests
from dotenv import load_dotenv, find_dotenv

# ---------------------------------------------------------------------------
# Logging (create logs dir; don't double-add handlers if already configured)
# ---------------------------------------------------------------------------
LOG_DIR = Path("logs")
CACHE_DIR = Path("cache")
for p in (LOG_DIR, CACHE_DIR):
    p.mkdir(parents=True, exist_ok=True)

_root = logging.getLogger()
if not _root.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(LOG_DIR / "error_emailer.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------
class EmailPriority(Enum):
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"
    CRITICAL = "critical"   # CHANGE: support explicit CRITICAL


@dataclass
class ErrorNotifierConfig:
    # Brevo (Sendinblue)
    api_key: Optional[str] = None
    api_endpoint: str = "https://api.brevo.com/v3/smtp/email"

    # Sender/recipients
    sender_name: str = "Python Pro Hub Automation"
    sender_email: str = "automation@pythonprohub.com"
    recipient_email: str | List[str] = "alerts@pythonprohub.com"  # accepts list or comma string
    cc: str | List[str] | None = None
    bcc: str | List[str] | None = None
    reply_to: Optional[str] = None

    # Priority-based recipient overrides (keys: 'HIGH'|'NORMAL'|'LOW'|'CRITICAL')
    priority_overrides_to: Dict[str, str | List[str]] = field(default_factory=dict)

    # Subject & branding
    subject_prefix: Optional[str] = None
    subject_suffix: Optional[str] = None
    service_name_env_var: str = "SERVICE_NAME"

    # Behavior
    timeout: int = 12
    dry_run: bool = False
    dedupe_seconds: int = 300        # suppress identical messages within window
    max_summary_chars: int = 6000
    rate_limit_per_min: int = 20     # basic per-process throttle

    # Ignore/silence patterns (regex strings, case-insensitive)
    ignore_patterns: List[str] = field(default_factory=list)

    # Persistence
    dedupe_store: Path = CACHE_DIR / "error_email_dedupe.json"
    ratelimit_store: Path = CACHE_DIR / "error_email_ratelimit.json"

    # Optional fallback channel (Slack webhook)
    slack_webhook_url: Optional[str] = None

    # Env placeholders supported in these fields via ${ENV}
    _env_fields: List[str] = field(default_factory=lambda: [
        "sender_email", "recipient_email", "cc", "bcc", "reply_to", "slack_webhook_url"
    ])


# ---------------------------------------------------------------------------
# ErrorNotifier
# ---------------------------------------------------------------------------
class ErrorNotifier:
    """
    Production-friendly error emailer via Brevo with:
      • ${ENV} resolution for addresses and webhook
      • Duplicate suppression across *restarts* (file-backed)
      • Simple per-minute rate limit (file-backed)
      • HTML + text bodies, priority headers
      • Priority routing, subject prefix/suffix
      • Optional Slack webhook fallback when Brevo fails
      • Ignore regexes for noisy/known errors
      • Attachments (paths or bytes)
      • Dry-run mode for safe staging

    Public API:
      - send_error_email(error_summary, recipient=None, priority=EmailPriority.HIGH,
                         additional_data=None, subject=None,
                         attachment_paths=None, attachments=None,
                         bypass_dedupe=False, bypass_rate_limit=False) -> bool
      - notify_exception(exc, context=None, **kwargs) -> bool
    """

    def __init__(self, app_settings: Dict):
        load_dotenv(find_dotenv(usecwd=True))
        self.config = self._load_config(app_settings)
        self._session = requests.Session()

        # Pre-compile ignore regexes
        self._ignore_regexes: List[re.Pattern] = []
        for pat in (self.config.ignore_patterns or []):
            try:
                self._ignore_regexes.append(re.compile(str(pat), re.IGNORECASE))
            except re.error as e:
                logger.warning("Invalid ignore pattern %r: %s", pat, e)

        try:
            self._validate_config()
            logger.info(
                "ErrorNotifier ready (endpoint=%s, dry_run=%s, rate=%d/min).",
                self.config.api_endpoint,
                self.config.dry_run,
                self.config.rate_limit_per_min,
            )
            self._is_initialized = True
        except ValueError as e:
            logger.critical("ErrorNotifier config invalid: %s", e, exc_info=True)
            self._is_initialized = False
        except Exception as e:
            logger.critical("ErrorNotifier init error: %s", e, exc_info=True)
            self._is_initialized = False

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------
    def send_error_email(
        self,
        error_summary: str,
        recipient: Optional[str | List[str]] = None,
        priority: EmailPriority = EmailPriority.HIGH,
        additional_data: Optional[Dict[str, Any]] = None,
        subject: Optional[str] = None,
        *,
        attachment_paths: Optional[List[str]] = None,
        attachments: Optional[List[Tuple[str, bytes]]] = None,
        bypass_dedupe: bool = False,
        bypass_rate_limit: bool = False,
    ) -> bool:
        if not self._is_initialized:
            logger.warning("ErrorNotifier not initialized; skipping send.")
            return False

        # Ignore/silence checks
        if self._should_ignore(error_summary):
            logger.info("Suppressed by ignore_patterns.")
            return True

        # Recipients (priority overrides first)
        to_list = None
        pr_key = priority.name.upper()
        pr_override = self.config.priority_overrides_to.get(pr_key)
        if pr_override:
            to_list = self._as_list(pr_override)
        if not to_list:
            to_list = self._as_list(recipient or self.config.recipient_email)

        cc_list = self._as_list(self.config.cc)
        bcc_list = self._as_list(self.config.bcc)
        if not to_list:
            logger.error("No valid recipients configured.")
            return False
        for addr in to_list + cc_list + bcc_list:
            if not self._valid_email(addr):
                logger.error("Invalid recipient email: %r", addr)
                return False

        # Duplicate suppression
        if not bypass_dedupe and self._is_duplicate(error_summary, priority, additional_data):
            logger.info("Suppressed duplicate error email (recently sent).")
            return True

        # Rate limiting
        if not bypass_rate_limit and not self._check_ratelimit_allow():
            logger.warning("Rate limit reached; queuing to Slack only (if configured).")
            return self._slack_fallback_only(error_summary, priority, additional_data, subject)

        # Build payload
        try:
            dedupe_key = self._dedupe_key(error_summary, priority, additional_data)
            payload = self._build_payload(
                error_summary=error_summary,
                to_list=to_list,
                cc_list=cc_list,
                bcc_list=bcc_list,
                priority=priority,
                additional_data=self._auto_enrich(additional_data),
                subject_override=subject,
                attachment_paths=attachment_paths,
                attachments=attachments,
                idempotency_key=dedupe_key,  # CHANGE: include idempotency header
            )
        except Exception as e:
            logger.error("Failed to build email payload: %s", e, exc_info=True)
            return False

        # Dry-run
        if self.config.dry_run:
            logger.info("[DRY-RUN] Would send error email: %s", json.dumps(_preview_payload(payload), ensure_ascii=False)[:900])
            self._record_send(success=True)  # count toward rate to mimic prod pacing
            return True

        # Send with small retry
        headers = {
            "api-key": self.config.api_key or "",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        for attempt in range(1, 4):
            try:
                resp = self._session.post(
                    self.config.api_endpoint,
                    json=payload,
                    headers=headers,
                    timeout=self.config.timeout,
                )
                if 200 <= resp.status_code < 300:
                    self._record_send(success=True)
                    logger.info("Error email sent (priority=%s).", priority.value.upper())
                    return True

                # 429 -> treat as rate-limited (no more retries here)
                if resp.status_code == 429:
                    logger.warning("Email send rate-limited [HTTP 429]: %s", _truncate(resp.text, 300))
                    break

                # Retry only on 5xx
                if 500 <= resp.status_code < 600 and attempt < 3:
                    logger.warning(
                        "Email send failed (try %d) [HTTP %s]: %s",
                        attempt,
                        resp.status_code,
                        _truncate(resp.text, 300),
                    )
                    time.sleep(min(1.5 * attempt, 5.0))
                    continue

                logger.error("Email send failed [HTTP %s]: %s", resp.status_code, _truncate(resp.text, 500))
                break

            except requests.RequestException as rexc:
                logger.warning("Network error (try %d): %s", attempt, rexc, exc_info=True)
                time.sleep(min(1.5 * attempt, 5.0))
                continue

        # Fallback to Slack if configured
        self._record_send(success=False)
        return self._slack_fallback_only(error_summary, priority, additional_data, subject)

    def notify_exception(
        self,
        exc: BaseException,
        context: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> bool:
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        ctx = dict(context or {})
        ctx.setdefault("ExceptionType", type(exc).__name__)
        try:
            ctx.setdefault("ExceptionMessage", str(exc))
        except Exception:
            pass
        return self.send_error_email(tb, additional_data=ctx, **kwargs)

    # -----------------------------------------------------------------------
    # Internals
    # -----------------------------------------------------------------------
    def _load_config(self, app_settings: Dict) -> ErrorNotifierConfig:
        cfg = ErrorNotifierConfig()
        error_settings = (app_settings or {}).get("error_notifications", {}) or {}
        api_keys = (app_settings or {}).get("api_keys", {}) or {}

        # API key: env first, then settings
        cfg.api_key = os.getenv("BREVO_API_KEY") or api_keys.get("brevo") or None

        # Basic fields
        cfg.api_endpoint = error_settings.get("api_endpoint", cfg.api_endpoint)
        cfg.sender_name = error_settings.get("sender_name", cfg.sender_name)
        cfg.sender_email = error_settings.get("sender_email", cfg.sender_email)
        cfg.recipient_email = error_settings.get("recipient_email", cfg.recipient_email)
        cfg.cc = error_settings.get("cc", cfg.cc)
        cfg.bcc = error_settings.get("bcc", cfg.bcc)
        cfg.reply_to = error_settings.get("reply_to", cfg.reply_to)
        cfg.timeout = int(error_settings.get("timeout", cfg.timeout))
        cfg.dry_run = bool(error_settings.get("dry_run", cfg.dry_run))
        cfg.dedupe_seconds = int(error_settings.get("dedupe_seconds", cfg.dedupe_seconds))
        cfg.max_summary_chars = int(error_settings.get("max_summary_chars", cfg.max_summary_chars))
        cfg.rate_limit_per_min = int(error_settings.get("rate_limit_per_min", cfg.rate_limit_per_min))
        cfg.slack_webhook_url = error_settings.get("slack_webhook_url", cfg.slack_webhook_url)

        # New controls
        cfg.subject_prefix = error_settings.get("subject_prefix", cfg.subject_prefix)
        cfg.subject_suffix = error_settings.get("subject_suffix", cfg.subject_suffix)
        cfg.service_name_env_var = error_settings.get("service_name_env_var", cfg.service_name_env_var)
        cfg.ignore_patterns = list(error_settings.get("ignore_patterns", cfg.ignore_patterns) or [])

        # Priority routes
        pr = error_settings.get("priority_routes", {}) or {}
        if isinstance(pr, dict):
            # resolve ${ENV} placeholders and accept "a@x.com,b@y.com" or list
            parsed: Dict[str, str | List[str]] = {}
            for k, v in pr.items():
                key = str(k).upper()
                val = self._resolve_env_placeholder(v)
                parsed[key] = val  # _as_list is applied later
            cfg.priority_overrides_to = parsed

        # Resolve ${ENV_VAR} placeholders on basic address fields
        for f in cfg._env_fields:
            val = getattr(cfg, f)
            setattr(cfg, f, self._resolve_env_placeholder(val))

        return cfg

    def _validate_config(self) -> None:
        if not self.config.api_key and not self.config.dry_run:
            raise ValueError("BREVO_API_KEY missing; set env or provide in settings. Or enable dry_run for testing.")
        if not self.config.api_endpoint:
            raise ValueError("api_endpoint required.")
        if not self._valid_email(self.config.sender_email):
            raise ValueError(f"Invalid sender_email: {self.config.sender_email!r}")

        # At least one recipient (string, list, or env-resolved)
        if not self._as_list(self.config.recipient_email):
            raise ValueError("recipient_email required (string or list).")

        # Prepare persistent stores
        for store in (self.config.dedupe_store, self.config.ratelimit_store):
            store.parent.mkdir(parents=True, exist_ok=True)
            if not store.exists():
                store.write_text("{}", encoding="utf-8")

    @staticmethod
    def _resolve_env_placeholder(value: Optional[str | List[str]]) -> Optional[str | List[str]]:
        """
        CHANGE: If a ${VAR} is *unresolved*, return None (so it can be dropped),
        not the literal '${VAR}' which would later fail email validation.
        """
        if value is None:
            return None
        if isinstance(value, list):
            out: List[str] = []
            for v in value:
                resolved = ErrorNotifier._resolve_env_placeholder(v)  # type: ignore
                if isinstance(resolved, str) and resolved:
                    out.append(resolved)
            return out
        v = str(value).strip()
        if v.startswith("${") and v.endswith("}"):
            env_name = v[2:-1]
            env_val = os.getenv(env_name)
            return env_val if env_val else None  # CHANGE: drop if unresolved
        return v or None

    @staticmethod
    def _valid_email(addr: str) -> bool:
        if not addr or "\n" in addr or "\r" in addr:
            return False
        return bool(re.match(r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$", addr))

    @staticmethod
    def _as_list(recipients: str | List[str] | None) -> List[str]:
        """
        CHANGE: normalize after env resolution; ignore empties; validate emails.
        Accept 'a@x.com,b@y.com' or ['a@x.com','b@y.com'] or None.
        """
        if not recipients:
            return []
        if isinstance(recipients, list):
            vals = recipients
        else:
            vals = [r.strip() for r in str(recipients).replace(";", ",").split(",")]
        cleaned = []
        for r in vals:
            if not r:
                continue
            if ErrorNotifier._valid_email(r):
                cleaned.append(r)
        return cleaned

    def _should_ignore(self, summary: str) -> bool:
        if not self._ignore_regexes:
            return False
        s = self._sanitize_summary(summary or "")
        return any(pat.search(s) for pat in self._ignore_regexes)

    # ---------------- Duplicate suppression (file-backed) -------------------
    def _is_duplicate(
        self,
        summary: str,
        priority: EmailPriority,
        additional: Optional[Dict[str, Any]],
    ) -> bool:
        try:
            now = time.time()
            key = self._dedupe_key(summary, priority, additional)
            data = self._load_json(self.config.dedupe_store) or {}
            last_ts = float(data.get(key, 0))
            if now - last_ts < self.config.dedupe_seconds:
                return True
            # update & prune
            data[key] = now
            # prune old entries
            cutoff = now - max(self.config.dedupe_seconds * 4, 1200)
            for k, ts in list(data.items()):
                try:
                    if float(ts) < cutoff:
                        data.pop(k, None)
                except Exception:
                    data.pop(k, None)
            self._save_json_atomic(data, self.config.dedupe_store)
            return False
        except Exception:
            # on any error, fail open (do not suppress)
            return False

    @staticmethod
    def _dedupe_key(summary: str, priority: EmailPriority, additional: Optional[Dict[str, Any]]) -> str:
        # Normalize + redact before hashing to avoid trivial diffs
        normalized = ErrorNotifier._sanitize_summary(summary or "")
        # Include handful of stable context keys if present (script, env, run id)
        keys = []
        for k in ("Script", "Env", "Run ID", "ExceptionType"):
            if additional and k in additional:
                keys.append(f"{k}={additional[k]}")
        raw = "|".join([priority.value, normalized] + keys)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    # ---------------- Rate limit (file-backed token bucket) -----------------
    def _check_ratelimit_allow(self) -> bool:
        try:
            now = time.time()
            data = self._load_json(self.config.ratelimit_store) or {}
            bucket = int(data.get("bucket", self.config.rate_limit_per_min))
            last_ts = float(data.get("last_ts", now))

            # Refill based on elapsed time
            capacity = max(1, self.config.rate_limit_per_min)
            rate_per_sec = capacity / 60.0
            bucket = min(capacity, int(bucket + (now - last_ts) * rate_per_sec))
            if bucket <= 0:
                # deny and persist
                data.update({"bucket": 0, "last_ts": now})
                self._save_json_atomic(data, self.config.ratelimit_store)
                return False

            # consume one token
            bucket -= 1
            data.update({"bucket": bucket, "last_ts": now})
            self._save_json_atomic(data, self.config.ratelimit_store)
            return True
        except Exception:
            # fail open on errors
            return True

    def _record_send(self, success: bool) -> None:
        # no-op hook (kept for symmetry / potential metrics)
        pass

    # ---------------- Payload builders --------------------------------------
    def _build_payload(
        self,
        *,
        error_summary: str,
        to_list: List[str],
        cc_list: List[str],
        bcc_list: List[str],
        priority: EmailPriority,
        additional_data: Optional[Dict[str, Any]],
        subject_override: Optional[str],
        attachment_paths: Optional[List[str]],
        attachments: Optional[List[Tuple[str, bytes]]],
        idempotency_key: str,  # CHANGE: pass through for header
    ) -> Dict[str, Any]:
        clean_summary = self._sanitize_summary(error_summary)
        if len(clean_summary) > self.config.max_summary_chars:
            clean_summary = clean_summary[: self.config.max_summary_chars] + "\n\n[... truncated ...]"

        env = os.getenv("ENV", "").strip().upper()
        service = os.getenv(self.config.service_name_env_var, "Pipeline")
        prefix_emoji = {
            EmailPriority.CRITICAL: "🛑",
            EmailPriority.HIGH: "🚨",
            EmailPriority.NORMAL: "⚠️",
            EmailPriority.LOW: "ℹ️",
        }.get(priority, "⚠️")
        env_tag = f" [{env}]" if env else ""
        subject = subject_override or f"{prefix_emoji} {service} Error{env_tag}"
        if self.config.subject_prefix:
            subject = f"{self.config.subject_prefix} {subject}".strip()
        if self.config.subject_suffix:
            subject = f"{subject} {self.config.subject_suffix}".strip()

        html_block = self._render_html(clean_summary, priority, additional_data)
        text_block = self._render_text(clean_summary, priority, additional_data)

        headers = self._priority_headers(priority)
        headers["X-Idempotency-Key"] = idempotency_key  # CHANGE: provider-side dedupe hint

        payload: Dict[str, Any] = {
            "sender": {"name": self.config.sender_name, "email": self.config.sender_email},
            "to": [{"email": e} for e in to_list],
            "subject": subject,
            "htmlContent": html_block,
            "textContent": text_block,
            "tags": ["automation-error", priority.value, env or "ENV-NA"],
            "headers": headers,
        }
        if cc_list:
            payload["cc"] = [{"email": e} for e in cc_list]
        if bcc_list:
            payload["bcc"] = [{"email": e} for e in bcc_list]
        if self.config.reply_to and self._valid_email(str(self.config.reply_to)):
            payload["replyTo"] = {"email": str(self.config.reply_to)}

        att_list = self._prepare_attachments(attachment_paths, attachments)
        if att_list:
            payload["attachment"] = att_list

        return payload

    @staticmethod
    def _sanitize_summary(summary: str) -> str:
        s = summary or ""
        # Redact common secrets
        s = re.sub(r"(?i)\b(api[_-]?key|token|secret|bearer)\b\s*[:=]\s*([A-Za-z0-9_\-\.]{8,})", r"\1: [REDACTED]", s)
        s = re.sub(r"(?i)\b(password)\b\s*[:=]\s*\S+", r"\1: [REDACTED]", s)
        # De-noise UUIDs and 0x addresses to improve dedupe
        s = re.sub(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\b", "[UUID]", s, flags=re.I)
        s = re.sub(r"\b0x[0-9a-fA-F]+\b", "0xADDR", s)
        # Collapse weird whitespace
        s = re.sub(r"[ \t]+\n", "\n", s)
        return s.strip()

    @staticmethod
    def _priority_headers(priority: EmailPriority) -> Dict[str, str]:
        # CHANGE: CRITICAL treated as highest priority for MUA hinting
        mapping = {
            EmailPriority.CRITICAL: "1",
            EmailPriority.HIGH: "1",
            EmailPriority.NORMAL: "3",
            EmailPriority.LOW: "5",
        }
        return {"X-Priority": mapping.get(priority, "3"), "X-Mailer": "PythonProHub-ErrorNotifier"}

    def _render_html(self, escaped_summary: str, priority: EmailPriority, additional: Optional[Dict[str, Any]]) -> str:
        esc = html.escape(escaped_summary)
        meta = self._format_additional_data_html(additional)
        return f"""
<html>
  <body style="font-family:Arial,Helvetica,sans-serif;line-height:1.55;color:#222;">
    <h2 style="margin:0 0 8px 0;">PythonProHub Automation Error Report</h2>
    <p style="margin:0 0 12px 0;"><strong>Date &amp; Time:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
    <p style="margin:0 0 12px 0;"><strong>Priority:</strong> {priority.value.upper()}</p>
    <p style="margin:0 0 12px 0;">Summary:</p>
    <pre style="background:#f6f8fa;padding:12px;border-radius:6px;white-space:pre-wrap;word-wrap:break-word;border:1px solid #eee;">{esc}</pre>
    {meta}
    <p style="margin-top:18px;color:#666;font-size:12px;"><em>This is an automated message from your Python Pro Hub pipeline.</em></p>
  </body>
</html>
""".strip()

    def _render_text(self, summary: str, priority: EmailPriority, additional: Optional[Dict[str, Any]]) -> str:
        lines = [
            "PythonProHub Automation Error Report",
            f"Date & Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Priority: {priority.value.upper()}",
            "",
            "Summary:",
            summary.strip(),
        ]
        if additional:
            lines.append("")
            lines.append("Additional Context:")
            for k, v in additional.items():
                lines.append(f"- {str(k)}: {str(v)}")
        lines.append("")
        lines.append("This is an automated message from your Python Pro Hub pipeline.")
        return "\n".join(lines)

    @staticmethod
    def _format_additional_data_html(data: Optional[Dict[str, Any]]) -> str:
        if not data:
            return ""
        items = "\n".join(
            f"<li><strong>{html.escape(str(k))}:</strong> {html.escape(str(v))}</li>"
            for k, v in data.items()
        )
        return f"<h3 style='margin:16px 0 8px 0;'>Additional Context</h3><ul style='margin:0 0 16px 18px;'>{items}</ul>"

    def _prepare_attachments(
        self,
        attachment_paths: Optional[List[str]],
        attachments: Optional[List[Tuple[str, bytes]]],
    ) -> List[Dict[str, str]]:
        out: List[Dict[str, str]] = []
        # Paths
        for p in (attachment_paths or []):
            try:
                fp = Path(p)
                if fp.exists() and fp.is_file():
                    b = fp.read_bytes()
                    out.append({"content": base64.b64encode(b).decode("utf-8"), "name": fp.name})
            except Exception as e:
                logger.warning("Failed to attach %s: %s", p, e)
        # In-memory
        for item in (attachments or []):
            try:
                name, blob = item
                out.append({"content": base64.b64encode(blob).decode("utf-8"), "name": str(name)})
            except Exception as e:
                logger.warning("Failed to attach in-memory item: %s", e)
        return out

    def _auto_enrich(self, ctx: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        data = dict(ctx or {})
        data.setdefault("Host", platform.node() or os.getenv("HOSTNAME", ""))
        data.setdefault("PID", os.getpid())
        data.setdefault("CWD", os.getcwd())
        if os.getenv("GIT_SHA"):
            data.setdefault("Git SHA", os.getenv("GIT_SHA"))
        return data

    # ---------------- Slack fallback ----------------------------------------
    def _slack_fallback_only(
        self,
        error_summary: str,
        priority: EmailPriority,
        additional: Optional[Dict[str, Any]],
        subject: Optional[str],
    ) -> bool:
        webhook = (self.config.slack_webhook_url or "").strip()
        if not webhook:
            logger.info("Slack webhook not configured; no fallback sent.")
            return False
        try:
            env = os.getenv("ENV", "").upper()
            prefix = {
                EmailPriority.CRITICAL: "🛑",
                EmailPriority.HIGH: "🚨",
                EmailPriority.NORMAL: "⚠️",
                EmailPriority.LOW: "ℹ️",
            }.get(priority, "⚠️")
            title = subject or f"{prefix} Error {f'[{env}]' if env else ''}"
            text = f"*{title}*\n```\n{self._sanitize_summary(error_summary)[:3500]}\n```"
            if additional:
                extras = "\n".join([f"• *{k}*: `{v}`" for k, v in additional.items()])
                text += f"\n{extras}"
            resp = self._session.post(webhook, json={"text": text}, timeout=8)
            ok = 200 <= resp.status_code < 300
            logger.info("Slack fallback %s.", "sent" if ok else f"failed [{resp.status_code}]")
            return ok
        except Exception as e:
            logger.warning("Slack fallback error: %s", e)
            return False

    # ---------------- Small file JSON helpers --------------------------------
    @staticmethod
    def _load_json(path: Path) -> Optional[dict]:
        try:
            if path.exists():
                return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
        return None

    @staticmethod
    def _save_json_atomic(data: dict, path: Path) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _truncate(s: Optional[str], n: int) -> str:
    if not s:
        return ""
    return s if len(s) <= n else s[: n - 3] + "..."

def _preview_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Strip large fields for log previews.
    """
    preview = dict(payload)
    preview["htmlContent"] = f"[{len(str(preview.get('htmlContent','')))} chars]"
    preview["textContent"] = f"[{len(str(preview.get('textContent','')))} chars]"
    if "attachment" in preview:
        preview["attachment"] = f"[{len(payload.get('attachment', []))} attachments]"
    return preview


# ---------------------------------------------------------------------------
# CLI test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    load_dotenv(find_dotenv(usecwd=True))

    demo_settings = {
        "api_keys": {"brevo": os.getenv("BREVO_API_KEY", "")},
        "error_notifications": {
            "sender_name": "Test Automation",
            "sender_email": os.getenv("SENDER_EMAIL", "test-sender@example.com"),
            "recipient_email": os.getenv("RECIPIENT_EMAIL", "test-recipient@example.com"),
            "cc": os.getenv("RECIPIENT_CC"),     # "a@x.com,b@y.com" or unset
            "bcc": os.getenv("RECIPIENT_BCC"),
            "reply_to": os.getenv("REPLY_TO"),
            "api_endpoint": os.getenv("BREVO_API_ENDPOINT", "https://api.brevo.com/v3/smtp/email"),
            "timeout": 10,
            "dry_run": os.getenv("ERROR_NOTIFIER_DRY_RUN", "1") not in ("0", "false", "no"),
            "dedupe_seconds": 10,
            "rate_limit_per_min": int(os.getenv("ERROR_EMAILS_PER_MIN", "20")),
            "slack_webhook_url": os.getenv("SLACK_WEBHOOK_URL"),
            # noisy examples can be silenced
            "ignore_patterns": [r"Temporary network glitch", r"Timeout connecting to .*? but retry succeeded"],
            # priority routes
            "priority_routes": {
                "CRITICAL": os.getenv("CRITICAL_ALERTS_TO", "${CRITICAL_ALERTS_TO}"),
                "HIGH": os.getenv("HIGH_ALERTS_TO", "${HIGH_ALERTS_TO}"),
                "LOW": os.getenv("LOW_ALERTS_TO", "${LOW_ALERTS_TO}"),
            },
            "subject_prefix": os.getenv("SUBJECT_PREFIX"),
            "subject_suffix": os.getenv("SUBJECT_SUFFIX"),
            "service_name_env_var": os.getenv("SERVICE_NAME_ENV", "SERVICE_NAME"),
        },
    }

    notifier = ErrorNotifier(demo_settings)

    if notifier._is_initialized:
        try:
            raise RuntimeError("Example failure from CLI test — something exploded 🔥 id=0xDEADBEEF uuid=123e4567-e89b-12d3-a456-426614174000")
        except Exception as e:
            notifier.notify_exception(
                e,
                context={
                    "Script": "daily_publish.py",
                    "Run ID": "xyz123",
                    "Env": os.getenv("ENV", "dev"),
                },
                priority=EmailPriority.HIGH,
                subject="Automation Crash Report (Demo)",
                # Try an attachment path if you want:
                # attachment_paths=["logs/error_emailer.log"],
            )

        notifier.send_error_email(
            "Minor: One article failed to publish, others successful.",
            priority=EmailPriority.NORMAL,
            additional_data={"Article Title": "Failed Article", "Reason": "429 from WP API"},
            subject="Automation Warning (Demo)",
        )

        # Example with bypass knobs and attachment bytes
        notifier.send_error_email(
            "Critical pipeline halt after retries.",
            priority=EmailPriority.CRITICAL,
            additional_data={"Script": "scheduler.py", "Env": os.getenv("ENV", "dev")},
            attachments=[("context.json", json.dumps({"ok": False}, indent=2).encode("utf-8"))],
            bypass_dedupe=True,
            bypass_rate_limit=True,
            subject="Emergency Send (Bypass)",
        )
    else:
        logger.warning("Notifier not initialized; skipping demo sends.")
