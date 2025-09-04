# send_newsletter.py

import os
import re
import html
import ssl
import time
import json
import logging
from dataclasses import dataclass, field
from email.utils import formataddr, make_msgid
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import List, Dict, Optional, Union, Tuple
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse, urlunparse, urlencode, parse_qsl

from dotenv import load_dotenv, find_dotenv
from openai import OpenAI  # SDK v1

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

_root = logging.getLogger()
if not _root.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(LOG_DIR / "newsletter.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class NewsletterConfig:
    enabled: bool = False
    sender_name: str = "PythonProHub"
    sender_email: Optional[str] = None
    recipient_email: Union[str, List[str], None] = None
    cc: Union[str, List[str], None] = None
    bcc: Union[str, List[str], None] = None
    reply_to: Optional[str] = None            # NEW

    smtp_server: Optional[str] = None
    smtp_port: int = 587
    email_username: Optional[str] = None
    email_password: Optional[str] = None
    use_ssl: bool = False
    use_tls: bool = True
    timeout: int = 20
    dry_run: bool = False  # if True, build and log email but do not send

    min_articles_for_newsletter: int = 1
    subject_prefix: str = "Daily PythonProHub Digest"
    preheader_text: str = "Fresh Python guides, tips, and demos inside."
    list_unsubscribe: Optional[str] = None            # e.g., "mailto:unsubscribe@example.com"
    one_click_unsubscribe_url: Optional[str] = None   # e.g., "https://…/unsubscribe?token=…"

    # OpenAI
    openai_model: str = "gpt-4o-mini"
    openai_enabled: bool = True
    max_teaser_tokens: int = 220
    temperature: float = 0.5

    # UTM
    utm_source: str = "pythonprohub"
    utm_medium: str = "email"
    utm_campaign: str = "daily_digest"

    # Pricing (USD per 1K tokens). Provide via .env to keep flexible.
    pricing_in_per_1k: float = 0.0
    pricing_out_per_1k: float = 0.0

    # Social caption generation (stored locally for later posting)
    social_enabled: bool = True
    social_platforms: List[str] = field(default_factory=lambda: ["twitter", "linkedin"])

    # Allow ${ENV} placeholders in these fields
    _env_fields: List[str] = field(default_factory=lambda: [
        "sender_email", "smtp_server", "email_username", "email_password",
        "recipient_email", "cc", "bcc",
        "list_unsubscribe", "reply_to"  # NEW
    ])


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def _resolve_env_placeholder(value: Optional[Union[str, List[str]]]) -> Optional[Union[str, List[str]]]:
    if value is None:
        return value
    if isinstance(value, list):
        return [_resolve_env_placeholder(v) for v in value]  # type: ignore
    if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
        return os.getenv(value[2:-1], "")
    return value


def _as_list(recipients: Union[str, List[str], None]) -> List[str]:
    if not recipients:
        return []
    if isinstance(recipients, list):
        vals = [r.strip() for r in recipients if r and r.strip()]
    else:
        vals = [r.strip() for r in recipients.replace(";", ",").split(",") if r.strip()]
    # Filter to valid emails only (prevents SMTP 550s / header issues)
    return [r for r in vals if _valid_email(r)]


def _valid_email(addr: str) -> bool:
    if not addr or "\n" in addr or "\r" in addr:
        return False
    return bool(re.match(r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$", addr))


def _sanitize_subject(s: str) -> str:
    s = (s or "").replace("\r", " ").replace("\n", " ").strip()
    s = re.sub(r"\s{2,}", " ", s)
    s = re.sub(r"[^\w\s\-\:\&\|,\.]", "", s)
    return s[:180]


def _append_utm(url: str, *, source: str, medium: str, campaign: str, content: Optional[str] = None) -> str:
    if not url or not url.startswith(("http://", "https://")):
        return url
    parsed = urlparse(url)
    q = dict(parse_qsl(parsed.query, keep_blank_values=True))
    q.update({"utm_source": source, "utm_medium": medium, "utm_campaign": campaign})
    if content:
        q["utm_content"] = content
    new_q = urlencode(q, doseq=True)
    return urlunparse(parsed._replace(query=new_q))


def _angle(s: str) -> str:
    """Wrap a header value in angle brackets if not present."""
    s = (s or "").strip()
    if not s:
        return s
    return s if s.startswith("<") and s.endswith(">") else f"<{s}>"


# ---------------------------------------------------------------------------
# Newsletter sender
# ---------------------------------------------------------------------------
class NewsletterSender:
    """
    Sends a simple daily digest email. Optionally uses OpenAI to generate
    a short teaser paragraph for the newsletter body and social captions.
    """

    def __init__(self, app_settings: Dict, openai_client: Optional[OpenAI] = None):
        self.app_settings = app_settings or {}
        raw_cfg = (self.app_settings.get("newsletter") or {}).copy()

        # Load env for ${VAR} placeholders
        load_dotenv(find_dotenv(usecwd=True))

        # Build config with env overrides
        self.config = NewsletterConfig(
            enabled=bool(raw_cfg.get("enabled", False)),
            sender_name=raw_cfg.get("sender_name", "PythonProHub"),
            sender_email=raw_cfg.get("sender_email"),
            recipient_email=raw_cfg.get("recipient_email"),
            cc=raw_cfg.get("cc"),
            bcc=raw_cfg.get("bcc"),
            reply_to=raw_cfg.get("reply_to"),  # NEW
            smtp_server=raw_cfg.get("smtp_server"),
            smtp_port=int(raw_cfg.get("smtp_port", 587) or 587),
            email_username=raw_cfg.get("email_username"),
            email_password=raw_cfg.get("email_password"),
            use_ssl=bool(raw_cfg.get("use_ssl", False)),
            use_tls=bool(raw_cfg.get("use_tls", True)),
            timeout=int(raw_cfg.get("timeout", 20) or 20),
            dry_run=bool(raw_cfg.get("dry_run", False)),
            min_articles_for_newsletter=int(raw_cfg.get("min_articles_for_newsletter", 1) or 1),
            subject_prefix=str(raw_cfg.get("subject_prefix", "Daily PythonProHub Digest")),
            preheader_text=str(raw_cfg.get("preheader_text", "Fresh Python guides, tips, and demos inside.")),
            list_unsubscribe=raw_cfg.get("list_unsubscribe"),
            one_click_unsubscribe_url=raw_cfg.get("one_click_unsubscribe_url"),
            openai_model=str(raw_cfg.get("openai_model", "gpt-4o-mini")),
            openai_enabled=bool(raw_cfg.get("openai_enabled", True)),
            max_teaser_tokens=int(raw_cfg.get("max_teaser_tokens", 220) or 220),
            temperature=float(raw_cfg.get("temperature", 0.5) or 0.5),
            utm_source=str(raw_cfg.get("utm_source", "pythonprohub")),
            utm_medium=str(raw_cfg.get("utm_medium", "email")),
            utm_campaign=str(raw_cfg.get("utm_campaign", "daily_digest")),
            pricing_in_per_1k=float(os.getenv("OPENAI_PRICE_IN_PER_1K", raw_cfg.get("pricing_in_per_1k", 0.0) or 0.0)),
            pricing_out_per_1k=float(os.getenv("OPENAI_PRICE_OUT_PER_1K", raw_cfg.get("pricing_out_per_1k", 0.0) or 0.0)),
            social_enabled=bool(raw_cfg.get("social_enabled", True)),
            social_platforms=list(raw_cfg.get("social_platforms", ["twitter", "linkedin"])),
        )

        # Resolve ${ENV} placeholders
        for f in self.config._env_fields:
            val = getattr(self.config, f)
            setattr(self.config, f, _resolve_env_placeholder(val))

        # OpenAI client (optional)
        if openai_client:
            self.openai_client = openai_client
            logger.info("NewsletterSender initialized with provided OpenAI client.")
        else:
            api_key = self.app_settings.get("api_keys", {}).get("openai") or os.getenv("OPENAI_API_KEY")
            if api_key and self.config.openai_enabled:
                self.openai_client = OpenAI(api_key=api_key)
                logger.info("NewsletterSender created its own OpenAI client.")
            else:
                self.openai_client = None
                if self.config.openai_enabled:
                    logger.warning("No OPENAI_API_KEY found; digest summary will be basic text.")

        # Validate minimal SMTP / email config (unless dry-run)
        self._validate_email_config()

    # -------------------- Public API --------------------
    def send_digest_email(self, published_articles: List[Dict]) -> Dict[str, Union[str, int, float, bool]]:
        result = {
            "sent": False,
            "recipients": 0,
            "model": self.config.openai_model,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "usd_cost_estimate": 0.0,
            "dry_run": self.config.dry_run,
        }

        if not self.config.enabled:
            logger.info("Newsletter sending is disabled in settings.")
            return result

        if len(published_articles) < self.config.min_articles_for_newsletter:
            logger.info(
                "Not enough articles (%d) for newsletter. Minimum is %d.",
                len(published_articles), self.config.min_articles_for_newsletter
            )
            return result

        # Prepare content
        today = datetime.now().strftime('%Y-%m-%d')
        subject = _sanitize_subject(f"{self.config.subject_prefix} — {today}")
        text_summary, usage = self._generate_digest_summary(published_articles)
        result.update(usage)

        # Build bodies
        html_body, text_body = self._build_body(subject, text_summary, published_articles, utm_content=today)

        # Build message
        msg = self._assemble_message(subject, html_body, text_body)

        # Dry-run / preview files
        if self.config.dry_run:
            self._write_previews(today, html_body, text_body)
            logger.info("[DRY-RUN] Newsletter prepared for %s recipients.", len(self._all_recipients()))
            logger.info("[DRY-RUN] Subject: %s", subject)
            result["recipients"] = len(self._all_recipients())
            return result

        # Send with basic retry
        sent_ok = self._send_with_retry(msg, max_attempts=3)
        result["sent"] = sent_ok
        result["recipients"] = len(self._all_recipients())

        # Optional: generate social captions for later posting
        try:
            if self.config.social_enabled:
                self._render_social_captions(today, published_articles)
        except Exception as e:
            logger.warning("Social caption rendering failed: %s", e)

        return result

    # -------------------- Internals --------------------
    def _write_previews(self, today: str, html_body: str, text_body: str) -> None:
        html_path = LOG_DIR / f"newsletter_preview_{today}.html"
        txt_path = LOG_DIR / f"newsletter_preview_{today}.txt"
        try:
            html_path.write_text(html_body, encoding="utf-8")
            txt_path.write_text(text_body, encoding="utf-8")
        except Exception as e:
            logger.warning("Failed to write preview files: %s", e)

    def _validate_email_config(self) -> None:
        if self.config.dry_run:
            return
        missing = []
        if not self.config.smtp_server:
            missing.append("smtp_server")
        if not self.config.sender_email:
            missing.append("sender_email")
        if not self._all_recipients():
            missing.append("recipient_email/cc/bcc")
        if not self.config.use_ssl:  # TLS/plain needs creds
            if not self.config.email_username or not self.config.email_password:
                missing.append("email_username/email_password")
        if missing:
            logger.error("Newsletter settings incomplete: %s", ", ".join(missing))
        # Validate addresses (warnings only; _all_recipients filters bad ones)
        for addr in [self.config.sender_email, self.config.reply_to]:
            if addr and not _valid_email(addr):
                logger.warning("Suspicious email address: %r", addr)

    def _all_recipients(self) -> List[str]:
        to = _as_list(self.config.recipient_email)
        cc = _as_list(self.config.cc)
        bcc = _as_list(self.config.bcc)
        return [a for a in (to + cc + bcc) if a]

    def _generate_digest_summary(self, articles: List[Dict]) -> Tuple[str, Dict[str, Union[int, float, str]]]:
        usage = {
            "model": self.config.openai_model,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "usd_cost_estimate": 0.0,
        }

        if not self.openai_client:
            lines = [f"- {a.get('title','Untitled')} — {a.get('url','N/A')}" for a in articles]
            return "Today's highlights:\n" + "\n".join(lines), usage

        titles = "\n".join([f"- {a.get('title', 'Untitled')} (URL: {a.get('url', 'N/A')})" for a in articles])
        prompt = (
            "Write a short, upbeat newsletter teaser (2–3 sentences) inviting readers to explore the links. "
            "Be concrete about themes, avoid fluff, avoid emojis, avoid passive voice. "
            "No code. No phrases that reveal AI involvement.\n\n"
            f"Articles:\n{titles}"
        )
        try:
            resp = self.openai_client.chat.completions.create(
                model=self.config.openai_model,
                messages=[
                    {"role": "system", "content": "You are a concise newsletter copywriter."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=int(self.config.max_teaser_tokens),
                temperature=float(self.config.temperature),
            )
            teaser = (resp.choices[0].message.content or "").strip()
            teaser = re.sub(r"\s{2,}", " ", teaser)[:600]

            try:
                p = int(getattr(resp, "usage").prompt_tokens)  # type: ignore[attr-defined]
                c = int(getattr(resp, "usage").completion_tokens)  # type: ignore[attr-defined]
                t = int(getattr(resp, "usage").total_tokens)  # type: ignore[attr-defined]
            except Exception:
                p = c = t = 0

            cost = (p / 1000.0) * float(self.config.pricing_in_per_1k) + (c / 1000.0) * float(self.config.pricing_out_per_1k)
            usage.update({
                "prompt_tokens": p,
                "completion_tokens": c,
                "total_tokens": t,
                "usd_cost_estimate": round(cost, 6),
            })
            logger.info("Teaser tokens: prompt=%s, completion=%s, total=%s, est_cost=$%s",
                        p, c, t, round(cost, 6))
            return teaser, usage
        except Exception as e:
            logger.error("OpenAI summary generation failed: %s", e)
            lines = [f"- {a.get('title','Untitled')} — {a.get('url','N/A')}" for a in articles]
            return "Today's highlights:\n" + "\n".join(lines), usage

    def _build_body(self, subject: str, summary_text: str, articles: List[Dict], *, utm_content: str) -> tuple[str, str]:
        items_html = []
        items_text = []
        for a in articles:
            raw_url = str(a.get("url", "#"))
            url = _append_utm(
                raw_url,
                source=self.config.utm_source,
                medium=self.config.utm_medium,
                campaign=self.config.utm_campaign,
                content=utm_content,
            )
            title = str(a.get("title", "Untitled Article"))
            desc = str(a.get("meta_description") or a.get("meta") or "No description available.")

            items_html.append(
                f'<h3 style="margin:16px 0;"><a href="{html.escape(url)}">{html.escape(title)}</a></h3>'
                f'<p style="margin:8px 0;">{html.escape(desc)}</p>'
                f'<p style="margin:8px 0;"><a href="{html.escape(url)}">Read more &raquo;</a></p>'
                '<hr style="border:none;border-top:1px solid #eee;margin:24px 0;"/>'
            )
            items_text.append(f"{title}\n{desc}\n{url}\n")

        articles_html = "\n".join(items_html)
        articles_text = "\n".join(items_text)

        preheader = html.escape(self.config.preheader_text or "")

        html_content = f"""\
<html>
  <body style="font-family:Arial,Helvetica,sans-serif;line-height:1.6;color:#222;">
    <span style="display:none!important;visibility:hidden;opacity:0;height:0;width:0;overflow:hidden;">
      {preheader}
    </span>
    <h1 style="font-size:20px;margin:0 0 12px 0;">{html.escape(subject)}</h1>
    <p>{html.escape(summary_text)}</p>
    <hr style="border:none;border-top:1px solid #eee;margin:24px 0;"/>
    {articles_html}
    <p style="font-size:12px;color:#666;margin-top:28px;">
      You’re receiving this because you subscribed to PythonProHub updates.
      {'<br/>Unsubscribe: ' + html.escape(self.config.list_unsubscribe) if self.config.list_unsubscribe else ''}
    </p>
  </body>
</html>
""".strip()

        text_content = "\n".join(
            [
                subject,
                "",
                self.config.preheader_text or "",
                "",
                summary_text,
                "",
                "-" * 40,
                "",
                articles_text,
                "",
                "You’re receiving this because you subscribed to PythonProHub updates.",
                (f"Unsubscribe: {self.config.list_unsubscribe}" if self.config.list_unsubscribe else ""),
            ]
        ).strip()

        return html_content, text_content

    def _assemble_message(self, subject: str, html_body: str, text_body: str) -> MIMEMultipart:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = formataddr((self.config.sender_name, self.config.sender_email or ""))
        to_list = _as_list(self.config.recipient_email)
        cc_list = _as_list(self.config.cc)
        bcc_list = _as_list(self.config.bcc)

        if to_list:
            msg["To"] = ", ".join(to_list)
        if cc_list:
            msg["Cc"] = ", ".join(cc_list)
        if self.config.reply_to and _valid_email(self.config.reply_to):
            msg["Reply-To"] = self.config.reply_to

        # List-Unsubscribe (both forms if available)
        lu_values = []
        if self.config.list_unsubscribe:
            lu_values.append(_angle(str(self.config.list_unsubscribe)))
        if self.config.one_click_unsubscribe_url:
            lu_values.append(_angle(str(self.config.one_click_unsubscribe_url)))
            msg["List-Unsubscribe-Post"] = "List-Unsubscribe=One-Click"
        if lu_values:
            msg["List-Unsubscribe"] = ", ".join(lu_values)

        # Message-ID for better deliverability/logging
        msg["Message-ID"] = make_msgid(domain=(self.config.sender_email or "pythonprohub.com").split("@")[-1])

        part_txt = MIMEText(text_body, "plain", "utf-8")
        part_html = MIMEText(html_body, "html", "utf-8")
        msg.attach(part_txt)
        msg.attach(part_html)

        # stash for sending
        msg._all_recipients = to_list + cc_list + bcc_list  # type: ignore[attr-defined]
        return msg

    def _send_with_retry(self, msg: MIMEMultipart, max_attempts: int = 3) -> bool:
        recipients: List[str] = getattr(msg, "_all_recipients", [])  # type: ignore[attr-defined]
        if not recipients:
            logger.error("No recipients; aborting send.")
            return False

        port = self.config.smtp_port or (465 if self.config.use_ssl else 587)

        for attempt in range(1, max_attempts + 1):
            try:
                if self.config.use_ssl or port == 465:
                    context = ssl.create_default_context()
                    import smtplib
                    with smtplib.SMTP_SSL(self.config.smtp_server, port, timeout=self.config.timeout, context=context) as server:
                        self._login_and_send(server, msg, recipients)
                else:
                    import smtplib
                    with smtplib.SMTP(self.config.smtp_server, port, timeout=self.config.timeout) as server:
                        if self.config.use_tls or port == 587:
                            server.ehlo()
                            server.starttls(context=ssl.create_default_context())
                            server.ehlo()
                        self._login_and_send(server, msg, recipients)

                logger.info("Newsletter sent to %s (count=%d).", recipients, len(recipients))
                return True
            except Exception as e:
                logger.warning("Send attempt %d/%d failed: %s", attempt, max_attempts, e, exc_info=True)
                if attempt < max_attempts:
                    time.sleep(min(1.5 * attempt, 5.0))
                else:
                    logger.error("Failed to send newsletter after %d attempts.", max_attempts)
        return False

    def _login_and_send(self, server, msg: MIMEMultipart, recipients: List[str]) -> None:
        if self.config.email_username and self.config.email_password:
            server.login(self.config.email_username, self.config.email_password)
        server.sendmail(self.config.sender_email, recipients, msg.as_string())

    # -------------------- Social captions (optional) --------------------
    def _render_social_captions(self, today: str, articles: List[Dict]) -> None:
        out_dir = Path("output/social")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{today}_digest.json"

        top = articles[:3]
        lines = [f"- {a.get('title','Untitled')} — {_append_utm(str(a.get('url','#')), source=self.config.utm_source, medium='social', campaign=self.config.utm_campaign, content=today)}" for a in top]
        base_summary = "\n".join(lines)

        captions: Dict[str, List[str]] = {}
        if not self.openai_client:
            captions = {
                "twitter": [f"Today on PythonProHub:\n{base_summary}"],
                "linkedin": [f"Your daily Python digest:\n{base_summary}"],
            }
        else:
            system = "You write concise, high-signal social captions that feel human and avoid buzzwords and emojis."
            for platform in self.config.social_platforms:
                prompt = (
                    f"Platform: {platform}\n"
                    "Create 2 short captions promoting today's Python digest. "
                    "Include 1–2 links from the list, vary structure, avoid passive voice and hype, no emojis. "
                    "Add one crisp hook per caption.\n\n"
                    f"Links:\n{base_summary}"
                )
                try:
                    resp = self.openai_client.chat.completions.create(
                        model=self.config.openai_model,
                        messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                        max_tokens=260,
                        temperature=0.6,
                    )
                    text = (resp.choices[0].message.content or "").strip()
                    items = [p.strip("-• ").strip() for p in text.split("\n") if p.strip()]
                    captions[platform] = items[:2] if items else []
                except Exception as e:
                    logger.warning("Caption generation failed for %s: %s", platform, e)
                    captions[platform] = []

        payload = {
            "date": today,
            "platforms": self.config.social_platforms,
            "captions": captions,
        }
        try:
            out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
            logger.info("Saved social captions to %s", out_path)
        except Exception as e:
            logger.warning("Failed to save social captions: %s", e)


# ---------------------------------------------------------------------------
# Example direct run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    load_dotenv(find_dotenv(usecwd=True))
    mock_app_settings = {
        "newsletter": {
            "enabled": True,
            "sender_name": os.getenv("SENDER_NAME", "PythonProHub"),
            "sender_email": os.getenv("SENDER_EMAIL") or "test@example.com",
            "recipient_email": os.getenv("RECIPIENT_EMAIL") or "recipient@example.com",
            "smtp_server": os.getenv("SMTP_SERVER") or "smtp.gmail.com",
            "smtp_port": int(os.getenv("SMTP_PORT") or 587),
            "email_username": os.getenv("EMAIL_USER"),
            "email_password": os.getenv("EMAIL_PASS"),
            "min_articles_for_newsletter": 1,
            "use_ssl": os.getenv("SMTP_USE_SSL", "false").lower() in ("1", "true", "yes"),
            "use_tls": os.getenv("SMTP_USE_TLS", "true").lower() in ("1", "true", "yes"),
            "dry_run": os.getenv("NEWSLETTER_DRY_RUN", "0").lower() in ("1", "true", "yes"),
            "list_unsubscribe": os.getenv("LIST_UNSUBSCRIBE"),
            "one_click_unsubscribe_url": os.getenv("ONE_CLICK_UNSUB_URL"),
            "reply_to": os.getenv("REPLY_TO"),  # NEW
            "preheader_text": os.getenv("NEWSLETTER_PREHEADER", "Fresh Python guides, tips, and demos inside."),
            "openai_model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            "openai_enabled": os.getenv("OPENAI_ENABLED", "1").lower() in ("1", "true", "yes"),
            "utm_source": os.getenv("UTM_SOURCE", "pythonprohub"),
            "utm_medium": os.getenv("UTM_MEDIUM", "email"),
            "utm_campaign": os.getenv("UTM_CAMPAIGN", "daily_digest"),
            "social_enabled": os.getenv("SOCIAL_ENABLED", "1").lower() in ("1", "true", "yes"),
            "social_platforms": os.getenv("SOCIAL_PLATFORMS", "twitter,linkedin").split(","),
            "pricing_in_per_1k": os.getenv("OPENAI_PRICE_IN_PER_1K") or 0.0,
            "pricing_out_per_1k": os.getenv("OPENAI_PRICE_OUT_PER_1K") or 0.0,
        },
        "api_keys": {"openai": os.getenv("OPENAI_API_KEY")},
    }

    sender = NewsletterSender(mock_app_settings, openai_client=None)
    test_articles = [
        {
            "title": "Latest Python Tips for Beginners",
            "url": "https://pythonprohub.com/tips",
            "meta_description": "Essential tips to kickstart your Python journey.",
        },
        {
            "title": "Deep Dive into AsyncIO",
            "url": "https://pythonprohub.com/asyncio-deep-dive",
            "meta_description": "Explore advanced concepts in Python's async programming.",
        },
    ]
    report = sender.send_digest_email(test_articles)
    logger.info("Result: %s", report)
