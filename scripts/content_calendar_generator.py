# content_calendar_generator.py

import argparse
import csv
import hashlib
import json
import logging
import math
import os
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# ---------------------------- Logging ---------------------------------
LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "content_calendar_generator.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# ---------------------------- Canonical 20 niches ----------------------
TWENTY_NICHES: List[str] = [
    "Web Development",
    "Data Science and Analytics",
    "Machine Learning and AI",
    "Automation and Scripting",
    "Cybersecurity and Ethical Hacking",
    "Python for Finance",
    "Educational Python",
    "Web Scraping and Data Extraction",
    "Python Tips",
    "Scientific & Numerical Computing",
    "DevOps, Cloud & Infrastructure",
    "Data Engineering & Pipelines",
    "Desktop GUI & Apps",
    "IoT, Embedded & Hardware",
    "Testing, Quality & Types",
    "MLOps & Production AI",
    "Geospatial & GIS",
    "Game Development with Python",
    "APIs & Integrations",
    "Data Visualization & Storytelling",
]

_CANON_MAP = {n.lower(): n for n in TWENTY_NICHES}

# ---------------------------- Helpers ---------------------------------

def _read_json(path: Optional[Path]) -> Optional[Any]:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        logger.warning("JSON file not found: %s", p)
        return None
    try:
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error("Failed to parse JSON at %s: %s", p, e)
        raise

def _atomic_write_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    tmp.replace(path)

def _date_range(start: datetime, end: datetime) -> Iterable[datetime]:
    days = (end - start).days
    for i in range(days + 1):
        yield start + timedelta(days=i)

def _norm_topic(s: str) -> str:
    return " ".join((s or "").lower().split())

def _weekday_name(dt: datetime) -> str:
    return dt.strftime("%A")  # Monday..Sunday

def _stable_id(date: str, niche: str, title: str) -> str:
    # Normalize to ensure stability across minor whitespace/case changes
    return hashlib.sha1(f"{date}|{niche}|{_norm_topic(title)}".encode("utf-8")).hexdigest()

def _safe_int(x: Any, default: int) -> int:
    try:
        return int(x)
    except Exception:
        return default

def _canonicalize_niche(name: str) -> Tuple[str, bool]:
    """Return (canonical_name, changed?). Falls back to original if unknown."""
    if not name:
        return name, False
    low = str(name).strip().lower()
    if low in _CANON_MAP:
        canon = _CANON_MAP[low]
        return canon, canon != name
    return name, False

def canonicalize_pools(pools: Dict[str, List[str]], enable: bool = True) -> Dict[str, List[str]]:
    """Merge keys by canonical 20-niche names (case-insensitive)."""
    if not enable:
        return pools
    merged: Dict[str, List[str]] = {}
    unknown: List[str] = []
    for raw, topics in pools.items():
        canon, changed = _canonicalize_niche(raw)
        if changed:
            logger.info("Canonicalizing niche name '%s' -> '%s'", raw, canon)
        if canon == raw and canon not in _CANON_MAP.values():
            unknown.append(raw)
        merged.setdefault(canon, []).extend(list(topics or []))
    if unknown:
        logger.warning("Unknown niche names (not in 20-niche taxonomy): %s", sorted(set(unknown)))
    # Ensure all 20 niches exist as keys (even if empty) for weights to bind to
    for n in TWENTY_NICHES:
        merged.setdefault(n, [])
    return merged

# ------------------------ Optional KeyphraseTracker --------------------

class _KeyphraseAdapter:
    """
    Optional adapter around your KeyphraseTracker (if importable).
    If not present, we still enforce local cooldown on repeated keyphrases.
    """
    def __init__(self) -> None:
        self.tracker = None
        try:
            from keyphrase_tracker import KeyphraseTracker, KeyphraseConfig  # type: ignore
            self.tracker = KeyphraseTracker(KeyphraseConfig())
            logger.info("KeyphraseTracker detected and will be used for uniqueness.")
        except Exception:
            logger.info("KeyphraseTracker not available; using in-memory keyphrase checks only.")

    def is_used(self, phrase: str) -> bool:
        if not phrase:
            return False
        if self.tracker:
            try:
                return self.tracker.is_keyphrase_used(phrase, check_fuzzy=True)
            except Exception:
                return False
        return False

    def save(self, phrase: str, slug: Optional[str] = None) -> None:
        if self.tracker:
            try:
                self.tracker.save_keyphrase(phrase, associated_slug=slug)
            except Exception:
                pass

# ---------------------------- Config ----------------------------------

DEFAULT_OUTPUT = Path("config/content_calendar.json")

@dataclass
class CalendarConfig:
    start: datetime
    end: datetime
    per_day: int
    output: Path
    seed: int
    strategy: str  # balanced | random (balanced recommended)
    pools_file: Optional[Path]
    weights_file: Optional[Path]
    weights_inline: Optional[str]
    weights: Dict[str, float]
    overrides_file: Optional[Path]
    themes_file: Optional[Path]
    blackouts_file: Optional[Path]
    cooldown_niche: int
    cooldown_keyphrase: int
    max_per_day_per_niche: int
    best_of: int
    validate_only: bool
    dry_run: bool
    resume: bool
    # NEW:
    print_summary: bool
    csv_out: Optional[Path]
    no_weekends: bool              # NEW
    canonicalize_niches: bool      # NEW

# ------------------------ Loading & Preflight --------------------------

def _twenty_niches_fallback_pools() -> Dict[str, List[str]]:
    # Minimal but useful defaults; real runs should provide a pools JSON.
    return {
        "Web Development": [
            "FastAPI REST API in 30 Minutes",
            "Django Caching Strategies (2025)",
            "Flask vs FastAPI: When to Choose Which",
        ],
        "Data Science and Analytics": [
            "EDA with pandas: a practical flow",
            "Polars vs pandas: Speed Showdown",
            "Great Expectations quickstart",
        ],
        "Machine Learning and AI": [
            "From scikit-learn to PyTorch: a gentle path",
            "Feature Engineering patterns in 2025",
            "Hyperparameter tuning with Optuna",
        ],
        "Automation and Scripting": [
            "Task automation with Python + cron",
            "click vs argparse: CLI ergonomics",
            "Automate spreadsheets with openpyxl",
        ],
        "Cybersecurity and Ethical Hacking": [
            "Intro to Python pentesting tooling",
            "Hashing & salting correctly in Python",
            "Secure secrets handling with dotenv",
        ],
        "Python for Finance": [
            "Backtesting with vectorbt",
            "Risk metrics with NumPy & pandas",
            "Quandl/Alpha Vantage data pipelines",
        ],
        "Educational Python": [
            "Teaching Python with Jupyter",
            "Common beginner pitfalls explained",
            "Writing testable examples for students",
        ],
        "Web Scraping and Data Extraction": [
            "Playwright vs Selenium for scraping",
            "Respectful scraping & rate limiting",
            "Parse HTML with Selectolax/lxml",
        ],
        "Python Tips": [
            "5 powerful uses of enumerate()",
            "Mastering f-strings",
            "functools.lru_cache deep dive",
        ],
        "Scientific & Numerical Computing": [
            "NumPy broadcasting mental model",
            "Numba jit: when it helps",
            "SciPy optimizers overview",
        ],
        "DevOps, Cloud & Infrastructure": [
            "Packaging with uv/poetry in CI",
            "Docker multi-stage builds for Python",
            "AWS Lambda + Python best practices",
        ],
        "Data Engineering & Pipelines": [
            "Airflow vs Prefect in 2025",
            "Delta Lake & Parquet essentials",
            "Data validation patterns in ETL",
        ],
        "Desktop GUI & Apps": [
            "PySide6: building a settings panel",
            "Packaging desktop apps with Briefcase",
            "Tkinter: modern styling tips",
        ],
        "IoT, Embedded & Hardware": [
            "MicroPython on ESP32 quickstart",
            "Raspberry Pi sensor logging",
            "Async IO for hardware polling",
        ],
        "Testing, Quality & Types": [
            "pytest fixtures you’ll reuse",
            "Type hints beyond basics (mypy)",
            "Property-based tests with hypothesis",
        ],
        "MLOps & Production AI": [
            "Model versioning with MLflow",
            "Feature stores overview",
            "Serving with FastAPI + BentoML",
        ],
        "Geospatial & GIS": [
            "GeoPandas basics",
            "Rasterio quick tour",
            "Interactive maps with Folium",
        ],
        "Game Development with Python": [
            "Arcade/Pygame shooter in 150 lines",
            "Game loops & delta time",
            "Sprite collisions explained",
        ],
        "APIs & Integrations": [
            "OAuth2 clients with httpx",
            "Retry & backoff patterns",
            "GraphQL clients in Python",
        ],
        "Data Visualization & Storytelling": [
            "Matplotlib fundamentals that scale",
            "Altair declarative charts",
            "Plotly interactive dashboards",
        ],
    }

def load_pools(pools_file: Optional[Path]) -> Dict[str, List[str]]:
    pools = _read_json(pools_file)
    if isinstance(pools, dict) and pools:
        fixed: Dict[str, List[str]] = {}
        for niche, arr in pools.items():
            if isinstance(arr, list):
                fixed[str(niche)] = [str(x).strip() for x in arr if str(x).strip()]
        if fixed:
            return fixed
    logger.warning("Using fallback topic pools for the 20-niche taxonomy (provide a pools JSON for real runs).")
    return _twenty_niches_fallback_pools()

def dedupe_pools(pools: Dict[str, List[str]]) -> Tuple[Dict[str, List[str]], int, int]:
    within_removed = 0
    cross_removed = 0
    seen_global = set()
    out: Dict[str, List[str]] = {}
    for niche, topics in pools.items():
        seen_local = set()
        uniq = []
        for t in topics:
            key = _norm_topic(t)
            if key in seen_local:
                within_removed += 1
                continue
            seen_local.add(key)
            uniq.append(t)
        final = []
        for t in uniq:
            key = _norm_topic(t)
            if key in seen_global:
                cross_removed += 1
                continue
            seen_global.add(key)
            final.append(t)
        out[niche] = final
    return out, within_removed, cross_removed

def apply_exclusions(pools: Dict[str, List[str]], overrides: Optional[Dict[str, Any]]) -> Dict[str, List[str]]:
    if not overrides:
        return pools
    excl = set(_norm_topic(x) for x in overrides.get("exclude_topics", []) if isinstance(x, str))
    if not excl:
        return pools
    out: Dict[str, List[str]] = {}
    for niche, topics in pools.items():
        out[niche] = [t for t in topics if _norm_topic(t) not in excl]
    return out

def preflight(cfg: CalendarConfig,
              pools: Dict[str, List[str]],
              blackouts: Optional[Dict[str, Any]],
              overrides: Optional[Dict[str, Any]]) -> None:
    # Validate core args
    if cfg.end < cfg.start:
        raise SystemExit("End date must be on/after start date.")
    if cfg.per_day <= 0:
        raise SystemExit("--per-day must be > 0.")
    if cfg.max_per_day_per_niche < 0:
        raise SystemExit("--max-per-day-per-niche must be >= 0.")

    # Effective schedulable days
    blocked_dates = set()
    if blackouts:
        blocked_dates |= set(blackouts.get("date_blackouts", []))
    if overrides:
        blocked_dates |= set(overrides.get("exclude_dates", []))

    def _schedulable(d: datetime) -> bool:
        if cfg.no_weekends and d.weekday() >= 5:
            return False
        return d.strftime("%Y-%m-%d") not in blocked_dates

    total_days = sum(1 for d in _date_range(cfg.start, cfg.end) if _schedulable(d))
    total_needed = total_days * cfg.per_day
    total_available = sum(len(v) for v in pools.values())

    unknown = [k for k in pools.keys() if k not in _CANON_MAP.values()]
    if unknown:
        logger.warning("Preflight: %d niche name(s) are not in the 20-niche taxonomy: %s", len(unknown), sorted(unknown))

    logger.info(
        "Preflight: days=%d | per_day=%d | needed=%d | available=%d | niches=%d | no_weekends=%s",
        total_days, cfg.per_day, total_needed, total_available, len(pools), cfg.no_weekends,
    )
    if total_available < total_needed:
        raise SystemExit(f"Not enough topics overall: need {total_needed}, have {total_available}.")

    if cfg.strategy == "balanced":
        per_niche_req = math.ceil(total_needed / max(1, len(pools)))
        for niche, arr in pools.items():
            if len(arr) < per_niche_req:
                logger.warning(
                    "Niche '%s' has %d topics; balanced scheduling may underflow (needs ~%d).",
                    niche, len(arr), per_niche_req,
                )

# ----------------------------- Themes/Weights --------------------------

def merge_weights(base: Dict[str, float],
                  day_overrides: Optional[Dict[str, Any]],
                  allowed: Optional[List[str]]) -> Dict[str, float]:
    w = dict(base)
    if allowed:
        for k in list(w.keys()):
            if k not in allowed:
                w[k] = 0.0
    if day_overrides and isinstance(day_overrides.get("weights"), dict):
        for k, v in day_overrides["weights"].items():
            try:
                w[k] = float(v)
            except Exception:
                pass
    for k in list(w.keys()):
        w[k] = max(0.0, float(w[k]))
    return w

def weights_from_args(weights_file: Optional[Path],
                      weights_inline: Optional[str],
                      pools: Dict[str, List[str]]) -> Dict[str, float]:
    w: Dict[str, float] = {}
    src = None
    unknown_warned = False
    if weights_file:
        data = _read_json(weights_file)
        if isinstance(data, dict):
            for k, v in data.items():
                try:
                    w[str(k)] = float(v)
                except Exception:
                    pass
            src = weights_file
    if (not w) and weights_inline:
        try:
            data = json.loads(weights_inline)
            if isinstance(data, dict):
                for k, v in data.items():
                    try:
                        w[str(k)] = float(v)
                    except Exception:
                        pass
                src = "CLI --weights"
        except Exception as e:
            logger.error("Failed to parse --weights JSON: %s", e)
    if not w:
        w = {k: 1.0 for k in pools.keys()}
        src = "uniform"
    else:
        # Warn on weights for unknown niches and ensure all pool niches exist
        for k in list(w.keys()):
            if k not in pools:
                if not unknown_warned:
                    logger.warning("Weights contain unknown niche '%s' (not in pools); it will be ignored.", k)
                unknown_warned = True
                del w[k]
        for k in pools.keys():
            w.setdefault(k, 1.0)
    logger.info("Using niche weights from %s.", src)
    return w

# ----------------------------- Scheduling --------------------------------

class Scheduler:
    def __init__(self,
                 cfg: CalendarConfig,
                 pools: Dict[str, List[str]],
                 weights: Dict[str, float],
                 overrides: Optional[Dict[str, Any]],
                 themes: Optional[Dict[str, Any]],
                 blackouts: Optional[Dict[str, Any]],
                 kpadapter: _KeyphraseAdapter):
        import random
        self.cfg = cfg
        self.rng = random.Random(cfg.seed)
        self.pools: Dict[str, List[str]] = {k: list(v) for k, v in pools.items()}
        for niche in self.pools:
            self.rng.shuffle(self.pools[niche])  # deterministic
        self.weights_base = weights
        self.overrides = overrides or {}
        self.themes = themes or {}
        self.blackouts = blackouts or {}
        self.kpadapter = kpadapter

        self.last_day_by_niche: Dict[str, Optional[datetime]] = {k: None for k in self.pools}
        self.last_day_by_keyphrase: Dict[str, Optional[datetime]] = {}
        self.used_topic_keys: set[str] = set()
        self.violations_niche_cd = 0
        self.violations_kp_cd = 0
        self.violations_theme = 0

        self.existing_rows: List[Dict[str, Any]] = []
        if self.cfg.resume and self.cfg.output.exists():
            try:
                existing = _read_json(self.cfg.output) or []
                if isinstance(existing, list):
                    self.existing_rows = existing
                    for row in existing:
                        try:
                            dt = datetime.fromisoformat(row["date"])
                        except Exception:
                            continue
                        niche = row.get("niche", "")
                        topic = row.get("title", "")
                        key = _norm_topic(topic)
                        self.used_topic_keys.add(key)
                        prev = self.last_day_by_niche.get(niche)
                        self.last_day_by_niche[niche] = max(prev, dt) if prev else dt
                        self.last_day_by_keyphrase[_norm_topic(row.get("keyphrase", topic))] = dt
                logger.info("Resume: loaded %d existing rows from %s.", len(self.existing_rows), self.cfg.output)
            except Exception as e:
                logger.warning("Resume requested but failed to parse existing output: %s", e)

    def _is_date_blocked(self, d: datetime, ds: str) -> bool:
        if self.cfg.no_weekends and d.weekday() >= 5:
            return True
        if ds in set(self.blackouts.get("date_blackouts", [])):
            return True
        if ds in set(self.overrides.get("exclude_dates", [])):
            return True
        return False

    def _day_theme(self, d: datetime) -> Tuple[Optional[List[str]], Dict[str, Any]]:
        day_cfg = {}
        allowed = None
        dc = (self.themes.get("dates") or {}).get(d.strftime("%Y-%m-%d"))
        if isinstance(dc, dict):
            day_cfg = dict(dc)
            allowed = list(dc.get("allow", [])) if isinstance(dc.get("allow"), list) else None
        wc = (self.themes.get("weekdays") or {}).get(_weekday_name(d))
        if isinstance(wc, dict):
            if "allow" in wc and isinstance(wc["allow"], list):
                allowed = list(set(allowed or []) & set(wc["allow"])) if allowed else list(wc["allow"])
            if "weights" in wc and isinstance(wc["weights"], dict):
                if "weights" in day_cfg and isinstance(day_cfg["weights"], dict):
                    day_cfg["weights"].update(wc["weights"])
                else:
                    day_cfg["weights"] = dict(wc["weights"])
        return allowed, day_cfg

    def _is_niche_blocked(self, ds: str, niche: str) -> bool:
        if niche in set(self.blackouts.get("global_niche_blackouts", [])):
            return True
        nb = self.blackouts.get("niche_blackouts", {}).get(ds)
        if isinstance(nb, list) and niche in nb:
            return True
        return False

    def _pop_topic_respecting_keyphrase(self, niche: str, today: datetime) -> Optional[str]:
        cd = self.cfg.cooldown_keyphrase
        arr = self.pools.get(niche, [])

        # First pass: respect cooldown AND skip phrases marked used by external tracker
        for i in range(len(arr)):
            topic = arr[i]
            tkey = _norm_topic(topic)
            if tkey in self.used_topic_keys:
                continue
            if self.kpadapter.is_used(topic):
                continue
            last = self.last_day_by_keyphrase.get(tkey)
            if (last is None) or ((today - last).days >= cd):
                del arr[i]
                return topic

        # Second pass: allow cooldown violation, still avoid duplicates this run
        for i in range(len(arr)):
            topic = arr[i]
            tkey = _norm_topic(topic)
            if tkey in self.used_topic_keys:
                continue
            del arr[i]
            self.violations_kp_cd += 1
            return topic

        return None

    def _weighted_cycle(self, weights: Dict[str, float]) -> List[str]:
        items: List[str] = []
        for niche, w in weights.items():
            if w <= 0.0:
                continue
            rep = max(1, int(round(w)))
            items.extend([niche] * rep)
        self.rng.shuffle(items)
        order: List[str] = []
        for n in items:
            if not order or order[-1] != n:
                order.append(n)
        if not order:
            order = [n for n, arr in self.pools.items() if arr]
            self.rng.shuffle(order)
        return order

    def _max_per_day_niche_ok(self, rows_today: List[Dict[str, Any]], niche: str) -> bool:
        if self.cfg.max_per_day_per_niche <= 0:
            return True
        used = sum(1 for r in rows_today if r["niche"] == niche)
        return used < self.cfg.max_per_day_per_niche

    def _apply_pin_rows(self, today: datetime, rows_today: List[Dict[str, Any]], used_topics_today: set) -> None:
        pins = [p for p in (self.overrides.get("pins") or [])
                if str(p.get("date")) == today.strftime("%Y-%m-%d")]
        for p in pins:
            niche = p.get("niche")
            topic = p.get("topic")
            if not (niche and topic):
                continue
            if self._is_niche_blocked(today.strftime("%Y-%m-%d"), niche):
                logger.warning("Pin ignored (niche blacked out): %s | %s", today.strftime("%Y-%m-%d"), niche)
                continue
            if not self._max_per_day_niche_ok(rows_today, niche):
                logger.warning("Pin ignored (hit per-day-per-niche cap): %s | %s", today.strftime("%Y-%m-%d"), niche)
                continue
            key = _norm_topic(topic)
            if key in self.used_topic_keys or key in used_topics_today:
                logger.warning("Pin ignored (duplicate topic): %s | %s", today.strftime("%Y-%m-%d"), topic)
                continue
            kp = _norm_topic(p.get("keyphrase") or topic)
            last_kp = self.last_day_by_keyphrase.get(kp)
            if last_kp and (today - last_kp).days < self.cfg.cooldown_keyphrase:
                self.violations_kp_cd += 1
            last_n = self.last_day_by_niche.get(niche)
            if last_n and (today - last_n).days < self.cfg.cooldown_niche:
                self.violations_niche_cd += 1

            row = make_row(today, niche, topic)
            row["keyphrase"] = p.get("keyphrase") or topic
            rows_today.append(row)
            used_topics_today.add(key)
            self.used_topic_keys.add(key)
            self.last_day_by_keyphrase[kp] = today
            self.last_day_by_niche[niche] = today

    def generate(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []

        start_date = self.cfg.start
        existing_by_date: Dict[str, List[Dict[str, Any]]] = {}
        if self.existing_rows:
            for r in self.existing_rows:
                if "date" in r:
                    existing_by_date.setdefault(r["date"], []).append(r)
            for ds, rows in sorted(existing_by_date.items()):
                try:
                    d = datetime.fromisoformat(ds)
                except Exception:
                    continue
                if self.cfg.start <= d <= self.cfg.end:
                    out.extend(rows)
            in_range = [datetime.fromisoformat(ds) for ds in existing_by_date.keys()
                        if self.cfg.start <= datetime.fromisoformat(ds) <= self.cfg.end]
            if in_range:
                start_date = max(in_range) + timedelta(days=1)

        for today in _date_range(start_date, self.cfg.end):
            ds = today.strftime("%Y-%m-%d")
            if self._is_date_blocked(today, ds):
                logger.info("Skipping %s (blocked/weekend).", ds)
                continue

            rows_today: List[Dict[str, Any]] = []
            used_topics_today: set[str] = set()

            self._apply_pin_rows(today, rows_today, used_topics_today)
            if len(rows_today) > self.cfg.per_day:
                rows_today = rows_today[: self.cfg.per_day]

            if len(rows_today) < self.cfg.per_day:
                remaining = self.cfg.per_day - len(rows_today)
                allowed, day_cfg = self._day_theme(today)

                # Strategy: random ignores numeric weights but still honors "allowed"
                if self.cfg.strategy == "random":
                    weights_today = {
                        k: (1.0 if (not allowed or k in allowed) else 0.0)
                        for k in self.pools.keys()
                    }
                else:
                    weights_today = merge_weights(self.weights_base, day_cfg, allowed)

                order = self._weighted_cycle(weights_today)

                attempts = 0
                guard = 20 * max(1, len(order))
                while remaining > 0 and attempts < guard:
                    attempts += 1
                    picked = False
                    for niche in order:
                        if remaining == 0:
                            break
                        if self._is_niche_blocked(ds, niche):
                            continue
                        if not self.pools.get(niche):
                            continue
                        if not self._max_per_day_niche_ok(rows_today, niche):
                            continue
                        last = self.last_day_by_niche.get(niche)
                        if last and (today - last).days < self.cfg.cooldown_niche:
                            continue
                        topic = self._pop_topic_respecting_keyphrase(niche, today)
                        if not topic:
                            continue
                        if allowed and niche not in allowed:
                            self.violations_theme += 1
                        row = make_row(today, niche, topic)
                        rows_today.append(row)
                        used_topics_today.add(_norm_topic(topic))
                        self.used_topic_keys.add(_norm_topic(topic))
                        self.last_day_by_niche[niche] = today
                        self.last_day_by_keyphrase[_norm_topic(row["keyphrase"])] = today
                        remaining -= 1
                        picked = True
                        try:
                            self.kpadapter.save(row["keyphrase"], slug=None)
                        except Exception:
                            pass
                    if not picked:
                        relaxed = False
                        for niche in order:
                            if remaining == 0:
                                break
                            if self._is_niche_blocked(ds, niche):
                                continue
                            if not self._max_per_day_niche_ok(rows_today, niche):
                                continue
                            if not self.pools.get(niche):
                                continue
                            topic = self._pop_topic_respecting_keyphrase(niche, today)
                            if not topic:
                                continue
                            self.violations_niche_cd += 1
                            if allowed and niche not in (allowed or []):
                                self.violations_theme += 1
                            row = make_row(today, niche, topic)
                            rows_today.append(row)
                            self.used_topic_keys.add(_norm_topic(topic))
                            self.last_day_by_niche[niche] = today
                            self.last_day_by_keyphrase[_norm_topic(row["keyphrase"])] = today
                            remaining -= 1
                            relaxed = True
                        if not relaxed:
                            break

            out.extend(rows_today)

        return out

# ----------------------------- Scoring ----------------------------------

def score_calendar(rows: List[Dict[str, Any]],
                   cooldown_niche: int,
                   cooldown_kp: int,
                   max_per_day_per_niche: int) -> float:
    """
    Lower is better. Penalize cooldown violations, day imbalance & niche imbalance.
    """
    by_day: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_day.setdefault(r["date"], []).append(r)

    penalty = 0.0
    niche_counts_total: Dict[str, int] = {}
    last_date_by_niche: Dict[str, Optional[datetime]] = {}
    last_date_by_kp: Dict[str, Optional[datetime]] = {}

    for ds in sorted(by_day.keys()):
        day_rows = by_day[ds]
        per_niche: Dict[str, int] = {}
        for r in day_rows:
            ni = r["niche"]
            per_niche[ni] = per_niche.get(ni, 0) + 1
            niche_counts_total[ni] = niche_counts_total.get(ni, 0) + 1

            try:
                dt = datetime.fromisoformat(ds)
            except Exception:
                continue
            last_n = last_date_by_niche.get(ni)
            if last_n and (dt - last_n).days < cooldown_niche:
                penalty += 2.0
            last_date_by_niche[ni] = dt

            kp = _norm_topic(r.get("keyphrase") or r.get("title"))
            last_k = last_date_by_kp.get(kp)
            if last_k and (dt - last_k).days < cooldown_kp:
                penalty += 1.0
            last_date_by_kp[kp] = dt

        for ni, c in per_niche.items():
            if c > max_per_day_per_niche > 0:
                penalty += 0.5 * (c - max_per_day_per_niche)

        if day_rows:
            unique_niches = len(per_niche)
            diversity_score = len(day_rows) - unique_niches
            penalty += 0.05 * diversity_score

    if niche_counts_total:
        counts = list(niche_counts_total.values())
        mean = sum(counts) / len(counts)
        variance = sum((c - mean) ** 2 for c in counts) / len(counts)
        penalty += 0.02 * variance

    return penalty

# ----------------------------- Rows ------------------------------------

def make_row(date_obj: datetime, niche: str, topic: str) -> Dict[str, Any]:
    ds = date_obj.strftime("%Y-%m-%d")
    row = {
        "id": _stable_id(ds, niche, topic),
        "date": ds,
        "niche": niche,
        "title": topic,
        "keyphrase": topic,
        "priority": 1,
    }
    return row

# ----------------------------- Summaries/CSV ---------------------------

def _log_summary(rows: List[Dict[str, Any]]) -> None:
    by_niche = Counter(r["niche"] for r in rows)
    by_day = Counter(r["date"] for r in rows)
    logger.info("Summary: rows=%d | days=%d | niches=%d",
                len(rows), len(by_day), len(by_niche))
    logger.info("Per-niche: %s",
                dict(sorted(by_niche.items(), key=lambda x: (-x[1], x[0]))))
    first_10_days = dict(sorted(by_day.items())[:10])
    logger.info("First 10 days: %s", first_10_days)

def _write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["id", "date", "niche", "title", "keyphrase", "priority"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    logger.info("Wrote CSV %s", path)

# ----------------------------- CLI -------------------------------------

def parse_args() -> CalendarConfig:
    p = argparse.ArgumentParser(description="Generate a content calendar JSON.")
    p.add_argument("--start", type=str, required=True, help="Start date (YYYY-MM-DD)")
    p.add_argument("--end", type=str, required=True, help="End date (YYYY-MM-DD)")
    p.add_argument("--per-day", type=int, default=5, help="Articles per day")
    p.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT), help="Output JSON path")
    p.add_argument("--seed", type=int, default=1337, help="PRNG seed for shuffling")
    p.add_argument("--strategy", choices=["balanced", "random"], default="balanced", help="Scheduling strategy")
    p.add_argument("--pools-file", type=str, default="config/topic_pools.json", help="Path to topic pools JSON")

    p.add_argument("--weights", type=str, default=None, help="Inline JSON mapping of niche->weight")
    p.add_argument("--weights-file", type=str, default=None, help="Path to niche weights JSON")

    p.add_argument("--overrides-file", type=str, default="config/calendar_overrides.json", help="Pins/exclusions JSON")
    p.add_argument("--themes-file", type=str, default="config/calendar_themes.json", help="Weekday/date themes JSON")
    p.add_argument("--blackouts-file", type=str, default="config/calendar_blackouts.json", help="Blackouts JSON")

    p.add_argument("--cooldown-niche", type=int, default=3, help="Min days between same niche")
    p.add_argument("--cooldown-keyphrase", type=int, default=28, help="Min days between same keyphrase")
    p.add_argument("--max-per-day-per-niche", type=int, default=1, help="Max items per niche per day (0 = unlimited)")

    p.add_argument("--best-of", type=int, default=1, help="Generate N candidates with different seeds and pick the best")
    p.add_argument("--validate-only", action="store_true", help="Only validate inputs and print a summary; do not write")
    p.add_argument("--dry-run", action="store_true", help="Compute but do not write output")
    p.add_argument("--resume", action="store_true", help="Merge with existing output; continue from last date within range")

    # NEW:
    p.add_argument("--print-summary", action="store_true",
                   help="Log counts per niche/day after generation")
    p.add_argument("--csv-out", type=str, default=None,
                   help="Optional CSV path to export calendar rows")
    p.add_argument("--no-weekends", action="store_true",
                   help="Skip Saturdays/Sundays automatically (in addition to blackouts)")
    p.add_argument("--canonicalize-niches", dest="canonicalize_niches", action="store_true",
                   help="Map pool keys to the canonical 20-niche taxonomy (default on)")
    p.add_argument("--no-canonicalize-niches", dest="canonicalize_niches", action="store_false",
                   help="Disable niche canonicalization")

    p.set_defaults(canonicalize_niches=True)

    a = p.parse_args()
    return CalendarConfig(
        start=datetime.fromisoformat(a.start),
        end=datetime.fromisoformat(a.end),
        per_day=int(a.per_day),
        output=Path(a.output),
        seed=int(a.seed),
        strategy=a.strategy,
        pools_file=Path(a.pools_file) if a.pools_file else None,
        weights_file=Path(a.weights_file) if a.weights_file else None,
        weights_inline=a.weights if a.weights else None,
        weights={},  # set after pools load
        overrides_file=Path(a.overrides_file) if a.overrides_file else None,
        themes_file=Path(a.themes_file) if a.themes_file else None,
        blackouts_file=Path(a.blackouts_file) if a.blackouts_file else None,
        cooldown_niche=_safe_int(a.cooldown_niche, 3),
        cooldown_keyphrase=_safe_int(a.cooldown_keyphrase, 28),
        max_per_day_per_niche=_safe_int(a.max_per_day_per_niche, 1),
        best_of=max(1, int(a.best_of)),
        validate_only=bool(a.validate_only),
        dry_run=bool(a.dry_run),
        resume=bool(a.resume),
        # NEW:
        print_summary=bool(a.print_summary),
        csv_out=Path(a.csv_out) if a.csv_out else None,
        no_weekends=bool(a.no_weekends),
        canonicalize_niches=bool(a.canonicalize_niches),
    )

# ----------------------------- Main ------------------------------------

def main() -> None:
    cfg = parse_args()

    pools_raw = load_pools(cfg.pools_file)
    # Canonicalize pool keys to 20-niche taxonomy (case-insensitive)
    pools_raw = canonicalize_pools(pools_raw, enable=cfg.canonicalize_niches)

    overrides = _read_json(cfg.overrides_file) or {}
    themes = _read_json(cfg.themes_file) or {}
    blackouts = _read_json(cfg.blackouts_file) or {}

    pools_raw, wdup, xdup = dedupe_pools(pools_raw)
    if wdup or xdup:
        logger.info("Dedupe: removed %d within-niche and %d cross-niche duplicates.", wdup, xdup)
    pools = apply_exclusions(pools_raw, overrides)

    cfg.weights = weights_from_args(cfg.weights_file, cfg.weights_inline or os.environ.get("CALENDAR_WEIGHTS_JSON"), pools)

    preflight(cfg, pools, blackouts, overrides)
    if cfg.validate_only:
        logger.info("Validate-only: inputs look good. Exiting without generation.")
        return

    candidates: List[Tuple[float, List[Dict[str, Any]]]] = []
    kp_adapter = _KeyphraseAdapter()

    for i in range(cfg.best_of):
        seed_variant = cfg.seed + i * 97
        cfg_variant = CalendarConfig(**{**cfg.__dict__, "seed": seed_variant})

        sched = Scheduler(cfg_variant, pools, cfg.weights, overrides, themes, blackouts, kp_adapter)
        rows = sched.generate()
        sc = score_calendar(rows, cfg.cooldown_niche, cfg.cooldown_keyphrase, cfg.max_per_day_per_niche)
        logger.info(
            "Candidate %d/%d seed=%d -> rows=%d score=%.3f (violations: niche=%d kp=%d theme=%d)",
            i + 1, cfg.best_of, seed_variant, len(rows), sc,
            sched.violations_niche_cd, sched.violations_kp_cd, sched.violations_theme
        )
        candidates.append((sc, rows))

    candidates.sort(key=lambda x: x[0])
    best_score, best_rows = candidates[0]
    logger.info("Selected best candidate: score=%.3f, rows=%d", best_score, len(best_rows))

    if cfg.print_summary:
        _log_summary(best_rows)

    if cfg.dry_run:
        logger.info("Dry-run: not writing output.")
        if cfg.csv_out:
            _write_csv(best_rows, cfg.csv_out)
        return

    _atomic_write_json(best_rows, cfg.output)
    logger.info("Wrote %s with %d rows.", cfg.output, len(best_rows))

    if cfg.csv_out:
        _write_csv(best_rows, cfg.csv_out)

if __name__ == "__main__":
    main()
