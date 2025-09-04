# topic_collector.py

import argparse
import json
import logging
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Iterable, Tuple, Set, Optional
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup  # GitHub Trending & fallback HTML parsing
from xml.etree import ElementTree as ET  # simple RSS/Atom parsing

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------
LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "topic_collector.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("topic_collector")

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
DEFAULT_OUTPUT = Path("config/topic_pools.json")
SEEN_FILE = Path("cache/seen_topics.json")

DEFAULT_UA = "pythonprohub-topic-collector/1.1 (+https://pythonprohub.com)"
HEADERS = {"User-Agent": DEFAULT_UA}

# ------------- Niches (all 20) ---------------------------------------
NICHES: List[str] = [
    # Original 9
    "Web Development",
    "Data Science and Analytics",
    "Machine Learning and AI",
    "Automation and Scripting",
    "Cybersecurity and Ethical Hacking",
    "Python for Finance",
    "Educational Python",
    "Web Scraping and Data Extraction",
    "Python Tips",
    # New 11
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

# ------------- Keyword routing (fast heuristic) ----------------------
# You can extend/tune these in-place.
ROUTING: Dict[str, Set[str]] = {
    "Web Development": {
        "flask", "django", "fastapi", "asgi", "starlette", "wsgi",
        "websocket", "rest", "api", "oauth", "frontend", "http", "graphql",
        "gunicorn", "nginx", "celery", "jinja", "drf", "htmx", "web",
        "cors", "pydantic", "streamlit", "dash", "uvicorn",
    },
    "Data Science and Analytics": {
        "pandas", "numpy", "matplotlib", "seaborn", "polars", "scipy",
        "eda", "statistics", "dataframe", "plotly", "vaex", "duckdb",
        "analytics", "etl", "parquet", "arrow", "bi", "dashboard",
    },
    "Machine Learning and AI": {
        "pytorch", "tensorflow", "keras", "scikit", "scikit-learn",
        "xgboost", "lightgbm", "transformer", "llm", "gpt", "rag", "bert",
        "diffusion", "yolo", "ml", "deep learning", "vision", "nlp",
        "torch", "autogen", "wandb", "weights & biases", "embedding",
    },
    "Automation and Scripting": {
        "selenium", "playwright", "argparse", "smtplib", "watchdog",
        "cron", "subprocess", "cli", "automation", "script", "pyautogui",
        "pydub", "openpyxl", "pyinstaller", "apscheduler", "rich",
        "pathlib", "logging", "requests", "httpx",
    },
    "Cybersecurity and Ethical Hacking": {
        "security", "xss", "sql injection", "ctf", "exploit", "malware",
        "phishing", "yara", "forensics", "nmap", "scapy", "csrf",
        "bug bounty", "hardening", "threat", "vulnerability", "owasp",
    },
    "Python for Finance": {
        "finance", "trading", "backtesting", "yfinance", "alpaca",
        "quant", "portfolio", "risk", "var", "alpha", "factor", "market",
        "arima", "garch", "sharpe", "options", "alpaca-trade-api",
    },
    "Educational Python": {
        "beginner", "tutorial", "guide", "how to", "learn", "basics",
        "classes", "dataclass", "typing", "pep", "dunder", "exception",
        "comprehension", "generator", "decorator", "context manager",
    },
    "Web Scraping and Data Extraction": {
        "beautifulsoup", "scrapy", "playwright", "selenium", "crawler",
        "scrape", "web scraping", "requests-html", "http", "parsing",
        "bs4", "xpath", "selector",
    },
    "Python Tips": {
        "tip", "trick", "pythonic", "idiomatic", "one-liner", "f-string",
        "enumerate", "zip", "lru_cache", "itertools", "deque", "walrus",
        "refactor", "list comprehension",
    },
    # New 11 buckets
    "Scientific & Numerical Computing": {
        "numba", "jax", "scipy", "sympy", "fortran", "lapack", "blas",
        "finite element", "ode", "pde", "numexpr", "cython", "mpi4py",
    },
    "DevOps, Cloud & Infrastructure": {
        "docker", "kubernetes", "k8s", "helm", "terraform", "ansible",
        "aws", "gcp", "azure", "lambda", "cloud run", "cloudwatch",
        "prometheus", "grafana", "sentry", "ci/cd", "github actions",
    },
    "Data Engineering & Pipelines": {
        "apache spark", "pyspark", "airflow", "dbt", "orchestration",
        "kafka", "flink", "beam", "ingestion", "elt", "olap", "iceberg",
        "delta lake", "medallion", "warehouse", "lakehouse",
    },
    "Desktop GUI & Apps": {
        "pyqt", "pyside", "tkinter", "kivy", "wxpython", "flet",
        "electron", "desktop app", "gui",
    },
    "IoT, Embedded & Hardware": {
        "raspberry pi", "microcontroller", "micropython", "arduino",
        "sensor", "gpio", "serial", "uart", "spi", "i2c", "edge",
    },
    "Testing, Quality & Types": {
        "pytest", "unit test", "coverage", "property-based", "hypothesis",
        "mypy", "ruff", "flake8", "black", "typing", "type checker",
        "lint", "static analysis", "mutation testing",
    },
    "MLOps & Production AI": {
        "mlflow", "bentoml", "triton", "torchserve", "kserve",
        "feature store", "monitoring", "drift", "prompt", "rag", "vector db",
        "llamaindex", "langchain", "modal", "vertex ai", "sagemaker",
    },
    "Geospatial & GIS": {
        "geopandas", "shapely", "folium", "rasterio", "gdal", "geocoding",
        "qgis", "cartography", "geospatial", "srtm", "geotiff",
    },
    "Game Development with Python": {
        "pygame", "arcade", "panda3d", "godot", "unity python", "opengl",
        "shader", "sprite", "collision", "game loop",
    },
    "APIs & Integrations": {
        "zapier", "slack api", "discord bot", "notion api", "google api",
        "webhook", "stripe", "stripe api", "payment", "oauth2", "grpc", "soap",
    },
    "Data Visualization & Storytelling": {
        "matplotlib", "seaborn", "altair", "bokeh", "plotly", "holoviews",
        "storytelling", "chart", "viz", "ggplot", "sankey", "heatmap",
    },
}

# Python relevance filter
PYTHON_FILTER = re.compile(
    r"\bpython\b|flask|django|fastapi|pandas|numpy|pytorch|tensorflow|scikit|polars|asyncio|pytest|jupyter|notebook|duckdb|scrapy|selenium|playwright|pydantic|sqlalchemy|airflow|dbt|pyqt|kivy|geopandas|shapely",
    re.I,
)


@dataclass
class CollectConfig:
    output: Path = DEFAULT_OUTPUT
    per_niche: int = 150        # cap NEW items per niche (this run)
    max_per_source: int = 60    # cap per source before routing/dedupe
    total_cap: int = 300        # cap per niche in final file
    dry_run: bool = False
    sleep_ms: int = 120         # base delay between calls (politeness)
    jitter_ms: int = 80         # random jitter added to each delay
    retries: int = 2            # simple retry count per request
    timeout_s: int = 15         # network timeout per request
    sources: List[str] = None   # None => default set; or list of names


# ---------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------
def _norm_title(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

def _fingerprint(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()

def _load_json(path: Path) -> Optional[dict]:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("Failed to read %s: %s", path, e)
    return None

def _save_json_atomic_map(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)

def _extend_unique(dst: List[str], src: Iterable[str], cap: int, already: Set[str]) -> None:
    for t in src:
        if len(dst) >= cap:
            break
        fp = _fingerprint(t)
        if fp and fp not in already:
            dst.append(t)
            already.add(fp)

def _route_niche(title: str) -> str:
    t = title.lower()
    for niche, kws in ROUTING.items():
        if any(kw in t for kw in kws):
            return niche
    if "python" in t:
        return "Educational Python"
    return "Educational Python"

def _python_relevant(title: str) -> bool:
    return bool(PYTHON_FILTER.search(title))

def _sleep_polite(cfg: CollectConfig) -> None:
    delay = (cfg.sleep_ms + random.randint(0, max(0, cfg.jitter_ms))) / 1000.0
    time.sleep(delay)

# ---------------------------------------------------------------------
# HTTP session with retry
# ---------------------------------------------------------------------
def _make_session(cfg: CollectConfig) -> requests.Session:
    s = requests.Session()
    s.headers.update(HEADERS)
    s.timeout = cfg.timeout_s  # type: ignore[attr-defined]
    return s

def _get(session: requests.Session, url: str, cfg: CollectConfig) -> Optional[requests.Response]:
    for attempt in range(1, cfg.retries + 2):  # e.g., retries=2 -> attempts: 1,2,3
        try:
            r = session.get(url, timeout=cfg.timeout_s)
            if r.status_code in (429, 500, 502, 503, 504):
                raise requests.HTTPError(f"HTTP {r.status_code}")
            return r
        except Exception as e:
            if attempt <= cfg.retries:
                backoff = min(3.0, 0.4 * attempt)
                logger.warning("GET failed (%s). Retry %d/%d in %.1fs: %s", url, attempt, cfg.retries, backoff, e)
                time.sleep(backoff)
            else:
                logger.error("GET failed permanently for %s: %s", url, e)
                return None
        finally:
            _sleep_polite(cfg)

# ---------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------
def fetch_hn(session: requests.Session, max_items: int, cfg: CollectConfig) -> List[str]:
    """Hacker News top stories (filter for Python in title). Reduced per-run requests by ~40%."""
    url_ids = "https://hacker-news.firebaseio.com/v0/topstories.json"
    ids_resp = _get(session, url_ids, cfg)
    if not ids_resp:
        return []
    try:
        ids = ids_resp.json() or []
        # Previously: max_items * 5; reduce to *3 to be polite without losing much coverage.
        ids = ids[: min(max_items * 3, 250)]
    except Exception as e:
        logger.warning("HN ids parse failed: %s", e)
        return []

    out: List[str] = []
    for i, sid in enumerate(ids):
        item_url = f"https://hacker-news.firebaseio.com/v0/item/{sid}.json"
        r = _get(session, item_url, cfg)
        if not r:
            continue
        try:
            data = r.json() or {}
            title = data.get("title") or ""
            if title and _python_relevant(title):
                out.append(_norm_title(title))
        except Exception:
            pass
        if len(out) >= max_items:
            break
    return out

def _parse_rss_titles(session: requests.Session, url: str, max_items: int, cfg: CollectConfig) -> List[str]:
    r = _get(session, url, cfg)
    if not r or r.status_code >= 400:
        logger.warning("RSS fetch failed: %s", url)
        return []
    try:
        root = ET.fromstring(r.content)
    except Exception as e:
        logger.warning("RSS parse failed for %s: %s", url, e)
        return []

    titles: List[str] = []

    # Atom
    for e in root.findall(".//{http://www.w3.org/2005/Atom}entry"):
        t = e.findtext("{http://www.w3.org/2005/Atom}title") or ""
        if t:
            titles.append(_norm_title(t))

    # RSS 2.0
    for i in root.findall(".//item"):
        t = i.findtext("title") or ""
        if t:
            titles.append(_norm_title(t))

    return titles[:max_items]

def fetch_reddit(session: requests.Session, max_items: int, cfg: CollectConfig) -> List[str]:
    feeds = [
        "https://www.reddit.com/r/Python/.rss",
               "https://www.reddit.com/r/learnpython/.rss",
        "https://www.reddit.com/r/django/.rss",
        "https://www.reddit.com/r/machinelearning/.rss",
        "https://www.reddit.com/r/dataengineering/.rss",
    ]
    out: List[str] = []
    for url in feeds:
        out.extend(_parse_rss_titles(session, url, max_items, cfg))
    return [t for t in out if _python_relevant(t)][:max_items]

def fetch_stackoverflow(session: requests.Session, max_items: int, cfg: CollectConfig) -> List[str]:
    url = "https://stackoverflow.com/feeds/tag?tagnames=python&sort=newest"
    titles = _parse_rss_titles(session, url, max_items * 2, cfg)
    clean = [re.sub(r"^\[python\]\s*", "", t, flags=re.I) for t in titles]
    return clean[:max_items]

def fetch_github_trending(session: requests.Session, max_items: int, cfg: CollectConfig) -> List[str]:
    url = "https://github.com/trending/python?since=daily"
    r = _get(session, url, cfg)
    if not r:
        return []
    try:
        soup = BeautifulSoup(r.text, "html.parser")
        cards = soup.select("article.Box-row")
        out: List[str] = []
        for c in cards[: max_items]:
            repo = (c.select_one("h2 a") or {}).get_text(strip=True).replace("\n", " ")
            desc = (c.select_one("p") or {}).get_text(" ", strip=True)
            if repo:
                title = f"GitHub Trending: {repo}" + (f" — {desc}" if desc else "")
                out.append(_norm_title(title))
        return out
    except Exception as e:
        logger.warning("GitHub Trending parse failed: %s", e)
        return []

def fetch_pypi_recent(session: requests.Session, max_items: int, cfg: CollectConfig) -> List[str]:
    url = "https://pypi.org/rss/updates.xml"
    titles = _parse_rss_titles(session, url, max_items * 2, cfg)
    keep: List[str] = []
    for t in titles:
        m = re.match(r"^(.+?)\s+\d+\.\d+.*$", t)
        name = m.group(1) if m else t
        keep.append(f"New on PyPI: {name}")
    return keep[:max_items]

def fetch_planetpython(session: requests.Session, max_items: int, cfg: CollectConfig) -> List[str]:
    url = "https://planetpython.org/rss20.xml"
    titles = _parse_rss_titles(session, url, max_items * 2, cfg)
    return [t for t in titles if _python_relevant(t)][:max_items]

def fetch_realpython(session: requests.Session, max_items: int, cfg: CollectConfig) -> List[str]:
    url = "https://realpython.com/atom.xml"
    titles = _parse_rss_titles(session, url, max_items * 2, cfg)
    return [t for t in titles if _python_relevant(t)][:max_items]

def fetch_devto_python(session: requests.Session, max_items: int, cfg: CollectConfig) -> List[str]:
    url = "https://dev.to/feed/tag/python"
    titles = _parse_rss_titles(session, url, max_items * 2, cfg)
    return [t for t in titles if _python_relevant(t)][:max_items]


# Registry so --sources can pick and choose
SOURCE_FUNCS = {
    "hn": fetch_hn,
    "reddit": fetch_reddit,
    "stackoverflow": fetch_stackoverflow,
    "github_trending": fetch_github_trending,
    "pypi_recent": fetch_pypi_recent,
    "planetpython": fetch_planetpython,
    "realpython": fetch_realpython,
    "devto": fetch_devto_python,
}

DEFAULT_SOURCES = ["hn", "reddit", "stackoverflow", "github_trending", "pypi_recent", "planetpython"]

# ---------------------------------------------------------------------
# Collect & Merge
# ---------------------------------------------------------------------
def collect(cfg: CollectConfig) -> Tuple[Dict[str, List[str]], Set[str], List[str]]:
    logger.info("Collecting topics… per_niche=%d, max_per_source=%d", cfg.per_niche, cfg.max_per_source)
    candidates: List[str] = []

    session = _make_session(cfg)
    sources = cfg.sources or DEFAULT_SOURCES

    for name in sources:
        fn = SOURCE_FUNCS.get(name)
        if not fn:
            logger.warning("Unknown source '%s' — skipping.", name)
            continue
        try:
            chunk = fn(session, cfg.max_per_source, cfg)
            candidates.extend(chunk)
            logger.info("Source %-16s → %3d items", name, len(chunk))
        except Exception as e:
            logger.error("Source '%s' failed: %s", name, e)

    logger.info("Fetched %d raw titles before filtering.", len(candidates))

    # Filter Python relevance and normalize
    filtered = [_norm_title(t) for t in candidates if _python_relevant(t)]
    logger.info("After python filter: %d", len(filtered))

    # De-dup within this run
    seen_run: Set[str] = set()
    unique_now: List[str] = []
    for t in filtered:
        fp = _fingerprint(t)
        if fp not in seen_run:
            seen_run.add(fp)
            unique_now.append(t)

    logger.info("After intra-run dedupe: %d", len(unique_now))

    # Load historical seen to avoid repeating across runs
    seen_data = _load_json(SEEN_FILE) or {}
    seen_hist: Set[str] = set(seen_data.get("fingerprints", []))

    unique_across_runs = [t for t in unique_now if _fingerprint(t) not in seen_hist]
    logger.info(
        "After cross-run dedupe: %d (skipped %d previously seen)",
        len(unique_across_runs),
        len(unique_now) - len(unique_across_runs),
    )

    # Route to niches and cap per niche
    per_niche: Dict[str, List[str]] = {n: [] for n in NICHES}
    per_niche_seen: Dict[str, Set[str]] = {n: set() for n in NICHES}
    for t in unique_across_runs:
        n = _route_niche(t)
        _extend_unique(per_niche[n], [t], cfg.per_niche, per_niche_seen[n])

    # Remove empties
    per_niche = {k: v for k, v in per_niche.items() if v}
    logger.info("Routed into %d niches.", len(per_niche))
    return per_niche, seen_hist, unique_now


def merge_into_topic_pools(collected: Dict[str, List[str]], out_path: Path, per_niche_cap_total: int) -> Dict[str, List[str]]:
    """Merge with existing pools, keep deduped, respect per_niche_cap_total."""
    existing = _load_json(out_path) or {}
    if not isinstance(existing, dict):
        existing = {}
    for n in NICHES:
        existing.setdefault(n, [])

    # Fingerprints per niche
    fp_per_niche: Dict[str, Set[str]] = {
        n: {_fingerprint(t) for t in existing.get(n, [])} for n in NICHES
    }

    for niche, new_titles in collected.items():
        out_list = existing.get(niche, [])
        for t in new_titles:
            fp = _fingerprint(t)
            if fp not in fp_per_niche[niche]:
                out_list.append(t)
                fp_per_niche[niche].add(fp)
            if len(out_list) >= per_niche_cap_total:
                break
        existing[niche] = out_list

    return existing


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def parse_args() -> CollectConfig:
    p = argparse.ArgumentParser(description="Collect trending Python topics into topic_pools.json")
    p.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Path to topic_pools.json")
    p.add_argument("--per-niche", type=int, default=150, help="Max NEW items per niche per run (post-dedupe)")
    p.add_argument("--max-per-source", type=int, default=60, help="Max items fetched per source")
    p.add_argument("--total-cap", type=int, default=300, help="Total cap per niche in the final pool file")
    p.add_argument("--sleep-ms", type=int, default=120, help="Base delay between network calls (politeness)")
    p.add_argument("--jitter-ms", type=int, default=80, help="Random jitter added to each delay")
    p.add_argument("--retries", type=int, default=2, help="Retries per request")
    p.add_argument("--timeout", type=int, default=15, help="Request timeout (seconds)")
    p.add_argument(
        "--sources",
        type=str,
        default=",".join(DEFAULT_SOURCES),
        help=f"Comma-separated sources. Available: {', '.join(sorted(SOURCE_FUNCS.keys()))}",
    )
    p.add_argument("--dry-run", action="store_true", help="Print summary only; do not write files")
    args = p.parse_args()
    return CollectConfig(
        output=Path(args.output),
        per_niche=args.per_niche,
        max_per_source=args.max_per_source,
        total_cap=args.total_cap,
        dry_run=args.dry_run,
        sleep_ms=args.sleep_ms,
        jitter_ms=args.jitter_ms,
        retries=args.retries,
        timeout_s=args.timeout,
        sources=[s.strip() for s in args.sources.split(",") if s.strip()],
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    cfg = parse_args()
    collected, seen_hist, unique_now = collect(cfg)

    if cfg.dry_run:
        for niche, items in collected.items():
            logger.info("Niche: %s (+%d)", niche, len(items))
        logger.info("Dry-run complete. No files written.")
        return

    # Merge with existing pools & save
    merged = merge_into_topic_pools(collected, cfg.output, per_niche_cap_total=cfg.total_cap)
    _save_json_atomic_map(merged, cfg.output)
    logger.info("Updated %s", cfg.output)

    # Update seen fingerprints (intra-run + historical)
    new_seen = set(seen_hist) | {_fingerprint(t) for t in unique_now}
    _save_json_atomic_map({"fingerprints": sorted(new_seen)}, SEEN_FILE)
    logger.info("Updated %s (seen=%d)", SEEN_FILE, len(new_seen))


if __name__ == "__main__":
    main()
