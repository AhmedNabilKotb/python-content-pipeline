# title_synthesizer.py

import re
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional

try:
    # package import
    from .utils.utils import slugify  # type: ignore
except Exception:
    # fallback
    import re as _re
    def slugify(t: str) -> str:
        t = (t or "").lower()
        t = _re.sub(r"[^a-z0-9]+", "-", t).strip("-")
        t = _re.sub(r"-{2,}", "-", t)
        return t or "article"


# ------------------------------------------------------------------
# Config (tunable without breaking existing API)
# ------------------------------------------------------------------

@dataclass
class TitleConfig:
    min_len: int = 40
    max_len: int = 60
    brand_suffix: Optional[str] = None          # e.g., "PythonProHub"
    brand_every_n: int = 0                      # 0 = never, 3 = every 3rd title, etc.
    forbid_emojis: bool = True
    forbid_clickbait: bool = True
    ensure_keyphrase_presence: bool = True
    # Light SERP width guidance (approx px). We still hard-clamp with min/max_len.
    serp_px_target: int = 570                   # ~Google desktop safe zone
    serp_px_soft_cap: int = 600

# Default (keeps previous behavior)
_CFG = TitleConfig()

# ------------------------------------------------------------------
# Heuristics and regexes
# ------------------------------------------------------------------

_WORD_RE = re.compile(r"\w+(?:'\w+)?", re.UNICODE)
_SEP_RE = re.compile(r"\s*[:—-]\s*")
_EMOJI_RE = re.compile(r"[\U00010000-\U0010ffff]", flags=re.UNICODE)

# Expanded, tech-leaning power words (kept tasteful)
POWER_WORDS = {
    "guide", "playbook", "cheatsheet", "blueprint", "patterns", "anti-patterns",
    "hands-on", "end-to-end", "production", "best practices", "deep dive",
    "from scratch", "in practice", "under the hood", "architecture",
    "profiling", "benchmark", "performance", "observability",
    "testing", "reliability", "security"
}

# Soft-penalize clickbait
CLICKBAIT = {
    "ultimate", "insane", "shocking", "you won't believe", "mind-blowing",
    "secret sauce", "killer", "crazy", "unbelievable"
}

ACRONYMS = {
    "api", "jwt", "sql", "sqlmodel", "http", "grpc", "gpu", "ml", "ai",
    "etl", "dbt", "ssh", "ci", "cd", "rpc", "db", "ui", "ux", "sdk", "ide",
}

# Proper-case for common tech terms (applied after Title Case)
PROPER_CASE = {
    "fastapi": "FastAPI",
    "pydantic": "Pydantic",
    "sqlalchemy": "SQLAlchemy",
    "postgres": "Postgres",
    "postgresql": "PostgreSQL",
    "polars": "Polars",
    "duckdb": "DuckDB",
    "pandas": "Pandas",
    "numpy": "NumPy",
    "numba": "Numba",
    "pytorch": "PyTorch",
    "tensorflow": "TensorFlow",
    "scikit-learn": "scikit-learn",
    "scikit": "scikit-learn",
    "lightgbm": "LightGBM",
    "xgboost": "XGBoost",
    "ray": "Ray",
    "dask": "Dask",
    "openai": "OpenAI",
    "opentelemetry": "OpenTelemetry",
    "grpc": "gRPC",
    "http": "HTTP",
    "https": "HTTPS",
    "kafka": "Kafka",
    "redis": "Redis",
    "celery": "Celery",
}

# Character width estimate for SERP (rough)
# Wider chars get bigger weights; used only for soft guidance.
_SERP_CHAR_W = {
    **{c: 8 for c in "mwMW"},
    **{c: 7 for c in "ABCDEFGHNOQRSTUVXYZ1234567890-—: "},
    **{c: 6 for c in "abcdeghnopqrstuvxyz"},
    **{c: 5 for c in "ijklr;,"},
    **{c: 4 for c in ".!|"},
}
def _serp_width_px(s: str) -> int:
    return sum(_SERP_CHAR_W.get(ch, 6) for ch in s)


@dataclass
class TitleCandidate:
    text: str
    score: float
    components: Dict[str, float]


# -------------------- scoring helpers --------------------

def _len_score(title: str, cfg: TitleConfig) -> float:
    n = len(title)
    if n < cfg.min_len or n > cfg.max_len:
        return max(0.0, 1.0 - (abs(n - 50) / 50.0))
    return 1.0 - (abs(n - 50) / 50.0) * 0.2

def _serp_score(title: str, cfg: TitleConfig) -> float:
    px = _serp_width_px(title)
    if px <= cfg.serp_px_target:
        return 1.0
    if px >= cfg.serp_px_soft_cap:
        return 0.6
    # linear falloff between target and soft cap
    span = max(1, cfg.serp_px_soft_cap - cfg.serp_px_target)
    return max(0.6, 1.0 - (px - cfg.serp_px_target) / span * 0.4)

def _keyphrase_score(title: str, keyphrase: str) -> float:
    t = title.lower()
    k = keyphrase.lower().strip()
    if not k:
        return 0.6
    m = re.search(rf"\b{re.escape(k)}\b", t) or re.search(re.escape(k), t)
    if not m:
        return 0.2
    pos = m.start()
    return max(0.6, 1.0 - (pos / max(1, len(t))))

def _power_word_score(title: str) -> float:
    t = title.lower()
    hits = sum(1 for pw in POWER_WORDS if pw in t)
    return min(1.0, 0.25 + 0.15 * hits)

def _clickbait_penalty(title: str) -> float:
    t = title.lower()
    if any(cb in t for cb in CLICKBAIT):
        return 0.75
    return 1.0

def _distinctiveness_penalty(title: str, used_titles: List[str]) -> float:
    base = slugify(title).replace("-", " ")
    for u in used_titles[:200]:
        s = slugify(u).replace("-", " ")
        bt = set(base.split())
        st = set(s.split())
        if not bt or not st:
            continue
        overlap = len(bt & st) / max(1, len(bt | st))
        if overlap >= 0.7:
            return 0.7
    return 1.0

def _ctr_heuristic(title: str) -> float:
    t = title.lower()
    boost = 0.0
    if re.search(r"\[(.*?)\]", title): boost += 0.2
    if ":" in title or "—" in title or "-" in title: boost += 0.15
    if re.search(r"\b\d{1,3}\b", title): boost += 0.15
    if re.search(r"[A-Z]{6,}", title): boost -= 0.2
    if "things" in t or "stuff" in t: boost -= 0.1
    return max(0.0, min(1.0, 0.5 + boost))


# -------------------- formatting helpers --------------------

def _normalize_punctuation(t: str) -> str:
    t = _SEP_RE.sub(" — ", t.strip())
    t = re.sub(r"\s{2,}", " ", t)
    return t.rstrip(" -–—:,")

def _title_case(s: str) -> str:
    small = {"a","an","the","and","but","or","for","nor","on","at","to","from","by","of","in","with","as","via"}
    parts = re.split(r"(\s+)", s.strip())
    out = []
    for i, token in enumerate(parts):
        if token.isspace():
            out.append(token); continue
        w = token
        lw = w.lower()
        if lw in ACRONYMS:
            out.append(lw.upper())
        elif i != 0 and i != len(parts)-1 and lw in small:
            out.append(lw)
        else:
            out.append(lw[:1].upper() + lw[1:])
    return "".join(out)

def _apply_proper_case(s: str) -> str:
    def repl(m: re.Match) -> str:
        w = m.group(0)
        lw = w.lower()
        return PROPER_CASE.get(lw, w)
    # replace whole-word matches
    keys = sorted(PROPER_CASE.keys(), key=len, reverse=True)
    for k in keys:
        s = re.sub(rf"\b{re.escape(k)}\b", PROPER_CASE[k], s, flags=re.IGNORECASE)
    return s

def _smart_truncate(t: str, limit: int) -> str:
    if len(t) <= limit:
        return t
    cut = t[:limit]
    space = cut.rfind(" ")
    if space > max(0, limit - 12):
        cut = cut[:space]
    return cut.rstrip(" -–—:,")

def _strip_emojis(t: str) -> str:
    return _EMOJI_RE.sub("", t)

def _maybe_brand(t: str, cfg: TitleConfig, index_hint: int) -> str:
    if not cfg.brand_suffix or cfg.brand_every_n <= 0:
        return t
    if index_hint % cfg.brand_every_n != 0:
        return t
    brand = f"{t} | {cfg.brand_suffix}"
    return brand if len(brand) <= cfg.max_len else t


# -------------------- candidate synthesis --------------------

def _niche_patterns(base: str, k: str, niche: str) -> List[str]:
    """
    Generate niche-aware patterns; prefer technical, credible phrasing.
    """
    n = (niche or "").lower().strip()
    kp = k or base

    common = [
        f"{kp}: production patterns that scale",
        f"{kp} in Python — hands-on guide",
        f"{kp} under the hood — performance & security",
        f"{kp} in practice: end-to-end walkthrough",
        f"Modern {kp}: best practices & anti-patterns",
        f"{kp}: design decisions, trade-offs, and testing",
    ]

    web = [
        f"{kp} with FastAPI & Pydantic: deep dive",
        f"{kp} in FastAPI — auth, JWT, and rate limiting",
        f"{kp} with asyncio & SQLModel: patterns & pitfalls",
        f"{kp} in production: observability & error budgets",
    ]

    data = [
        f"{kp} with Polars & DuckDB — faster pipelines",
        f"{kp} in Pandas vs Polars: vectorization & memory",
        f"{kp}: profiling, benchmarks, and I/O bottlenecks",
        f"{kp} with Dask/Ray — scaling beyond one machine",
    ]

    ml = [
        f"{kp} with scikit-learn — feature engineering & leakage",
        f"{kp} in PyTorch: data pipelines & evaluation",
        f"{kp}: SHAP, calibration, and drift monitoring",
        f"{kp} to production: MLOps, CI/CD, and tests",
    ]

    devex = [
        f"{kp} with uv/poetry: packaging for teams",
        f"{kp}: mypy, Ruff, and type-safe refactors",
        f"{kp} — logging, tracing, and OpenTelemetry",
        f"{kp} with pytest: fixtures, parametrization, coverage",
    ]

    if "web" in n:
        return common + web
    if "data" in n:
        return common + data
    if "ml" in n or "machine" in n or "ai" in n:
        return common + ml
    if "devops" in n or "tooling" in n or "productivity" in n or "engineering" in n:
        return common + devex
    return common + web[:2] + data[:2] + ml[:1]

def synthesize_titles(
    topic: str,
    keyphrase: str,
    niche: str,
    used_titles: Optional[List[str]] = None
) -> List[str]:
    """
    Make a diverse set of candidate titles. Keep them concise, modern, and technical.
    """
    k = (keyphrase or "").strip()
    b = (topic or "").strip()
    used_titles = used_titles or []

    base = b if not k else k
    patterns = _niche_patterns(base, k, niche)

    if not k:
        # ensure non-keyphrase variants still look solid
        base_norm = slugify(b).replace("-", " ").title()
        extra = [
            f"{base_norm}: production patterns that scale",
            f"{base_norm} — hands-on guide",
        ]
        patterns = patterns + extra

    # Normalize & uniquify
    seen = set()
    out = []
    for p in patterns:
        t = _normalize_punctuation(p.strip().rstrip("."))
        if t.lower() in seen:
            continue
        seen.add(t.lower())
        out.append(t)
    return out


# -------------------- selection & normalization --------------------

def _final_normalize(title: str, keyphrase: str, cfg: TitleConfig) -> str:
    t = _normalize_punctuation(title)

    # Ensure keyphrase presence (soft)
    if cfg.ensure_keyphrase_presence and keyphrase and keyphrase.lower() not in t.lower():
        proposal = f"{keyphrase}: {t}"
        if len(proposal) <= cfg.max_len:
            t = proposal

    # Title case + proper casing of known tech terms
    t = _title_case(t)
    t = _apply_proper_case(t)

    # Clamp to window
    if len(t) > cfg.max_len:
        t = _smart_truncate(t, cfg.max_len)

    # Pad if too short
    if len(t) < cfg.min_len:
        tail = " — Patterns & Pitfalls"
        if len(t) + len(tail) <= cfg.max_len:
            t = t + tail

    return t

def _avoid_duplicates(t: str, used_titles: List[str], cfg: TitleConfig) -> str:
    base = slugify(t).replace("-", " ")
    for u in used_titles[:200]:
        s = slugify(u).replace("-", " ")
        bt = set(base.split())
        st = set(s.split())
        if not bt or not st:
            continue
        overlap = len(bt & st) / max(1, len(bt | st))
        if overlap >= 0.75:
            addon = " — A Practical Playbook"
            if len(t) + len(addon) <= cfg.max_len:
                return t + addon
            return _smart_truncate(t, cfg.max_len)
    return t

def _clean_for_policy(t: str, cfg: TitleConfig) -> str:
    if cfg.forbid_emojis:
        t = _strip_emojis(t)
    if cfg.forbid_clickbait and any(cb in t.lower() for cb in CLICKBAIT):
        # Remove the offending phrase softly (first occurrence)
        for cb in CLICKBAIT:
            t = re.sub(cb, "", t, flags=re.IGNORECASE).strip()
        t = _normalize_punctuation(t)
    return t


# -------------------- public API --------------------

def pick_best_title(candidates: List[str], keyphrase: str, used_titles: List[str], cfg: TitleConfig = _CFG) -> TitleCandidate:
    best: Optional[TitleCandidate] = None
    for t in candidates:
        comp = {
            "len": _len_score(t, cfg),
            "serp": _serp_score(t, cfg),
            "key": _keyphrase_score(t, keyphrase),
            "power": _power_word_score(t),
            "ctr": _ctr_heuristic(t),
        }
        base_score = (
            0.27 * comp["len"] +
            0.18 * comp["serp"] +
            0.25 * comp["key"] +
            0.16 * comp["ctr"] +
            0.14 * comp["power"]
        )
        distinct = _distinctiveness_penalty(t, used_titles)
        clickbait = _clickbait_penalty(t)
        score = base_score * distinct * clickbait

        cand = TitleCandidate(text=t, score=score, components={**comp, "distinct": distinct, "clickbait": clickbait})
        if not best or cand.score > best.score:
            best = cand
    return best or TitleCandidate(text=candidates[0], score=0.0, components={})

def generate_title(
    topic: str,
    keyphrase: str,
    niche: str,
    used_titles: Optional[List[str]] = None,
    *,
    cfg: TitleConfig = _CFG,
    index_hint: int = 1,   # for optional brand injection cadence
) -> str:
    """
    Backward-compatible main entry.
    - Adds soft SERP-width awareness
    - Avoids clickbait/emojis if configured
    - Keeps within Yoast-friendly length window (min_len/max_len)
    """
    used_titles = used_titles or []
    candidates = synthesize_titles(topic, keyphrase, niche, used_titles)
    best = pick_best_title(candidates, keyphrase, used_titles, cfg)
    t = _final_normalize(best.text.strip(), keyphrase.strip(), cfg)
    t = _avoid_duplicates(t, used_titles, cfg)
    t = _clean_for_policy(t, cfg)
    t = _maybe_brand(t, cfg, index_hint)
    if len(t) > cfg.max_len:
        t = _smart_truncate(t, cfg.max_len)
    return _normalize_punctuation(t)

def generate_title_plus_alts(
    topic: str,
    keyphrase: str,
    niche: str,
    used_titles: Optional[List[str]] = None,
    *,
    n_alternatives: int = 4,
    cfg: TitleConfig = _CFG,
    index_hint: int = 1,
) -> Tuple[str, List[str], Dict[str, Any]]:
    """
    Produce a chosen title + N alternates (already normalized), plus a debug report.
    Deterministic given the same inputs.
    """
    used_titles = used_titles or []
    cands = synthesize_titles(topic, keyphrase, niche, used_titles)

    scored: List[TitleCandidate] = []
    for c in cands:
        cand = pick_best_title([c], keyphrase, used_titles, cfg)
        scored.append(cand)

    # Sort by score, desc
    scored.sort(key=lambda x: (-x.score, x.text.lower()))
    normalized: List[str] = []
    for idx, s in enumerate(scored):
        t = _final_normalize(s.text.strip(), keyphrase.strip(), cfg)
        t = _avoid_duplicates(t, used_titles + normalized, cfg)
        t = _clean_for_policy(t, cfg)
        t = _normalize_punctuation(t)
        normalized.append(t)

    primary = _maybe_brand(normalized[0], cfg, index_hint) if normalized else ""
    alts = []
    for t in normalized[1: 1 + n_alternatives]:
        alts.append(t if len(t) <= cfg.max_len else _smart_truncate(t, cfg.max_len))

    report = {
        "chosen_score": scored[0].score if scored else 0.0,
        "components": scored[0].components if scored else {},
        "candidates_considered": len(scored),
    }
    return primary, alts, report
