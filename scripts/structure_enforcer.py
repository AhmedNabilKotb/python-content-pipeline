# structure_enforcer.py

import re
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# ----------------------------- Data ---------------------------------

@dataclass
class ArticleStructure:
    title: str
    content: str
    article_type: str = "guide"
    word_count: int = 0


class ArticleStructureEnforcer:
    """
    Enforce a professional, pipeline-friendly structure:
      - Optional single H1 (if `title` provided; otherwise preserve existing H1 if present)
      - Stable section set per article type
      - Extract existing H2/H3 sections when possible (with alias matching)
      - Deterministic (no randomness)
      - Section length tuned by article_type and niche (20 niches supported)
      - Niche-aware keyword extraction
      - Scaffolds use plain ``` fences (no language tag)
      - Normalizes existing code fences to plain ```
    """

    # Templates that align with QualityGate's section aliases
    PROFESSIONAL_TEMPLATES: Dict[str, List[str]] = {
        "tutorial": [
            "Real-World Problem Statement",
            "Technical Deep Dive",
            "Implementation Walkthrough",
            "Performance Benchmarks",
            "Alternative Approaches",
            "Production Considerations",
            "Further Reading",
        ],
        "guide": [
            "Comprehensive Overview",
            "Key Concepts Explained",
            "Best Practices",
            "Common Pitfalls",
            "Advanced Techniques",
            "Use Case Examples",
            "Summary Cheat Sheet",
        ],
        "analysis": [
            "Technology Landscape",
            "Comparative Analysis",
            "Technical Specifications",
            "Implementation Challenges",
            "Performance Metrics",
            "Adoption Recommendations",
            "Future Outlook",
        ],
    }

    TEMPLATE_INTROS: Dict[str, str] = {
        "tutorial": "This hands-on tutorial walks through implementing {topic} in production environments.",
        "guide": "This comprehensive guide covers everything you need to know about {topic} from basics to advanced usage.",
        "analysis": "This in-depth analysis examines {topic} from technical specifications to real-world implementation.",
    }

    # Guidance for every section name in templates (tutorial/guide/analysis)
    SECTION_GUIDANCE: Dict[str, str] = {
        # tutorial
        "Real-World Problem Statement": "Describe the concrete business/technical problem and why {topic} is the right fit.",
        "Technical Deep Dive": "Explain core concepts, architecture, and key mechanisms behind {topic}.",
        "Implementation Walkthrough": "Provide step-by-step code with brief explanations for each block.",
        "Performance Benchmarks": "Show measured results (latency, throughput, memory). Explain methodology and environment.",
        "Alternative Approaches": "Compare at least two alternatives and detail trade-offs.",
        "Production Considerations": "Cover deployment, monitoring, logging, scaling, backups, and rollback strategies.",
        "Further Reading": "List official docs and 1–2 highly reputable references.",
        # guide
        "Comprehensive Overview": "Summarize what {topic} is, when to use it, and common misconceptions.",
        "Key Concepts Explained": "Define the fundamental ideas and how they interact.",
        "Best Practices": "List practical do's and don'ts with short rationales.",
        "Common Pitfalls": "Highlight frequent mistakes and how to avoid them.",
        "Advanced Techniques": "Offer 2–3 advanced patterns with brief examples.",
        "Use Case Examples": "Show realistic scenarios with inputs/outputs.",
        "Summary Cheat Sheet": "Provide a quick-reference list of commands, APIs, or patterns.",
        # analysis
        "Technology Landscape": "Position {topic} among alternatives and adjacent tools.",
        "Comparative Analysis": "Compare features/cost/performance with 2–3 options.",
        "Technical Specifications": "List versions, limits, supported platforms, and constraints.",
        "Implementation Challenges": "Call out integration risks and mitigations.",
        "Performance Metrics": "Reference reproducible metrics and how they were obtained.",
        "Adoption Recommendations": "Suggest when/how to adopt, with prerequisites.",
        "Future Outlook": "Discuss roadmap, ecosystem maturity, and likely trends.",
    }

    # Base keywords for extraction from raw content
    BASE_SECTION_KEYWORDS: Dict[str, List[str]] = {
        "Real-World Problem Statement": ["problem", "challenge", "issue", "pain point", "use case"],
        "Technical Deep Dive": ["technical", "architecture", "design", "under the hood", "deep dive"],
        "Implementation Walkthrough": ["code", "example", "implementation", "step", "walkthrough"],
        "Performance Benchmarks": ["performance", "benchmark", "latency", "throughput", "memory", "cpu", "ram"],
        "Alternative Approaches": ["alternative", "other way", "different approach", "trade-off", "tradeoff"],
        "Production Considerations": ["production", "deployment", "monitoring", "observability", "scaling", "slo", "sla"],
        "Further Reading": ["reference", "documentation", "resource", "learn more", "further reading"],
        "Comprehensive Overview": ["overview", "introduction", "summary", "what is", "why"],
        "Key Concepts Explained": ["concept", "principle", "core idea", "terminology"],
        "Best Practices": ["best practice", "recommended", "guideline", "do's", "donts", "do not"],
        "Common Pitfalls": ["pitfall", "gotcha", "mistake", "anti-pattern"],
        "Advanced Techniques": ["advanced", "pattern", "technique", "strategy"],
        "Use Case Examples": ["use case", "example", "case study"],
        "Summary Cheat Sheet": ["cheat sheet", "summary", "recap", "quick reference"],
        "Technology Landscape": ["landscape", "ecosystem", "market", "panorama"],
        "Comparative Analysis": ["compare", "versus", "vs", "differ", "comparison"],
        "Technical Specifications": ["specification", "specs", "limits", "constraints", "version"],
        "Implementation Challenges": ["challenge", "risk", "issue", "concern"],
        "Performance Metrics": ["metric", "benchmark", "kpi", "measure"],
        "Adoption Recommendations": ["recommendation", "adopt", "rollout", "migration"],
        "Future Outlook": ["future", "roadmap", "trend", "outlook"],
    }

    # Aliases to map existing headings to canonical section names (case-insensitive)
    SECTION_ALIASES: Dict[str, List[str]] = {
        # tutorial
        "Real-World Problem Statement": ["problem statement", "the problem", "why this matters", "background", "real world problem"],
        "Technical Deep Dive": ["deep dive", "architecture", "design details", "under the hood", "technical details"],
        "Implementation Walkthrough": ["implementation", "walkthrough", "setup", "getting started", "how to", "step by step"],
        "Performance Benchmarks": ["benchmarks", "performance", "results", "performance results", "performance metrics"],
        "Alternative Approaches": ["alternatives", "other approaches", "options", "trade-offs", "tradeoffs"],
        "Production Considerations": ["production", "deployment", "operations", "observability", "monitoring", "scaling"],
        "Further Reading": ["references", "resources", "learn more", "reading"],
        # guide
        "Comprehensive Overview": ["overview", "what is", "introduction", "intro", "summary"],
        "Key Concepts Explained": ["key concepts", "concepts", "fundamentals", "core ideas", "terminology"],
        "Best Practices": ["dos and donts", "dos & donts", "guidelines", "recommendations"],
        "Common Pitfalls": ["pitfalls", "gotchas", "anti-patterns", "mistakes"],
        "Advanced Techniques": ["advanced topics", "advanced patterns", "expert techniques"],
        "Use Case Examples": ["examples", "use cases", "case studies"],
        "Summary Cheat Sheet": ["cheat sheet", "summary sheet", "quick reference", "recap"],
        # analysis
        "Technology Landscape": ["landscape", "ecosystem overview", "market overview"],
        "Comparative Analysis": ["comparison", "versus", "vs", "alternatives"],
        "Technical Specifications": ["specifications", "specs", "limits & constraints"],
        "Implementation Challenges": ["challenges", "risks", "issues"],
        "Performance Metrics": ["metrics", "kpis", "benchmark results"],
        "Adoption Recommendations": ["recommendations", "adoption guidance", "rollout plan"],
        "Future Outlook": ["outlook", "roadmap", "trends"],
    }

    # ---------- 20 Niche Canonicalization ----------
    NICHE_ALIASES: Dict[str, str] = {
        "web development": "web development",
        "data science and analytics": "data science",
        "data science analytics": "data science",
        "machine learning and ai": "machine learning",
        "automation and scripting": "automation",
        "cybersecurity and ethical hacking": "cybersecurity",
        "python for finance": "finance",
        "educational python": "education",
        "web scraping and data extraction": "web scraping",
        "python tips": "python tips",
        "scientific numerical computing": "scientific computing",
        "devops cloud infrastructure": "devops",
        "data engineering pipelines": "data engineering",
        "desktop gui apps": "desktop gui",
        "iot embedded hardware": "iot",
        "testing quality types": "testing",
        "mlops production ai": "mlops",
        "geospatial gis": "geospatial",
        "game development with python": "game development",
        "apis integrations": "apis",
        "data visualization storytelling": "data visualization",
    }

    # ---------- 20 Niche Keyword Boosts ----------
    NICHE_KEYWORD_BOOST: Dict[str, Dict[str, List[str]]] = {
        "web development": {
            "Implementation Walkthrough": ["django", "flask", "fastapi", "react", "vue", "next.js", "routing"],
            "Performance Benchmarks": ["ttfb", "lcp", "cls", "cdn", "cache", "gzip", "brotli"],
            "Common Pitfalls": ["cors", "csrf", "n+1", "blocking", "race condition"],
            "Production Considerations": ["reverse proxy", "nginx", "gunicorn", "uvicorn", "scaling"],
        },
        "data science": {
            "Key Concepts Explained": ["pandas", "numpy", "scipy", "eda", "feature engineering"],
            "Use Case Examples": ["regression", "classification", "clustering", "timeseries"],
            "Performance Metrics": ["accuracy", "rmse", "precision", "recall"],
        },
        "machine learning": {
            "Technical Deep Dive": ["pytorch", "tensorflow", "regularization", "overfitting", "gradient", "optimizer"],
            "Performance Benchmarks": ["latency", "throughput", "f1", "auc", "roc", "inference"],
            "Production Considerations": ["drift", "feature store", "monitoring", "canary", "shadow"],
        },
        "automation": {
            "Implementation Walkthrough": ["airflow", "crontab", "schedule", "selenium", "pyautogui", "rpa"],
            "Production Considerations": ["idempotent", "retry", "backoff", "dead letter", "queue", "monitoring"],
            "Best Practices": ["orchestration", "state", "recovery", "alerting"],
        },
        "cybersecurity": {
            "Technical Deep Dive": ["owasp", "xss", "csrf", "sqli", "cve", "mitre", "threat model"],
            "Best Practices": ["least privilege", "mfa", "tls", "rotate keys", "zero trust", "secrets"],
            "Common Pitfalls": ["injection", "hardcoded", "plaintext", "weak cipher", "default creds"],
        },
        "finance": {
            "Technical Specifications": ["decimal", "precision", "rounding", "double entry", "ledger"],
            "Performance Metrics": ["p95", "latency", "throughput", "risk", "var", "sharpe"],
            "Production Considerations": ["audit", "compliance", "sox", "gdpr", "reporting"],
        },
        "education": {
            "Key Concepts Explained": ["beginner", "curriculum", "exercise", "notebook", "lesson plan"],
            "Use Case Examples": ["assignments", "quizzes", "projects"],
            "Summary Cheat Sheet": ["syntax", "operators", "builtins", "stdlib"],
        },
        "web scraping": {
            "Implementation Walkthrough": ["requests", "beautifulsoup", "lxml", "scrapy", "xpath", "css selector"],
            "Production Considerations": ["robots.txt", "rate limit", "captcha", "proxy", "user agent"],
            "Common Pitfalls": ["ban", "blocked", "dynamic content", "javascript rendering"],
        },
        "python tips": {
            "Best Practices": ["pep8", "typing", "mypy", "black", "flake8"],
            "Advanced Techniques": ["context manager", "decorator", "generator", "iterable", "walrus"],
            "Summary Cheat Sheet": ["pathlib", "itertools", "functools", "dataclasses"],
        },
        "scientific computing": {
            "Technical Deep Dive": ["numpy", "scipy", "cython", "numba", "blas", "lapack", "vectorization"],
            "Performance Benchmarks": ["simd", "matrix", "solver", "precision", "profiling"],
            "Implementation Walkthrough": ["array", "ndarray", "broadcast", "ufunc"],
        },
        "devops": {
            "Production Considerations": ["kubernetes", "helm", "docker", "observability", "rollback", "slo", "sli", "sre"],
            "Implementation Walkthrough": ["terraform", "ansible", "ci", "cd", "pipeline", "github actions"],
            "Comparative Analysis": ["rest", "grpc", "mesh", "service discovery"],
        },
        "data engineering": {
            "Implementation Walkthrough": ["etl", "elt", "spark", "dask", "dbt", "airflow", "kafka", "parquet"],
            "Performance Benchmarks": ["shuffle", "partition", "spill", "io", "throughput"],
            "Production Considerations": ["schema evolution", "cdc", "late data", "backfill", "orchestration"],
        },
        "desktop gui": {
            "Implementation Walkthrough": ["pyqt", "pyside", "tkinter", "kivy", "pysimplegui"],
            "Production Considerations": ["pyinstaller", "packaging", "cross-platform", "code signing"],
            "Common Pitfalls": ["event loop", "threading", "blocking ui"],
        },
        "iot": {
            "Technical Specifications": ["micropython", "circuitpython", "gpio", "uart", "spi", "i2c"],
            "Production Considerations": ["mqtt", "power", "latency", "firmware", "ota"],
            "Performance Metrics": ["throughput", "packet loss", "rssi"],
        },
        "testing": {
            "Best Practices": ["pytest", "unittest", "coverage", "tdd", "fixtures", "mocks"],
            "Implementation Walkthrough": ["ci", "cd", "matrix", "runner", "pipeline"],
            "Common Pitfalls": ["flaky", "shared state", "timing", "non-deterministic"],
        },
        "mlops": {
            "Production Considerations": ["serving", "bentoml", "fastapi", "canary", "shadow", "monitoring", "drift"],
            "Performance Metrics": ["p95", "p99", "latency", "throughput", "qps", "cost"],
            "Implementation Challenges": ["feature store", "online vs batch", "rollout", "versioning"],
        },
        "geospatial": {
            "Technical Specifications": ["geopandas", "shapely", "postgis", "epsg", "proj", "rasterio"],
            "Use Case Examples": ["routing", "isochrone", "heatmap", "choropleth"],
            "Performance Metrics": ["tile", "index", "rtree", "geohash"],
        },
        "game development": {
            "Implementation Walkthrough": ["pygame", "panda3d", "godot", "sprite", "physics"],
            "Performance Metrics": ["fps", "delta time", "frame time", "draw call"],
            "Advanced Techniques": ["state machine", "entity component system", "ai"],
        },
        "apis": {
            "Technical Specifications": ["openapi", "swagger", "rest", "graphql", "grpc", "rate limit", "idempotency"],
            "Comparative Analysis": ["rest", "graphql", "grpc", "websocket"],
            "Production Considerations": ["versioning", "deprecation", "backward compatibility", "oauth2", "jwt"],
        },
        "data visualization": {
            "Use Case Examples": ["matplotlib", "seaborn", "plotly", "altair", "storytelling"],
            "Best Practices": ["annotation", "layout", "legend", "color scale", "accessibility"],
            "Comparative Analysis": ["bar", "line", "heatmap", "scatter", "dashboard"],
        },
    }

    # ---------- Per-Article-Type Bounds ----------
    DEFAULT_SECTION_BOUNDS = {"min": 150, "max": 800}
    ARTICLE_TYPE_BOUNDS: Dict[str, Dict[str, Tuple[int, int]]] = {
        "tutorial": {
            "Implementation Walkthrough": (200, 900),
            "Performance Benchmarks": (180, 800),
            "Further Reading": (60, 300),
        },
        "guide": {
            "Summary Cheat Sheet": (80, 300),
            "Use Case Examples": (160, 800),
        },
        "analysis": {
            "Comparative Analysis": (200, 800),
            "Performance Metrics": (150, 700),
            "Future Outlook": (120, 600),
        },
    }

    # ---------- 20 Niche Bounds Deltas ----------
    NICHE_BOUNDS_DELTA: Dict[str, Dict[str, Tuple[int, int]]] = {
        "web development": {
            "Implementation Walkthrough": (+40, +60),
            "Performance Benchmarks": (+30, +60),
            "Common Pitfalls": (+20, +40),
        },
        "data science": {
            "Key Concepts Explained": (+30, +60),
            "Use Case Examples": (+20, +40),
            "Performance Metrics": (+10, +30),
        },
        "machine learning": {
            "Technical Deep Dive": (+50, +100),
            "Performance Benchmarks": (+50, +100),
        },
        "automation": {
            "Implementation Walkthrough": (+40, +80),
            "Production Considerations": (+30, +60),
        },
        "cybersecurity": {
            "Best Practices": (+30, +60),
            "Common Pitfalls": (+30, +60),
            "Technical Specifications": (+10, +20),
        },
        "finance": {
            "Technical Specifications": (+40, +80),
            "Production Considerations": (+20, +40),
            "Performance Metrics": (+20, +40),
        },
        "education": {
            "Key Concepts Explained": (+30, +60),
            "Use Case Examples": (+20, +40),
            "Summary Cheat Sheet": (+10, +20),
        },
        "web scraping": {
            "Implementation Walkthrough": (+40, +80),
            "Production Considerations": (+20, +40),
        },
        "python tips": {
            "Best Practices": (+30, +60),
            "Summary Cheat Sheet": (+10, +30),
        },
        "scientific computing": {
            "Technical Deep Dive": (+40, +80),
            "Performance Benchmarks": (+50, +100),
        },
        "devops": {
            "Production Considerations": (+60, +120),
            "Implementation Walkthrough": (+40, +80),
            "Comparative Analysis": (+30, +60),
        },
        "data engineering": {
            "Implementation Walkthrough": (+50, +100),
            "Performance Benchmarks": (+40, +80),
        },
        "desktop gui": {
            "Implementation Walkthrough": (+40, +80),
            "Use Case Examples": (+20, +40),
        },
        "iot": {
            "Technical Specifications": (+40, +80),
            "Production Considerations": (+30, +60),
            "Performance Metrics": (+20, +40),
        },
        "testing": {
            "Best Practices": (+40, +80),
            "Common Pitfalls": (+30, +60),
            "Implementation Walkthrough": (+20, +40),
        },
        "mlops": {
            "Production Considerations": (+50, +100),
            "Performance Metrics": (+40, +80),
            "Implementation Challenges": (+30, +60),
        },
        "geospatial": {
            "Technical Specifications": (+30, +60),
            "Use Case Examples": (+30, +60),
            "Performance Metrics": (+20, +40),
        },
        "game development": {
            "Implementation Walkthrough": (+40, +80),
            "Performance Metrics": (+30, +60),
            "Advanced Techniques": (+20, +40),
        },
        "apis": {
            "Technical Specifications": (+40, +80),
            "Comparative Analysis": (+30, +60),
            "Production Considerations": (+30, +60),
        },
        "data visualization": {
            "Use Case Examples": (+40, +80),
            "Best Practices": (+30, +60),
            "Comparative Analysis": (+20, +40),
        },
    }

    def __init__(self):
        self.quality_standards = {
            "min_code_blocks": 3,
            "min_references": 2,
            "min_examples": 2,
            "min_section_words": self.DEFAULT_SECTION_BOUNDS["min"],
            "max_section_words": self.DEFAULT_SECTION_BOUNDS["max"],
        }

    # ----------------------------- Public API -----------------------------

    def determine_article_type(self, topic: str, keyphrase: str) -> str:
        tutorial_keywords = {"tutorial", "how to", "step by step", "guide", "implement", "walkthrough"}
        analysis_keywords = {"analysis", "comparison", "benchmark", "performance", "review", "versus", "vs"}
        t = (topic or "").lower()
        k = (keyphrase or "").lower()
        if any(kw in t or kw in k for kw in tutorial_keywords):
            return "tutorial"
        if any(kw in t or kw in k for kw in analysis_keywords):
            return "analysis"
        return "guide"

    def enforce_structure(
        self,
        content: str,
        topic: str,
        keyphrase: str,
        *,
        title: Optional[str] = None,
        niche: Optional[str] = None,
    ) -> str:
        """
        Returns a fully structured Markdown article string.
        """
        content = (content or "").strip()
        # Normalize existing code fences to plain ```
        content = self._normalize_code_fences(content)

        # Decide article type & template
        article_type = self.determine_article_type(topic, keyphrase)
        template = self.PROFESSIONAL_TEMPLATES[article_type]

        # Title handling: prefer provided title, else extract existing H1
        extracted_title = self._extract_h1(content)
        final_title = (title or extracted_title or "").strip()

        # If we will inject our own H1, remove all H1 from body
        if final_title:
            content = self._strip_all_h1(content)

        # Parse existing sections and map by canonical aliases
        existing_sections_raw = self._parse_sections(content)  # List of (heading_text, body)
        existing_sections = self._index_sections_by_norm(existing_sections_raw)

        parts: List[str] = []
        if final_title:
            parts.append(f"# {final_title}\n\n")

        intro = self.TEMPLATE_INTROS[article_type].format(topic=topic)
        parts.append(f"{intro}\n\n")

        canon_niche = self._canon_niche(niche) if niche else None
        for section in template:
            section_body = self._get_existing_section_by_alias(
                canonical=section,
                existing_sections=existing_sections,
            )

            if section_body is None:
                # try to mine a paragraph that looks relevant
                keywords = self._get_keywords(section, canon_niche)
                mined = self._find_paragraph_by_keywords(content, keywords)
                section_body = mined if mined else self._scaffold_section(section, topic)

            min_w, max_w = self._section_length_bounds(section, article_type, canon_niche)
            section_body = self._enforce_section_length(section_body, min_w=min_w, max_w=max_w)

            # Post-process specific sections
            if section in ("Further Reading",):
                section_body = self._ensure_min_references(section_body)

            parts.append(f"## {section}\n\n{section_body}\n\n")

        out = "".join(parts).strip() + "\n"
        # Make sure any truncated code block is properly closed
        out = self._ensure_closed_fences(out)
        return out

    def validate_structure(self, content: str, article_type: str) -> bool:
        template = self.PROFESSIONAL_TEMPLATES[article_type]
        present = {self._norm(h) for h in re.findall(r'^#{2,3}\s+(.+)$', content, re.MULTILINE)}
        missing = [sec for sec in template if self._norm(sec) not in present]
        if missing:
            logger.warning(f"Missing sections: {missing}")
            return False
        return True

    # ----------------------------- Internals ------------------------------

    def _parse_sections(self, content: str) -> List[Tuple[str, str]]:
        """
        Return list of (heading_text, body) for H2/H3 sections in order of appearance.
        """
        sections: List[Tuple[str, str]] = []
        pattern = re.compile(r'^(#{2,3})\s+(.+)$', re.MULTILINE)
        matches = list(pattern.finditer(content))
        for i, m in enumerate(matches):
            heading_text = m.group(2).strip()
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
            body = content[start:end].strip()
            sections.append((heading_text, body))
        return sections

    def _index_sections_by_norm(self, sections: List[Tuple[str, str]]) -> Dict[str, str]:
        """
        Map normalized heading -> body; if duplicate headings appear, keep the first.
        """
        out: Dict[str, str] = {}
        for heading, body in sections:
            key = self._norm(heading)
            if key not in out:
                out[key] = body
        return out

    def _get_existing_section_by_alias(self, *, canonical: str, existing_sections: Dict[str, str]) -> Optional[str]:
        """
        Try exact normalized match; if not found, try alias patterns for the canonical section.
        """
        canon_norm = self._norm(canonical)
        if canon_norm in existing_sections:
            return existing_sections[canon_norm]

        aliases = self.SECTION_ALIASES.get(canonical, [])
        for alias in aliases:
            a_norm = self._norm(alias)
            # exact alias
            if a_norm in existing_sections:
                return existing_sections[a_norm]
            # fuzzy: alias token appears as a word in the heading norm
            for k in existing_sections.keys():
                if re.search(rf'\b{re.escape(a_norm)}\b', k):
                    return existing_sections[k]
        return None

    def _extract_or_generate_section(
        self,
        *,
        section: str,
        topic: str,
        existing_sections: Dict[str, str],
        raw_content: str,
        niche: Optional[str],
    ) -> str:
        # (kept for backward-compat; now routed through _get_existing_section_by_alias)
        normalized = self._norm(section)
        if normalized in existing_sections:
            return existing_sections[normalized]
        keywords = self._get_keywords(section, niche)
        candidate = self._find_paragraph_by_keywords(raw_content, keywords)
        if candidate:
            return candidate
        guidance = self.SECTION_GUIDANCE.get(section, f"Discuss {section.lower()} aspects of {{topic}}.")
        scaffold = self._make_scaffold(guidance.format(topic=topic), section=section)
        return scaffold

    def _get_keywords(self, section: str, niche: Optional[str]) -> List[str]:
        base = list(self.BASE_SECTION_KEYWORDS.get(section, []))
        if not niche:
            return base
        extras = self.NICHE_KEYWORD_BOOST.get(niche, {}).get(section, [])
        seen = set()
        merged: List[str] = []
        for tok in base + extras:
            t = tok.lower().strip()
            if t and t not in seen:
                seen.add(t)
                merged.append(t)
        return merged

    def _find_paragraph_by_keywords(self, content: str, keywords: List[str]) -> Optional[str]:
        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
        for p in paragraphs:
            pl = p.lower()
            if any(k in pl for k in keywords):
                return p
        return None

    def _scaffold_section(self, section: str, topic: str) -> str:
        guidance = self.SECTION_GUIDANCE.get(section, f"Discuss {section.lower()} aspects of {topic}.")
        return self._make_scaffold(guidance, section)

    def _make_scaffold(self, guidance: str, section: str) -> str:
        bullets = []
        if section in ("Implementation Walkthrough", "Advanced Techniques"):
            bullets += [
                "- Outline the steps briefly, then show code blocks:",
                "```",
                "# Step 1",
                "# Step 2",
                "# Step 3",
                "```",
            ]
        if section in ("Performance Benchmarks", "Performance Metrics"):
            bullets += [
                "- Describe setup (hardware, dataset, versions).",
                "- Present results (table) and discuss trade-offs.",
            ]
        if section in ("Alternative Approaches", "Comparative Analysis"):
            bullets += [
                "- Compare 2–3 options and when to prefer each.",
                "- Note caveats, pitfalls, and migration cost.",
            ]
        if section in ("Further Reading", "Summary Cheat Sheet"):
            bullets += [
                "- Add 2+ reputable references (official docs preferred).",
            ]
        if section in ("Best Practices", "Production Considerations"):
            bullets += [
                "- Provide 5–8 concise bullets the reader can apply immediately.",
            ]
        base = f"{guidance}\n\n" + "\n".join(bullets) if bullets else guidance
        return base

    # ---- Bounds -----------------------------------------------------------

    def _section_length_bounds(
        self, section: str, article_type: str, niche: Optional[str]
    ) -> Tuple[int, int]:
        min_w = self.quality_standards["min_section_words"]
        max_w = self.quality_standards["max_section_words"]

        type_over = self.ARTICLE_TYPE_BOUNDS.get(article_type, {})
        if section in type_over:
            min_w, max_w = type_over[section]

        if niche:
            deltas = self.NICHE_BOUNDS_DELTA.get(niche, {})
            if section in deltas:
                dm, dM = deltas[section]
                min_w = max(60, min_w + dm)
                max_w = max(min_w + 100, max_w + dM)

        return min_w, max_w

    def _enforce_section_length(self, text: str, *, min_w: int, max_w: int) -> str:
        words = self._split_words(text)
        if len(words) > max_w:
            trimmed = " ".join(words[:max_w])
            trimmed = self._close_unfinished_code_block(trimmed, original=text)
            return trimmed
        if len(words) < min_w:
            needed = min_w - len(words)
            pad_items = max(3, min(8, needed // 20))
            checklist = "\n".join([f"- Add detail point {i+1}." for i in range(pad_items)])
            return f"{text.rstrip()}\n\n{checklist}"
        return text

    def _close_unfinished_code_block(self, trimmed: str, *, original: str) -> str:
        open_count = len(re.findall(r'```', trimmed))
        if open_count % 2 != 0:
            if '```' in original:
                return trimmed.rstrip() + "\n```"
        return trimmed

    # ----------------------------- Utilities ------------------------------

    @staticmethod
    def _strip_all_h1(content: str) -> str:
        # Remove any H1 line(s) at line start
        return re.sub(r'^\#\s+.*\n?', '', content, flags=re.MULTILINE).lstrip()

    @staticmethod
    def _extract_h1(content: str) -> Optional[str]:
        m = re.search(r'^\s*#\s+(.+?)\s*$', content or "", flags=re.MULTILINE)
        return m.group(1).strip() if m else None

    @staticmethod
    def _split_words(text: str) -> List[str]:
        return re.findall(r'\b\w+\b', text or "")

    @staticmethod
    def _norm(h: str) -> str:
        return re.sub(r'\s+', ' ', (h or "")).strip().lower()

    def _canon_niche(self, niche: Optional[str]) -> Optional[str]:
        if not niche:
            return None
        raw = str(niche).strip().lower()
        raw_norm = re.sub(r'[^a-z0-9]+', ' ', raw).strip()
        return self.NICHE_ALIASES.get(raw_norm, raw_norm)

    @staticmethod
    def _normalize_code_fences(md: str) -> str:
        """
        Convert any language-tagged or tilde fences to plain triple backticks.
        """
        if not md:
            return md
        # Normalize tildes to backticks
        md = re.sub(r'~{3,}([^\n]*)\n', '```\n', md)
        # Strip language tags on code fences (```python -> ```)
        md = re.sub(r'```[a-zA-Z0-9_+\-]+\s*\n', '```\n', md)
        # Collapse multiple backticks to 3
        md = re.sub(r'`{4,}\n', '```\n', md)
        return md

    @staticmethod
    def _ensure_closed_fences(md: str) -> str:
        """
        Ensure total number of ``` is even by appending a closing fence if necessary.
        """
        count = len(re.findall(r'```', md or ""))
        if count % 2 != 0:
            return (md or "").rstrip() + "\n```"
        return md

    def _ensure_min_references(self, body: str) -> str:
        """
        If 'Further Reading' body lacks at least two links, add a deterministic checklist note.
        (We avoid injecting specific URLs to keep this module deterministic.)
        """
        link_count = len(re.findall(r'\(https?://[^)]+\)', body or ""))
        if link_count >= 2:
            return body
        needed = 2 - link_count
        bullets = "\n".join(f"- Add reputable reference {i+1} (official docs preferred)." for i in range(needed))
        return (body.rstrip() + "\n\n" + bullets).strip()
