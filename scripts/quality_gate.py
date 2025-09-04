import re
import logging
from typing import Dict, List, Tuple, Optional, Set, Any, Union
from dataclasses import dataclass
from enum import Enum
from urllib.parse import urlparse, urlsplit, urlunsplit, parse_qsl, urlencode

logger = logging.getLogger(__name__)


class ContentType(Enum):
    TUTORIAL = "tutorial"
    REFERENCE = "reference"
    CONCEPTUAL = "conceptual"
    NEWS = "news"
    COMPARISON = "comparison"
    REVIEW = "review"


@dataclass
class QualityMetrics:
    structure_score: int
    technical_score: int
    readability_score: int
    practicality_score: int
    originality_score: int
    seo_score: int
    total_score: int
    failed_checks: List[str]
    warnings: List[str]
    suggestions: List[str]


class QualityGate:
    """
    Comprehensive quality checks for long-form technical articles (Markdown or MD+HTML).
    Deterministic, stdlib-only, conservative (heuristic) by design.

    Key improvements (readability-focused):
      • Weighted readability mode with configurable weights (transitions, passive, long-sentences, paragraphs)
      • Thresholds sourced from settings.yoast_compliance (keeps parity with Yoast preflight)
      • Sentence-start transition detection compatible with preflight
      • Adaptive "near miss" pass logic (from settings.quality_gate)
      • Backwards-compatible legacy (penalty) readability scoring
      • New evaluate(...) API to integrate with DailyAutomation
    """

    # ------------------------- Config -------------------------

    # Content-type specific minimum scores (slightly more realistic for technical content)
    MINIMUM_SCORES = {
        ContentType.TUTORIAL: {
            "structure": 80, "technical": 80, "readability": 75,
            "practicality": 85, "originality": 65, "seo": 60, "total": 78
        },
        ContentType.REFERENCE: {
            "structure": 80, "technical": 85, "readability": 75,
            "practicality": 70, "originality": 50, "seo": 75, "total": 75
        },
        ContentType.CONCEPTUAL: {
            "structure": 78, "technical": 70, "readability": 85,
            "practicality": 60, "originality": 75, "seo": 65, "total": 74
        },
        ContentType.NEWS: {
            "structure": 70, "technical": 60, "readability": 80,
            "practicality": 50, "originality": 70, "seo": 80, "total": 70
        },
        ContentType.COMPARISON: {
            "structure": 80, "technical": 75, "readability": 80,
            "practicality": 80, "originality": 70, "seo": 75, "total": 78
        },
        ContentType.REVIEW: {
            "structure": 75, "technical": 70, "readability": 80,
            "practicality": 75, "originality": 75, "seo": 80, "total": 75
        }
    }

    # Content-type specific standards
    QUALITY_STANDARDS = {
        ContentType.TUTORIAL: {
            "min_length": 1500, "max_length": 6000,
            "min_headings": 6, "max_headings": 20,
            "min_code_blocks": 4, "max_code_blocks": 15,
            "min_references": 3, "min_examples": 4,
            "readability_threshold": 70,
            "min_images": 1, "max_images": 10,
        },
        ContentType.REFERENCE: {
            "min_length": 800, "max_length": 4000,
            "min_headings": 8, "max_headings": 25,
            "min_code_blocks": 3, "max_code_blocks": 20,
            "min_references": 4, "min_examples": 2,
            "readability_threshold": 65,
            "min_images": 0, "max_images": 5,
        },
        ContentType.CONCEPTUAL: {
            "min_length": 1200, "max_length": 5000,
            "min_headings": 5, "max_headings": 15,
            "min_code_blocks": 1, "max_code_blocks": 8,
            "min_references": 3, "min_examples": 2,
            "readability_threshold": 75,
            "min_images": 1, "max_images": 8,
        },
        ContentType.NEWS: {
            "min_length": 600, "max_length": 2500,
            "min_headings": 3, "max_headings": 10,
            "min_code_blocks": 0, "max_code_blocks": 5,
            "min_references": 2, "min_examples": 1,
            "readability_threshold": 80,
            "min_images": 1, "max_images": 6,
        },
        ContentType.COMPARISON: {
            "min_length": 1800, "max_length": 5500,
            "min_headings": 7, "max_headings": 18,
            "min_code_blocks": 2, "max_code_blocks": 12,
            "min_references": 4, "min_examples": 3,
            "readability_threshold": 75,
            "min_images": 2, "max_images": 8,
        },
        ContentType.REVIEW: {
            "min_length": 1200, "max_length": 4000,
            "min_headings": 5, "max_headings": 15,
            "min_code_blocks": 1, "max_code_blocks": 8,
            "min_references": 3, "min_examples": 2,
            "readability_threshold": 80,
            "min_images": 2, "max_images": 10,
        }
    }

    SECTION_ALIASES = {
        "introduction": {
            "introduction", "overview", "comprehensive overview",
            "real-world problem statement", "getting started", "background",
            "what is", "understanding", "the basics", "preface", "foreword"
        },
        "implementation": {
            "implementation", "implementation walkthrough", "technical deep dive",
            "step-by-step guide", "walkthrough", "architecture", "design",
            "how to", "building", "creating", "developing", "setup",
            "installation", "configuration"
        },
        "examples": {
            "examples", "use case examples", "case studies",
            "worked example", "example", "demo", "demos",
            "practical example", "real-world example", "sample",
            "code examples", "usage examples"
        },
        "conclusion": {
            "conclusion", "summary cheat sheet", "further reading",
            "wrap-up", "closing thoughts", "next steps", "summary",
            "key takeaways", "final thoughts", "recap", "summary",
            "concluding remarks", "final words"
        },
    }

    # Transition starters (align with preflight/finalizer so metrics agree)
    _TRANSITIONS = {
        "therefore", "however", "in addition", "consequently", "as a result",
        "meanwhile", "crucially", "similarly", "in contrast", "specifically",
        "moving on", "ultimately", "to illustrate", "in essence", "furthermore",
        "moreover", "on the other hand", "additionally", "for example", "in practice"
    }

    # Build a robust sentence-start detector similar to preflight
    _TRANSITION_TERMS_PATTERN = "|".join(
        sorted([re.escape(t).replace(r"\ ", r"\s+") for t in _TRANSITIONS], key=len, reverse=True)
    )
    _TRANSITION_START_RE = re.compile(
        rf"""
        ^\s*
        (?:\*\*|__|\*|_)?                  # optional opening emphasis
        (?P<tw>(?:{_TRANSITION_TERMS_PATTERN}))
        (?:\*\*|__|\*|_)?                  # optional closing emphasis
        \s*
        (?:,|:|—|–|-)?                     # comma/colon/dash (optional)
        \s*
        """,
        re.IGNORECASE | re.VERBOSE,
    )

    # ------------------------- Patterns -------------------------

    _FENCED_CODE_RE = re.compile(r"(?s)(?m)^\s{0,3}(?:```|~~~)[^\n]*\n.*?\n?\s{0,3}(?:```|~~~)\s*$")
    _ANY_HEADING_RE = re.compile(r"(?m)^\s{0,3}#{1,6}\s+(.+?)\s*$")
    _H1_RE = re.compile(r"(?m)^\s{0,3}#\s+(.+?)\s*$")
    _H2_RE = re.compile(r"(?m)^\s{0,3}##\s+(.+?)\s*$")
    _H2_H3_RE = re.compile(r"(?m)^\s{0,3}#{2,3}\s+(.+?)\s*$")

    # Links: Markdown, bare URLs, and HTML <a href="...">
    _URL_RE = re.compile(r"https?://[^\s)\]]+[^\s)\].,;:!?]", re.IGNORECASE)
    _MD_LINK_RE = re.compile(r"\[[^\]]+\]\((https?://[^)]+)\)")
    _HTML_A_RE = re.compile(r'<a[^>]+href=["\'](?P<href>[^"\']+)["\']', re.IGNORECASE)

    # Images: Markdown and HTML
    _IMAGE_FULL_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
    _IMAGE_RE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")
    _IMG_HTML_RE = re.compile(r'<img[^>]+src=["\'](?P<src>[^"\']+)["\'][^>]*>', re.IGNORECASE)
    _IMG_HTML_ALT_RE = re.compile(r'<img[^>]*alt=["\'](?P<alt>[^"\']*)["\'][^>]*>', re.IGNORECASE)

    _WORD_RE = re.compile(r"\b\w+\b", re.UNICODE)
    _SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
    _PASSIVE_VOICE_RE = re.compile(r"\b(am|are|is|was|were|be|being|been)\s+\w{3,}ed\b", re.IGNORECASE)

    # URL tracking cleanup (helps domain diversity & duplicate counting)
    _TRACKING_KEYS = {"fbclid", "gclid", "yclid", "mc_eid", "mc_cid"}
    _TRACKING_PREFIXES = ("utm_", "ref_")

    # ------------------------- Init (robust) -------------------------

    def __init__(self,
                 content_type: Union[ContentType, str, Dict[str, Any], None] = ContentType.TUTORIAL,
                 settings: Optional[Dict[str, Any]] = None):
        """
        Accepts:
          - ContentType enum or string alias (e.g., "article", "tutorial", ...)
          - or a full `settings` dict (from settings.json); if provided, thresholds/weights are read from it.
        """
        # Settings (optional; enables parity with Yoast preflight)
        self.settings = settings or {}
        yoast = (self.settings.get("yoast_compliance") or {}) if isinstance(self.settings, dict) else {}

        # Readability knobs (defaults are safe if not present)
        self.readability_mode: str = str(yoast.get("readability_mode", "weighted")).lower()
        self.adaptive_readability: bool = bool(yoast.get("adaptive_readability", True))

        # Thresholds aligned with Yoast preflight
        self.trans_min_pct: float = float(yoast.get("min_transition_word_percent", 30))
        self.passive_max_pct: float = float(yoast.get("passive_voice_max_percent", 7))
        self.long_sentence_max_pct: float = float(yoast.get("max_long_sentence_percent", 20))
        self.paragraph_words_max: int = int(yoast.get("max_paragraph_words", 200))

        # Weights for weighted mode
        default_weights = {"transitions": 0.35, "passive": 0.30, "long_sentence": 0.25, "paragraph": 0.10}
        w = yoast.get("readability_weights", default_weights) or default_weights
        # Normalize weights to sum=1
        s = sum(float(w.get(k, 0)) for k in default_weights.keys()) or 1.0
        self.readability_weights = {k: float(w.get(k, 0)) / s for k in default_weights.keys()}

        # Quality-gate (additional) controls
        qg = (self.settings.get("quality_gate") or {}) if isinstance(self.settings, dict) else {}
        self.qg_min_readability: float = float(qg.get("min_readability", 55))
        self.qg_allow_near_miss: bool = bool(qg.get("allow_near_miss", True))
        self.qg_near_miss_margin: float = float(qg.get("near_miss_margin", 2.0))
        self.qg_min_seo: float = float(qg.get("min_seo", 60))
        self.qg_max_warnings: int = int(qg.get("max_warnings", 3))
        self.qg_min_word_count: int = int(qg.get("min_word_count", 1100))

        # Content type wiring
        ct = self._coerce_content_type(content_type if not isinstance(content_type, dict) else content_type)
        self.content_type: ContentType = ct
        self.quality_standards = self.QUALITY_STANDARDS[ct]
        self.minimum_scores = self.MINIMUM_SCORES[ct]
        logger.info(f"[QualityGate] Using content type: {self.content_type.value} | readability_mode={self.readability_mode}")

    @classmethod
    def _content_type_from_settings(cls, cfg: Dict[str, Any]) -> Optional[str]:
        """Extract content_type string from various plausible places in a settings dict."""
        try_paths = [
            ("quality_gate", "content_type"),
            ("content_strategy", "content_type"),
            ("content_type",),
        ]
        for path in try_paths:
            cur: Any = cfg
            ok = True
            for key in path:
                if isinstance(cur, dict) and key in cur:
                    cur = cur[key]
                else:
                    ok = False
                    break
            if ok and isinstance(cur, str) and cur.strip():
                return cur.strip()
        return None

    @classmethod
    def _coerce_content_type(cls, value: Union[ContentType, str, Dict[str, Any], None]) -> ContentType:
        """Convert mixed inputs into a valid ContentType."""
        if isinstance(value, ContentType):
            return value

        if isinstance(value, dict):
            s = cls._content_type_from_settings(value)
            value = s if isinstance(s, str) else "article"

        if isinstance(value, str):
            s = value.strip().lower()
            aliases = {
                "article": ContentType.TUTORIAL,
                "how-to": ContentType.TUTORIAL, "howto": ContentType.TUTORIAL,
                "guide": ContentType.TUTORIAL, "walkthrough": ContentType.TUTORIAL, "pillar": ContentType.TUTORIAL,
                "tutorial": ContentType.TUTORIAL,

                "reference": ContentType.REFERENCE, "api": ContentType.REFERENCE,
                "docs": ContentType.REFERENCE, "documentation": ContentType.REFERENCE,

                "conceptual": ContentType.CONCEPTUAL, "concepts": ContentType.CONCEPTUAL, "explainer": ContentType.CONCEPTUAL,

                "news": ContentType.NEWS, "update": ContentType.NEWS, "changelog": ContentType.NEWS, "announcement": ContentType.NEWS,

                "comparison": ContentType.COMPARISON, "vs": ContentType.COMPARISON, "versus": ContentType.COMPARISON,
                "compare": ContentType.COMPARISON, "alternative": ContentType.COMPARISON,

                "review": ContentType.REVIEW, "evaluation": ContentType.REVIEW, "assessment": ContentType.REVIEW, "rating": ContentType.REVIEW,
            }
            if s in aliases:
                return aliases[s]
            for ct in ContentType:
                if s == ct.value:
                    return ct
            logger.warning(f"[QualityGate] Unknown content_type '{value}', defaulting to 'tutorial'.")
            return ContentType.TUTORIAL

        return ContentType.TUTORIAL

    # ------------------------- Public API -------------------------

    def evaluate(self, *,
                 title: str,
                 meta: Optional[str],
                 content: str,
                 yoast_report: Optional[Dict[str, Any]] = None,
                 niche: Optional[str] = None,
                 keyphrase: Optional[str] = None,
                 topic: Optional[str] = None) -> Tuple[bool, List[str], Dict[str, Any]]:
        """
        Lightweight gate used by the runner after Yoast preflight.
        Returns (ok, reasons, extra_metrics) and DOES NOT duplicate Yoast gating unless far off.

        Strategy:
          • Compute internal readability (weighted or legacy) with detailed sub-metrics
          • Compare with settings.quality_gate.{min_readability, allow_near_miss, near_miss_margin}
          • If it's a near miss and allow_near_miss=True, pass with warning
          • Always return diagnostics for visibility
        """
        reasons: List[str] = []
        extra: Dict[str, Any] = {}

        # Basic length guard (cheap)
        wc = self._word_count(content or "")
        if wc < self.qg_min_word_count:
            reasons.append(f"word_count below minimum ({wc} < {self.qg_min_word_count})")

        # Readability components (aligned to Yoast-style)
        comps = self._readability_components(content)
        if self.readability_mode == "weighted":
            read_internal = int(round(self._compute_weighted_readability(comps)))
        else:
            read_internal = self._legacy_readability_score(content)

        yoast_read = None
        if isinstance(yoast_report, dict):
            try:
                yoast_read = float(yoast_report.get("readability"))
            except Exception:
                yoast_read = None

        # Prefer the better of internal vs Yoast as the effective score
        candidates = [x for x in [yoast_read, read_internal] if isinstance(x, (int, float))]
        effective_read = max(candidates) if candidates else read_internal

        # Gate (lenient near-miss policy)
        if effective_read < self.qg_min_readability - self.qg_near_miss_margin:
            reasons.append(
                f"readability below minimum (score {effective_read:.1f} < {self.qg_min_readability})"
            )
        elif effective_read < self.qg_min_readability:
            if self.qg_allow_near_miss:
                reasons.append(
                    f"near miss: readability {effective_read:.1f} within {self.qg_near_miss_margin} points of minimum {self.qg_min_readability}"
                )
            else:
                reasons.append(
                    f"readability below minimum (score {effective_read:.1f} < {self.qg_min_readability})"
                )

        # Optional SEO nudge (do not double-fail if Yoast already enforced earlier)
        yoast_seo = None
        if isinstance(yoast_report, dict):
            try:
                yoast_seo = float(yoast_report.get("seo"))
            except Exception:
                yoast_seo = None
        if yoast_seo is not None and yoast_seo < self.qg_min_seo:
            reasons.append(f"seo score low (Yoast {yoast_seo:.1f} < {self.qg_min_seo})")

        # Diagnostics
        extra.update({
            "word_count": wc,
            "mode": self.readability_mode,
            "readability_internal": read_internal,
            "readability_yoast": yoast_read,
            "readability_effective": effective_read,
            "readability_components": comps,
            "readability_weights": self.readability_weights,
            "thresholds": {
                "trans_min_pct": self.trans_min_pct,
                "passive_max_pct": self.passive_max_pct,
                "long_sentence_max_pct": self.long_sentence_max_pct,
                "paragraph_words_max": self.paragraph_words_max,
                "qg_min_readability": self.qg_min_readability,
                "qg_near_miss_margin": self.qg_near_miss_margin
            }
        })

        ok = not any("below minimum" in r for r in reasons)
        return ok, reasons, extra

    def check_article(self, article: Dict) -> Tuple[bool, QualityMetrics]:
        """
        Full, detailed check for batch/offline usage (kept for backward-compat).
        Uses the configured readability mode for the readability_score.
        """
        content = (article.get("content") or "").strip()
        title = (article.get("title") or "").strip()
        keyphrase = (article.get("keyphrase") or "").strip()
        meta_description = (article.get("meta_description") or "").strip()
        site_domain = (article.get("site_domain") or "").strip().lower()

        failed_checks: List[str] = []
        warnings: List[str] = []
        suggestions: List[str] = []

        if not content:
            failed_checks.append("Empty content")
            return False, QualityMetrics(0, 0, 0, 0, 0, 0, 0, failed_checks, warnings, suggestions)

        analysis_content = self._get_sample_for_analysis(content)

        # Readability score (mode-aware)
        if self.readability_mode == "weighted":
            read_comps = self._readability_components(analysis_content)
            readability_score = int(round(self._compute_weighted_readability(read_comps)))
        else:
            readability_score = self._legacy_readability_score(analysis_content)

        scores = {
            "structure": self._check_structure(content, title),
            "technical": self._check_technical_depth(content),
            "readability": readability_score,
            "practicality": self._check_practical_elements(content),
            "originality": self._check_original_insights(content),
            "seo": self._check_seo_elements(
                content=content,
                title=title,
                keyphrase=keyphrase or None,
                meta=meta_description or None,
                site_domain=site_domain or None,
            ),
        }

        # Code quality contributes a small bonus to technical
        code_quality_bonus = self._check_code_quality(content)
        if code_quality_bonus > 0:
            scores["technical"] = min(100, scores["technical"] + code_quality_bonus)
            suggestions.append(f"Code quality bonus: +{code_quality_bonus} points")

        total_score = sum(scores.values()) // len(scores)

        # Check minimum scores
        for metric, score in scores.items():
            if score < self.minimum_scores[metric]:
                failed_checks.append(f"{metric} below minimum (score {score} < {self.minimum_scores[metric]})")

        # Readability hard threshold (content-type specific)
        if scores["readability"] < self.quality_standards["readability_threshold"]:
            failed_checks.append(
                f"readability below threshold (score {scores['readability']} < {self.quality_standards['readability_threshold']})"
            )

        # Borderline warnings
        for metric, score in scores.items():
            if self.minimum_scores[metric] <= score < self.minimum_scores[metric] + 10:
                warnings.append(f"{metric} score is borderline ({score}/100)")

        # Suggestions
        suggestions.extend(self._generate_suggestions(scores, content, title, keyphrase, meta_description))

        metrics = QualityMetrics(
            structure_score=scores["structure"],
            technical_score=scores["technical"],
            readability_score=scores["readability"],
            practicality_score=scores["practicality"],
            originality_score=scores["originality"],
            seo_score=scores["seo"],
            total_score=total_score,
            failed_checks=failed_checks,
            warnings=warnings,
            suggestions=suggestions,
        )
        return len(failed_checks) == 0, metrics

    # ------------------------- Readability (new) -------------------------

    def _readability_components(self, content: str) -> Dict[str, float]:
        """
        Compute Yoast-aligned readability components on NON-code text:
          • transitions_pct = % sentences starting with a transition
          • passive_pct     = % sentences matching passive pattern
          • long_sentence_pct= % sentences with >20 words
          • paragraph_max_words = max words found in any paragraph
        """
        # Remove code & strip links/anchors for readability metrics
        stripped = self._FENCED_CODE_RE.sub("\n", content)
        stripped = self._MD_LINK_RE.sub(" ", stripped)
        stripped = self._URL_RE.sub(" ", stripped)
        stripped = self._HTML_A_RE.sub(" ", stripped)

        sentences = [s.strip() for s in self._SENTENCE_SPLIT_RE.split(stripped) if s.strip()]
        words = self._WORD_RE.findall(stripped)

        # Sentences → words per sentence
        sent_word_counts = [len(self._WORD_RE.findall(s)) for s in sentences] if sentences else []
        long_sentence_hits = sum(1 for n in sent_word_counts if n > 20)
        long_sentence_pct = 100.0 * long_sentence_hits / max(1, len(sentences))

        # Passive voice
        passive_hits = sum(1 for s in sentences if self._PASSIVE_VOICE_RE.search(s))
        passive_pct = 100.0 * passive_hits / max(1, len(sentences))

        # Transition starts — sentence-based, robust to emphasis & punctuation
        transition_hits = sum(1 for s in sentences if self._TRANSITION_START_RE.match(s))
        transitions_pct = 100.0 * transition_hits / max(1, len(sentences))

        # Paragraphs
        paragraphs = [p.strip() for p in stripped.split("\n\n") if p.strip()]
        paragraph_max_words = max((len(self._WORD_RE.findall(p)) for p in paragraphs), default=0)

        return {
            "transitions_pct": transitions_pct,
            "passive_pct": passive_pct,
            "long_sentence_pct": long_sentence_pct,
            "paragraph_max_words": float(paragraph_max_words),
            "avg_sentence_len": (len(words) / max(1, len(sentences))) if sentences else 0.0,  # diagnostic only
        }

    def _compute_weighted_readability(self, comps: Dict[str, float]) -> float:
        """
        Map each component to 0..100, then combine by weights.
        Scaling:
          • transitions: 100 if >= trans_min_pct, else linear up to min
          • passive:     100 if <= passive_max_pct, else linear down to 0
          • long_sent:   100 if <= long_sentence_max_pct, else linear down to 0
          • paragraph:   100 if max_words <= paragraph_words_max, else decay to 0 by 2× limit
        """
        # transitions
        if comps["transitions_pct"] >= self.trans_min_pct:
            r_trans = 100.0
        else:
            r_trans = max(0.0, 100.0 * comps["transitions_pct"] / max(1e-6, self.trans_min_pct))

        # passive
        if comps["passive_pct"] <= self.passive_max_pct:
            r_pass = 100.0
        else:
            over = comps["passive_pct"] - self.passive_max_pct
            r_pass = max(0.0, 100.0 - (over / max(1e-6, 100.0 - self.passive_max_pct)) * 100.0)

        # long sentences
        if comps["long_sentence_pct"] <= self.long_sentence_max_pct:
            r_long = 100.0
        else:
            over = comps["long_sentence_pct"] - self.long_sentence_max_pct
            r_long = max(0.0, 100.0 - (over / max(1e-6, 100.0 - self.long_sentence_max_pct)) * 100.0)

        # paragraph max length — linear decay until 2× limit
        if comps["paragraph_max_words"] <= self.paragraph_words_max:
            r_para = 100.0
        else:
            cap = self.paragraph_words_max * 2.0
            over = min(comps["paragraph_max_words"], cap) - self.paragraph_words_max
            r_para = max(0.0, 100.0 - (over / max(1e-6, self.paragraph_words_max)) * 100.0)

        w = self.readability_weights
        # ensure weights cover keys (defensive)
        total_w = sum(w.get(k, 0.0) for k in ("transitions", "passive", "long_sentence", "paragraph")) or 1.0
        score = (
            w.get("transitions", 0.0) * r_trans +
            w.get("passive", 0.0) * r_pass +
            w.get("long_sentence", 0.0) * r_long +
            w.get("paragraph", 0.0) * r_para
        ) / total_w

        return max(0.0, min(100.0, score))

    def _legacy_readability_score(self, content: str) -> int:
        """
        Original penalty-style scorer (kept for compatibility). Uses avg lengths,
        passive ratio, paragraph size, and transition % (paragraph-based before).
        """
        score = 100

        stripped = self._FENCED_CODE_RE.sub("\n", content)
        stripped = self._MD_LINK_RE.sub(" ", stripped)
        stripped = self._URL_RE.sub(" ", stripped)
        stripped = self._HTML_A_RE.sub(" ", stripped)

        sentences = [s.strip() for s in self._SENTENCE_SPLIT_RE.split(stripped) if s.strip()]
        words = self._WORD_RE.findall(stripped)

        if not sentences or not words:
            return 0

        # Sentence and word length — relaxed for technical writing
        avg_sentence_len = len(words) / max(1, len(sentences))
        avg_word_len = sum(len(w) for w in words) / max(1, len(words))

        if avg_sentence_len > 29:
            score -= 22
        elif avg_sentence_len > 23:
            score -= 10

        if avg_word_len > 6.2:
            score -= 15
        elif avg_word_len > 5.7:
            score -= 8

        # Passive voice
        passive_hits = len(self._PASSIVE_VOICE_RE.findall(stripped))
        passive_ratio = passive_hits / max(1, len(sentences))
        if passive_ratio > 0.25:
            score -= 22
        elif passive_ratio > 0.15:
            score -= 10

        # Paragraph length
        paragraphs = [p.strip() for p in stripped.split("\n\n") if p.strip()]
        if paragraphs:
            max_para_words = max(len(self._WORD_RE.findall(p)) for p in paragraphs)
            if max_para_words > 260:
                score -= 12
            elif max_para_words > 200:
                score -= 5

        # Transition-word presence (paragraph-based legacy)
        trans_pct = self._transition_pct_paragraphs(stripped)
        if trans_pct < 12:
            score -= 12
        elif trans_pct < 20:
            score -= 6

        return max(0, score)

    # ------------------------- Enhanced Checks (unchanged) -------------------------

    def _check_structure(self, content: str, title: str) -> int:
        """Check article structure, organization, and length."""
        score = 100

        # Headings analysis
        h1s = self._H1_RE.findall(content)
        h2s = self._H2_RE.findall(content)
        h2_h3 = self._H2_H3_RE.findall(content)
        headings_norm = {self._normalize_heading(h) for h in (h1s + h2_h3)}

        # Heading counts
        if len(h2_h3) < self.quality_standards["min_headings"]:
            score -= 25
        elif len(h2_h3) > self.quality_standards["max_headings"]:
            score -= 15

        # Title validation (tolerant)
        if len(h1s) != 1:
            score -= 12
        else:
            if not self._titles_equivalent(h1s[0], title):
                score -= 4

        # Heading hierarchy sanity: H3 before any H2
        first_h2 = self._H2_RE.search(content)
        first_h3 = re.search(r"(?m)^\s{0,3}###\s+", content)
        if first_h3 and (not first_h2 or first_h3.start() < first_h2.start()):
            score -= 5

        # Duplicate H2s (exact duplicates) — penalize lightly
        if len(h2s) != len(set(self._norm_title(h) for h in h2s)):
            score -= 5

        # Required sections
        missing_buckets = []
        for bucket, aliases in self.SECTION_ALIASES.items():
            if not self._section_present(aliases, headings_norm):
                missing_buckets.append(bucket)
        if missing_buckets:
            score -= len(missing_buckets) * 10

        # Length validation
        word_count = self._word_count(content)
        if word_count < self.quality_standards["min_length"]:
            score -= 28
        elif word_count > self.quality_standards["max_length"]:
            score -= 12

        # Image validation & missing alt text penalty (MD + HTML)
        md_images = self._IMAGE_FULL_RE.findall(content)
        html_imgs = self._IMG_HTML_RE.findall(content)
        total_imgs = len(md_images) + len(html_imgs)

        if total_imgs < self.quality_standards["min_images"]:
            score -= 10
        elif total_imgs > self.quality_standards["max_images"]:
            score -= 5

        # ALT text check (MD + HTML)
        missing_alt_md = sum(1 for alt, _ in md_images if not (alt or "").strip())
        missing_alt_html = 0
        for m in self._IMG_HTML_RE.finditer(content):
            tag = m.group(0)
            alt_m = self._IMG_HTML_ALT_RE.search(tag)
            if not alt_m or not (alt_m.group("alt") or "").strip():
                missing_alt_html += 1
        if (missing_alt_md + missing_alt_html) > 0:
            score -= min(10, (missing_alt_md + missing_alt_html) * 3)

        return max(0, score)

    def _check_technical_depth(self, content: str) -> int:
        """Check technical depth and signals of rigor."""
        score = 100

        # Code blocks analysis
        code_blocks = len(self._FENCED_CODE_RE.findall(content))
        if code_blocks < self.quality_standards["min_code_blocks"]:
            score -= 30
        elif code_blocks > self.quality_standards["max_code_blocks"]:
            score -= 20

        # Unique references (MD, bare URL, and HTML anchors)
        all_urls = self._collect_urls(content)
        if len(all_urls) < self.quality_standards["min_references"]:
            score -= 25

        # Technical indicators
        technical_indicators = [
            "complexity", "performance", "benchmark", "optimization",
            "trade-off", "tradeoffs", "latency", "throughput", "scalability",
            "big-o", "space complexity", "time complexity", "memory usage",
            "profiling", "load testing", "concurrency", "race condition",
            "algorithm", "data structure", "pattern", "framework", "library"
        ]
        cl = content.lower()
        found_indicators = sum(1 for i in technical_indicators if i in cl)
        if found_indicators < 3:
            score -= 20

        return max(0, score)

    def _check_practical_elements(self, content: str) -> int:
        """Check for practical, actionable content."""
        score = 100
        cl = content.lower()

        examples = len(re.findall(r"\bexample\b|for instance|e\.g\.|case study|walkthrough|exercise|challenge", cl))
        tips = len(re.findall(r"\btip:\b|recommendation|best practice|pro tip|guideline", cl))

        if examples < self.quality_standards["min_examples"]:
            score -= 35
        if tips < 2:
            score -= 25

        actionable_phrases = [
            "you should", "we recommend", "best approach", "consider using",
            "avoid", "do not", "ensure that", "quick check", "make sure",
            "always", "never", "important to", "critical to", "essential"
        ]
        actionable_count = sum(1 for p in actionable_phrases if p in cl)
        if actionable_count < 4:
            score -= 20

        return max(0, score)

    def _check_original_insights(self, content: str) -> int:
        """Check for original insights using enhanced signals."""
        score = 100
        cl = content.lower()

        insight_indicators = [
            "in practice", "empirical", "based on testing", "measured",
            "results show", "trade-off", "pitfall", "anti-pattern",
            "caveat", "limitation", "gotcha", "counterexample",
            "what makes this different", "key insight", "important consideration",
            "surprisingly", "contrary to popular belief", "common misconception"
        ]
        count = sum(1 for i in insight_indicators if i in cl)
        if count < 3:
            score -= 45
        elif count < 5:
            score -= 20

        contrast_terms = ["unlike", "whereas", "however", "compared to", "versus", "vs.", "while"]
        if not any(t in cl for t in contrast_terms):
            score -= 15

        return max(0, score)

    def _check_seo_elements(
        self,
        *,
        content: str,
        title: str,
        keyphrase: Optional[str] = None,
        meta: Optional[str] = None,
        site_domain: Optional[str] = None,
    ) -> int:
        """
        Score SEO signals. Starts from 100 and subtracts on issues.
        Counts MD links, bare URLs, and HTML anchors; handles tracking params.
        """
        score = 100
        cl = content.lower()
        title_lower = (title or "").lower()
        key = (keyphrase or "").lower().strip()

        # Title length & keyphrase
        if not (40 <= len(title) <= 60):
            score -= 8
        if key and key not in title_lower:
            score -= 10

        # First paragraph contains keyphrase or salient title term
        paras = content.split("\n\n")
        first_para = (paras[0] if paras else content[:200]).lower()
        salient_terms = [w for w in title_lower.split() if len(w) > 3][:3]
        if key:
            if key not in first_para:
                score -= 8
        elif salient_terms and not any(t in first_para for t in salient_terms):
            score -= 6

        # Headings with keyphrase or salient terms
        headings = " ".join(self._ANY_HEADING_RE.findall(content)).lower()
        if key:
            if key not in headings:
                score -= 8
        elif salient_terms and not any(t in headings for t in salient_terms):
            score -= 6

        # Meta description length (if provided)
        if meta is not None:
            if not (120 <= len(meta.strip()) <= 160):
                score -= 8
        else:
            if len(first_para) < 100:
                score -= 4

        # Link checks (MD + bare + HTML anchors)
        urls = self._collect_urls(content)
        if len(urls) < 2:
            score -= 10

        # Domain diversity
        domains = {self._norm_domain(u) for u in urls if "://" in u}
        domains.discard("")  # safety
        if len(domains) < 2:
            score -= 5

        # Internal link presence if site_domain provided
        if site_domain:
            site_domain = site_domain.lower().replace("www.", "")
            internal = [d for d in domains if site_domain in d]
            if not internal:
                score -= 5

        # Images alt text quality (MD + HTML)
        md_imgs = self._IMAGE_FULL_RE.findall(content)
        html_img_tags = list(self._IMG_HTML_RE.finditer(content))
        empty_alt = sum(1 for alt, _ in md_imgs if not (alt or "").strip())
        for tag_m in html_img_tags:
            tag = tag_m.group(0)
            alt_m = self._IMG_HTML_ALT_RE.search(tag)
            if not alt_m or not (alt_m.group("alt") or "").strip():
                empty_alt += 1
        if empty_alt:
            score -= min(10, empty_alt * 3)

        return max(0, min(100, score))

    def _check_code_quality(self, content: str) -> int:
        """Basic code quality indicators in code blocks (returns bonus up to +20)."""
        bonus = 0
        blocks = self._FENCED_CODE_RE.findall(content)

        for block in blocks:
            lines_raw = block.splitlines()
            lang_line = lines_raw[0] if lines_raw else ""
            has_lang = bool(re.search(r"```[a-zA-Z]+|~~~[a-zA-Z]+", lang_line))
            if has_lang:
                bonus += 2  # reward specifying language for syntax highlighting

            body_lines = [ln for ln in lines_raw[1:-1] if ln.strip()]
            if not body_lines:
                continue

            comments = sum(1 for line in body_lines if line.strip().startswith(("#", "//")))
            comment_ratio = comments / max(1, len(body_lines))
            if comment_ratio >= 0.1:
                bonus += 3

            # Descriptive variable/function names heuristic
            descriptive = sum(1 for ln in body_lines if re.search(r"\b[a-z_]{8,}\b", ln))
            if descriptive / max(1, len(body_lines)) > 0.25:
                bonus += 2

            bad_patterns = [r"\bbare except\b", r"\bprint\(", r"TODO"]
            if not any(re.search(p, " ".join(body_lines), re.IGNORECASE) for p in bad_patterns):
                bonus += 2

        return min(20, bonus)

    # ------------------------- Utilities -------------------------

    def _get_sample_for_analysis(self, content: str, max_chars: int = 10000) -> str:
        """Get representative sample for analysis of very long articles."""
        if len(content) <= max_chars:
            return content
        third = max_chars // 3
        sixth = max_chars // 6
        return "\n".join([
            content[:third],
            content[len(content)//2 - sixth: len(content)//2 + sixth],
            content[-third:]
        ])

    @staticmethod
    def _normalize_heading(h: str) -> str:
        h2 = re.sub(r"\s+", " ", h).strip().lower()
        h2 = re.sub(r"^[^\w]+", "", h2)
        return h2

    @staticmethod
    def _word_count(text: str) -> int:
        return len(re.findall(r"\b\w+\b", text))

    @staticmethod
    def _strip_tracking(url: str) -> str:
        try:
            parts = urlsplit(url)
            kept = []
            for k, v in parse_qsl(parts.query, keep_blank_values=True):
                kl = k.lower()
                if any(kl.startswith(p) for p in QualityGate._TRACKING_PREFIXES) or kl in QualityGate._TRACKING_KEYS:
                    continue
                kept.append((k, v))
            return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(kept, doseq=True), parts.fragment))
        except Exception:
            return url

    def _collect_urls(self, content: str) -> Set[str]:
        """Collect unique URLs from Markdown links, bare URLs, and HTML anchors with tracking stripped."""
        urls = set(self._MD_LINK_RE.findall(content))
        urls |= set(self._URL_RE.findall(content))
        urls |= {m.group("href") for m in self._HTML_A_RE.finditer(content)}
        return {self._strip_tracking(u) for u in urls}

    @staticmethod
    def _norm_domain(url: str) -> str:
        try:
            return urlparse(url).netloc.lower().replace("www.", "")
        except Exception:
            return ""

    def _section_present(self, aliases: Set[str], headings_norm: Set[str]) -> bool:
        for heading in headings_norm:
            for alias in aliases:
                if (
                    heading == alias
                    or heading.startswith(alias + " ")
                    or re.search(rf"\b{re.escape(alias)}\b", heading)
                ):
                    return True
        return False

    def _norm_title(self, s: str) -> str:
        """Normalize titles for tolerant equality checks."""
        s = s.strip().lower()
        s = re.sub(r"\s+", " ", s)
        s = re.sub(r"[^\w\s\-]", "", s)
        return s

    def _titles_equivalent(self, h1: str, title: str) -> bool:
        """
        Tolerant equality: ignore em-dash suffixes, numeric variants, and small wording drift.
        """
        a = self._norm_title(h1)
        b = self._norm_title(title)
        if a == b:
            return True
        # Strip trivial suffixes like " — 1", "-1 in python", etc.
        a2 = re.sub(r"\s[-–—:]\s?\d+\b.*$", "", a)
        b2 = re.sub(r"\s[-–—:]\s?\d+\b.*$", "", b)
        if a2 == b2:
            return True
        # Token overlap heuristic
        ta, tb = set(a2.split()), set(b2.split())
        inter = len(ta & tb)
        return inter >= max(1, min(len(ta), len(tb)) - 2)

    # Legacy paragraph-based transition %
    def _transition_pct_paragraphs(self, text: str) -> float:
        paras = [p.strip() for p in text.split("\n\n") if p.strip()]
        if not paras:
            return 0.0
        starters = 0
        for p in paras:
            first = p.split(" ", 1)[0].strip(" ,").lower()
            if any(first.startswith(t) for t in self._TRANSITIONS):
                starters += 1
        return 100.0 * starters / max(1, len(paras))

    # ------------------------- Reporting -------------------------

    def generate_quality_report(
        self,
        metrics: QualityMetrics,
        content_type: Optional[ContentType] = None
    ) -> str:
        """Generate detailed quality report with content-type context."""
        ct = content_type or self.content_type
        mins = self.MINIMUM_SCORES[ct]

        report = [
            f"QUALITY ASSESSMENT REPORT - {ct.value.upper()}",
            "=" * 60,
            f"Structure:    {metrics.structure_score}/100 (min: {mins['structure']})",
            f"Technical:    {metrics.technical_score}/100 (min: {mins['technical']})",
            f"Readability:  {metrics.readability_score}/100 (min: {mins['readability']})",
            f"Practicality: {metrics.practicality_score}/100 (min: {mins['practicality']})",
            f"Originality:  {metrics.originality_score}/100 (min: {mins['originality']})",
            f"SEO:          {metrics.seo_score}/100 (min: {mins['seo']})",
            f"Overall:      {metrics.total_score}/100 (min: {mins['total']})",
            "",
        ]

        if metrics.failed_checks:
            report.append("❌ FAILED CHECKS:")
            for check in metrics.failed_checks:
                report.append(f"   • {check}")
            report.append("")

        if metrics.warnings:
            report.append("⚠️  WARNINGS:")
            for warning in metrics.warnings:
                report.append(f"   • {warning}")
            report.append("")

        if metrics.suggestions:
            report.append("💡 SUGGESTIONS FOR IMPROVEMENT:")
            for suggestion in metrics.suggestions:
                report.append(f"   • {suggestion}")
            report.append("")

        if not metrics.failed_checks:
            report.append("✅ ALL CHECKS PASSED")

        return "\n".join(report)

    # ------------------------- Helper Methods -------------------------

    @classmethod
    def detect_content_type(cls, content: str, title: str) -> ContentType:
        """Auto-detect content type based on content and title."""
        text = f"{title} {content}".lower()

        if re.search(r"\b(how to|tutorial|step[-\s]?by[-\s]?step|guide|walkthrough)\b", text):
            return ContentType.TUTORIAL
        if re.search(r"\b(reference|api|documentation|specification|manual)\b", text):
            return ContentType.REFERENCE
        if re.search(r"\b(vs\.?|versus|comparison|compare|alternative)\b", text):
            return ContentType.COMPARISON
        if re.search(r"\b(news|update|release|announcement|launch)\b", text):
            return ContentType.NEWS
        if re.search(r"\b(review|rating|score|evaluation|assessment)\b", text):
            return ContentType.REVIEW
        return ContentType.CONCEPTUAL

    def update_content_type(self, new_type: ContentType) -> None:
        """Dynamically update content type and corresponding standards."""
        self.content_type = new_type
        self.quality_standards = self.QUALITY_STANDARDS[new_type]
        self.minimum_scores = self.MINIMUM_SCORES[new_type]

    def get_content_standards(self) -> Dict:
        """Get current content type standards for inspection."""
        return {
            "content_type": self.content_type.value,
            "standards": self.quality_standards,
            "minimum_scores": self.minimum_scores
        }


# ------------------------- Utility Functions -------------------------

def create_quality_gate(article: Dict, settings: Optional[Dict[str, Any]] = None) -> QualityGate:
    """Factory function to create appropriate QualityGate for article."""
    content_type = QualityGate.detect_content_type(
        article.get("content", ""),
        article.get("title", "")
    )
    return QualityGate(content_type, settings=settings)


def batch_quality_check(articles: List[Dict], settings: Optional[Dict[str, Any]] = None) -> Dict:
    """Perform quality check on multiple articles and return statistics."""
    results = {
        "total_articles": len(articles),
        "passed": 0,
        "failed": 0,
        "average_score": 0.0,
        "content_type_distribution": {},
        "common_issues": []
    }

    total_score = 0
    issue_counter: Dict[str, int] = {}

    for article in articles:
        gate = create_quality_gate(article, settings=settings)
        passed, metrics = gate.check_article(article)

        if passed:
            results["passed"] += 1
        else:
            results["failed"] += 1
            for issue in metrics.failed_checks:
                issue_counter[issue] = issue_counter.get(issue, 0) + 1

        total_score += metrics.total_score

        # Track content type distribution
        ct_val = gate.content_type.value
        results["content_type_distribution"][ct_val] = \
            results["content_type_distribution"].get(ct_val, 0) + 1

    if articles:
        results["average_score"] = round(total_score / len(articles), 2)

    # Top 5 common issues
    results["common_issues"] = sorted(
        issue_counter.items(),
        key=lambda x: x[1],
        reverse=True
    )[:5]

    return results
