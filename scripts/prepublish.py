# prepublish.py
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

try:
    # package-relative import
    from .yoast_preflight import YoastPreflight  # type: ignore
except Exception:  # pragma: no cover
    from yoast_preflight import YoastPreflight  # type: ignore

try:
    # package-relative import
    from .seo_finalizer import finalize as deterministic_finalize  # type: ignore
except Exception:  # pragma: no cover
    from seo_finalizer import finalize as deterministic_finalize  # type: ignore


def run_prepublish(
    settings: Dict[str, Any],
    content: str,
    title: str,
    meta: str,
    keyphrase: str,
    niche: str,
    article_linker: Optional[Any] = None,
    logger: Optional[Any] = None,
) -> Tuple[str, str, str, Dict[str, Any]]:
    """
    1) Run YoastPreflight.fix() using knobs from settings.json
    2) Optionally run seo_finalizer.finalize() if:
         - finalizer.enabled is True, AND
         - (only_when_close == False) OR (we are within run_if_within_points of both targets)
       Accept finalizer output only if it doesn't regress SEO/Readability beyond allowed points.
    Returns: (content, title, meta, report_dict)
    """
    yp = YoastPreflight(settings, article_linker=article_linker)

    # Pass 1: deterministic, code-safe, iterative preflight
    content1, title1, meta1, rep1 = yp.fix(
        content=content, title=title, meta=meta, keyphrase=keyphrase, niche=niche
    )

    fin_cfg = (settings.get("finalizer") or {})
    if not fin_cfg.get("enabled", True):
        return content1, title1, meta1, {"preflight": rep1, "finalizer": None, "finalizer_skipped": True}

    # Check closeness to targets
    gap = int(fin_cfg.get("run_if_within_points", 5))
    only_when_close = bool(fin_cfg.get("only_when_close", True))

    need_seo = max(0.0, yp.target_seo - float(rep1.get("seo", 0.0)))
    need_read = max(0.0, yp.target_read - float(rep1.get("readability", 0.0)))
    within = (need_seo <= gap) and (need_read <= gap)

    if only_when_close and not within:
        # Respect the "only when close" guardrail
        return content1, title1, meta1, {
            "preflight": rep1,
            "finalizer": None,
            "finalizer_skipped": True,
            "reason": f"outside close window (SEO Δ={need_seo:.1f}, READ Δ={need_read:.1f}, gap={gap})",
        }

    # Pass 2: deterministic finalizer (rule-based normalizer)
    t2, m2, c2, fin_report = deterministic_finalize(
        title=title1,
        meta=meta1,
        content_md=content1,
        keyphrase=keyphrase,
        knobs=(settings.get("yoast_compliance") or {}),
    )

    # Rescore using same scorer to compare fairly
    seo2, read2, metrics2, warn2 = yp._score(c2, keyphrase)  # type: ignore[attr-defined]
    allow_seo_reg = int(fin_cfg.get("allow_seo_regression_points", 0))
    allow_read_reg = int(fin_cfg.get("allow_read_regression_points", 0))

    base_seo = float(rep1.get("seo", 0.0))
    base_read = float(rep1.get("readability", 0.0))

    # Reject if we regress too far
    if (seo2 + allow_seo_reg) < base_seo or (read2 + allow_read_reg) < base_read:
        if logger:
            logger.info(
                f"[finalizer] rejected: SEO {seo2:.1f} vs {base_seo:.1f} (allow {allow_seo_reg}), "
                f"READ {read2:.1f} vs {base_read:.1f} (allow {allow_read_reg})"
            )
        return content1, title1, meta1, {
            "preflight": rep1,
            "finalizer": {
                "accepted": False,
                "report": fin_report,
                "metrics": {"seo": seo2, "readability": read2, **metrics2, "warnings": warn2},
            },
        }

    # Accept the finalizer pass
    if logger:
        logger.info(f"[finalizer] accepted: SEO {seo2:.1f}, READ {read2:.1f}")

    return c2, t2, m2, {
        "preflight": rep1,
        "finalizer": {
            "accepted": True,
            "report": fin_report,
            "metrics": {"seo": seo2, "readability": read2, **metrics2, "warnings": warn2},
        },
    }

