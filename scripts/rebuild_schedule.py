# scripts/rebuild_schedule.py
import argparse
import json
import random
import re
import sys
from pathlib import Path
from datetime import date, timedelta

TOPICS_FILE_DEFAULT = Path("config/topic_pools.json")
SETTINGS_FILE = Path("config/settings.json")
OUT_DEFAULT = Path("config/publishing_schedule.json")


def _norm_title(s: str) -> str:
    """Normalize a title for dedupe/lookup."""
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def _load_json(path: Path):
    text = path.read_text(encoding="utf-8")
    # tolerate // and /* */ comments
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    text = re.sub(r"^\s*//.*$", "", text, flags=re.M)
    return json.loads(text)


def _read_articles_per_day() -> int:
    try:
        if SETTINGS_FILE.exists():
            cfg = _load_json(SETTINGS_FILE)
            return int(cfg.get("content_strategy", {}).get("articles_per_day", 5))
    except Exception:
        pass
    return 5


def _flatten_topic_pools(pools_obj):
    """
    Returns a list of (title, keyphrase, niche) tuples from two supported shapes:

    1) {"pools":[{"niche":"…","topics":[{"title":"…","keyphrase":"…"} or "string"]}, …]}
    2) {"Niche A": [{"title":"…","keyphrase":"…"} or "string"], "Niche B": [...]}
    """
    flat = []

    # shape 1: {"pools":[...]}
    if isinstance(pools_obj, dict) and isinstance(pools_obj.get("pools"), list):
        for pool in pools_obj.get("pools", []):
            niche = (pool.get("niche") or "").strip()
            topics = pool.get("topics") or []
            for t in topics:
                if isinstance(t, str):
                    title = t.strip()
                    keyphrase = title
                elif isinstance(t, dict):
                    title = (t.get("title") or "").strip()
                    keyphrase = (t.get("keyphrase") or title).strip()
                else:
                    continue
                if title:
                    flat.append((title, keyphrase, niche))
        return flat

    # shape 2: { "Niche": [ "topic" | {title,keyphrase}, ... ], ... }
    if isinstance(pools_obj, dict):
        for niche, topics in pools_obj.items():
            if niche == "pools":  # already handled above; ignore here
                continue
            if not isinstance(topics, list):
                continue
            n = (niche or "").strip()
            for t in topics:
                if isinstance(t, str):
                    title = t.strip()
                    keyphrase = title
                elif isinstance(t, dict):
                    title = (t.get("title") or "").strip()
                    keyphrase = (t.get("keyphrase") or title).strip()
                else:
                    continue
                if title:
                    flat.append((title, keyphrase, n))
        return flat

    # unknown shape
    raise ValueError("Unsupported topic_pools.json shape.")


def _preserve_published_flags(out_items, existing_items):
    """Carry over published=True from an existing schedule by normalized title."""
    if not existing_items:
        return out_items

    existing_published = {_norm_title(i.get("topic", "")): bool(i.get("published")) for i in existing_items}
    for item in out_items:
        nt = _norm_title(item.get("topic", ""))
        if existing_published.get(nt, False):
            item["published"] = True
    return out_items


def build_schedule(
    start: date,
    per_day: int,
    topics_path: Path,
    out_path: Path,
    shuffle: bool = False,
    preserve_existing: bool = True,
):
    if not topics_path.exists():
        print(f"ERROR: topics file not found: {topics_path}", file=sys.stderr)
        sys.exit(2)

    pools_obj = _load_json(topics_path)
    items = _flatten_topic_pools(pools_obj)

    if not items:
        print("ERROR: No topics found in topic_pools.json.", file=sys.stderr)
        sys.exit(2)

    # de-dupe by normalized title, keep first occurrence
    seen = set()
    deduped = []
    for title, kp, niche in items:
        nt = _norm_title(title)
        if nt in seen:
            continue
        seen.add(nt)
        deduped.append((title, kp, niche))

    if shuffle:
        random.shuffle(deduped)

    # load existing schedule to optionally preserve published flags
    existing = []
    if preserve_existing and out_path.exists():
        try:
            existing = _load_json(out_path)
            if not isinstance(existing, list):
                existing = []
        except Exception:
            existing = []

    # lay out days
    d = start
    out = []
    for i, (title, kp, niche) in enumerate(deduped):
        if i and i % per_day == 0:
            d += timedelta(days=1)
        out.append(
            {
                "date": d.isoformat(),
                "topic": title,
                "keyphrase": kp,
                "niche": niche,
                "published": False,
            }
        )

    if preserve_existing and existing:
        out = _preserve_published_flags(out, existing)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    # summary
    total_days = (len(out) + per_day - 1) // per_day
    print(f"Wrote {len(out)} scheduled items across ~{total_days} day(s) to {out_path}")
    print(f"Start date: {start.isoformat()} | Articles/day: {per_day} | Shuffle: {shuffle} | Preserve flags: {preserve_existing}")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="Rebuild config/publishing_schedule.json from config/topic_pools.json"
    )
    ap.add_argument(
        "--start",
        help="Start date YYYY-MM-DD (default 2025-08-22)",
        default="2025-08-22",
    )
    ap.add_argument(
        "--per-day",
        type=int,
        help="Articles per day (default from settings.json content_strategy.articles_per_day, else 5)",
        default=None,
    )
    ap.add_argument(
        "--from",
        dest="topics_path",
        type=Path,
        default=TOPICS_FILE_DEFAULT,
        help="Path to topic_pools.json",
    )
    ap.add_argument(
        "--out",
        dest="out_path",
        type=Path,
        default=OUT_DEFAULT,
        help="Where to write publishing_schedule.json",
    )
    ap.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle topic order before scheduling",
    )
    ap.add_argument(
        "--no-preserve",
        dest="preserve",
        action="store_false",
        help="Do not carry over published=True flags from existing schedule",
    )
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    try:
        y, m, d = map(int, args.start.split("-"))
        start = date(y, m, d)
    except Exception:
        print("ERROR: --start must be YYYY-MM-DD", file=sys.stderr)
        sys.exit(2)

    per_day = args.per_day if args.per_day is not None else _read_articles_per_day()
    if per_day <= 0:
        print("ERROR: --per-day must be >= 1", file=sys.stderr)
        sys.exit(2)

    build_schedule(
        start=start,
        per_day=per_day,
        topics_path=args.topics_path,
        out_path=args.out_path,
        shuffle=args.shuffle,
        preserve_existing=args.preserve,
    )


if __name__ == "__main__":
    main()
