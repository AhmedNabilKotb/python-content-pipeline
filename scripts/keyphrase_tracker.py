# keyphrase_tracker.py

import os
import json
import re
import time
import tempfile
from pathlib import Path
from typing import List, Set, Dict, Optional, Tuple
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

# --- Logging ---------------------------------------------------------------
LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Avoid duplicating handlers if the app configured logging already
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(LOG_DIR / "keyphrase_tracker.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
logger = logging.getLogger(__name__)

# --- Data ------------------------------------------------------------------

@dataclass
class KeyphraseMetadata:
    """Stores metadata for a single keyphrase."""
    phrase: str
    first_used_date: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))
    last_used_date: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))
    use_count: int = 1
    associated_slugs: List[str] = field(default_factory=list)
    # Optional enrichments:
    niches: List[str] = field(default_factory=list)
    titles: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "phrase": self.phrase,
            "first_used_date": self.first_used_date,
            "last_used_date": self.last_used_date,
            "use_count": self.use_count,
            "associated_slugs": self.associated_slugs,
            "niches": self.niches,
            "titles": self.titles,
        }

    @classmethod
    def from_dict(cls, data: Dict):
        return cls(
            phrase=data.get("phrase", ""),
            first_used_date=data.get("first_used_date", datetime.now(timezone.utc).isoformat(timespec="seconds")),
            last_used_date=data.get("last_used_date", datetime.now(timezone.utc).isoformat(timespec="seconds")),
            use_count=int(data.get("use_count", 1)),
            associated_slugs=list(data.get("associated_slugs", [])),
            niches=[str(n).lower() for n in list(data.get("niches", []))],
            titles=list(data.get("titles", [])),
        )

@dataclass
class KeyphraseConfig:
    storage_path: Path = Path("config/used_keyphrases.json")
    semantic_suffixes: List[str] = field(
        default_factory=lambda: [
            "guide", "tutorial", "in python", "tips",
            "how to", "best practices", "examples", "advanced"
        ]
    )
    max_retries: int = 1000
    fuzzy_match_threshold: float = 0.8
    max_entries: int = 5000
    max_keyphrase_length: int = 80
    reserved_phrases: List[str] = field(default_factory=lambda: [])
    ban_prefixes: List[str] = field(default_factory=lambda: ["http://", "https://", "www."])
    backup_keep: int = 5
    lock_timeout_s: float = 5.0
    lock_poll_interval_s: float = 0.05
    # Controls
    days_between_reuse_per_niche: int = 14
    prefer_year_suffix: bool = True
    dedupe_on_load: bool = True
    # Optional caps for lists to prevent unbounded growth
    max_titles_per_phrase: int = 50
    max_slugs_per_phrase: int = 50

# --- Tracker ---------------------------------------------------------------

class KeyphraseTracker:
    """
    Robust keyphrase de-duplication with:
      • Fuzzy similarity (Jaccard on lightly-stemmed tokens)
      • Semantic & numeric variant generation
      • Per-niche reuse cooldown (days_between_reuse_per_niche)
      • Atomic saves + exclusive lock + rotating backups
      • Capacity limits + utilities (stats, removal, reset)
      • Title-to-keyphrase helper for pipeline integration
    """

    def __init__(self, config: Optional[KeyphraseConfig] = None):
        self.config = config if config else KeyphraseConfig()
        # Allow env vars and ~ in storage_path (e.g., "${DATA_DIR}/used_keyphrases.json")
        self.config.storage_path = self._resolve_storage_path(self.config.storage_path)
        self.config.storage_path.parent.mkdir(parents=True, exist_ok=True)

        self._lock_path = self.config.storage_path.with_suffix(self.config.storage_path.suffix + ".lock")
        self._used_keyphrases: Dict[str, KeyphraseMetadata] = self._load_used_keyphrases()
        self._keyphrase_word_sets: Dict[str, Set[str]] = {
            kp_meta.phrase.lower(): self._get_word_set(kp_meta.phrase)
            for kp_meta in self._used_keyphrases.values()
        }
        logger.info(f"KeyphraseTracker initialized. Loaded {len(self._used_keyphrases)} keyphrases.")

    # --- Locking -------------------------------------------------------------

    def _acquire_lock(self) -> bool:
        """
        Exclusive lock via atomic create; safe across processes.
        Removes a stale lock if older than ~10× timeout.
        """
        deadline = time.time() + float(self.config.lock_timeout_s)
        while time.time() < deadline:
            try:
                fd = os.open(str(self._lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                with os.fdopen(fd, "w", encoding="utf-8") as h:
                    h.write(f"pid={os.getpid()} ts={datetime.now(timezone.utc).isoformat(timespec='seconds')}")
                return True
            except FileExistsError:
                # stale lock guard
                try:
                    age = time.time() - self._lock_path.stat().st_mtime
                    if age > max(1.0, self.config.lock_timeout_s) * 10:
                        logger.warning("Stale keyphrase lock detected; removing.")
                        self._lock_path.unlink(missing_ok=True)
                        continue
                except Exception:
                    pass
                time.sleep(self.config.lock_poll_interval_s)
            except Exception as e:
                logger.error(f"Unexpected lock error: {e}")
                time.sleep(self.config.lock_poll_interval_s)
        logger.warning(f"Lock timeout waiting for {self._lock_path}")
        return False

    def _release_lock(self) -> None:
        try:
            if self._lock_path.exists():
                self._lock_path.unlink(missing_ok=True)
        except Exception:
            pass

    # --- Internals ----------------------------------------------------------

    @staticmethod
    def _now_utc() -> datetime:
        return datetime.now(timezone.utc)

    @staticmethod
    def _resolve_storage_path(p: Path) -> Path:
        # Expand ${VAR}, $VAR, and ~ anywhere in the path
        s = os.path.expandvars(os.path.expanduser(str(p)))
        return Path(s)

    def _normalize(self, phrase: str) -> str:
        p = (phrase or "").strip().lower()
        # collapse separators and trim
        p = re.sub(r"[\s\-–—:·|]+", " ", p)
        p = re.sub(r"\s+", " ", p).strip()
        # trim repeated "in python"
        p = re.sub(r"\b(in python)\b(?=.*\1)", "", p).strip()
        p = p[: self.config.max_keyphrase_length].strip()
        return p

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\b[a-z0-9]+\b", (text or "").lower())

    def _light_stem(self, w: str) -> str:
        # very naive: plurals and common endings
        if len(w) > 4 and w.endswith("ies"):
            return w[:-3] + "y"
        if len(w) > 3 and w.endswith("es"):
            return w[:-2]
        if len(w) > 3 and w.endswith("s"):
            return w[:-1]
        return w

    def _get_word_set(self, phrase: str) -> Set[str]:
        toks = [self._light_stem(t) for t in self._tokenize(phrase)]
        return set(toks)

    def _calculate_jaccard_similarity(self, set1: Set[str], set2: Set[str]) -> float:
        if not set1 and not set2:
            return 1.0
        union = set1 | set2
        return (len(set1 & set2) / len(union)) if union else 0.0

    def _is_reserved_or_banned(self, phrase: str) -> bool:
        p = phrase.strip().lower()
        if not p:
            return True
        for pre in self.config.ban_prefixes:
            if p.startswith(pre):
                return True
        for bad in self.config.reserved_phrases:
            if p == bad.strip().lower():
                return True
        return False

    @staticmethod
    def _parse_iso(s: str) -> Optional[datetime]:
        if not s:
            return None
        try:
            # Accept "....Z" (Zulu) and offsets
            s2 = s.replace("Z", "+00:00")
            dt = datetime.fromisoformat(s2)
            # If naive, assume UTC
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except Exception:
            return None

    def _recently_used_in_niche(self, meta: KeyphraseMetadata, niche: Optional[str]) -> bool:
        """Respect per-niche cooldown if a niche is provided."""
        if not niche or self.config.days_between_reuse_per_niche <= 0:
            return False
        niche_l = niche.lower()
        if meta.niches and niche_l in (n.lower() for n in meta.niches):
            last = self._parse_iso(meta.last_used_date)
            if last:
                return (self._now_utc() - last) < timedelta(days=self.config.days_between_reuse_per_niche)
        return False

    def _jsonc_load(self, p: Path) -> Optional[object]:
        """
        Tolerant loader for JSONC-like files:
          • strips // line comments and /* block */ comments
          • removes trailing commas (best-effort)
        """
        try:
            raw = p.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        except Exception as e:
            logger.error(f"Failed reading keyphrase file: {e}")
            return None
        # Remove /* ... */ block comments
        raw = re.sub(r"/\*.*?\*/", "", raw, flags=re.S)
        # Remove // ... line comments
        raw = re.sub(r"(^|\s)//.*?$", r"\1", raw, flags=re.M)
        # Remove trailing commas (objects & arrays)
        raw = re.sub(r",\s*([}\]])", r"\1", raw)
        raw = raw.strip()
        if not raw:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            logger.warning(f"Error loading keyphrases (JSONC parse): {e}. Starting with empty storage.")
            return None

    def _load_used_keyphrases(self) -> Dict[str, KeyphraseMetadata]:
        loaded: Dict[str, KeyphraseMetadata] = {}
        p = self.config.storage_path
        try:
            data = None
            if p.exists():
                data = self._jsonc_load(p)

            if not data:
                msg = "Loaded %d keyphrases."
                logger.info(msg, 0)
                return loaded

            # Accept both list (current) and dict (legacy) formats
            if isinstance(data, list):
                items: List[Dict] = data
            elif isinstance(data, dict):
                items = []
                for k, v in data.items():
                    v = v or {}
                    v.setdefault("phrase", k)
                    items.append(v)
            else:
                raise ValueError("Unexpected keyphrase storage format")

            for item in items:
                kp = KeyphraseMetadata.from_dict(item)
                n = self._normalize(kp.phrase)
                if not n:
                    continue
                kp.phrase = n
                # dedupe associated arrays cheaply
                kp.associated_slugs = list(dict.fromkeys([s for s in kp.associated_slugs if s]))[: self.config.max_slugs_per_phrase]
                kp.niches = list(dict.fromkeys([s.lower() for s in kp.niches if s]))
                kp.titles = list(dict.fromkeys([s for s in kp.titles if s]))[: self.config.max_titles_per_phrase]
                # keep newest if duplicates appear in file
                if n in loaded:
                    a = loaded[n]
                    if (kp.last_used_date or "") > (a.last_used_date or ""):
                        loaded[n] = kp
                else:
                    loaded[n] = kp

            msg = "Loaded %d keyphrases (deduped by normalized phrase)." if self.config.dedupe_on_load else "Loaded %d keyphrases."
            logger.info(msg, len(loaded))
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Error loading keyphrases: {e}. Starting with empty storage.")
        except Exception as e:
            logger.error(f"Unexpected error loading keyphrases: {e}")
        return loaded

    def _rotate_backups(self) -> None:
        """Keep small rolling backups of the JSON file."""
        base = self.config.storage_path
        if not base.exists():
            return
        ts = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        backup = base.with_suffix(base.suffix + f".bak.{ts}")
        try:
            backup.write_text(base.read_text(encoding="utf-8"), encoding="utf-8")
        except Exception:
            return
        # prune old backups
        try:
            backups = sorted(base.parent.glob(base.name + ".bak.*"), key=lambda p: p.name, reverse=True)
            for old in backups[self.config.backup_keep:]:
                old.unlink(missing_ok=True)
        except Exception:
            pass

    def _atomic_write_json(self, data: List[Dict]) -> bool:
        """Same-dir temp + fsync for robustness across crashes."""
        target = self.config.storage_path
        tmp_fd, tmp_path = tempfile.mkstemp(prefix=target.name + ".", dir=str(target.parent))
        ok = False
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, target)
            ok = True
        except Exception as e:
            logger.error(f"Atomic write failed: {e}")
        finally:
            try:
                if not ok and os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass
        return ok

    def _save_keyphrases(self) -> bool:
        if not self._acquire_lock():
            logger.error("Could not acquire keyphrase storage lock.")
            return False
        try:
            # capacity control: keep the newest N by last_used_date
            if len(self._used_keyphrases) > self.config.max_entries:
                all_items = list(self._used_keyphrases.values())
                all_items.sort(key=lambda m: m.last_used_date, reverse=True)
                trimmed = {m.phrase: m for m in all_items[: self.config.max_entries]}
                dropped = len(self._used_keyphrases) - len(trimmed)
                self._used_keyphrases = trimmed
                if dropped > 0:
                    logger.info(f"Trimmed keyphrase store by {dropped} entries (capacity={self.config.max_entries}).")

            data_to_save = [kp.to_dict() for kp in self._used_keyphrases.values()]
            # rotate before write
            self._rotate_backups()
            ok = self._atomic_write_json(data_to_save)
            if ok:
                logger.info(f"Saved {len(self._used_keyphrases)} keyphrases.")
            return ok
        except Exception as e:
            logger.error(f"Failed to save keyphrases: {e}")
            return False
        finally:
            self._release_lock()

    # --- Public API --------------------------------------------------------

    def is_keyphrase_used(
        self,
        keyphrase: str,
        check_fuzzy: bool = False,
        niche: Optional[str] = None,
    ) -> bool:
        """
        Returns True if the keyphrase (or a fuzzy-similar one) should be considered "used".
        Semantics:
          • If a niche is provided, we only block reuse if it was used RECENTLY in the SAME niche (cooldown).
          • If no niche is provided, any prior use blocks reuse globally.
        """
        normalized = self._normalize(keyphrase)
        if not normalized:
            return False

        meta = self._used_keyphrases.get(normalized)
        if meta:
            if niche:
                # Only block if same-niche cooldown is still in effect
                return self._recently_used_in_niche(meta, niche)
            # Global check (no niche filtering) → used if seen at least once
            return True

        if check_fuzzy:
            current_words = self._get_word_set(normalized)
            if not current_words:
                return False
            for stored_phrase, stored_words in self._keyphrase_word_sets.items():
                sim = self._calculate_jaccard_similarity(current_words, stored_words)
                if sim >= self.config.fuzzy_match_threshold:
                    logger.info(f"Fuzzy match: '{keyphrase}' ~ '{stored_phrase}' (sim={sim:.2f})")
                    if niche:
                        meta2 = self._used_keyphrases.get(stored_phrase)
                        # Block only if recently used in THIS niche
                        if meta2 and self._recently_used_in_niche(meta2, niche):
                            return True
                        # else allow reuse (continue scanning to detect a conflicting same-niche hit)
                        continue
                    # No niche filtering → considered used
                    return True
        return False

    def save_keyphrase(
        self,
        keyphrase: str,
        associated_slug: Optional[str] = None,
        niche: Optional[str] = None,
        title: Optional[str] = None,
    ) -> bool:
        normalized = self._normalize(keyphrase)
        if not normalized or self._is_reserved_or_banned(normalized):
            logger.error("Attempted to save an empty/invalid/banned keyphrase.")
            return False

        now = self._now_utc().isoformat(timespec="seconds")

        if normalized in self._used_keyphrases:
            kp = self._used_keyphrases[normalized]
            kp.last_used_date = now
            kp.use_count += 1
            if associated_slug and associated_slug not in kp.associated_slugs:
                kp.associated_slugs.append(associated_slug)
                if len(kp.associated_slugs) > self.config.max_slugs_per_phrase:
                    kp.associated_slugs = kp.associated_slugs[-self.config.max_slugs_per_phrase :]
            if niche:
                niche_l = niche.lower()
                if niche_l not in kp.niches:
                    kp.niches.append(niche_l)
            if title and title not in kp.titles:
                kp.titles.append(title)
                if len(kp.titles) > self.config.max_titles_per_phrase:
                    kp.titles = kp.titles[-self.config.max_titles_per_phrase :]
            logger.info(f"Updated keyphrase: '{kp.phrase}' (use_count={kp.use_count})")
        else:
            niche_l = niche.lower() if niche else None
            kp = KeyphraseMetadata(
                phrase=normalized,
                associated_slugs=[associated_slug] if associated_slug else [],
                niches=[niche_l] if niche_l else [],
                titles=[title] if title else [],
            )
            self._used_keyphrases[normalized] = kp
            self._keyphrase_word_sets[normalized] = self._get_word_set(normalized)
            logger.info(f"Added new keyphrase: '{normalized}'")

        return self._save_keyphrases()

    # --- Generation helpers -------------------------------------------------

    def _semantic_variants(self, base: str) -> List[str]:
        """Generate semantic variants (bounded length)."""
        year = str(self._now_utc().year)
        base_short = base[: self.config.max_keyphrase_length].strip()
        variants = []
        for suf in self.config.semantic_suffixes:
            v = f"{base_short} {suf}".strip()
            variants.append(v[: self.config.max_keyphrase_length])
        if self.config.prefer_year_suffix:
            variants.append(f"{base_short} {year}")
            variants.append(f"{base_short} {year} guide")
        # dashed style (sometimes better for slugs/titles)
        variants.append(f"{base_short} - advanced")
        # unique preserving order
        return list(dict.fromkeys(variants))

    def _title_to_base_keyphrase(self, title: str) -> str:
        """
        Clean a title to a solid base keyphrase:
          - remove trailing punctuation/em-dashes/colons
          - collapse whitespace and separators
          - remove duplicate 'in python'
        """
        t = (title or "").strip()
        # drop bracketed notes at end: "Title (2025)" or "[Tutorial]"
        t = re.sub(r"\s*[\[(]\s*[^)\]]{1,30}\s*[\])]\s*$", "", t)
        # replace separators with space
        t = re.sub(r"[\-–—:·|/]+", " ", t)
        t = re.sub(r"\s+", " ", t).strip()
        # normalize
        t = self._normalize(t)
        return t

    def ensure_unique_for_title(
        self,
        title: str,
        *,
        niche: Optional[str] = None,
        content_slug: Optional[str] = None,
        allow_numeric: bool = True,
        fuzzy_check: bool = True,
    ) -> str:
        """
        Turn an article title into a unique, de-duplicated keyphrase.
        Respects per-niche cooldown and fuzzy similarity.
        """
        base = self._title_to_base_keyphrase(title)
        if not base or self._is_reserved_or_banned(base):
            raise ValueError("Invalid base keyphrase derived from title")

        # Try the base
        if not self.is_keyphrase_used(base, check_fuzzy=fuzzy_check, niche=niche):
            self.save_keyphrase(base, associated_slug=content_slug, niche=niche, title=title)
            logger.info(f"Using original keyphrase '{base}'.")
            return base

        # Semantic variants first
        for candidate in self._semantic_variants(base):
            if not self.is_keyphrase_used(candidate, check_fuzzy=fuzzy_check, niche=niche):
                self.save_keyphrase(candidate, associated_slug=content_slug, niche=niche, title=title)
                logger.info(f"Using semantic variant: '{candidate}'")
                return candidate

        # Then numeric variants
        if allow_numeric:
            for i in range(1, self.config.max_retries + 1):
                candidate = f"{base}-{i}"
                if not self.is_keyphrase_used(candidate, check_fuzzy=False, niche=niche):
                    self.save_keyphrase(candidate, associated_slug=content_slug, niche=niche, title=title)
                    logger.info(f"Using numerical variant: '{candidate}'")
                    return candidate

        logger.critical(
            f"Failed to generate unique keyphrase after {self.config.max_retries} attempts for '{base}' (niche={niche})."
        )
        raise RuntimeError(f"Exhausted all keyphrase variants for '{base}'")

    # --- Existing API kept for compatibility --------------------------------

    def get_unique_keyphrase(self, base_keyphrase: str, content_slug: Optional[str] = None) -> str:
        """Compatibility wrapper (no niche awareness)."""
        base = self._normalize(base_keyphrase)
        if not base or self._is_reserved_or_banned(base):
            raise ValueError("Invalid base keyphrase")

        if not self.is_keyphrase_used(base, check_fuzzy=True):
            self.save_keyphrase(base, associated_slug=content_slug)
            logger.info(f"Using original keyphrase '{base}'.")
            return base

        for candidate in self._semantic_variants(base):
            if not self.is_keyphrase_used(candidate, check_fuzzy=True):
                self.save_keyphrase(candidate, associated_slug=content_slug)
                logger.info(f"Using semantic variant: '{candidate}'")
                return candidate

        for i in range(1, self.config.max_retries + 1):
            candidate = f"{base}-{i}"
            if not self.is_keyphrase_used(candidate, check_fuzzy=False):
                self.save_keyphrase(candidate, associated_slug=content_slug)
                logger.info(f"Using numerical variant: '{candidate}'")
                return candidate

        logger.critical(f"Failed to generate unique keyphrase after {self.config.max_retries} attempts for '{base}'.")
        raise RuntimeError(f"Exhausted all keyphrase variants for '{base}'")

    def reset_storage(self) -> bool:
        self._used_keyphrases.clear()
        self._keyphrase_word_sets.clear()
        logger.info("Reset keyphrase storage.")
        return self._save_keyphrases()

    def remove_keyphrase(self, phrase: str) -> bool:
        """Remove a keyphrase explicitly."""
        n = self._normalize(phrase)
        if n in self._used_keyphrases:
            self._used_keyphrases.pop(n, None)
            self._keyphrase_word_sets.pop(n, None)
            logger.info(f"Removed keyphrase: '{n}'")
            return self._save_keyphrases()
        logger.info(f"Keyphrase not found for removal: '{n}'")
        return False

    def get_all_keyphrases(self) -> List[KeyphraseMetadata]:
        return list(self._used_keyphrases.values())

    def find_similar_keyphrases(self, query_phrase: str, threshold: Optional[float] = None) -> List[Tuple[str, float]]:
        query_words = self._get_word_set(self._normalize(query_phrase))
        if not query_words:
            return []

        results: List[Tuple[str, float]] = []
        effective_threshold = threshold if threshold is not None else self.config.fuzzy_match_threshold

        for stored_phrase, stored_words in self._keyphrase_word_sets.items():
            if stored_phrase == self._normalize(query_phrase):
                continue
            sim = self._calculate_jaccard_similarity(query_words, stored_words)
            if sim >= effective_threshold:
                results.append((self._used_keyphrases[stored_phrase].phrase, sim))

        return sorted(results, key=lambda x: x[1], reverse=True)

    def stats(self) -> Dict[str, int]:
        """Basic stats for monitoring."""
        total = len(self._used_keyphrases)
        uses = sum(k.use_count for k in self._used_keyphrases.values())
        # recent 7d usage
        seven_days_ago = self._now_utc() - timedelta(days=7)
        recent = 0
        for m in self._used_keyphrases.values():
            dt = self._parse_iso(m.last_used_date)
            if dt and dt >= seven_days_ago:
                recent += 1
        return {"phrases": total, "uses": uses, "recent_7d": recent}
