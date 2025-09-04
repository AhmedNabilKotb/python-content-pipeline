import os
import logging
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------
LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "code_cleaner.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
@dataclass
class CodeCleanerConfig:
    backup_dir_name: str = "backups/cleaned_articles"
    max_backups_per_article: int = 3

    # Code fence normalization
    normalize_language_tags: bool = True
    retag_untagged_as_python: bool = True
    code_block_tag: str = "python"  # final tag for pythonic blocks
    protect_special_blocks: List[str] = field(default_factory=lambda: ["mermaid", "math", "latex", "plantuml", "html"])

    # Heuristics & content tweaks
    strip_trailing_whitespace: bool = True
    ensure_final_newline: bool = True
    max_consecutive_blank_lines_in_code: int = 2
    strip_repl_prompts: bool = False  # remove ">>> " / "... " in python blocks
    expand_tabs_to_spaces: Optional[int] = None  # e.g., 4 to expand, None to keep

    # Markdown spacing fixes
    ensure_blank_line_before_fences: bool = True
    ensure_blank_line_after_fences: bool = True  # NEW

    # Line ending normalization
    normalize_newlines: bool = True  # NEW (CRLF/CR → LF)

    # Optionally force a single fence style (e.g., "```"); None keeps original
    canonical_fence: Optional[str] = None  # NEW

    # Aliases that should be normalized (info-string first token only)
    language_aliases: Dict[str, str] = field(
        default_factory=lambda: {
            # python
            "py": "python",
            "python3": "python",
            "py3": "python",
            "pycon": "python",
            # shells
            "sh": "bash",
            "shell": "bash",
            "console": "bash",
            "zsh": "bash",
            "shell-session": "bash",
            # data/config
            "yml": "yaml",
            "json5": "json",
            "toml": "toml",
            "ini": "ini",
            # web
            "js": "javascript",
            "ts": "typescript",
            # windows
            "ps1": "powershell",
            "powershell": "powershell",
        }
    )


# ---------------------------------------------------------------------
# Cleaner
# ---------------------------------------------------------------------
class CodeCleaner:
    """
    Cleans article markdown:

      • Parses fenced code blocks robustly (``` or ~~~), preserving unknown attrs.
      • Normalizes language tags (keeps non-Python tags; respects special blocks).
      • Tags untagged blocks as Python only if they *look* like Python (heuristic+AST).
      • Optional REPL prompt stripping and tab expansion inside code.
      • Squashes huge blank runs, trims trailing whitespace, ensures final newline.
      • Ensures a blank line *before* and (optionally) *after* fenced blocks.
      • Optional newline normalization (CRLF/CR → LF).
      • Optional canonical fence marker.
      • Safe timestamped backups + retention.
      • Emits a small per-file report to logs (blocks scanned/retagged, etc.).
    """

    # Match either ``` or ~~~ fences with an info string (lang + optional attrs)
    # Accept *any* info-string content up to the newline.
    _FENCE_RE = re.compile(
        r"(?P<fence>```|~~~)(?P<info>[^\n]*)\n(?P<code>.*?)(?:\n)?(?P=fence)",
        flags=re.DOTALL,
    )

    def __init__(self, app_settings: Dict, project_base_dir: Path):
        self.project_base_dir = Path(project_base_dir).resolve()
        self.config = CodeCleanerConfig()

        cleaning_settings = (app_settings or {}).get("code_cleaning", {})
        # Scalars
        self.config.backup_dir_name = cleaning_settings.get("backup_dir", self.config.backup_dir_name)
        self.config.max_backups_per_article = int(
            cleaning_settings.get("max_backups_per_article", self.config.max_backups_per_article)
        )
        self.config.code_block_tag = cleaning_settings.get("code_block_tag", self.config.code_block_tag)
        self.config.normalize_language_tags = bool(
            cleaning_settings.get("normalize_language_tags", self.config.normalize_language_tags)
        )
        self.config.retag_untagged_as_python = bool(
            cleaning_settings.get("retag_untagged_as_python", self.config.retag_untagged_as_python)
        )
        self.config.strip_trailing_whitespace = bool(
            cleaning_settings.get("strip_trailing_whitespace", self.config.strip_trailing_whitespace)
        )
        self.config.ensure_final_newline = bool(
            cleaning_settings.get("ensure_final_newline", self.config.ensure_final_newline)
        )
        self.config.max_consecutive_blank_lines_in_code = int(
            cleaning_settings.get("max_consecutive_blank_lines_in_code", self.config.max_consecutive_blank_lines_in_code)
        )
        self.config.strip_repl_prompts = bool(cleaning_settings.get("strip_repl_prompts", self.config.strip_repl_prompts))
        self.config.expand_tabs_to_spaces = cleaning_settings.get("expand_tabs_to_spaces", self.config.expand_tabs_to_spaces)
        self.config.ensure_blank_line_before_fences = bool(
            cleaning_settings.get("ensure_blank_line_before_fences", self.config.ensure_blank_line_before_fences)
        )
        self.config.ensure_blank_line_after_fences = bool(
            cleaning_settings.get("ensure_blank_line_after_fences", self.config.ensure_blank_line_after_fences)
        )
        self.config.normalize_newlines = bool(
            cleaning_settings.get("normalize_newlines", self.config.normalize_newlines)
        )
        self.config.canonical_fence = cleaning_settings.get("canonical_fence", self.config.canonical_fence)

        # Lists / dicts
        if isinstance(cleaning_settings.get("language_aliases"), dict):
            self.config.language_aliases.update(cleaning_settings["language_aliases"])
        if isinstance(cleaning_settings.get("protect_special_blocks"), list):
            self.config.protect_special_blocks = [str(x).lower() for x in cleaning_settings["protect_special_blocks"]]

        self.backup_full_path = self.project_base_dir / self.config.backup_dir_name
        self._setup_directories()

    # ---------------------- setup ----------------------
    def _setup_directories(self) -> None:
        self.backup_full_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"CodeCleaner backup directory ensured: {self.backup_full_path}")

    # ---------------------- heuristics -----------------
    @staticmethod
    def _looks_like_python(code: str) -> bool:
        """
        Conservative: quick keyword scan, then attempt ast.parse when feasible.
        """
        c = (code or "").strip()
        if not c:
            return False

        # Cheap signals
        kw_hits = 0
        patterns = [
            r"^\s*def\s+\w+\(",
            r"^\s*class\s+\w+\s*(\(|:)",
            r"^\s*if\s+__name__\s*==\s*['\"]__main__['\"]\s*:",
            r"^\s*(from\s+\w[\w\.]*\s+import\s+|import\s+\w+)",
            r"^\s*@\w+",
            r"\basync\s+def\b|\bawait\b",
            r"\bwith\s+\w+",
            r"\bprint\s*\(",
        ]
        for pat in patterns:
            if re.search(pat, c, flags=re.MULTILINE):
                kw_hits += 1
                if kw_hits >= 2:
                    return True

        # Try AST (best-effort; tolerate SyntaxError for snippets)
        try:
            if len(c) < 10000:
                import ast
                ast.parse(c)
                return True
        except Exception:
            pass
        return False

    @staticmethod
    def _looks_like_json(code: str) -> bool:
        c = (code or "").strip()
        if not (c.startswith("{") or c.startswith("[")):
            return False
        # Quick sanity
        return (c.count("{") + c.count("[")) >= 1

    @staticmethod
    def _looks_like_yaml(code: str) -> bool:
        c = (code or "").strip()
        return bool(re.search(r"^\s*\w+\s*:\s*.+", c, flags=re.MULTILINE))

    @staticmethod
    def _looks_like_bash(code: str) -> bool:
        c = (code or "").strip()
        if c.startswith("#!/bin/bash") or c.startswith("#!/usr/bin/env bash"):
            return True
        return bool(re.search(r"^\s*(\$ |sudo\s+|apt-get\s+|pip\s+|python3?\s+|export\s+\w+=)", c, re.MULTILINE))

    @staticmethod
    def _looks_like_html(code: str) -> bool:
        c = (code or "").strip().lower()
        return c.startswith("<!doctype") or c.startswith("<html") or "<div" in c or "<script" in c

    @staticmethod
    def _looks_like_sql(code: str) -> bool:
        c = (code or "").strip().lower()
        return bool(re.search(r"\b(select|insert|update|delete|create|drop|alter)\b", c))

    @staticmethod
    def _looks_like_javascript(code: str) -> bool:
        c = (code or "").strip()
        return bool(
            re.search(r"\b(function\s+\w+\(|=>|const\s+\w+\s*=|let\s+\w+\s*=|import\s+.+from\s+['\"])", c)
        )

    # ---------------------- helpers -------------------
    def _strip_repl_prompts(self, code: str) -> str:
        if not self.config.strip_repl_prompts:
            return code
        out: List[str] = []
        for ln in code.splitlines():
            s = ln.lstrip()
            if s.startswith(">>> "):
                out.append(s[4:])
            elif s.startswith("... "):
                out.append(s[4:])
            else:
                out.append(ln)
        return "\n".join(out)

    def _squash_blank_runs(self, code: str) -> str:
        """
        Ensure at most `max_consecutive_blank_lines_in_code` blank lines in a row.
        """
        max_run = max(0, int(self.config.max_consecutive_blank_lines_in_code))
        if max_run == 0:
            # remove all blank-only lines runs altogether
            return re.sub(r"(?:\n[ \t]*){2,}", "\n", code)
        # Replace any run longer than max_run with exactly max_run newlines
        pattern = r"(?:\n[ \t]*){" + str(max_run + 1) + r",}"
        return re.sub(pattern, "\n" * max_run, code)

    def _apply_within_code(self, code: str) -> str:
        if self.config.expand_tabs_to_spaces:
            try:
                code = code.expandtabs(int(self.config.expand_tabs_to_spaces))
            except Exception:
                pass
        code = code.strip("\n")
        code = self._strip_repl_prompts(code)
        code = self._squash_blank_runs(code)
        if self.config.strip_trailing_whitespace:
            code = re.sub(r"[ \t]+$", "", code, flags=re.MULTILINE)
        return code

    @staticmethod
    def _split_info_string(info: str) -> Tuple[str, str]:
        """
        Split an info string into (language_token, rest_attrs).
        Accepts forms like: "python", "python title='x.py' linenums", "", "bash {cmd}"
        """
        s = (info or "").strip()
        if not s:
            return "", ""
        m = re.match(r"^([A-Za-z0-9_+\-\.#]*)(.*)$", s)
        if not m:
            return s.lower(), ""
        lang = (m.group(1) or "").strip().lower()
        attrs = (m.group(2) or "").rstrip()
        return lang, attrs

    def _normalize_lang(self, lang: str) -> str:
        if not lang:
            return ""
        mapped = self.config.language_aliases.get(lang.lower(), lang.lower())
        if mapped == "python":
            return self.config.code_block_tag
        return mapped

    # ---------------------- content ops ----------------
    def _has_blank_line_before(self, text: str, idx: int) -> bool:
        """Return True if there's a blank line immediately before idx."""
        start = max(0, idx - 512)
        return re.search(r"\n\s*\n$", text[start:idx]) is not None

    def _ensure_blank_line_after_fences(self, text: str) -> str:
        """
        Ensure exactly one blank line after every fenced code block (unless EOF).
        """
        if not self.config.ensure_blank_line_after_fences:
            return text

        new_parts: List[str] = []
        pos = 0
        for m in self._FENCE_RE.finditer(text):
            end = m.end()
            new_parts.append(text[pos:end])
            # Count immediate newlines after the fence
            i = end
            nl_count = 0
            while i < len(text) and text[i] == "\n":
                nl_count += 1
                i += 1
            if i >= len(text):
                pos = i
                continue
            if nl_count == 0:
                new_parts.append("\n\n")  # add one blank line
            elif nl_count == 1:
                new_parts.append("\n")    # make it a blank line (total 2)
            pos = end + nl_count
        new_parts.append(text[pos:])
        return "".join(new_parts)

    def _fix_code_blocks(self, content: str, report: Dict[str, int]) -> str:
        """
        Normalize fenced blocks with careful language handling and
        (optionally) ensure a blank line before each fence.
        """
        special = set(self.config.protect_special_blocks)
        original = content  # for before-context checks inside the replacer

        def repl(m: re.Match) -> str:
            fence = m.group("fence")
            info_raw = m.group("info") or ""
            code = m.group("code") or ""

            # Optionally enforce a canonical fence marker
            fence_out = self.config.canonical_fence if self.config.canonical_fence else fence

            lang_raw, attrs = self._split_info_string(info_raw)
            lang_norm = lang_raw.lower()

            # Protect special blocks (mermaid/math/etc.)
            if lang_norm in special:
                return m.group(0)

            # Normalize code inner text
            code = self._apply_within_code(code)

            # Decide language—preserve explicit non-empty lang unless aliasable
            out_lang = lang_raw
            changed_lang = False

            if self.config.normalize_language_tags:
                if lang_raw:
                    mapped = self._normalize_lang(lang_raw)
                    if mapped != lang_raw:
                        out_lang = mapped
                        changed_lang = True
                else:
                    if self.config.retag_untagged_as_python:
                        if self._looks_like_python(code):
                            out_lang = self.config.code_block_tag
                            changed_lang = True
                        elif self._looks_like_bash(code):
                            out_lang = "bash"
                            changed_lang = True
                        elif self._looks_like_json(code):
                            out_lang = "json"
                            changed_lang = True
                        elif self._looks_like_yaml(code):
                            out_lang = "yaml"
                            changed_lang = True
                        elif self._looks_like_html(code):
                            out_lang = "html"
                            changed_lang = True
                        elif self._looks_like_sql(code):
                            out_lang = "sql"
                            changed_lang = True
                        elif self._looks_like_javascript(code):
                            out_lang = "javascript"
                            changed_lang = True

            # Book-keeping for report
            report["blocks_scanned"] += 1
            if changed_lang:
                report["blocks_retagged"] += 1
            if self.config.strip_repl_prompts and (">>> " in m.group("code") or "... " in m.group("code")):
                report["repl_prompts_stripped"] += 1

            # Rebuild info string (preserve attrs exactly)
            info_out = out_lang.strip()
            if attrs:
                info_out = (info_out + " " + attrs).strip()

            # Ensure blank line BEFORE fence (for proper Markdown rendering)
            prefix = ""
            if self.config.ensure_blank_line_before_fences and m.start() > 0:
                if not self._has_blank_line_before(original, m.start()):
                    prefix = "\n\n"

            # Rebuild full fence (keep a single newline after fence; body-level hygiene runs later)
            rebuilt = f"{fence_out}{info_out}\n{code}\n{fence_out}"
            return prefix + rebuilt

        out = self._FENCE_RE.sub(repl, content)
        # Optionally ensure blank line AFTER fences in a second pass
        out = self._ensure_blank_line_after_fences(out)
        return out

    def _normalize_newlines_text(self, text: str) -> str:
        if not self.config.normalize_newlines:
            return text
        # Convert CRLF/CR to LF
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        return text

    def _post_text_hygiene(self, text: str) -> str:
        """File-level hygiene outside fences."""
        if self.config.strip_trailing_whitespace:
            text = re.sub(r"[ \t]+$", "", text, flags=re.MULTILINE)
        # Collapse multiple trailing newlines to a single newline
        text = re.sub(r"\n{3,}\Z", "\n\n", text)
        if self.config.ensure_final_newline and not text.endswith("\n"):
            text += "\n"
        return text

    def clean_article_content(self, article_content: str, article_slug: str) -> str:
        """
        Clean a markdown string and return the cleaned content.
        """
        if not article_content:
            logger.warning("No content provided for cleaning.")
            return ""

        report = {"blocks_scanned": 0, "blocks_retagged": 0, "repl_prompts_stripped": 0}

        # Normalize newlines first for predictable regex behavior
        text = self._normalize_newlines_text(article_content)

        cleaned = self._fix_code_blocks(text, report)
        cleaned = self._post_text_hygiene(cleaned)

        if cleaned != article_content:
            logger.info(
                "CodeCleaner: '%s' cleaned (blocks=%d, retagged=%d, repl_stripped=%d).",
                article_slug, report["blocks_scanned"], report["blocks_retagged"], report["repl_prompts_stripped"]
            )
        else:
            logger.info("CodeCleaner: '%s' no changes needed (blocks=%d).", article_slug, report["blocks_scanned"])

        return cleaned

    # ---------------------- file ops -------------------
    def _create_backup(self, filepath: Path) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = self.backup_full_path / f"{filepath.stem}_{timestamp}{filepath.suffix}"
        backup_path.write_text(Path(filepath).read_text(encoding="utf-8"), encoding="utf-8")
        logger.info(f"Created backup: {backup_path}")
        return backup_path

    def _cleanup_old_backups(self) -> None:
        backups = sorted(
            self.backup_full_path.glob(f"*.*"),
            key=os.path.getmtime,
            reverse=True,
        )
        # Retain the last N backups per original filename stem
        by_stem: Dict[str, List[Path]] = {}
        for p in backups:
            stem = p.name.split("_")[0]
            by_stem.setdefault(stem, []).append(p)
        total_removed = 0
        for stem, items in by_stem.items():
            for old in items[self.config.max_backups_per_article:]:
                try:
                    old.unlink()
                    total_removed += 1
                except Exception as e:
                    logger.error(f"Failed to remove backup {old}: {e}")
        if total_removed:
            logger.info("Removed %d old backup(s).", total_removed)

    def clean_article_file(self, filepath: Path) -> bool:
        """
        Clean a file in place with backup + retention.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            logger.error(f"File does not exist: {filepath}")
            return False

        try:
            self._create_backup(filepath)
            original = filepath.read_text(encoding="utf-8")
            cleaned = self.clean_article_content(original, filepath.stem)
            if cleaned != original:
                filepath.write_text(cleaned, encoding="utf-8")
                logger.info(f"Wrote cleaned content: {filepath}")
            self._cleanup_old_backups()
            return True
        except Exception as e:
            logger.error(f"Failed to clean file {filepath}: {e}", exc_info=True)
            return False


# ---------------------- In-memory helper (NEW) -------------------

def pre_publish_sanitize(content: str, app_settings: Optional[Dict] = None, slug: str = "pre_qc") -> str:
    """
    In-memory sanitization for pipeline usage (no file I/O, no backups).

    You can override cleaner behavior at runtime by providing:
      app_settings = {
          "code_cleaning_runtime": {
              # Any of these optional keys:
              "strip_repl_prompts": True,
              "expand_tabs_to_spaces": 4,
              "ensure_blank_line_before_fences": True,
              "ensure_blank_line_after_fences": True,
              "normalize_newlines": True,
              "normalize_language_tags": True,
              "retag_untagged_as_python": True,
              "canonical_fence": "```",  # or "~~~"
          }
      }
    """
    settings = app_settings or {}
    cleaner = CodeCleaner(settings, project_base_dir=Path("."))

    # Apply runtime overrides (if present) without touching file-oriented knobs
    overrides = (settings.get("code_cleaning_runtime") or {})
    if overrides:
        cfg = cleaner.config
        for key, val in overrides.items():
            if hasattr(cfg, key):
                setattr(cfg, key, val)

    try:
        return cleaner.clean_article_content(content, slug)
    except Exception as e:
        logger.error(f"pre_publish_sanitize failed: {e}", exc_info=True)
        # Fail-safe: return original content if anything goes wrong
        return content


__all__ = ["CodeCleaner", "CodeCleanerConfig", "pre_publish_sanitize"]
