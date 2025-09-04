# scripts/utils/jsonc.py
import json
import re
from pathlib import Path

# Strip // line comments and /* ... */ block comments
_LINE = re.compile(r"^\s*//.*$", re.MULTILINE)
_BLOCK = re.compile(r"/\*.*?\*/", re.DOTALL)

def load_jsonc(path: Path):
    """
    Read JSON that may contain // or /* */ comments.
    Returns a Python object like json.load(s) would.
    """
    text = Path(path).read_text(encoding="utf-8")
    text = _BLOCK.sub("", _LINE.sub("", text))
    return json.loads(text)

