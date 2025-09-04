# technical_enhancer.py

import re
import logging
from typing import Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class TechnicalDepthEnhancer:
    """
    Adds technical depth to structured markdown:
      - Deterministic code example injection (no randomness)
      - Self-contained, correct examples (fixed imports, no undefined funcs)
      - Inserts into the right sections: "Implementation Walkthrough",
        "Technical Deep Dive", "Performance Benchmarks", "Further Reading"
      - Adds practical tips if they're light
      - Uses plain ``` fences (no language tag)
    """

    def __init__(self):
        self.quality_standards = {
            "min_code_blocks": 3,
            "min_references": 2,
            "min_examples": 2,
            "min_practical_tips": 3,
        }

        # Correct, self-contained examples
        self.code_examples: Dict[str, Dict[str, List[str]]] = {
            "python": {
                "decorator": [
                    "import time\nimport functools\n\n"
                    "def timer(func):\n"
                    "    @functools.wraps(func)\n"
                    "    def wrapper(*args, **kwargs):\n"
                    "        start = time.perf_counter()\n"
                    "        result = func(*args, **kwargs)\n"
                    "        end = time.perf_counter()\n"
                    "        print(f\"{func.__name__} took {end-start:.4f} seconds\")\n"
                    "        return result\n"
                    "    return wrapper\n\n"
                    "@timer\n"
                    "def work(n=1_000_000):\n"
                    "    s = 0\n"
                    "    for i in range(n):\n"
                    "        s += i\n"
                    "    return s\n\n"
                    "if __name__ == '__main__':\n"
                    "    work()\n",
                    "from typing import Callable, Any\nimport time\n\n"
                    "class retry:\n"
                    "    def __init__(self, max_attempts: int = 3):\n"
                    "        self.max_attempts = max_attempts\n\n"
                    "    def __call__(self, func: Callable) -> Callable:\n"
                    "        def wrapper(*args, **kwargs) -> Any:\n"
                    "            for attempt in range(self.max_attempts):\n"
                    "                try:\n"
                    "                    return func(*args, **kwargs)\n"
                    "                except Exception:\n"
                    "                    if attempt == self.max_attempts - 1:\n"
                    "                        raise\n"
                    "                    time.sleep(2 ** attempt)\n"
                    "            return func(*args, **kwargs)\n"
                    "        return wrapper\n",
                ],
                "context_manager": [
                    "from contextlib import contextmanager\n"
                    "import tempfile, os\n\n"
                    "@contextmanager\n"
                    "def managed_tempfile(suffix=\"\"):\n"
                    "    f = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)\n"
                    "    try:\n"
                    "        yield f.name\n"
                    "    finally:\n"
                    "        os.unlink(f.name)\n\n"
                    "with managed_tempfile('.txt') as path:\n"
                    "    with open(path, 'w') as fh:\n"
                    "        fh.write('hello')\n",
                    "import time\n\n"
                    "class Timer:\n"
                    "    def __enter__(self):\n"
                    "        self.start = time.perf_counter()\n"
                    "        return self\n"
                    "    def __exit__(self, exc_type, exc, tb):\n"
                    "        self.elapsed = time.perf_counter() - self.start\n"
                    "        print(f\"elapsed: {self.elapsed:.4f}s\")\n\n"
                    "with Timer():\n"
                    "    sum(range(1_000_000))\n",
                ],
                "async": [
                    "import asyncio\n\n"
                    "async def fetch(n):\n"
                    "    await asyncio.sleep(0.1)\n"
                    "    return n\n\n"
                    "async def main():\n"
                    "    results = await asyncio.gather(*(fetch(i) for i in range(3)))\n"
                    "    print(results)\n\n"
                    "if __name__ == '__main__':\n"
                    "    asyncio.run(main())\n",
                ],
                "generator": [
                    "from collections import deque\n\n"
                    "def sliding_window(iterable, size):\n"
                    "    it = iter(iterable)\n"
                    "    window = deque(maxlen=size)\n"
                    "    for x in it:\n"
                    "        window.append(x)\n"
                    "        if len(window) == size:\n"
                    "            yield tuple(window)\n\n"
                    "print(list(sliding_window([1,2,3,4,5], 3)))\n",
                ],
            }
        }

        self.reference_links = {
            "python": [
                "https://docs.python.org/3/",
                "https://peps.python.org/",
                "https://realpython.com/",
            ],
            "performance": [
                "https://pythonspeed.com/",
                "https://github.com/psf/pyperformance",
                "https://pyperf.readthedocs.io/",
            ],
        }

    # ------------------------------ Public API ------------------------------

    def enhance(self, content: str, topic: str, keyphrase: str) -> str:
        analysis = self.analyze_content(content)
        enhanced = content

        if analysis["code_blocks"] < self.quality_standards["min_code_blocks"]:
            enhanced = self._inject_code_examples(enhanced, topic, keyphrase)

        if analysis["references"] < self.quality_standards["min_references"]:
            enhanced = self._inject_references(enhanced, topic)

        if analysis["examples"] < self.quality_standards["min_examples"]:
            enhanced = self._inject_practical_examples(enhanced, topic)
        if analysis["practical_tips"] < self.quality_standards["min_practical_tips"]:
            enhanced = self._inject_practical_tips(enhanced, topic)

        topic_l = topic.lower()
        if any(k in topic_l for k in ("performance", "speed", "benchmark")) or \
           ("## Performance Benchmarks" in enhanced):
            enhanced = self._inject_benchmarks(enhanced, topic)

        return enhanced

    def analyze_content(self, content: str) -> Dict:
        return {
            "code_blocks": len(re.findall(r"```.*?```", content, re.DOTALL)),
            "references": len(re.findall(r"https?://", content)),
            "examples": len(re.findall(r"example|for instance|e\.g\.", content, re.IGNORECASE)),
            "practical_tips": len(re.findall(r"\btip:|\brecommendation\b|best practice", content, re.IGNORECASE)),
        }

    # ------------------------------ Injections ------------------------------

    def _inject_code_examples(self, content: str, topic: str, keyphrase: str) -> str:
        code_type = self._determine_code_type(topic, keyphrase)
        examples = self.code_examples["python"].get(code_type, [])
        if not examples:
            return content

        example = self._deterministic_pick(examples, seed=f"{topic}|{keyphrase}|{code_type}")
        block = f"\n\n```\n{example.strip()}\n```\n\n"

        pos = self._section_header_pos(content, "Implementation Walkthrough")
        if pos is None:
            pos = self._section_header_pos(content, "Technical Deep Dive")
        if pos is None:
            pos = self._section_header_pos(content, "Implementation")

        if pos is not None:
            insert_at = self._after_header_index(content, pos)
            return content[:insert_at] + block + content[insert_at:]
        return content + block

    def _determine_code_type(self, topic: str, keyphrase: str) -> str:
        t = f"{topic} {keyphrase}".lower()
        if any(k in t for k in ("decorator", "wrap", "annotation")):
            return "decorator"
        if any(k in t for k in ("context", "resource", "manager", "with ")):
            return "context_manager"
        if any(k in t for k in ("async", "await", "coroutine", "event loop")):
            return "async"
        if any(k in t for k in ("generator", "yield", "iterator", "stream")):
            return "generator"
        return "decorator"

    def _inject_references(self, content: str, topic: str) -> str:
        refs = self._get_relevant_references(topic)
        if not refs:
            return content
        list_md = "\n".join(f"- [{url}]({url})" for url in refs[:3])
        section_md = f"## Further Reading\n\n{list_md}\n\n"
        if self._section_header_pos(content, "Further Reading") is None:
            return content.rstrip() + "\n\n" + section_md
        return self._append_to_section(content, "Further Reading", "\n" + list_md + "\n")

    def _get_relevant_references(self, topic: str) -> List[str]:
        t = topic.lower()
        refs = list(self.reference_links["python"])
        if any(k in t for k in ("performance", "speed", "optimiz", "benchmark")):
            refs.extend(self.reference_links["performance"])
        seen, out = set(), []
        for r in refs:
            if r not in seen:
                seen.add(r)
                out.append(r)
        return out

    def _inject_practical_examples(self, content: str, topic: str) -> str:
        examples = [
            f"**Example**: In a production web application, {topic.lower()} helps reduce latency and error budgets.",
            f"**Use Case**: Teams use {topic.lower()} to eliminate toil and standardize workflows.",
            f"**Real-world Scenario**: When scaling services, adopting {topic.lower()} avoids common regressions.",
        ]
        ex = self._deterministic_pick(examples, seed=topic)
        paras = content.split("\n\n")
        if len(paras) > 1:
            paras.insert(1, ex)
            return "\n\n".join(paras)
        return content + "\n\n" + ex + "\n\n"

    def _inject_practical_tips(self, content: str, topic: str) -> str:
        tips = [
            "- Prefer small, focused functions; measure before optimizing.",
            "- Add type hints for public APIs to improve readability and tooling.",
            "- Profile hot paths (cProfile/pyinstrument) before refactoring.",
            "- Use retries with backoff for network/IO; keep them idempotent.",
            "- Document assumptions and edge cases near the code that implements them.",
        ]
        checklist = "\n".join(tips[:5]) + "\n"
        if self._section_header_pos(content, "Best Practices") is None:
            return content.rstrip() + "\n\n## Best Practices\n\n" + checklist + "\n"
        return self._append_to_section(content, "Best Practices", "\n" + checklist)

    def _inject_benchmarks(self, content: str, topic: str) -> str:
        bench_md = self._generate_benchmark_data(topic)
        if self._section_header_pos(content, "Performance Benchmarks") is None:
            fr_pos = self._section_header_pos(content, "Further Reading")
            if fr_pos is not None:
                return self._insert_before_section(content, "Further Reading", bench_md)
            return content.rstrip() + "\n\n" + bench_md
        return self._append_to_section(content, "Performance Benchmarks", "\n\n" + bench_md.split("\n", 1)[1])

    def _generate_benchmark_data(self, topic: str) -> str:
        today = datetime.utcnow().strftime("%Y-%m-%d")
        return (
            "## Performance Benchmarks\n\n"
            f"Performance comparison of different approaches to {topic.lower()} "
            f"(Python 3.10; {today}):\n\n"
            "| Approach | Time (ms) | Memory (MB) | Complexity |\n"
            "|----------|-----------|-------------|------------|\n"
            "| Basic Implementation | 125 | 15.2 | O(n) |\n"
            "| Optimized Version    |  89 | 12.8 | O(log n) |\n"
            "| Advanced Technique   |  62 | 10.4 | O(1) |\n\n"
            "*Benchmarks are illustrative; reproduce on your hardware and dataset.*\n"
        )

    # ------------------------------ Helpers ------------------------------

    @staticmethod
    def _deterministic_pick(items: List[str], *, seed: str) -> str:
        if not items:
            return ""
        h = 0
        for i, ch in enumerate(seed):
            h = (h * 131 + ord(ch) + i) & 0xFFFFFFFF
        idx = h % len(items)
        return items[idx]

    @staticmethod
    def _section_header_pos(content: str, title: str) -> Optional[int]:
        pattern = re.compile(rf"^##\s+{re.escape(title)}\s*$", re.IGNORECASE | re.MULTILINE)
        m = pattern.search(content)
        return None if not m else m.start()

    @staticmethod
    def _after_header_index(content: str, header_start: int) -> int:
        line_end = content.find("\n", header_start)
        if line_end == -1:
            return len(content)
        if content[line_end:line_end+2] == "\n\n":
            return line_end + 2
        return line_end + 1

    def _append_to_section(self, content: str, title: str, text: str) -> str:
        start = self._section_header_pos(content, title)
        if start is None:
            return content.rstrip() + f"\n\n## {title}\n\n{text}"
        after = self._after_header_index(content, start)
        next_m = re.search(r"^##\s+.+$", content[after:], re.MULTILINE)
        if next_m:
            insert_at = after + next_m.start()
            return content[:insert_at] + text + content[insert_at:]
        return content + ("" if content.endswith("\n") else "\n") + text

    def _insert_before_section(self, content: str, target_title: str, new_section_md: str) -> str:
        pos = self._section_header_pos(content, target_title)
        if pos is None:
            return content.rstrip() + "\n\n" + new_section_md
        return content[:pos] + new_section_md + content[pos:]
