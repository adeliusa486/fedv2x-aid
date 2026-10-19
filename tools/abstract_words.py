"""Count the words of the abstract of a LaTeX manuscript (Discover IoT limit: under 250 words).

Usage: python tools/abstract_words.py path/to/manuscript.tex
"""
import re
import sys
from pathlib import Path

tex = Path(sys.argv[1]).read_text(encoding="utf8")
start = tex.index(r"\begin{abstract}") + len(r"\begin{abstract}")
abstract = tex[start:tex.index(r"\end{abstract}")]
for a, b in ((r"$\times$", "x"), ("{,}", ","), (r"\,", " "), (r"\%", "%"), ("--", "-")):
    abstract = abstract.replace(a, b)
words = re.findall(r"\S+", abstract)
print(f"abstract words: {len(words)}")
sys.exit(0 if len(words) < 250 else 1)
