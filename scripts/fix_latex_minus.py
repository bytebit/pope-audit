"""Mechanical fix: '$-$X' -> '$-X' (the former breaks math mode)."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = [ROOT / "paper" / "latex" / "acl" / "main.tex",
         ROOT / "paper" / "latex" / "main.tex"]

for p in FILES:
    s = p.read_text(encoding="utf-8")
    s2, n = re.subn(r"\$-\$(\d)", r"$- \1".replace("- ", "-"), s)
    p.write_text(s2, encoding="utf-8")
    left = s2.count("$-$")
    print(f"{p.name}: replaced {n}, remaining '$-$': {left}")
