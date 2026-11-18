"""Wrap the wide single-column tables of the ACL version in \resizebox so they
fit the column width (removes overfull-hbox warnings)."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEX = ROOT / "paper" / "latex" / "acl" / "main.tex"
TARGETS = {"tab:protocol", "tab:qwen", "tab:chair", "tab:detect",
           "tab:variants", "tab:noise"}

src = TEX.read_text(encoding="utf-8")
out = []
pos = 0
for m in re.finditer(r"\\begin\{table\}(?!\*)(.*?)\\end\{table\}", src, re.S):
    block = m.group(0)
    label = re.search(r"\\label\{(tab:[^}]+)\}", block)
    if label and label.group(1) in TARGETS and "resizebox" not in block:
        wrapped = block.replace(
            "\\begin{tabular}",
            "\\resizebox{\\columnwidth}{!}{%\n\\begin{tabular}", 1)
        wrapped = wrapped.replace(
            "\\end{tabular}",
            "\\end{tabular}\n}", 1)
        out.append(src[pos:m.start()])
        out.append(wrapped)
        pos = m.end()
out.append(src[pos:])
new = "".join(out)
TEX.write_text(new, encoding="utf-8")
print("resizebox occurrences now:", new.count("resizebox"))
