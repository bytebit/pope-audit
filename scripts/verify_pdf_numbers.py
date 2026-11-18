"""Check that key numbers/terms of the paper appear in the compiled PDF."""

import sys
from pathlib import Path

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
pdf = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "paper" / "latex" / "acl" / "main.pdf"

reader = PdfReader(str(pdf))
text = " ".join(" ".join((p.extract_text() or "").split()) for p in reader.pages)
# PDF text extraction sometimes splits a word ("LLaV A"); compare a
# whitespace-free variant as well so kerning does not read as a missing term.
flat = text.replace(" ", "")

checks = ["LLaVA-NeXT", "OneVision", "Qwen2.5-VL", "42.4", "7.4", "1.2",
          "92.8", "idiosyncrasy", "CHAIR", "0.674", "ResDec"]
print(f"{pdf.name}: {len(reader.pages)} pages, unresolved '??': {text.count('??')}")
missing = []
for c in checks:
    status = "found" if (c in text or c.replace(" ", "") in flat) else "MISSING"
    if status == "MISSING":
        missing.append(c)
    print(f"   {c}: {status}")
if missing:
    sys.exit(f"missing terms: {missing}")
