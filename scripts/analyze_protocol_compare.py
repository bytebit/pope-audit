"""Analyze fix/break patterns of ResDec vs Regular from a protocol_compare file."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def yn(t: str) -> str:
    return "yes" if t.startswith("yes") else "no"


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else None
    if path is None:
        cands = sorted((ROOT / "outputs").glob("protocol_compare_*.json"))
        if not cands:
            print("no protocol_compare file found")
            return
        path = str(cands[-1])
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    items = d["items"]
    for run_key in [k for k in d["runs"] if "ResDec" in k]:
        reg_key = run_key.split("/")[0] + "/Regular"
        reg = d["runs"][reg_key]["preds"]
        res = d["runs"][run_key]["preds"]
        fixes, breaks = [], []
        for i, (r, s, it) in enumerate(zip(reg, res, items)):
            r_ok = yn(r) == it["answer"]
            s_ok = yn(s) == it["answer"]
            if r_ok != s_ok:
                (fixes if s_ok else breaks).append((i, r, s, it))
        m = d["runs"][run_key]["metrics"]
        print(f"== {run_key}: Acc={m['accuracy']:.4f} F1={m['f1']:.4f} | "
              f"fixes={len(fixes)} breaks={len(breaks)} net={len(fixes) - len(breaks)}")
        print("  FIXED (Regular wrong -> ResDec right):")
        for i, r, s, it in fixes[:12]:
            print(f"    #{i} [{it['question']}] ans={it['answer']} "
                  f"reg={yn(r)}({r[:32]!r}) res={yn(s)}({s[:32]!r})")
        print("  BROKE (Regular right -> ResDec wrong):")
        for i, r, s, it in breaks[:10]:
            print(f"    #{i} [{it['question']}] ans={it['answer']} "
                  f"reg={yn(r)}({r[:32]!r}) res={yn(s)}({s[:32]!r})")


if __name__ == "__main__":
    main()
