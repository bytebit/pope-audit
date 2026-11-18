"""Inspect PAS detection records: negative examples, score distributions."""

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    files = sorted((ROOT / "outputs").glob("pas_detection_*.json"))
    path = sys.argv[1] if len(sys.argv) > 1 else str(files[-1])
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    recs = d["records"]
    neg = [r for r in recs if r["label"] == 0]
    pos = [r for r in recs if r["label"] == 1]
    print(f"neg: {len(neg)}  pos: {len(pos)}")
    print("neg objects:", Counter(r["object"] for r in neg).most_common(15))
    print("\nsample neg captions:")
    for r in neg[:12]:
        print(f"  [{r['object']}] pas={r['pas']:.3f} svar={r['svar']:.3f} nll={r['nll']:.3f} ent={r['entropy']:.3f}")
        print(f"    {r['caption'][:110]}")
    print("\nscore means (pos vs neg):")
    for det in ["pas", "svar", "nll", "entropy"]:
        p = np.array([r[det] for r in pos])
        n = np.array([r[det] for r in neg])
        print(f"  {det:8s} pos {p.mean():.4f}±{p.std():.4f}  neg {n.mean():.4f}±{n.std():.4f}")


if __name__ == "__main__":
    main()
