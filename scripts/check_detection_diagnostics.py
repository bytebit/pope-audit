"""Post-hoc diagnostics for token-level detection AUROC:
- AUROC on early vs late caption tokens
- per-object-class AUROC
- negative record counts by object class vs the object's global frequency
"""

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import sklearn.metrics as skm

ROOT = Path(__file__).resolve().parents[1]


def auroc(labels, scores):
    if len(set(labels.tolist())) < 2 or len(set(scores.tolist())) < 2:
        return float("nan")
    return float(skm.roc_auc_score(labels, scores))


def main() -> None:
    files = sorted((ROOT / "outputs").glob("pas_detection_*.json"))
    path = sys.argv[1] if len(sys.argv) > 1 else str(files[-1])
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    recs = d["records"]
    labels = np.array([r["label"] for r in recs])
    print(f"records: {len(recs)} pos={int(labels.sum())} neg={int((1 - labels).sum())}")

    for det in ["pas", "svar", "nll", "entropy"]:
        scores = np.array([r[det] for r in recs])
        early = [i for i, r in enumerate(recs) if r["token_idx"] < 20]
        late = [i for i, r in enumerate(recs) if r["token_idx"] >= 20]
        print(f"\n{det}: all={auroc(labels, scores):.4f} "
              f"early(<20)={auroc(labels[early], scores[early]):.4f} (n={len(early)}) "
              f"late(>=20)={auroc(labels[late], scores[late]):.4f} (n={len(late)})")

    print("\ntop negative classes (caption mentions absent objects):")
    neg = [r for r in recs if r["label"] == 0]
    for obj, c in Counter(r["object"] for r in neg).most_common(20):
        # global present frequency of this class among positive mentions
        total = sum(1 for r in recs if r["object"] == obj)
        pos_c = sum(1 for r in recs if r["object"] == obj and r["label"] == 1)
        print(f"  {obj:16s} neg={c:4d} total={total:4d} pos_ratio={pos_c / max(1, total):.2f}")


if __name__ == "__main__":
    main()
