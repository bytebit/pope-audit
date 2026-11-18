"""Label-noise stress test for token-level hallucination detectors.

Flips detection labels with probability p (multiple seeds) and recomputes
AUROC for PAS / SVAR / NLL / Entropy, producing robustness curves.
"""

import json
import sys
from pathlib import Path

import numpy as np
import sklearn.metrics as skm

ROOT = Path(__file__).resolve().parents[1]

NOISE_LEVELS = [0.0, 0.05, 0.1, 0.2, 0.3]
SEEDS = [0, 1, 2, 3, 4]
DETECTORS = ["pas", "svar", "nll", "entropy"]


def auroc(labels, scores):
    if len(set(labels.tolist())) < 2 or len(set(scores.tolist())) < 2:
        return float("nan")
    return float(skm.roc_auc_score(labels, scores))


def main() -> None:
    files = sorted((ROOT / "outputs").glob("pas_detection_*.json"))
    if not files:
        print("no pas_detection results found")
        return
    path = sys.argv[1] if len(sys.argv) > 1 else str(files[-1])
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    records = d["records"]
    labels = np.array([r["label"] for r in records], dtype=int)
    scores = {det: np.array([r[det] for r in records], dtype=float) for det in DETECTORS}
    n_pos = int(labels.sum())
    print(f"records: {len(records)}  pos={n_pos} neg={len(records) - n_pos}")

    print("\n=== AUROC under label noise (mean over 5 seeds) ===")
    header = "noise   " + "".join(f"{d:>12s}" for d in DETECTORS)
    print(header)
    table = {}
    for p in NOISE_LEVELS:
        vals = {det: [] for det in DETECTORS}
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            flip = rng.random(len(labels)) < p
            noisy = labels.copy()
            noisy[flip] = 1 - noisy[flip]
            for det in DETECTORS:
                vals[det].append(auroc(noisy, scores[det]))
        row = f"{p:6.2f}"
        for det in DETECTORS:
            m = float(np.nanmean(vals[det]))
            row += f"{m:12.4f}"
        print(row)
        table[str(p)] = {det: round(float(np.nanmean(vals[det])), 4) for det in DETECTORS}

    out = ROOT / "outputs" / "label_noise_summary.json"
    out.write_text(json.dumps({"source": path, "noise_levels": NOISE_LEVELS,
                               "seeds": SEEDS, "auroc": table}, indent=1), encoding="utf-8")
    print("saved ->", out)


if __name__ == "__main__":
    main()
