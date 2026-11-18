"""Compare the 5000x512 PAS run against the PAS paper's reported numbers and
recompute label-noise robustness."""

import json
import sys
from pathlib import Path

import numpy as np
import sklearn.metrics as skm

ROOT = Path(__file__).resolve().parents[1]

PAPER_GREEDY = {"pas": 0.842, "svar": 0.815, "nll": 0.565, "entropy": 0.716}
DETECTORS = ["pas", "svar", "nll", "entropy"]
NOISE = [0.0, 0.05, 0.1, 0.2, 0.3]
SEEDS = [0, 1, 2, 3, 4]


def auroc(labels, scores):
    if len(set(labels.tolist())) < 2 or len(set(scores.tolist())) < 2:
        return float("nan")
    return float(skm.roc_auc_score(labels, scores))


def main() -> None:
    files = sorted((ROOT / "outputs").glob("pas_detection_*.json"))
    if not files:
        print("no results")
        return
    path = sys.argv[1] if len(sys.argv) > 1 else str(files[-1])
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    recs = d["records"]
    labels = np.array([r["label"] for r in recs], dtype=int)
    lens = [r.get("gen_len", 0) for r in recs]
    print(f"source: {path}")
    print(f"records={len(recs)} pos={int(labels.sum())} neg={int((1 - labels).sum())} "
          f"neg_ratio={(1 - labels).mean():.1%}")
    if lens:
        print(f"gen_len (per record): min={min(lens)} median={int(np.median(lens))} "
              f"p90={int(np.percentile(lens, 90))} max={max(lens)}")

    print("\n=== AUROC: ours vs paper (greedy, LLaVA-1.5-7B MSCOCO) ===")
    print(f"{'detector':10s} {'ours':>8s} {'paper':>8s} {'delta':>8s}")
    clean = {}
    for det in DETECTORS:
        scores = np.array([r[det] for r in recs], dtype=float)
        a = auroc(labels, scores)
        clean[det] = a
        p = PAPER_GREEDY[det]
        print(f"{det:10s} {a:8.4f} {p:8.3f} {a - p:+8.3f}")

    print("\n=== label noise (mean over 5 seeds) ===")
    header = "noise   " + "".join(f"{x:>10s}" for x in DETECTORS)
    print(header)
    noise_tbl = {}
    for p in NOISE:
        vals = {det: [] for det in DETECTORS}
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            flip = rng.random(len(labels)) < p
            noisy = labels.copy()
            noisy[flip] = 1 - noisy[flip]
            for det in DETECTORS:
                vals[det].append(auroc(noisy, np.array([r[det] for r in recs], dtype=float)))
        row = f"{p:6.2f}"
        for det in DETECTORS:
            row += f"{float(np.nanmean(vals[det])):10.4f}"
        print(row)
        noise_tbl[str(p)] = {det: round(float(np.nanmean(vals[det])), 4) for det in DETECTORS}

    out = ROOT / "outputs" / "pas_5000_summary.json"
    out.write_text(json.dumps({
        "source": path,
        "records": len(recs),
        "pos": int(labels.sum()),
        "neg": int((1 - labels).sum()),
        "clean_auroc": clean,
        "paper_greedy": PAPER_GREEDY,
        "label_noise": noise_tbl,
    }, indent=1), encoding="utf-8")
    print("saved ->", out)


if __name__ == "__main__":
    main()
