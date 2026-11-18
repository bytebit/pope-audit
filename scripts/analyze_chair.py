"""Paired significance tests for the CHAIR comparison (Regular vs methods)."""

import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
RNG = np.random.default_rng(0)


def main() -> None:
    f = sorted((ROOT / "outputs").glob("chair_*.json"))[-1]
    d = json.loads(f.read_text(encoding="utf-8"))
    res = d["results"]
    print(f"source: {f.name}  images={d['images']}  max_new_tokens={d['max_new_tokens']}\n")
    print("| method | CHAIR_S | CHAIR_I | mentions/cap |")
    print("| --- | --- | --- | --- |")
    for m, r in res.items():
        print(f"| {m} | {r['chair_s']:.4f} | {r['chair_i']:.4f} | {r['mentions_per_caption']} |")

    base = res["Regular"]["per_image"]
    print("\npaired tests vs Regular (per image, n=%d):" % len(base))
    for m in res:
        if m == "Regular":
            continue
        other = res[m]["per_image"]
        # caption-level: has >=1 hallucinated mention
        b = sum(1 for x, y in zip(base, other)
                if not x["hallucinated"] and y["hallucinated"])       # became hallucinating
        c = sum(1 for x, y in zip(base, other)
                if x["hallucinated"] and not y["hallucinated"])       # became clean
        p_cap = binomtest(min(b, c), b + c, 0.5).pvalue if b + c else 1.0
        # mention-level CHAIR_I difference (paired bootstrap over images)
        diffs = []
        n_base = [len(x["mentions"]) or 1 for x in base]
        n_other = [len(y["mentions"]) or 1 for y in other]
        h_base = [len(x["hallucinated"]) for x in base]
        h_other = [len(y["hallucinated"]) for y in other]
        a = np.array([hb / nb for hb, nb in zip(h_base, n_base)])
        bb = np.array([ho / no for ho, no in zip(h_other, n_other)])
        delta = bb - a
        boots = [delta[RNG.integers(0, len(delta), len(delta))].mean() for _ in range(2000)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        print(f"  {m:8s} CHAIR_S change: +{b} / -{c} captions, McNemar p={p_cap:.3g}; "
              f"mean CHAIR_I delta = {delta.mean():+.4f} [{lo:+.4f},{hi:+.4f}]")


if __name__ == "__main__":
    main()
