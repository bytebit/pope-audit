"""Paired significance tests (McNemar exact/chi-square) for every mitigation
comparison: baseline vs method, per subset and prompt, with the artifact
component isolated."""

import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, chi2

ROOT = Path(__file__).resolve().parents[1]


def yn(t: str) -> str:
    return "yes" if t.startswith("yes") else "no"


def mcnemar(reg, method, golds):
    """Return (b, c) for a 2x2 table where b = reg wrong / method right,
    c = reg right / method wrong, plus exact p-value."""
    b = sum(1 for r, s, g in zip(reg, method, golds)
            if yn(r) != g and yn(s) == g)
    c = sum(1 for r, s, g in zip(reg, method, golds)
            if yn(r) == g and yn(s) != g)
    n = b + c
    p = binomtest(b, n, 0.5).pvalue if n > 0 else 1.0
    return b, c, p


def main() -> None:
    baselines = {
        "ins-yesno": json.loads((ROOT / "outputs" / "pope_baseline_20260829_001941.json").read_text(encoding="utf-8")),
        "plain": json.loads((ROOT / "outputs" / "pope_baseline_20260828_210602.json").read_text(encoding="utf-8")),
    }
    resdec = json.loads((ROOT / "outputs" / "pope_resdec_20260829_015107.json").read_text(encoding="utf-8"))

    print("=== McNemar tests: ResDec (ins-yesno) vs baseline (4-bit LLaVA) ===")
    print(f"{'subset':12s} {'b (fix)':>8s} {'c (break)':>10s} {'p (exact)':>12s} "
          f"{'b_echo':>7s} {'b_genuine':>10s} {'p_genuine':>11s}")
    for subset in ["adversarial", "popular", "random"]:
        reg = [x["pred"] for x in baselines["ins-yesno"][subset]["predictions"]]
        me = resdec[subset]["predictions"]
        mp = [x["pred"] for x in me]
        golds = [x["answer"] for x in baselines["ins-yesno"][subset]["predictions"]]
        b, c, p = mcnemar(reg, mp, golds)
        # restrict to non-echo baseline errors (genuine corrections only)
        b_gen = sum(1 for r, s, g in zip(reg, mp, golds)
                    if yn(r) != g and yn(s) == g and "assistant" not in r)
        c_gen = sum(1 for r, s, g in zip(reg, mp, golds)
                    if yn(r) == g and yn(s) != g and "assistant" not in r)
        n_gen = b_gen + c_gen
        p_gen = binomtest(b_gen, n_gen, 0.5).pvalue if n_gen > 0 else 1.0
        print(f"{subset:12s} {b:8d} {c:10d} {p:12.3g} {b - b_gen:7d} {b_gen:10d} {p_gen:11.3g}")

    print("\n=== McNemar tests: VCD (full 3000x3) vs baseline ===")
    for f in sorted((ROOT / "outputs").glob("vcd_full_*_*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        subset = d["subset"]
        for pname in ["ins-yesno", "plain"]:
            rk, vk = f"{pname}/Regular", f"{pname}/VCD"
            if vk not in d["runs"]:
                continue
            reg, mp = d["runs"][rk]["preds"], d["runs"][vk]["preds"]
            golds = [it["answer"] for it in d["items"]]
            b, c, p = mcnemar(reg, mp, golds)
            b_gen = sum(1 for r, s, g in zip(reg, mp, golds)
                        if yn(r) != g and yn(s) == g and "assistant" not in r)
            c_gen = sum(1 for r, s, g in zip(reg, mp, golds)
                        if yn(r) == g and yn(s) != g and "assistant" not in r)
            n_gen = b_gen + c_gen
            p_gen = binomtest(b_gen, n_gen, 0.5).pvalue if n_gen > 0 else 1.0
            print(f"{subset:12s} {pname:9s} b={b:4d} c={c:3d} p={p:.3g} | "
                  f"genuine: b={b_gen:3d} c={c_gen:3d} p={p_gen:.3g}")

    print("\n=== McNemar tests: Qwen2.5-VL (VCD vs baseline) ===")
    for f in sorted((ROOT / "outputs").glob("qwen_pope_*_205435.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        for pname in ["ins-yesno", "plain"]:
            rk, vk = f"{pname}/Regular", f"{pname}/VCD"
            if vk not in d["runs"]:
                continue
            reg, mp = d["runs"][rk]["preds"], d["runs"][vk]["preds"]
            golds = [it["answer"] for it in d["items"]]
            b, c, p = mcnemar(reg, mp, golds)
            print(f"{d['subset']:12s} {pname:9s} b={b:3d} c={c:3d} p={p:.3g} (echo=0)")


if __name__ == "__main__":
    main()
