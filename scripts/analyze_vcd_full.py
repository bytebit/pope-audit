"""Summarize full-scale VCD runs: per subset x prompt table + bootstrap CIs."""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RNG = np.random.default_rng(0)


def yn(t: str) -> bool:
    return t.startswith("yes")


def ci(vals, stat_fn, n_boot=2000):
    arr = np.array(vals)
    stats = [stat_fn(arr[RNG.integers(0, len(arr), size=len(arr))]) for _ in range(n_boot)]
    return np.percentile(stats, [2.5, 97.5])


def main() -> None:
    files = sorted((ROOT / "outputs").glob("vcd_full_*_*.json"))
    if not files:
        print("no vcd_full results")
        return
    by_subset = {}
    for f in files:
        d = json.loads(f.read_text(encoding="utf-8"))
        by_subset[d["subset"]] = d

    print("=== VCD full-scale: metrics + decomposition ===")
    for subset in ["adversarial", "popular", "random"]:
        d = by_subset.get(subset)
        if d is None:
            print(f"{subset}: missing")
            continue
        print(f"\n[{subset}] n={d['n']}")
        for pname in ["ins-yesno", "plain"]:
            rk, vk = f"{pname}/Regular", f"{pname}/VCD"
            if vk not in d["runs"]:
                continue
            rm, vm = d["runs"][rk]["metrics"], d["runs"][vk]["metrics"]
            rp, vp = d["runs"][rk]["preds"], d["runs"][vk]["preds"]
            golds = [it["answer"] for it in d["items"]]
            gain = np.array([1.0 if (yn(s) == (g == "yes")) else 0.0
                             for s, g in zip(vp, golds)]) - np.array(
                [1.0 if (yn(r) == (g == "yes")) else 0.0 for r, g in zip(rp, golds)])
            lo, hi = ci(gain, lambda x: x.mean())
            dec = d["runs"][vk].get("decomp", {})
            print(f"  {pname:9s} Reg={rm['accuracy']:.4f}/{rm['f1']:.4f} "
                  f"VCD={vm['accuracy']:.4f}/{vm['f1']:.4f} "
                  f"gain={gain.mean():+.4f} [{lo:+.4f},{hi:+.4f}] "
                  f"decomp={dec}")

    print("\n=== Summary table (for paper) ===")
    print("| subset | prompt | Regular | VCD | gain [95% CI] | fix_echo / fix_genuine / break_genuine |")
    print("| --- | --- | --- | --- | --- | --- |")
    for subset in ["adversarial", "popular", "random"]:
        d = by_subset.get(subset)
        if d is None:
            continue
        for pname in ["ins-yesno", "plain"]:
            rk, vk = f"{pname}/Regular", f"{pname}/VCD"
            if vk not in d["runs"]:
                continue
            rm, vm = d["runs"][rk]["metrics"], d["runs"][vk]["metrics"]
            rp, vp = d["runs"][rk]["preds"], d["runs"][vk]["preds"]
            golds = [it["answer"] for it in d["items"]]
            gain = np.array([1.0 if (yn(s) == (g == "yes")) else 0.0
                             for s, g in zip(vp, golds)]) - np.array(
                [1.0 if (yn(r) == (g == "yes")) else 0.0 for r, g in zip(rp, golds)])
            lo, hi = ci(gain, lambda x: x.mean())
            dec = d["runs"][vk].get("decomp", {})
            print(f"| {subset} | {pname} | {rm['accuracy']*100:.2f} / {rm['f1']*100:.2f} | "
                  f"{vm['accuracy']*100:.2f} / {vm['f1']*100:.2f} | "
                  f"{gain.mean()*100:+.2f} [{lo*100:+.2f}, {hi*100:+.2f}] | "
                  f"{dec.get('fix_echo', 0)} / {dec.get('fix_genuine', 0)} / {dec.get('break_genuine', 0)} |")


if __name__ == "__main__":
    main()
