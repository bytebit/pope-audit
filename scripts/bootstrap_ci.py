"""Bootstrap 95% CIs for the paper's headline numbers:
- echo rate (Regular, ins-yesno) per subset
- ResDec/VCD raw gains and echo/g genuine flip decomposition (per item)
"""

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RNG = np.random.default_rng(0)


def ci(values, stat_fn, n_boot=2000):
    vals = np.array(values)
    stats = []
    for _ in range(n_boot):
        idx = RNG.integers(0, len(vals), size=len(vals))
        stats.append(stat_fn(vals[idx]))
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return lo, hi


def yn(t: str) -> bool:
    return t.startswith("yes")


def main() -> None:
    reg_file = sorted((ROOT / "outputs").glob("pope_baseline_20260829_*.json"))[-1]
    res_file = sorted((ROOT / "outputs").glob("pope_resdec_20260829_*.json"))[-1]
    reg = json.loads(reg_file.read_text(encoding="utf-8"))
    res = json.loads(res_file.read_text(encoding="utf-8"))

    print("=== Regular(ins-yesno): echo rate (per subset, 95% CI) ===")
    for subset in ["adversarial", "popular", "random"]:
        preds = [x["pred"] for x in reg[subset]["predictions"]]
        echo = np.array([1.0 if "assistant" in p else 0.0 for p in preds])
        lo, hi = ci(echo, lambda x: x.mean())
        print(f"  {subset:12s} echo={echo.mean():.4f}  CI=[{lo:.4f}, {hi:.4f}]")

    print("\n=== ResDec: raw gain + flip decomposition (per subset, 95% CI) ===")
    for subset in ["adversarial", "popular", "random"]:
        rp = [x["pred"] for x in reg[subset]["predictions"]]
        sp = [x["pred"] for x in res[subset]["predictions"]]
        golds = [x["answer"] for x in reg[subset]["predictions"]]
        gain = np.array([1.0 if (yn(s) == (g == "yes")) else 0.0
                         for s, g in zip(sp, golds)]) - np.array(
            [1.0 if (yn(r) == (g == "yes")) else 0.0 for r, g in zip(rp, golds)])
        lo, hi = ci(gain, lambda x: x.mean())
        print(f"  {subset:12s} gain={gain.mean():+.4f}  CI=[{lo:+.4f}, {hi:+.4f}]")
        # flip decomposition CI
        fix_echo = np.array([1.0 if (not (yn(r) == (g == "yes"))) and (yn(s) == (g == "yes")) and "assistant" in r else 0.0
                             for r, s, g in zip(rp, sp, golds)])
        fix_gen = np.array([1.0 if (not (yn(r) == (g == "yes"))) and (yn(s) == (g == "yes")) and "assistant" not in r else 0.0
                            for r, s, g in zip(rp, sp, golds)])
        lo1, hi1 = ci(fix_echo, lambda x: x.mean())
        lo2, hi2 = ci(fix_gen, lambda x: x.mean())
        print(f"    fix_echo={fix_echo.mean():.4f} [{lo1:.4f},{hi1:.4f}]  "
              f"fix_genuine={fix_gen.mean():.4f} [{lo2:.4f},{hi2:.4f}]")

    print("\n=== VCD (500-item pilots): gain + decomposition CI ===")
    for f in sorted((ROOT / "outputs").glob("vcd_compare_*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        for pname in [k.split("/")[0] for k in d["runs"] if k.endswith("/VCD")]:
            rp = d["runs"][f"{pname}/Regular"]["preds"]
            vp = d["runs"][f"{pname}/VCD"]["preds"]
            golds = [it["answer"] for it in d["items"]]
            gain = np.array([1.0 if (yn(s) == (g == "yes")) else 0.0 for s, g in zip(vp, golds)]) - np.array(
                [1.0 if (yn(r) == (g == "yes")) else 0.0 for r, g in zip(rp, golds)])
            lo, hi = ci(gain, lambda x: x.mean())
            decomp = d["runs"][f"{pname}/VCD"]["decomp"]
            print(f"  {f.name} {pname}: subset={d['subset']} n={d['n']} "
                  f"gain={gain.mean():+.4f} [{lo:+.4f},{hi:+.4f}] "
                  f"decomp={decomp}")


if __name__ == "__main__":
    main()
