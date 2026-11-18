"""Recompute echo rates and VCD flip decompositions from saved Qwen2.5-VL runs."""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RNG = np.random.default_rng(0)


def yn(t: str) -> str:
    return "yes" if t.startswith("yes") else "no"


def ci(vals, n_boot=2000):
    arr = np.array(vals)
    stats = [arr[RNG.integers(0, len(arr), size=len(arr))].mean() for _ in range(n_boot)]
    return np.percentile(stats, [2.5, 97.5])


def main() -> None:
    files = sorted((ROOT / "outputs").glob("qwen_pope_*.json"))
    # keep only real runs (n > 50)
    runs = []
    for f in files:
        d = json.loads(f.read_text(encoding="utf-8"))
        if d.get("n", 0) > 50:
            runs.append((f.name, d))
    if not runs:
        print("no qwen runs found")
        return
    print("| subset | prompt | Regular Acc/F1 | VCD Acc/F1 | gain [95% CI] | echo (Reg) |")
    print("| --- | --- | --- | --- | --- | --- |")
    for name, d in runs:
        items = d["items"]
        for pname in ["plain", "ins-yesno"]:
            rk, vk = f"{pname}/Regular", f"{pname}/VCD"
            if rk not in d["runs"] or vk not in d["runs"]:
                continue
            rp, vp = d["runs"][rk]["preds"], d["runs"][vk]["preds"]
            rm, vm = d["runs"][rk]["metrics"], d["runs"][vk]["metrics"]
            gain = np.array([1.0 if (yn(s) == it["answer"]) else 0.0 for s, it in zip(vp, items)]) - \
                np.array([1.0 if (yn(r) == it["answer"]) else 0.0 for r, it in zip(rp, items)])
            lo, hi = ci(gain)
            fe = fg = be = bg = 0
            for r, s, it in zip(rp, vp, items):
                r_ok, s_ok = yn(r) == it["answer"], yn(s) == it["answer"]
                if r_ok == s_ok:
                    continue
                echo = "assistant" in r
                if s_ok:
                    fe += int(echo)
                    fg += int(not echo)
                else:
                    be += int(echo)
                    bg += int(not echo)
            echo_rate = sum(1 for p in rp if "assistant" in p) / len(rp)
            print(f"| {d['subset']} | {pname} | {rm['accuracy']*100:.2f} / {rm['f1']*100:.2f} | "
                  f"{vm['accuracy']*100:.2f} / {vm['f1']*100:.2f} | "
                  f"{gain.mean()*100:+.2f} [{lo*100:+.2f}, {hi*100:+.2f}] | {echo_rate:.1%} |")
            print(f"<!-- {name} {d['subset']}/{pname}: fix_echo={fe} fix_genuine={fg} "
                  f"break_echo={be} break_genuine={bg} net={fe + fg - be - bg} -->")


if __name__ == "__main__":
    main()
