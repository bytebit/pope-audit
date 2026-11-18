"""Aggregate the echo-rate / mitigation results across every model we tested.

Reads outputs/model_*_*.json (new models) and outputs/qwen_pope_*.json (Qwen2.5-VL)
and prints one table suitable for the paper, plus the VCD decomposition where
both Regular and VCD were run.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

BACKBONES = {
    "llava-1.5-7b-hf": "Vicuna-7B (LLaVA-1.5)",
    "llava-v1.6-vicuna-7b-hf": "Vicuna-7B (LLaVA-NeXT)",
    "llava-onevision-qwen2-7b-ov-hf": "Qwen2-7B (LLaVA-OneVision)",
    "qwen2.5-vl-7b-instruct": "Qwen2.5-7B (Qwen2.5-VL)",
    "qwen2.5-vl-3b-instruct": "Qwen2.5-3B",
}


def yn(t: str) -> str:
    return "yes" if t.startswith("yes") else "no"


def collect() -> dict:
    rows = {}
    for f in sorted((ROOT / "outputs").glob("model_*_*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        rows[(d["tag"], d["subset"])] = d
    for f in sorted((ROOT / "outputs").glob("qwen_pope_*_205435.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        rows[("qwen2.5-vl-7b-instruct", d["subset"])] = d
    return rows


def main() -> None:
    rows = collect()
    if not rows:
        print("no per-model results yet")
        return
    print("| model (LLM backbone) | subset | prompt | Acc | F1 | echo rate | VCD gain |")
    print("| --- | --- | --- | --- | --- | --- | --- |")
    summary = {}
    for (tag, subset), d in sorted(rows.items()):
        items = d["items"]
        for pname in ["plain", "ins-yesno"]:
            rk = f"{pname}/Regular"
            if rk not in d["runs"]:
                continue
            m = d["runs"][rk]["metrics"]
            preds = d["runs"][rk]["preds"]
            echo = sum(1 for p in preds if "assistant" in p) / len(preds)
            vk = f"{pname}/VCD"
            gain = ""
            if vk in d["runs"]:
                vp = d["runs"][vk]["preds"]
                correct_r = sum(1 for p, it in zip(preds, items) if yn(p) == it["answer"])
                correct_v = sum(1 for p, it in zip(vp, items) if yn(p) == it["answer"])
                gain = f"{(correct_v - correct_r) / len(items) * 100:+.1f}"
            backbone = BACKBONES.get(tag, tag)
            print(f"| {backbone} | {subset} | {pname} | {m['accuracy']*100:.1f} | "
                  f"{m['f1']*100:.1f} | {echo*100:.1f}% | {gain} |")
            summary.setdefault((backbone, pname), []).append(echo)

    print("\n=== echo rate by model and prompt (mean over subsets) ===")
    print("| model (backbone) | plain | ins-yesno |")
    print("| --- | --- | --- |")
    models = sorted({k[0] for k in summary})
    for m in models:
        plain = summary.get((m, "plain"), [])
        ins = summary.get((m, "ins-yesno"), [])
        p = f"{sum(plain)/len(plain)*100:.1f}%" if plain else "-"
        i = f"{sum(ins)/len(ins)*100:.1f}%" if ins else "-"
        print(f"| {m} | {p} | {i} |")


if __name__ == "__main__":
    main()
