"""Print a compact summary of every per-model POPE run saved so far."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

for f in sorted((ROOT / "outputs").glob("model_*.json")):
    d = json.loads(f.read_text(encoding="utf-8"))
    print(f"=== {d['tag']} / {d['subset']} (n={d['n']})")
    for key, v in d["runs"].items():
        m = v["metrics"]
        n = len(v["preds"])
        echo = v.get("echo", sum(1 for p in v["preds"] if "assistant" in p))
        print(f"   {key:26s} Acc={m['accuracy']:.4f} F1={m['f1']:.4f} "
              f"echo={echo}/{n} ({echo / n * 100:.1f}%)")
