"""Re-run POPE items that produced empty predictions in single-image mode."""

import json
import sys
import time
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_pope import load_model, predict_batch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    subset = sys.argv[1] if len(sys.argv) > 1 else "random"
    result_path = sorted((ROOT / "outputs").glob("pope_baseline_*.json"))[-1]
    results = json.loads(result_path.read_text(encoding="utf-8"))
    preds = results[subset]["predictions"]
    items = json.loads((ROOT / "data" / "pope" / f"{subset}.json").read_text(encoding="utf-8"))

    empty_idx = [i for i, p in enumerate(preds) if not p["pred"]]
    print(f"{subset}: {len(empty_idx)} empty predictions -> re-running in single-image mode")

    processor, model = load_model(str(ROOT / "models" / "llava-1.5-7b-hf"))
    still_empty = 0
    for i in tqdm(empty_idx):
        it = items[i]
        pred = predict_batch(processor, model, [ROOT / it["image"]], [it["question"]], 8)[0]
        if not pred:
            still_empty += 1
            print("STILL EMPTY:", it["image"].split("/")[-1], "|", it["question"], "| answer:", it["answer"])
    print(f"single-image re-run: {still_empty}/{len(empty_idx)} still empty")


if __name__ == "__main__":
    main()
