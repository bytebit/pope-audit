"""Debug VCD generation on 2 items with step-by-step prints."""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_pope import apply_prompt_template, load_model  # noqa: E402
from vcd import vcd_generate_batch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    items = json.loads((ROOT / "data" / "pope" / "random.json").read_text(encoding="utf-8"))[:2]
    print("Loading model...", flush=True)
    processor, model = load_model(str(ROOT / "models" / "llava-1.5-7b-hf"))
    print("model loaded", flush=True)
    t0 = time.time()
    preds = vcd_generate_batch(
        processor, model,
        [ROOT / it["image"] for it in items],
        [apply_prompt_template("ins-yesno", it["question"]) for it in items],
        max_new_tokens=8,
    )
    print("preds:", preds, "time:", round(time.time() - t0, 1), flush=True)


if __name__ == "__main__":
    main()
