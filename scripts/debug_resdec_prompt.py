"""Compare Regular vs ResDec under two prompt templates (100 items)."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_pope import compute_metrics, load_model, predict_batch  # noqa: E402
from resdec import resdec_generate_batch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    items = json.loads((ROOT / "data" / "pope" / "random.json").read_text(encoding="utf-8"))[:100]
    print("Loading model...")
    processor, model = load_model(str(ROOT / "models" / "llava-1.5-7b-hf"))

    def batched(prompt_fn, decode_fn):
        pairs = []
        for i in range(0, 100, 8):
            chunk = items[i:i + 8]
            qs = [prompt_fn(it["question"]) for it in chunk]
            preds = decode_fn([ROOT / it["image"] for it in chunk], qs)
            pairs.extend(zip(preds, [it["answer"] for it in chunk]))
        return pairs

    for label, prompt_fn in [
        ("plain", lambda q: q),
        ("pope-style", lambda q: f"Question: {q}\nAnswer:"),
    ]:
        reg = batched(prompt_fn, lambda imgs, qs: predict_batch(processor, model, imgs, qs, 8))
        res = batched(prompt_fn, lambda imgs, qs: resdec_generate_batch(
            processor, model, imgs, qs, max_new_tokens=8, alpha=0.5, beta=0.1, K=256, W=15))
        mr = compute_metrics(reg)
        ms = compute_metrics(res)
        print(f"[{label}] Regular Acc={mr['accuracy']:.4f} F1={mr['f1']:.4f} | "
              f"ResDec Acc={ms['accuracy']:.4f} F1={ms['f1']:.4f} | "
              f"flips={sum(1 for a, b in zip(reg, res) if (a[0].startswith('yes') != b[0].startswith('yes')))}")


if __name__ == "__main__":
    main()
