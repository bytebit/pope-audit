"""Compare grouped-Regular vs ResDec predictions on the same 100 items."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_pope import compute_metrics, item_prompt_len, load_model, predict_batch  # noqa: E402
from resdec import resdec_generate_batch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    items = json.loads((ROOT / "data" / "pope" / "random.json").read_text(encoding="utf-8"))[:100]
    print("Loading model...")
    processor, model = load_model(str(ROOT / "models" / "llava-1.5-7b-hf"))

    lengths = [item_prompt_len(processor, it["question"]) for it in items]
    order = sorted(range(100), key=lambda i: lengths[i])
    batches = []
    for i in order:
        if batches and lengths[i] == lengths[batches[-1][-1]] and len(batches[-1]) < 8:
            batches[-1].append(i)
        else:
            batches.append([i])

    def run(fn):
        pairs = []
        for idxs in batches:
            chunk = [items[i] for i in idxs]
            preds = fn(chunk)
            pairs.extend((pred, items[i]["answer"]) for i, pred in zip(idxs, preds))
        pairs_sorted = sorted(zip(order, pairs), key=lambda x: x[0])
        return [pred for _, (pred, _ans) in pairs_sorted]

    reg = run(lambda chunk: predict_batch(processor, model, [ROOT / it["image"] for it in chunk],
                                          [it["question"] for it in chunk], 8))
    res = run(lambda chunk: resdec_generate_batch(processor, model, [ROOT / it["image"] for it in chunk],
                                                  [it["question"] for it in chunk],
                                                  max_new_tokens=8, alpha=0.5, beta=0.1, K=256, W=15))

    for label, preds in (("Regular", reg), ("ResDec", res)):
        m = compute_metrics(list(zip(preds, [it["answer"] for it in items])))
        print(f"{label}: Acc={m['accuracy']:.4f} F1={m['f1']:.4f} empty={sum(1 for p in preds if not p)}")

    flips = 0
    print("\nPer-item diffs (Regular correct but ResDec wrong / vice versa):")
    for i, (rp, sp, it) in enumerate(zip(reg, res, items)):
        r_ok = rp.startswith("yes") == (it["answer"] == "yes")
        s_ok = sp.startswith("yes") == (it["answer"] == "yes")
        if r_ok != s_ok:
            flips += 1
            print(f"  #{i} [{it['question']}] ans={it['answer']} | Reg={'yes' if rp.startswith('yes') else 'no'}({rp[:28]!r}) | Res={'yes' if sp.startswith('yes') else 'no'}({sp[:28]!r})")
    print(f"total flips: {flips}")


if __name__ == "__main__":
    main()
