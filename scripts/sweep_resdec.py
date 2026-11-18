"""Hyperparameter sweep for ResDec on a fixed 100-item POPE sample (single model load)."""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_pope import compute_metrics, item_prompt_len, load_model, predict_batch  # noqa: E402
from resdec import resdec_generate_batch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

CONFIGS = [
    {"K": 256, "W": 15, "valley": "argmin", "label": "K256-W15-argmin"},
    {"K": 128, "W": 15, "valley": "argmin", "label": "K128-W15-argmin"},
    {"K": 512, "W": 15, "valley": "argmin", "label": "K512-W15-argmin"},
    {"K": 256, "W": 10, "valley": "argmin", "label": "K256-W10-argmin"},
    {"K": 256, "W": 20, "valley": "argmin", "label": "K256-W20-argmin"},
    {"K": 256, "W": 15, "valley": "lastmin", "label": "K256-W15-lastmin"},
]


def main() -> None:
    subset = "random"
    items = json.loads((ROOT / "data" / "pope" / f"{subset}.json").read_text(encoding="utf-8"))[:100]
    print(f"Loading model...")
    processor, model = load_model(str(ROOT / "models" / "llava-1.5-7b-hf"))

    # Regular baseline on the same 100 items
    pairs = []
    for i in range(0, 100, 8):
        chunk = items[i:i + 8]
        preds = predict_batch(processor, model, [ROOT / it["image"] for it in chunk],
                              [it["question"] for it in chunk], 8)
        pairs.extend(zip(preds, [it["answer"] for it in chunk]))
    m = compute_metrics(pairs)
    print(f"Regular   Acc={m['accuracy']:.4f} F1={m['f1']:.4f} empty={sum(1 for p, _ in pairs if not p)}")

    lengths = [item_prompt_len(processor, it["question"]) for it in items]
    order = sorted(range(100), key=lambda i: lengths[i])
    for cfg in CONFIGS:
        batches = []
        for i in order:
            if batches and lengths[i] == lengths[batches[-1][-1]] and len(batches[-1]) < 8:
                batches[-1].append(i)
            else:
                batches.append([i])
        pairs = []
        t0 = time.time()
        for idxs in batches:
            chunk = [items[i] for i in idxs]
            preds = resdec_generate_batch(
                processor, model,
                [ROOT / it["image"] for it in chunk],
                [it["question"] for it in chunk],
                max_new_tokens=8, alpha=0.5, beta=0.1,
                K=cfg["K"], W=cfg["W"], valley_mode=cfg["valley"],
            )
            pairs.extend((pred, items[i]["answer"]) for i, pred in zip(idxs, preds))
        pairs = [dict(zip(order, pairs))[i] for i in range(100)]
        m = compute_metrics(pairs)
        print(f"{cfg['label']:18s} Acc={m['accuracy']:.4f} F1={m['f1']:.4f} "
              f"empty={sum(1 for p, _ in pairs if not p)} time={time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
