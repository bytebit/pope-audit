"""ResDec candidate-pool-size (K) sensitivity on a fixed 300-item sample."""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_pope import apply_prompt_template, compute_metrics, item_prompt_len, load_model  # noqa: E402
from resdec import resdec_generate_batch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    ks = [int(x) for x in (sys.argv[2:] or ["64", "128", "256", "512"])]
    items = json.loads((ROOT / "data" / "pope" / "random.json").read_text(encoding="utf-8"))[:n]
    print("Loading model...")
    processor, model = load_model(str(ROOT / "models" / "llava-1.5-7b-hf"))

    qs_all = [apply_prompt_template("ins-yesno", it["question"]) for it in items]
    lengths = [item_prompt_len(processor, q) for q in qs_all]
    order = sorted(range(n), key=lambda i: lengths[i])
    batches = []
    for i in order:
        if batches and lengths[i] == lengths[batches[-1][-1]] and len(batches[-1]) < 4:
            batches[-1].append(i)
        else:
            batches.append([i])

    for K in ks:
        pairs = []
        t0 = time.time()
        for idxs in batches:
            chunk = [items[i] for i in idxs]
            preds = resdec_generate_batch(
                processor, model,
                [ROOT / it["image"] for it in chunk],
                [apply_prompt_template("ins-yesno", it["question"]) for it in chunk],
                max_new_tokens=8, alpha=0.5, beta=0.1, K=K, W=8,
            )
            pairs.extend((i, p) for i, p in zip(idxs, preds))
        preds = [p for _, p in sorted(pairs, key=lambda x: x[0])]
        m = compute_metrics(list(zip(preds, [it["answer"] for it in items])))
        print(f"K={K:4d}  Acc={m['accuracy']:.4f} F1={m['f1']:.4f} "
              f"empty={sum(1 for p in preds if not p)} time={time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
