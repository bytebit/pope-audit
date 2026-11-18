"""Inspect ResDec output format with longer generation (max_new_tokens=16)
to confirm the 'yes\\n...' artifact is a short-generation quirk."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_pope import apply_prompt_template, item_prompt_len, load_model  # noqa: E402
from resdec import resdec_generate_batch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    items = json.loads((ROOT / "data" / "pope" / "random.json").read_text(encoding="utf-8"))[:n]
    print("Loading model...")
    processor, model = load_model(str(ROOT / "models" / "llava-1.5-7b-hf"))

    qs_all = [apply_prompt_template("ins-yesno", it["question"]) for it in items]
    lengths = [item_prompt_len(processor, q) for q in qs_all]
    order = sorted(range(n), key=lambda i: lengths[i])
    batches = []
    for i in order:
        if batches and lengths[i] == lengths[batches[-1][-1]] and len(batches[-1]) < 8:
            batches[-1].append(i)
        else:
            batches.append([i])

    preds = {}
    for idxs in batches:
        chunk = [items[i] for i in idxs]
        out = resdec_generate_batch(
            processor, model,
            [ROOT / it["image"] for it in chunk],
            [apply_prompt_template("ins-yesno", it["question"]) for it in chunk],
            max_new_tokens=16, alpha=0.5, beta=0.1, K=256, W=8,
        )
        preds.update(zip(idxs, out))

    preds = [preds[i] for i in range(n)]
    yes = sum(1 for p in preds if p.startswith("yes"))
    no = sum(1 for p in preds if p.startswith("no"))
    has_nl = sum(1 for p in preds if "\n" in p)
    print(f"n={n} yes={yes} no={no} other={n - yes - no} with_newline={has_nl}")
    print("\nSample outputs (first 12):")
    for i in range(12):
        print(f"  #{i} {preds[i]!r}")


if __name__ == "__main__":
    main()
