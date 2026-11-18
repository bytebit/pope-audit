"""Screen POPE prompt templates on a small sample: which template makes
ResDec's historical signal informative (vs template-no-bias artifact)."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_pope import compute_metrics, item_prompt_len, load_model, predict_batch  # noqa: E402
from resdec import resdec_generate_batch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

TEMPLATES = {
    "plain": lambda q: q,
    "ins-yesno": lambda q: f"{q} Please answer yes or no.",
    "Q-ins-yesno-A": lambda q: f"Question: {q} Please answer yes or no.\nAnswer:",
    "Q-A": lambda q: f"Question: {q}\nAnswer:",
    "Q-word-A": lambda q: f"Question: {q}\nAnswer with the word yes or no.",
}


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    which = sys.argv[2:] or list(TEMPLATES)
    items = json.loads((ROOT / "data" / "pope" / "random.json").read_text(encoding="utf-8"))[:n]
    print(f"Loading model...")
    processor, model = load_model(str(ROOT / "models" / "llava-1.5-7b-hf"))

    def grouped(fn):
        """Order items by prompt length so each batch has zero left-padding
        (avoids the 4bit+transformers5.9 NaN empty-answer bug)."""
        lengths = [item_prompt_len(processor, fn(it["question"])) for it in items]
        order = sorted(range(len(items)), key=lambda i: lengths[i])
        batches = []
        for i in order:
            if batches and lengths[i] == lengths[batches[-1][-1]] and len(batches[-1]) < 8:
                batches[-1].append(i)
            else:
                batches.append([i])
        return [(i, items[i]) for idxs in batches for i in idxs]

    for name in which:
        fn = TEMPLATES[name]
        flat = grouped(fn)
        reg_pairs, res_pairs = [], []
        for start in range(0, len(flat), 8):
            block = flat[start:start + 8]
            its = [it for _, it in block]
            rp = predict_batch(processor, model,
                               [ROOT / it["image"] for it in its],
                               [fn(it["question"]) for it in its], 8)
            sp = resdec_generate_batch(
                processor, model,
                [ROOT / it["image"] for it in its],
                [fn(it["question"]) for it in its],
                max_new_tokens=8, alpha=0.5, beta=0.1, K=256, W=8)
            reg_pairs.extend(zip([i for i, _ in block], rp, [it["answer"] for _, it in block]))
            res_pairs.extend(zip([i for i, _ in block], sp, [it["answer"] for _, it in block]))
        reg = [(p, a) for _, p, a in sorted(reg_pairs)]
        res = [(p, a) for _, p, a in sorted(res_pairs)]
        mr = compute_metrics(reg)
        ms = compute_metrics(res)
        flips = [(i, it["question"], reg[i][0], res[i][0], it["answer"])
                 for i, it in enumerate(items)
                 if (reg[i][0].startswith("yes") != res[i][0].startswith("yes"))]
        print(f"[{name}] Regular Acc={mr['accuracy']:.4f} F1={mr['f1']:.4f} | "
              f"ResDec Acc={ms['accuracy']:.4f} F1={ms['f1']:.4f} | flips={len(flips)}")
        for i, q, r, s, a in flips[:12]:
            print(f"    #{i} [{q}] ans={a} reg={r[:24]!r} res={s[:24]!r} "
                  f"{'FIX' if (s.startswith('yes') == (a == 'yes')) else 'BREAK'}")


if __name__ == "__main__":
    main()
