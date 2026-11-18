"""Head-to-head protocol comparison: Regular vs ResDec across prompt templates
and historical windows W (paper default W=8) on a fixed POPE sample.

Saves per-item predictions so we can re-derive metrics under any parsing.
"""

import argparse
import json
import time
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_pope import compute_metrics, item_prompt_len, load_model, predict_batch  # noqa: E402
from resdec import resdec_generate_batch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

TEMPLATES = {
    "plain": lambda q: q,
    "ins-yesno": lambda q: f"{q} Please answer yes or no.",
}


def grouped_batches(processor, items, fn):
    """Order items by prompt length -> zero left-padding batches of size 8."""
    lengths = [item_prompt_len(processor, fn(it["question"])) for it in items]
    order = sorted(range(len(items)), key=lambda i: lengths[i])
    batches = []
    for i in order:
        if batches and lengths[i] == lengths[batches[-1][-1]] and len(batches[-1]) < 8:
            batches[-1].append(i)
        else:
            batches.append([i])
    return batches


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--subset", default="random")
    ap.add_argument("--windows", type=int, nargs="+", default=[8, 15])
    ap.add_argument("--prompts", nargs="+", default=list(TEMPLATES))
    args = ap.parse_args()

    items = json.loads((ROOT / "data" / "pope" / f"{args.subset}.json").read_text(encoding="utf-8"))[: args.n]
    print(f"Loading model...")
    t0 = time.time()
    processor, model = load_model(str(ROOT / "models" / "llava-1.5-7b-hf"))
    print(f"Model loaded in {time.time() - t0:.1f}s")

    results = {}
    for pname in args.prompts:
        fn = TEMPLATES[pname]
        batches = grouped_batches(processor, items, fn)

        # Regular
        pairs = []
        for idxs in batches:
            chunk = [items[i] for i in idxs]
            preds = predict_batch(
                processor, model,
                [ROOT / it["image"] for it in chunk],
                [fn(it["question"]) for it in chunk], 8,
            )
            pairs.extend((i, p) for i, p in zip(idxs, preds))
        reg_preds = [p for _, p in sorted(pairs, key=lambda x: x[0])]
        m = compute_metrics(list(zip(reg_preds, [it["answer"] for it in items])))
        key = f"{pname}/Regular"
        results[key] = {"preds": reg_preds, "metrics": m}
        print(f"[{key}] Acc={m['accuracy']:.4f} F1={m['f1']:.4f} "
              f"empty={sum(1 for p in reg_preds if not p)}")

        for W in args.windows:
            pairs = []
            t_start = time.time()
            for idxs in batches:
                chunk = [items[i] for i in idxs]
                preds = resdec_generate_batch(
                    processor, model,
                    [ROOT / it["image"] for it in chunk],
                    [fn(it["question"]) for it in chunk],
                    max_new_tokens=8, alpha=0.5, beta=0.1, K=256, W=W,
                )
                pairs.extend((i, p) for i, p in zip(idxs, preds))
            res_preds = [p for _, p in sorted(pairs, key=lambda x: x[0])]
            m = compute_metrics(list(zip(res_preds, [it["answer"] for it in items])))
            flips = sum(
                1 for r, s in zip(reg_preds, res_preds)
                if (r.startswith("yes") != s.startswith("yes"))
            )
            key = f"{pname}/ResDec-W{W}"
            results[key] = {"preds": res_preds, "metrics": m}
            print(f"[{key}] Acc={m['accuracy']:.4f} F1={m['f1']:.4f} "
                  f"empty={sum(1 for p in res_preds if not p)} flips_vs_regular={flips} "
                  f"time={time.time() - t_start:.0f}s")

    # Serialize (without giant per-run duplication of everything, keep metrics + preds)
    out = {
        "subset": args.subset,
        "n": args.n,
        "items": [
            {"question": it["question"], "answer": it["answer"]} for it in items
        ],
        "runs": {
            k: {"metrics": v["metrics"], "preds": v["preds"]} for k, v in results.items()
        },
    }
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = ROOT / "outputs" / f"protocol_compare_{stamp}.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
