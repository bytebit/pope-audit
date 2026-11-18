"""POPE evaluation with ResDec decoding (LLaVA-1.5-7B, 4bit)."""

import argparse
import json
import time
from pathlib import Path

from tqdm import tqdm

from eval_pope import apply_prompt_template, compute_metrics, item_prompt_len, load_model
from resdec import resdec_generate_batch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(Path(__file__).resolve().parents[1] / "models" / "llava-1.5-7b-hf"))
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--subsets", nargs="+", default=["random"], choices=["random", "popular", "adversarial"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--topk", type=int, default=256)
    ap.add_argument("--window", type=int, default=15)
    ap.add_argument("--valley", default="argmin", choices=["argmin", "lastmin"])
    ap.add_argument("--prompt-template", default="plain", choices=["plain", "ins-yesno"])
    args = ap.parse_args()

    root: Path = args.root
    outputs_dir = root / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {args.model}")
    processor, model = load_model(args.model)
    print(f"ResDec hyperparams: alpha={args.alpha} beta={args.beta} K={args.topk} W={args.window} valley={args.valley}")

    all_results = {}
    for subset in args.subsets:
        data_path = root / "data" / "pope" / f"{subset}.json"
        if not data_path.exists():
            print(f"Missing data: {data_path}")
            continue
        items = json.loads(data_path.read_text(encoding="utf-8"))
        if args.limit > 0:
            items = items[: args.limit]

        # group by identical prompt length -> zero left-padding batches
        qs_all = [apply_prompt_template(args.prompt_template, it["question"]) for it in items]
        lengths = [item_prompt_len(processor, q) for q in qs_all]
        order = sorted(range(len(items)), key=lambda i: lengths[i])
        batches = []
        for i in order:
            if batches and lengths[i] == lengths[batches[-1][-1]] and len(batches[-1]) < args.batch_size:
                batches[-1].append(i)
            else:
                batches.append([i])

        pairs = []
        t_start = time.time()
        for idxs in tqdm(batches, desc=f"ResDec/POPE/{subset}"):
            chunk = [items[i] for i in idxs]
            preds = resdec_generate_batch(
                processor,
                model,
                [root / it["image"] for it in chunk],
                [apply_prompt_template(args.prompt_template, it["question"]) for it in chunk],
                max_new_tokens=args.max_new_tokens,
                alpha=args.alpha,
                beta=args.beta,
                K=args.topk,
                W=args.window,
                valley_mode=args.valley,
            )
            pairs.extend((pred, items[i]["answer"]) for i, pred in zip(idxs, preds))
        by_idx = {i: p for i, p in zip(order, pairs)}
        pairs = [by_idx[i] for i in range(len(items))]

        metrics = compute_metrics(pairs)
        metrics["num_samples"] = len(pairs)
        metrics["elapsed_sec"] = round(time.time() - t_start, 1)
        metrics["predictions"] = [
            {"question": it["question"], "pred": p, "answer": a}
            for it, (p, a) in zip(items, pairs)
        ]
        metrics["hyperparams"] = {
            "alpha": args.alpha, "beta": args.beta, "K": args.topk,
            "W": args.window, "valley": args.valley,
        }
        all_results[subset] = metrics
        print(f"ResDec/POPE/{subset}: {metrics}")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = outputs_dir / f"pope_resdec_{stamp}.json"
    out_path.write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
