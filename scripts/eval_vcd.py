"""Compare Regular vs VCD on a POPE sample with echo-artifact decomposition."""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_pope import apply_prompt_template, compute_metrics, item_prompt_len, load_model, predict_batch  # noqa: E402
from vcd import vcd_generate_batch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def grouped_batches(processor, items, fn, bs=4):
    lengths = [item_prompt_len(processor, fn(it["question"])) for it in items]
    order = sorted(range(len(items)), key=lambda i: lengths[i])
    batches = []
    for i in order:
        if batches and lengths[i] == lengths[batches[-1][-1]] and len(batches[-1]) < bs:
            batches[-1].append(i)
        else:
            batches.append([i])
    return batches


def yn(t: str) -> str:
    return "yes" if t.startswith("yes") else "no"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--subsets", nargs="+", default=["random"],
                    choices=["random", "popular", "adversarial"])
    ap.add_argument("--prompts", nargs="+", default=["plain", "ins-yesno"])
    ap.add_argument("--batch-size", type=int, default=4)
    args = ap.parse_args()

    print("Loading model...")
    processor, model = load_model(str(ROOT / "models" / "llava-1.5-7b-hf"))

    # reuse existing full Regular baselines when available (deterministic greedy)
    baseline_files = {}
    for pname in args.prompts:
        cand = {
            "plain": ROOT / "outputs" / "pope_baseline_20260828_210602.json",
            "ins-yesno": ROOT / "outputs" / "pope_baseline_20260829_001941.json",
        }.get(pname)
        if cand and cand.exists():
            baseline_files[pname] = json.loads(cand.read_text(encoding="utf-8"))

    stamp = time.strftime("%Y%m%d_%H%M%S")
    for subset in args.subsets:
        out_path = ROOT / "outputs" / f"vcd_full_{subset}_{stamp}.json"
        if out_path.exists():
            print(f"skip {subset}: {out_path} exists", flush=True)
            continue
        items = json.loads((ROOT / "data" / "pope" / f"{subset}.json").read_text(encoding="utf-8"))[: args.n]
        out = {"subset": subset, "n": args.n,
               "items": [{"question": it["question"], "answer": it["answer"]} for it in items],
               "runs": {}}
        for pname in args.prompts:
            fn = lambda q, pname=pname: apply_prompt_template(pname, q)  # noqa: E731
            batches = grouped_batches(processor, items, fn, args.batch_size)
            for label, decode_fn in [
                ("Regular", lambda imgs, qs: predict_batch(processor, model, imgs, qs, 8)),
                ("VCD", lambda imgs, qs: vcd_generate_batch(processor, model, imgs, qs, max_new_tokens=8)),
            ]:
                key = f"{pname}/{label}"
                if label == "Regular" and pname in baseline_files and args.n >= len(items):
                    preds = [x["pred"] for x in baseline_files[pname][subset]["predictions"]][: args.n]
                    m = compute_metrics(list(zip(preds, [it["answer"] for it in items])))
                    print(f"[{key}] Acc={m['accuracy']:.4f} F1={m['f1']:.4f} (baseline reused)", flush=True)
                else:
                    pairs = []
                    t0 = time.time()
                    for idxs in batches:
                        chunk = [items[i] for i in idxs]
                        preds = decode_fn(
                            [ROOT / it["image"] for it in chunk],
                            [fn(it["question"]) for it in chunk],
                        )
                        pairs.extend((i, p) for i, p in zip(idxs, preds))
                    preds = [p for _, p in sorted(pairs, key=lambda x: x[0])]
                    m = compute_metrics(list(zip(preds, [it["answer"] for it in items])))
                    print(f"[{key}] Acc={m['accuracy']:.4f} F1={m['f1']:.4f} "
                          f"time={time.time() - t0:.0f}s", flush=True)
                out["runs"][key] = {"metrics": m, "preds": preds}

        # decomposition of VCD flips vs Regular (echo vs genuine)
        for pname in args.prompts:
            reg = out["runs"][f"{pname}/Regular"]["preds"]
            vcd = out["runs"][f"{pname}/VCD"]["preds"]
            fe = fg = be = bg = 0
            for r, s, it in zip(reg, vcd, items):
                r_ok = yn(r) == it["answer"]
                s_ok = yn(s) == it["answer"]
                if r_ok == s_ok:
                    continue
                echo = "assistant" in r
                if s_ok:
                    if echo:
                        fe += 1
                    else:
                        fg += 1
                else:
                    if echo:
                        be += 1
                    else:
                        bg += 1
            print(f"[{subset}/{pname}] VCD flips: fix_echo={fe} fix_genuine={fg} "
                  f"break_echo={be} break_genuine={bg} net={fe + fg - be - bg}", flush=True)
            out["runs"][f"{pname}/VCD"]["decomp"] = {
                "fix_echo": fe, "fix_genuine": fg, "break_echo": be, "break_genuine": bg,
            }
        out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"saved -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
