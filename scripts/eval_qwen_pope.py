"""POPE evaluation on Qwen2.5-VL: Regular vs VCD, echo rate + flip decomposition."""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_pope import PROMPT_TEMPLATES, compute_metrics  # noqa: E402
from qwen_vl import load_qwen, predict_batch, vcd_generate_batch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def yn(t: str) -> str:
    return "yes" if t.startswith("yes") else "no"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "models" / "qwen2.5-vl-7b-instruct"))
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--subsets", nargs="+", default=["random"],
                    choices=["random", "popular", "adversarial"])
    ap.add_argument("--prompts", nargs="+", default=["plain", "ins-yesno"])
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--bits", type=int, default=4, choices=[4, 8, 16])
    ap.add_argument("--methods", nargs="+", default=["Regular", "VCD"])
    args = ap.parse_args()

    print(f"Loading Qwen2.5-VL ({args.bits}-bit): {args.model}", flush=True)
    t0 = time.time()
    processor, model = load_qwen(args.model, bits=args.bits)
    print(f"model loaded in {time.time() - t0:.1f}s", flush=True)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    for subset in args.subsets:
        out_path = ROOT / "outputs" / f"qwen_pope_{subset}_{stamp}.json"
        items = json.loads((ROOT / "data" / "pope" / f"{subset}.json").read_text(encoding="utf-8"))[: args.n]
        out = {"model": args.model, "bits": args.bits, "subset": subset, "n": args.n,
               "items": [{"question": it["question"], "answer": it["answer"]} for it in items],
               "runs": {}}
        for pname in args.prompts:
            fn = lambda q, p=pname: PROMPT_TEMPLATES[p](q)  # noqa: E731
            for method in args.methods:
                decode_fn = (lambda imgs, qs: predict_batch(processor, model, imgs, qs, 8)) if method == "Regular" \
                    else (lambda imgs, qs: vcd_generate_batch(processor, model, imgs, qs, max_new_tokens=8))
                pairs = []
                t_start = time.time()
                for i in range(0, len(items), args.batch_size):
                    chunk = items[i:i + args.batch_size]
                    preds = decode_fn(
                        [ROOT / it["image"] for it in chunk],
                        [fn(it["question"]) for it in chunk],
                    )
                    pairs.extend((i + j, p) for j, p in enumerate(preds))
                preds = [p for _, p in sorted(pairs)]
                m = compute_metrics(list(zip(preds, [it["answer"] for it in items])))
                echo = sum(1 for p in preds if "assistant" in p)
                key = f"{pname}/{method}"
                out["runs"][key] = {"metrics": m, "preds": preds, "echo": echo}
                print(f"[{subset}/{key}] Acc={m['accuracy']:.4f} F1={m['f1']:.4f} "
                      f"echo={echo}/{len(preds)} ({echo / len(preds):.1%}) "
                      f"time={time.time() - t_start:.0f}s", flush=True)
        # decomposition vs Regular (same prompt)
        for pname in args.prompts:
            rk, vk = f"{pname}/Regular", f"{pname}/VCD"
            if vk not in out["runs"] or rk not in out["runs"]:
                continue
            reg, vcd = out["runs"][rk]["preds"], out["runs"][vk]["preds"]
            fe = fg = be = bg = 0
            for r, s, it in zip(reg, vcd, items):
                r_ok, s_ok = yn(r) == it["answer"], yn(s) == it["answer"]
                if r_ok == s_ok:
                    continue
                echo = "assistant" in r
                if s_ok:
                    fe += echo
                    fg += (not echo)
                else:
                    be += echo
                    bg += (not echo)
            print(f"[{subset}/{pname}] VCD flips: fix_echo={fe} fix_genuine={fg} "
                  f"break_echo={be} break_genuine={bg} net={fe + fg - be - bg}", flush=True)
            out["runs"][vk]["decomp"] = {"fix_echo": fe, "fix_genuine": fg,
                                         "break_echo": be, "break_genuine": bg}
        out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"saved -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
