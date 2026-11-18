"""Model-agnostic POPE echo / mitigation evaluation.

Works for any HF-native LLaVA-family checkpoint (LLaVA-1.5, LLaVA-NeXT,
LLaVA-OneVision) with AutoProcessor + AutoModelForImageTextToText:
  - Regular generation (greedy)
  - VCD (two KV-cache branches; extra processor keys are passed on prefill only)
Reports accuracy/F1 and the ASSISTANT-echo rate per prompt template, and the
VCD flip decomposition when both methods are run.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_pope import PROMPT_TEMPLATES, compute_metrics, load_model  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def build_inputs(processor, image_paths, questions):
    messages = [
        [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q}]}]
        for q in questions
    ]
    images = [Image.open(p).convert("RGB") for p in image_paths]
    texts = [processor.apply_chat_template(m, add_generation_prompt=True) for m in messages]
    return processor(text=texts, images=images, padding=True, return_tensors="pt")


def regular_generate(processor, model, image_paths, questions, max_new_tokens=8):
    inputs = build_inputs(processor, image_paths, questions)
    inputs = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[1]
    with torch.inference_mode():
        out = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens, num_beams=1)
    return [
        processor.decode(out[i][input_len:], skip_special_tokens=True).strip().lower()
        for i in range(out.shape[0])
    ]


def vcd_generate(processor, model, image_paths, questions, max_new_tokens=8,
                 alpha=1.0, beta=0.1, noise_std=0.1):
    inputs = build_inputs(processor, image_paths, questions)
    inputs = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    B = input_ids.shape[0]
    pixel_key = "pixel_values" if "pixel_values" in inputs else None
    if pixel_key is None:
        raise RuntimeError("no pixel_values in processor output")

    noise = torch.randn_like(inputs[pixel_key]) * noise_std
    pre_o = dict(inputs)
    pre_d = dict(inputs)
    pre_d[pixel_key] = inputs[pixel_key] + noise

    with torch.inference_mode():
        out = model(**pre_o, use_cache=True)
        past_o, cur_o = out.past_key_values, out.logits[:, -1, :]
        out_d = model(**pre_d, use_cache=True)
        past_d, cur_d = out_d.past_key_values, out_d.logits[:, -1, :]

    gen = [[] for _ in range(B)]
    for _ in range(max_new_tokens):
        p = torch.softmax(cur_o, dim=-1)
        head = p >= beta * p.max(dim=-1, keepdim=True).values
        contrast = (1.0 + alpha) * cur_o - alpha * cur_d
        contrast = contrast.masked_fill(~head, float("-inf"))
        tok = contrast.argmax(dim=-1)
        for b in range(B):
            gen[b].append(int(tok[b]))
        next_ids = tok.unsqueeze(1)
        attention_mask = torch.cat(
            [attention_mask, torch.ones(B, 1, dtype=attention_mask.dtype, device=attention_mask.device)], dim=1)
        with torch.inference_mode():
            out = model(input_ids=next_ids, attention_mask=attention_mask,
                        past_key_values=past_o, use_cache=True)
            past_o, cur_o = out.past_key_values, out.logits[:, -1, :]
            out_d = model(input_ids=next_ids, attention_mask=attention_mask,
                          past_key_values=past_d, use_cache=True)
            past_d, cur_d = out_d.past_key_values, out_d.logits[:, -1, :]
    return [
        processor.decode(torch.tensor(gen[b], device=model.device), skip_special_tokens=True).strip().lower()
        for b in range(B)
    ]


def yn(t: str) -> str:
    return "yes" if t.startswith("yes") else "no"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True, help="short name used in output files")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--subsets", nargs="+", default=["random", "adversarial"])
    ap.add_argument("--prompts", nargs="+", default=["plain", "ins-yesno"])
    ap.add_argument("--methods", nargs="+", default=["Regular", "VCD"])
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--attn-impl", default="eager", help="eager | sdpa (OneVision needs sdpa)")
    args = ap.parse_args()

    print(f"loading {args.model} ({args.bits}-bit)", flush=True)
    processor, model = load_model(args.model, bits=args.bits, attn_impl=args.attn_impl)
    print("model loaded", flush=True)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    for subset in args.subsets:
        items = json.loads((ROOT / "data" / "pope" / f"{subset}.json").read_text(encoding="utf-8"))[: args.n]
        out = {"model": args.model, "tag": args.tag, "bits": args.bits, "subset": subset, "n": args.n,
               "items": [{"question": it["question"], "answer": it["answer"]} for it in items],
               "runs": {}}
        for pname in args.prompts:
            fn = lambda q, p=pname: PROMPT_TEMPLATES[p](q)  # noqa: E731
            for method in args.methods:
                decode = regular_generate if method == "Regular" else vcd_generate
                preds = []
                t0 = time.time()
                for i in range(0, len(items), args.batch_size):
                    chunk = items[i:i + args.batch_size]
                    preds.extend(decode(processor, model,
                                        [ROOT / it["image"] for it in chunk],
                                        [fn(it["question"]) for it in chunk]))
                m = compute_metrics(list(zip(preds, [it["answer"] for it in items])))
                echo = sum(1 for p in preds if "assistant" in p)
                key = f"{pname}/{method}"
                out["runs"][key] = {"metrics": m, "preds": preds, "echo": echo}
                print(f"[{args.tag}/{subset}/{key}] Acc={m['accuracy']:.4f} F1={m['f1']:.4f} "
                      f"echo={echo}/{len(preds)} ({echo / len(preds):.1%}) "
                      f"time={time.time() - t0:.0f}s", flush=True)
        for pname in args.prompts:
            rk, vk = f"{pname}/Regular", f"{pname}/VCD"
            if rk not in out["runs"] or vk not in out["runs"]:
                continue
            reg, vcd = out["runs"][rk]["preds"], out["runs"][vk]["preds"]
            fe = fg = be = bg = 0
            for r, s, it in zip(reg, vcd, items):
                r_ok, s_ok = yn(r) == it["answer"], yn(s) == it["answer"]
                if r_ok == s_ok:
                    continue
                echo = "assistant" in r
                fe += int(echo and s_ok)
                fg += int((not echo) and s_ok)
                be += int(echo and not s_ok)
                bg += int((not echo) and not s_ok)
            out["runs"][vk]["decomp"] = {"fix_echo": fe, "fix_genuine": fg,
                                         "break_echo": be, "break_genuine": bg}
            print(f"[{args.tag}/{subset}/{pname}] VCD flips: fix_echo={fe} fix_genuine={fg} "
                  f"break_echo={be} break_genuine={bg} net={fe + fg - be - bg}", flush=True)
        path = ROOT / "outputs" / f"model_{args.tag}_{subset}_{stamp}.json"
        path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
        print("saved ->", path, flush=True)


if __name__ == "__main__":
    main()
