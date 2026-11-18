"""CHAIR evaluation (free-form captioning) for Regular / ResDec / VCD on the
COCO images used by POPE.

CHAIR_S = fraction of captions containing >=1 hallucinated object class
CHAIR_I = hallucinated object mentions / all object mentions
Ground truth: COCO instance annotations (80 classes), longest-match-first
matching, identical to the detection pipeline.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_pas_detection import build_plurals, match_object_tokens  # noqa: E402
from eval_pope import load_model  # noqa: E402
from resdec import resdec_generate_batch  # noqa: E402
from vcd import vcd_generate_batch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROMPT = "Please help me describe the image in detail."


def mentions_of(caption_ids, tokenizer, variants):
    raw = tokenizer.convert_ids_to_tokens(caption_ids.tolist())
    token_texts = [t.replace("\u2581", " ") for t in raw]
    caption = "".join(token_texts)
    return [name for name, _ in match_object_tokens(caption, token_texts, variants)], caption


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=int, default=300)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--methods", nargs="+", default=["Regular", "ResDec", "VCD"])
    args = ap.parse_args()

    image_map = json.loads((ROOT / "data" / "detection" / "coco_image_map.json").read_text(encoding="utf-8"))
    ann = json.loads((ROOT / "data" / "detection" / "instances_val2014.json").read_text(encoding="utf-8"))
    class_names = [c["name"] for c in ann["categories"]]
    variants = build_plurals(class_names)
    cat_id2name = {c["id"]: c["name"] for c in ann["categories"]}
    img_present = {}
    for a in ann["annotations"]:
        img_present.setdefault(a["image_id"], set()).add(cat_id2name[a["category_id"]])

    by_coco = {}
    for local, cid in image_map.items():
        by_coco.setdefault(cid, local)
    items = sorted(by_coco.items())[: args.images]
    print(f"images: {len(items)}", flush=True)

    processor, model = load_model(str(ROOT / "models" / "llava-1.5-7b-hf"), bits=4)
    results = {}
    for method in args.methods:
        t0 = time.time()
        caps = []
        bs = args.batch_size
        for start in tqdm(range(0, len(items), bs), desc=method):
            chunk = items[start:start + bs]
            imgs = [ROOT / local for _, local in chunk]
            qs = [PROMPT] * len(chunk)
            if method == "Regular":
                from eval_pope import build_inputs
                inputs = build_inputs(processor, imgs, qs)
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                with torch.inference_mode():
                    out = model.generate(**inputs, do_sample=False,
                                         max_new_tokens=args.max_new_tokens, num_beams=1)
                gen = out[:, inputs["input_ids"].shape[1]:]
            elif method == "ResDec":
                # resdec helper returns text; re-tokenize not needed if we score text
                texts = resdec_generate_batch(processor, model, imgs, qs,
                                              max_new_tokens=args.max_new_tokens,
                                              alpha=0.5, beta=0.1, K=256, W=8)
                caps.extend(texts)
                continue
            else:
                texts = vcd_generate_batch(processor, model, imgs, qs,
                                           max_new_tokens=args.max_new_tokens)
                caps.extend(texts)
                continue
            for b in range(gen.shape[0]):
                txt = processor.decode(gen[b], skip_special_tokens=True)
                caps.append(txt.strip().lower())
        # score
        n_cap_with_hall = 0
        n_hall = n_mentions = 0
        per_image = []
        for (cid, local), cap in zip(items, caps):
            present = img_present.get(cid, set())
            # character-level token list: keeps char->token offsets aligned with
            # the caption so no match is dropped (we only use the class names)
            tok_texts = list(cap)
            ment = [name for name, _ in match_object_tokens(cap, tok_texts, variants)]
            hall = [m for m in ment if m not in present]
            n_mentions += len(ment)
            n_hall += len(hall)
            n_cap_with_hall += int(len(hall) > 0)
            per_image.append({"coco_id": cid, "caption": cap, "mentions": ment, "hallucinated": hall})
        results[method] = {
            "chair_s": round(n_cap_with_hall / len(caps), 4),
            "chair_i": round(n_hall / max(1, n_mentions), 4),
            "mentions_per_caption": round(n_mentions / len(caps), 2),
            "caption_len_tokens": round(sum(len(c.split()) for c in caps) / len(caps), 1),
            "n_captions": len(caps),
            "per_image": per_image,
            "time_sec": round(time.time() - t0, 1),
        }
        r = results[method]
        print(f"[{method}] CHAIR_S={r['chair_s']:.4f} CHAIR_I={r['chair_i']:.4f} "
              f"mentions/cap={r['mentions_per_caption']} time={r['time_sec']:.0f}s", flush=True)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = ROOT / "outputs" / f"chair_{stamp}.json"
    out.write_text(json.dumps({"images": len(items), "prompt": PROMPT,
                               "max_new_tokens": args.max_new_tokens,
                               "results": results}, ensure_ascii=False), encoding="utf-8")
    print("saved ->", out)


if __name__ == "__main__":
    main()
