"""Diagnose which PAS implementation variant (if any) reproduces the paper's
AUROC: sweep attention layers and prelim definitions.

Variants per object token k (layer l, heads averaged):
  PAS_all   : sum_j A(k, j), j in (image_end, k)          -- all text before k
  PAS_gen   : sum_j A(k, j), j in [answer_start, k)       -- generated tokens only
  PAS_inst  : sum_j A(k, j), j in (image_end, answer_start) -- instruction only
  SVAR      : sum_j A(k, j), j in [0, image_end]          -- image tokens
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import sklearn.metrics as skm
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_pas_detection import build_plurals, match_object_tokens  # noqa: E402
from eval_pope import load_model  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROMPT = "Please help me describe the image in detail."
LAYERS = [0, 4, 8, 16, 24, 31]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=int, default=100)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--bits", type=int, default=4, choices=[4, 8, 16])
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

    print(f"loading model ({args.bits}-bit)...", flush=True)
    processor, model = load_model(str(ROOT / "models" / "llava-1.5-7b-hf"), bits=args.bits)
    lm = model.model.language_model

    captured = {l: [] for l in LAYERS}
    handles = []
    for l in LAYERS:
        def make_hook(layer):
            def hook(module, args, kwargs, output):
                w = output[1] if isinstance(output, (tuple, list)) else None
                if w is not None and isinstance(w, torch.Tensor):
                    captured[layer].append(w.detach().float().cpu())
            return hook
        handles.append(lm.layers[l].self_attn.register_forward_hook(make_hook(l), with_kwargs=True))

    records = []
    bs = args.batch_size
    for start in range(0, len(items), bs):
        chunk = items[start:start + bs]
        imgs = [Image.open(ROOT / local).convert("RGB") for _, local in chunk]
        msgs = [[{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}] for _ in chunk]
        texts = [processor.apply_chat_template(m, add_generation_prompt=True) for m in msgs]
        inputs = processor(text=texts, images=imgs, padding=True, return_tensors="pt").to(model.device)
        input_len = inputs["input_ids"].shape[1]
        for l in LAYERS:
            captured[l].clear()
        with torch.inference_mode():
            out = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                pixel_values=inputs["pixel_values"],
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
                num_beams=1,
            )
        gen_ids = out[:, input_len:]
        image_end = int((inputs["input_ids"][0] == model.config.image_token_id).nonzero(as_tuple=False).max())
        for b in range(len(chunk)):
            cid = chunk[b][0]
            present = img_present.get(cid, set())
            raw_tokens = processor.tokenizer.convert_ids_to_tokens(gen_ids[b].tolist())
            token_texts = [t.replace("\u2581", " ") for t in raw_tokens]
            caption = "".join(token_texts)
            for name, tok_idx in match_object_tokens(caption, token_texts, variants):
                if tok_idx >= len(captured[0]):
                    continue
                rec = {"coco_id": cid, "object": name, "token_idx": tok_idx,
                       "label": int(name in present)}
                for l in LAYERS:
                    att = captured[l][tok_idx][b][0]  # [H, L]
                    k_pos = input_len + tok_idx
                    def mass(lo, hi):
                        if hi <= lo:
                            return 0.0
                        return float(att[:, lo:hi].sum(dim=1).mean())
                    rec[f"L{l}_pas_all"] = mass(image_end + 1, k_pos)
                    rec[f"L{l}_pas_gen"] = mass(input_len, k_pos)
                    rec[f"L{l}_pas_inst"] = mass(image_end + 1, input_len)
                    rec[f"L{l}_svar"] = mass(0, image_end + 1)
                    # Official PAS convention (github.com/lanl/pas): the
                    # attention row of the token *before* the object mention,
                    # summed over positions after the prompt (generated tokens,
                    # causally including the row's own position).
                    if tok_idx >= 1:
                        att_prev = captured[l][tok_idx - 1][b][0]
                        rec[f"L{l}_pas_official"] = float(
                            att_prev[:, input_len:k_pos].sum(dim=1).mean())
                        rec[f"L{l}_pas_official_alltext"] = float(
                            att_prev[:, image_end + 1:k_pos].sum(dim=1).mean())
                    else:
                        rec[f"L{l}_pas_official"] = float("nan")
                        rec[f"L{l}_pas_official_alltext"] = float("nan")
                records.append(rec)
        print(f"  batch {start // bs + 1}/{(len(items) + bs - 1) // bs} "
              f"records={len(records)}", flush=True)

    for h in handles:
        h.remove()

    labels = np.array([r["label"] for r in records])
    print(f"\ntokens={len(records)} pos={int(labels.sum())} neg={int((1 - labels).sum())}")
    print("\n=== AUROC by layer and variant ===")
    variants_list = ["pas_all", "pas_gen", "pas_inst", "svar",
                     "pas_official", "pas_official_alltext"]
    header = "layer  " + "".join(f"{v:>12s}" for v in variants_list)
    print(header)
    out_tbl = {}
    for l in LAYERS:
        row = f"{l:5d}  "
        out_tbl[l] = {}
        for v in variants_list:
            s = np.array([r[f"L{l}_{v}"] for r in records])
            ok = ~np.isnan(s)
            a = float(skm.roc_auc_score(labels[ok], s[ok])) if len(set(labels[ok].tolist())) > 1 else float("nan")
            out_tbl[l][v] = round(a, 4)
            row += f"{a:12.4f}"
        print(row)
    save = ROOT / "outputs" / f"pas_variants_{time.strftime('%Y%m%d_%H%M%S')}.json"
    save.write_text(json.dumps({"bits": args.bits, "images": len(items), "records": len(records),
                                "auroc": out_tbl}, indent=1), encoding="utf-8")
    print("saved ->", save)


if __name__ == "__main__":
    main()
