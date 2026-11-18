"""PAS-style object-hallucination detection on the 500-image COCO-POPE subset.

Protocol (following PAS, CVPR 2026):
  - prompt: "Please help me describe the image in detail." (plain, greedy)
  - capture layer-0 attention + per-step scores during caption generation
  - CHAIR-style string matching to find object-class token spans
  - detectors: PAS (prelim attention), SVAR (image attention), NLL, Entropy
  - AUROC over object tokens labeled present/absent from COCO instance GT
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_pope import load_model  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROMPT = "Please help me describe the image in detail."


def build_plurals(names):
    variants = {}
    for n in names:
        last = n.split()[-1]
        if last.endswith("s") or last.endswith("x") or last.endswith("ch") or last.endswith("sh"):
            plural = n + "es"
        elif last.endswith("y"):
            plural = n[:-1] + "ies"
        else:
            plural = n + "s"
        variants[n] = {n.lower(), plural.lower()}
    variants["person"].add("people")
    return variants


def match_object_tokens(caption_text, token_texts, variants):
    """Longest-match-first object detection with overlap removal.

    Multi-word classes are matched before their single-word substrings
    (e.g. "teddy bear" before "bear"); overlapping spans are skipped.
    Returns list of (object_name, last_token_idx).
    """
    norm = caption_text.lower()
    cands = sorted(
        ((name, v) for name, vset in variants.items() for v in vset),
        key=lambda x: (-len(x[1]), x[0]),
    )
    spans = []  # (start, end, name)
    for name, v in cands:
        start = 0
        while True:
            pos = norm.find(v, start)
            if pos < 0:
                break
            before_ok = pos == 0 or not norm[pos - 1].isalnum()
            end = pos + len(v)
            after_ok = end >= len(norm) or not norm[end].isalnum()
            if before_ok and after_ok and not any(
                pos < e and end > s for s, e, _ in spans
            ):
                spans.append((pos, end, name))
                start = end
                continue
            start = pos + 1
    out = []
    for _, end, name in spans:
        cum = 0
        tok_idx = None
        for i, t in enumerate(token_texts):
            cum += len(t)
            if cum >= end:
                tok_idx = i
                break
        if tok_idx is not None:
            out.append((name, tok_idx))
    return out


def compute_metrics(records):
    import sklearn.metrics as skm
    out = {}
    labels = np.array([r["label"] for r in records], dtype=int)
    for det in ["pas", "svar", "nll", "entropy"]:
        scores = np.array([r[det] for r in records], dtype=float)
        if len(set(scores.tolist())) < 2 or len(set(labels.tolist())) < 2:
            out[det] = None
            continue
        out[det] = round(float(skm.roc_auc_score(labels, scores)), 4)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="max images (0 = all 500)")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--coco-ids-file", type=str, default="",
                    help="JSON list of coco image ids; overrides the POPE map")
    ap.add_argument("--image-dir", type=str, default="",
                    help="directory containing {coco_id}.jpg for coco-ids-file runs")
    ap.add_argument("--checkpoint", type=str, default="",
                    help="resume from this checkpoint json and continue appending")
    ap.add_argument("--save-every", type=int, default=0,
                    help="save incremental checkpoint every N batches (0 = off)")
    ap.add_argument("--bits", type=int, default=4, choices=[4, 8, 16],
                    help="quantization: 4-bit NF4, 8-bit, or fp16")
    args = ap.parse_args()

    image_map = json.loads((ROOT / "data" / "detection" / "coco_image_map.json").read_text(encoding="utf-8"))
    ann = json.loads((ROOT / "data" / "detection" / "instances_val2014.json").read_text(encoding="utf-8"))
    class_names = [c["name"] for c in ann["categories"]]
    variants = build_plurals(class_names)
    print(f"COCO classes: {len(class_names)}")
    # per-image present-object sets (lazy, from COCO instance annotations)
    cat_id2name = {c["id"]: c["name"] for c in ann["categories"]}
    img_present = {}
    for a in ann["annotations"]:
        img_present.setdefault(a["image_id"], set()).add(cat_id2name[a["category_id"]])

    def get_present(cid):
        return img_present.get(cid, set())

    # unique images: either from a coco-id list + image dir, or POPE map
    if args.coco_ids_file:
        coco_ids = json.loads(Path(args.coco_ids_file).read_text(encoding="utf-8"))
        img_dir = Path(args.image_dir)
        items = []
        for cid in coco_ids:
            p = img_dir / f"{cid}.jpg"
            if not p.exists():
                # fall back to local POPE image
                local = None
                for loc, c in image_map.items():
                    if int(c) == int(cid):
                        local = ROOT / loc
                        break
                if local is not None:
                    p = local
            if p.exists():
                items.append((int(cid), str(p)))
        items.sort()
    else:
        by_coco = {}
        for local, cid in image_map.items():
            by_coco.setdefault(cid, local)
        items = sorted(by_coco.items())
    if args.limit:
        items = items[: args.limit]
    print(f"images to run: {len(items)}")

    print("Loading model...")
    t0 = time.time()
    processor, model = load_model(str(ROOT / "models" / "llava-1.5-7b-hf"), bits=args.bits)
    print(f"model loaded in {time.time() - t0:.1f}s")

    attn = model.model.language_model.layers[0].self_attn
    captured = []

    def hook(module, args, kwargs, output):
        w = output[1] if isinstance(output, (tuple, list)) else None
        if w is not None and isinstance(w, torch.Tensor):
            captured.append(w.detach().float().cpu())

    handle = attn.register_forward_hook(hook, with_kwargs=True)

    records = []
    done_batches = 0
    if args.checkpoint and Path(args.checkpoint).exists():
        ckpt = json.loads(Path(args.checkpoint).read_text(encoding="utf-8"))
        records = ckpt.get("records", [])
        done_batches = ckpt.get("done_batches", 0)
        print(f"resumed from checkpoint: {len(records)} records, {done_batches} batches done")
    bs = args.batch_size
    total_batches = (len(items) + bs - 1) // bs
    for bi, start in enumerate(tqdm(range(0, len(items), bs), desc="PAS detection")):
        if bi < done_batches:
            continue
        chunk = items[start:start + bs]
        imgs = [Image.open(ROOT / local).convert("RGB") for _, local in chunk]
        texts = [PROMPT] * len(chunk)
        messages = [
            [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": t}]}]
            for t in texts
        ]
        text_strs = [processor.apply_chat_template(m, add_generation_prompt=True) for m in messages]
        inputs = processor(text=text_strs, images=imgs, padding=True, return_tensors="pt").to(model.device)
        del captured[:]
        with torch.inference_mode():
            out = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                pixel_values=inputs["pixel_values"],
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
                num_beams=1,
                output_scores=True,
                return_dict_in_generate=True,
            )
        attn_rows = [c for c in captured]  # list over steps: [B,H,1,L_step]
        gen_ids = out.sequences[:, inputs["input_ids"].shape[1]:]
        scores = [s.float().cpu() for s in out.scores]  # [B,V] per step
        image_end = int((inputs["input_ids"][0] == model.config.image_token_id).nonzero(as_tuple=False).max())

        for b in range(len(chunk)):
            cid = chunk[b][0]
            present = get_present(cid)
            cap_ids = gen_ids[b]
            gen_len = int((cap_ids != processor.tokenizer.eos_token_id).sum()) if processor.tokenizer.eos_token_id is not None else int(cap_ids.numel())
            if int(cap_ids[-1]) == processor.tokenizer.eos_token_id:
                gen_len = int(cap_ids.numel()) - 1
            raw_tokens = processor.tokenizer.convert_ids_to_tokens(cap_ids.tolist())
            # SentencePiece marks spaces with U+2581; rebuild per-token text so
            # the concatenation exactly equals the decoded caption.
            token_texts = [t.replace("\u2581", " ") for t in raw_tokens]
            caption_text = "".join(token_texts)
            matches = match_object_tokens(caption_text, token_texts, variants)
            for name, tok_idx in matches:
                if tok_idx >= len(attn_rows):
                    continue
                att = attn_rows[tok_idx][b]  # [H,1,L]
                att = att[0]  # [H,L]
                pas = float(att[:, image_end + 1:].sum(dim=1).mean())
                svar = float(att[:, : image_end + 1].sum(dim=1).mean())
                logp = torch.log_softmax(scores[tok_idx], dim=-1)[b]
                nll = float(-logp[cap_ids[tok_idx]])
                p = logp.exp()
                entropy = float(-(p * logp).sum())
                records.append({
                    "coco_id": cid,
                    "object": name,
                    "token_idx": tok_idx,
                    "gen_len": gen_len,
                    "label": int(name in present),
                    "pas": pas,
                    "svar": svar,
                    "nll": nll,
                    "entropy": entropy,
                    "caption": caption_text,
                })
        if args.save_every and args.checkpoint and (bi + 1) % args.save_every == 0:
            Path(args.checkpoint).write_text(
                json.dumps({"records": records, "done_batches": bi + 1},
                           ensure_ascii=False),
                encoding="utf-8",
            )
    handle.remove()

    if args.save_every and args.checkpoint:
        Path(args.checkpoint).write_text(
            json.dumps({"records": records, "done_batches": total_batches},
                       ensure_ascii=False),
            encoding="utf-8",
        )

    metrics = compute_metrics(records)
    n_pos = sum(1 for r in records if r["label"])
    print(f"\ntokens: {len(records)}  pos={n_pos} neg={len(records) - n_pos}")
    for k, v in metrics.items():
        print(f"  AUROC {k:8s}: {v}")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = ROOT / "outputs" / f"pas_detection_{stamp}.json"
    payload = {
        "prompt": PROMPT,
        "max_new_tokens": args.max_new_tokens,
        "num_images": len(items),
        "auroc": metrics,
        "records": records,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print("saved ->", out_path)


if __name__ == "__main__":
    main()
