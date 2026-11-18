"""POPE evaluation with a 4-bit LLaVA-1.5-7B on RTX 2060 12G (baseline)."""

import argparse
import json
import time
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

PROMPT_TEMPLATES = {
    "plain": lambda q: q,
    "ins-yesno": lambda q: f"{q} Please answer yes or no.",
}


def load_model(model_id: str, bits: int = 4, attn_impl: str = "eager"):
    """Load the model with bits ∈ {4, 8, 16} (16 = fp16 without quantization)."""
    from transformers import AutoProcessor, BitsAndBytesConfig

    if bits == 4:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    elif bits == 8:
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    else:
        bnb_config = None
    try:
        from transformers import AutoModelForImageTextToText as ModelCls
    except ImportError:
        try:
            from transformers import AutoModelForVision2Seq as ModelCls
        except ImportError:
            from transformers import LlavaForConditionalGeneration as ModelCls

    processor = AutoProcessor.from_pretrained(model_id)
    model = ModelCls.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        attn_implementation=attn_impl,
    )
    model.eval()
    return processor, model


def build_inputs(processor, image_paths, questions):
    messages = [
        [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": q},
                ],
            }
        ]
        for q in questions
    ]
    images = [Image.open(p).convert("RGB") for p in image_paths]
    texts = [processor.apply_chat_template(m, add_generation_prompt=True) for m in messages]
    inputs = processor(text=texts, images=images, padding=True, return_tensors="pt")
    return inputs


def predict_batch(processor, model, image_paths, questions, max_new_tokens: int) -> list[str]:
    inputs = build_inputs(processor, image_paths, questions)
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)
    pixel_values = inputs["pixel_values"].to(model.device)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            num_beams=1,
        )

    # Generated tokens occupy the positions after the (padded) input length,
    # regardless of padding side. Slicing by per-row real length is wrong here.
    padded_len = output_ids.shape[1] - max_new_tokens
    results = []
    for i in range(output_ids.shape[0]):
        generated = processor.decode(output_ids[i][padded_len:], skip_special_tokens=True)
        results.append(generated.strip().lower())
    return results


def predict_one(processor, model, image_path: Path, question: str, max_new_tokens: int) -> str:
    return predict_batch(processor, model, [image_path], [question], max_new_tokens)[0]


def item_prompt_len(processor, question: str) -> int:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    return len(processor.tokenizer(text, add_special_tokens=True)["input_ids"])


def apply_prompt_template(template: str, question: str) -> str:
    return PROMPT_TEMPLATES[template](question)


def answer_is_yes(text: str) -> bool:
    return text.startswith("yes")


def compute_metrics(pairs):
    tp = sum(1 for p, a in pairs if answer_is_yes(p) and a == "yes")
    fp = sum(1 for p, a in pairs if answer_is_yes(p) and a == "no")
    fn = sum(1 for p, a in pairs if not answer_is_yes(p) and a == "yes")
    tn = sum(1 for p, a in pairs if not answer_is_yes(p) and a == "no")
    acc = (tp + tn) / max(1, len(pairs))
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    return {"accuracy": round(acc, 4), "precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(Path(__file__).resolve().parents[1] / "models" / "llava-1.5-7b-hf"))
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--subsets", nargs="+", default=["random"], choices=["random", "popular", "adversarial"])
    ap.add_argument("--limit", type=int, default=0, help="only evaluate first N items per subset (0 = all)")
    ap.add_argument("--batch-size", type=int, default=1, help="images per generation call (1 = exact single-image protocol)")
    ap.add_argument(
        "--group-by-length",
        action="store_true",
        help="sort items by prompt length before batching so every batch has zero left-padding "
        "(avoids a 4bit+fp16 NaN issue in transformers 5.9 with padded batches)",
    )
    ap.add_argument("--max-new-tokens", type=int, default=8)
    ap.add_argument("--prompt-template", default="plain", choices=list(PROMPT_TEMPLATES))
    ap.add_argument("--bits", type=int, default=4, choices=[4, 8, 16],
                    help="quantization: 4-bit NF4, 8-bit, or 16 = fp16 (no quantization)")
    args = ap.parse_args()

    root: Path = args.root
    outputs_dir = root / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {args.model}")
    t0 = time.time()
    processor, model = load_model(args.model, bits=args.bits)
    print(f"Model loaded in {time.time() - t0:.1f}s; VRAM used: {torch.cuda.memory_allocated() / 2**30:.2f} GiB")
    print(f"Prompt template: {args.prompt_template}")

    all_results = {}
    for subset in args.subsets:
        data_path = root / "data" / "pope" / f"{subset}.json"
        if not data_path.exists():
            print(f"Missing data: {data_path} (run download_pope.py first)")
            continue
        items = json.loads(data_path.read_text(encoding="utf-8"))
        if args.limit > 0:
            items = items[: args.limit]

        if args.group_by_length:
            # keep original order in the results; batches are grouped by identical prompt length
            qs = [apply_prompt_template(args.prompt_template, it["question"]) for it in items]
            lengths = [item_prompt_len(processor, q) for q in qs]
            order = sorted(range(len(items)), key=lambda i: lengths[i])
            batches = []
            for i in order:
                if batches and lengths[i] == lengths[batches[-1][-1]] and len(batches[-1]) < args.batch_size:
                    batches[-1].append(i)
                else:
                    batches.append([i])
        else:
            batches = [list(range(i, min(i + args.batch_size, len(items)))) for i in range(0, len(items), args.batch_size)]

        pairs = []
        t_start = time.time()
        for idxs in tqdm(batches, desc=f"POPE/{subset}"):
            chunk = [items[i] for i in idxs]
            preds = predict_batch(
                processor,
                model,
                [root / it["image"] for it in chunk],
                [apply_prompt_template(args.prompt_template, it["question"]) for it in chunk],
                args.max_new_tokens,
            )
            pairs.extend((pred, items[i]["answer"]) for i, pred in zip(idxs, preds))

        if args.group_by_length:
            # reorder pairs back to the original item order
            by_idx = {i: p for i, p in zip(order, pairs)}
            pairs = [by_idx[i] for i in range(len(items))]

        metrics = compute_metrics(pairs)
        metrics["num_samples"] = len(pairs)
        metrics["elapsed_sec"] = round(time.time() - t_start, 1)
        metrics["predictions"] = [
            {"question": it["question"], "pred": p, "answer": a}
            for it, (p, a) in zip(items, pairs)
        ]
        all_results[subset] = metrics
        print(f"POPE/{subset}: {metrics}")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = outputs_dir / f"pope_baseline_{stamp}.json"
    out_path.write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
