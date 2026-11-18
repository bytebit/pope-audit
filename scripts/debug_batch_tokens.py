"""Compare raw generated tokens for a failing batch vs single-image mode."""

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_pope import build_inputs, load_model  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    subset = "random"
    items = json.loads((ROOT / "data" / "pope" / f"{subset}.json").read_text(encoding="utf-8"))
    results = json.loads(sorted((ROOT / "outputs").glob("pope_baseline_*.json"))[-1].read_text(encoding="utf-8"))
    preds = results[subset]["predictions"]
    empty_idx = [i for i, p in enumerate(preds) if not p["pred"]]
    print("empty indices sample:", empty_idx[:20])

    processor, model = load_model(str(ROOT / "models" / "llava-1.5-7b-hf"))
    print("padding_side:", processor.tokenizer.padding_side, "| pad_id:", processor.tokenizer.pad_token_id,
          "| eos_id:", processor.tokenizer.eos_token_id)

    start = (empty_idx[0] // 8) * 8
    batch_items = items[start:start + 8]
    print(f"inspecting batch {start}-{start+7}, empty rows:",
          [i - start for i in empty_idx if start <= i < start + 8])

    # single-image generation for the failing row
    failing = empty_idx[0]
    it = items[failing]
    for mode, kwargs in [
        ("single", {"image_paths": [ROOT / it["image"]], "questions": [it["question"]]}),
        ("batch8", {
            "image_paths": [ROOT / b["image"] for b in batch_items],
            "questions": [b["question"] for b in batch_items],
        }),
    ]:
        inputs = build_inputs(processor, kwargs["image_paths"], kwargs["questions"])
        input_ids = inputs["input_ids"].to(model.device)
        attention_mask = inputs["attention_mask"].to(model.device)
        pixel_values = inputs["pixel_values"].to(model.device)
        with torch.inference_mode():
            out = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                do_sample=False,
                max_new_tokens=8,
                num_beams=1,
            )
        print(f"\n== {mode} == shape={tuple(out.shape)} input_shape={tuple(input_ids.shape)}")
        if mode == "single":
            gen = out[0][input_ids.shape[1]:]
            print("raw gen ids:", gen.tolist())
            print("decoded:", repr(processor.decode(gen, skip_special_tokens=True)))
            # also first-token logits
            with torch.inference_mode():
                logits = model(input_ids=input_ids, attention_mask=attention_mask,
                               pixel_values=pixel_values).logits[0, -1]
            top = torch.topk(logits, 5)
            print("top5 first-token:", [(processor.tokenizer.decode([i]), round(float(p), 2)) for p, i in zip(top.values, top.indices)])
        else:
            row = failing - start
            gen = out[row][input_ids.shape[1]:]
            print(f"row {row} raw gen ids:", gen.tolist())
            print(f"row {row} decoded:", repr(processor.decode(gen, skip_special_tokens=True)))
            with torch.inference_mode():
                logits = model(input_ids=input_ids, attention_mask=attention_mask,
                               pixel_values=pixel_values).logits[row, -1]
            top = torch.topk(logits, 5)
            print(f"row {row} top5 first-token:", [(processor.tokenizer.decode([i]), round(float(p), 2)) for p, i in zip(top.values, top.indices)])
            # print all rows' first generated token
            for r in range(8):
                g = out[r][input_ids.shape[1]:]
                print(f"  row {r}: first={processor.decode([g[0].item()])!r} decoded={processor.decode(g, skip_special_tokens=True)!r}")


if __name__ == "__main__":
    main()
