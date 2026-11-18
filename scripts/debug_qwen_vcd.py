"""Debug Qwen2.5-VL VCD: manual decode loop with correct mRoPE positions."""

import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qwen_vl import build_inputs, load_qwen, predict_batch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def vcd_step_forward(model, full_ids, past, attention_mask):
    past_len = past.get_seq_length()
    cache_position = torch.arange(past_len, full_ids.shape[1], device=full_ids.device)
    mi = model.prepare_inputs_for_generation(
        full_ids, past_key_values=past, attention_mask=attention_mask,
        cache_position=cache_position, use_cache=True,
    )
    return model(**mi)


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    items = json.loads((ROOT / "data" / "pope" / "random.json").read_text(encoding="utf-8"))[:n]
    processor, model = load_qwen(str(ROOT / "models" / "qwen2.5-vl-7b-instruct"), bits=4)
    imgs = [ROOT / it["image"] for it in items]
    qs = [it["question"] + " Please answer yes or no." for it in items]

    reg = predict_batch(processor, model, imgs, qs, 8)
    print("regular:", reg, flush=True)

    # single item: Qwen2.5-VL batched generation with padding is broken in this
    # transformers version (right-padding + dynamic image tokens), so use batch 1
    imgs, qs = imgs[:1], qs[:1]
    inputs = build_inputs(processor, imgs, qs)
    inputs = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
    input_ids, am = inputs["input_ids"], inputs["attention_mask"]

    t0 = time.time()
    with torch.inference_mode():
        out = model(input_ids=input_ids, attention_mask=am, pixel_values=inputs["pixel_values"],
                    image_grid_thw=inputs.get("image_grid_thw"), use_cache=True)
        past_o, cur_o = out.past_key_values, out.logits[:, -1, :]
        noise = torch.randn_like(inputs["pixel_values"]) * 0.1
        out_d = model(input_ids=input_ids, attention_mask=am,
                      pixel_values=inputs["pixel_values"] + noise,
                      image_grid_thw=inputs.get("image_grid_thw"), use_cache=True)
        past_d, cur_d = out_d.past_key_values, out_d.logits[:, -1, :]

        gen = [[] for _ in range(input_ids.shape[0])]
        full = input_ids
        for _ in range(8):
            p = torch.softmax(cur_o, dim=-1)
            head = p >= 0.1 * p.max(dim=-1, keepdim=True).values
            contrast = 2.0 * cur_o - 1.0 * cur_d
            contrast = contrast.masked_fill(~head, float("-inf"))
            tok = contrast.argmax(dim=-1)
            for b in range(len(gen)):
                gen[b].append(int(tok[b]))
            next_ids = tok.unsqueeze(1)
            full = torch.cat([full, next_ids], dim=1)
            am = torch.cat([am, torch.ones(am.shape[0], 1, dtype=am.dtype, device=am.device)], dim=1)
            out = vcd_step_forward(model, full, past_o, am)
            past_o, cur_o = out.past_key_values, out.logits[:, -1, :]
            out_d = vcd_step_forward(model, full, past_d, am)
            past_d, cur_d = out_d.past_key_values, out_d.logits[:, -1, :]
    preds = []
    for b in range(len(gen)):
        txt = processor.batch_decode([torch.tensor(gen[b], device=model.device)], skip_special_tokens=True)[0]
        preds.append(txt.strip().lower())
    print("vcd:", preds, f"time={time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
