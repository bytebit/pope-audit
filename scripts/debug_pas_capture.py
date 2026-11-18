"""Smoke test: capture layer-0 attention during caption generation and
compute the PAS score for object tokens on a single image."""

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_pope import load_model  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    items = json.loads((ROOT / "data" / "pope" / "random.json").read_text(encoding="utf-8"))
    it = items[idx * 6]
    img_path = ROOT / it["image"]

    processor, model = load_model(str(ROOT / "models" / "llava-1.5-7b-hf"))
    lm = model.model.language_model
    print("lm type:", type(lm))
    inner = lm.model if hasattr(lm, "model") else lm
    print("lm.model:", type(inner), "layers:", len(inner.layers))
    attn = inner.layers[0].self_attn
    print("layer0 attn:", type(attn))

    captured = []

    def hook(module, args, kwargs, output):
        # output may be tuple (attn_output, attn_weights, past_kv) or (attn_output, attn_weights)
        attn_weights = output[1] if isinstance(output, (tuple, list)) else None
        if attn_weights is not None and isinstance(attn_weights, torch.Tensor):
            captured.append(attn_weights.detach().float().cpu())

    handle = attn.register_forward_hook(hook, with_kwargs=True)

    from PIL import Image
    messages = [
        [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Please help me describe the image in detail."},
                ],
            }
        ]
    ]
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=text, images=[Image.open(img_path).convert("RGB")],
                       padding=True, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=24,
            num_beams=1,
            output_attentions=True,
            return_dict_in_generate=True,
        )
    handle.remove()
    print("captured attention tensors:", len(captured))
    if captured:
        shapes = sorted(set(str(c.shape) for c in captured))
        print("shapes:", shapes[:6], "..." if len(shapes) > 6 else "")
    gen = processor.decode(out.sequences[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print("caption:", gen)


if __name__ == "__main__":
    main()
