"""Diagnose LLaVA-OneVision garbage output: inspect processor inputs, raw
generated tokens, and the effect of attention implementation."""

import json
import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_pope import load_model  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MODEL = str(ROOT / "models" / "llava-onevision-qwen2-7b-ov-hf")


def run(model_id, bits, attn_impl):
    from transformers import AutoProcessor, BitsAndBytesConfig
    try:
        from transformers import AutoModelForImageTextToText as Cls
    except ImportError:
        from transformers import LlavaOnevisionForConditionalGeneration as Cls

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.float16,
                             bnb_4bit_use_double_quant=True) if bits == 4 else None
    proc = AutoProcessor.from_pretrained(model_id)
    model = Cls.from_pretrained(model_id, quantization_config=bnb, device_map="auto",
                                torch_dtype=torch.float16, attn_implementation=attn_impl)
    model.eval()

    items = json.loads((ROOT / "data" / "pope" / "random.json").read_text(encoding="utf-8"))[:2]
    msgs = [[{"role": "user", "content": [{"type": "image"},
             {"type": "text", "text": it["question"] + " Please answer yes or no."}]}] for it in items]
    texts = [proc.apply_chat_template(m, add_generation_prompt=True) for m in msgs]
    images = [Image.open(ROOT / it["image"]).convert("RGB") for it in items]
    inputs = proc(text=texts[0], images=[images[0]], return_tensors="pt")
    print("processor keys:", list(inputs.keys()))
    for k, v in inputs.items():
        if hasattr(v, "shape"):
            print(f"   {k}: {tuple(v.shape)} {v.dtype}")
    ids = inputs["input_ids"][0]
    print("input_len:", ids.numel(), "| first ids:", ids[:12].tolist(),
          "| last ids:", ids[-8:].tolist())

    inputs = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
    with torch.inference_mode():
        out = model.generate(**inputs, do_sample=False, max_new_tokens=12, num_beams=1)
    gen = out[0][inputs["input_ids"].shape[1]:]
    print("raw generated ids:", gen.tolist())
    print("decoded (skip special):", repr(proc.decode(gen, skip_special_tokens=True)))
    print("decoded (keep special):", repr(proc.decode(gen, skip_special_tokens=False)))


if __name__ == "__main__":
    bits = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    attn = sys.argv[2] if len(sys.argv) > 2 else "eager"
    print(f"=== bits={bits} attn={attn}")
    run(MODEL, bits, attn)
