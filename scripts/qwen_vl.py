"""Qwen2.5-VL utilities: loading, batched POPE generation, and VCD decoding.

Mirrors the LLaVA helpers in eval_pope.py / vcd.py so that the same audit
pipeline (prompt templates, echo detection, flip decomposition) can be applied
to a second model family.
"""

import torch
from PIL import Image


def load_qwen(model_id: str, bits: int = 4):
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

    if bits == 4:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    elif bits == 8:
        bnb = BitsAndBytesConfig(load_in_8bit=True)
    else:
        bnb = None

    processor = AutoProcessor.from_pretrained(model_id)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=bnb,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model.eval()
    return processor, model


def build_inputs(processor, image_paths, questions):
    messages = [
        [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q}]}]
        for q in questions
    ]
    images = [Image.open(p).convert("RGB") for p in image_paths]
    texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages]
    return processor(text=texts, images=images, padding=True, return_tensors="pt")


def predict_batch(processor, model, image_paths, questions, max_new_tokens: int = 8) -> list[str]:
    inputs = build_inputs(processor, image_paths, questions)
    inputs = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[1]
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            num_beams=1,
        )
    results = []
    for i in range(out.shape[0]):
        gen = processor.batch_decode(
            [out[i][input_len:]], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        results.append(gen.strip().lower())
    return results


def vcd_generate_batch(
    processor,
    model,
    image_paths,
    questions,
    max_new_tokens: int = 8,
    alpha: float = 1.0,
    beta: float = 0.1,
    noise_std: float = 0.1,
) -> list[str]:
    """Visual contrastive decoding for Qwen2.5-VL (two KV-cache branches)."""
    inputs = build_inputs(processor, image_paths, questions)
    inputs = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    pixel_values = inputs["pixel_values"]
    grid = inputs.get("image_grid_thw")
    mm_token_type_ids = inputs.get("mm_token_type_ids")
    B = input_ids.shape[0]

    noise = torch.randn(pixel_values.shape, device=model.device, dtype=pixel_values.dtype) * noise_std
    pixel_dist = pixel_values + noise

    with torch.inference_mode():
        # mRoPE position ids: compute once for the prompt, then advance by one
        # per decode step (text tokens advance uniformly in all three dims).
        if mm_token_type_ids is not None:
            pos_ids = model.model.get_rope_index(
                input_ids, mm_token_type_ids, image_grid_thw=grid, attention_mask=attention_mask
            )[0]
        else:
            pos_ids = model.model.get_rope_index(input_ids, image_grid_thw=grid, attention_mask=attention_mask)[0]
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=grid,
            position_ids=pos_ids,
            use_cache=True,
        )
        past_orig = out.past_key_values
        cur_orig = out.logits[:, -1, :]
        out_d = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_dist,
            image_grid_thw=grid,
            position_ids=pos_ids,
            use_cache=True,
        )
        past_dist = out_d.past_key_values
        cur_dist = out_d.logits[:, -1, :]

    gen_tokens = [[] for _ in range(B)]
    step_pos = pos_ids[:, :, -1:] + 1  # [3, B, 1]
    for _step in range(max_new_tokens):
        p_orig = torch.softmax(cur_orig, dim=-1)
        head = p_orig >= beta * p_orig.max(dim=-1, keepdim=True).values
        contrast = (1.0 + alpha) * cur_orig - alpha * cur_dist
        contrast = contrast.masked_fill(~head, float("-inf"))
        tok = contrast.argmax(dim=-1).tolist()
        next_ids = torch.tensor(tok, dtype=torch.long, device=model.device).unsqueeze(1)
        for b in range(B):
            gen_tokens[b].append(tok[b])
        attention_mask = torch.cat(
            [attention_mask, torch.ones(B, 1, dtype=attention_mask.dtype, device=attention_mask.device)],
            dim=1,
        )
        with torch.inference_mode():
            out = model(
                input_ids=next_ids,
                attention_mask=attention_mask,
                past_key_values=past_orig,
                position_ids=step_pos,
                use_cache=True,
            )
            cur_orig = out.logits[:, -1, :]
            past_orig = out.past_key_values
            out_d = model(
                input_ids=next_ids,
                attention_mask=attention_mask,
                past_key_values=past_dist,
                position_ids=step_pos,
                use_cache=True,
            )
            cur_dist = out_d.logits[:, -1, :]
            past_dist = out_d.past_key_values
        step_pos = step_pos + 1

    return [
        processor.batch_decode(
            [torch.tensor(gen_tokens[b], device=model.device)], skip_special_tokens=True
        )[0].strip().lower()
        for b in range(B)
    ]
