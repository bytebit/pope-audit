"""VCD: Visual Contrastive Decoding (Leng et al., ICLR 2024).

At each decoding step, contrast the logits from the original image with the
logits from a Gaussian-noised image, restricted to the adaptive plausible
token set (beta=0.1), then take the argmax. Default alpha=1.0.
"""

import torch

from eval_pope import build_inputs


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
    inputs = build_inputs(processor, image_paths, questions)
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)
    pixel_values = inputs["pixel_values"].to(model.device)
    B = input_ids.shape[0]

    # VCD: add Gaussian noise to the (CLIP-normalized) pixel values directly,
    # fresh noise per batch, no clamping (following the paper's implementation).
    noise = torch.randn(pixel_values.shape, device=model.device) * noise_std
    pixel_dist = pixel_values + noise

    with torch.inference_mode():
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            use_cache=True,
        )
        past_orig = out.past_key_values
        out_d = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_dist,
            use_cache=True,
        )
        past_dist = out_d.past_key_values
        cur_orig = out.logits[:, -1, :]
        cur_dist = out_d.logits[:, -1, :]

    gen_tokens = [[] for _ in range(B)]
    for _step in range(max_new_tokens):
        # adaptive plausible constraint on the original distribution
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
                use_cache=True,
            )
            cur_orig = out.logits[:, -1, :]
            past_orig = out.past_key_values
            out_d = model(
                input_ids=next_ids,
                attention_mask=attention_mask,
                past_key_values=past_dist,
                use_cache=True,
            )
            cur_dist = out_d.logits[:, -1, :]
            past_dist = out_d.past_key_values

    results = []
    for b in range(B):
        ids = torch.tensor(gen_tokens[b], dtype=torch.long, device=model.device)
        results.append(processor.decode(ids, skip_special_tokens=True).strip().lower())
    return results
