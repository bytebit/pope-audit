"""Locate where NaN enters the Llava forward for failing batch rows."""

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_pope import build_inputs, load_model  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def count_nan(x: torch.Tensor, label: str) -> None:
    n = int(torch.isnan(x.float()).sum())
    print(f"  {label}: shape={tuple(x.shape)} nan={n}")


def main() -> None:
    subset = "random"
    items = json.loads((ROOT / "data" / "pope" / f"{subset}.json").read_text(encoding="utf-8"))
    results = json.loads(sorted((ROOT / "outputs").glob("pope_baseline_*.json"))[-1].read_text(encoding="utf-8"))
    preds = results[subset]["predictions"]
    empty_idx = [i for i, p in enumerate(preds) if not p["pred"]]
    start = (empty_idx[0] // 8) * 8
    batch_items = items[start:start + 8]

    processor, model = load_model(str(ROOT / "models" / "llava-1.5-7b-hf"))
    inputs = build_inputs(processor, [ROOT / b["image"] for b in batch_items], [b["question"] for b in batch_items])
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)
    pixel_values = inputs["pixel_values"].to(model.device)

    count_nan(pixel_values, "pixel_values")

    with torch.inference_mode():
        vision = model.model.vision_tower(pixel_values.to(model.model.vision_tower.dtype))
    count_nan(vision.last_hidden_state, "vision last_hidden_state")

    # image features (after projection) used by the model
    with torch.inference_mode():
        feats = model.model.get_image_features(
            pixel_values=pixel_values,
            vision_feature_layer=None,
            vision_feature_select_strategy=None,
            return_dict=True,
        ).pooler_output
    if isinstance(feats, (list, tuple)):
        feats = torch.cat(feats, dim=0)
    count_nan(feats, "projected image features")
    print("  image token id:", model.config.image_token_id)
    print("  image features shape:", tuple(feats.shape))
    print("  <image> count per row:", (input_ids == model.config.image_token_id).sum(dim=1).tolist())

    hooks = []
    for idx, layer in enumerate(model.model.language_model.layers):
        if idx in (0, 1, 15, 31):
            hooks.append(layer.register_forward_hook(
                lambda m, a, o: count_nan(o[0], f"llm layer{idx} out")
            ))

    with torch.inference_mode():
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
        )
    count_nan(out.logits, "final logits")
    for h in hooks:
        h.remove()


if __name__ == "__main__":
    main()
