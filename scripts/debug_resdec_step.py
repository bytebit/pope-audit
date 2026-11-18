"""Inspect one ResDec decoding step: candidate pool, JSD, residual, fused logits."""

import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_pope import build_inputs, load_model  # noqa: E402
from resdec import resdec_step  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def topk_str(logits, k=8):
    vals, idxs = torch.topk(logits, k)
    return [(int(i), round(float(v), 2)) for v, i in zip(vals, idxs)]


def main() -> None:
    item_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    items = json.loads((ROOT / "data" / "pope" / "random.json").read_text(encoding="utf-8"))
    it = items[item_idx]
    print(f"item #{item_idx}: {it['question']}  answer={it['answer']}")

    processor, model = load_model(str(ROOT / "models" / "llava-1.5-7b-hf"))
    inputs = build_inputs(processor, [ROOT / it["image"]], [it["question"]])
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)
    pixel_values = inputs["pixel_values"].to(model.device)

    with torch.inference_mode():
        out = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values, use_cache=True)
    logits = out.logits[0]  # [L, V]
    mask = input_ids[0] != model.config.image_token_id
    pos = mask.nonzero(as_tuple=False).squeeze(-1)
    cur = logits[pos[-1]]
    hist = logits[pos[-16:-1]]
    print("cur top8:", topk_str(cur))
    print("history positions:", pos[-16:-1].tolist(), "len", hist.shape[0])
    for i in range(hist.shape[0]):
        print(f"  hist[{i}]: top3", topk_str(hist[i], 3))

    # manually replicate the step with detailed prints
    K, alpha, beta, W = 256, 0.5, 0.1, 15
    pool = torch.topk(cur, K).indices
    from resdec import _jsd_log
    lp = F.log_softmax(hist, dim=-1)
    lq = lp[:, pool] - lp[:, pool].logsumexp(dim=-1, keepdim=True)
    jsd = torch.stack([_jsd_log(lq[i], lq[i + 1]) for i in range(lq.shape[0] - 1)])
    print("jsd:", [round(float(x), 4) for x in jsd])
    print("valley (argmin):", int(jsd.argmin()))
    conf = -(lp[:, pool].mean(dim=-1))
    print("conf:", [round(float(x), 4) for x in conf])
    valley = int(jsd.argmin())
    conf_delta = conf[valley:]
    w = conf_delta / conf_delta.sum().clamp_min(1e-12)
    print("weights:", [round(float(x), 4) for x in w])
    res = torch.zeros_like(cur)
    res[pool] = (w[:, None] * hist[valley:][:, pool]).sum(dim=0)
    print("res top8:", topk_str(res))
    fused = (1 - alpha) * cur + alpha * res
    print("fused top8:", topk_str(fused))
    pcur = torch.softmax(cur, dim=-1)
    thresh = beta * pcur.max()
    print("head threshold:", round(float(thresh), 5))
    head = (pcur >= thresh).nonzero(as_tuple=False).squeeze(-1)
    print("head size:", head.numel(), "head top10 ids:", head[:10].tolist())
    fused2 = fused.clone()
    fused2[pcur < thresh] = float("-inf")
    print("final top8:", topk_str(fused2))
    tok = int(fused2.argmax())
    print("chosen token:", tok, repr(processor.decode([tok])))


if __name__ == "__main__":
    main()
