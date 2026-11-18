"""Detailed per-item diagnosis of ResDec on POPE: history implicit inference,
pool composition, JSD curve, valley, confidence weights, residual signal."""

import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_pope import build_inputs, load_model, predict_batch  # noqa: E402
from resdec import _jsd_log  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
YES, NO = 3869, 1939


def decode_tok(processor, t: int) -> str:
    return repr(processor.decode([t]).strip())


def main() -> None:
    idxs = [int(x) for x in sys.argv[1:]] or [16, 29, 34, 43, 67, 87, 20, 5, 10]
    W = 8  # paper default
    K = 256
    alpha, beta = 0.5, 0.1

    items = json.loads((ROOT / "data" / "pope" / "random.json").read_text(encoding="utf-8"))
    processor, model = load_model(str(ROOT / "models" / "llava-1.5-7b-hf"))

    reg = predict_batch(processor, model,
                        [ROOT / items[i]["image"] for i in idxs],
                        [items[i]["question"] for i in idxs], 8)

    for k, i in enumerate(idxs):
        it = items[i]
        inputs = build_inputs(processor, [ROOT / it["image"]], [it["question"]])
        input_ids = inputs["input_ids"].to(model.device)
        attention_mask = inputs["attention_mask"].to(model.device)
        pixel_values = inputs["pixel_values"].to(model.device)
        with torch.inference_mode():
            out = model(input_ids=input_ids, attention_mask=attention_mask,
                        pixel_values=pixel_values, use_cache=True)
        logits = out.logits[0]
        mask = input_ids[0] != model.config.image_token_id
        pos = mask.nonzero(as_tuple=False).squeeze(-1)
        cur = logits[pos[-1]]
        hist = logits[pos[-W - 1:-1]]

        ids = input_ids[0][pos[-W - 1:]]
        toks = [decode_tok(processor, int(t)) for t in ids]

        pc = F.softmax(cur, dim=-1)
        pool = torch.topk(cur, K).indices
        yes_rank = int((pool == YES).nonzero(as_tuple=False).flatten()[0]) if (pool == YES).any() else None
        no_rank = int((pool == NO).nonzero(as_tuple=False).flatten()[0]) if (pool == NO).any() else None

        lp = F.log_softmax(hist, dim=-1)
        lq = lp[:, pool] - lp[:, pool].logsumexp(dim=-1, keepdim=True)
        jsd = torch.stack([_jsd_log(lq[j], lq[j + 1]) for j in range(lq.shape[0] - 1)])
        valley = int(jsd.argmin())
        conf = -(lq.mean(dim=-1))
        conf_delta = conf[valley:]
        w = conf_delta / conf_delta.sum().clamp_min(1e-12)
        res = torch.zeros_like(cur)
        res[pool] = (w[:, None] * hist[valley:][:, pool]).sum(dim=0)
        fused = (1 - alpha) * cur + alpha * res
        head = pc >= beta * pc.max()
        fused = fused.masked_fill(~head, float("-inf"))

        yes_p = float(pc[YES])
        no_p = float(pc[NO])
        print("=" * 88)
        print(f"item#{i} [{it['question']}]  answer={it['answer']}  regular={reg[k]!r}")
        print(f"  current: P(yes)={yes_p:.4f} P(no)={no_p:.4f}  pool_rank yes={yes_rank} no={no_rank}  top1={decode_tok(processor, int(cur.argmax()))}")
        print("  history positions (token -> top3):")
        for j in range(hist.shape[0]):
            vals, idxs = torch.topk(hist[j], 3)
            top3 = ", ".join(f"{decode_tok(processor, int(t))}:{float(v):.1f}" for v, t in zip(vals, idxs))
            py, pn = float(lp[j][YES]), float(lp[j][NO])
            print(f"    [{j}] {toks[j]:>18} -> {top3}   | logP(yes)={py:.2f} logP(no)={pn:.2f}")
        print(f"  jsd: {[round(float(x), 3) for x in jsd]}  valley={valley}")
        print(f"  conf: {[round(float(x), 3) for x in conf]}")
        print(f"  weights: {[round(float(x), 3) for x in w]}")
        y_res, n_res = float(res[YES]), float(res[NO])
        y_cur, n_cur = float(cur[YES]), float(cur[NO])
        y_fus, n_fus = float(fused[YES]), float(fused[NO])
        print(f"  res: yes={y_res:.2f} no={n_res:.2f} | cur: yes={y_cur:.2f} no={n_cur:.2f} | fused: yes={y_fus:.2f} no={n_fus:.2f} | argmax={decode_tok(processor, int(fused.argmax()))}")
        if yes_rank is not None and no_rank is not None:
            print(f"  pool entropy(hist)={[round(float(-lq[j].mean()), 3) for j in range(lq.shape[0])]}")


if __name__ == "__main__":
    main()
