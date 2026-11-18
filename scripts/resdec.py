"""ResDec: Residual Decoding (CVPR 2026) — history-aware residual guidance decoding.

Faithful re-implementation of "Residual Decoding: Mitigating Hallucinations in
Large Vision-Language Models via History-Aware Residual Guidance" based on the
paper's method section (Eq. 4-10). Defaults: alpha=0.5, beta=0.1 (paper default),
K (candidate pool) = 256, W (history window) = 15.
"""

import torch
import torch.nn.functional as F

from eval_pope import build_inputs


def _jsd_log(log_p: torch.Tensor, log_q: torch.Tensor) -> torch.Tensor:
    """JSD between two log-probability vectors (stable; 0*log0 handled)."""
    log_m = torch.logaddexp(log_p, log_q) - torch.log(torch.tensor(2.0, device=log_p.device))
    p = log_p.exp()
    q = log_q.exp()
    # p*(log p - log m) is 0 when p == 0 (torch: 0 * finite = 0)
    kl_pm = (p * (log_p - log_m)).sum(-1)
    kl_qm = (q * (log_q - log_m)).sum(-1)
    return 0.5 * (kl_pm + kl_qm)


def resdec_step(
    cur_logits: torch.Tensor,
    hist_logits: torch.Tensor,
    alpha: float,
    beta: float,
    K: int,
    valley_mode: str = "argmin",
) -> int:
    """Apply one ResDec step.

    cur_logits: [V] logits of the current time step.
    hist_logits: [h, V] logits of the previous time steps (h <= W).
    """
    V = cur_logits.numel()
    pool = torch.topk(cur_logits, min(K, V)).indices  # [K] candidate token pool

    # log-space probabilities (stable: no exact zeros / -inf)
    lp = F.log_softmax(hist_logits, dim=-1)  # [h, V]
    lq = lp[:, pool] - lp[:, pool].logsumexp(dim=-1, keepdim=True)  # [h, K] pool-normalized

    # JSD between adjacent steps over the candidate pool (Eq. 4)
    if lq.shape[0] >= 2:
        jsd = torch.stack([_jsd_log(lq[i], lq[i + 1]) for i in range(lq.shape[0] - 1)])
        if valley_mode == "argmin":
            valley = int(jsd.argmin())
        else:  # last minimum (rightmost bottom of the U-shape)
            valley = int(jsd.shape[0] - 1 - torch.flip(jsd, dims=[0]).argmin())
    else:
        valley = 0

    # confidence-weighted aggregation over SAP+EDP steps (Eq. 6-7).
    # C_i measures the entropy of the candidate-token distribution *within* the
    # pool (pool-normalized), matching the paper's "low entropy of the token
    # distribution within Omega_t" wording.
    conf = -(lq.mean(dim=-1))  # [h]
    conf_delta = conf[valley:]
    w = conf_delta / conf_delta.sum().clamp_min(1e-12)
    res = torch.zeros_like(cur_logits)
    res[pool] = (w[:, None] * hist_logits[valley:][:, pool]).sum(dim=0)

    # fuse history and present (Eq. 9)
    fused = (1.0 - alpha) * cur_logits + alpha * res

    # head truncation using the current distribution (Eq. 10)
    pcur = F.softmax(cur_logits, dim=-1)
    fused = fused.masked_fill(pcur < beta * pcur.max(), float("-inf"))
    return int(fused.argmax())


def resdec_generate_batch(
    processor,
    model,
    image_paths,
    questions,
    max_new_tokens: int = 8,
    alpha: float = 0.5,
    beta: float = 0.1,
    K: int = 256,
    W: int = 15,
    valley_mode: str = "argmin",
) -> list[str]:
    """Generate with ResDec for a batch (caller should group items by prompt
    length so the batch has zero left-padding; see eval_pope_resdec.py)."""
    inputs = build_inputs(processor, image_paths, questions)
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)
    pixel_values = inputs["pixel_values"].to(model.device)
    B, L = input_ids.shape
    image_token_id = model.config.image_token_id

    with torch.inference_mode():
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            use_cache=True,
        )
        logits = out.logits  # [B, L, V]
        past = out.past_key_values

        cur = []
        history = []
        for b in range(B):
            mask = input_ids[b] != image_token_id
            pos = mask.nonzero(as_tuple=False).squeeze(-1)
            if pos.numel() == 0:
                pos = torch.arange(L, device=input_ids.device)
            # last text position -> current logits of step 0;
            # the W text positions before it -> initial history (Eq. 8)
            h_idx = pos[-W - 1:-1]
            cur.append(logits[b, pos[-1]])
            history.append(logits[b, h_idx])
        cur = torch.stack(cur)  # [B, V]

    gen_tokens = [[] for _ in range(B)]
    for _step in range(max_new_tokens):
        tok = [
            resdec_step(cur[b], history[b][-W:], alpha, beta, K, valley_mode)
            for b in range(B)
        ]
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
                past_key_values=past,
                use_cache=True,
            )
            new_logits = out.logits[:, -1, :]  # [B, V]
            past = out.past_key_values

        for b in range(B):
            history[b] = torch.cat([history[b], cur[b].unsqueeze(0)], dim=0)
        cur = new_logits

    results = []
    for b in range(B):
        ids = torch.tensor(gen_tokens[b], dtype=torch.long, device=model.device)
        results.append(processor.decode(ids, skip_special_tokens=True).strip().lower())
    return results
