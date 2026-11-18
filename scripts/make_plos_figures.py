"""Generate the paper's figures from the saved result files (real data only)."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def yn(t: str) -> str:
    return "yes" if t.startswith("yes") else "no"


def qa_no_orange_on_blue(path: Path, tol: int = 12, allowed: int = 5) -> None:
    """Fail if an orange element (e.g. a legend swatch) sits on a blue bar.

    Matplotlib's default ``loc="best"`` can drop the legend inside the axes and
    on top of a bar; the blue handle then blends in while the orange one shows
    as a stray block.  This check catches that regression in the saved PNG.
    """
    from PIL import Image

    a = np.asarray(Image.open(path).convert("RGB")).astype(int)
    blue = np.abs(a - np.array([76, 114, 176])).max(axis=2) <= tol
    orange = np.abs(a - np.array([221, 132, 82])).max(axis=2) <= tol
    cols = np.where(blue.any(axis=0))[0]
    groups, start, prev = [], cols[0], cols[0]
    for c in cols[1:]:
        if c - prev > 3:
            groups.append((start, prev))
            start = c
        prev = c
    groups.append((start, prev))
    bad = []
    for x0, x1 in groups:
        ys = np.where(blue[:, x0:x1 + 1].any(axis=1))[0]
        n = orange[ys.min():ys.max() + 1, x0:x1 + 1].sum()
        if n > allowed:
            bad.append((x0, x1, int(n)))
    if bad:
        raise RuntimeError(f"orange pixels overlap blue bars in columns {bad}")
    print(f"fig1 QA: no orange overlap inside blue bars ({path.name})")


def fig1_echo_and_decomposition() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 2.55))

    # (a) echo rate by checkpoint, computed from the saved runs (ins-yesno)
    def echo_from_popesubset(path, subsets, key=None):
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        rates = []
        for s in subsets:
            rec = d[s] if key is None else d
            preds = (rec["predictions"] if key is None
                     else d["runs"][f"{key}/Regular"]["preds"])
            preds = [p["pred"] if isinstance(p, dict) else p for p in preds]
            rates.append(100 * sum(1 for p in preds if "assistant" in p) / len(preds))
        return sum(rates) / len(rates)

    def echo_from_model(pattern, subsets):
        rates = []
        for s in subsets:
            files = sorted((ROOT / "outputs").glob(pattern.format(subset=s)))
            if not files:
                continue
            d = json.loads(files[-1].read_text(encoding="utf-8"))
            if "ins-yesno/Regular" not in d.get("runs", {}):
                continue
            preds = d["runs"]["ins-yesno/Regular"]["preds"]
            rates.append(100 * sum(1 for p in preds if "assistant" in p) / len(preds))
        return sum(rates) / len(rates) if rates else 0.0

    three = ["adversarial", "popular", "random"]
    labels = ["LLaVA-1.5\n4-bit", "LLaVA-1.5\n8-bit", "LLaVA-NeXT\n4-bit",
              "OneVision\n4-bit", "Qwen2.5-VL\n4-bit"]
    rates = [
        echo_from_popesubset(ROOT / "outputs" / "pope_baseline_20260829_001941.json", three),
        echo_from_popesubset(ROOT / "outputs" / "pope_baseline_20260915_233143.json", ["random"]),
        echo_from_model("model_llava-next-vicuna-7b_{subset}_*.json", ["random", "adversarial"]),
        echo_from_model("model_llava-onevision-qwen2-7b_{subset}_*.json", ["random"]),
        echo_from_model("qwen_pope_{subset}_20260915_205435.json", ["random", "adversarial"]),
    ]
    colors = ["#4C72B0", "#4C72B0", "#55A868", "#55A868", "#DD8452"]
    axes[0].bar(labels, rates, color=colors)
    # A bar of height zero is invisible, which makes panel (a) look as if only
    # the LLaVA checkpoints were plotted. Mark every zero at the baseline so all
    # five checkpoints are visibly present without inflating the value.
    for i, r in enumerate(rates):
        if r <= 0.0:
            axes[0].scatter([i], [0.0], color=colors[i], s=34, zorder=5,
                            clip_on=False)
    for i, r in enumerate(rates):
        axes[0].text(i, r + 1.0, f"{r:.1f}%", ha="center", fontsize=8)
    axes[0].set_ylabel("echo rate (ins-yesno)")
    axes[0].set_ylim(0, 50)
    axes[0].set_title("(a) The echo appears in one checkpoint only", fontsize=9)
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].tick_params(axis="x", labelsize=8)

    # (b) mitigation gain decomposition (ResDec full-3000, per subset)
    subsets = ["adversarial", "popular", "random"]
    res = json.loads((ROOT / "outputs" / "pope_resdec_20260829_015107.json").read_text(encoding="utf-8"))
    reg = json.loads((ROOT / "outputs" / "pope_baseline_20260829_001941.json").read_text(encoding="utf-8"))
    fix_echo, fix_gen = [], []
    for s in subsets:
        rp = [x["pred"] for x in reg[s]["predictions"]]
        mp = [x["pred"] for x in res[s]["predictions"]]
        golds = [x["answer"] for x in reg[s]["predictions"]]
        fe = sum(1 for r, m, g in zip(rp, mp, golds)
                 if yn(r) != g and yn(m) == g and "assistant" in r)
        fg = sum(1 for r, m, g in zip(rp, mp, golds)
                 if yn(r) != g and yn(m) == g and "assistant" not in r)
        fix_echo.append(fe / len(rp) * 100)
        fix_gen.append(fg / len(rp) * 100)
    x = np.arange(len(subsets))
    axes[1].bar(x - 0.2, fix_echo, 0.4, label="echo repair", color="#4C72B0")
    axes[1].bar(x + 0.2, fix_gen, 0.4, label="genuine fix", color="#DD8452")
    for i in range(len(subsets)):
        axes[1].text(i - 0.2, fix_echo[i] + 0.15, f"{fix_echo[i]:.1f}", ha="center", fontsize=8)
        axes[1].text(i + 0.2, fix_gen[i] + 0.15, f"{fix_gen[i]:.1f}", ha="center", fontsize=8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([s[:9] for s in subsets])
    axes[1].set_ylabel("% of items corrected by ResDec")
    axes[1].set_title("(b) ResDec corrections are ~90% echo repair", fontsize=9)
    # Reserve headroom and keep the legend above the bars: with loc="best" it
    # was drawn over the random-subset bar, leaving a stray orange swatch.
    axes[1].set_ylim(0, 14)
    axes[1].legend(fontsize=8, frameon=False, loc="upper left", ncol=2,
                   handlelength=1.2, columnspacing=1.0, borderaxespad=0.4)
    axes[1].grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "Fig1.png", dpi=300)
    fig.savefig(FIG / "Fig1.tif", dpi=300,
                pil_kwargs={"compression": "tiff_lzw"})
    qa_no_orange_on_blue(FIG / "Fig1.png")
    plt.close(fig)


def fig2_attention_variants() -> None:
    files = sorted((ROOT / "outputs").glob("pas_variants_*.json"))   # newest run has the official-convention column
    d = json.loads(files[-1].read_text(encoding="utf-8"))
    tbl = d["auroc"]
    layers = sorted(int(k) for k in tbl)
    variants = [("pas_all", "prelim: all text (PAS as defined)", "#C44E52", "-"),
                ("pas_gen", "prelim: generated tokens only", "#8172B2", "--"),
                ("pas_official", "prelim: authors' code convention", "#937860", ":"),
                ("pas_inst", "instruction tokens", "#55A868", "-"),
                ("svar", "image tokens", "#4C72B0", "-.")]
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    for key, label, color, ls in variants:
        ys = [tbl[str(l)][key] for l in layers]
        ax.plot(layers, ys, marker="o", ms=3.5, color=color, ls=ls, label=label)
    ax.axhline(0.5, color="gray", lw=0.8, ls=":")
    ax.set_xlabel("layer")
    ax.set_ylabel("AUROC (hallucination detection)")
    ax.set_ylim(0.3, 0.75)
    ax.set_title("Where the detection signal actually lives\n(LLaVA-1.5-7B, 500 images, 128-token captions)", fontsize=9)
    ax.legend(fontsize=7.5, frameon=False, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "Fig2.png", dpi=300)
    fig.savefig(FIG / "Fig2.tif", dpi=300,
                pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


def main() -> None:
    fig1_echo_and_decomposition()
    fig2_attention_variants()
    print("figures written to", FIG)


if __name__ == "__main__":
    main()
