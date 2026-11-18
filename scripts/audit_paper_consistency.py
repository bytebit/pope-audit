"""Recompute the manuscript's headline numbers from the evidence files and
compare them with the values printed in the paper.

Usage (from exp_topicA):   python scripts/audit_paper_consistency.py
Exit code 0 = every check passed (SKIPs allowed), 1 = at least one FAIL.

Parsing rule used throughout: an answer counts as "yes" iff the decoded string
starts with "yes" (the paper's first-word rule).
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"


def yn(text: str) -> str:
    return "yes" if text.startswith("yes") else "no"


def load(pattern: str, newest: bool = True):
    files = sorted(OUT.glob(pattern))
    if not files:
        return None
    return json.loads((files[-1] if newest else files[0]).read_text(encoding="utf-8"))


def load_merged(pattern: str):
    """Merge every file matching pattern for the same tag+subset.

    Runs are accumulated across files because different methods for the same
    checkpoint were sometimes saved by separate script invocations (e.g. the
    Regular runs of LLaVA-NeXT live in one file, its VCD runs in another).
    """
    files = sorted(OUT.glob(pattern))
    if not files:
        return None
    merged = None
    for f in files:
        d = json.loads(f.read_text(encoding="utf-8"))
        if merged is None:
            merged = {"tag": d["tag"], "subset": d["subset"], "n": d["n"],
                      "items": d["items"], "runs": {}}
        merged["runs"].update(d.get("runs", {}))
    return merged


def echo_rate(preds) -> float:
    return sum(1 for p in preds if "assistant" in p) / len(preds)


def accuracy(preds, golds) -> float:
    return sum(1 for p, g in zip(preds, golds) if yn(p) == g) / len(preds)


RESULTS = []


def check(label, got, want, tol):
    ok = got is not None and abs(got - want) <= tol
    RESULTS.append((ok, label, got, want, tol))


def skip(label, reason):
    RESULTS.append((None, label, None, reason, None))


def main() -> None:
    # ---------- LLaVA-1.5: echo rate and ResDec decomposition -------------
    base_ins = load("pope_baseline_20260829_001941.json")
    base_plain = load("pope_baseline_20260828_210602.json")
    resdec = load("pope_resdec_20260829_015107.json")
    if all(d is not None for d in (base_ins, base_plain, resdec)):
        for subset, want_echo in [("adversarial", 0.248), ("popular", 0.288), ("random", 0.231)]:
            preds = [x["pred"] for x in base_ins[subset]["predictions"]]
            check(f"echo rate LLaVA-1.5 ins-yesno / {subset}", echo_rate(preds), want_echo, 0.005)
        for subset, want_gain, want_fe, want_fg in [
                ("adversarial", 0.0653, 257, 27),
                ("popular", 0.0757, 260, 34),
                ("random", 0.0830, 250, 28)]:
            reg = [x["pred"] for x in base_ins[subset]["predictions"]]
            golds = [x["answer"] for x in base_ins[subset]["predictions"]]
            meth = [x["pred"] for x in resdec[subset]["predictions"]]
            gain = accuracy(meth, golds) - accuracy(reg, golds)
            fe = sum(1 for r, s, g in zip(reg, meth, golds)
                     if yn(r) != g and yn(s) == g and "assistant" in r)
            fg = sum(1 for r, s, g in zip(reg, meth, golds)
                     if yn(r) != g and yn(s) == g and "assistant" not in r)
            check(f"ResDec gain / {subset}", gain, want_gain, 0.0005)
            check(f"ResDec fix_echo / {subset}", fe, want_fe, 0)
            check(f"ResDec fix_genuine / {subset}", fg, want_fg, 0)
        for subset in ("adversarial", "popular", "random"):
            preds = [x["pred"] for x in base_plain[subset]["predictions"]]
            check(f"echo rate LLaVA-1.5 plain / {subset}", echo_rate(preds), 0.0, 0.0)
    else:
        skip("LLaVA-1.5 echo + ResDec block", "baseline or ResDec results missing")

    # ---------- ResDec plain-template pilot (quoted in Section 5.3) -------
    pilot = load("protocol_compare_20260828_225818.json")
    if pilot and "plain/ResDec-W8" in pilot.get("runs", {}):
        pr = pilot["runs"]["plain/Regular"]["preds"]
        pm = pilot["runs"]["plain/ResDec-W8"]["preds"]
        pg = [x["answer"] for x in pilot["items"]]
        pfix = sum(1 for a, b, g in zip(pr, pm, pg) if yn(a) != g and yn(b) == g)
        pbrk = sum(1 for a, b, g in zip(pr, pm, pg) if yn(a) == g and yn(b) != g)
        check("plain pilot n", pilot["n"], 300, 0)
        check("plain pilot fixed items", pfix, 7, 0)
        check("plain pilot broken items", pbrk, 2, 0)
        check("plain pilot net gain (pts)", 100 * (pfix - pbrk) / pilot["n"], 1.67, 0.02)
    else:
        skip("plain-template pilot", "missing")

    # ---------- VCD full-scale -------------------------------------------
    vcd_files = sorted(OUT.glob("vcd_full_*_*.json"))
    if vcd_files:
        vcd = {}
        for f in vcd_files:
            d = json.loads(f.read_text(encoding="utf-8"))
            vcd[d["subset"]] = d
        want = {
            ("adversarial", "ins-yesno"): (0.0647, 243, 26),
            ("popular", "ins-yesno"): (0.0700, 253, 32),
            ("random", "ins-yesno"): (0.0743, 239, 21),
            ("adversarial", "plain"): (0.0013, 0, 42),
            ("popular", "plain"): (-0.0033, 0, 33),
            ("random", "plain"): (-0.0043, 0, 29),
        }
        for (subset, prompt), (wg, wfe, wfg) in want.items():
            d = vcd.get(subset)
            if d is None or f"{prompt}/VCD" not in d["runs"]:
                skip(f"VCD {subset}/{prompt}", "run missing")
                continue
            reg = d["runs"][f"{prompt}/Regular"]["preds"]
            meth = d["runs"][f"{prompt}/VCD"]["preds"]
            golds = [it["answer"] for it in d["items"]]
            gain = accuracy(meth, golds) - accuracy(reg, golds)
            fe = sum(1 for r, s, g in zip(reg, meth, golds)
                     if yn(r) != g and yn(s) == g and "assistant" in r)
            fg = sum(1 for r, s, g in zip(reg, meth, golds)
                     if yn(r) != g and yn(s) == g and "assistant" not in r)
            check(f"VCD gain / {subset}/{prompt}", gain, wg, 0.0005)
            check(f"VCD fix_echo / {subset}/{prompt}", fe, wfe, 0)
            check(f"VCD fix_genuine / {subset}/{prompt}", fg, wfg, 0)
    else:
        skip("VCD full-scale block", "no vcd_full_*.json found")

    # ---------- additional checkpoints ------------------------------------
    model_wants = [
        ("model_llava-next-vicuna-7b_random_*.json", "ins-yesno/Regular", 0.0, 0.0),
        ("model_llava-next-vicuna-7b_adversarial_*.json", "ins-yesno/Regular", 0.0, 0.0),
        ("model_llava-onevision-qwen2-7b_random_*.json", "ins-yesno/Regular", 0.0, 0.0),
    ]
    for pattern, key, want_echo, tol in model_wants:
        d = load_merged(pattern)
        if d is None or key not in d.get("runs", {}):
            skip(f"echo {pattern}", "run missing")
            continue
        check(f"echo {d['tag']}/{d['subset']} {key}", echo_rate(d["runs"][key]["preds"]), want_echo, tol)

    next_vcd = load_merged("model_llava-next-vicuna-7b_random_*.json")
    if next_vcd and "ins-yesno/VCD" in next_vcd.get("runs", {}):
        reg = next_vcd["runs"]["ins-yesno/Regular"]["preds"]
        meth = next_vcd["runs"]["ins-yesno/VCD"]["preds"]
        golds = [it["answer"] for it in next_vcd["items"]]
        check("LLaVA-NeXT VCD gain / random / ins-yesno",
              accuracy(meth, golds) - accuracy(reg, golds), -0.0120, 0.0005)
    else:
        skip("LLaVA-NeXT VCD gain", "VCD run not present in the saved file")

    # ---------- CHAIR ------------------------------------------------------
    chair = load("chair_*.json")
    if chair:
        want = {"Regular": (0.6000, 0.2153), "ResDec": (0.7600, 0.2858), "VCD": (0.7267, 0.2426)}
        for method, (ws, wi) in want.items():
            if method not in chair["results"]:
                skip(f"CHAIR {method}", "missing")
                continue
            per = chair["results"][method]["per_image"]
            s = sum(1 for x in per if x["hallucinated"]) / len(per)
            mentions = sum(len(x["mentions"]) for x in per)
            hall = sum(len(x["hallucinated"]) for x in per)
            check(f"CHAIR_S {method}", s, ws, 0.002)
            check(f"CHAIR_I {method}", hall / max(1, mentions), wi, 0.002)
    else:
        skip("CHAIR block", "chair_*.json missing")

    # ---------- detection --------------------------------------------------
    pas5000 = load("pas_detection_20260829_172214.json")
    if pas5000:
        for det, want in [("pas", 0.387), ("svar", 0.613), ("nll", 0.489), ("entropy", 0.487)]:
            check(f"AUROC 5000x512 / {det}", pas5000["auroc"][det], want, 0.003)
    else:
        skip("5000x512 detection block", "missing")

    pas8 = load("pas_detection_20260915_195610.json")
    if pas8:
        for det, want in [("pas", 0.413), ("svar", 0.587), ("nll", 0.521), ("entropy", 0.518)]:
            check(f"AUROC 8-bit 500x128 / {det}", pas8["auroc"][det], want, 0.003)
    else:
        skip("8-bit detection block", "missing")

    variants = load("pas_variants_*.json")
    if variants:
        # Values quoted in the text and in the attention-decomposition table.
        WANT = {
            "0":  [("pas_all", 0.4116), ("pas_gen", 0.3944), ("pas_inst", 0.6745),
                   ("svar", 0.5884), ("pas_official", 0.3466)],
            "16": [("pas_all", 0.3627), ("pas_gen", 0.3595), ("pas_inst", 0.4610),
                   ("svar", 0.6374), ("pas_official", 0.3312)],
        }
        for layer, items in WANT.items():
            for key, want in items:
                check(f"attention variants layer {layer} / {key}",
                      variants["auroc"][layer].get(key), want, 0.002)
    else:
        skip("attention-variant block", "missing")

    # ---------- report -----------------------------------------------------
    passed = sum(1 for ok, *_ in RESULTS if ok is True)
    failed = [r for r in RESULTS if r[0] is False]
    skipped = [r for r in RESULTS if r[0] is None]
    print(f"checks: {passed} passed, {len(failed)} failed, {len(skipped)} skipped\n")
    print("| status | check | recomputed | in paper | tol |")
    print("| --- | --- | --- | --- | --- |")
    for ok, label, got, want, tol in RESULTS:
        if ok is None:
            print(f"| SKIP | {label} | - | {want} | - |")
        else:
            status = "PASS" if ok else "**FAIL**"
            got_s = f"{got:.4f}" if isinstance(got, float) else str(got)
            want_s = f"{want:.4f}" if isinstance(want, float) else str(want)
            print(f"| {status} | {label} | {got_s} | {want_s} | {tol} |")
    if failed:
        print("\nFAILING CHECKS — the manuscript or the evidence must be updated:")
        for _, label, got, want, tol in failed:
            print(f"  - {label}: recomputed {got}, paper says {want} (tol {tol})")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
