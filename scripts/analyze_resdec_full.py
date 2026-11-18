"""Full ResDec reproduction analysis: dual parsing, echo-artifact decomposition,
and comparison with the paper's Table 1 numbers."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PAPER = {
    "adversarial": {"reg": (76.03, 76.26), "res": (84.50, 83.69)},
    "popular": {"reg": (79.98, 79.34), "res": (86.97, 87.22)},
    "random": {"reg": (83.49, 82.28), "res": (90.21, 89.87)},
}


def parse_ss(t: str) -> str:
    return "yes" if t.startswith("yes") else "no"


def parse_official(t: str) -> str:
    if "." in t:
        t = t.split(".")[0]
    t = t.replace(",", "")
    words = t.split(" ")
    return "no" if any(w in ("No", "no", "not") for w in words) else "yes"


def metrics(preds, golds, parse):
    tp = sum(1 for p, a in zip(preds, golds) if parse(p) == "yes" and a == "yes")
    fp = sum(1 for p, a in zip(preds, golds) if parse(p) == "yes" and a == "no")
    fn = sum(1 for p, a in zip(preds, golds) if parse(p) == "no" and a == "yes")
    tn = sum(1 for p, a in zip(preds, golds) if parse(p) == "no" and a == "no")
    acc = (tp + tn) / len(preds)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return acc, prec, rec, f1


def main() -> None:
    reg_file = sorted((ROOT / "outputs").glob("pope_baseline_20260829_*.json"))[-1]
    res_file = sorted((ROOT / "outputs").glob("pope_resdec_20260829_*.json"))[-1]
    reg_data = json.loads(reg_file.read_text(encoding="utf-8"))
    res_data = json.loads(res_file.read_text(encoding="utf-8"))

    print(f"Regular file: {reg_file.name}  |  ResDec file: {res_file.name}\n")
    print("=== Subset comparison (Acc/F1, startswith parsing) ===")
    print(f"{'subset':12s} {'Reg ours':>18s} {'Reg paper':>18s} {'Res ours':>18s} "
          f"{'Res paper':>18s} {'gain ours':>10s} {'gain paper':>11s}")
    for subset in ["adversarial", "popular", "random"]:
        rp = [x["pred"] for x in reg_data[subset]["predictions"]]
        sp = [x["pred"] for x in res_data[subset]["predictions"]]
        golds = [x["answer"] for x in reg_data[subset]["predictions"]]
        a_reg, _, _, f_reg = metrics(rp, golds, parse_ss)
        a_res, _, _, f_res = metrics(sp, golds, parse_ss)
        pr, pf = PAPER[subset]["reg"]
        sr, sf = PAPER[subset]["res"]
        print(f"{subset:12s} {a_reg*100:6.2f}/{f_reg*100:6.2f} "
              f"{pr:6.2f}/{pf:6.2f} "
              f"{a_res*100:6.2f}/{f_res*100:6.2f} "
              f"{sr:6.2f}/{sf:6.2f} "
              f"{(a_res-a_reg)*100:+6.2f} {sr-pr:+7.2f}")
    # averages
    ra = sum(metrics([x["pred"] for x in reg_data[s]["predictions"]],
                     [x["answer"] for x in reg_data[s]["predictions"]], parse_ss)[0]
             for s in ["adversarial", "popular", "random"]) / 3
    sa = sum(metrics([x["pred"] for x in res_data[s]["predictions"]],
                     [x["answer"] for x in res_data[s]["predictions"]], parse_ss)[0]
             for s in ["adversarial", "popular", "random"]) / 3
    print(f"{'AVG':12s} {ra*100:6.2f}/--  {'':>13s} {sa*100:6.2f}/--  {'':>13s} "
          f"{(sa-ra)*100:+6.2f} +7.40")

    print("\n=== Dual-parsing sensitivity (ours, Acc/F1) ===")
    print(f"{'subset':12s} {'Reg/ss':>16s} {'Reg/off':>16s} {'Res/ss':>16s} {'Res/off':>16s}")
    for subset in ["adversarial", "popular", "random"]:
        rp = [x["pred"] for x in reg_data[subset]["predictions"]]
        sp = [x["pred"] for x in res_data[subset]["predictions"]]
        golds = [x["answer"] for x in reg_data[subset]["predictions"]]
        a1, _, _, f1_ = metrics(rp, golds, parse_ss)
        a2, _, _, f2 = metrics(rp, golds, parse_official)
        a3, _, _, f3 = metrics(sp, golds, parse_ss)
        a4, _, _, f4 = metrics(sp, golds, parse_official)
        print(f"{subset:12s} {a1*100:7.2f}/{f1_*100:6.2f} {a2*100:7.2f}/{f2*100:6.2f} "
              f"{a3*100:7.2f}/{f3*100:6.2f} {a4*100:7.2f}/{f4*100:6.2f}")

    print("\n=== Error decomposition (Regular, startswith view) ===")
    for subset in ["adversarial", "popular", "random"]:
        rp = [x["pred"] for x in reg_data[subset]["predictions"]]
        golds = [x["answer"] for x in reg_data[subset]["predictions"]]
        echo = sum(1 for p in rp if "assistant" in p)
        wrong = sum(1 for p, a in zip(rp, golds) if parse_ss(p) != a)
        wrong_echo = sum(1 for p, a in zip(rp, golds)
                         if parse_ss(p) != a and "assistant" in p)
        wrong_genuine = wrong - wrong_echo
        print(f"  {subset:12s} wrong={wrong:5d} | echo-artifact wrong={wrong_echo:5d} "
              f"| genuine wrong={wrong_genuine:5d} | echo total={echo:5d}")

    print("\n=== ResDec fix/break decomposition (vs Regular, startswith view) ===")
    for subset in ["adversarial", "popular", "random"]:
        rp = [x["pred"] for x in reg_data[subset]["predictions"]]
        sp = [x["pred"] for x in res_data[subset]["predictions"]]
        golds = [x["answer"] for x in reg_data[subset]["predictions"]]
        fix_echo = fix_gen = brk_echo = brk_gen = 0
        for r, s, a in zip(rp, sp, golds):
            r_ok = parse_ss(r) == a
            s_ok = parse_ss(s) == a
            if r_ok == s_ok:
                continue
            is_echo = "assistant" in r
            if s_ok:
                if is_echo:
                    fix_echo += 1
                else:
                    fix_gen += 1
            else:
                if is_echo:
                    brk_echo += 1
                else:
                    brk_gen += 1
        print(f"  {subset:12s} fix_echo={fix_echo:4d} fix_genuine={fix_gen:4d} "
              f"break_echo={brk_echo:3d} break_genuine={brk_gen:3d} "
              f"| net={fix_echo + fix_gen - brk_echo - brk_gen:+4d}")


if __name__ == "__main__":
    main()
