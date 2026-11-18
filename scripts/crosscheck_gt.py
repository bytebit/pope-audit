"""Cross-check detection negatives against POPE's own ground truth:
how many objects labeled 'absent' by COCO instances are marked 'present'
by POPE's yes-questions for the same image (GT disagreement)."""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    files = sorted((ROOT / "outputs").glob("pas_detection_*.json"))
    d = json.loads(files[-1].read_text(encoding="utf-8"))
    image_map = json.loads((ROOT / "data" / "detection" / "coco_image_map.json").read_text(encoding="utf-8"))
    local2coco = {local: str(cid) for local, cid in image_map.items()}
    pope_present = {}
    pat = re.compile(r"^Is there an? (.+) in the ima?ge\?$")
    for subset in ["random", "popular", "adversarial"]:
        items = json.loads((ROOT / "data" / "pope" / f"{subset}.json").read_text(encoding="utf-8"))
        for it in items:
            if it["answer"] != "yes":
                continue
            key = local2coco.get(it["image"])
            m = pat.match(it["question"])
            if key and m:
                pope_present.setdefault(key, set()).add(m.group(1).lower())

    neg = [r for r in d["records"] if r["label"] == 0]
    agree = sum(1 for r in neg if r["object"] in pope_present.get(str(r["coco_id"]), set()))
    print(f"negatives={len(neg)}  POPE-says-present={agree} ({agree / len(neg):.1%})")
    for r in neg[:15]:
        pp = r["object"] in pope_present.get(str(r["coco_id"]), set())
        print(f"  coco {r['coco_id']} {r['object']:16s} pope_present={pp} "
              f"pas={r['pas']:.3f} nll={r['nll']:.3f}")


if __name__ == "__main__":
    main()
