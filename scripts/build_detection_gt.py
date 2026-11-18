"""Build per-image object ground truth for the PAS-style detection eval.

The lmms-lab/POPE parquet preserves the original POPE item order, so local
images data/pope/images/{subset}/{subset}_{i:04d}.jpg correspond 1:1 to the
original POPE JSONL entries (which carry COCO_val2014_*.jpg filenames).
This script verifies that mapping and saves:
  - data/detection/coco_image_map.json       : local file -> COCO image id
  - data/detection/pope_objects_per_image.json : per-image present/absent
    object lists derived from POPE annotations (fallback GT)
"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TMP = Path(r"C:\Users\Administrator\AppData\Local\Temp")

SUBSETS = ["random", "popular", "adversarial"]


def main() -> None:
    coco_id_pat = re.compile(r"COCO_val2014_(\d+)\.jpg$")
    mapping = {}  # local image path -> coco image id
    per_image = {}  # coco image id -> {"present": set, "absent": set}
    mismatches = 0

    for subset in SUBSETS:
        orig_path = TMP / f"coco_pope_{subset}.json"
        orig = [json.loads(l) for l in orig_path.read_text(encoding="utf-8").splitlines()]
        local = json.loads((ROOT / "data" / "pope" / f"{subset}.json").read_text(encoding="utf-8"))
        assert len(orig) == len(local), (subset, len(orig), len(local))
        for i, (o, l) in enumerate(zip(orig, local)):
            if o["text"].strip() != l["question"].strip() or o["label"].lower() != l["answer"].lower():
                mismatches += 1
                if mismatches < 6:
                    print(f"MISMATCH {subset} #{i}: orig={o} local={l}")
            m = coco_id_pat.search(o["image"])
            if not m:
                raise ValueError(f"no coco id in {o['image']}")
            coco_id = int(m.group(1))
            local_img = str((ROOT / l["image"]).resolve()).replace("\\", "/")
            if local_img in mapping and mapping[local_img] != coco_id:
                raise ValueError(f"conflicting coco id for {local_img}")
            mapping[local_img] = coco_id
            rec = per_image.setdefault(coco_id, {"present": set(), "absent": set()})
            rec["present" if l["answer"] == "yes" else "absent"].add(l["question"])

    print(f"mismatches: {mismatches}")
    print(f"unique local images: {len(mapping)}, unique coco ids: {len(per_image)}")
    # check conflicts (same object both present and absent for one image)
    conflicts = 0
    for cid, rec in per_image.items():
        both = rec["present"] & rec["absent"]
        if both:
            conflicts += 1
            if conflicts < 5:
                print(f"  conflict coco {cid}: {both}")
    print(f"images with present/absent conflicts: {conflicts}")

    out_dir = ROOT / "data" / "detection"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "coco_image_map.json").write_text(
        json.dumps(mapping, indent=1), encoding="utf-8")
    (out_dir / "pope_objects_per_image.json").write_text(
        json.dumps(
            {str(k): {"present": sorted(v["present"]), "absent": sorted(v["absent"])}
             for k, v in per_image.items()},
            indent=1,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print("saved ->", out_dir)


if __name__ == "__main__":
    main()
