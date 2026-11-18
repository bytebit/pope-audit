"""Build per-image ground-truth object lists from COCO instances_val2014
for the 500 COCO images used by POPE."""

import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ann = json.loads((ROOT / "data" / "detection" / "instances_val2014.json").read_text(encoding="utf-8"))
    cat_id2name = {c["id"]: c["name"] for c in ann["categories"]}
    img_objs = defaultdict(set)
    for a in ann["annotations"]:
        img_objs[a["image_id"]].add(cat_id2name[a["category_id"]])
    print("COCO val2014 images with instances:", len(img_objs))

    image_map = json.loads((ROOT / "data" / "detection" / "coco_image_map.json").read_text(encoding="utf-8"))
    coco_ids = sorted(set(image_map.values()))
    print("our coco ids:", len(coco_ids))
    missing = [c for c in coco_ids if c not in img_objs]
    print("coco ids missing from instances:", len(missing), missing[:10])

    gt = {str(c): {"present": sorted(img_objs.get(c, set()))} for c in coco_ids}
    out = ROOT / "data" / "detection" / "coco_gt_per_image.json"
    out.write_text(json.dumps(gt, indent=1), encoding="utf-8")
    print("saved ->", out)
    print("example:", coco_ids[0], gt[str(coco_ids[0])]["present"][:12])
    n_obj = sum(len(v["present"]) for v in gt.values())
    print("avg objects/image:", round(n_obj / len(coco_ids), 2))


if __name__ == "__main__":
    main()
