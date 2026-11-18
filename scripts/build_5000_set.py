"""Sample 5000 COCO val2014 images for the PAS original-protocol replication.

Saves:
  - data/detection/val2014_5000_ids.json : list of 5000 COCO image ids (seed 0)
  - data/detection/val2014_5000_missing.json : ids not yet present locally
Reuses the 500 local POPE images when they overlap with the sample.
"""

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
N = 5000
SEED = 0


def main() -> None:
    ann = json.loads((ROOT / "data" / "detection" / "instances_val2014.json").read_text(encoding="utf-8"))
    all_ids = sorted(a["id"] for a in ann["images"])
    rng = random.Random(SEED)
    sample = sorted(rng.sample(all_ids, N))
    print(f"val2014 images: {len(all_ids)}  sampled: {len(sample)}")

    image_map = json.loads((ROOT / "data" / "detection" / "coco_image_map.json").read_text(encoding="utf-8"))
    local_by_coco = {}
    for local, cid in image_map.items():
        local_by_coco.setdefault(int(cid), local)
    overlap = [c for c in sample if c in local_by_coco]
    missing = [c for c in sample if c not in local_by_coco]
    print(f"already local (POPE overlap): {len(overlap)}  missing: {len(missing)}")

    out_dir = ROOT / "data" / "detection"
    (out_dir / "val2014_5000_ids.json").write_text(json.dumps(sample), encoding="utf-8")
    (out_dir / "val2014_5000_missing.json").write_text(json.dumps(missing), encoding="utf-8")
    print("saved ->", out_dir / "val2014_5000_ids.json")
    print("saved ->", out_dir / "val2014_5000_missing.json")


if __name__ == "__main__":
    main()
