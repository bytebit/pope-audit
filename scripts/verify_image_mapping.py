"""Verify the local-image -> COCO-id mapping by comparing md5 hashes against
official COCO val2014 images for a sample of ids."""

import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    image_map = json.loads((ROOT / "data" / "detection" / "coco_image_map.json").read_text(encoding="utf-8"))
    by_coco = {}
    for local, cid in image_map.items():
        by_coco.setdefault(cid, local)
    sample = sorted(by_coco.items())[:n]
    ok = 0
    with tempfile.TemporaryDirectory() as tmp:
        for cid, local in sample:
            url = f"http://images.cocodataset.org/val2014/COCO_val2014_{cid:012d}.jpg"
            dst = Path(tmp) / f"{cid}.jpg"
            r = subprocess.run(
                ["curl.exe", "-L", "--max-time", "60", "-s", "-o", str(dst), url],
                capture_output=True,
            )
            if not dst.exists() or dst.stat().st_size < 1000:
                print(f"coco {cid}: download failed")
                continue
            h_local = md5(ROOT / local)
            h_coco = md5(dst)
            match = h_local == h_coco
            ok += int(match)
            print(f"coco {cid} local={local.split('/')[-1]} "
                  f"match={match} size_local={(ROOT / local).stat().st_size} size_coco={dst.stat().st_size}")
    print(f"\nmatch rate: {ok}/{n}")


if __name__ == "__main__":
    main()
