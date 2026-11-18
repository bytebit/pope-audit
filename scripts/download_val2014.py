"""Download missing COCO val2014 images (parallel)."""

import json
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = "http://images.cocodataset.org/val2014/COCO_val2014_{:012d}.jpg"


def fetch(cid: int, out_dir: Path, workers: int) -> None:
    dst = out_dir / f"{cid}.jpg"
    if dst.exists() and dst.stat().st_size > 1000:
        return "skip"
    url = BASE.format(cid)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "codex-audit/1.0"})
        with urllib.request.urlopen(req, timeout=60) as r, open(dst, "wb") as f:
            f.write(r.read())
        return "ok"
    except Exception as e:  # noqa: BLE001
        return f"err:{e}"


def main() -> None:
    ids = json.loads((ROOT / "data" / "detection" / "val2014_5000_missing.json").read_text(encoding="utf-8"))
    out_dir = ROOT / "data" / "detection" / "val2014"
    out_dir.mkdir(parents=True, exist_ok=True)
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    done = ok = err = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fetch, c, out_dir, workers): c for c in ids}
        for f in as_completed(futs):
            res = f.result()
            done += 1
            if res == "ok":
                ok += 1
            elif res == "skip":
                pass
            else:
                err += 1
            if done % 500 == 0:
                print(f"{done}/{len(ids)} ok={ok} err={err}", flush=True)
    print(f"done. ok={ok} err={err} files={len(list(out_dir.glob('*.jpg')))}")


if __name__ == "__main__":
    main()
