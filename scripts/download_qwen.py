"""Robust per-file downloader for Qwen2.5-VL (avoids stalled snapshot_download)."""

import os
import sys
import time
from pathlib import Path

REPO = "Qwen/Qwen2.5-VL-7B-Instruct"
ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT / "models" / "qwen2.5-vl-7b-instruct"


def main() -> None:
    from huggingface_hub import HfApi, hf_hub_download

    LOCAL.mkdir(parents=True, exist_ok=True)
    files = [f for f in HfApi().list_repo_files(REPO) if not f.startswith(".")]
    # config files first, weights last
    files.sort(key=lambda f: (f.endswith(".safetensors"), f))
    print(f"{len(files)} files to fetch", flush=True)
    for f in files:
        target = LOCAL / f
        if target.exists() and target.stat().st_size > 1000:
            print(f"skip {f}", flush=True)
            continue
        for attempt in range(1, 4):
            t0 = time.time()
            try:
                hf_hub_download(repo_id=REPO, filename=f, local_dir=str(LOCAL))
                size = target.stat().st_size / 1e6 if target.exists() else -1
                print(f"ok   {f}  {size:.0f} MB  {time.time() - t0:.0f}s", flush=True)
                break
            except Exception as e:  # noqa: BLE001
                print(f"retry {attempt} {f}: {type(e).__name__} {e}", flush=True)
                time.sleep(5)
    print("done", flush=True)


if __name__ == "__main__":
    main()
