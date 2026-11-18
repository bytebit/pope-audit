"""Robust per-file model downloader (retries each file; avoids stalled
snapshot_download). Usage:
    python scripts/download_model_robust.py <repo_id> <local_dir>
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    repo = sys.argv[1]
    local = Path(sys.argv[2])
    if not local.is_absolute():
        local = ROOT / local
    local.mkdir(parents=True, exist_ok=True)

    from huggingface_hub import HfApi, hf_hub_download

    files = [f for f in HfApi().list_repo_files(repo) if not f.startswith(".")]
    files.sort(key=lambda f: (f.endswith(".safetensors"), f))
    print(f"{repo}: {len(files)} files", flush=True)

    for f in files:
        target = local / f
        if target.exists() and target.stat().st_size > 1000:
            print(f"skip {f}", flush=True)
            continue
        for attempt in range(1, 6):
            t0 = time.time()
            try:
                hf_hub_download(repo_id=repo, filename=f, local_dir=str(local))
                mb = target.stat().st_size / 1e6 if target.exists() else -1
                print(f"ok   {f}  {mb:.0f} MB  {time.time() - t0:.0f}s", flush=True)
                break
            except Exception as e:  # noqa: BLE001
                print(f"retry {attempt} {f}: {type(e).__name__} {str(e)[:80]}", flush=True)
                time.sleep(5)
    print("done", flush=True)


if __name__ == "__main__":
    main()
