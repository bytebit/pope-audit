"""Download llava-hf/llava-1.5-7b-hf from hf-mirror with resume + retry.

The three big safetensors shards are downloaded sequentially; every attempt
resumes from the local byte offset, so a stalled connection never wastes work.
"""

import subprocess
import sys
import time
from pathlib import Path

BASE = "https://hf-mirror.com/llava-hf/llava-1.5-7b-hf/resolve/main"
OUT = Path(__file__).resolve().parents[1] / "models" / "llava-1.5-7b-hf"

EXPECTED = {
    "model-00001-of-00003.safetensors": 4992930200,
    "model-00002-of-00003.safetensors": 4957878440,
    "model-00003-of-00003.safetensors": 4176137408,
}

SMALL = [
    "README.md",
    "added_tokens.json",
    "chat_template.jinja",
    "chat_template.json",
    "config.json",
    "generation_config.json",
    "model.safetensors.index.json",
    "preprocessor_config.json",
    "processor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
]


def fetch_one(name: str, expected: int | None = None, max_attempts: int = 120) -> bool:
    url = f"{BASE}/{name}"
    out = OUT / name
    for attempt in range(1, max_attempts + 1):
        cur = out.stat().st_size if out.exists() else 0
        if expected is not None and cur >= expected:
            print(f"[OK] {name} complete ({cur} bytes)", flush=True)
            return True
        subprocess.run(
            ["curl.exe", "-L", "-C", "-", "--max-time", "300", "-s", "-o", str(out), url],
            check=False,
        )
        cur = out.stat().st_size if out.exists() else 0
        print(f"[{name}] attempt={attempt} size={cur}", flush=True)
        if expected is not None and cur >= expected:
            print(f"[OK] {name} complete", flush=True)
            return True
        if expected is None and cur > 0 and attempt == 1:
            return True
        time.sleep(5)
    return False


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for name in SMALL:
        fetch_one(name)
    for name, size in EXPECTED.items():
        fetch_one(name, size)
    ok = all(fetch_one(name, size, max_attempts=1) for name, size in EXPECTED.items())
    print("ALL DONE" if ok else "INCOMPLETE")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
