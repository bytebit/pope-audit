"""Download the POPE dataset (questions + images) and convert to local JSON + image files."""

import argparse
import json
import re
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

REPO = "lmms-lab/POPE"
FILES = {
    "adversarial": "Full/adversarial-00000-of-00001.parquet",
    "popular": "Full/popular-00000-of-00001.parquet",
    "random": "Full/random-00000-of-00001.parquet",
}


def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", str(s))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = ap.parse_args()
    root: Path = args.root

    parquet_dir = root / "data" / "pope" / "parquet"
    parquet_dir.mkdir(parents=True, exist_ok=True)

    for subset, filename in FILES.items():
        target = parquet_dir / Path(filename).name
        if not target.exists():
            print(f"[1/2] downloading {filename} ...")
            hf_hub_download(
                repo_id=REPO,
                filename=filename,
                repo_type="dataset",
                local_dir=parquet_dir,
            )
        else:
            print(f"[1/2] {target.name} already present")

        print(f"[2/2] converting {subset} ...")
        df = pd.read_parquet(target)
        images_dir = root / "data" / "pope" / "images" / subset
        images_dir.mkdir(parents=True, exist_ok=True)

        records = []
        for i, row in enumerate(df.itertuples(index=False)):
            img = row.image
            if isinstance(img, dict):
                blob = img.get("bytes")
                if blob is None and img.get("path"):
                    blob = Path(img["path"]).read_bytes()
            elif isinstance(img, (bytes, bytearray)):
                blob = bytes(img)
            else:
                blob = None

            img_name = f"{subset}_{i:04d}.jpg"
            img_path = images_dir / img_name
            if blob is not None and not img_path.exists():
                img_path.write_bytes(blob)

            records.append(
                {
                    "question_id": str(row.question_id),
                    "image": str(img_path.relative_to(root)).replace("\\", "/"),
                    "question": str(row.question),
                    "answer": str(row.answer).strip().lower(),
                    "category": str(getattr(row, "category", "")),
                }
            )

        out = root / "data" / "pope" / f"{subset}.json"
        out.write_text(json.dumps(records, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"  -> {len(records)} records -> {out}")

    print("POPE data ready.")


if __name__ == "__main__":
    main()

