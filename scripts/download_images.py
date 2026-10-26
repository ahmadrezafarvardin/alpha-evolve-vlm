"""Download CharXiv images into charxiv/images/ and verify them.

    python scripts/download_images.py          # skips the download if all images are present
    python scripts/download_images.py --force  # re-download

Exits non-zero unless every figure listed in the val + test metadata (1,000 + 1,323 = 2,323) is
present. charxiv/README.md says 2,333 images; that is a typo, the archive holds 2,323.
"""
import sys
import json
import shutil
import zipfile
import argparse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IMAGE_DIR = ROOT / "charxiv" / "images"
URL = "https://huggingface.co/datasets/princeton-nlp/CharXiv/resolve/main/images.zip"
DEV_SAMPLES = 128


def count_images():
    return sum(1 for _ in IMAGE_DIR.glob("*.jpg"))


def download(zip_path):
    print(f"Downloading {URL}")
    with urllib.request.urlopen(URL) as resp, open(zip_path, "wb") as f:
        total = int(resp.headers.get("Content-Length", 0))
        done = 0
        while chunk := resp.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done / 1e6:.0f} / {total / 1e6:.0f} MB", end="", flush=True)
    print()


def extract(zip_path):
    # Flatten: the archive may store files under a top-level folder.
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            if member.is_dir() or not member.filename.endswith(".jpg"):
                continue
            with zf.open(member) as src, open(IMAGE_DIR / Path(member.filename).name, "wb") as dst:
                shutil.copyfileobj(src, dst)


def expected_figure_ids():
    ids = set()
    for split in ("val", "test"):
        with open(ROOT / "charxiv" / "data" / f"image_metadata_{split}.json") as f:
            ids |= set(json.load(f))
    return ids


def dev_figure_ids():
    with open(ROOT / "charxiv" / "data" / "descriptive_val.json") as f:
        data = json.load(f)
    ids, n = [], 0
    for entry in data.values():
        if n >= DEV_SAMPLES:
            break
        ids.append(entry["figure_id"])
        n += len(entry["qids"])
    return ids


def missing_images():
    present = {p.stem for p in IMAGE_DIR.glob("*.jpg")}
    return sorted(expected_figure_ids() - present, key=int)


def verify():
    count, expected = count_images(), len(expected_figure_ids())
    missing = missing_images()
    missing_dev = [i for i in dev_figure_ids() if str(i) in missing]
    if count != expected or missing:
        sys.exit(
            f"\n{'!' * 70}\nIMAGE CHECK FAILED: found {count} images, expected {expected}; "
            f"{len(missing)} missing (first: {missing[:10]}), of which dev figures: {missing_dev}\n{'!' * 70}"
        )
    print(f"OK: {count} images in {IMAGE_DIR}, matching val + test metadata (all dev figures present)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="re-download even if images are present")
    args = parser.parse_args()

    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    if args.force or missing_images():
        zip_path = IMAGE_DIR / "images.zip"
        download(zip_path)
        extract(zip_path)
        zip_path.unlink()
    verify()


if __name__ == "__main__":
    main()
