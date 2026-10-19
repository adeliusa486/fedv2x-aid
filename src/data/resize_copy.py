#!/usr/bin/env python3
"""Copy a list of merged-set images, resized to at most ``--size`` pixels on the long side, with their labels.

Training reads every image once per epoch; the street photos of Mapillary Vistas and MTSD are up to 4000 x 3000
pixels, and decoding them, not the GPU, limits the speed. Ultralytics resizes every image to 640 pixels on the
long side before augmentation, so a copy resized once to 640 pixels (Lanczos, JPEG quality 95) gives nearly the
same training input. Labels (YOLO format, normalised coordinates) are copied unchanged. The source dataset is
only read.

    python src/data/resize_copy.py --list $FEDV2X_WORK/round0/train.txt --out $FEDV2X_WORK/round0_data --split train
"""
import argparse
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def one(args):
    src, out_img, out_lab, size, labels_dir = args
    src = Path(src)
    dst = Path(out_img) / (src.stem + ".jpg")
    if not dst.exists():
        with Image.open(src) as im:
            if im.format == "JPEG":
                im.draft("RGB", (size, size))
            im = im.convert("RGB")
            w, h = im.size
            r = size / max(w, h)
            if r < 1:
                im = im.resize((max(1, round(w * r)), max(1, round(h * r))), Image.LANCZOS)
            im.save(dst, quality=95)
    lab = Path(str(src.parent).replace("\\images\\", "\\" + labels_dir + "\\").replace("/images/", "/" + labels_dir + "/")) / (src.stem + ".txt")
    if lab.exists():
        shutil.copyfile(lab, Path(out_lab) / (src.stem + ".txt"))
    return 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--size", type=int, default=640)
    ap.add_argument("--labels", default="labels", help="labels (merged) or labels_human")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    paths = [p.strip() for p in Path(a.list).read_text().splitlines() if p.strip()]
    out_img = Path(a.out) / "images" / a.split
    out_lab = Path(a.out) / "labels" / a.split
    out_img.mkdir(parents=True, exist_ok=True)
    out_lab.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        n = sum(ex.map(one, [(p, out_img, out_lab, a.size, a.labels) for p in paths], chunksize=64))
    (Path(a.out) / f"{a.split}.txt").write_text("\n".join(str(out_img / (Path(p).stem + ".jpg")) for p in paths) + "\n")
    print(f"{n} images -> {out_img} in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
