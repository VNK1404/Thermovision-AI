#!/usr/bin/env python3
"""
Prepare a YOLOv8-style dataset from `thermal_detection_results/annotated_images` and the
corresponding `thermal_detection_results/labels` files.

Features:
 - Copies images into train/val/test splits
 - Copies/creates label TXT files in YOLO format (class x_center y_center w h)
 - If label files contain only a class id (e.g. `0`), you can opt-in to create a full-image
   bounding box for that class (useful to convert image-level labels to detection labels).

Usage:
  python prepare_yolo_dataset.py --images_dir thermal_detection_results/annotated_images \
    --labels_dir thermal_detection_results/labels --out yolo_from_thermal --split 0.8 0.1 0.1 --force_full_bbox

Notes / assumptions:
 - If a label file already contains YOLO-style lines (more than one value per line), it will be copied as-is.
 - If a label file contains a single integer per file, that integer is treated as the class id.
 - When creating a full-image bbox we write: class_id 0.5 0.5 1.0 1.0 (normalized center & size)
 - The script will create a `data.yaml` in the output folder compatible with `train.py`
"""

import argparse
from pathlib import Path
import random
import shutil
import sys


def read_label_file(label_path: Path):
    """Return contents of a label file as list of stripped lines (or empty list)."""
    try:
        txt = label_path.read_text(encoding='utf-8').strip()
        if not txt:
            return []
        lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
        return lines
    except Exception:
        return []


def looks_like_yolo_line(line: str) -> bool:
    parts = line.split()
    # YOLO line is usually: class x_center y_center width height (5+ numbers)
    if len(parts) >= 5:
        try:
            [float(p) for p in parts[1:5]]
            int(parts[0])
            return True
        except Exception:
            return False
    return False


def prepare_dataset(images_dir: Path, labels_dir: Path, out_dir: Path, split=(0.8, 0.1, 0.1), force_full_bbox=False, seed=42):
    out_dir = out_dir.resolve()
    images_dir = images_dir.resolve()
    labels_dir = labels_dir.resolve()

    if not images_dir.exists():
        raise FileNotFoundError(f"Images dir not found: {images_dir}")

    out_train_img = out_dir / 'train' / 'images'
    out_train_lbl = out_dir / 'train' / 'labels'
    out_val_img = out_dir / 'valid' / 'images'
    out_val_lbl = out_dir / 'valid' / 'labels'
    out_test_img = out_dir / 'test' / 'images'
    out_test_lbl = out_dir / 'test' / 'labels'

    for p in [out_train_img, out_train_lbl, out_val_img, out_val_lbl, out_test_img, out_test_lbl]:
        p.mkdir(parents=True, exist_ok=True)

    # Gather images
    img_files = [p for p in images_dir.iterdir() if p.suffix.lower() in ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')]
    img_files = sorted(img_files)
    if not img_files:
        raise FileNotFoundError(f"No images found in {images_dir}")

    random.seed(seed)
    random.shuffle(img_files)

    n = len(img_files)
    n_train = int(split[0] * n)
    n_val = int(split[1] * n)
    # remainder -> test
    train_files = img_files[:n_train]
    val_files = img_files[n_train:n_train + n_val]
    test_files = img_files[n_train + n_val:]

    def copy_partition(files, img_out, lbl_out):
        for img in files:
            # Determine matching label file by stem
            stem = img.stem
            # There may be labels with same stem and different extensions; try to match
            candidate = labels_dir / (stem + '.txt')
            # fallback: try matching by filename without the trailing suffix after first underscore
            if not candidate.exists():
                # try other label name patterns
                alt = None
                for f in labels_dir.iterdir():
                    if f.is_file() and f.stem == stem:
                        alt = f
                        break
                candidate = alt or candidate

            # Copy image
            shutil.copy2(img, img_out / img.name)

            if candidate and candidate.exists():
                lines = read_label_file(candidate)
                if all(looks_like_yolo_line(ln) for ln in lines):
                    # copy label file as-is
                    (lbl_out / (img.stem + '.txt')).write_text('\n'.join(lines), encoding='utf-8')
                else:
                    # treat as image-level label (single integer per file)
                    try:
                        cls = int(lines[0].split()[0]) if lines else None
                    except Exception:
                        cls = None
                    if cls is None:
                        # write empty label
                        (lbl_out / (img.stem + '.txt')).write_text('', encoding='utf-8')
                    else:
                        if force_full_bbox:
                            # write a full-image bbox (center 0.5,0.5 size 1.0,1.0)
                            (lbl_out / (img.stem + '.txt')).write_text(f"{cls} 0.5 0.5 1.0 1.0\n", encoding='utf-8')
                        else:
                            # no bbox available, write empty (will be treated as no detection)
                            (lbl_out / (img.stem + '.txt')).write_text('', encoding='utf-8')
            else:
                # no label file found -> write empty label
                (lbl_out / (img.stem + '.txt')).write_text('', encoding='utf-8')

    copy_partition(train_files, out_train_img, out_train_lbl)
    copy_partition(val_files, out_val_img, out_val_lbl)
    copy_partition(test_files, out_test_img, out_test_lbl)

    # Create data.yaml
    data_yaml = out_dir / 'data.yaml'
    classes = set()
    # inspect some label files to get class names (best-effort)
    for lbl in list(out_train_lbl.glob('*.txt'))[:50]:
        lines = read_label_file(lbl)
        for ln in lines:
            parts = ln.split()
            if parts:
                try:
                    classes.add(int(parts[0]))
                except:
                    pass

    # sort class ids and create names mapping
    cls_ids = sorted(list(classes)) if classes else [0]
    names_lines = []
    for cid in cls_ids:
        # best-effort names
        if cid == 0:
            nm = 'FIRE'
        else:
            nm = f'CLASS_{cid}'
        names_lines.append(f"  {cid}: {nm}")

    data_yaml.write_text("""train: train/images
val: valid/images
test: test/images

names:
""" + "\n".join(names_lines), encoding='utf-8')

    print(f"Prepared dataset at: {out_dir}")
    print(f"  train: {len(list(out_train_img.glob('*')))} images")
    print(f"  valid: {len(list(out_val_img.glob('*')))} images")
    print(f"  test:  {len(list(out_test_img.glob('*')))} images")
    print(f"Wrote data.yaml -> {data_yaml}")


def main():
    parser = argparse.ArgumentParser(description='Prepare YOLOv8 dataset from thermal_detection_results annotated images')
    parser.add_argument('--images_dir', required=True, help='Path to images folder (annotated_images)')
    parser.add_argument('--labels_dir', required=True, help='Path to labels folder')
    parser.add_argument('--out', default='yolo_from_thermal', help='Output folder for YOLO dataset')
    parser.add_argument('--split', nargs=3, type=float, default=(0.8, 0.1, 0.1), help='Train/val/test split fractions')
    parser.add_argument('--force_full_bbox', action='store_true', help='If label files only contain class id, create a full-image bbox for positives')
    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    labels_dir = Path(args.labels_dir)
    out_dir = Path(args.out)

    prepare_dataset(images_dir, labels_dir, out_dir, split=tuple(args.split), force_full_bbox=args.force_full_bbox)


if __name__ == '__main__':
    main()
