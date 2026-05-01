"""Convert COCO annotations to YOLOv8 format and split into train/val/test.

Run:
    python ml/data/prepare_dataset.py
"""
import json
import random
import shutil
from pathlib import Path

RAW_IMAGES_DIR = Path("ml/data/raw/val2017")
COCO_ANN_FILE = Path("ml/data/raw/annotations/instances_val2017.json")
OUTPUT_DIR = Path("ml/data/processed/yolo_dataset")
PERSON_CAT_ID = 1
SPLITS = {"train": 0.7, "val": 0.2, "test": 0.1}


def coco_bbox_to_yolo(bbox: list, img_w: int, img_h: int) -> tuple:
    x, y, w, h = bbox
    cx = (x + w / 2) / img_w
    cy = (y + h / 2) / img_h
    return cx, cy, w / img_w, h / img_h


def prepare() -> None:
    with open(COCO_ANN_FILE) as f:
        coco = json.load(f)

    img_meta = {img["id"]: img for img in coco["images"]}
    img_anns: dict[int, list] = {}
    for ann in coco["annotations"]:
        if ann["category_id"] == PERSON_CAT_ID:
            img_anns.setdefault(ann["image_id"], []).append(ann["bbox"])

    img_ids = list(img_anns.keys())
    random.seed(42)
    random.shuffle(img_ids)
    n = len(img_ids)
    train_end = int(n * SPLITS["train"])
    val_end = train_end + int(n * SPLITS["val"])
    split_map = (
        {i: "train" for i in img_ids[:train_end]}
        | {i: "val" for i in img_ids[train_end:val_end]}
        | {i: "test" for i in img_ids[val_end:]}
    )

    for split in SPLITS:
        (OUTPUT_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / split / "labels").mkdir(parents=True, exist_ok=True)

    copied = 0
    for img_id, split in split_map.items():
        meta = img_meta[img_id]
        src = RAW_IMAGES_DIR / meta["file_name"]
        if not src.exists():
            continue
        shutil.copy(src, OUTPUT_DIR / split / "images" / meta["file_name"])
        label_path = OUTPUT_DIR / split / "labels" / (Path(meta["file_name"]).stem + ".txt")
        with open(label_path, "w") as lf:
            for bbox in img_anns[img_id]:
                cx, cy, w, h = coco_bbox_to_yolo(bbox, meta["width"], meta["height"])
                lf.write(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
        copied += 1

    counts = {s: sum(1 for v in split_map.values() if v == s) for s in SPLITS}
    print(f"Dataset prepared in {OUTPUT_DIR}")
    for s, c in counts.items():
        print(f"  {s}: {c} images")


if __name__ == "__main__":
    prepare()
