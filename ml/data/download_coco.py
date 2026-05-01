"""Download COCO 2017 val subset (person class only) for YOLOv8 fine-tuning.

Run:
    python ml/data/download_coco.py

Then annotate custom frames in Label Studio and run prepare_dataset.py.
"""
import json
import shutil
from pathlib import Path

COCO_VAL_IMAGES_URL = "http://images.cocodataset.org/zips/val2017.zip"
COCO_ANNS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
RAW_DIR = Path("ml/data/raw")
PERSON_CATEGORY_ID = 1


def extract_person_images(
    ann_path: Path, images_dir: Path, out_dir: Path, max_images: int = 300
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(ann_path) as f:
        coco = json.load(f)

    person_image_ids = {
        ann["image_id"]
        for ann in coco["annotations"]
        if ann["category_id"] == PERSON_CATEGORY_ID
    }
    id_to_filename = {img["id"]: img["file_name"] for img in coco["images"]}
    selected = list(person_image_ids)[:max_images]

    copied = 0
    for img_id in selected:
        src = images_dir / id_to_filename[img_id]
        if src.exists():
            shutil.copy(src, out_dir / src.name)
            copied += 1
    print(f"Copied {copied} person images to {out_dir}")


if __name__ == "__main__":
    print("Step 1: Download and unzip COCO val2017 manually:")
    print(f"  Images:      {COCO_VAL_IMAGES_URL}")
    print(f"  Annotations: {COCO_ANNS_URL}")
    print(f"  Extract both to: {RAW_DIR}/")
    print()
    print("Step 2: Run extract_person_images() after extraction:")
    print("  from ml.data.download_coco import extract_person_images")
    print("  extract_person_images(")
    print("      ann_path=Path('ml/data/raw/annotations/instances_val2017.json'),")
    print("      images_dir=Path('ml/data/raw/val2017'),")
    print("      out_dir=Path('ml/data/raw/person_images'),")
    print("  )")
