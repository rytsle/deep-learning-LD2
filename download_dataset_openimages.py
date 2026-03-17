"""
Download images from Open Images v7 for classification.
Classes: Light bulb, Battery, Light switch, Car
100 images per class, no bounding boxes needed.

Requirements:
    pip install fiftyone

Usage:
    python download_dataset.py
"""

import fiftyone as fo
import fiftyone.zoo as foz
import os
import shutil

# --- Config ---
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "custom_data")
IMAGES_PER_CLASS = 100
SPLITS = ["train", "validation", "test"]

CLASS_MAP = {
    "Light bulb": "light_bulb",
    "Light switch": "light_switch",
    "Car":        "car",
}

def download_class(class_name, folder_name, images_per_class=100):
    """Download images for a single class across splits."""
    collected = []

    for split in SPLITS:
        remaining = images_per_class - len(collected)
        if remaining <= 0:
            break

        print(f"  [{split}] Fetching up to {remaining} images for '{class_name}'...")
        try:
            dataset = foz.load_zoo_dataset(
                "open-images-v7",
                split=split,
                label_types=["detections"],  # needed to filter by class
                classes=[class_name],
                max_samples=remaining,
                dataset_name=f"oi_{folder_name}_{split}",  # unique name per run
                overwrite=True,
            )
            collected.extend([s.filepath for s in dataset])
        except Exception as e:
            print(f"  Warning: could not load split '{split}': {e}")

    return collected[:images_per_class]


def main():
    print(f"Output directory: {os.path.abspath(OUTPUT_DIR)}\n")

    for class_name, folder_name in CLASS_MAP.items():
        class_dir = os.path.join(OUTPUT_DIR, folder_name)
        os.makedirs(class_dir, exist_ok=True)

        print(f"=== Downloading: {class_name} → {folder_name}/ ===")
        filepaths = download_class(class_name, folder_name, IMAGES_PER_CLASS)

        # Copy images into the class folder with clean sequential names
        for i, src in enumerate(filepaths):
            ext = os.path.splitext(src)[-1].lower() or ".jpg"
            dst = os.path.join(class_dir, f"{folder_name}_{i:04d}{ext}")
            shutil.copy2(src, dst)

        print(f"  Saved {len(filepaths)} images to {class_dir}/\n")

    # Print final summary
    print("=== Done ===")
    for folder_name in CLASS_MAP.values():
        class_dir = os.path.join(OUTPUT_DIR, folder_name)
        count = len(os.listdir(class_dir))
        print(f"  {folder_name}/: {count} images")

    print(f"\nDataset structure:")
    print(f"  {OUTPUT_DIR}/")
    for folder_name in CLASS_MAP.values():
        print(f"    {folder_name}/   (use as class label)")


if __name__ == "__main__":
    main()