import os
from pathlib import Path
from PIL import Image


def clean_dataset(data_dir="./data"):
    """
    Scans the dataset directory and deletes any images that PIL cannot open.
    """
    data_path = Path(data_dir)
    corrupted_count = 0

    print(f"Scanning {data_dir} for corrupted images...")

    # rglob finds all .jpg files recursively in all subfolders (Cat, Dog)
    for img_path in data_path.rglob("*.jpg"):
        try:
            # img.verify() checks the file header without loading the whole image into memory
            with Image.open(img_path) as img:
                img.verify()
        except Exception as e:
            print(f"Deleting corrupted file: {img_path} | Error: {e}")
            os.remove(img_path)
            corrupted_count += 1

    print(f"Cleanup complete! Removed {corrupted_count} corrupted images.")


if __name__ == "__main__":
    clean_dataset()