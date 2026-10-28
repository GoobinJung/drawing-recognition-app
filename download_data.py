from pathlib import Path
from urllib.request import urlretrieve

import numpy as np


CLASS_NAMES = ["cat", "car", "tree", "umbrella", "clock"]

BASE_URL = (
    "https://storage.googleapis.com/"
    "quickdraw_dataset/full/numpy_bitmap"
)

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data" / "raw"


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for name in CLASS_NAMES:
        file_path = DATA_DIR / f"{name}.npy"

        if not file_path.exists():
            url = f"{BASE_URL}/{name}.npy"
            temporary_path = DATA_DIR / f"{name}.npy.part"

            print(f"Downloading {name}...", flush=True)
            urlretrieve(url, temporary_path)
            temporary_path.replace(file_path)
        else:
            print(f"Using existing file: {name}.npy")

        images = np.load(
            file_path,
            mmap_mode="r",
            allow_pickle=False,
        )

        if images.ndim != 2 or images.shape[1] != 784:
            raise ValueError(f"Unexpected image shape: {name}")

        if images.dtype != np.uint8:
            raise ValueError(f"Unexpected image type: {name}")

        print(f"{name}: {len(images):,} images")
        print(f"Shape: {images.shape}, type: {images.dtype}")

    print("\nAll five datasets are ready.")
    print(f"Saved to: {DATA_DIR}")


if __name__ == "__main__":
    main()