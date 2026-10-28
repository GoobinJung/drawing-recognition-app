from pathlib import Path

import numpy as np

from download_data import CLASS_NAMES, DATA_DIR


OUTPUT_DIR = Path(__file__).resolve().parent / "data" / "processed"
SEED = 42
SAMPLES_PER_CLASS = 2000

SPLIT_RANGES = {
    "train": (0, 1400),
    "validation": (1400, 1700),
    "test": (1700, 2000),
}


def main():
    rng = np.random.default_rng(SEED)
    seen_images = set()

    splits = {
        name: {"images": [], "labels": [], "source_rows": []}
        for name in SPLIT_RANGES
    }

    for label, class_name in enumerate(CLASS_NAMES):
        raw_images = np.load(
            DATA_DIR / f"{class_name}.npy",
            mmap_mode="r",
            allow_pickle=False,
        )

        selected_images = []
        selected_rows = []

        for row in rng.permutation(len(raw_images)):
            image = raw_images[row]

            # Skip blank images and exact duplicates.
            if not np.any(image):
                continue

            image_key = image.tobytes()

            if image_key in seen_images:
                continue

            seen_images.add(image_key)
            selected_images.append(image.copy())
            selected_rows.append(int(row))

            if len(selected_images) == SAMPLES_PER_CLASS:
                break

        if len(selected_images) != SAMPLES_PER_CLASS:
            raise ValueError(f"Not enough unique images: {class_name}")

        images = np.stack(selected_images).reshape(-1, 28, 28)
        source_rows = np.array(selected_rows, dtype=np.int64)

        for split_name, (start, end) in SPLIT_RANGES.items():
            splits[split_name]["images"].append(images[start:end])
            splits[split_name]["labels"].append(
                np.full(end - start, label, dtype=np.int64)
            )
            splits[split_name]["source_rows"].append(
                source_rows[start:end]
            )

        print(f"{class_name}: selected {len(images):,} images")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for split_name, parts in splits.items():
        images = np.concatenate(parts["images"])
        labels = np.concatenate(parts["labels"])
        source_rows = np.concatenate(parts["source_rows"])

        order = rng.permutation(len(labels))

        np.savez_compressed(
            OUTPUT_DIR / f"{split_name}.npz",
            images=images[order],
            labels=labels[order],
            source_rows=source_rows[order],
            class_names=np.array(CLASS_NAMES),
            seed=np.array(SEED),
        )

        print(f"{split_name}: {len(labels):,} images, shape={images.shape}")

    print("\nDataset preparation finished.")


if __name__ == "__main__":
    main()