from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"


def main():
    with np.load(
        DATA_DIR / "processed" / "train.npz",
        allow_pickle=False,
    ) as data:
        images = data["images"]
        labels = data["labels"]
        class_names = data["class_names"].tolist()

    rng = np.random.default_rng(42)
    sheet = Image.new("RGB", (550, 680), "white")
    draw = ImageDraw.Draw(sheet)

    for row, class_name in enumerate(class_names):
        indices = np.flatnonzero(labels == row)
        selected = rng.choice(indices, size=5, replace=False)

        draw.text(
            (10, row * 130 + 8),
            class_name,
            fill="black",
        )

        for column, index in enumerate(selected):
            picture = Image.fromarray(images[index])
            picture = picture.resize(
                (100, 100),
                Image.Resampling.NEAREST,
            )

            sheet.paste(
                picture,
                (column * 110 + 5, row * 130 + 25),
            )

    draw.text(
        (10, 655),
        "Source: Google Quick, Draw! Dataset (CC BY 4.0)",
        fill="black",
    )

    output_path = DATA_DIR / "preview.png"
    sheet.save(output_path)

    print(f"Preview saved to: {output_path}")
    print("Each row shows five training images from one category.")


if __name__ == "__main__":
    main()