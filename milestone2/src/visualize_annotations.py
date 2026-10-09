from pathlib import Path
import csv

from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[2]

METADATA_DIR = PROJECT_ROOT / "milestone2" / "metadata"
GENERATED_DIR = PROJECT_ROOT / "milestone2" / "data" / "generated"
OUTPUT_DIR = PROJECT_ROOT / "milestone2" / "results" / "debug_visualizations"

MANIFEST = METADATA_DIR / "composite_manifest_debug.csv"


def read_manifest():
    with MANIFEST.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = read_manifest()

    grouped = {}

    for row in rows:
        grouped.setdefault(
            row["composite_image"],
            []
        ).append(row)

    for image_rel_path, annotations in grouped.items():

        image_path = PROJECT_ROOT / image_rel_path

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            draw = ImageDraw.Draw(image)

            for row in annotations:

                x_min = int(row["x_min"])
                y_min = int(row["y_min"])
                x_max = int(row["x_max"])
                y_max = int(row["y_max"])

                label = (
                    f"{row['object_id']} "
                    f"{row['object_name']}"
                )

                draw.rectangle(
                    [x_min, y_min, x_max - 1, y_max - 1],
                    outline="red",
                    width=4,
                )

                text_x = x_min + 8
                text_y = y_min + 8

                draw.text(
                    (text_x, text_y),
                    label,
                    fill="red",
                )

            output_name = (
                f"annotated_{image_path.name}"
            )

            output_path = OUTPUT_DIR / output_name

            image.save(
                output_path,
                format="JPEG",
                quality=95,
            )

    print("\nANNOTATION VISUALIZATION")
    print("=" * 60)
    print(f"Images visualized: {len(grouped)}")
    print(f"Output folder: {OUTPUT_DIR.relative_to(PROJECT_ROOT)}")
    print("Visualization complete.")


if __name__ == "__main__":
    main()