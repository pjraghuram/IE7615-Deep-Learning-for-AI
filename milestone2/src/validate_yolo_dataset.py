from pathlib import Path
from collections import Counter

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "milestone2" / "configs" / "data.yaml"

EXPECTED_CLASSES = 73
EXPECTED_SPLIT_COUNTS = {
    "train": 280,
    "val": 60,
    "test": 60,
}
EXPECTED_LABELS_PER_IMAGE = 4


def main():

    print("\nYOLO DATASET VALIDATION")
    print("=" * 60)

    # --------------------------------------------------
    # Load YAML
    # --------------------------------------------------

    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if config["nc"] != EXPECTED_CLASSES:
        raise ValueError(
            f"Expected {EXPECTED_CLASSES} classes, "
            f"found {config['nc']}."
        )

    if len(config["names"]) != EXPECTED_CLASSES:
        raise ValueError(
            "Class-name count does not match nc."
        )

    dataset_root = (
    PROJECT_ROOT / config["path"]
    ).resolve()

    if not dataset_root.exists():
        raise FileNotFoundError(
            f"Dataset root does not exist: {dataset_root}"
        )

    print(f"Dataset root: {dataset_root}")
    print(f"Classes:      {config['nc']}")

    # --------------------------------------------------
    # Validate each split
    # --------------------------------------------------

    total_images = 0
    total_labels = 0
    total_boxes = 0

    global_class_counts = Counter()

    for split in ("train", "val", "test"):

        image_dir = dataset_root / "images" / split
        label_dir = dataset_root / "labels" / split

        if not image_dir.exists():
            raise FileNotFoundError(
                f"Missing image folder: {image_dir}"
            )

        if not label_dir.exists():
            raise FileNotFoundError(
                f"Missing label folder: {label_dir}"
            )

        images = sorted(
            list(image_dir.glob(f"{split}_??????.jpg"))
        )

        labels = sorted(
            list(label_dir.glob(f"{split}_??????.txt"))
        )

        expected_count = EXPECTED_SPLIT_COUNTS[split]

        if len(images) != expected_count:
            raise ValueError(
                f"{split}: expected {expected_count} final images, "
                f"found {len(images)}."
            )

        if len(labels) != expected_count:
            raise ValueError(
                f"{split}: expected {expected_count} final labels, "
                f"found {len(labels)}."
            )

        image_stems = {p.stem for p in images}
        label_stems = {p.stem for p in labels}

        missing_labels = image_stems - label_stems
        missing_images = label_stems - image_stems

        if missing_labels:
            raise ValueError(
                f"{split}: images without labels: "
                f"{sorted(missing_labels)}"
            )

        if missing_images:
            raise ValueError(
                f"{split}: labels without images: "
                f"{sorted(missing_images)}"
            )

        split_class_counts = Counter()
        split_boxes = 0

        for label_path in labels:

            lines = [
                line.strip()
                for line in label_path.read_text(
                    encoding="utf-8"
                ).splitlines()
                if line.strip()
            ]

            if len(lines) != EXPECTED_LABELS_PER_IMAGE:
                raise ValueError(
                    f"{label_path.name}: expected "
                    f"{EXPECTED_LABELS_PER_IMAGE} labels, "
                    f"found {len(lines)}."
                )

            classes_in_image = []

            for line_number, line in enumerate(lines, start=1):

                parts = line.split()

                if len(parts) != 5:
                    raise ValueError(
                        f"{label_path.name}, line {line_number}: "
                        f"expected 5 values, found {len(parts)}."
                    )

                class_id = int(parts[0])

                if not 0 <= class_id < EXPECTED_CLASSES:
                    raise ValueError(
                        f"{label_path.name}: invalid "
                        f"class ID {class_id}."
                    )

                x_center, y_center, width, height = map(
                    float,
                    parts[1:]
                )

                if not (
                    0.0 <= x_center <= 1.0
                    and 0.0 <= y_center <= 1.0
                    and 0.0 < width <= 1.0
                    and 0.0 < height <= 1.0
                ):
                    raise ValueError(
                        f"{label_path.name}: invalid normalized box "
                        f"{x_center}, {y_center}, {width}, {height}"
                    )

                classes_in_image.append(class_id)
                split_class_counts[class_id] += 1
                global_class_counts[class_id] += 1
                split_boxes += 1

            if len(classes_in_image) != len(set(classes_in_image)):
                raise ValueError(
                    f"{label_path.name}: duplicate class "
                    f"inside one composite."
                )

        missing_classes = (
            set(range(EXPECTED_CLASSES))
            - set(split_class_counts)
        )

        if missing_classes:
            raise ValueError(
                f"{split}: missing classes "
                f"{sorted(missing_classes)}"
            )

        print(f"\n{split.upper()}")
        print(f"  Images:       {len(images)}")
        print(f"  Labels:       {len(labels)}")
        print(f"  Boxes:        {split_boxes}")
        print(
            f"  Class range:  "
            f"{min(split_class_counts.values())}"
            f"–{max(split_class_counts.values())}"
        )
        print("  Pairing:      PASS")
        print("  Label format: PASS")
        print("  Class cover:  PASS")

        total_images += len(images)
        total_labels += len(labels)
        total_boxes += split_boxes

    # --------------------------------------------------
    # Global checks
    # --------------------------------------------------

    if set(global_class_counts) != set(range(EXPECTED_CLASSES)):
        raise ValueError(
            "Global dataset does not contain all 73 classes."
        )

    print("\nGLOBAL SUMMARY")
    print(f"  Images:       {total_images}")
    print(f"  Label files:  {total_labels}")
    print(f"  Bounding boxes: {total_boxes}")
    print(f"  Classes:      {len(global_class_counts)}")

    print("\nFINAL CHECKS")
    print("  data.yaml:          PASS")
    print("  Image-label pairs:  PASS")
    print("  YOLO format:        PASS")
    print("  Coordinate ranges:  PASS")
    print("  Class IDs 0–72:     PASS")
    print("  All classes present: PASS")

    print("\nYOLO dataset validation complete.")


if __name__ == "__main__":
    main()