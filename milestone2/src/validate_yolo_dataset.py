from pathlib import Path
from collections import Counter, defaultdict

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "milestone2" / "configs" / "data.yaml"

EXPECTED_CLASSES = 73

EXPECTED_SPLIT_COUNTS = {
    "train": 280,
    "val": 60,
    "test": 60,
}

MIN_LABELS_PER_IMAGE = 3
MAX_LABELS_PER_IMAGE = 5

EXPECTED_TOTAL_IMAGES = 400
EXPECTED_TOTAL_LABEL_FILES = 400

# Based on the current deterministic dataset generation run.
EXPECTED_TOTAL_BOXES = 1606


def main():

    print("\nYOLO DATASET VALIDATION")
    print("=" * 60)

    # --------------------------------------------------
    # Load YAML
    # --------------------------------------------------

    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Missing configuration file: {CONFIG_PATH}"
        )

    with CONFIG_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    required_yaml_keys = {
        "path",
        "train",
        "val",
        "test",
        "nc",
        "names",
    }

    if not required_yaml_keys.issubset(
        config.keys()
    ):
        missing = (
            required_yaml_keys
            - set(config.keys())
        )

        raise ValueError(
            f"data.yaml missing required keys: "
            f"{sorted(missing)}"
        )

    # --------------------------------------------------
    # Validate class configuration
    # --------------------------------------------------

    if config["nc"] != EXPECTED_CLASSES:
        raise ValueError(
            f"Expected {EXPECTED_CLASSES} classes, "
            f"found {config['nc']}."
        )

    names = config["names"]

    if len(names) != EXPECTED_CLASSES:
        raise ValueError(
            "Class-name count does not match nc."
        )

    dataset_root = (
        PROJECT_ROOT
        / config["path"]
    ).resolve()

    if not dataset_root.exists():
        raise FileNotFoundError(
            f"Dataset root does not exist: "
            f"{dataset_root}"
        )

    print(
        f"Dataset root: {dataset_root}"
    )

    print(
        f"Classes:      {config['nc']}"
    )

    # --------------------------------------------------
    # Global counters
    # --------------------------------------------------

    total_images = 0
    total_labels = 0
    total_boxes = 0

    global_class_counts = Counter()

    labels_per_image_distribution = Counter()

    global_image_stems = set()
    global_label_stems = set()

    # --------------------------------------------------
    # Validate each split
    # --------------------------------------------------

    for split in (
        "train",
        "val",
        "test",
    ):

        image_dir = (
            dataset_root
            / "images"
            / split
        )

        label_dir = (
            dataset_root
            / "labels"
            / split
        )

        if not image_dir.exists():
            raise FileNotFoundError(
                f"Missing image folder: "
                f"{image_dir}"
            )

        if not label_dir.exists():
            raise FileNotFoundError(
                f"Missing label folder: "
                f"{label_dir}"
            )

        # --------------------------------------------------
        # Find final production files
        # --------------------------------------------------

        images = sorted(
            image_dir.glob(
                f"{split}_??????.jpg"
            )
        )

        labels = sorted(
            label_dir.glob(
                f"{split}_??????.txt"
            )
        )

        expected_count = (
            EXPECTED_SPLIT_COUNTS[
                split
            ]
        )

        if len(images) != expected_count:
            raise ValueError(
                f"{split}: expected "
                f"{expected_count} final images, "
                f"found {len(images)}."
            )

        if len(labels) != expected_count:
            raise ValueError(
                f"{split}: expected "
                f"{expected_count} final labels, "
                f"found {len(labels)}."
            )

        # --------------------------------------------------
        # Pairing checks
        # --------------------------------------------------

        image_stems = {
            path.stem
            for path in images
        }

        label_stems = {
            path.stem
            for path in labels
        }

        missing_labels = (
            image_stems
            - label_stems
        )

        missing_images = (
            label_stems
            - image_stems
        )

        if missing_labels:
            raise ValueError(
                f"{split}: images without "
                f"labels: "
                f"{sorted(missing_labels)}"
            )

        if missing_images:
            raise ValueError(
                f"{split}: labels without "
                f"images: "
                f"{sorted(missing_images)}"
            )

        # Ensure names are unique globally.
        overlap_images = (
            global_image_stems
            & image_stems
        )

        overlap_labels = (
            global_label_stems
            & label_stems
        )

        if overlap_images:
            raise ValueError(
                "Duplicate image stem detected "
                "across splits: "
                f"{sorted(overlap_images)}"
            )

        if overlap_labels:
            raise ValueError(
                "Duplicate label stem detected "
                "across splits: "
                f"{sorted(overlap_labels)}"
            )

        global_image_stems.update(
            image_stems
        )

        global_label_stems.update(
            label_stems
        )

        # --------------------------------------------------
        # Validate label contents
        # --------------------------------------------------

        split_class_counts = Counter()
        split_boxes = 0

        split_label_distribution = Counter()

        for label_path in labels:

            lines = [
                line.strip()
                for line
                in label_path.read_text(
                    encoding="utf-8"
                ).splitlines()
                if line.strip()
            ]

            label_count = len(lines)

            if not (
                MIN_LABELS_PER_IMAGE
                <= label_count
                <= MAX_LABELS_PER_IMAGE
            ):
                raise ValueError(
                    f"{label_path.name}: "
                    f"expected "
                    f"{MIN_LABELS_PER_IMAGE}-"
                    f"{MAX_LABELS_PER_IMAGE} "
                    f"labels, found "
                    f"{label_count}."
                )

            split_label_distribution[
                label_count
            ] += 1

            labels_per_image_distribution[
                label_count
            ] += 1

            classes_in_image = []

            for (
                line_number,
                line,
            ) in enumerate(
                lines,
                start=1,
            ):

                parts = line.split()

                if len(parts) != 5:
                    raise ValueError(
                        f"{label_path.name}, "
                        f"line {line_number}: "
                        f"expected 5 values, "
                        f"found {len(parts)}."
                    )

                # --------------------------------------------------
                # Parse class ID
                # --------------------------------------------------

                try:
                    class_id = int(
                        parts[0]
                    )
                except ValueError as exc:
                    raise ValueError(
                        f"{label_path.name}, "
                        f"line {line_number}: "
                        f"invalid class ID "
                        f"{parts[0]}."
                    ) from exc

                if not (
                    0
                    <= class_id
                    < EXPECTED_CLASSES
                ):
                    raise ValueError(
                        f"{label_path.name}: "
                        f"invalid class ID "
                        f"{class_id}."
                    )

                # --------------------------------------------------
                # Parse YOLO box
                # --------------------------------------------------

                try:
                    (
                        x_center,
                        y_center,
                        width,
                        height,
                    ) = map(
                        float,
                        parts[1:],
                    )
                except ValueError as exc:
                    raise ValueError(
                        f"{label_path.name}, "
                        f"line {line_number}: "
                        f"invalid numeric "
                        f"YOLO coordinates."
                    ) from exc

                if not (
                    0.0 <= x_center <= 1.0
                    and 0.0 <= y_center <= 1.0
                    and 0.0 < width <= 1.0
                    and 0.0 < height <= 1.0
                ):
                    raise ValueError(
                        f"{label_path.name}: "
                        f"invalid normalized box "
                        f"{x_center}, "
                        f"{y_center}, "
                        f"{width}, "
                        f"{height}"
                    )

                # Stronger boundary check:
                # box edges must stay inside image.
                x_min = (
                    x_center
                    - width / 2
                )

                x_max = (
                    x_center
                    + width / 2
                )

                y_min = (
                    y_center
                    - height / 2
                )

                y_max = (
                    y_center
                    + height / 2
                )

                tolerance = 1e-6

                if not (
                    -tolerance
                    <= x_min
                    <= 1.0 + tolerance
                    and -tolerance
                    <= y_min
                    <= 1.0 + tolerance
                    and -tolerance
                    <= x_max
                    <= 1.0 + tolerance
                    and -tolerance
                    <= y_max
                    <= 1.0 + tolerance
                ):
                    raise ValueError(
                        f"{label_path.name}: "
                        f"bounding box extends "
                        f"outside normalized image "
                        f"bounds."
                    )

                classes_in_image.append(
                    class_id
                )

                split_class_counts[
                    class_id
                ] += 1

                global_class_counts[
                    class_id
                ] += 1

                split_boxes += 1

            # --------------------------------------------------
            # Unique class per composite
            # --------------------------------------------------

            if len(
                classes_in_image
            ) != len(
                set(classes_in_image)
            ):
                raise ValueError(
                    f"{label_path.name}: "
                    f"duplicate class inside "
                    f"one composite."
                )

        # --------------------------------------------------
        # Split class coverage
        # --------------------------------------------------

        missing_classes = (
            set(
                range(
                    EXPECTED_CLASSES
                )
            )
            - set(
                split_class_counts
            )
        )

        if missing_classes:
            raise ValueError(
                f"{split}: missing classes "
                f"{sorted(missing_classes)}"
            )

        # --------------------------------------------------
        # Split report
        # --------------------------------------------------

        print(
            f"\n{split.upper()}"
        )

        print(
            f"  Images:       "
            f"{len(images)}"
        )

        print(
            f"  Labels:       "
            f"{len(labels)}"
        )

        print(
            f"  Boxes:        "
            f"{split_boxes}"
        )

        print(
            f"  3-object imgs: "
            f"{split_label_distribution[3]}"
        )

        print(
            f"  4-object imgs: "
            f"{split_label_distribution[4]}"
        )

        print(
            f"  5-object imgs: "
            f"{split_label_distribution[5]}"
        )

        print(
            f"  Class range:  "
            f"{min(split_class_counts.values())}"
            f"–"
            f"{max(split_class_counts.values())}"
        )

        print(
            "  Pairing:      PASS"
        )

        print(
            "  Label format: PASS"
        )

        print(
            "  Coordinates:  PASS"
        )

        print(
            "  Unique class: PASS"
        )

        print(
            "  Class cover:  PASS"
        )

        total_images += len(
            images
        )

        total_labels += len(
            labels
        )

        total_boxes += (
            split_boxes
        )

    # --------------------------------------------------
    # Global checks
    # --------------------------------------------------

    if total_images != EXPECTED_TOTAL_IMAGES:
        raise ValueError(
            f"Expected "
            f"{EXPECTED_TOTAL_IMAGES} "
            f"total images, found "
            f"{total_images}."
        )

    if total_labels != EXPECTED_TOTAL_LABEL_FILES:
        raise ValueError(
            f"Expected "
            f"{EXPECTED_TOTAL_LABEL_FILES} "
            f"label files, found "
            f"{total_labels}."
        )

    if total_boxes != EXPECTED_TOTAL_BOXES:
        raise ValueError(
            f"Expected "
            f"{EXPECTED_TOTAL_BOXES} "
            f"bounding boxes, found "
            f"{total_boxes}."
        )

    if set(
        global_class_counts
    ) != set(
        range(
            EXPECTED_CLASSES
        )
    ):
        raise ValueError(
            "Global dataset does not "
            "contain all 73 classes."
        )

    # --------------------------------------------------
    # Final report
    # --------------------------------------------------

    print(
        "\nGLOBAL SUMMARY"
    )

    print(
        f"  Images:         "
        f"{total_images}"
    )

    print(
        f"  Label files:    "
        f"{total_labels}"
    )

    print(
        f"  Bounding boxes: "
        f"{total_boxes}"
    )

    print(
        f"  Classes:        "
        f"{len(global_class_counts)}"
    )

    print(
        "\nObjects per composite:"
    )

    print(
        f"  3 objects: "
        f"{labels_per_image_distribution[3]}"
    )

    print(
        f"  4 objects: "
        f"{labels_per_image_distribution[4]}"
    )

    print(
        f"  5 objects: "
        f"{labels_per_image_distribution[5]}"
    )

    print(
        "\nFINAL CHECKS"
    )

    print(
        "  data.yaml:              PASS"
    )

    print(
        "  Image-label pairs:      PASS"
    )

    print(
        "  YOLO format:            PASS"
    )

    print(
        "  Coordinate ranges:      PASS"
    )

    print(
        "  Boxes inside image:     PASS"
    )

    print(
        "  3-5 labels per image:   PASS"
    )

    print(
        "  Unique classes/image:   PASS"
    )

    print(
        "  Class IDs 0-72:         PASS"
    )

    print(
        "  All classes present:    PASS"
    )

    print(
        "  Split counts correct:   PASS"
    )

    print(
        "  Total box count:        PASS"
    )

    print(
        "\nYOLO dataset validation complete."
    )


if __name__ == "__main__":
    main()