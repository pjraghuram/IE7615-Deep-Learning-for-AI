from pathlib import Path
import csv
import random

from PIL import Image, ImageOps


# --------------------------------------------------
# Configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

METADATA_DIR = PROJECT_ROOT / "milestone2" / "metadata"
GENERATED_DIR = PROJECT_ROOT / "milestone2" / "data" / "generated"

SPLIT_MANIFEST = METADATA_DIR / "source_split_manifest.csv"
COMPOSITE_MANIFEST = METADATA_DIR / "composite_manifest_debug.csv"

IMAGE_DIR = GENERATED_DIR / "images"
LABEL_DIR = GENERATED_DIR / "labels"

RANDOM_SEED = 42

# Prototype only
NUM_COMPOSITES = {
    "train": 6,
    "val": 3,
    "test": 3,
}

# Four source images concatenated into a 2 x 2 grid
OBJECTS_PER_COMPOSITE = 4

OUTPUT_WIDTH = 640
OUTPUT_HEIGHT = 640

GRID_ROWS = 2
GRID_COLS = 2

CELL_WIDTH = OUTPUT_WIDTH // GRID_COLS
CELL_HEIGHT = OUTPUT_HEIGHT // GRID_ROWS

JPEG_QUALITY = 95


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def read_csv(path):
    """Read a CSV file into a list of dictionaries."""
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def ensure_directories():
    """Create output folders if they do not already exist."""
    for split in ("train", "val", "test"):
        (IMAGE_DIR / split).mkdir(parents=True, exist_ok=True)
        (LABEL_DIR / split).mkdir(parents=True, exist_ok=True)


def clean_debug_outputs():
    """
    Remove only outputs previously produced by this debug generator.

    This keeps repeated prototype runs reproducible without touching
    unrelated files.
    """
    for split in ("train", "val", "test"):

        for path in (IMAGE_DIR / split).glob(f"{split}_debug_*.jpg"):
            path.unlink()

        for path in (LABEL_DIR / split).glob(f"{split}_debug_*.txt"):
            path.unlink()


def load_split_rows():
    """Load and validate the source split manifest."""
    if not SPLIT_MANIFEST.exists():
        raise FileNotFoundError(
            "source_split_manifest.csv was not found. "
            "Run create_source_splits.py first."
        )

    rows = read_csv(SPLIT_MANIFEST)

    required_columns = {
        "source_path",
        "object_id",
        "object_name",
        "class_index",
        "md5",
        "split",
    }

    if not rows:
        raise ValueError("Source split manifest is empty.")

    if not required_columns.issubset(rows[0].keys()):
        raise ValueError(
            "source_split_manifest.csv is missing required columns."
        )

    return rows


def prepare_image(source_path):
    """
    Load one single-object source image and place it inside one grid cell.

    ImageOps.fit preserves the cell size consistently while cropping
    as needed to fill the 320 x 320 region.
    """
    with Image.open(source_path) as image:

        image = image.convert("RGB")

        fitted = ImageOps.fit(
            image,
            (CELL_WIDTH, CELL_HEIGHT),
            method=Image.Resampling.LANCZOS,
            centering=(0.5, 0.5),
        )

        return fitted


def get_cell_box(cell_index):
    """
    Return pixel bounding-box coordinates for a 2 x 2 grid cell.

    Format:
        x_min, y_min, x_max, y_max
    """
    row = cell_index // GRID_COLS
    col = cell_index % GRID_COLS

    x_min = col * CELL_WIDTH
    y_min = row * CELL_HEIGHT

    x_max = x_min + CELL_WIDTH
    y_max = y_min + CELL_HEIGHT

    return x_min, y_min, x_max, y_max


def pixel_box_to_yolo(x_min, y_min, x_max, y_max):
    """
    Convert pixel bounding box to normalized YOLO format:

        x_center y_center width height
    """

    box_width = x_max - x_min
    box_height = y_max - y_min

    x_center = x_min + box_width / 2
    y_center = y_min + box_height / 2

    return (
        x_center / OUTPUT_WIDTH,
        y_center / OUTPUT_HEIGHT,
        box_width / OUTPUT_WIDTH,
        box_height / OUTPUT_HEIGHT,
    )


def validate_yolo_box(x_center, y_center, width, height):
    """Validate normalized YOLO coordinates."""
    values = (x_center, y_center, width, height)

    if not all(0.0 <= value <= 1.0 for value in values):
        raise ValueError(
            f"Invalid YOLO bounding box: {values}"
        )

    if width <= 0 or height <= 0:
        raise ValueError(
            f"YOLO bounding box has invalid size: {values}"
        )


# --------------------------------------------------
# Composite generation
# --------------------------------------------------

def generate_composite(
    split,
    composite_number,
    selected_rows,
):
    """Generate one multi-object composite and its YOLO labels."""

    if len(selected_rows) != OBJECTS_PER_COMPOSITE:
        raise ValueError(
            f"Expected {OBJECTS_PER_COMPOSITE} objects, "
            f"received {len(selected_rows)}."
        )

    composite = Image.new(
        "RGB",
        (OUTPUT_WIDTH, OUTPUT_HEIGHT),
    )

    image_name = (
        f"{split}_debug_{composite_number:04d}.jpg"
    )

    label_name = (
        f"{split}_debug_{composite_number:04d}.txt"
    )

    image_path = IMAGE_DIR / split / image_name
    label_path = LABEL_DIR / split / label_name

    yolo_lines = []
    metadata_rows = []

    for cell_index, row in enumerate(selected_rows):

        source_path = PROJECT_ROOT / row["source_path"]

        if not source_path.exists():
            raise FileNotFoundError(
                f"Missing source image: {source_path}"
            )

        tile = prepare_image(source_path)

        x_min, y_min, x_max, y_max = get_cell_box(
            cell_index
        )

        composite.paste(
            tile,
            (x_min, y_min),
        )

        (
            x_center,
            y_center,
            box_width,
            box_height,
        ) = pixel_box_to_yolo(
            x_min,
            y_min,
            x_max,
            y_max,
        )

        validate_yolo_box(
            x_center,
            y_center,
            box_width,
            box_height,
        )

        class_index = int(row["class_index"])

        yolo_lines.append(
            f"{class_index} "
            f"{x_center:.6f} "
            f"{y_center:.6f} "
            f"{box_width:.6f} "
            f"{box_height:.6f}"
        )

        metadata_rows.append(
            {
                "composite_image": (
                    image_path
                    .relative_to(PROJECT_ROOT)
                    .as_posix()
                ),
                "label_file": (
                    label_path
                    .relative_to(PROJECT_ROOT)
                    .as_posix()
                ),
                "split": split,
                "object_position": cell_index + 1,
                "source_path": row["source_path"],
                "source_md5": row["md5"],
                "object_id": row["object_id"],
                "object_name": row["object_name"],
                "class_index": class_index,
                "x_min": x_min,
                "y_min": y_min,
                "x_max": x_max,
                "y_max": y_max,
                "x_center_norm": f"{x_center:.6f}",
                "y_center_norm": f"{y_center:.6f}",
                "width_norm": f"{box_width:.6f}",
                "height_norm": f"{box_height:.6f}",
            }
        )

    composite.save(
        image_path,
        format="JPEG",
        quality=JPEG_QUALITY,
    )

    label_path.write_text(
        "\n".join(yolo_lines) + "\n",
        encoding="utf-8",
    )

    return metadata_rows


# --------------------------------------------------
# Integrity checks
# --------------------------------------------------

def verify_no_cross_split_sources(metadata_rows):
    """
    Ensure no source image appears in more than one generated split.
    """

    source_to_splits = {}

    for row in metadata_rows:

        source_path = row["source_path"]
        split = row["split"]

        source_to_splits.setdefault(
            source_path,
            set(),
        ).add(split)

    violations = {
        source: splits
        for source, splits in source_to_splits.items()
        if len(splits) > 1
    }

    if violations:
        raise ValueError(
            "Source-image leakage detected across generated splits."
        )


def verify_no_cross_split_hashes(metadata_rows):
    """
    Ensure byte-identical source images cannot appear in
    multiple generated splits.
    """

    hash_to_splits = {}

    for row in metadata_rows:

        digest = row["source_md5"]
        split = row["split"]

        hash_to_splits.setdefault(
            digest,
            set(),
        ).add(split)

    violations = {
        digest: splits
        for digest, splits in hash_to_splits.items()
        if len(splits) > 1
    }

    if violations:
        raise ValueError(
            "Duplicate-image leakage detected across generated splits."
        )


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    print("\nDEBUG COMPOSITE GENERATION")
    print("=" * 60)

    ensure_directories()
    clean_debug_outputs()

    rows = load_split_rows()

    rows_by_split = {
        split: [
            row
            for row in rows
            if row["split"] == split
        ]
        for split in ("train", "val", "test")
    }

    rng = random.Random(RANDOM_SEED)

    all_metadata_rows = []

    used_source_paths = set()

    for split in ("train", "val", "test"):

        available_rows = sorted(
            rows_by_split[split],
            key=lambda row: row["source_path"],
        )

        rng.shuffle(available_rows)

        required_sources = (
            NUM_COMPOSITES[split]
            * OBJECTS_PER_COMPOSITE
        )

        if required_sources > len(available_rows):
            raise ValueError(
                f"Not enough source images in {split}. "
                f"Need {required_sources}, "
                f"found {len(available_rows)}."
            )

        split_pointer = 0

        print(
            f"\nGenerating {NUM_COMPOSITES[split]} "
            f"{split} composites..."
        )

        for composite_number in range(
            1,
            NUM_COMPOSITES[split] + 1,
        ):

            selected_rows = []

            selected_classes = set()

            # Prefer four different object classes
            # inside each composite.
            while len(selected_rows) < OBJECTS_PER_COMPOSITE:

                if split_pointer >= len(available_rows):
                    raise RuntimeError(
                        f"Ran out of unused {split} source images."
                    )

                candidate = available_rows[split_pointer]
                split_pointer += 1

                if candidate["source_path"] in used_source_paths:
                    continue

                if candidate["object_id"] in selected_classes:
                    continue

                selected_rows.append(candidate)
                selected_classes.add(
                    candidate["object_id"]
                )

                used_source_paths.add(
                    candidate["source_path"]
                )

            metadata_rows = generate_composite(
                split=split,
                composite_number=composite_number,
                selected_rows=selected_rows,
            )

            all_metadata_rows.extend(metadata_rows)

    # --------------------------------------------------
    # Validate generated prototype
    # --------------------------------------------------

    verify_no_cross_split_sources(
        all_metadata_rows
    )

    verify_no_cross_split_hashes(
        all_metadata_rows
    )

    # --------------------------------------------------
    # Save composite manifest
    # --------------------------------------------------

    fieldnames = [
        "composite_image",
        "label_file",
        "split",
        "object_position",
        "source_path",
        "source_md5",
        "object_id",
        "object_name",
        "class_index",
        "x_min",
        "y_min",
        "x_max",
        "y_max",
        "x_center_norm",
        "y_center_norm",
        "width_norm",
        "height_norm",
    ]

    with COMPOSITE_MANIFEST.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(all_metadata_rows)

    # --------------------------------------------------
    # Final validation summary
    # --------------------------------------------------

    expected_composites = sum(
        NUM_COMPOSITES.values()
    )

    expected_objects = (
        expected_composites
        * OBJECTS_PER_COMPOSITE
    )

    generated_images = []

    generated_labels = []

    for split in ("train", "val", "test"):

        generated_images.extend(
            (IMAGE_DIR / split).glob(
                f"{split}_debug_*.jpg"
            )
        )

        generated_labels.extend(
            (LABEL_DIR / split).glob(
                f"{split}_debug_*.txt"
            )
        )

    if len(generated_images) != expected_composites:
        raise ValueError(
            "Unexpected number of generated images."
        )

    if len(generated_labels) != expected_composites:
        raise ValueError(
            "Unexpected number of generated label files."
        )

    if len(all_metadata_rows) != expected_objects:
        raise ValueError(
            "Unexpected number of object annotations."
        )

    print("\nGeneration summary:")
    print(f"  Train composites: {NUM_COMPOSITES['train']}")
    print(f"  Val composites:   {NUM_COMPOSITES['val']}")
    print(f"  Test composites:  {NUM_COMPOSITES['test']}")
    print(f"  Total composites: {expected_composites}")
    print(f"  Objects/image:    {OBJECTS_PER_COMPOSITE}")
    print(f"  Total boxes:      {expected_objects}")

    print("\nOutput geometry:")
    print(
        f"  Composite size: "
        f"{OUTPUT_WIDTH} x {OUTPUT_HEIGHT}"
    )
    print(
        f"  Grid: "
        f"{GRID_ROWS} x {GRID_COLS}"
    )
    print(
        f"  Cell size: "
        f"{CELL_WIDTH} x {CELL_HEIGHT}"
    )

    print("\nIntegrity checks:")
    print("  Image/label count: PASS")
    print("  Source split isolation: PASS")
    print("  MD5 split isolation: PASS")
    print("  YOLO coordinate validation: PASS")

    print("\nMetadata:")
    print(
        f"  {COMPOSITE_MANIFEST.relative_to(PROJECT_ROOT)}"
    )

    print("\nDebug composite generation complete.")


if __name__ == "__main__":
    main()