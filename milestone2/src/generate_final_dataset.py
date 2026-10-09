from pathlib import Path
import csv
import random
import re
from collections import Counter, defaultdict

from PIL import Image, ImageOps


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

METADATA_DIR = PROJECT_ROOT / "milestone2" / "metadata"
GENERATED_DIR = PROJECT_ROOT / "milestone2" / "data" / "generated"

SOURCE_SPLIT_MANIFEST = (
    METADATA_DIR / "source_split_manifest.csv"
)

FINAL_MANIFEST = (
    METADATA_DIR / "composite_manifest.csv"
)

CLASS_DISTRIBUTION = (
    METADATA_DIR / "class_distribution.csv"
)

IMAGE_DIR = GENERATED_DIR / "images"
LABEL_DIR = GENERATED_DIR / "labels"

RANDOM_SEED = 42


# ------------------------------------------------------------
# Group members
# ------------------------------------------------------------

GROUP_MEMBERS = [
    "Hemanth Rajaboyina",
    "Jaya Raghu Ram Penugonda",
    "Nikhil Kudupudi",
    "Sri Lakshmi Swetha Jalluri",
]


# ------------------------------------------------------------
# Required composite counts PER student
# ------------------------------------------------------------

COMPOSITES_PER_MEMBER = {
    "train": 70,
    "val": 15,
    "test": 15,
}

# 70 + 15 + 15 = 100 images per student


# ------------------------------------------------------------
# Composite geometry
# ------------------------------------------------------------

OBJECTS_PER_COMPOSITE = 4

OUTPUT_WIDTH = 640
OUTPUT_HEIGHT = 640

GRID_ROWS = 2
GRID_COLS = 2

CELL_WIDTH = OUTPUT_WIDTH // GRID_COLS
CELL_HEIGHT = OUTPUT_HEIGHT // GRID_ROWS

JPEG_QUALITY = 95


# ============================================================
# Helpers
# ============================================================

def read_csv(path):
    """Read CSV file into a list of dictionaries."""
    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        return list(csv.DictReader(file))


def ensure_output_directories():
    """Create generated YOLO folders."""
    for split in ("train", "val", "test"):
        (IMAGE_DIR / split).mkdir(
            parents=True,
            exist_ok=True,
        )

        (LABEL_DIR / split).mkdir(
            parents=True,
            exist_ok=True,
        )


def clean_previous_final_outputs():
    """
    Remove ONLY final production outputs.

    Debug files such as:
        train_debug_0001.jpg
    are preserved.
    """

    final_image_pattern = re.compile(
        r"^(train|val|test)_\d{6}\.jpg$"
    )

    final_label_pattern = re.compile(
        r"^(train|val|test)_\d{6}\.txt$"
    )

    for split in ("train", "val", "test"):

        image_folder = IMAGE_DIR / split
        label_folder = LABEL_DIR / split

        for path in image_folder.iterdir():
            if (
                path.is_file()
                and final_image_pattern.fullmatch(path.name)
            ):
                path.unlink()

        for path in label_folder.iterdir():
            if (
                path.is_file()
                and final_label_pattern.fullmatch(path.name)
            ):
                path.unlink()


def prepare_source_image(source_path):
    """
    Convert a source image to RGB and fit it into
    one 320 × 320 grid region.

    ImageOps.fit preserves the output geometry and
    center-crops where required.
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
    Get pixel bounds for one location in the 2×2 grid.

    Returns:
        x_min, y_min, x_max, y_max
    """

    row = cell_index // GRID_COLS
    col = cell_index % GRID_COLS

    x_min = col * CELL_WIDTH
    y_min = row * CELL_HEIGHT

    x_max = x_min + CELL_WIDTH
    y_max = y_min + CELL_HEIGHT

    return (
        x_min,
        y_min,
        x_max,
        y_max,
    )


def pixel_box_to_yolo(
    x_min,
    y_min,
    x_max,
    y_max,
):
    """
    Convert pixel bounding box into normalized
    YOLO format:

        x_center y_center width height
    """

    width = x_max - x_min
    height = y_max - y_min

    x_center = x_min + width / 2
    y_center = y_min + height / 2

    return (
        x_center / OUTPUT_WIDTH,
        y_center / OUTPUT_HEIGHT,
        width / OUTPUT_WIDTH,
        height / OUTPUT_HEIGHT,
    )


def validate_yolo_box(
    x_center,
    y_center,
    width,
    height,
):
    """Verify normalized YOLO coordinates."""

    if not (
        0 <= x_center <= 1
        and 0 <= y_center <= 1
        and 0 < width <= 1
        and 0 < height <= 1
    ):
        raise ValueError(
            "Invalid normalized YOLO box: "
            f"{x_center}, {y_center}, "
            f"{width}, {height}"
        )


# ============================================================
# Load and prepare source pools
# ============================================================

def load_source_pools(rng):

    if not SOURCE_SPLIT_MANIFEST.exists():
        raise FileNotFoundError(
            "source_split_manifest.csv is missing. "
            "Run create_source_splits.py first."
        )

    rows = read_csv(SOURCE_SPLIT_MANIFEST)

    required_columns = {
        "source_path",
        "object_id",
        "object_name",
        "class_index",
        "md5",
        "split",
    }

    if not rows:
        raise ValueError(
            "source_split_manifest.csv is empty."
        )

    if not required_columns.issubset(rows[0].keys()):
        raise ValueError(
            "Source split manifest is missing "
            "required columns."
        )

    pools = {
        "train": defaultdict(list),
        "val": defaultdict(list),
        "test": defaultdict(list),
    }

    for row in rows:

        split = row["split"]
        object_id = row["object_id"]

        if split not in pools:
            raise ValueError(
                f"Unexpected split: {split}"
            )

        pools[split][object_id].append(row)

    # Sort first, then shuffle reproducibly.
    for split in pools:

        for object_id in pools[split]:

            class_rows = sorted(
                pools[split][object_id],
                key=lambda r: r["source_path"],
            )

            rng.shuffle(class_rows)

            pools[split][object_id] = class_rows

    return pools


# ============================================================
# Balanced class selection
# ============================================================

def select_balanced_classes(
    split,
    pools,
    source_pointer,
    class_usage,
    rng,
):
    """
    Select four different classes.

    Classes with the lowest current usage are preferred.
    Random tie-breaking keeps generation deterministic
    while avoiding a rigid repeated ordering.
    """

    candidates = []

    for object_id in sorted(pools[split].keys()):

        available_count = len(
            pools[split][object_id]
        )

        used_count = source_pointer[
            split
        ][object_id]

        remaining = available_count - used_count

        if remaining > 0:
            candidates.append(object_id)

    if len(candidates) < OBJECTS_PER_COMPOSITE:
        raise RuntimeError(
            f"Not enough classes with remaining "
            f"source images in {split}."
        )

    # Random value is used only as a deterministic
    # tie breaker because rng has a fixed seed.
    scored = [
        (
            class_usage[split][object_id],
            rng.random(),
            object_id,
        )
        for object_id in candidates
    ]

    scored.sort()

    selected = [
        item[2]
        for item in scored[:OBJECTS_PER_COMPOSITE]
    ]

    if len(set(selected)) != OBJECTS_PER_COMPOSITE:
        raise RuntimeError(
            "Duplicate class selected within "
            "one composite."
        )

    return selected


def take_source_row(
    split,
    object_id,
    pools,
    source_pointer,
):
    """Take the next unused source image for a class."""

    pointer = source_pointer[
        split
    ][object_id]

    class_pool = pools[
        split
    ][object_id]

    if pointer >= len(class_pool):
        raise RuntimeError(
            f"No unused source images remain for "
            f"{object_id} in {split}."
        )

    row = class_pool[pointer]

    source_pointer[
        split
    ][object_id] += 1

    return row


# ============================================================
# Generate one composite
# ============================================================

def generate_one_composite(
    split,
    composite_number,
    member,
    selected_rows,
):

    image_name = (
        f"{split}_{composite_number:06d}.jpg"
    )

    label_name = (
        f"{split}_{composite_number:06d}.txt"
    )

    image_path = (
        IMAGE_DIR / split / image_name
    )

    label_path = (
        LABEL_DIR / split / label_name
    )

    composite = Image.new(
        "RGB",
        (OUTPUT_WIDTH, OUTPUT_HEIGHT),
    )

    annotation_lines = []
    manifest_rows = []

    for position, row in enumerate(
        selected_rows
    ):

        source_path = (
            PROJECT_ROOT / row["source_path"]
        )

        if not source_path.exists():
            raise FileNotFoundError(
                f"Missing source image: "
                f"{source_path}"
            )

        tile = prepare_source_image(
            source_path
        )

        (
            x_min,
            y_min,
            x_max,
            y_max,
        ) = get_cell_box(position)

        composite.paste(
            tile,
            (x_min, y_min),
        )

        (
            x_center,
            y_center,
            width_norm,
            height_norm,
        ) = pixel_box_to_yolo(
            x_min,
            y_min,
            x_max,
            y_max,
        )

        validate_yolo_box(
            x_center,
            y_center,
            width_norm,
            height_norm,
        )

        class_index = int(
            row["class_index"]
        )

        annotation_lines.append(
            f"{class_index} "
            f"{x_center:.6f} "
            f"{y_center:.6f} "
            f"{width_norm:.6f} "
            f"{height_norm:.6f}"
        )

        manifest_rows.append(
            {
                "composite_image":
                    image_path.relative_to(
                        PROJECT_ROOT
                    ).as_posix(),

                "label_file":
                    label_path.relative_to(
                        PROJECT_ROOT
                    ).as_posix(),

                "split": split,

                "generator_member": member,

                "object_position":
                    position + 1,

                "source_path":
                    row["source_path"],

                "source_md5":
                    row["md5"],

                "object_id":
                    row["object_id"],

                "object_name":
                    row["object_name"],

                "class_index":
                    class_index,

                "x_min":
                    x_min,

                "y_min":
                    y_min,

                "x_max":
                    x_max,

                "y_max":
                    y_max,

                "x_center_norm":
                    f"{x_center:.6f}",

                "y_center_norm":
                    f"{y_center:.6f}",

                "width_norm":
                    f"{width_norm:.6f}",

                "height_norm":
                    f"{height_norm:.6f}",
            }
        )

    composite.save(
        image_path,
        format="JPEG",
        quality=JPEG_QUALITY,
    )

    label_path.write_text(
        "\n".join(annotation_lines) + "\n",
        encoding="utf-8",
    )

    return manifest_rows


# ============================================================
# Integrity checks
# ============================================================

def run_integrity_checks(
    manifest_rows,
):

    # --------------------------------------------------------
    # No source reuse
    # --------------------------------------------------------

    source_paths = [
        row["source_path"]
        for row in manifest_rows
    ]

    if len(source_paths) != len(
        set(source_paths)
    ):
        raise ValueError(
            "A source image was reused."
        )

    # --------------------------------------------------------
    # No MD5 reuse
    # --------------------------------------------------------

    hashes = [
        row["source_md5"]
        for row in manifest_rows
    ]

    if len(hashes) != len(
        set(hashes)
    ):
        raise ValueError(
            "An exact duplicate source image "
            "was reused."
        )

    # --------------------------------------------------------
    # Every composite has four annotations
    # --------------------------------------------------------

    composite_counts = Counter(
        row["composite_image"]
        for row in manifest_rows
    )

    invalid = {
        image: count
        for image, count
        in composite_counts.items()
        if count != OBJECTS_PER_COMPOSITE
    }

    if invalid:
        raise ValueError(
            "One or more composites do not "
            "contain exactly four annotations."
        )

    # --------------------------------------------------------
    # Four unique classes per composite
    # --------------------------------------------------------

    classes_by_composite = defaultdict(
        set
    )

    for row in manifest_rows:
        classes_by_composite[
            row["composite_image"]
        ].add(
            row["object_id"]
        )

    for image, classes in (
        classes_by_composite.items()
    ):
        if len(classes) != (
            OBJECTS_PER_COMPOSITE
        ):
            raise ValueError(
                f"{image} does not contain "
                f"four unique classes."
            )


# ============================================================
# Save manifests
# ============================================================

def save_final_manifest(
    manifest_rows,
):

    fieldnames = [
        "composite_image",
        "label_file",
        "split",
        "generator_member",
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

    with FINAL_MANIFEST.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(
            manifest_rows
        )


def save_class_distribution(
    manifest_rows,
):

    counts = Counter(
        (
            row["split"],
            row["object_id"],
            row["object_name"],
        )
        for row in manifest_rows
    )

    with CLASS_DISTRIBUTION.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:

        fieldnames = [
            "split",
            "object_id",
            "object_name",
            "annotation_count",
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for (
            split,
            object_id,
            object_name,
        ), count in sorted(
            counts.items()
        ):

            writer.writerow(
                {
                    "split": split,
                    "object_id": object_id,
                    "object_name": object_name,
                    "annotation_count": count,
                }
            )


# ============================================================
# Main
# ============================================================

def main():

    print(
        "\nFINAL MILESTONE 2 DATASET GENERATION"
    )
    print("=" * 65)

    rng = random.Random(
        RANDOM_SEED
    )

    ensure_output_directories()
    clean_previous_final_outputs()

    pools = load_source_pools(
        rng
    )

    source_pointer = {
        split: defaultdict(int)
        for split in (
            "train",
            "val",
            "test",
        )
    }

    class_usage = {
        split: Counter()
        for split in (
            "train",
            "val",
            "test",
        )
    }

    manifest_rows = []

    split_composite_number = {
        "train": 0,
        "val": 0,
        "test": 0,
    }

    # --------------------------------------------------------
    # Generate exactly 100 composites per student
    # --------------------------------------------------------

    for member in GROUP_MEMBERS:

        print(
            f"\nGenerating images for: "
            f"{member}"
        )

        for split in (
            "train",
            "val",
            "test",
        ):

            requested = (
                COMPOSITES_PER_MEMBER[
                    split
                ]
            )

            for _ in range(
                requested
            ):

                split_composite_number[
                    split
                ] += 1

                selected_classes = (
                    select_balanced_classes(
                        split=split,
                        pools=pools,
                        source_pointer=source_pointer,
                        class_usage=class_usage,
                        rng=rng,
                    )
                )

                selected_rows = []

                for object_id in (
                    selected_classes
                ):

                    row = take_source_row(
                        split=split,
                        object_id=object_id,
                        pools=pools,
                        source_pointer=source_pointer,
                    )

                    selected_rows.append(
                        row
                    )

                    class_usage[
                        split
                    ][object_id] += 1

                rows_created = (
                    generate_one_composite(
                        split=split,
                        composite_number=(
                            split_composite_number[
                                split
                            ]
                        ),
                        member=member,
                        selected_rows=selected_rows,
                    )
                )

                manifest_rows.extend(
                    rows_created
                )

    # --------------------------------------------------------
    # Validate everything
    # --------------------------------------------------------

    run_integrity_checks(
        manifest_rows
    )

    save_final_manifest(
        manifest_rows
    )

    save_class_distribution(
        manifest_rows
    )

    # --------------------------------------------------------
    # Summaries
    # --------------------------------------------------------

    composite_counts = Counter()

    member_composites = defaultdict(
        set
    )

    split_composites = defaultdict(
        set
    )

    for row in manifest_rows:

        image = row[
            "composite_image"
        ]

        member_composites[
            row["generator_member"]
        ].add(image)

        split_composites[
            row["split"]
        ].add(image)

    total_composites = len(
        {
            row["composite_image"]
            for row in manifest_rows
        }
    )

    total_annotations = len(
        manifest_rows
    )

    print("\nDataset summary:")
    print(
        f"  Train composites: "
        f"{len(split_composites['train'])}"
    )
    print(
        f"  Val composites:   "
        f"{len(split_composites['val'])}"
    )
    print(
        f"  Test composites:  "
        f"{len(split_composites['test'])}"
    )
    print(
        f"  Total composites: "
        f"{total_composites}"
    )

    print(
        f"  Total annotations: "
        f"{total_annotations}"
    )

    print("\nPer-student generation:")

    for member in GROUP_MEMBERS:
        count = len(
            member_composites[
                member
            ]
        )

        print(
            f"  {member}: {count}"
        )

        if count < 100:
            raise ValueError(
                f"{member} has fewer "
                f"than 100 composites."
            )

    print("\nClass-balance check:")

    for split in (
        "train",
        "val",
        "test",
    ):

        counts = list(
            class_usage[
                split
            ].values()
        )

        print(
            f"  {split}: "
            f"min={min(counts)}, "
            f"max={max(counts)}"
        )

    print("\nIntegrity checks:")
    print(
        "  No source reuse: PASS"
    )
    print(
        "  No MD5 reuse: PASS"
    )
    print(
        "  Four unique classes/image: PASS"
    )
    print(
        "  Four annotations/image: PASS"
    )
    print(
        "  Student minimum requirement: PASS"
    )

    print("\nReproducibility:")
    print(
        f"  Random seed: "
        f"{RANDOM_SEED}"
    )

    print("\nMetadata files:")
    print(
        f"  "
        f"{FINAL_MANIFEST.relative_to(PROJECT_ROOT)}"
    )
    print(
        f"  "
        f"{CLASS_DISTRIBUTION.relative_to(PROJECT_ROOT)}"
    )

    print(
        "\nFinal dataset generation complete."
    )


if __name__ == "__main__":
    main()