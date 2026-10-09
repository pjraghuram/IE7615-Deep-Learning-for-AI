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
EXPECTED_CLASS_COUNT = 73


# ------------------------------------------------------------
# TA-required exclusions
# ------------------------------------------------------------

EXCLUDED_SOURCE_IMAGES = {
    "images_OBJ054/OBJ054_017.jpg",
}


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
# Composite counts
# ------------------------------------------------------------

COMPOSITES_PER_MEMBER = {
    "train": 70,
    "val": 15,
    "test": 15,
}


# ------------------------------------------------------------
# Composite geometry
# ------------------------------------------------------------

MIN_OBJECTS_PER_COMPOSITE = 3
MAX_OBJECTS_PER_COMPOSITE = 5

TILE_SIZE = 224
JPEG_QUALITY = 95


# ============================================================
# Basic helpers
# ============================================================

def read_csv(path):
    """Read a CSV file into a list of dictionaries."""

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        return list(
            csv.DictReader(file)
        )


def ensure_output_directories():
    """Create YOLO output directories."""

    for split in (
        "train",
        "val",
        "test",
    ):
        (
            IMAGE_DIR / split
        ).mkdir(
            parents=True,
            exist_ok=True,
        )

        (
            LABEL_DIR / split
        ).mkdir(
            parents=True,
            exist_ok=True,
        )


def clean_previous_final_outputs():
    """
    Remove only production outputs.

    Debug files are left untouched.
    """

    image_pattern = re.compile(
        r"^(train|val|test)_\d{6}\.jpg$"
    )

    label_pattern = re.compile(
        r"^(train|val|test)_\d{6}\.txt$"
    )

    for split in (
        "train",
        "val",
        "test",
    ):

        image_folder = (
            IMAGE_DIR / split
        )

        label_folder = (
            LABEL_DIR / split
        )

        for path in image_folder.iterdir():

            if (
                path.is_file()
                and image_pattern.fullmatch(
                    path.name
                )
            ):
                path.unlink()

        for path in label_folder.iterdir():

            if (
                path.is_file()
                and label_pattern.fullmatch(
                    path.name
                )
            ):
                path.unlink()


# ============================================================
# Compact layout logic
# ============================================================

def get_layout(object_count):
    """
    Return compact grid geometry.

    3 objects:
        2 x 2 grid
        448 x 448 canvas
        1 empty cell

    4 objects:
        2 x 2 grid
        448 x 448 canvas

    5 objects:
        2 x 3 grid
        672 x 448 canvas
        1 empty cell
    """

    if object_count in (
        3,
        4,
    ):
        rows = 2
        cols = 2

    elif object_count == 5:
        rows = 2
        cols = 3

    else:
        raise ValueError(
            f"Unsupported object count: "
            f"{object_count}. "
            f"Expected 3, 4, or 5."
        )

    canvas_width = (
        cols * TILE_SIZE
    )

    canvas_height = (
        rows * TILE_SIZE
    )

    return (
        rows,
        cols,
        canvas_width,
        canvas_height,
    )


def prepare_source_image(
    source_path,
):
    """
    Convert one source image to RGB and fit
    it into a 224 x 224 tile.
    """

    with Image.open(
        source_path
    ) as image:

        image = image.convert(
            "RGB"
        )

        fitted = ImageOps.fit(
            image,
            (
                TILE_SIZE,
                TILE_SIZE,
            ),
            method=Image.Resampling.LANCZOS,
            centering=(
                0.5,
                0.5,
            ),
        )

        return fitted


def choose_grid_cells(
    object_count,
    rows,
    cols,
    rng,
):
    """
    Choose non-overlapping grid cells.

    Because each object occupies exactly one cell,
    overlap is impossible by construction.
    """

    all_cells = [
        (
            row,
            col,
        )
        for row in range(
            rows
        )
        for col in range(
            cols
        )
    ]

    if object_count > len(
        all_cells
    ):
        raise ValueError(
            f"Cannot place "
            f"{object_count} objects "
            f"in a {rows}x{cols} grid."
        )

    selected_cells = (
        rng.sample(
            all_cells,
            object_count,
        )
    )

    return selected_cells


def cell_to_pixel_box(
    row,
    col,
):
    """
    Convert grid position into pixel coordinates.
    """

    x_min = (
        col * TILE_SIZE
    )

    y_min = (
        row * TILE_SIZE
    )

    x_max = (
        x_min + TILE_SIZE
    )

    y_max = (
        y_min + TILE_SIZE
    )

    return (
        x_min,
        y_min,
        x_max,
        y_max,
    )


def boxes_overlap(
    box_a,
    box_b,
):
    """
    Return True if two boxes overlap
    with positive area.

    Edge-touching is allowed.
    """

    (
        ax1,
        ay1,
        ax2,
        ay2,
    ) = box_a

    (
        bx1,
        by1,
        bx2,
        by2,
    ) = box_b

    return (
        ax1 < bx2
        and ax2 > bx1
        and ay1 < by2
        and ay2 > by1
    )


# ============================================================
# YOLO helpers
# ============================================================

def pixel_box_to_yolo(
    x_min,
    y_min,
    x_max,
    y_max,
    canvas_width,
    canvas_height,
):
    """
    Convert pixel coordinates to normalized
    YOLO format:

        x_center y_center width height
    """

    box_width = (
        x_max - x_min
    )

    box_height = (
        y_max - y_min
    )

    x_center = (
        x_min
        + box_width / 2
    )

    y_center = (
        y_min
        + box_height / 2
    )

    return (
        x_center
        / canvas_width,

        y_center
        / canvas_height,

        box_width
        / canvas_width,

        box_height
        / canvas_height,
    )


def validate_yolo_box(
    x_center,
    y_center,
    width,
    height,
):
    """Validate normalized YOLO coordinates."""

    if not (
        0.0 <= x_center <= 1.0
        and 0.0 <= y_center <= 1.0
        and 0.0 < width <= 1.0
        and 0.0 < height <= 1.0
    ):
        raise ValueError(
            "Invalid YOLO box: "
            f"{x_center}, "
            f"{y_center}, "
            f"{width}, "
            f"{height}"
        )

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
        <= x_max
        <= 1.0 + tolerance

        and -tolerance
        <= y_min
        <= 1.0 + tolerance

        and -tolerance
        <= y_max
        <= 1.0 + tolerance
    ):
        raise ValueError(
            "YOLO bounding box extends "
            "outside normalized image bounds."
        )


# ============================================================
# Load source pools
# ============================================================

def load_source_pools(
    rng,
):

    if not (
        SOURCE_SPLIT_MANIFEST.exists()
    ):
        raise FileNotFoundError(
            "source_split_manifest.csv "
            "is missing. "
            "Run create_source_splits.py first."
        )

    rows = read_csv(
        SOURCE_SPLIT_MANIFEST
    )

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
            "source_split_manifest.csv "
            "is empty."
        )

    if not (
        required_columns.issubset(
            rows[0].keys()
        )
    ):
        raise ValueError(
            "source_split_manifest.csv "
            "is missing required columns."
        )

    # --------------------------------------------------------
    # TA exclusion check
    # --------------------------------------------------------

    leaked_exclusions = {
        row["source_path"]
        for row in rows
        if row["source_path"]
        in EXCLUDED_SOURCE_IMAGES
    }

    if leaked_exclusions:
        raise ValueError(
            "Excluded source image found "
            "in source split manifest: "
            + ", ".join(
                sorted(
                    leaked_exclusions
                )
            )
        )

    pools = {
        "train":
            defaultdict(list),

        "val":
            defaultdict(list),

        "test":
            defaultdict(list),
    }

    for row in rows:

        split = row[
            "split"
        ]

        object_id = row[
            "object_id"
        ]

        if split not in pools:
            raise ValueError(
                f"Unexpected split: "
                f"{split}"
            )

        class_index = int(
            row[
                "class_index"
            ]
        )

        if not (
            0
            <= class_index
            < EXPECTED_CLASS_COUNT
        ):
            raise ValueError(
                f"Invalid class index "
                f"{class_index} for "
                f"{row['source_path']}."
            )

        pools[
            split
        ][
            object_id
        ].append(
            row
        )

    # --------------------------------------------------------
    # Confirm every split contains all classes
    # --------------------------------------------------------

    expected_ids = {
        f"OBJ{i:03d}"
        for i in range(
            1,
            EXPECTED_CLASS_COUNT + 1,
        )
    }

    for split in pools:

        missing = (
            expected_ids
            - set(
                pools[
                    split
                ].keys()
            )
        )

        if missing:
            raise ValueError(
                f"{split}: missing source "
                f"pools for classes "
                f"{sorted(missing)}"
            )

        # Sort then shuffle deterministically.
        for object_id in (
            pools[
                split
            ]
        ):

            class_rows = sorted(
                pools[
                    split
                ][
                    object_id
                ],
                key=lambda row:
                    row[
                        "source_path"
                    ],
            )

            rng.shuffle(
                class_rows
            )

            pools[
                split
            ][
                object_id
            ] = class_rows

    return pools


# ============================================================
# Balanced class selection
# ============================================================

def select_balanced_classes(
    split,
    object_count,
    pools,
    source_pointer,
    class_usage,
    rng,
):
    """
    Select distinct classes while keeping
    overall class usage balanced.
    """

    candidates = []

    for object_id in sorted(
        pools[
            split
        ].keys()
    ):

        available_count = len(
            pools[
                split
            ][
                object_id
            ]
        )

        used_count = (
            source_pointer[
                split
            ][
                object_id
            ]
        )

        remaining = (
            available_count
            - used_count
        )

        if remaining > 0:
            candidates.append(
                object_id
            )

    if len(
        candidates
    ) < object_count:

        raise RuntimeError(
            f"Not enough classes with "
            f"unused source images "
            f"in {split}. "
            f"Need {object_count}, "
            f"found {len(candidates)}."
        )

    scored = [
        (
            class_usage[
                split
            ][
                object_id
            ],

            rng.random(),

            object_id,
        )
        for object_id
        in candidates
    ]

    scored.sort()

    selected = [
        item[2]
        for item in scored[
            :object_count
        ]
    ]

    if len(
        set(
            selected
        )
    ) != object_count:

        raise RuntimeError(
            "Duplicate class selected "
            "within one composite."
        )

    return selected


def take_source_row(
    split,
    object_id,
    pools,
    source_pointer,
):
    """Take next unused source image."""

    pointer = (
        source_pointer[
            split
        ][
            object_id
        ]
    )

    class_pool = (
        pools[
            split
        ][
            object_id
        ]
    )

    if pointer >= len(
        class_pool
    ):
        raise RuntimeError(
            f"No unused source image "
            f"remains for "
            f"{object_id} in {split}."
        )

    row = class_pool[
        pointer
    ]

    source_pointer[
        split
    ][
        object_id
    ] += 1

    return row


# ============================================================
# Generate one composite
# ============================================================

def generate_one_composite(
    split,
    composite_number,
    member,
    selected_rows,
    rng,
):

    object_count = len(
        selected_rows
    )

    if not (
        MIN_OBJECTS_PER_COMPOSITE
        <= object_count
        <= MAX_OBJECTS_PER_COMPOSITE
    ):
        raise ValueError(
            f"Composite must contain "
            f"{MIN_OBJECTS_PER_COMPOSITE}-"
            f"{MAX_OBJECTS_PER_COMPOSITE} "
            f"objects."
        )

    (
        grid_rows,
        grid_cols,
        canvas_width,
        canvas_height,
    ) = get_layout(
        object_count
    )

    selected_cells = (
        choose_grid_cells(
            object_count=
                object_count,

            rows=
                grid_rows,

            cols=
                grid_cols,

            rng=
                rng,
        )
    )

    image_name = (
        f"{split}_"
        f"{composite_number:06d}.jpg"
    )

    label_name = (
        f"{split}_"
        f"{composite_number:06d}.txt"
    )

    image_path = (
        IMAGE_DIR
        / split
        / image_name
    )

    label_path = (
        LABEL_DIR
        / split
        / label_name
    )

    composite = Image.new(
        "RGB",
        (
            canvas_width,
            canvas_height,
        ),
        color=(
            255,
            255,
            255,
        ),
    )

    annotation_lines = []
    manifest_rows = []
    pixel_boxes = []

    for (
        position,
        (
            row_data,
            cell,
        ),
    ) in enumerate(
        zip(
            selected_rows,
            selected_cells,
        ),
        start=1,
    ):

        source_path = (
            PROJECT_ROOT
            / row_data[
                "source_path"
            ]
        )

        if not (
            source_path.exists()
        ):
            raise FileNotFoundError(
                f"Missing source image: "
                f"{source_path}"
            )

        if (
            row_data[
                "source_path"
            ]
            in EXCLUDED_SOURCE_IMAGES
        ):
            raise ValueError(
                "Excluded source image "
                "reached generation: "
                f"{row_data['source_path']}"
            )

        (
            grid_row,
            grid_col,
        ) = cell

        (
            x_min,
            y_min,
            x_max,
            y_max,
        ) = cell_to_pixel_box(
            grid_row,
            grid_col,
        )

        new_box = (
            x_min,
            y_min,
            x_max,
            y_max,
        )

        for existing_box in (
            pixel_boxes
        ):

            if boxes_overlap(
                new_box,
                existing_box,
            ):
                raise RuntimeError(
                    "Overlapping pasted "
                    "regions detected."
                )

        pixel_boxes.append(
            new_box
        )

        tile = (
            prepare_source_image(
                source_path
            )
        )

        composite.paste(
            tile,
            (
                x_min,
                y_min,
            ),
        )

        (
            x_center,
            y_center,
            width_norm,
            height_norm,
        ) = pixel_box_to_yolo(
            x_min=
                x_min,

            y_min=
                y_min,

            x_max=
                x_max,

            y_max=
                y_max,

            canvas_width=
                canvas_width,

            canvas_height=
                canvas_height,
        )

        validate_yolo_box(
            x_center,
            y_center,
            width_norm,
            height_norm,
        )

        class_index = int(
            row_data[
                "class_index"
            ]
        )

        if not (
            0
            <= class_index
            < EXPECTED_CLASS_COUNT
        ):
            raise ValueError(
                f"Invalid class index "
                f"{class_index}."
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
                    image_path
                    .relative_to(
                        PROJECT_ROOT
                    )
                    .as_posix(),

                "label_file":
                    label_path
                    .relative_to(
                        PROJECT_ROOT
                    )
                    .as_posix(),

                "split":
                    split,

                "generator_member":
                    member,

                "object_count":
                    object_count,

                "object_position":
                    position,

                "source_path":
                    row_data[
                        "source_path"
                    ],

                "source_md5":
                    row_data[
                        "md5"
                    ],

                "object_id":
                    row_data[
                        "object_id"
                    ],

                "object_name":
                    row_data[
                        "object_name"
                    ],

                "class_index":
                    class_index,

                "grid_rows":
                    grid_rows,

                "grid_cols":
                    grid_cols,

                "grid_row":
                    grid_row,

                "grid_col":
                    grid_col,

                "canvas_width":
                    canvas_width,

                "canvas_height":
                    canvas_height,

                "tile_size":
                    TILE_SIZE,

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
        quality=
            JPEG_QUALITY,
    )

    label_path.write_text(
        "\n".join(
            annotation_lines
        )
        + "\n",
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
    # Explicit exclusion
    # --------------------------------------------------------

    leaked_exclusions = {
        row["source_path"]
        for row in manifest_rows
        if row["source_path"]
        in EXCLUDED_SOURCE_IMAGES
    }

    if leaked_exclusions:
        raise ValueError(
            "Excluded source image leaked "
            "into generated dataset."
        )

    # --------------------------------------------------------
    # No source reuse
    # --------------------------------------------------------

    source_paths = [
        row[
            "source_path"
        ]
        for row
        in manifest_rows
    ]

    if len(
        source_paths
    ) != len(
        set(
            source_paths
        )
    ):
        raise ValueError(
            "A source image was reused."
        )

    # --------------------------------------------------------
    # No exact duplicate reuse
    # --------------------------------------------------------

    hashes = [
        row[
            "source_md5"
        ]
        for row
        in manifest_rows
    ]

    if len(
        hashes
    ) != len(
        set(
            hashes
        )
    ):
        raise ValueError(
            "An exact duplicate source "
            "image was reused."
        )

    # --------------------------------------------------------
    # Composite checks
    # --------------------------------------------------------

    rows_by_composite = (
        defaultdict(list)
    )

    for row in manifest_rows:

        rows_by_composite[
            row[
                "composite_image"
            ]
        ].append(
            row
        )

    for (
        image,
        rows,
    ) in (
        rows_by_composite.items()
    ):

        annotation_count = len(
            rows
        )

        if not (
            MIN_OBJECTS_PER_COMPOSITE
            <= annotation_count
            <= MAX_OBJECTS_PER_COMPOSITE
        ):
            raise ValueError(
                f"{image} has "
                f"{annotation_count} "
                f"annotations."
            )

        metadata_counts = {
            int(
                row[
                    "object_count"
                ]
            )
            for row in rows
        }

        if metadata_counts != {
            annotation_count
        }:
            raise ValueError(
                f"{image} has inconsistent "
                f"object_count metadata."
            )

        class_ids = [
            row[
                "object_id"
            ]
            for row in rows
        ]

        if len(
            class_ids
        ) != len(
            set(
                class_ids
            )
        ):
            raise ValueError(
                f"{image} contains "
                f"duplicate classes."
            )

        boxes = [
            (
                int(
                    row[
                        "x_min"
                    ]
                ),
                int(
                    row[
                        "y_min"
                    ]
                ),
                int(
                    row[
                        "x_max"
                    ]
                ),
                int(
                    row[
                        "y_max"
                    ]
                ),
            )
            for row in rows
        ]

        for (
            index,
            box_a,
        ) in enumerate(
            boxes
        ):

            for box_b in (
                boxes[
                    index + 1:
                ]
            ):

                if boxes_overlap(
                    box_a,
                    box_b,
                ):
                    raise ValueError(
                        f"{image} contains "
                        f"overlapping regions."
                    )

        (
            expected_rows,
            expected_cols,
            expected_width,
            expected_height,
        ) = get_layout(
            annotation_count
        )

        for row in rows:

            if (
                int(
                    row[
                        "grid_rows"
                    ]
                )
                != expected_rows
            ):
                raise ValueError(
                    f"{image}: incorrect "
                    f"grid row metadata."
                )

            if (
                int(
                    row[
                        "grid_cols"
                    ]
                )
                != expected_cols
            ):
                raise ValueError(
                    f"{image}: incorrect "
                    f"grid column metadata."
                )

            if (
                int(
                    row[
                        "canvas_width"
                    ]
                )
                != expected_width
            ):
                raise ValueError(
                    f"{image}: incorrect "
                    f"canvas width."
                )

            if (
                int(
                    row[
                        "canvas_height"
                    ]
                )
                != expected_height
            ):
                raise ValueError(
                    f"{image}: incorrect "
                    f"canvas height."
                )

    # --------------------------------------------------------
    # Cross-split leakage checks
    # --------------------------------------------------------

    source_to_splits = (
        defaultdict(set)
    )

    hash_to_splits = (
        defaultdict(set)
    )

    for row in manifest_rows:

        source_to_splits[
            row[
                "source_path"
            ]
        ].add(
            row[
                "split"
            ]
        )

        hash_to_splits[
            row[
                "source_md5"
            ]
        ].add(
            row[
                "split"
            ]
        )

    source_violations = {
        source: splits
        for (
            source,
            splits,
        ) in (
            source_to_splits.items()
        )
        if len(
            splits
        ) > 1
    }

    hash_violations = {
        digest: splits
        for (
            digest,
            splits,
        ) in (
            hash_to_splits.items()
        )
        if len(
            splits
        ) > 1
    }

    if source_violations:
        raise ValueError(
            "Source-image leakage "
            "detected across splits."
        )

    if hash_violations:
        raise ValueError(
            "Duplicate-image leakage "
            "detected across splits."
        )


# ============================================================
# Save metadata
# ============================================================

def save_final_manifest(
    manifest_rows,
):

    fieldnames = [
        "composite_image",
        "label_file",
        "split",
        "generator_member",
        "object_count",
        "object_position",
        "source_path",
        "source_md5",
        "object_id",
        "object_name",
        "class_index",
        "grid_rows",
        "grid_cols",
        "grid_row",
        "grid_col",
        "canvas_width",
        "canvas_height",
        "tile_size",
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

        writer = (
            csv.DictWriter(
                file,
                fieldnames=
                    fieldnames,
            )
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
            row[
                "split"
            ],
            row[
                "object_id"
            ],
            row[
                "object_name"
            ],
        )
        for row
        in manifest_rows
    )

    fieldnames = [
        "split",
        "object_id",
        "object_name",
        "annotation_count",
    ]

    with CLASS_DISTRIBUTION.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:

        writer = (
            csv.DictWriter(
                file,
                fieldnames=
                    fieldnames,
            )
        )

        writer.writeheader()

        for (
            (
                split,
                object_id,
                object_name,
            ),
            count,
        ) in sorted(
            counts.items()
        ):

            writer.writerow(
                {
                    "split":
                        split,

                    "object_id":
                        object_id,

                    "object_name":
                        object_name,

                    "annotation_count":
                        count,
                }
            )


# ============================================================
# Main
# ============================================================

def main():

    print(
        "\nFINAL MILESTONE 2 "
        "DATASET GENERATION"
    )

    print(
        "=" * 65
    )

    rng = random.Random(
        RANDOM_SEED
    )

    ensure_output_directories()

    clean_previous_final_outputs()

    pools = load_source_pools(
        rng
    )

    source_pointer = {
        split:
            defaultdict(int)
        for split in (
            "train",
            "val",
            "test",
        )
    }

    class_usage = {
        split:
            Counter()
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
    # Generate composites
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

                object_count = (
                    rng.randint(
                        MIN_OBJECTS_PER_COMPOSITE,
                        MAX_OBJECTS_PER_COMPOSITE,
                    )
                )

                selected_classes = (
                    select_balanced_classes(
                        split=
                            split,

                        object_count=
                            object_count,

                        pools=
                            pools,

                        source_pointer=
                            source_pointer,

                        class_usage=
                            class_usage,

                        rng=
                            rng,
                    )
                )

                selected_rows = []

                for object_id in (
                    selected_classes
                ):

                    row = (
                        take_source_row(
                            split=
                                split,

                            object_id=
                                object_id,

                            pools=
                                pools,

                            source_pointer=
                                source_pointer,
                        )
                    )

                    selected_rows.append(
                        row
                    )

                    class_usage[
                        split
                    ][
                        object_id
                    ] += 1

                created_rows = (
                    generate_one_composite(
                        split=
                            split,

                        composite_number=
                            split_composite_number[
                                split
                            ],

                        member=
                            member,

                        selected_rows=
                            selected_rows,

                        rng=
                            rng,
                    )
                )

                manifest_rows.extend(
                    created_rows
                )

    # --------------------------------------------------------
    # Validate and save
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
    # Summary
    # --------------------------------------------------------

    member_composites = (
        defaultdict(set)
    )

    split_composites = (
        defaultdict(set)
    )

    rows_by_composite = (
        defaultdict(list)
    )

    for row in manifest_rows:

        image = row[
            "composite_image"
        ]

        member_composites[
            row[
                "generator_member"
            ]
        ].add(
            image
        )

        split_composites[
            row[
                "split"
            ]
        ].add(
            image
        )

        rows_by_composite[
            image
        ].append(
            row
        )

    object_count_distribution = (
        Counter(
            len(rows)
            for rows in (
                rows_by_composite.values()
            )
        )
    )

    layout_distribution = (
        Counter()
    )

    for rows in (
        rows_by_composite.values()
    ):

        first = rows[0]

        layout_distribution[
            (
                int(
                    first[
                        "object_count"
                    ]
                ),

                f"{first['grid_rows']}"
                f"x"
                f"{first['grid_cols']}",

                f"{first['canvas_width']}"
                f"x"
                f"{first['canvas_height']}",
            )
        ] += 1

    total_composites = len(
        rows_by_composite
    )

    total_annotations = len(
        manifest_rows
    )

    print(
        "\nDataset summary:"
    )

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

    print(
        "\nObjects per composite:"
    )

    for object_count in range(
        MIN_OBJECTS_PER_COMPOSITE,
        MAX_OBJECTS_PER_COMPOSITE + 1,
    ):

        print(
            f"  {object_count} objects: "
            f"{object_count_distribution[object_count]}"
        )

    print(
        "\nCompact layouts:"
    )

    for (
        (
            object_count,
            grid_shape,
            canvas_shape,
        ),
        count,
    ) in sorted(
        layout_distribution.items()
    ):

        print(
            f"  {object_count} objects -> "
            f"{grid_shape} grid / "
            f"{canvas_shape} canvas: "
            f"{count}"
        )

    print(
        "\nPer-member allocation:"
    )

    for member in GROUP_MEMBERS:

        count = len(
            member_composites[
                member
            ]
        )

        print(
            f"  {member}: "
            f"{count}"
        )

        if count != 100:

            raise ValueError(
                f"{member} should have "
                f"100 allocated composites, "
                f"found {count}."
            )

    print(
        "\nClass-balance check:"
    )

    for split in (
        "train",
        "val",
        "test",
    ):

        counts = [
            class_usage[
                split
            ].get(
                f"OBJ{i:03d}",
                0,
            )
            for i in range(
                1,
                EXPECTED_CLASS_COUNT + 1,
            )
        ]

        print(
            f"  {split}: "
            f"min={min(counts)}, "
            f"max={max(counts)}"
        )

    print(
        "\nIntegrity checks:"
    )

    print(
        "  Explicit exclusion absent: PASS"
    )

    print(
        "  No source reuse: PASS"
    )

    print(
        "  No MD5 reuse: PASS"
    )

    print(
        "  No cross-split source leakage: PASS"
    )

    print(
        "  No cross-split MD5 leakage: PASS"
    )

    print(
        "  3-5 objects per composite: PASS"
    )

    print(
        "  Unique classes per composite: PASS"
    )

    print(
        "  Non-overlapping placements: PASS"
    )

    print(
        "  Every pasted region labeled: PASS"
    )

    print(
        "  Compact layout metadata: PASS"
    )

    print(
        "\nGeometry:"
    )

    print(
        f"  Source tile size: "
        f"{TILE_SIZE} x {TILE_SIZE}"
    )

    print(
        "  3 objects: 2x2 grid -> 448 x 448"
    )

    print(
        "  4 objects: 2x2 grid -> 448 x 448"
    )

    print(
        "  5 objects: 2x3 grid -> 672 x 448"
    )

    print(
        "\nReproducibility:"
    )

    print(
        f"  Random seed: "
        f"{RANDOM_SEED}"
    )

    print(
        "\nMetadata files:"
    )

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