from pathlib import Path
import csv
import random
from collections import Counter, defaultdict

# --------------------------------------------------
# Configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

METADATA_DIR = PROJECT_ROOT / "milestone2" / "metadata"

SOURCE_MANIFEST = METADATA_DIR / "source_manifest.csv"
SPLIT_MANIFEST = METADATA_DIR / "source_split_manifest.csv"
DUPLICATE_REPORT = METADATA_DIR / "duplicate_report.csv"

EXPECTED_CLASS_COUNT = 73

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 42


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def read_csv(path):
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def class_index(object_id):
    """
    Convert canonical object ID to zero-based YOLO class index.

    OBJ001 -> 0
    OBJ002 -> 1
    ...
    OBJ073 -> 72
    """
    number = int(object_id.replace("OBJ", ""))
    return number - 1


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    if not SOURCE_MANIFEST.exists():
        raise FileNotFoundError(
            "source_manifest.csv does not exist. "
            "Run audit_source_dataset.py first."
        )

    rows = read_csv(SOURCE_MANIFEST)

    # --------------------------------------------------
    # 1. Keep only real object images
    # --------------------------------------------------

    object_rows = [
        row
        for row in rows
        if row["image_type"] == "object"
    ]

    print("\nSOURCE SPLIT CREATION")
    print("=" * 60)

    print(f"All source images:        {len(rows)}")
    print(f"Object-image candidates:  {len(object_rows)}")
    print(
        f"Background images ignored: "
        f"{len(rows) - len(object_rows)}"
    )

    # --------------------------------------------------
    # 2. Group object images by MD5
    # --------------------------------------------------

    by_hash = defaultdict(list)

    for row in object_rows:
        by_hash[row["md5"]].append(row)

    retained_rows = []
    duplicate_rows = []

    cross_class_duplicate_groups = []

    for md5, group in sorted(by_hash.items()):

        # Deterministic canonical choice:
        # lexicographically smallest source path.
        ordered_group = sorted(
            group,
            key=lambda row: row["source_path"]
        )

        canonical = ordered_group[0]
        retained_rows.append(canonical)

        object_ids = sorted(
            {row["object_id"] for row in ordered_group}
        )

        if len(object_ids) > 1:
            cross_class_duplicate_groups.append(
                {
                    "md5": md5,
                    "object_ids": object_ids,
                    "paths": [
                        row["source_path"]
                        for row in ordered_group
                    ],
                }
            )

        for duplicate in ordered_group[1:]:
            duplicate_rows.append(
                {
                    "duplicate_path": duplicate["source_path"],
                    "duplicate_object_id": duplicate["object_id"],
                    "canonical_path": canonical["source_path"],
                    "canonical_object_id": canonical["object_id"],
                    "md5": md5,
                }
            )

    # A byte-identical image assigned to different object IDs
    # indicates a label conflict and should not be silently accepted.
    if cross_class_duplicate_groups:

        print("\nERROR: Exact duplicates found across different classes.")

        for group in cross_class_duplicate_groups:
            print(
                f"\nMD5: {group['md5']}\n"
                f"Classes: {group['object_ids']}"
            )

            for path in group["paths"]:
                print(f"  {path}")

        raise ValueError(
            "Cross-class duplicate images detected. "
            "Resolve before creating splits."
        )

    print(f"Unique object images:     {len(retained_rows)}")
    print(f"Duplicate object files:   {len(duplicate_rows)}")

    # --------------------------------------------------
    # 3. Group unique object images by class
    # --------------------------------------------------

    by_class = defaultdict(list)

    for row in retained_rows:
        by_class[row["object_id"]].append(row)

    expected_ids = [
        f"OBJ{i:03d}"
        for i in range(1, EXPECTED_CLASS_COUNT + 1)
    ]

    found_ids = sorted(by_class.keys())

    if found_ids != expected_ids:
        missing = sorted(
            set(expected_ids) - set(found_ids)
        )

        unexpected = sorted(
            set(found_ids) - set(expected_ids)
        )

        raise ValueError(
            f"Object classes do not match expected IDs.\n"
            f"Missing: {missing}\n"
            f"Unexpected: {unexpected}"
        )

    # --------------------------------------------------
    # 4. Deterministic per-class train/val/test split
    # --------------------------------------------------

    rng = random.Random(RANDOM_SEED)

    split_rows = []

    per_class_summary = []

    for object_id in expected_ids:

        class_rows = sorted(
            by_class[object_id],
            key=lambda row: row["source_path"]
        )

        # Shuffle using the fixed global RNG.
        rng.shuffle(class_rows)

        total = len(class_rows)

        n_train = int(total * TRAIN_RATIO)
        n_val = int(total * VAL_RATIO)
        n_test = total - n_train - n_val

        if min(n_train, n_val, n_test) <= 0:
            raise ValueError(
                f"{object_id} does not contain enough images "
                "for train/val/test splitting."
            )

        assignments = (
            [("train", row) for row in class_rows[:n_train]]
            + [
                ("val", row)
                for row in class_rows[n_train:n_train + n_val]
            ]
            + [
                ("test", row)
                for row in class_rows[n_train + n_val:]
            ]
        )

        for split, row in assignments:

            split_rows.append(
                {
                    "source_path": row["source_path"],
                    "object_id": row["object_id"],
                    "object_name": row["object_name"],
                    "class_index": class_index(row["object_id"]),
                    "md5": row["md5"],
                    "split": split,
                }
            )

        per_class_summary.append(
            (
                object_id,
                total,
                n_train,
                n_val,
                n_test,
            )
        )

    # --------------------------------------------------
    # 5. Integrity checks
    # --------------------------------------------------

    source_paths = [
        row["source_path"]
        for row in split_rows
    ]

    if len(source_paths) != len(set(source_paths)):
        raise ValueError(
            "A source image was assigned more than once."
        )

    hash_to_splits = defaultdict(set)

    for row in split_rows:
        hash_to_splits[row["md5"]].add(row["split"])

    leaking_hashes = {
        md5: splits
        for md5, splits in hash_to_splits.items()
        if len(splits) > 1
    }

    if leaking_hashes:
        raise ValueError(
            "Exact duplicate hashes appear across multiple splits."
        )

    class_split_counts = Counter(
        (row["object_id"], row["split"])
        for row in split_rows
    )

    for object_id in expected_ids:

        for split in ("train", "val", "test"):

            if class_split_counts[(object_id, split)] == 0:
                raise ValueError(
                    f"{object_id} is missing from {split}."
                )

    # --------------------------------------------------
    # 6. Save source split manifest
    # --------------------------------------------------

    split_fieldnames = [
        "source_path",
        "object_id",
        "object_name",
        "class_index",
        "md5",
        "split",
    ]

    with SPLIT_MANIFEST.open(
        "w",
        encoding="utf-8",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=split_fieldnames
        )

        writer.writeheader()
        writer.writerows(
            sorted(
                split_rows,
                key=lambda row: (
                    row["split"],
                    row["object_id"],
                    row["source_path"],
                )
            )
        )

    # --------------------------------------------------
    # 7. Save duplicate report
    # --------------------------------------------------

    duplicate_fieldnames = [
        "duplicate_path",
        "duplicate_object_id",
        "canonical_path",
        "canonical_object_id",
        "md5",
    ]

    with DUPLICATE_REPORT.open(
        "w",
        encoding="utf-8",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=duplicate_fieldnames
        )

        writer.writeheader()
        writer.writerows(duplicate_rows)

    # --------------------------------------------------
    # 8. Summary
    # --------------------------------------------------

    split_counts = Counter(
        row["split"]
        for row in split_rows
    )

    print("\nSplit counts:")
    print(f"  Train: {split_counts['train']}")
    print(f"  Val:   {split_counts['val']}")
    print(f"  Test:  {split_counts['test']}")
    print(f"  Total: {len(split_rows)}")

    class_totals = [
        total
        for _, total, _, _, _ in per_class_summary
    ]

    print("\nUnique object images per class:")
    print(f"  Minimum: {min(class_totals)}")
    print(f"  Maximum: {max(class_totals)}")

    print("\nLeakage checks:")
    print("  Source path overlap: PASS")
    print("  Exact duplicate overlap: PASS")
    print("  All 73 classes in train: PASS")
    print("  All 73 classes in val:   PASS")
    print("  All 73 classes in test:  PASS")

    print("\nConfiguration:")
    print(
        f"  Ratios: "
        f"{TRAIN_RATIO:.2f} / {VAL_RATIO:.2f} / {TEST_RATIO:.2f}"
    )
    print(f"  Random seed: {RANDOM_SEED}")

    print("\nFiles created:")
    print(
        f"  {SPLIT_MANIFEST.relative_to(PROJECT_ROOT)}"
    )
    print(
        f"  {DUPLICATE_REPORT.relative_to(PROJECT_ROOT)}"
    )

    print("\nSource split creation complete.")


if __name__ == "__main__":
    main()