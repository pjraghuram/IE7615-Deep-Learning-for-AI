from pathlib import Path
import csv
import hashlib
import re
from collections import Counter, defaultdict


# --------------------------------------------------
# Configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

OBJECT_NAMES_CSV = PROJECT_ROOT / "object_names.csv"

OUTPUT_DIR = PROJECT_ROOT / "milestone2" / "metadata"
OUTPUT_MANIFEST = OUTPUT_DIR / "source_manifest.csv"

EXPECTED_CLASS_COUNT = 73

VALID_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
}

# TA-provided exclusion:
# OBJ054_017.jpg is not a valid photo of the assigned object.
EXCLUDED_SOURCE_IMAGES = {
    "images_OBJ054/OBJ054_017.jpg",
}

FOLDER_PATTERN = re.compile(
    r"^images_(OBJ\d{3})$"
)

FILENAME_ID_PATTERN = re.compile(
    r"^OBJ(\d{1,3})",
    re.IGNORECASE,
)


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def md5_hash(
    file_path: Path,
    chunk_size: int = 8192,
) -> str:
    """
    Return the MD5 hash of a file.

    MD5 is used only for exact-duplicate detection.
    """
    md5 = hashlib.md5()

    with file_path.open("rb") as file:
        while chunk := file.read(chunk_size):
            md5.update(chunk)

    return md5.hexdigest()


def load_object_names() -> dict[str, str]:
    """
    Load and validate the OBJECT_ID -> OBJECT_NAME mapping
    from object_names.csv.
    """

    if not OBJECT_NAMES_CSV.exists():
        raise FileNotFoundError(
            f"Missing required file: {OBJECT_NAMES_CSV}"
        )

    mapping: dict[str, str] = {}

    with OBJECT_NAMES_CSV.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:

        reader = csv.DictReader(file)

        required_columns = {
            "OBJECT_ID",
            "OBJECT_NAME",
        }

        if not reader.fieldnames:
            raise ValueError(
                "object_names.csv has no header."
            )

        if not required_columns.issubset(
            reader.fieldnames
        ):
            raise ValueError(
                "object_names.csv must contain "
                "OBJECT_ID and OBJECT_NAME columns."
            )

        for row in reader:
            object_id = row["OBJECT_ID"].strip()
            object_name = row["OBJECT_NAME"].strip()

            if object_id in mapping:
                raise ValueError(
                    f"Duplicate OBJECT_ID in CSV: {object_id}"
                )

            mapping[object_id] = object_name

    return mapping


def normalize_filename_object_id(
    filename: str,
) -> str | None:
    """
    Extract an OBJ identifier from a filename
    and normalize it to OBJ### format.

    Examples:
        OBJ007_001.jpg -> OBJ007
        OBJ07_033.jpg  -> OBJ007
        OBJ060_1.JPG   -> OBJ060
    """

    match = FILENAME_ID_PATTERN.match(
        filename
    )

    if not match:
        return None

    number = int(match.group(1))

    return f"OBJ{number:03d}"


def is_background_image(
    file_path: Path,
) -> bool:
    """
    Detect background-only files while supporting
    known naming variations.

    Examples:
        OBJ001_bg_001.jpg
        OBJ013_BG001.jpg
        OBJ072_bg_01.jpg
    """

    return "_bg" in file_path.stem.lower()


# --------------------------------------------------
# Main audit
# --------------------------------------------------

def main() -> None:

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    object_names = load_object_names()

    expected_ids = [
        f"OBJ{i:03d}"
        for i in range(
            1,
            EXPECTED_CLASS_COUNT + 1,
        )
    ]

    # --------------------------------------------------
    # Discover class folders
    # --------------------------------------------------

    class_dirs = sorted(
        path
        for path in PROJECT_ROOT.iterdir()
        if path.is_dir()
        and FOLDER_PATTERN.fullmatch(
            path.name
        )
    )

    if len(class_dirs) != EXPECTED_CLASS_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_CLASS_COUNT} class folders, "
            f"but found {len(class_dirs)}."
        )

    # --------------------------------------------------
    # Audit containers
    # --------------------------------------------------

    records = []

    extension_counts = Counter()
    class_counts = Counter()
    object_counts = Counter()
    background_counts = Counter()

    hashes = defaultdict(list)

    filename_warnings = []
    unrecognized_filenames = []

    excluded_files = []

    found_ids = []

    # --------------------------------------------------
    # Scan source dataset
    # --------------------------------------------------

    for class_dir in class_dirs:

        folder_match = FOLDER_PATTERN.fullmatch(
            class_dir.name
        )

        if folder_match is None:
            raise ValueError(
                f"Invalid class folder name: "
                f"{class_dir.name}"
            )

        folder_object_id = (
            folder_match.group(1)
        )

        found_ids.append(
            folder_object_id
        )

        if folder_object_id not in object_names:
            raise ValueError(
                f"{folder_object_id} exists as a folder "
                f"but is missing from object_names.csv."
            )

        for file_path in sorted(
            class_dir.iterdir()
        ):

            if not file_path.is_file():
                continue

            extension = (
                file_path.suffix.lower()
            )

            if extension not in VALID_EXTENSIONS:
                continue

            relative_path = (
                file_path
                .relative_to(PROJECT_ROOT)
                .as_posix()
            )

            # --------------------------------------------------
            # Explicit TA-required exclusions
            # --------------------------------------------------

            if (
                relative_path
                in EXCLUDED_SOURCE_IMAGES
            ):
                excluded_files.append(
                    relative_path
                )
                continue

            # --------------------------------------------------
            # Folder is authoritative class source
            # --------------------------------------------------

            object_id = folder_object_id

            filename_object_id = (
                normalize_filename_object_id(
                    file_path.name
                )
            )

            if filename_object_id is None:

                unrecognized_filenames.append(
                    relative_path
                )

            elif (
                filename_object_id
                != folder_object_id
            ):

                filename_warnings.append(
                    {
                        "path": relative_path,
                        "folder_id":
                            folder_object_id,
                        "filename_id":
                            filename_object_id,
                    }
                )

            # --------------------------------------------------
            # Determine image type
            # --------------------------------------------------

            image_type = (
                "background"
                if is_background_image(
                    file_path
                )
                else "object"
            )

            # --------------------------------------------------
            # Exact duplicate hash
            # --------------------------------------------------

            digest = md5_hash(
                file_path
            )

            # --------------------------------------------------
            # Manifest record
            # --------------------------------------------------

            records.append(
                {
                    "source_path":
                        relative_path,
                    "object_id":
                        object_id,
                    "object_name":
                        object_names[
                            object_id
                        ],
                    "image_type":
                        image_type,
                    "extension":
                        extension,
                    "md5":
                        digest,
                }
            )

            # --------------------------------------------------
            # Counters
            # --------------------------------------------------

            extension_counts[
                extension
            ] += 1

            class_counts[
                object_id
            ] += 1

            if image_type == "background":
                background_counts[
                    object_id
                ] += 1
            else:
                object_counts[
                    object_id
                ] += 1

            hashes[
                digest
            ].append(
                relative_path
            )

    # --------------------------------------------------
    # Validate class IDs
    # --------------------------------------------------

    if found_ids != expected_ids:

        missing = sorted(
            set(expected_ids)
            - set(found_ids)
        )

        unexpected = sorted(
            set(found_ids)
            - set(expected_ids)
        )

        raise ValueError(
            "Class folder sequence mismatch.\n"
            f"Missing: {missing}\n"
            f"Unexpected: {unexpected}"
        )

    csv_ids = sorted(
        object_names.keys()
    )

    if csv_ids != expected_ids:

        missing = sorted(
            set(expected_ids)
            - set(csv_ids)
        )

        unexpected = sorted(
            set(csv_ids)
            - set(expected_ids)
        )

        raise ValueError(
            "object_names.csv ID mismatch.\n"
            f"Missing: {missing}\n"
            f"Unexpected: {unexpected}"
        )

    # --------------------------------------------------
    # Validate explicit exclusions
    # --------------------------------------------------

    actual_exclusions = set(
        excluded_files
    )

    missing_exclusions = (
        EXCLUDED_SOURCE_IMAGES
        - actual_exclusions
    )

    if missing_exclusions:
        raise ValueError(
            "Expected excluded source image(s) "
            "were not found:\n"
            + "\n".join(
                sorted(
                    missing_exclusions
                )
            )
        )

    manifest_paths = {
        record["source_path"]
        for record in records
    }

    leaked_exclusions = (
        manifest_paths
        & EXCLUDED_SOURCE_IMAGES
    )

    if leaked_exclusions:
        raise ValueError(
            "Excluded source image(s) leaked "
            "into the manifest:\n"
            + "\n".join(
                sorted(
                    leaked_exclusions
                )
            )
        )

    # --------------------------------------------------
    # Duplicate analysis
    # --------------------------------------------------

    duplicate_groups = {
        digest: paths
        for digest, paths
        in hashes.items()
        if len(paths) > 1
    }

    duplicate_extra_files = sum(
        len(paths) - 1
        for paths
        in duplicate_groups.values()
    )

    # --------------------------------------------------
    # Save manifest
    # --------------------------------------------------

    fieldnames = [
        "source_path",
        "object_id",
        "object_name",
        "image_type",
        "extension",
        "md5",
    ]

    with OUTPUT_MANIFEST.open(
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
            records
        )

    # --------------------------------------------------
    # Report
    # --------------------------------------------------

    total_objects = sum(
        record["image_type"]
        == "object"
        for record in records
    )

    total_backgrounds = sum(
        record["image_type"]
        == "background"
        for record in records
    )

    print(
        "\nSOURCE DATASET AUDIT"
    )

    print(
        "=" * 60
    )

    print(
        f"Class folders:           "
        f"{len(class_dirs)}"
    )

    print(
        f"CSV class mappings:      "
        f"{len(object_names)}"
    )

    print(
        f"Included image files:    "
        f"{len(records)}"
    )

    print(
        f"Explicitly excluded:     "
        f"{len(excluded_files)}"
    )

    print(
        f"Object images:           "
        f"{total_objects}"
    )

    print(
        f"Background-only images:  "
        f"{total_backgrounds}"
    )

    # --------------------------------------------------
    # Exclusions
    # --------------------------------------------------

    print(
        "\nExplicit exclusions:"
    )

    for path in sorted(
        excluded_files
    ):
        print(
            f"  {path}"
        )

    print(
        "  Exclusion validation: PASS"
    )

    # --------------------------------------------------
    # Extensions
    # --------------------------------------------------

    print(
        "\nExtensions:"
    )

    for extension, count in sorted(
        extension_counts.items()
    ):
        print(
            f"  {extension}: {count}"
        )

    # --------------------------------------------------
    # Per-class ranges
    # --------------------------------------------------

    print(
        "\nPer-class total image range:"
    )

    print(
        f"  Minimum: "
        f"{min(class_counts.values())}"
    )

    print(
        f"  Maximum: "
        f"{max(class_counts.values())}"
    )

    print(
        "\nPer-class object-image range:"
    )

    print(
        f"  Minimum: "
        f"{min(object_counts.get(i, 0) for i in expected_ids)}"
    )

    print(
        f"  Maximum: "
        f"{max(object_counts.get(i, 0) for i in expected_ids)}"
    )

    print(
        "\nBackground-image range:"
    )

    print(
        f"  Minimum: "
        f"{min(background_counts.get(i, 0) for i in expected_ids)}"
    )

    print(
        f"  Maximum: "
        f"{max(background_counts.get(i, 0) for i in expected_ids)}"
    )

    # --------------------------------------------------
    # Duplicate summary
    # --------------------------------------------------

    print(
        "\nExact duplicate check:"
    )

    print(
        f"  Duplicate hash groups: "
        f"{len(duplicate_groups)}"
    )

    print(
        f"  Extra duplicate files: "
        f"{duplicate_extra_files}"
    )

    # --------------------------------------------------
    # Filename checks
    # --------------------------------------------------

    print(
        "\nFilename checks:"
    )

    if unrecognized_filenames:

        print(
            f"  Unrecognized filenames: "
            f"{len(unrecognized_filenames)}"
        )

        for path in (
            unrecognized_filenames
        ):
            print(
                f"    {path}"
            )

    else:

        print(
            "  All filenames contain "
            "a recognizable OBJ ID."
        )

    if filename_warnings:

        print(
            f"  Folder/filename ID "
            f"warnings: "
            f"{len(filename_warnings)}"
        )

        for warning in (
            filename_warnings
        ):

            print(
                f"    {warning['path']} "
                f"(folder="
                f"{warning['folder_id']}, "
                f"filename="
                f"{warning['filename_id']})"
            )

    else:

        print(
            "  Folder/filename class "
            "IDs agree."
        )

    # --------------------------------------------------
    # Final status
    # --------------------------------------------------

    print(
        "\nClass ID validation: PASS"
    )

    print(
        "Explicit exclusion validation: PASS"
    )

    print(
        "\nManifest saved to:"
    )

    print(
        f"  {OUTPUT_MANIFEST.relative_to(PROJECT_ROOT)}"
    )

    print(
        "\nAudit complete."
    )


if __name__ == "__main__":
    main()