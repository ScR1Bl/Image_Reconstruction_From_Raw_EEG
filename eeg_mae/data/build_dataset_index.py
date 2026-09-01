"""Build the deterministic image-level split used with THINGS-EEG2.

The official training partition contains ten images for each of 1,654 concepts.
We use images 1-8 for training, image 9 for validation, and image 10 for an
untouched test. All four EEG repetitions and all ten participants remain grouped
with their image, preventing repetitions of one stimulus from crossing splits.
"""

from __future__ import annotations

import argparse
import csv
import re
import zipfile
from pathlib import Path, PurePosixPath

FIELDS = (
    "zip_path",
    "inner_npy_path",
    "eeg_path",
    "label",
    "subject_id",
    "split",
    "image_index",
    "repeat_index",
    "image_within_concept",
    "length",
    "concept",
    "image_file",
    "eeg_key",
)
IMAGE_PATTERN = re.compile(r"^(?P<label>\d{5})_(?P<concept>.+)/(?P<file>.+\.jpg)$")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the THINGS-EEG2 image holdout index")
    parser.add_argument("--dataset-root", type=Path, default=Path("data/things_eeg2_osf"))
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def training_images(archive_path: Path) -> list[tuple[int, str, str]]:
    images: list[tuple[int, str, str]] = []
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.namelist():
            path = PurePosixPath(member)
            if len(path.parts) != 3 or path.parts[0] != "training_images":
                continue
            match = IMAGE_PATTERN.match("/".join(path.parts[1:]))
            if match:
                images.append(
                    (int(match.group("label")) - 1, match.group("concept"), match.group("file"))
                )
    images.sort(key=lambda item: (item[0], item[2]))
    if len(images) != 16_540 or {label for label, _, _ in images} != set(range(1_654)):
        raise RuntimeError("expected 16,540 training images from 1,654 concepts")
    return images


def main() -> None:
    args = arguments()
    root = args.dataset_root.resolve()
    output = args.output or root / "preprocessed_train_all_subjects_holdout_index.csv"
    images = training_images(root / "image_set" / "training_images.zip")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for subject in range(10):
            subject_name = f"sub-{subject + 1:02d}"
            for image_index, (label, concept, image_file) in enumerate(images):
                within = image_index % 10
                split = "train" if within < 8 else "val" if within == 8 else "test"
                for repetition in range(4):
                    writer.writerow(
                        {
                            "zip_path": str(Path("preprocessed") / f"{subject_name}.zip"),
                            "inner_npy_path": f"{subject_name}/preprocessed_eeg_training.npy",
                            "eeg_path": "",
                            "label": label,
                            "subject_id": subject,
                            "split": split,
                            "image_index": image_index,
                            "repeat_index": repetition,
                            "image_within_concept": within,
                            "length": 100,
                            "concept": concept,
                            "image_file": image_file,
                            "eeg_key": "preprocessed_eeg_data",
                        }
                    )
    print(f"wrote {output} (661,600 rows)")


if __name__ == "__main__":
    main()
