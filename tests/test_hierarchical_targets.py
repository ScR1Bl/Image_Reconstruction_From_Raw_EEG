from __future__ import annotations

import csv
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from eeg_mae.data.build_hierarchical_visual_targets import (
    collect_image_records,
    records_fingerprint,
)
from eeg_mae.data.hierarchical_targets import HierarchicalTargetBank


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_records_are_unique_and_ordered_by_image_index(tmp_path: Path) -> None:
    index = tmp_path / "index.csv"
    with index.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("image_file", "image_index", "split"))
        writer.writeheader()
        writer.writerow({"image_file": "b.jpg", "image_index": 1, "split": "val"})
        writer.writerow({"image_file": "a.jpg", "image_index": 0, "split": "train"})
        writer.writerow({"image_file": "a.jpg", "image_index": 0, "split": "train"})
    images = tmp_path / "images.zip"
    with zipfile.ZipFile(images, "w") as archive:
        archive.writestr("root/concept/b.jpg", b"b")
        archive.writestr("root/concept/a.jpg", b"a")

    records = collect_image_records(index, images)

    assert [record.filename for record in records] == ["a.jpg", "b.jpg"]
    assert [record.position for record in records] == [0, 1]
    assert records_fingerprint(records) == records_fingerprint(records)


def test_records_reject_inconsistent_split(tmp_path: Path) -> None:
    index = tmp_path / "index.csv"
    index.write_text(
        "image_file,image_index,split\na.jpg,0,train\na.jpg,0,test\n", encoding="utf-8"
    )
    images = tmp_path / "images.zip"
    with zipfile.ZipFile(images, "w") as archive:
        archive.writestr("a.jpg", b"a")

    with pytest.raises(RuntimeError, match="inconsistent index"):
        collect_image_records(index, images)


def test_bank_exposes_only_requested_split(tmp_path: Path) -> None:
    index = tmp_path / "index.csv"
    rows = [
        {
            "position": 0,
            "image_index": 10,
            "filename": "train.jpg",
            "split": "train",
            "zip_member": "train.jpg",
            "crc32": 1,
            "file_size": 1,
        },
        {
            "position": 1,
            "image_index": 11,
            "filename": "test.jpg",
            "split": "test",
            "zip_member": "test.jpg",
            "crc32": 2,
            "file_size": 1,
        },
    ]
    with index.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    array = np.arange(8, dtype=np.float16).reshape(2, 4)
    array_path = tmp_path / "synclr.float16.npy"
    np.save(array_path, array, allow_pickle=False)
    manifest = {
        "schema_version": 1,
        "index": index.name,
        "index_sha256": _sha256(index),
        "split_counts": {"train": 1, "val": 0, "test": 1},
        "teachers": {
            "synclr": {
                "array": array_path.name,
                "array_sha256": _sha256(array_path),
                "shape": [2, 4],
            }
        },
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    bank = HierarchicalTargetBank(tmp_path, "train", teachers=("synclr",))

    assert len(bank) == 1
    assert bank[0]["filename"] == "train.jpg"
    assert bank[0]["split"] == "train"
    assert bank[0]["targets"]["synclr"].shape == (4,)
