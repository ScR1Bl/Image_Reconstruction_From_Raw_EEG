from __future__ import annotations

import argparse
import csv
import io
import zipfile
from pathlib import Path

import numpy as np


def main() -> None:
    p = argparse.ArgumentParser(
        description="One-time float32 memmap cache for compressed THINGS-EEG2 training EEG"
    )
    p.add_argument(
        "--index", default="data/things_eeg2_osf/preprocessed_train_all_subjects_holdout_index.csv"
    )
    p.add_argument("--output", default="data/derived/eeg_float32_cache")
    p.add_argument("--chunk-images", type=int, default=256)
    args = p.parse_args()
    index_path, output = Path(args.index).resolve(), Path(args.output).resolve()
    root = index_path.parent
    sources = set()
    with index_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            sources.add((row["zip_path"], row["inner_npy_path"]))
    output.mkdir(parents=True, exist_ok=True)
    for number, (zip_name, inner_name) in enumerate(sorted(sources), 1):
        destination = output / f"{Path(zip_name).stem}__{Path(inner_name).stem}.npy"
        if destination.exists():
            existing = np.load(destination, mmap_mode="r")
            if existing.shape == (16540, 4, 17, 100) and existing.dtype == np.float32:
                print(
                    f"cache {number}/{len(sources)} already valid: {destination.name}", flush=True
                )
                continue
            raise RuntimeError(f"invalid existing cache file: {destination}")
        with zipfile.ZipFile(root / zip_name) as archive:
            packed = np.load(io.BytesIO(archive.read(inner_name)), allow_pickle=True).item()
            source = packed["preprocessed_eeg_data"]
        target = np.lib.format.open_memmap(
            destination, mode="w+", dtype=np.float32, shape=source.shape
        )
        for start in range(0, source.shape[0], args.chunk_images):
            target[start : start + args.chunk_images] = source[
                start : start + args.chunk_images
            ].astype(np.float32)
        target.flush()
        del target, source, packed
        verified = np.load(destination, mmap_mode="r")
        if (
            verified.shape != (16540, 4, 17, 100)
            or verified.dtype != np.float32
            or not np.isfinite(verified[::1000]).all()
        ):
            raise RuntimeError(f"cache verification failed: {destination}")
        print(f"cache {number}/{len(sources)} created and verified: {destination.name}", flush=True)


if __name__ == "__main__":
    main()
