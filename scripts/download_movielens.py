#!/usr/bin/env python3
"""Download and extract MovieLens datasets with checksum validation."""

from __future__ import annotations

import argparse
import hashlib
import logging
import shutil
import sys
from pathlib import Path
from typing import Dict

import requests
from requests.exceptions import HTTPError
from zipfile import ZipFile


LOGGER = logging.getLogger("movielens_download")

DEFAULT_VARIANT = "25m"
DATASET_URLS: Dict[str, str] = {
    "latest": "https://files.grouplens.org/datasets/movielens/ml-latest.zip",
    "latest-small": "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip",
    "25m": "https://files.grouplens.org/datasets/movielens/ml-25m.zip",
    "20m": "https://files.grouplens.org/datasets/movielens/ml-20m.zip",
    "10m": "https://files.grouplens.org/datasets/movielens/ml-10m.zip",
    "1m": "https://files.grouplens.org/datasets/movielens/ml-1m.zip",
    "100k": "https://files.grouplens.org/datasets/movielens/ml-100k.zip",
}

EXPECTED_FILES = {
    "ratings": "ratings.csv",
    "movies": "movies.csv",
    "tags": "tags.csv",
    "links": "links.csv",
}

CHECKSUM_SUFFIX = ".sha256"
CHUNK_SIZE = 1 << 20  # 1 MiB


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s %(message)s")


def compute_checksum(path: Path) -> str:
    sha256 = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(CHUNK_SIZE), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def existing_dataset_is_valid(target_dir: Path, archive_path: Path) -> bool:
    if not archive_path.exists():
        LOGGER.debug("Archive %s does not exist", archive_path)
        return False

    checksum_file = archive_path.with_suffix(archive_path.suffix + CHECKSUM_SUFFIX)
    if not checksum_file.exists():
        LOGGER.debug("Checksum file %s missing", checksum_file)
        return False

    recorded_checksum = checksum_file.read_text().strip()
    current_checksum = compute_checksum(archive_path)
    if recorded_checksum != current_checksum:
        LOGGER.warning(
            "Checksum mismatch for %s (expected %s, got %s)",
            archive_path,
            recorded_checksum,
            current_checksum,
        )
        return False

    missing = [name for name in EXPECTED_FILES.values() if not (target_dir / name).exists()]
    if missing:
        LOGGER.warning("Missing files in %s: %s", target_dir, ", ".join(missing))
        return False

    return True


def download_archive(url: str, destination: Path) -> None:
    LOGGER.info("Downloading %s", url)
    response = requests.get(url, stream=True, timeout=60)
    try:
        response.raise_for_status()
    except HTTPError as exc:  # pragma: no cover - just defensive logging
        raise SystemExit(f"Failed to download {url}: {exc}") from exc

    total = 0
    with destination.open("wb") as fh:
        for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
            if chunk:
                fh.write(chunk)
                total += len(chunk)
    LOGGER.info("Downloaded %.2f MB", total / (1024 * 1024))


def extract_archive(archive_path: Path, extract_to: Path) -> None:
    LOGGER.info("Extracting %s to %s", archive_path, extract_to)
    with ZipFile(archive_path) as zip_file:
        zip_file.extractall(path=extract_to)


def move_dataset_contents(extracted_root: Path, output_dir: Path) -> None:
    LOGGER.debug("Searching for dataset files in %s", extracted_root)
    csv_parent = None
    for child in extracted_root.iterdir():
        if child.is_dir() and all((child / file_name).exists() for file_name in EXPECTED_FILES.values()):
            csv_parent = child
            break
    if csv_parent is None:
        raise FileNotFoundError(
            f"Could not locate expected files ({', '.join(EXPECTED_FILES.values())}) in {extracted_root}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    for file_name in EXPECTED_FILES.values():
        source = csv_parent / file_name
        destination = output_dir / file_name
        LOGGER.debug("Copying %s -> %s", source, destination)
        shutil.copy2(source, destination)


def record_checksum(archive_path: Path) -> None:
    checksum = compute_checksum(archive_path)
    checksum_path = archive_path.with_suffix(archive_path.suffix + CHECKSUM_SUFFIX)
    checksum_path.write_text(checksum)
    LOGGER.debug("Recorded checksum %s in %s", checksum, checksum_path)


def clean_temp_directory(temp_dir: Path) -> None:
    if temp_dir.exists():
        shutil.rmtree(temp_dir)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and extract MovieLens datasets.")
    parser.add_argument(
        "--variant",
        choices=sorted(DATASET_URLS),
        default=DEFAULT_VARIANT,
        help="Which MovieLens variant to download (default: %(default)s)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory where data should be stored (default: %(default)s)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if data already exists",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    configure_logging(args.verbose)

    url = DATASET_URLS[args.variant]
    data_dir: Path = args.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)

    archive_name = Path(url).name
    archive_path = data_dir / archive_name
    dataset_output_dir = data_dir / "movielens"
    variant_dir = dataset_output_dir / args.variant

    if not args.force and existing_dataset_is_valid(variant_dir, archive_path):
        LOGGER.info("Dataset already downloaded and verified at %s", variant_dir)
        return 0

    temp_dir = data_dir / "tmp_download"
    clean_temp_directory(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    download_archive(url, archive_path)
    record_checksum(archive_path)

    extract_archive(archive_path, temp_dir)
    move_dataset_contents(temp_dir, variant_dir)
    clean_temp_directory(temp_dir)

    LOGGER.info("MovieLens %s dataset extracted to %s", args.variant, variant_dir)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
