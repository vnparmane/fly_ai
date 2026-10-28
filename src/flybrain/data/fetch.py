"""Data fetching module for FlyWire connectome release 783."""

import hashlib
import json
import logging
from pathlib import Path
from urllib.request import urlopen, Request
from tqdm import tqdm

logger = logging.getLogger(__name__)

URLS = {
    "783": {
        "Connectivity_783.parquet": "https://raw.githubusercontent.com/philshiu/Drosophila_brain_model/main/Connectivity_783.parquet",
        "Completeness_783.csv": "https://raw.githubusercontent.com/philshiu/Drosophila_brain_model/main/Completeness_783.csv",
        "Supplemental_file1_neuron_annotations.tsv": "https://raw.githubusercontent.com/flyconnectome/flywire_annotations/main/supplemental_files/Supplemental_file1_neuron_annotations.tsv",
    }
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def _download_file(url: str, dest: Path) -> None:
    logger.info(f"Downloading {url} -> {dest}")
    req = Request(url, headers={"User-Agent": "FlyBrainArcade/0.1.0"})
    with urlopen(req) as resp, open(dest, "wb") as f:
        total = int(resp.headers.get("content-length", 0))
        with tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as pbar:
            while chunk := resp.read(1024 * 1024):
                f.write(chunk)
                pbar.update(len(chunk))


def ensure_flywire(release: str = "783", dest: Path = Path("data/raw")) -> dict[str, Path]:
    """Ensure FlyWire release files are downloaded and verified with MANIFEST.json.

    Parameters
    ----------
    release : str
        Connectome materialization release (default: "783")
    dest : Path
        Target directory for raw data files

    Returns
    -------
    dict[str, Path]
        Map of logical filename to local Path
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    manifest_path = dest / "MANIFEST.json"

    if release not in URLS:
        raise ValueError(f"Unknown release '{release}'. Known releases: {list(URLS.keys())}")

    manifest = {}
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

    file_paths: dict[str, Path] = {}
    manifest_updated = False

    for filename, url in URLS[release].items():
        file_path = dest / filename
        file_paths[filename] = file_path

        # If file doesn't exist, download it
        if not file_path.exists():
            _download_file(url, file_path)
            checksum = _sha256(file_path)
            manifest[filename] = {
                "sha256": checksum,
                "bytes": file_path.stat().st_size,
                "url": url,
            }
            manifest_updated = True
        else:
            # File exists: verify it's recorded in manifest and not corrupted
            if filename in manifest:
                expected_sha = manifest[filename]["sha256"]
                actual_sha = _sha256(file_path)
                if expected_sha != actual_sha:
                    raise IOError(
                        f"Checksum mismatch for {filename}! Expected {expected_sha}, got {actual_sha}. "
                        f"Delete {file_path} and rerun ensure_flywire() to re-download."
                    )
            else:
                # Add existing unrecorded file to manifest
                manifest[filename] = {
                    "sha256": _sha256(file_path),
                    "bytes": file_path.stat().st_size,
                    "url": url,
                }
                manifest_updated = True

    if manifest_updated or not manifest_path.exists():
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    return file_paths


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    paths = ensure_flywire()
    print("Files ensured:")
    for k, v in paths.items():
        print(f"  {k}: {v} ({v.stat().st_size / 1024 / 1024:.1f} MB)")
