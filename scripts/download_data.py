#!/usr/bin/env python3
"""Download and checksum the public granting-model dataset from Zenodo."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen


def md5sum(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    with urlopen(url) as response, temporary.open("wb") as output:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            output.write(chunk)
    temporary.replace(destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/model_config.json")
    parser.add_argument("--output-dir", default="data/raw")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))["data"]
    destination = Path(args.output_dir) / config["filename"]
    if not destination.exists() or md5sum(destination) != config["md5"]:
        download(config["url"], destination)
    checksum = md5sum(destination)
    if checksum != config["md5"]:
        raise RuntimeError(f"Checksum mismatch: expected {config['md5']}, received {checksum}")
    print(f"Ready: {destination} ({checksum})")


if __name__ == "__main__":
    main()

