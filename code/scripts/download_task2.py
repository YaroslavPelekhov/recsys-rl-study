"""Download only the pinned source needed by Task 2; preserve existing manifests."""

from pathlib import Path

import requests

from download_yambda import FILES, REVISION
from rec.io import file_hash, write_json
from rec.task2_data import read_json


def main():
    raw = Path("data/yambda")
    raw.mkdir(parents=True, exist_ok=True)
    name = "listens.parquet"
    remote, size, digest = FILES[name]
    target = raw / name
    url = f"https://huggingface.co/datasets/yandex/yambda/resolve/{REVISION}/{remote}?download=true"
    if not target.exists():
        temporary = raw / "listens.part"
        with requests.get(url, stream=True, timeout=(30, 120)) as response:
            response.raise_for_status()
            with temporary.open("wb") as stream:
                for block in response.iter_content(1024 * 1024):
                    stream.write(block)
        if temporary.stat().st_size != size or file_hash(temporary) != digest:
            raise ValueError("Downloaded source checksum mismatch")
        temporary.replace(target)
    if target.stat().st_size != size or file_hash(target) != digest:
        raise ValueError("Cached source checksum mismatch")
    path = raw / "source_manifest.json"
    if path.exists():
        old = read_json(path)
        if old["revision"] != REVISION or old["files"][name]["sha256"] != digest:
            raise ValueError("Existing source manifest refers to another revision")
    else:
        write_json(
            path,
            {
                "revision": REVISION,
                "files": {name: {"path": remote, "url": url, "bytes": size, "sha256": digest}},
            },
        )
    print("Pinned Yambda listens verified:", target)


if __name__ == "__main__":
    main()
