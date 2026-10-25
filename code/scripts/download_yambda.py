"""Download pinned Yambda files and retain complete histories of sampled users."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import requests

from rec.data import sample_users
from rec.io import file_hash, write_json

REVISION = "dd6f3a19eef5866e346c3270e098baa641a44948"
FILES = {
    "listens.parquet": (
        "flat/50m/listens.parquet",
        369110332,
        "eed9cbd094af1e189507d2f8132a0dc9653b90e65480125c7cdccd601e0592d1",
    ),
    "artist_item_mapping.parquet": (
        "artist_item_mapping.parquet",
        38274621,
        "cf1110aa2413ed70008ec9232d395660b2181ffda731002fb74da54e87b6f25d",
    ),
}


def download_and_sample(root):
    raw = Path("data/yambda")
    raw.mkdir(parents=True, exist_ok=True)
    manifest = {"revision": REVISION, "files": {}}
    for name, (remote, size, digest) in FILES.items():
        target = raw / name
        url = f"https://huggingface.co/datasets/yandex/yambda/resolve/{REVISION}/{remote}?download=true"
        if not target.exists():
            temporary = target.with_suffix(".part")
            with requests.get(url, stream=True, timeout=(30, 120)) as response:
                response.raise_for_status()
                with temporary.open("wb") as stream:
                    for block in response.iter_content(1024 * 1024):
                        stream.write(block)
            if temporary.stat().st_size != size or file_hash(temporary) != digest:
                raise ValueError(f"Source checksum mismatch: {name}")
            temporary.replace(target)
        if target.stat().st_size != size or file_hash(target) != digest:
            raise ValueError(f"Cached source checksum mismatch: {target}")
        manifest["files"][name] = {"path": remote, "url": url, "bytes": size, "sha256": digest}
    write_json(raw / "source_manifest.json", manifest)
    if (root / "sample.json").exists():
        if not (root / "sampled_events.parquet").exists():
            raise FileNotFoundError("Existing sample manifest has no retained event table")
        print("Existing sample preserved:", root)
        return
    population, minimum, maximum = [], [], []
    source = pq.ParquetFile(raw / "listens.parquet")
    for group in range(source.num_row_groups):
        frame = source.read_row_group(group, columns=["uid", "timestamp"])
        population.extend(frame["uid"].unique().to_pylist())
        values = frame["timestamp"].to_numpy()
        minimum.append(int(values.min()))
        maximum.append(int(values.max()))
    population = np.unique(population)
    chosen = sample_users(population, 1000, 2028)
    events = pd.read_parquet(raw / "listens.parquet", filters=[("uid", "in", chosen.tolist())])
    root.mkdir(parents=True, exist_ok=True)
    events.to_parquet(root / "sampled_events.parquet", index=False)
    write_json(
        root / "sample.json",
        {
            "seed": 2028,
            "sampled_user_ids": chosen.tolist(),
            "n_users": len(chosen),
            "source_users": len(population),
            "rows": len(events),
            "source_min_timestamp": min(minimum),
            "source_max_timestamp": max(maximum),
        },
    )
    print(f"Preserved {len(events):,} rows for {len(chosen)} sampled users in {root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("runs/v3"))
    download_and_sample(parser.parse_args().root)
