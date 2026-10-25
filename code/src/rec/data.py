from __future__ import annotations

import json
import logging
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from rec.io import file_hash, write_json

LOG = logging.getLogger(__name__)
KION_REVISION = "f69775be31fa5779907cf0a92ddedb70037fb5ae"
BASE_URL = f"https://raw.githubusercontent.com/irsafilo/KION_DATASET/{KION_REVISION}"
SOURCES = {"data_original.zip": "data_original.zip"}
SOURCE_SHA256 = {
    "data_original.zip": "50fc5b7db78cbda4c3d0ee5bb5bea4c416cff41e8af58aab0b661d0d3e9dfa9b"
}


def download(root: Path) -> None:
    """Download immutable source files; record and verify local checksums on reuse."""
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    previous = json.loads(manifest_path.read_text("utf-8")) if manifest_path.exists() else {}
    if previous and previous["revision"] != KION_REVISION:
        raise ValueError("Raw data directory belongs to a different source revision")
    manifest = {"revision": KION_REVISION, "files": {}}
    for name, remote_path in SOURCES.items():
        target = root / name
        url = f"{BASE_URL}/{remote_path}"
        if not target.exists():
            LOG.info("Downloading %s", url)
            partial = target.with_suffix(target.suffix + ".part")
            with requests.get(url, stream=True, timeout=(20, 180)) as response:
                response.raise_for_status()
                with partial.open("wb") as stream:
                    for chunk in response.iter_content(1024 * 1024):
                        stream.write(chunk)
            partial.replace(target)
        checksum = file_hash(target)
        if checksum != SOURCE_SHA256[name]:
            raise ValueError(f"Source checksum does not match pinned archive: {target}")
        old = previous.get("files", {}).get(name)
        if old and old["sha256"] != checksum:
            raise ValueError(f"Checksum mismatch: {target}")
        manifest["files"][name] = {"url": url, "bytes": target.stat().st_size, "sha256": checksum}
    write_json(manifest_path, manifest)


def sample_users(ids: np.ndarray, count: int, seed: int) -> np.ndarray:
    """Uniform user sample, independent of activity and time; preserve whole histories."""
    unique = np.sort(np.unique(ids))
    if count < 1:
        raise ValueError("n_users must be positive")
    return np.sort(
        np.random.default_rng(seed).choice(unique, min(count, len(unique)), replace=False)
    )


def load_sample(
    raw: Path, n_users: int, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    archive = raw / "data_original.zip"
    with zipfile.ZipFile(archive) as zipped:
        with zipped.open("data_original/interactions.csv") as stream:
            events = pd.read_csv(stream)
        with zipped.open("data_original/items.csv") as stream:
            items = pd.read_csv(stream).drop_duplicates("item_id").set_index("item_id")
        with zipped.open("data_original/users.csv") as stream:
            users = pd.read_csv(stream).drop_duplicates("user_id").set_index("user_id")
    required = {"user_id", "item_id", "last_watch_dt", "total_dur", "watched_pct"}
    if not required.issubset(events.columns):
        raise ValueError(f"Unexpected interaction schema: {events.columns.tolist()}")
    item_fields = {
        "genres",
        "directors",
        "countries",
        "content_type",
        "release_year",
        "age_rating",
        "for_kids",
    }
    if not item_fields.issubset(items.columns) or not {"age", "income", "sex", "kids_flg"}.issubset(
        users.columns
    ):
        raise ValueError("Unexpected metadata schema")
    events = events.rename(columns={"last_watch_dt": "timestamp"})
    events["timestamp"] = pd.to_datetime(events["timestamp"], errors="raise")
    if events[["user_id", "item_id", "timestamp"]].isna().any().any():
        raise ValueError("Null event identifiers or timestamps")
    selected = sample_users(events.user_id.to_numpy(), n_users, seed)
    source_rows = len(events)
    source_end = events.timestamp.max() + pd.Timedelta(days=1)
    events = events[events.user_id.isin(selected)].copy()
    # KION rows contain the last watch date and aggregates for a user-item pair.
    # Do not pretend these are atomic events; do not use duration/percentage as historical features.
    events = (
        events[["user_id", "item_id", "timestamp"]]
        .sort_values(["timestamp", "user_id", "item_id"], kind="stable")
        .reset_index(drop=True)
    )
    duplicates = int(events.duplicated(["user_id", "item_id"]).sum())
    users = users.reindex(selected)
    return (
        events,
        items,
        users,
        {
            "source_rows": source_rows,
            "sample_rows": len(events),
            "sample_users": len(selected),
            "sample_seed": seed,
            "sampled_user_ids": selected.tolist(),
            "min_timestamp": str(events.timestamp.min()),
            "source_end_exclusive": str(source_end),
            "duplicate_user_item_rows": duplicates,
            "sampling": "uniform over distinct source interaction users, retain every source row",
            "semantics": "last-watch records, not an atomic event or impression log",
        },
    )
