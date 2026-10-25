"""A checksum update must not hide a fabricated metric."""

import shutil
import sys
from pathlib import Path

import pandas as pd
import pytest

from rec.io import file_hash, write_json
from rec.task2_data import read_json

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from verify_task2b import verify  # noqa: E402


def test_replay_rejects_modified_ndcg_with_updated_checksum(tmp_path):
    source = Path("reports/task2b")
    if not source.exists():
        pytest.skip("Task2b evidence has not been published")
    destination = tmp_path / "altered"
    shutil.copytree(source / "evidence", destination / "evidence")
    path = destination / "evidence/query_metrics.parquet"
    frame = pd.read_parquet(path)
    frame.loc[0, "ndcg"] += 0.1
    frame.to_parquet(path, index=False)
    checksums = read_json(destination / "evidence/checksums.json")
    checksums["query_metrics.parquet"] = file_hash(path)
    write_json(destination / "evidence/checksums.json", checksums)
    with pytest.raises(AssertionError):
        verify(destination)
