"""Reuse immutable pilot data/RMs without duplicating large desktop caches."""

from pathlib import Path
import os
import shutil

from rec.io import file_hash, write_json

base, final = Path("runs/task2"), Path("runs/task2_final")
if final.exists():
    raise FileExistsError(final)
final.mkdir()
(base / "code_snapshot").mkdir(exist_ok=True)
for source in [*Path("src/rec").glob("task2*.py"), *Path("scripts").glob("*task2.py")]:
    shutil.copy2(source, base / "code_snapshot" / source.name)
for name in (
    "sampled_events.parquet",
    "sample.json",
    "catalog.npz",
    "train.npz",
    "search_x.npy",
    "search_queries.parquet",
    "search_labels.parquet",
    "data_audit.json",
    "environment.json",
):
    os.link((base / name).resolve(), final / name)
for directory in ("models", "logs"):
    (final / directory).mkdir()
    for source in (base / directory).glob("rm_*"):
        os.link(source.resolve(), final / directory / source.name)
write_json(
    final / "protocol.json",
    {
        "base_document_sha256": file_hash(Path("docs/TASK2_PROTOCOL.md")),
        "amendment_sha256": file_hash(Path("docs/TASK2_AMENDMENT_01.md")),
        "sample_sha256": file_hash(final / "sample.json"),
        "test_opened": False,
        "pilot_completed_rl_trials": 9,
    },
)
print("Reused full histories, train data and four RM checkpoints; test is closed.")
