"""Verify a standalone submission: file hashes, raw-result replay and paper bindings."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

CODE = Path(__file__).resolve().parents[1]
PACKAGE = CODE.parent


def read(path):
    return json.loads(path.read_text("utf-8"))


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify():
    for line in (PACKAGE / "SHA256SUMS.txt").read_text("utf-8").splitlines():
        expected, name = line.split("  ", 1)
        path = (PACKAGE / name).resolve()
        assert path.is_relative_to(PACKAGE.resolve()), name
        assert digest(path) == expected, f"Changed submission file: {name}"
    origin = read(PACKAGE / "SOURCE_ORIGIN.json")
    for name, entry in origin.items():
        assert digest(PACKAGE / name) == entry["sha256"], (
            f"Changed original source/evidence: {name}"
        )
    info = read(PACKAGE / "SUBMISSION.json")
    quality = read(PACKAGE / "PDF_QUALITY.json")
    assert digest(PACKAGE / "article.pdf") == info["paper_sha256"] == quality["pdf_sha256"]
    assert quality["page_count"] == info["pages"]
    assert all(p["outside_margins"] == 0 for p in quality["pages"])
    build = read(PACKAGE / "PAPER_BUILD.json")
    assert build["pdf_sha256"] == info["paper_sha256"]
    outputs = {}
    if info["task"] == 1:
        from rec.verification import verify_evidence

        for name in ("main", "replication", "v2", "v3", "v3_support", "v4"):
            outputs[name] = verify_evidence(CODE / "reports" / name)
        assert digest(CODE / "paper/article.tex") == build["source_sha256"]
        for name, sha in build["figure_sha256"].items():
            assert digest(CODE / name) == sha
        from write_paper import build_blocks, updated_source
        from paper_figures import figure_data

        blocks, claims = build_blocks()
        source = (CODE / "paper/article.tex").read_text("utf-8")
        assert source == updated_source(source, blocks)
        assert claims == read(CODE / "paper/evidence/claims.json")
        assert figure_data() == read(CODE / "paper/evidence/figure_statistics.json")
    else:
        from verify_task2 import verify as verify_first
        from verify_task2b import verify as verify_second

        outputs["cycle_a"] = verify_first(CODE / "reports/task2")
        outputs["cycle_b"] = verify_second(CODE / "reports/task2b")
        for name, sha in build["inputs"].items():
            assert digest(CODE / name) == sha
        for name, sha in read(CODE / "paper/submission_task2/evidence_inputs.json").items():
            assert digest(CODE / name) == sha
        numbers = (CODE / "paper/submission_task2/numbers.tex").read_text("utf-8")
        a, b = (
            read(CODE / "reports/task2/evidence/results.json"),
            read(CODE / "reports/task2b/evidence/results.json"),
        )
        for prefix, result, contrast in (
            ("A", a, a["primary_anchored_minus_ensemble"]),
            ("B", b, b["comparisons"]["gated-minus-continue_ce"]),
        ):
            values = {
                "Users": str(result["users"]),
                "Queries": str(result["queries"]),
                "Delta": f"{contrast['delta']:+.6f}",
                "Lo": f"{contrast['ci95'][0]:+.6f}",
                "Hi": f"{contrast['ci95'][1]:+.6f}",
            }
            for name, value in values.items():
                assert f"\\newcommand{{\\{prefix + name}}}{{{value}}}" in numbers
        gain = 100 * (b["methods"]["continue_ce"]["ndcg"] / b["methods"]["sft"]["ndcg"] - 1)
        assert f"\\newcommand{{\\Gain}}{{{gain:.1f}}}" in numbers
        for name, result in (("cycle_a", a), ("cycle_b", b)):
            table = (CODE / f"paper/submission_task2/{name}.tex").read_text("utf-8")
            for line in table.splitlines():
                if " & " not in line or "Метод" in line:
                    continue
                values = [float(x.strip().split()[0]) for x in line.split("&")[1:]]
                assert any(
                    np.allclose(values, [r["ndcg"], r["recall"], r["map"]], rtol=0, atol=0.00000051)
                    for r in result["methods"].values()
                )
    print(
        json.dumps(
            dict(
                status="verified",
                task=info["task"],
                pages=info["pages"],
                report_sets=len(outputs),
                paper_sha256=info["paper_sha256"],
                original_files=len(origin),
            ),
            ensure_ascii=False,
        )
    )
    return outputs


if __name__ == "__main__":
    verify()
