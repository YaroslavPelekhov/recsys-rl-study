"""Build the multi-file Task 2 LaTeX project using local XeLaTeX."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from rec.io import file_hash, write_json


def build():
    root = Path(__file__).resolve().parents[1]
    source = root / "paper/task2/article.tex"
    temporary = root / "tmp/latex-task2"
    temporary.mkdir(parents=True, exist_ok=True)
    executable = shutil.which("xelatex")
    if not executable:
        raise FileNotFoundError("Install XeLaTeX (TeX Live/MiKTeX)")
    args = [
        executable,
        "-interaction=nonstopmode",
        "-halt-on-error",
        "-file-line-error",
        f"-output-directory={temporary}",
        source.name,
    ]
    if "miktex" in executable.lower():
        args.insert(1, "--disable-installer")
    for number in (1, 2):
        result = subprocess.run(args, cwd=source.parent, capture_output=True)
        (temporary / f"pass{number}.txt").write_bytes(result.stdout)
        if result.returncode:
            raise RuntimeError(result.stdout.decode("utf-8", errors="replace")[-4000:])
    log = (temporary / "article.log").read_text("utf-8", errors="replace")
    warnings = [line for line in log.splitlines() if "Overfull" in line or "undefined" in line]
    if warnings:
        raise ValueError("\n".join(warnings))
    output = root / "output/pdf/task2-rl-paper.pdf"
    shutil.copyfile(temporary / "article.pdf", output)
    files = [
        source,
        source.parent / "numbers.tex",
        *sorted((source.parent / "figures").glob("*.pdf")),
    ]
    write_json(
        root / "output/pdf/task2-build.json",
        {
            "inputs": {p.relative_to(root).as_posix(): file_hash(p) for p in files},
            "pdf_sha256": file_hash(output),
            "passes": 2,
            "overfull_boxes": 0,
            "undefined_references": 0,
        },
    )
    print(output)


if __name__ == "__main__":
    build()
