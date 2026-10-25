"""Build the unified second paper; preserve both experiment-level papers."""

import shutil
import subprocess
from pathlib import Path

from rec.io import file_hash, write_json

ROOT = Path(__file__).resolve().parents[1]


def build():
    source = ROOT / "paper/submission_task2/article.tex"
    temporary = ROOT / "tmp/latex-submission-task2"
    temporary.mkdir(parents=True, exist_ok=True)
    engine = shutil.which("xelatex")
    if not engine:
        raise FileNotFoundError("XeLaTeX is required")
    args = [
        engine,
        "-interaction=nonstopmode",
        "-halt-on-error",
        "-file-line-error",
        f"-output-directory={temporary}",
        source.name,
    ]
    if "miktex" in engine.lower():
        args.insert(1, "--disable-installer")
    for number in (1, 2):
        result = subprocess.run(args, cwd=source.parent, capture_output=True)
        (temporary / f"pass{number}.txt").write_bytes(result.stdout)
        if result.returncode:
            raise RuntimeError(result.stdout.decode("utf-8", errors="replace")[-3000:])
    log = (temporary / "article.log").read_text("utf-8", errors="replace")
    problems = [s for s in log.splitlines() if "Overfull" in s or "undefined" in s]
    if problems:
        raise ValueError("\n".join(problems))
    output = ROOT / "output/pdf/task2-final-submission.pdf"
    shutil.copy2(temporary / "article.pdf", output)
    files = [
        source,
        *sorted(source.parent.glob("*.tex")),
        *sorted((source.parent / "figures").glob("*.pdf")),
        source.parent / "evidence_inputs.json",
    ]
    write_json(
        ROOT / "output/pdf/task2-final-build.json",
        dict(
            pdf_sha256=file_hash(output),
            inputs={p.relative_to(ROOT).as_posix(): file_hash(p) for p in files},
            passes=2,
            overfull_boxes=0,
            undefined_references=0,
        ),
    )
    shutil.copy2(
        ROOT / "output/pdf/research-paper.pdf", ROOT / "output/pdf/task1-final-submission.pdf"
    )
    print(output)


if __name__ == "__main__":
    build()
