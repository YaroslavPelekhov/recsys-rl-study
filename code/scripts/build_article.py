"""Rebuild the submitted paper into article-rebuilt.pdf, preserving the original."""

import json
import shutil
import subprocess
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
PACKAGE = CODE.parent


def main():
    info = json.loads((PACKAGE / "SUBMISSION.json").read_text("utf-8"))
    source = CODE / info["latex_main"]
    engine = shutil.which("xelatex")
    if not engine:
        raise SystemExit("Install XeLaTeX from TeX Live or MiKTeX")
    output = CODE / "tmp/submission-latex"
    output.mkdir(parents=True, exist_ok=True)
    arguments = [
        engine,
        "-interaction=nonstopmode",
        "-halt-on-error",
        f"-output-directory={output}",
        source.name,
    ]
    if "miktex" in engine.lower():
        arguments.insert(1, "--disable-installer")
    for number in (1, 2):
        result = subprocess.run(arguments, cwd=source.parent, capture_output=True)
        (output / f"pass{number}.txt").write_bytes(result.stdout)
        if result.returncode:
            raise RuntimeError(result.stdout.decode("utf-8", errors="replace")[-3000:])
    log = (output / "article.log").read_text("utf-8", errors="replace")
    assert "Overfull" not in log and "undefined" not in log
    target = PACKAGE / "article-rebuilt.pdf"
    shutil.copy2(output / "article.pdf", target)
    print(target)


if __name__ == "__main__":
    main()
