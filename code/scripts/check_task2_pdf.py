"""Render Task 2 PDF for visual QA; run in a PDF-capable Python environment."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path


def main():
    import pdfplumber
    from PIL import Image, ImageOps

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, default=Path("output/pdf/task2-rl-paper.pdf"))
    parser.add_argument("--poppler", default=shutil.which("pdftoppm"))
    parser.add_argument("--output", type=Path, default=Path("tmp/pdfs/task2"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    pages = []
    with pdfplumber.open(args.pdf) as document:
        for index, page in enumerate(document.pages, 1):
            text = page.extract_text() or ""
            assert "�" not in text and "□" not in text, f"Missing glyph on page{index}"
            assert any("а" <= ch <= "я" for ch in text), f"No Cyrillic on page{index}"
            outside = [
                c
                for c in page.chars
                if c["x0"] < 40
                or c["x1"] > page.width - 40
                or c["top"] < 23
                or c["bottom"] > page.height - 23
            ]
            assert not outside, f"Text outside margins on page{index}"
            pages.append({"page": index, "text_chars": len(text), "outside_margins": 0})
    if args.poppler:
        subprocess.run(
            [
                args.poppler,
                "-r",
                "100",
                "-png",
                str(args.pdf.resolve()),
                str((args.output / "page").resolve()),
            ],
            check=True,
        )
        images = []
        width = len(str(len(pages)))
        for number in range(1, len(pages) + 1):
            image_path = args.output / f"page-{number:0{width}d}.png"
            with Image.open(image_path) as source:
                thumbnail = source.copy()
            thumbnail.thumbnail((430, 610))
            images.append(ImageOps.expand(thumbnail, border=5, fill="#ccd4d9"))
        contact = Image.new("RGB", (len(images) * 440, 625), "white")
        for index, picture in enumerate(images):
            contact.paste(picture, (index * 440, 0))
        contact.save(args.output / "contact.png")
    (args.output / "quality.json").write_text(
        json.dumps(
            {
                "pages": pages,
                "page_count": len(pages),
                "pdf_sha256": hashlib.sha256(args.pdf.read_bytes()).hexdigest(),
            },
            indent=2,
        ),
        "utf-8",
    )
    print(json.dumps(pages))


if __name__ == "__main__":
    main()
