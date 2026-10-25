from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from rec.data import download
from rec.experiment import run
from rec.protocol import Config


def main() -> None:
    parser = argparse.ArgumentParser(description="Temporal relational feature-search experiments")
    parser.add_argument("command", choices=["download", "run", "report", "verify"])
    parser.add_argument("--config", type=Path, default=Path("configs/main.json"))
    parser.add_argument("--raw", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("runs/main"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports/main"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.command == "download":
        download(args.raw)
    elif args.command == "run":
        run(Config.load(args.config), args.raw, args.output)
    elif args.command == "report":
        from rec.report import build_report

        build_report(args.output, args.report_dir)
    else:
        from rec.verification import verify_evidence

        print(json.dumps(verify_evidence(args.report_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
