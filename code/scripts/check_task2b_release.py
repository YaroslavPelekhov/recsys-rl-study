"""Standalone-package compatibility entry point."""

import argparse

from verify_submission import verify

if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    verify()
