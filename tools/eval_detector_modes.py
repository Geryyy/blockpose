#!/usr/bin/env python3
"""Retired Python detector-mode benchmark.

Free discovery is now C++-only and the Python implementations of placed and
gripped verification were deliberately removed.  This legacy all-mode report
therefore has no valid implementation until those C++ query APIs exist.
"""

import argparse


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    parser.error(
        "retired: this report required deleted Python detector modes. "
        "Use the C++ discovery bridge benchmarks; reintroduce this tool only "
        "after native placed and gripped verification APIs are available."
    )


if __name__ == "__main__":
    raise SystemExit(main())
