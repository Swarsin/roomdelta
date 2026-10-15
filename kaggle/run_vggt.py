"""M0 handoff stub only; no imports of torch, VGGT or checkpoint downloads."""

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Kaggle VGGT handoff stub; actual inference/export is planned for M2.")
    parser.add_argument("--manifest", type=Path, required=True, help="ordered frame manifest supplied after M1")
    parser.add_argument("--output", type=Path, required=True, help="future portable reconstruction cache directory")
    parser.parse_args()
    parser.error("VGGT inference and cache import/export are not implemented in M0. No predictions were generated.")


if __name__ == "__main__":
    main()
