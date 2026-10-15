"""CLI placeholder only: actual evaluation belongs to M7."""

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="RoomDelta evaluation placeholder (not implemented until M7).")
    parser.add_argument("--config", type=Path, help="future evaluation configuration")
    parser.parse_args()
    parser.error("Evaluation is not implemented in M0. No metrics or tables were generated.")


if __name__ == "__main__":
    main()
