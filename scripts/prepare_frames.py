"""M1 before/after video preparation wrapper; uses the installed RoomDelta CLI."""

import sys

from roomdelta.pipeline import main


if __name__ == "__main__":
    raise SystemExit(main(["--stage", "extract-frames", *sys.argv[1:]]))
