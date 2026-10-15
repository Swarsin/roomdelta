"""M0 synthetic demo wrapper; run after installing RoomDelta."""

import sys

from roomdelta.pipeline import main


if __name__ == "__main__":
    raise SystemExit(main(["--stage", "synthetic", *sys.argv[1:]]))
