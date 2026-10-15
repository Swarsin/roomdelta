"""Stage-aware M0 CLI. Later milestones fail explicitly rather than simulating them."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from importlib import metadata
import json
import logging
from pathlib import Path
import platform
import sys
from typing import Sequence

import yaml

from roomdelta import __version__
from roomdelta.config import load_config
from roomdelta.visualization import save_synthetic_scene

LOGGER = logging.getLogger(__name__)
LATER_STAGES = {
    "extract-frames": "M1", "reconstruct": "M2", "volumetric": "M3",
    "verify": "M4", "semantics": "M5", "navigation": "M6", "evaluate": "M7", "all": "M1–M7",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="RoomDelta M0: package/contract scaffold and synthetic visualization.",
        epilog="Implemented stages: environment, synthetic. All reconstruction/change stages are pending.",
    )
    parser.add_argument("--stage", choices=["environment", "synthetic", *LATER_STAGES], default="all",
                        help="targeted stage (default: all; full pipeline is not implemented in M0)")
    parser.add_argument("--before", type=Path, help="before recording (M1+) ")
    parser.add_argument("--after", type=Path, help="after recording (M1+) ")
    parser.add_argument("--from-cache", type=Path, help="reconstruction cache directory (M2+) ")
    parser.add_argument("--output", type=Path, default=Path("outputs/m0"), help="output directory (default: outputs/m0)")
    parser.add_argument("--config", type=Path, help="YAML config; omitted uses built-in configs/default.yaml values")
    parser.add_argument("--check-cuda", action="store_true", help="explicitly import torch and test CUDA for the environment report")
    parser.add_argument("--verbose", action="store_true", help="enable debug logging")
    return parser


def environment_report(check_cuda: bool = False) -> dict[str, object]:
    """Capture real dependency/platform data; CUDA probing is explicitly opt-in."""
    report: dict[str, object] = {
        "python": platform.python_version(), "executable": sys.executable,
        "platform": platform.platform(), "roomdelta": __version__,
        "dependencies": {dist.metadata["Name"]: dist.version for dist in metadata.distributions()},
        "cuda_probe": "not requested; torch was not imported",
    }
    if check_cuda:
        try:
            import torch
        except ImportError:
            report["cuda_probe"] = {"status": "torch not installed"}
        else:
            available = torch.cuda.is_available()
            probe: dict[str, object] = {
                "torch_version": torch.__version__, "cuda_build": torch.version.cuda,
                "available": available,
                "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            }
            if available:
                probe["tensor_sum"] = torch.arange(4, device="cuda").sum().item()
            report["cuda_probe"] = probe
    return report


def _validate_inputs(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if (args.before is None) != (args.after is None):
        parser.error("provide both --before and --after")
    if args.from_cache is not None and args.before is not None:
        parser.error("--from-cache cannot be combined with --before/--after")
    for name in ("before", "after"):
        path = getattr(args, name)
        if path is not None and not path.is_file():
            parser.error(f"{name} recording does not exist or is not a file: {path}")
    if args.from_cache is not None and not args.from_cache.is_dir():
        parser.error(f"cache directory does not exist: {args.from_cache}")
    if args.stage in ("environment", "synthetic") and (args.before is not None or args.from_cache is not None):
        parser.error(f"--stage {args.stage} takes no recording/cache input")


def main(argv: Sequence[str] | None = None) -> int:
    """Run one M0 stage, saving effective config and actual environment metadata."""
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s: %(message)s")
    _validate_inputs(parser, args)
    if args.stage in LATER_STAGES:
        parser.error(f"stage '{args.stage}' is not implemented in M0 (requires {LATER_STAGES[args.stage]}). "
                     "Use --stage synthetic or --stage environment.")
    try:
        config = load_config(args.config)
        report = environment_report(args.check_cuda)
        args.output.mkdir(parents=True, exist_ok=True)
        artifacts: list[str] = []
        if args.stage == "synthetic":
            scene = save_synthetic_scene(args.output / "roomdelta.rrd", config.visualization)
            artifacts.append(scene.name)
        effective_config = asdict(config)
        (args.output / "effective_config.yaml").write_text(yaml.safe_dump(effective_config, sort_keys=False), encoding="utf-8")
        (args.output / "environment.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        run = {
            "schema_version": "roomdelta-run/1", "milestone": "M0", "stage": args.stage,
            "data_kind": "synthetic" if args.stage == "synthetic" else "environment-only",
            "created_at_utc": datetime.now(timezone.utc).isoformat(), "random_seed": config.random_seed,
            "execution_device": "cpu", "model_name": None, "checkpoint_revision": None,
            "inference_performed": False, "config": effective_config,
            "artifacts": artifacts + ["effective_config.yaml", "environment.json"],
        }
        (args.output / "run_manifest.json").write_text(json.dumps(run, indent=2) + "\n", encoding="utf-8")
    except (OSError, ValueError, RuntimeError) as exc:
        LOGGER.error("%s", exc)
        return 1
    LOGGER.info("M0 %s completed: %s", args.stage, args.output)
    if args.stage == "synthetic":
        LOGGER.info("SYNTHETIC ONLY; open with: rerun %s", args.output / "roomdelta.rrd")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
