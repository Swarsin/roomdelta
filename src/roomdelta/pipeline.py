"""Stage-aware M0/M1 CLI; later milestones fail rather than simulating results."""

import argparse
from dataclasses import asdict, replace
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
    "reconstruct": "M2", "volumetric": "M3",
    "verify": "M4", "semantics": "M5", "navigation": "M6", "evaluate": "M7", "all": "M1–M7",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="RoomDelta M1: video dataset preparation, M0 interfaces and synthetic visualization.",
        epilog="Implemented stages: environment, synthetic, extract-frames. Reconstruction/change stages are pending.",
    )
    parser.add_argument("--stage", choices=["environment", "synthetic", "extract-frames", *LATER_STAGES], default="all",
                        help="targeted stage (default: all; full pipeline is not implemented in M0/M1)")
    parser.add_argument("--before", type=Path, help="before recording (M1+) ")
    parser.add_argument("--after", type=Path, help="after recording (M1+) ")
    parser.add_argument("--from-cache", type=Path, help="reconstruction cache directory (M2+) ")
    parser.add_argument("--output", type=Path, help="output directory (default: outputs/frames for extraction, outputs/m0 otherwise)")
    parser.add_argument("--config", type=Path, help="YAML config; omitted uses built-in configs/default.yaml values")
    parser.add_argument("--check-cuda", action="store_true", help="explicitly import torch and test CUDA for the environment report")
    parser.add_argument("--verbose", action="store_true", help="enable debug logging")
    parser.add_argument("--scenario", help="scenario identifier; required for extract-frames")
    parser.add_argument("--data-kind", choices=["real", "synthetic"], default="real", help="video provenance (default: real); fixture inputs must use synthetic")
    parser.add_argument("--frames-per-recording", type=int, help="override selected frame count per video (default config: 12)")
    parser.add_argument("--candidate-stride", type=int, help="analyse every nth decoded frame")
    parser.add_argument("--min-blur-score", type=float, help="minimum grey-thumbnail Laplacian variance")
    parser.add_argument("--min-time-separation-s", type=float, help="minimum selected timestamp spacing")
    parser.add_argument("--target-size", type=int, nargs=2, metavar=("WIDTH", "HEIGHT"), help="processed image dimensions")
    parser.add_argument("--native-resolution", action="store_true", help="disable resize/pad/crop")
    parser.add_argument("--resize-mode", choices=["letterbox", "stretch", "center-crop"], help="override preprocessing mode")
    parser.add_argument("--aruco", action="store_true", default=None, help="enable optional marker corner detection")
    parser.add_argument("--aruco-dictionary", help="OpenCV marker dictionary name")
    parser.add_argument("--marker-side-m", type=float, help="user ruler-measured marker side in metres (not an estimated scale)")
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
    if args.stage == "extract-frames":
        if args.before is None or args.after is None or args.from_cache is not None:
            parser.error("extract-frames requires --before and --after videos, not --from-cache")
        if args.scenario is None:
            parser.error("extract-frames requires --scenario (e.g. unchanged, basic, occlusion, replacement)")
    elif any(getattr(args, name) is not None for name in
             ("scenario", "frames_per_recording", "candidate_stride", "min_blur_score", "min_time_separation_s",
              "target_size", "resize_mode", "aruco", "aruco_dictionary", "marker_side_m")) or args.native_resolution:
        parser.error("frame extraction options require --stage extract-frames")
    if args.native_resolution and args.target_size is not None:
        parser.error("--native-resolution cannot be combined with --target-size")


def main(argv: Sequence[str] | None = None) -> int:
    """Run one supported stage, saving effective config and actual environment metadata."""
    parser = build_parser()
    args = parser.parse_args(argv)
    args.output = args.output or Path("outputs/frames" if args.stage == "extract-frames" else "outputs/m0")
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s: %(message)s")
    _validate_inputs(parser, args)
    if args.stage in LATER_STAGES:
        parser.error(f"stage '{args.stage}' is not implemented in M0/M1 (requires {LATER_STAGES[args.stage]}). "
                     "Use --stage extract-frames, synthetic or environment.")
    try:
        config = load_config(args.config)
        overrides = {name: getattr(args, name) for name in
                     ("frames_per_recording", "candidate_stride", "min_blur_score", "min_time_separation_s", "resize_mode",
                      "aruco_dictionary", "marker_side_m") if getattr(args, name) is not None}
        if args.target_size is not None:
            overrides.update(target_width=args.target_size[0], target_height=args.target_size[1])
        if args.native_resolution:
            overrides.update(target_width=None, target_height=None)
        if args.aruco is not None:
            overrides["aruco_enabled"] = args.aruco
        config = replace(config, extraction=replace(config.extraction, **overrides))
        report = environment_report(args.check_cuda)
        artifacts: list[str] = []
        if args.stage == "extract-frames":
            from roomdelta.frames import prepare_frames

            manifest = prepare_frames(args.before, args.after, args.output, args.scenario, config.extraction, args.data_kind)
            artifacts.extend(["frames_manifest.json", "ground_truth.template.json", "contact_sheet_before.jpg", "contact_sheet_after.jpg"])
            artifacts.extend(frame["file_path"] for frame in manifest["frames"])
        else:
            args.output.mkdir(parents=True, exist_ok=True)
        if args.stage == "synthetic":
            scene = save_synthetic_scene(args.output / "roomdelta.rrd", config.visualization)
            artifacts.append(scene.name)
        effective_config = asdict(config)
        (args.output / "effective_config.yaml").write_text(yaml.safe_dump(effective_config, sort_keys=False), encoding="utf-8")
        (args.output / "environment.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        run = {
            "schema_version": "roomdelta-run/1", "milestone": "M1" if args.stage == "extract-frames" else "M0", "stage": args.stage,
            "data_kind": args.data_kind if args.stage == "extract-frames" else ("synthetic" if args.stage == "synthetic" else "environment-only"),
            "created_at_utc": datetime.now(timezone.utc).isoformat(), "random_seed": config.random_seed,
            "execution_device": "cpu", "model_name": None, "checkpoint_revision": None,
            "inference_performed": False, "config": effective_config,
            "artifacts": artifacts + ["effective_config.yaml", "environment.json"],
        }
        (args.output / "run_manifest.json").write_text(json.dumps(run, indent=2) + "\n", encoding="utf-8")
    except (OSError, ValueError, RuntimeError) as exc:
        LOGGER.error("%s", exc)
        return 1
    LOGGER.info("%s %s completed: %s", "M1 preparation" if args.stage == "extract-frames" else "M0", args.stage, args.output)
    if args.stage == "synthetic":
        LOGGER.info("SYNTHETIC ONLY; open with: rerun %s", args.output / "roomdelta.rrd")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
