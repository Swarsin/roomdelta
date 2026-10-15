# Kaggle bridge — inference stub

`run_vggt.py --help` documents the future manifest/output interface. Execution
exits with an explicit M2-pending error and writes nothing. No model, checkpoint,
GPU configuration, inference or cache import/export is implemented yet.

The Kaggle reconstruction runner saves results for the local RoomDelta pipeline.
M1 can prepare frames + an ordered manifest locally. M2 will inspect the actual VGGT API and
normalize predictions to `ReconstructionCache`. The user returns real cache files
to `outputs/<scenario>/reconstruction/`. The local `--from-cache` CLI is a scaffold
not a functioning cache reader. Input hashes, model revision, preprocessing,
calibration and dependency versions must be preserved for future cache keys.
