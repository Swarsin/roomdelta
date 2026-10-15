# Dataset preparation and video preprocessing

Frame preparation runs on CPU using OpenCV contrib headless. It does not load
torch, model weights or reconstruction caches. Its outputs are selected **2D**
frames/observations, not cameras, depths, metric scale or change predictions.
Synthetic tests exercise software behavior; authentic datasets still require
human capture, annotation and inspection.

## Supply actual recordings

Put a `before.mp4` / `after.mp4` pair for each scenario under `data/<scenario>/`:

| Scenario | What to record and annotate |
|---|---|
| `unchanged` | Physically unchanged room; vary the phone path/viewpoints only. Complete truth has no physical change events. |
| `basic` | Actually move a chair, remove a backpack and add a box; note identity and measure displacement where practical. |
| `occlusion` | An object remains physically present but is hidden in the after views; record this in notes, not as a removal. |
| `replacement` | Swap objects at an occupied location and introduce a blocking box; record actual removal/addition and identity in truth. |

For each pair:

- Film the same indoor area slowly, preferably 15–30 seconds per recording, with
  translation around the area, overlap between successive views, and clear views
  of unchanged walls/floor/background. Pause briefly to get sharp views.
- Keep the lens/zoom fixed, avoid abrupt turns, and use steady lighting. The
  software cannot infer that a panorama-only camera path has useful 3D baseline.
- Keep a marker fixed relative to the static room across both timepoints. Record
  the exact dictionary/ID and **ruler-measured** side of its outer black square
  (not the surrounding white margin and not the nominal print setting).
- Capture the marker clearly in several frames. If you supply one scalar side
  length, it applies to every detected marker; use only equally sized markers.
- Supply notes on actual changes, object identities, a measured movement distance
  with measurement method/uncertainty, and an independent measured distance/object
  for later scale checking. Measurements that are unavailable remain unknown.

The software does not manufacture these recordings or measurements. `data/` and
`outputs/` are excluded from Git; processing does not upload files anywhere.

## Working commands

Activate the existing environment and install the tested CPU dependencies:

```bash
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest -q
python scripts/prepare_frames.py --help
```

On Windows PowerShell, use `.\.venv\Scripts\Activate.ps1`; the Python commands
are the same. Windows commands are documented; execution has been tested on WSL2.

Prepare one pair with defaults (12 frames per video, 640×480 letterbox):

```bash
python -m roomdelta.pipeline --stage extract-frames --before data/basic/before.mp4 --after data/basic/after.mp4 --scenario basic --config configs/default.yaml --output outputs/basic/frames
```

The wrapper accepts identical options:

```bash
python scripts/prepare_frames.py --before data/basic/before.mp4 --after data/basic/after.mp4 --scenario basic --output outputs/basic/frames_custom --frames-per-recording 12 --candidate-stride 5 --min-blur-score 20 --min-time-separation-s 0.5 --target-size 640 480 --resize-mode letterbox
```

For optional marker detection, add `--aruco --aruco-dictionary DICT_4X4_50
--marker-side-m YOUR_MEASURED_SIDE_IN_METRES`, replacing the placeholder with your
actual positive measurement. Omitting size is allowed: detections then store
`measured_side_m: null`. The measurement is not converted to a reconstruction scale.

`--resize-mode stretch` resizes directly to the target; `center-crop` scales to
cover then crops the centre. `--native-resolution` disables all resize/pad/crop
but still applies display orientation, and cannot be combined with `--target-size`.
For portrait recordings use `--target-size 480 640`. Crop mode may cut markers/scene
regions out of the processed image; original detections remain recorded with a
visibility flag. Choose a **new output directory** for each rerun: existing ones
are rejected to preserve user annotations.

For synthetic video inputs explicitly add `--data-kind synthetic`; real phone
recordings default to `real`. This label is user-supplied provenance, not an
automatic authenticity test. Scenario names allow letters/digits/underscores/
hyphens, start with a letter/digit, and have at most 80 characters.

## Settings and selection policy

All defaults live in `configs/default.yaml` under `extraction`. CLI overrides
are saved in `effective_config.yaml`:

| Setting | Default | Meaning |
|---|---|---|
| `frames_per_recording` | 12 | Exact required selected count per video |
| `candidate_stride` | 5 | Score every fifth decoded frame; all frames are decoded |
| `min_blur_score` | 20.0 | Reject lower Laplacian variance scores |
| `min_time_separation_s` | 0.5 | Minimum selected timestamp spacing |
| `diversity_weight` | 0.35 | Appearance novelty weight vs normalized sharpness (YAML) |
| `target_width`, `target_height` | 640, 480 | Common processed resolution; both null means native |
| `resize_mode` | `letterbox` | Resize to fit, then black padding |
| `aruco_enabled` | false | Detect markers on selected original decoded images |
| `aruco_dictionary` | `DICT_4X4_50` | OpenCV dictionary; 4×4 through 7×7 families or ARUCO_ORIGINAL |
| `marker_side_m` | null | User-measured marker side, never inferred |

Sharpness is variance of a 3×3 Laplacian on 8-bit grayscale, downsampled so the
maximum dimension is at most 320 pixels. It is resolution/texture/compression-
dependent, **not a calibrated blur probability**. Uniform walls and motion blur
can both have low scores; codec artifacts can inflate them. Tune on inspected
real selections and keep the effective settings.

Usable candidates are partitioned into temporal bins. Each bin chooses a
spacing-compatible candidate weighted by relative sharpness and minimum RMS
distance of 32×32 grayscale thumbnails to previously selected candidates. Empty
bins are filled from the remaining pool. If greedy choices prevent a feasible
count, an earliest-spaced sharp-enough fallback is logged. Insufficient count
is an error, not silent reduction. Frames are finally sorted by decoded source index.

**Thumbnail difference and temporal spacing do not establish distinct camera
positions or independent observations.** Lighting, 2D motion or a rotating camera
can produce novelty. Similar images are not hard-rejected, so stationary controls
remain possible. Inspect overlap/viewpoints; physical pose validation is future work.

## Output layout and inspection

```text
outputs/basic/frames/
  before/frame_00000015.png      Names use decoded source frame indices
  after/frame_00000020.png       Actual selected numbers depend on the recording
  frames_manifest.json
  contact_sheet_before.jpg
  contact_sheet_after.jpg
  ground_truth.template.json
  effective_config.yaml
  environment.json
  run_manifest.json
```

The filenames above illustrate the naming convention; they are not claimed
results from a phone recording. Extraction uses lossless PNGs; contact sheets are
JPEG previews with index/timestamp/blur labels and green ArUco outlines. Saved
individual frames contain **no overlay**, preserving them for reconstruction.

Open both contact sheets and individual PNGs in an image viewer. Check sharpness,
room coverage, overlap, static background, marker clarity, orientation and black
padding/cropped content. Review the manifest's decoder/timestamp warnings, source
hashes, decoded counts, rejected count, selected indices and transforms. A
successfully prepared pair is not automatically an acceptable reconstruction dataset.

## Frame manifest schema: `roomdelta-frames/2`

`frames_manifest.json` is UTF-8 JSON, with no NaN/Inf. Major fields:

- `schema_version`, `scenario`, `data_kind`, `created_at_utc`, `opencv_version`.
- `path_base: manifest_directory`; `frame_order: before_then_after_source_index_ascending`.
- `coordinate_unit: pixels_only_no_3d_calibration`, `calibration_status: uncalibrated`.
- `selection_method`, `selection_caveat`, full `extraction_config`.
- `sources.before` / `sources.after`: source ID/filename/path/path kind/SHA-256,
  actual decoded count, advertised count (nullable), FPS (nullable), original
  raw decoded and upright `(W,H)`, decoder backend, orientation metadata/convention,
  applied clockwise quarter-turn, raw-to-upright transform, timestamp basis, warnings,
  candidate count, blur-rejected count and selected count.
- `frames`: before entries followed by after entries, each chronologically ordered.

Source paths are relative to the manifest where possible. Different Windows
drives require an absolute source path; image paths are always portable relative
POSIX-style paths. Transferring the frames and manifest together preserves their
image references; raw recordings need not accompany that transfer.

Each frame contains:

| Field | Shape/unit/meaning |
|---|---|
| `index` | Global stable contiguous manifest index, starting at 0 |
| `source_frame_index` | Zero-based decoded index within the recording, independent of manifest index |
| `scenario`, `timepoint`, `source_recording_id`, `source_filename` | Scenario and unambiguous source reference |
| `file_path`, `image_sha256` | Relative saved image path and actual checksum |
| `timestamp_s`, `timestamp_basis` | Source-relative seconds and how they were obtained |
| `original_size_wh`, `raw_decoded_size_wh` | Both name the raw decoded sensor `(W,H)` before display rotation |
| `upright_size_wh`, `processed_size_wh` | Display-oriented and final saved `(W,H)` |
| `rotation_degrees_clockwise` | Applied display correction: 0, 90, 180 or 270 |
| `raw_to_upright`, `upright_to_processed` | Separate 3×3 transforms R and S |
| `pixel_transform` | Complete finite raw-to-final transform A = S @ R, including rotation, resize, padding/crop |
| `valid`, `blur_score`, `quality_reasons` | All selected entries valid; heuristic sharpness/reasons |
| `preprocessing` | Actual resized `(W,H)`, padding/crop `(left,top,right,bottom)`, content `(x,y,w,h)`, mode/interpolation/pixel-centre convention |
| `aruco_detections` | List, empty if disabled/no detection; ID, dictionary, corners and supplied size |

`load_frame_records(Path('.../frames_manifest.json'))` in `roomdelta.frames`
loads entries into the existing `FrameRecord` contract and resolves image paths
from the manifest directory. It checks contract fields, contiguous ordering,
source references/indices and portable paths. It is not a reconstruction cache
reader and does not verify all image checksums/resolutions on load.
Version 2 also checks rotation decomposition/dimensions. Legacy version 1 remains
readable with its original raw-sensor semantics; old PNGs are not silently reinterpreted
as upright. Regenerate legacy extractions that had display-rotation metadata.

### Timestamp provenance and decode failures

Sequential decode establishes source indices. OpenCV `CAP_PROP_POS_MSEC` is used
only when all decoded timestamps are finite, nonnegative and increasing; this
is backend-reported timing, not a guaranteed exact phone presentation timestamp.
If timing is unusable and FPS is valid, **the entire recording** uses
`source_frame_index / FPS`, labelled `frame_index_over_fps_approximate`. This can
be inaccurate for variable-rate phone videos. Missing both usable timing and FPS
is an error; no timestamps are invented or described as exact.

An empty/unopenable video, resolution change, or a read failure before advertised
frame count is an error. OpenCV cannot always distinguish corruption from EOF when
frame-count metadata is absent; that limitation is recorded as a warning. Incorrect
advertised counts can cause conservative rejection. Frames are decoded a second
time sequentially for saving, avoiding approximate seeking. Inputs changing during
preparation fail. Files are staged and published only after pair preparation succeeds.

### Display orientation: exactly one rotation owner

Before the first read on **both decode passes**, OpenCV automatic orientation is
disabled and checked. RoomDelta then applies the metadata-derived quarter-turn
once, before quality analysis, resize and marker detection. A backend that cannot
disable auto-rotation is rejected before reading; manually rotating an already
auto-corrected image would be a double-rotation bug.

Rotation sign depends on the metadata API. OpenCV FFMPEG
`CAP_PROP_ORIENTATION_META` is a clockwise image-space correction, whereas FFprobe's
`side_data.rotation` / FFmpeg `av_display_rotation_get` uses counterclockwise angles.
Thus an FFprobe **−90°** display matrix corresponds to OpenCV **+90°**, i.e. a
clockwise pixel permutation. These values must not be passed interchangeably to
`cv2.rotate`. The manifest records `orientation_metadata_convention`,
`orientation_policy: disable_opencv_auto_then_rotate_once`,
`rotation_applied_by: roomdelta` and effective `automatic_rotation: false`.
The latter describes the decoder flag, not whether display correction was applied.

The mapping follows OpenCV 5.0's
[applyMetadataRotation](https://github.com/opencv/opencv/blob/5.0.0/modules/videoio/src/cap_interface.hpp)
and [FFMPEG angle conversion](https://github.com/opencv/opencv/blob/5.0.0/modules/videoio/src/cap_ffmpeg_impl.hpp).
Nonzero metadata from unverified non-FFMPEG backends and non-quarter-turn angles
fail clearly rather than guessing. Codec/backend support varies; inspect actual
phone outputs. FFmpeg CLI tools are used only for optional encoded regression
fixtures; normal extraction requires only the pinned OpenCV package.

### Exact preprocessing / later intrinsics

Pixel centres are integers, compatible with the existing geometry contract.
First rotate raw pixels to upright pixels using R. For raw width W and height H:

```text
R_90cw  = [[ 0, -1, H-1], [ 1,  0,   0], [0,0,1]]
R_180   = [[-1,  0, W-1], [ 0, -1, H-1], [0,0,1]]
R_270cw = [[ 0,  1,   0], [-1,  0, W-1], [0,0,1]]
```

These are lossless integer pixel permutations, not interpolated rotations.
OpenCV linear resize then maps `u' = sx*(u+0.5)-0.5` and likewise v. Using actual
integer resized dimensions, `sx = resized_W/upright_W`, `sy = resized_H/upright_H`:

```text
S = [[sx, 0, (sx-1)/2 + pad_left - crop_left],
     [0, sy, (sy-1)/2 + pad_top  - crop_top ],
     [0,  0,                 1             ]]
A = S @ R
```

Integer rounding can make sx and sy slightly different even in aspect-preserving
modes; the actual values are retained. Padding is black, and `content_rect_xywh`
identifies nonpadding content. Later depth validity must mask padding; black pixels
are not evidence of free space. Frame preparation does not infer intrinsics.

With no rotation, the existing helper can compute canonical `K_processed = S @ K`.
With rotation, `A @ K_raw` is a valid projection matrix for **raw camera axes**, but
it is not generally a standard upper-triangular OpenCV K with positive focal lengths.
The resize-only `transform_intrinsics` helper rejects rotations to prevent that
mistake. Future calibrated camera use must explicitly account for the camera-axis
convention rather than relabelling `A @ K_raw` as canonical K.

ArUco corners are detected in upright images, in OpenCV canonical marker corner
order. Version 2 stores 4×2 `corners_raw_decoded_px`, `corners_upright_px` and
`corners_processed_px`; `corners_original_px` aliases raw decoded coordinates for
compatibility. Applying A to the raw corners or S to upright corners gives the
same final positions. Cropped corners may lie outside the processed image.
`all_corners_in_processed_image` records this. Merely detecting a marker—even with
a known physical side—does not establish a metric reconstruction or camera pose.

## Ground truth schema and annotation

The installable JSON Schema (Draft 2020-12) is
`src/roomdelta/schemas/ground_truth.schema.json`. Copy the generated pending
`ground_truth.template.json` to `ground_truth.json`, preserving the original
template, and edit with actual observations. No existing truth file is overwritten.

Top-level required fields are `schema_version: roomdelta-ground-truth/1`, scenario,
real/synthetic provenance, `annotations_status: pending|complete`, annotator,
source recording filenames, +Z-up coordinate declaration, `origin_description`,
`unit: m`, events and notes. Pending empty events **do not mean unchanged**.
Mark complete only after human annotation; complete requires an annotator.

Each event requires a unique human ID, object label, `type: added|removed|moved`,
nullable `before_centroid`, nullable `after_centroid`, nullable
`measured_displacement`, and notes. Added has no before centroid; removed has no
after centroid. An occupied-location replacement can be annotated as a removed
and added object with identity notes; a hidden-but-present object is not removed.

- Known centroids are `{xyz_m: [x,y,z], uncertainty_m: nonnegative, method: ...}`.
- Known movement is `{distance_m: nonnegative, vector_m: [dx,dy,dz] or null,
  uncertainty_m: nonnegative, method: ...}`; vector is after minus before.
- `uncertainty_m` is a human estimate of absolute ±metre measurement uncertainty,
  not a confidence probability. State how you measured it in `method`.
- Unknown positions/displacement stay `null`; do not replace them with zero.
  Positions or direction vectors require a nonempty shared origin/axis description.
  A scalar measured displacement does not require absolute 3D positions.
- IDs must be unique; the schema enforces shape/types and duplicate whole events,
  but cross-field numeric consistency and ID-only uniqueness need human review.

For example, an **unmeasured annotation placeholder**, not a claim about your room:

```json
{
  "id": "ENTER_OBJECT_ID",
  "object_label": "ENTER_ACTUAL_LABEL",
  "type": "moved",
  "before_centroid": null,
  "after_centroid": null,
  "measured_displacement": null,
  "notes": "Enter actual movement and measurement notes."
}
```

Validate your JSON using the installed dev extra (replace the input path as needed):

```bash
python -c "import json, jsonschema; from importlib.resources import files; schema=json.loads(files('roomdelta').joinpath('schemas/ground_truth.schema.json').read_text()); truth=json.load(open('outputs/basic/frames/ground_truth.json', encoding='utf-8')); jsonschema.validate(truth, schema); print('Ground-truth schema valid; human review still required')"
```

No real phone input, metric scale, event identity or evaluation result is validated
by synthetic tests. Reconstruction remains a separate, unimplemented stage.
