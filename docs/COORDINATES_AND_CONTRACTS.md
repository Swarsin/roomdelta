# Coordinates and shared data contracts

The runtime definitions are in `src/roomdelta/contracts.py`. These are normalized
interfaces for later milestones, not a claim that a raw model already produces
these arrays. Portable cache serialization/checksums and cache-key computation
are deferred to M2.

## Coordinates, units and projection

- Canonical world is right-handed: +X right, +Y forward, **+Z up**. The world origin
  and horizontal orientation will be chosen during M2 alignment. This is not GPS.
- Camera axes follow OpenCV: **+X right, +Y down, +Z forward**. Both frames are
  right-handed; do not substitute an OpenGL camera convention.
- Homogeneous coordinates are column vectors mathematically:
  `X_c = R_cw X_w + t_cw`, with `T_cw = [[R_cw,t_cw],[0,0,0,1]]`.
  NumPy row-array storage implements the same equation as `X_w @ R_cw.T + t_cw`.
- `T_wc = inverse(T_cw)`. The world-space camera centre is `T_wc[:3,3]`,
  equivalently `-R_cw.T @ t_cw`. Rerun child-to-parent transforms log **T_wc**.
- Pinhole projection uses `q = K @ X_c`, then `(u,v) = (q_x/q_z,q_y/q_z)`.
  `K = [[fx,skew,cx],[0,fy,cy],[0,0,1]]`, in pixels. Positive camera-z is required.
  Pixel centres are at integer coordinates. In-image bounds are `0 <= u < W`,
  `0 <= v < H`; boundary points at W or H are outside.
- Stored depth is **camera-z**, not Euclidean camera-to-point range. For projection,
  points at zero/negative camera-z have NaN pixels and an invalid mask; out-of-image
  positive-z points retain their coordinates but are invalid.
- Dimensions are `(width,height)` in manifests; depth indexing is `[frame,v,u]`.
  A resize/crop affine A maps original pixels to processed pixels. Record A and
  transform canonical intrinsics with `K_processed = S @ K_upright` for resize/pad/crop S.
  M1's full `pixel_transform = S @ R` maps **raw decoded** to final pixels, including
  a proper display quarter-turn R. The geometry contract accepts scaled proper
  quarter-turns, rejecting reflection/shear. `A @ K_raw` after rotation is a projection
  matrix in raw camera axes, not generally canonical OpenCV K. The resize-only
  `transform_intrinsics` helper rejects that case; camera axes must be handled explicitly.
  See [dataset preparation](DATASET_PREPARATION.md) for exact pixel-centre offsets.

## Scale labels

`Calibration` contains a finite 4×4 `model_to_world` Sim(3): its linear block must
be `sR`, `s > 0`, with a proper rotation; anisotropic scale/reflections are rejected.
It is metadata describing the conversion from raw model coordinates, not a
transform to apply repeatedly to already normalized arrays.

| Status | Stored unit | Meaning |
|---|---|---|
| `uncalibrated` | `arbitrary` | Relative model-space scale; no metric distance claims |
| `measured` | `m` | Measured calibration; method and relative scale uncertainty required |
| `synthetic` | `m` | Analytic fixture defined in metres; no physical measurement/inference |

Relative scale uncertainty is a nonnegative fraction, not a probability or a
pixel error. Synthetic zero uncertainty refers only to exact fixture scale, not
real sensor reliability. Raw depths, camera translations and point maps must all
be converted consistently by M2 before a metric cache is accepted. A metric
`T_cw` is still rigid: do not put the similarity scale inside its rotation block.
World-coordinate normalization (+Z up) is also the M2 adapter's responsibility.

## Contracts

### FrameRecord

Scenario/timepoint (`before` or `after`), `Path` to the processed image, source
filename and timestamp in seconds, original/processed `(W,H)`, 3×3 pixel affine,
stable integer index, validity, optional nonnegative blur score and quality reasons.
`original_size_wh` always means raw decoded sensor dimensions, including when display
orientation swaps the axes. Version 2 manifests additionally name raw/upright sizes
and the rotation/resize decomposition; the complete affine includes both steps.
M1 adds optional `source_frame_index` (decoded index within the source video),
preserving backward compatibility for M0 callers. Manifest index remains global.
Paths are metadata; constructors do not access user files. The CLI checks provided
recording/cache paths. Invalid frames cannot have valid depth in a cache.

### ReconstructionCache

Schema version `roomdelta/1`, ordered frames, `K[N,3,3]`, `T_cw[N,4,4]`,
`depth[N,H,W]`, `depth_valid[N,H,W]`, `Calibration`, `ModelInfo`, optional
`depth_confidence[N,H,W]` and optional world points `[P,3]`. Arrays are NumPy CPU
arrays. All frames share a processed resolution and scenario; indices must equal
`0..N-1` in array order. Timepoints are derived from frames, avoiding duplicated
labels that could disagree. Proper rigid poses are required.

Depth values must all be finite; invalid raw NaN/Inf must be explicitly masked
and replaced with a finite placeholder (normally zero) by the future adapter.
Valid depth values are strictly positive. Masks must actually have boolean dtype.
Confidence is model-specific and need not lie in [0,1]; its interpretation must
be provided if confidence is present. Shape/finite/pose validation cannot establish
that an adapter used the correct original intrinsics: retain the preprocessing
affine and verify the `A @ K_original` conversion in the adapter.

`ModelInfo` records real vs synthetic provenance, model/checkpoint revision (both
present or both absent), execution device, random seed, preprocessing, dependency
versions, source SHA-256 digests and confidence interpretation. Synthetic scale
and provenance must agree. M2 will enforce completeness of real cache provenance
and compute cache keys from inputs, model revision, preprocessing and calibration.

### VoxelGrid / VoxelEvidence

Metric minimum-corner `origin_m[3]`, `bounds_m[2,3]` (min/max), positive voxel
size in metres, integer `(nx,ny,nz)` shape, before/after evidence and calibration.
Bounds must equal `[origin, origin + shape * voxel_size]`. Cell centre at index i
is `origin + (i + 0.5) * voxel_size`. Uncalibrated grids are rejected.

Per-timepoint evidence uses compact integer `[nx,ny,nz]` occupied/free counts,
state labels (`unknown=0`, `free=1`, `occupied=2`, `conflicting=3`), unsigned
reason flags, view-to-correlation-group mapping and quality/provenance metadata.
Reason-bit meanings and evidence aggregation are deferred to M3. Counts do not
by themselves establish independent observations or confirm a change.

### ChangeCandidate

ID, enum `added/removed/appearance/unknown`, nonempty integer voxel indices
`[M,3]`, centroid `[3]` and bounds `[2,3]` in metres, unique supporting frame IDs,
local static-alignment diagnostics and reasons. Centroid must lie in bounds.
`validate_context(grid_shape, frame_count)` rejects out-of-range voxel/view IDs.
Static diagnostics will be computed from trustworthy unchanged background in M4,
not candidate changed regions. Candidate construction does not confirm an event.

### ChangeEvent / ChangesDocument

Event ID/type (`added`, `removed`, `moved`, `replacement-candidate`, `unknown`),
semantic label (default `unknown object`), before/after optional centroids in metres,
nonnegative size `[3]`, disposition, evidence reasons, status
(`candidate/confirmed/abstained`), supporting frame IDs, crop paths and optional
Rerun entity ID/confidence. Added requires after centroid; removed requires before;
moved requires both. Confirmed requires a concrete event type and supporting IDs;
unknown and replacement candidates cannot be marked confirmed. These structural
checks do **not** replace M4 evidence verification or M5 correspondence checks.
Absence of points alone must never confirm removal. Occupied-to-occupied does not
establish unchanged identity. Confidence is a score, not calibrated probability.

The future `changes.json` document has explicit `schema_version`,
`coordinate_system`, `unit`, `source_scenario`, `model_info`, `calibration`, and
`events`, with unique event IDs and metric calibration. M0 defines the contract
but does not write a fake changes report.

## Ownership and validation limits

Dataclasses are frozen; NumPy buffers and metadata mappings are not deep-frozen.
Treat validated inputs as read-only and revalidate after edits. Shape/finite
validation is software correctness only. Pose reliability, static alignment,
physical scale, visibility, independence and real event identity require evidence
from later milestones and actual recordings.
