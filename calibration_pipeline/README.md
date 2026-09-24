# Task-oriented camera calibration pipeline

This pipeline estimates several camera models, evaluates them on held-out
images, maps residual error over the sensor, checks that distortion inversion
is numerically one-to-one, measures calibration stability by group bootstrap,
and evaluates candidate processing crops. When an independent pose-reference
dataset is supplied, crop selection is based on the localizer's actual pose
error rather than calibration-board reprojection error alone.

The output is a recommendation, an audit trail, diagnostic plots, and a JSON
snippet that can be copied into `high_rate_localizer/config/localizer.json`.
The source configuration is never modified automatically.

## 0-to-100 operating instructions

These stages are deliberately ordered. Do not skip the measurement and
held-out validation stages just because an optimizer reports a small training
RMS.

### 0 — Define the production requirement

Before taking calibration images, write down:

- maximum acceptable `z` RMSE;
- maximum acceptable 95th-percentile absolute `z` error;
- minimum pose availability;
- minimum marker count or acceptable pose-conditioning limit;
- the distances, attitudes, velocities, lighting and temperatures over which
  those requirements must hold.

Put the numerical acceptance requirements under
`crop_sweep.thresholds`. The numbers in `config.example.json` are examples,
not approved requirements for the flight controller.

### 5 — Freeze the camera configuration

Use exactly the production camera, lens, focus, aperture, sensor mode,
resolution and mount. For this localizer that normally means `640 x 400`, the
same 120 Hz sensor mode, fixed focus, 3000 us exposure, and production gain.
Disable autofocus, automatic lens correction, digital stabilization, automatic
zoom and any processing that can change between frames.

If focus, lens mounting, resolution, binning or ISP geometry changes later,
repeat calibration.

### 10 — Create the Python environment

From the repository root:

```sh
python3 -m venv calibration_pipeline/.venv
calibration_pipeline/.venv/bin/python -m pip install \
  -r calibration_pipeline/requirements.txt
```

The system Python may be used if it already provides NumPy, Matplotlib, and an
OpenCV build containing `cv2.aruco`.

Verify it:

```sh
calibration_pipeline/.venv/bin/python -c \
  'import cv2, numpy; print(cv2.__version__, hasattr(cv2, "aruco"))'
```

The last value must be `True`.

Run the pipeline's synthetic, detector, invertibility and end-to-end tests:

```sh
calibration_pipeline/.venv/bin/python -m unittest discover \
  -s calibration_pipeline/tests -v
```

### 15 — Copy and edit the configuration

```sh
cp calibration_pipeline/config.example.json \
  calibration_pipeline/config.production.json
```

Edit at least:

- `capture_metadata`, including camera serial, lens/focus, target identity and
  acquisition conditions;
- `image_size`;
- physical target dimensions;
- `dataset.images`;
- `dataset.group_regex`;
- model list;
- crop candidates;
- operational-validation path;
- every deployment threshold.

All relative paths are resolved relative to the configuration file.

### 20 — Generate the ChArUco board

```sh
calibration_pipeline/.venv/bin/python \
  calibration_pipeline/generate_charuco_board.py \
  --config calibration_pipeline/config.production.json
```

This writes a print-scale SVG, raster preview and metadata under
`calibration_pipeline/generated_board/`.

Print the SVG at **100% / actual size**. Disable fit-to-page scaling. Mount it
to a verified flat, rigid surface; foam board that bows is not adequate for a
millimetre-level depth study.

### 25 — Measure the physical target

After mounting, measure several spans of five or more squares in horizontal
and vertical directions. Divide each span by the number of squares and enter
the measured mean in `target.square_length_m`. Measure the black marker size
and update `marker_length_m` as well.

Record printer, paper, mount, measurements and date. A scale error in the
target becomes a scale error in estimated translation.

### 30 — Prepare independent capture groups

Create at least two genuinely independent capture sessions, preferably more:

```text
calibration_pipeline/data/images/
  session_01_cold/
  session_02_warm/
  session_03_remounted/
```

Configure `dataset.images` with a recursive glob and configure
`dataset.group_regex` so all frames from the same burst/session receive the
same group. The splitter never places a group in both training and validation.
This prevents adjacent video frames from masquerading as independent evidence.

Use the production capture assistant on the Raspberry Pi:

```sh
/usr/bin/python3 calibration_pipeline/capture_session.py \
  --production-config high_rate_localizer/config/localizer.json \
  --pipeline-config calibration_pipeline/config.production.json \
  --session session_01_cold \
  --output-dir calibration_pipeline/data/images \
  --auto
```

The utility reads resolution, frame rate, buffer count, YUV420 format,
exposure, analogue gain, brightness and contrast from the production localizer
configuration and applies them through Picamera2/libcamera. It always saves the
full luma image; `processing_crop` is recorded but intentionally not applied.

Install the Raspberry Pi camera binding with the OS package manager, normally:

```sh
sudo apt install python3-picamera2 python3-opencv
```

Picamera2 is commonly installed only for the system Python, which is why the
capture command uses `/usr/bin/python3`. That interpreter must also expose an
OpenCV build with `cv2.aruco`. The offline fitting pipeline can continue using
the virtual environment.

The preview keys are:

- `Space`: save a detected, sharp target after camera controls settle;
- `f`: save a diagnostic `forced_*.png`, even if target validation fails;
- `a`: toggle automatic capture;
- `r`: reset the live coverage/novelty state;
- `q` or `Esc`: finish and write the summary.

Automatic capture requires a sharp ChArUco detection, settled production
camera metadata, several stable frames, a cooldown, and either an
underrepresented image region or a sufficiently different pose. The overlay's
grid counts which image regions have actually contained detected target
corners.

For a headless SSH session, use `--headless --auto`; inspect
`latest_preview.jpg` from another terminal and stop with Ctrl-C. Use a new
session name after temperature changes or remounting. `--resume` is allowed
only when both configuration-file hashes still match.

The `--video PATH` option rehearses the UI without a camera but is explicitly
marked as a non-production source in the session metadata.

The example dataset glob admits only `session_*/frame_*.png`, so forced
diagnostic captures and `latest_preview.jpg` cannot silently enter calibration.

### 35 — Capture central target poses

Capture several distances and moderate board tilts. Do not capture only a
fronto-parallel board: focal length and distance are then strongly coupled.
Keep the board still during exposure and avoid reflections, clipping and
motion blur.

### 40 — Capture every edge and corner

Move the board center and its detected corners through the complete image,
including partial ChArUco views at the left, right, top and bottom boundaries.
The observations should densely cover all four corners, not merely place a
large board near the center.

More nearly identical frames do not replace missing edge coverage.

### 45 — Inspect raw production images

Before calibration, check representative images at 1:1 scale for:

- saturation or blooming around illuminated features;
- strong vignetting;
- asymmetric blur or defocus;
- motion blur;
- clipped target corners;
- a bowed target;
- resolution or crop changes;
- ISP sharpening or denoising artifacts.

Set `detection.minimum_laplacian_variance` only after observing the sharpness
distribution. It is a dataset-specific rejection threshold, not a universal
quality number.

### 50 — Run a first detection pass

```sh
calibration_pipeline/.venv/bin/python \
  calibration_pipeline/run_calibration.py \
  --config calibration_pipeline/config.production.json \
  --output calibration_pipeline/output/first_pass \
  --detect-only
```

The first run saves `observations.json`. Review `dataset.rejections` in
`detection_results.json` and `observation_coverage.png`. If an edge region is empty,
capture more data rather than fitting more distortion coefficients.

The command exits with status `2` when analysis succeeds but no model/ROI meets
all deployment thresholds. This is an actionable failed acceptance test, not a
pipeline crash.

### 55 — Freeze detections for repeatable model experiments

After detections have been reviewed, set:

```json
"observations_file": "output/first_pass/observations.json"
```

Subsequent runs then use identical subpixel measurements, making model and
threshold comparisons reproducible.

### 60 — Review the held-out split

Inspect `split_assignments.json`. Training and validation must contain
independent sessions and both should cover the sensor. Change `split.seed` or
capture additional sessions if one set lacks edge/corner observations.

Do not manually move difficult images out of validation because they make a
model look bad. Remove an image only for a documented acquisition defect.

### 65 — Run all candidate models

The example compares:

- `opencv5`: radial `k1,k2,k3` plus tangential `p1,p2`;
- `rational8`: adds denominator terms `k4,k5,k6`;
- `thin_prism12`: adds asymmetric thin-prism terms;
- `fisheye4_diagnostic`: an angle-polynomial model.

The current localizer can consume OpenCV 5-, 8- and 12-coefficient vectors.
Its PnP path is not the OpenCV fisheye API, so the example marks fisheye as
`deployment_compatible: false`. A winning fisheye result is evidence that the
runtime camera model should be upgraded; it is not silently exported.

### 70 — Inspect model validity, not only RMS

For every model inspect:

- `validation.residual_px`;
- `validation.spatial_bins`;
- `validation.horizontal_regions`;
- `mapping_validity`;
- `radial_monotonicity`;
- `bootstrap_stability`;
- the residual PNG and residual CSV.

`mapping_validity` samples the configured ROI, maps pixels to rays, reprojects
them, and checks finite round-trip error plus the sign of the local inverse
Jacobian. This detects non-invertible polynomial fits that can appear accurate
near the center.

The selected model is the simplest eligible model within
`selection.simpler_model_tolerance_px` of the best held-out metric. This avoids
buying extra coefficients for an insignificant training-only improvement.

### 75 — Build an independent operational pose dataset

A calibration target can validate pixel geometry, but it cannot by itself
prove flight `z` accuracy because the target pose is estimated from those same
pixels. For a production crop decision, capture marker-grid flights or
controlled static poses with an independent reference such as mocap.

Convert synchronized samples to the schema illustrated by
`operational_validation.example.json`. Each frame needs:

- production detector image points;
- corresponding marker-grid object/world points;
- reference object-to-camera `rvec` and `tvec` derived independently of image
  measurements;
- a `group` identifying an independent flight or trajectory cycle.

The convention is:

```text
X_camera = R(reference_rvec) * X_object + reference_tvec_m
```

If marker-grid object coordinates are world coordinates, mocap provides drone
position `C_wd` and rotation `R_wd`, and the mount provides camera position
`p_dc` and camera-to-drone rotation `R_dc`, compute:

```text
R_wc = R_wd * R_dc
C_wc = C_wd + R_wd * p_dc
R_cw = transpose(R_wc)
t_cw = -R_cw * C_wc
reference_rvec = Rodrigues(R_cw)
reference_tvec_m = t_cw
```

Verify these conventions with a known static pose before processing a flight.

Do not use the localizer's own PnP pose as the reference.

### 80 — Configure candidate crops and confidence bounds

List all trims worth testing under `crop_sweep.horizontal_trims_px`. An integer
means an equal left/right trim. A two-element array means `[left,right]`.
`explicit_rois` can evaluate arbitrary `[x,y,width,height]` rectangles.

Use a reasonably fine step near the expected boundary. Include the full frame,
the current 120-pixel trim and candidates on both sides of it.

The crop analysis re-solves every held-out pose for every candidate. Bootstrap
confidence intervals resample whole `group` blocks, not individual high-rate
frames, so temporal correlation does not create false confidence.
Set `minimum_points_per_pose` to the real solver requirement. The example uses
four because four-marker tiles are part of the production operating envelope.
Set `pose_solver` and `refine_lm` to the production path as well. The supplied
configuration uses `sqpnp` followed by Levenberg–Marquardt refinement, matching
the current high-rate localizer. Supported initializers are `sqpnp`,
`iterative`, `epnp`, `ippe`, and `ap3p`; `ap3p` is evaluated only when exactly
four points remain.

### 85 — Run the final pipeline

```sh
calibration_pipeline/.venv/bin/python \
  calibration_pipeline/run_calibration.py \
  --config calibration_pipeline/config.production.json \
  --output calibration_pipeline/output/final
```

For a fast diagnostic run, temporarily set `bootstrap.iterations` and
`crop_sweep.confidence.iterations` to zero. Restore them for the acceptance
run.

### 90 — Apply the acceptance decision

Read `report.md`, then inspect the complete `results.json`.

The deployment selector considers only runtime-compatible models and chooses:

1. candidates satisfying all configured thresholds;
2. the candidate retaining the largest image area;
3. the lowest held-out residual if areas tie;
4. the simpler model if accuracy also ties.

This implements the scientific rule “largest validated usable field of view,”
not “crop until the plot looks smooth.”

If `maximum_p95_absolute_z_error_mm` is configured but no independent
operational dataset is supplied, every crop fails with
`missing_operational_pose_validation`.

### 95 — Review and deploy the generated patch

If a candidate passes, the pipeline writes
`localizer_calibration_patch.json`. Review it and manually copy its
`calibration` and `processing_crop` objects into
`high_rate_localizer/config/localizer.json`.

Do not copy the `provenance` object into the runtime configuration. Preserve
the final pipeline config, raw images, observations, operational-validation
data and output directory together as the calibration certificate.

After deployment, run the localizer test suite and a short static marker-grid
test before flight.

### 100 — Confirm on a new day and monitor drift

Perform a confirmation flight/session that was not used for calibration,
threshold selection or crop selection. It must meet the same `z`, tail-error
and availability requirements.

Repeat calibration or at least the independent validation after lens remount,
focus change, camera replacement, mechanical impact, resolution change, or a
large temperature shift. Keep one rigid reference target for periodic drift
checks.

## Configuration reference

### `dataset`

- `images`: glob of calibration images. `**` recursion is supported.
- `observations_file`: optional frozen detection file. When it exists, image
  detection is skipped.
- `group_regex`: optional regular expression applied to each filename. The
  first capture group becomes the independent group identifier.

### `capture`

- `coverage_bins`: live horizontal/vertical image-coverage grid.
- `minimum_captures_per_bin`: observations required in every cell before the
  grid is shown as complete.
- `stable_frames` and `maximum_stable_motion_px`: target-stability gate based
  on median displacement of common ChArUco corners.
- `minimum_seconds_between_captures`: prevents burst duplicates.
- `minimum_descriptor_distance`: required change in normalized position,
  approximate pose or scale after covered cells are satisfied.
- `settings_settled_frames`: consecutive frames whose exposure, gain and frame
  duration metadata must agree with production settings.
- `preview_scale`: GUI enlargement factor.
- `headless_preview_interval_s`: update rate for `latest_preview.jpg`.

### `target`

Supported `type` values are `charuco`, `chessboard`, `circles`, and
`asymmetric_circles`. ChArUco uses `squares_x`, `squares_y`,
`square_length_m`, `marker_length_m`, and `dictionary`. Chessboards use the
inner-corner `columns` and `rows`. Circle grids use `columns`, `rows`, and
`spacing_m`.

### `validation.pose_anchor`

`central_radius` solves each held-out target pose using only points inside a
configured normalized radius, then evaluates all target points. This reduces
the ability of a per-image pose fit to absorb edge distortion. If too few
central points are present, `fallback: all` uses the complete observation and
increments `pose_anchor_all_point_fallbacks`.

### `model_analysis`

- `mapping_grid`: horizontal and vertical samples used for invertibility.
- `maximum_round_trip_error_px`: allowed pixel → ray → pixel error.
- `minimum_inverse_jacobian_determinant`: rejects folds and local orientation
  reversals.
- `radial_test_limit`: maximum undistorted radius used by the analytic radial
  monotonicity scan.

### `crop_sweep.thresholds`

- `require_mapping_valid`: requires the ROI's mapping to be finite,
  orientation-preserving and accurately invertible.
- `maximum_validation_p95_px`: upper limit for held-out reprojection residual;
  the upper 95% block-bootstrap bound is used when available.
- `minimum_availability_fraction`: minimum fraction of reference frames that
  retain enough markers and solve successfully.
- `minimum_spatial_bin_coverage_fraction`: minimum fraction of ROI grid cells
  containing enough held-out residual samples. This prevents an unobserved
  image region from passing merely because the fitted polynomial is smooth.
- `maximum_p95_absolute_z_error_mm`: upper limit for operational reference
  error; its upper 95% block-bootstrap bound is used when available.
- `maximum_median_normal_matrix_condition`: optional guard against poorly
  conditioned point layouts. Leave `null` until a task-specific limit has been
  established.

## Output reference

### `observations.json`

Contains every accepted view, full-resolution image coordinates, metric target
coordinates, sharpness, target coverage and group. It is the reproducible
input to model fitting.

### Capture-session outputs

Each session directory contains:

- `frame_*.png`: untouched full-frame luma images accepted for calibration;
- `forced_*.png`: manual diagnostic captures excluded by the example dataset
  glob;
- `session.json`: configuration paths, SHA-256 hashes, exact requested camera
  settings, source type and the intentionally ignored production crop;
- `manifest.jsonl`: one record per saved image with sensor timestamp, actual
  camera metadata, target detection, sharpness, coverage cells and approximate
  pose;
- `capture_summary.json`: duration, processed and saved counts, coverage matrix
  and reasons frames were not automatically captured;
- `latest_preview.jpg`: headless status preview, not part of the PNG dataset.

### `results.json`

Top-level entries:

- `dataset`: detection counts, rejection reasons, image size and split sizes.
- `operational_validation`: whether an independent pose reference was used.
- `models`: complete result for every requested model.
- `model_recommendation`: held-out full-frame model choice.
- `deployment_recommendation`: largest model/ROI pair satisfying all
  thresholds.
- `final_refit`: final coefficients after the chosen model family is refitted
  to every calibration observation, plus repeated mapping and independent-pose
  acceptance checks. The export is suppressed if this refit fails.
- `plots`: generated plot paths.
- `capture_metadata`: the immutable camera/target/acquisition identity copied
  from the configuration.
- `config_snapshot.json`: the complete configuration preserved beside the
  results.

For each model:

- `calibration`: matrix, coefficients, training RMS, per-view RMS, reported
  parameter standard deviations and compatibility.
- `validation`: global, signed, per-view, spatial-bin and left/center/right
  held-out residual statistics.
- `mapping_validity`: sampled numerical invertibility result.
- `radial_monotonicity`: first failure radius of the radial component, when
  applicable.
- `bootstrap_stability`: parameter distribution after resampling independent
  training groups.
- `crop_sweep`: result for every ROI.

Each crop result contains:

- `roi`: `[x,y,width,height]`, using a half-open upper boundary;
- `retained_image_fraction`;
- `mapping_validity` within that ROI;
- `validation_residual_px` and its group-bootstrap p95 interval;
- `calibration_pose_consistency`: crop pose versus the centrally anchored
  board pose; useful diagnostically but not independent ground truth;
- `operational_pose`: crop pose versus the independent reference, when given;
- `passes_thresholds` and exact `threshold_failures`.

Common statistics use population standard deviation and linear NumPy
quantiles. `rms` is `sqrt(mean(x^2))`. Pose `camera_z_error_mm` is computed
after converting object-to-camera pose to camera position in the object frame:

```text
camera_position_object = -R^T * t
```

`tvec_z_sigma_mm_per_px` is a local linearized uncertainty based on the PnP
projection Jacobian assuming one-pixel independent observation noise. It is a
relative conditioning diagnostic, not a replacement for empirical error.

### `residuals_<model>.csv`

One row per held-out target point: view, group, observed pixel, signed residual
and residual magnitude. This is suitable for independent plotting and model
diagnosis.

### `localizer_calibration_patch.json`

Written only when a deployable model/ROI passes every threshold. It contains
the camera matrix, distortion vector and processing crop expected by the
high-rate localizer.

## Important limitations

- Calibration cannot repair blur, saturation, blooming, target non-planarity,
  bad marker coordinates or timestamp error.
- The invertibility test is sampled. Increase `mapping_grid` for final
  validation if a model is close to a failure boundary.
- ChArUco corner residuals do not measure the centroid bias of production LED
  blobs. Independent operational validation is required for a production crop.
- Circle-grid calibration requires care: the observed ellipse center can
  differ from the projected physical circle center at oblique views.
- A calibration is valid only for the image geometry and focus state in which
  it was captured.
