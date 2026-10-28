# FLS Pose Scope

A local-first web viewer for `eye` output logs and swarm flight plans. Log mode
synchronizes a 3D scene, image-plane overlays, telemetry plots, frame tables,
and complete JSON inspection. Flight mode animates an SFL mission over the
marker floor, accumulates each down-facing camera footprint, and finds the
smallest set of unique marker windows needed for the certified route and one
distinct launch/landing home per drone.

## Marker grid generator

The **Marker grid** tab authors the `fls-marker-grid` JSON format entirely in
the browser. It describes an infinite, always-on HyperGrid by marker spacing
instead of materializing a finite ID matrix. Its tile size is always twice the
HyperGrid spacing, and `grid_origin` is the world-FLU position of the centre of
tile `(0, 0)`.

Each tile owns four HyperGrid markers in a centred 2×2 layout. Click tiles in
the pan-and-zoom canvas to install a second, tighter 2×2 group of four MyGrid
markers at its centre. MyGrid IDs form an asymmetric clockwise cyclic signature
that is unique under rotation (and reflection) across the exported map. Click
any marker to inspect its logical grid and global XYZ coordinates. Optional
per-tile event lists control on, off, static, and blinking-ID mode transitions at
arbitrary times in a simulated scene. MyGrids start off in blinking mode, so an
`on` event creates the ID blink pattern unless a prior event selected static
mode; a tile without events stays off. Encoding and Blender settings are embedded in the download;
the payload width is `ceil(log2(max_ids))` and the all-ones delimiter is one bit
longer than the payload, followed by `0`.

## Run locally

```sh
npm install
npm run dev
```

Open the printed local URL, then drag in a log `.json` or SFL `.yaml`/`.yml`
file (or use **Open log** / **Open SFL**). A synthetic blob-grid run and the
built-in `la_base.yaml` mission are available before a file is selected. Files
stay in the browser.

## Flight coverage

Open the **Flight** tab to play the built-in mission, or load another SFL YAML.
The bundled `marker_grid_20x20_h.json` geometry is used by default; **Grid
JSON** loads a different complete localization-grid file.

SFL does not define launch XY positions, so the viewer initially assigns each
drone so its camera starts above a distinct short-range-tile centre, or a
non-overlapping main-window centre when no short-range tiles are present. The
computed body XY and home-window assignment are listed in the flight inspector.
Each route rises vertically to `target[2]`, travels to `target[0:2]` at that
height, then follows
`[x, y, z, yaw, dt]` waypoints using the controller's relative-waypoint and
timing semantics. SFL yaw values use radians, matching the controller and
Blender exporter. Because SFL has no initial-yaw field, takeoff assumes the
controller default of zero yaw; takeoff and target transit use its
seventh-degree easing, with yaw easing to the target during transit. After the
waypoints, each drone follows the controller's horizontal return to its launch
XY and then descends there, so its computed takeoff spot is also its landing
spot.

The camera centre defaults to the Lightbender orchestrator's current
`[0.04, 0, 0]`-metre launch setting in drone FLU coordinates. The viewer
rotates that offset by flight yaw before projecting the footprint. All three
values are editable for another airframe or calibration.

Coverage uses the grid's physical pinhole-camera fields and working range. A
continuous swept-footprint pass records every crossed grid cell, including
cells crossed between display frames. A main-grid requirement is supported
only when a complete, uniquely decodable window is visible. Adaptive display
samples are supplemented by continuous visibility decomposition with a
nanometre-scale numerical boundary tolerance, and the reported minimum is an
exact constrained set cover: every drone's distinct home window is preselected,
and only the minimum additional windows needed by the in-range outbound,
waypoint, and return legs are added. The route-support percentage is
duration-weighted across all drones. If the route contains a true gap or leaves
the working range, the UI labels the result infeasible and reports only a
partial-route minimum. Vertical launch and landing are excluded while the
controller uses the assigned short-range tile; below-range horizontal flight is
unsupported.

## Supported records

- Current blob-grid camera poses and complete localization diagnostics.
- Main/short-range grid phases, landing tile coordinates, and both marker sets
  in the world and image-plane views.
- Current ArUco camera poses and marker world poses.
- Legacy marker-in-camera / camera-in-marker pose pairs.
- Historical `tvec`, `tvec_filtered`, and `yaw_pitch_roll` logs.
- Relative seconds, live epoch seconds, and historical epoch milliseconds.

Unknown and future fields remain available through **Raw JSON**, while all
known run metadata and selected-frame arrays are rendered as tables.

## Checks

```sh
npm test
npm run build
```

## Flight region in 3D

The **Flight region** tab defaults to a continuous analytic envelope for a level
camera guaranteed across every yaw. A sampled roll/pitch model is also available
for a finite marker installation. Load a localization grid JSON, or use **Visualize
flight region** in the marker generator. For `fls-marker-grid` files, selected
MyGrid tiles define the installed HyperGrid tile set; the infinite flag does
not add markers beyond those tiles in this view. A finite 4×4 example is built
in. MyGrid landing dots are excluded from flight support.

Set the maximum absolute roll and pitch (each independently), fixed yaw,
drone heights above the marker plane, configured processing ROI, dot area threshold, and
horizontal search border. The calibrated camera defaults come from
`high_rate_localizer/config/localizer.json`; another localizer config can be
loaded to change intrinsics, distortion, image size and camera mount.

The interactive 3D view shows level-flight support in blue and positions
supported at every sampled combined attitude in green. Height slices isolate
a cross section. Volume is a cell-center estimate over the displayed search
bounds, with 16 height layers and selectable XY/attitude resolutions. All
calculation runs locally in a worker. The support test projects a 16-point dot
contour, requires sufficient projected area, and checks the nearest four
visible markers for separation and non-collinearity. Supported cells indicate
geometric feasibility; intermediate attitudes, occlusion, exposure, image
thresholding, association and solver quality are not certified by this model.

### Level camera, every yaw

The default model assumes a downward level camera and pinhole projection.
It retains the camera intrinsics, processing ROI, principal point and mount
translation; lens distortion is excluded. Intersecting all yaw rotations of
the eroded footprint gives a disk centered on the drone origin, whose radius
is the minimum distance to the four footprint edges minus marker radius.
For each XY position, the smallest farthest-corner distance among complete
installed 2×2 blocks determines a continuous minimum-height boundary. The
upper height follows the projected dot-area limit.

This guarantees visibility of the same complete block through every yaw;
allowing different blocks at different yaw angles can yield a larger region.
The view shows a closed triangular surface, with continuous filled height
slices. Surface detail controls interpolation of the analytic boundary;
volume is computed from that mesh and remains approximate. The calculation
uses model-derived horizontal bounds and the selected height band. Exact
formulae and the derivation are in `doc/main.tex`.

### Fixed yaw and height slicing

Choose **Level camera · fixed yaw** to use an editable drone yaw (default 0°).
This analytic mode intersects each block's image-edge constraints to form its
continuous region. At a fixed height, the supported area is a union of rotated
rectangles; every-yaw slices instead intersect four disks per block. Slices
are computed directly from these boundaries, independently of the lower
surface mesh, to preserve narrow gaps and avoid interpolation bridges.

A height slider moves a cyan slicing plane and the filled 2D support area.
The full 3D region remains visible at reduced opacity. Toggle **Slice plane**
off to restore its normal opacity; moving the slider enables it again.
Orbit and zoom are retained while adjusting slice height.

The percentage margin control has been removed. All support modes use
`processing_crop` from the loaded camera configuration when enabled, and the
full image when disabled; the active ROI is shown beside the camera inputs.
For analytic modes, the horizontal mesh domain is derived from the model's
support bounds at the effective ceiling. The slicing plane is sized from the
selected maximum height (default 3 m), and the optional extra border adds
space around those model bounds. Marker-area limits can make the supported
ceiling lower than the selected height; the result summary reports this.

### Camera and tile overlays

Under **Scene overlays**, toggle the camera's frustum and ground footprint,
and the installed tile cuboids independently. Camera X/Y are world coordinates;
height is its optical-center distance above the marker plane. Roll/pitch/yaw
rotate the camera from the configured mount orientation, in degrees. Editing
this preview pose leaves the calculated support model intact. The orange
pinhole frustum uses the active processing ROI and is clipped at the floor;
if the ROI crosses the horizon, no bounded ground polygon is drawn.

Purple cuboids show each installed tile footprint extruded to the chosen
maximum drone height. Missing tiles stay empty. These are geometric tile
volumes, independent of the optical support and marker-area ceiling.
