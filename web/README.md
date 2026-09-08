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
