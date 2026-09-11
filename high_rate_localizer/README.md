# High-rate marker localizer

This is a standalone C++20 localizer for a downward-facing camera. It starts
on a four-marker MyGrid landing tile, decodes its file-defined blink packet,
publishes an initial FLU pose and yaw for the EKF reset, then tracks the
unlabelled static HyperGrid. During landing it switches back to the
controller-specified MyGrid tile without decoding its IDs again.

The production path uses libcamera. There is no preview, streaming, ArUco,
networking, or internal position filter.

## Build

On the Linux deployment machine, install libcamera, OpenCV 4, and
nlohmann-json development packages, then run:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

If libcamera is unavailable, CMake still builds the video runner and unit
tests. To explicitly request that configuration:

```sh
cmake -S . -B build -DFLS_LOCALIZER_ENABLE_LIBCAMERA=OFF
```

## Run

Production has one option:

```sh
./build/fls_localizer --config config/localizer.json
```

The MP4 fixture runner accepts a grid override and output directory:

```sh
./build/fls_localizer_video \
  --config config/video_test.json \
  --video render_lb_normal.mp4 \
  --trajectory render_lb_normal_trajectory.json \
  --grid config/hypergrid-mygrid-normal.json \
  --output-dir logs/normal
```

It can also perturb the exact Blender attitude before passing it to the
localizer as simulated EKF data:

```sh
./build/fls_localizer_video \
  --config config/video_test.json \
  --video render_lb_normal.mp4 \
  --trajectory render_lb_normal_trajectory.json \
  --orientation-bias-deg 1.0,-0.5,2.0 \
  --orientation-noise-stddev-deg 0.3,0.3,1.0 \
  --orientation-noise-seed 42
```

Bias and per-frame independent Gaussian-noise standard deviations use
roll,pitch,yaw order. A single value applies to all three axes. Errors are
composed onto the drone-to-world attitude in the drone/body frame. The seed
defaults to `0`, so repeated runs are reproducible; changing it generates
another noise realization. Omit the arguments (or pass zero values) to retain
the exact trajectory attitude.

`localizer.json` contains the calibrated physical-camera distortion.
`video_test.json` intentionally has zero distortion because the Blender
fixtures use an ideal pinhole camera.

Each session writes `log.json` and a 30 fps annotated `video.mp4`. Vision and
shared-memory publication continue at the camera rate. JSON serialization and
video encoding run on a background thread; video frames are dropped first if
debug output falls behind.

The production runner also accepts `--tag TAG`. When present, it writes
`log_TAG.json` and `video_TAG.mp4`, matching the filenames collected by the
LightBender orchestrator.

The JSON retains the `args`, `config`, and `frames` structure and can be opened
directly in the repository's web log viewer. Each successful tracking frame
contains separate `poses` entries with `pose_technique` set to `pnp` and
`shared_attitude`. Each entry logs these explicitly framed quantities:

- `marker_position_camera_m` and `marker_orientation_camera_rpy_rad`;
- `camera_position_world_flu_m` and
  `camera_orientation_world_flu_rpy_rad`;
- `drone_position_world_flu_m` and
  `drone_orientation_world_flu_rpy_rad`.

The PnP entry estimates rotation and translation from image/world
correspondences alone. The shared-attitude entry uses the controller's
drone-to-world quaternion for rotation and solves translation from the same
correspondences. Both use the configured camera mount to derive the drone pose.

Set which accepted tracking pose is published to the controller in the JSON
configuration:

```json
"shared_memory_pose_technique": "shared_attitude"
```

The allowed values are `shared_attitude` and `pnp`. The initial bootstrap pose
is necessarily published from PnP because the shared attitude becomes valid
only after the controller resets and acknowledges its EKF.

## Ground-truth trajectory

In Blender, select exactly one LightBender and use **Marker Grid → Camera
Rendering → Export Selected Trajectory**. The add-on exports one JSON sample
for every integer frame from `frame_start` through `frame_end`, using the
evaluated global transform. Positions are world FLU metres and quaternions are
drone-to-world in `x,y,z,w` order.

The fixture runner requires the matching trajectory. It checks frame rate and
frame count before processing, then supplies each exact or configured-perturbed
trajectory quaternion as the simulated shared-memory EKF attitude. For every
valid localized pose it records the 3D position RMSE

```text
frame_rmse = sqrt(dx^2 + dy^2 + dz^2)
cumulative_rmse = sqrt(sum(dx^2 + dy^2 + dz^2) / valid_pose_count)
```

Both values are drawn on the annotated video and stored in each frame's
`ground_truth` JSON object. Frames without a localization result use `null` for
the per-frame value; cumulative RMSE continues over previously valid poses.
The log metadata records the bias, noise standard deviations, and seed, while
each ground-truth entry records both the exact Blender quaternion and the
`simulated_ekf_quaternion_xyzw` supplied to the localizer.

## State handshake

| State | Meaning | Requested MyGrid mode |
| --- | --- | --- |
| `mygrid_decoding` | Collect and match the cyclic four-ID signature | Blink |
| `initial_pose_ready` | Position/yaw published; waiting for matching EKF generation | Static |
| `takeoff_tracking` | Shared EKF quaternion is authoritative | Static |
| `hypergrid_acquire` | HyperGrid pose is being confirmed | Static |
| `hypergrid_tracking` | Static lattice provides position | Off |
| `landing_acquire` | Known landing tile requested, HyperGrid remains fallback | Static |
| `landing_tracking` | Known MyGrid tile provides position | Static |
| `lost` | No valid measurement within the configured loss window | Unchanged |
| `fault` | Unrecoverable configuration or runtime error | Unchanged |

An OFF or STATIC request is advisory. HyperGrid correspondences are selected
only from predicted lattice nodes, and all known MyGrid locations are excluded.
Consequently an always-on MyGrid cannot enter the HyperGrid IPPE point set.

Along with the initial pose, shared memory contains a conservative HyperGrid
acquisition height computed from the calibrated focal lengths, sensor size,
marker spacing, and a 10% margin. It guarantees enough field of view for a 2x2
lattice under arbitrary lattice phase. The controller can climb to this height,
hold until `hypergrid_tracking`, and then continue takeoff.

## High-rate behavior

- Libcamera uses a four-buffer YUV420 stream and processes only its luma plane.
- Completed requests are latest-only; stale frames are returned immediately.
- Connected-component storage is reused between frames.
- Detection is hard-capped at 64 blobs.
- IPPE input is spatially selected and hard-capped at 16 points.
- Full-frame undistortion is avoided.
- IPPE produces the PnP-only pose. A second translation is solved against the
  shared EKF attitude, using the same matched marker correspondences.
- Absolute HyperGrid indices use the most recent anchored pose plus a bounded
  constant-velocity prediction. A cold start on an unlabelled lattice is never
  treated as an absolute position.

The bundled 640x400 fixtures process well within the 8.33 ms budget on the
development machine. Always benchmark the release build again on the target
computer.

## Marker-grid files

The combined grid files are self-contained. Their `encoding` object defines
payload width, delimiter, bit ordering, and bit duration. Their `mygrid.tiles`
objects define cyclic signatures and marker positions. HyperGrid markers are
not enumerated; their position is computed from `grid_origin` and
`hypergrid.marker_spacing`:

```text
P(i,j) = origin + ((i + 0.5) spacing, (j + 0.5) spacing, 0)
```

The loader rejects invalid geometry, inconsistent marker IDs, symmetric rings,
and duplicate cyclic ring signatures before capture starts.

See [SHARED_MEMORY.md](SHARED_MEMORY.md) for the controller ABI.
