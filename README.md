# FLS Marker Localization

Quantifies the 3d position and orientation of a 3D marker consisting of 4 points. Made for Raspberry Pi and Raspberry Camera.
This software is a part of the FLS prototype software stack: https://github.com/flyinglightspeck/FLS.

## Install Dependencies

```
sudo apt install libopencv-dev libeigen3-dev libcamera-dev nlohmann-json3-dev
```

## Make

```
mkdir build
cd build
cmake ..
make
```

Copy config to the build directory:

```
cp ../src/gs_camera_config.json .
```

The camera config file includes distortion coefficients, camera matrix, and marker configuration. To calibrate a camera and compute distortion coefficients and camera matrix, see https://github.com/flyinglightspeck/aruco-pose-estimation.

## Usage

Run for 10 seconds:

```
./eye -v -t 10
```

| Argument                  | Alias | Type   | Description                                           | Default Value      |
| ------------------------- | ----- | ------ | ----------------------------------------------------- | ------------------ |
| `--verbose`               | `-v`  | Flag   | Enables verbose logging                               | false              |
| `--preview`               | `-p`  | Flag   | Enables preview mode, requires a display              | false              |
| `--distance`              | `-d`  | Double | Initial perpendicular camera-to-grid distance (m)     | -1.0               |
| `--time`                  | `-t`  | Int    | Sets execution time in seconds (0 means no end time). | 0                  |
| `--save-frames`           | —     | Flag   | Enables saving frames as individual images            | false              |
| `--save-rate`             | —     | Double | Save frames per second                                | 1                  |
| `--save-video`            | `-s`  | Flag   | Enables saving video                                  | false              |
| `--video-fps`             | —     | Int    | Sets video frames per second                          | 30                 |
| `--video-path`            | —     | String | Path to save video                                    | empty              |
| `--json-path`             | —     | String | Path to save JSON log                                 | empty              |
| `--config`                | —     | String | Path to configuration file                            | camera_config.json |
| `--contrast`              | —     | Double | Image contrast adjustment                             | camera default     |
| `--brightness`            | —     | Double | Image brightness adjustment                           | camera default     |
| `--dark-blob-intensity`   | —     | Double | Normalized marker intensity for logic-0 states        | 0                  |
| `--exposure`              | —     | Int    | Exposure time                                         | camera default     |
| `--fps`                   | —     | Int    | Frame rate in frames per second                       | 120                |
| `--stream`                | —     | Flag   | Enables video streaming                               | false              |
| `--stream-port`           | —     | Int    | Port for video streaming                              | 8080               |
| `--stream-type`           | —     | String | Streaming protocol type (`http` or `udp`)             | http               |
| `--stream-rate`           | —     | Double | Target frames per second for streaming                | 10                 |
| `--aruco`                 | —     | Flag   | Enables ArUco marker detection mode                   | false              |
| `--video-input`           | —     | String | Path to a video file to use instead of live camera    | empty              |
| `--grid-map`              | —     | String | Blinking-marker grid JSON path                        | empty              |
| `--window-size`           | `-w`  | Int    | Side length of the unique grid lookup window          | 2                  |
| `--grid-center-ap3p`      | —     | Flag   | Use the frame-centred 2 x 2 grid window with AP3P     | false              |
| `--grid-rounding-tolerance` | —   | Double | Maximum grid-cell rounding residual                   | 0.30               |
| `--grid-max-marker-age`   | —     | Double | Maximum age of a decoded image point in seconds       | 0 (current frame)  |
| `--max-attitude-age`      | —     | Double | Maximum camera/quaternion timestamp difference        | 0.1 s              |
| `--camera-offset`         | —     | 3 doubles | Camera centre in drone FLU coordinates (m)          | 0 0 0              |

## Blinking Marker Grid Localization

Pass `--grid-map` in blob mode to turn decoded blinking markers into a global
camera pose. The pipeline undistorts image points, rectifies them onto the
marker plane, converts them to relative integer grid cells, looks up a complete
oriented `w x w` ID window, and solves PnP from the resulting image/world
correspondences.

Grid maps use a right-handed object/world frame. The JSON array is row-major:
increasing the row moves along world `-X`, increasing the column moves along
world `-Y`, and every marker lies at the same world `Z`; `+Z` is the marker-plane
normal. `grid_origin` is the world coordinate of cell `(0, 0)`, not the grid
centre. Maps generated with the coordinator's default centred layout set that
cell coordinate to `((rows - 1) * spacing / 2, (cols - 1) * spacing / 2, 0)`,
placing the geometric grid centre at world `(0, 0, 0)`.

OpenCV camera coordinates use `+X` right, `+Y` down, and `+Z` forward.
`solvePnP` returns `rvec` and `tvec` for the object/world-to-camera transform,
`X_camera = R(rvec) * X_world + tvec`. Consequently, `tvec` is the world-frame
origin expressed in camera coordinates. For a default centred map, that world
origin is also the grid centre.

For the 20 x 20 example map and its default 2 x 2 lookup window:

```
./eye \
  --grid-map ../../lightbender/orchestrator/marker_grid_20x20.json \
  --window-size 2 \
  --distance 1.5
```

`--window-size 2` requires all four cells of a contiguous 2 x 2 observation.
The configured map must have a unique row-major signature for every window at
the chosen size. Repeated marker IDs are supported; identity is preserved by
relative cell and image coordinate rather than by ID alone.

Add `--grid-center-ap3p` to use legacy-style pose solving. This mode selects
the complete visible 2 x 2 window whose four-point image centroid is closest
to the frame centre, then passes exactly those four image/world correspondence
pairs to `SOLVEPNP_AP3P`. Without the flag, the default IPPE pipeline uses all
matched grid markers and performs iterative refinement.

The camera matrix and distortion coefficients in `--config` are used for ray
rectification and must match the processed frame resolution. Grid localization
always reads the Crazyflie attitude from `/pos_shared_mem`; there is no static
RPY or pixel-scale fallback. The Python controller writes the quaternion in
SciPy order `[qx, qy, qz, qw]`. If `R_w_d(q)` maps drone vectors to world vectors
and

```
R_d_c = [ 0 -1  0
         -1  0  0
          0  0 -1 ],
```

then every frame uses `R_c_g = R_d_c^T * R_w_d(q)^T` both to intersect camera
rays with the grid plane and to select the physically consistent OpenCV PnP
solution. Missing, torn, invalid, or stale attitude is rejected for that frame;
grid mode never silently substitutes another orientation source. A nearly
edge-on plane is rejected because grid normalization is unstable there.
The shared slot supplies the latest host-timestamped quaternion: its timestamp
is compared with the camera capture timestamp, but samples are not buffered or
interpolated to the exact exposure instant.

`--distance` supplies only the initial perpendicular distance from the camera
centre to the marker plane. After every successful PnP solve, the next frame
uses `abs(camera_world_z - marker_plane_z)`, where `camera_world` is recovered
with the shared attitude. The last successful value is retained across failed
frames. Raw `tvec.z` is not used because it is optical-axis depth, not
perpendicular plane distance when the camera is tilted.

The first map lock still requires a complete decoder-confirmed window. After
that lock, spatial ID assignment runs independently of the blink decoder:
visible tracks that fit the rectified lattice, occupy an in-bounds unclaimed
map cell, and connect to the visible assigned frontier receive that cell's map
ID immediately. Off-lattice, duplicate-cell, out-of-bounds, and decoder/map
conflicts are excluded from localization. Assigned IDs are not written back to
the decoder. If no assigned track remains visible, inference pauses until
continuity or a newly decoded unique window restores the alignment.
When dim-state tracking is enabled, grid mode immediately retires a missed
track last seen near an image edge, so an entering marker cannot inherit a
departed marker's decoder or map identity. Otherwise it waits only through the
longest valid dark packet interval before retiring the track.

By default, decoded fallback observations must be visible in the current frame,
so image coordinates from different camera poses are never mixed while LEDs
blink off. If the camera is effectively stationary and the markers are not
synchronized, `--grid-max-marker-age <seconds>` can admit recently observed
decoded tracks instead.

`--dark-blob-intensity <value>` enables continuous tracking through logic-0
bits when the marker remains dimly visible. Values are normalized to `[0, 1]`
and must be below the bright-state threshold (`0.8`). The tracker detects
marker visibility at half the configured dark intensity while continuing to
classify pixels above `0.8` as logic 1. The default `0` preserves the original
on/off behavior.

Each frame log includes `blob_grid_localization`, with every decoded track's
freshness/eligibility, map-lock and spatial-assignment counts, normalization
residuals, relative cells, whether lookup ran, matched map cells/global points,
and PnP reprojection error. It also records the distance used for that frame
and the new camera-to-plane distance when solved. A successful `poses` record
contains the raw PnP `marker_position` (`tvec`) plus attitude-derived
world-frame `camera_position`, `camera_orientation`, `drone_position`, and
`drone_orientation`.

`--camera-offset X Y Z` is the camera centre relative to the drone origin,
expressed in drone FLU coordinates. The camera and drone positions are

```
p_w_c = -R_w_d R_d_c t_pnp
p_w_d = p_w_c - R_w_d p_d_c
```

The legacy 40-byte pose prefix in `/pos_shared_mem` publishes `p_w_d` and drone
RPY by default. With the zero offset default, camera and drone positions are
identical; pass the calibrated offset for flight hardware.

## ArUco Marker Detection Mode

When `--aruco` is passed, the system uses OpenCV's ArUco marker detector instead of the LED blob tracker. Given known world poses of ArUco markers (defined in the config file), it computes the camera's position and orientation in world coordinates via PnP.

### Config File Format

Add an `aruco_markers` section to your camera config JSON:

```json
{
  "camera_matrix": [...],
  "dist_coeffs": [...],
  "aruco_markers": {
    "dictionary": "DICT_4X4_50",
    "marker_size": 0.20,
    "markers": {
      "0": {
        "position": [0.0, 0.0, 0.0],
        "rotation_deg": [0, 0, 0]
      },
      "1": {
        "position": [0.5, 0.0, 0.0],
        "rotation_deg": [0, 0, 0]
      }
    }
  }
}
```

- **dictionary**: ArUco dictionary name (e.g. `DICT_4X4_50`, `DICT_6X6_250`)
- **marker_size**: Physical side length of markers in meters
- **markers**: Map of marker ID → world pose (position in meters, rotation as roll/pitch/yaw in degrees)

### Example

```
./eye --aruco -v -t 10
```

When multiple markers are visible, their camera pose estimates are fused via weighted average (weighted by inverse reprojection error).

## Video Input Mode

The software can process a pre-recorded video file instead of live camera input. This mode does **not** require libcamera or a physical camera, making it possible to run on macOS or any system with OpenCV.

### Build (Video Input Mode)

Build with the `VIDEO_INPUT_MODE` CMake option enabled:

```
mkdir build_video
cd build_video
cmake .. -DVIDEO_INPUT_MODE=ON
make
```

Or use the convenience script:

```
./build_video_mode.sh
```

### Usage

```
./eye --video-input path/to/video.mp4 --aruco -v -p --config camera_config.json
```

- The video plays back at its native FPS. Use `--fps` to override playback speed.
- The program exits when the video ends (or when `--time` expires, whichever comes first).
- All other features work identically: ArUco/blob detection, Kalman filtering, JSON logging, frame/video saving, and streaming.

## Visualize Logs

The repository includes **FLS Pose Scope**, a local-first web viewer for the
generated `log.json` files. It shows synchronized plots, image-space detections,
the camera and marker geometry in 3D, every frame diagnostic, and the complete
raw log.

```sh
cd web
npm install
npm run dev
```

Open the printed URL and drag a log into the viewer (or use **Open log**). The
file is parsed entirely in the browser and is not uploaded.

## Blink Marker

```
g++ -O2 blinker.cpp -o blinker -llgpio
```

```
./bin/blinker --fps <fps>
```
