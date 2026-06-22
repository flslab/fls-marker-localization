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
| `--distance`              | `-d`  | Double | Sets distance                                         | -1.0               |
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
| `--exposure`              | —     | Int    | Exposure time                                         | camera default     |
| `--fps`                   | —     | Int    | Frame rate in frames per second                       | 120                |
| `--stream`                | —     | Flag   | Enables video streaming                               | false              |
| `--stream-port`           | —     | Int    | Port for video streaming                              | 8080               |
| `--stream-type`           | —     | String | Streaming protocol type (`http` or `udp`)             | http               |
| `--stream-rate`           | —     | Double | Target frames per second for streaming                | 10                 |
| `--aruco`                 | —     | Flag   | Enables ArUco marker detection mode                   | false              |

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

## Visualize Logs

Use https://github.com/Hamedamz/cam-pose-vis repository to visualize the logs.

## Blink Marker

```
g++ -O2 blinker.cpp -o blinker -llgpio
```

```
./bin/blinker --fps <fps>
```
