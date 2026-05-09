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

## Visualize Logs

Use https://github.com/Hamedamz/cam-pose-vis repository to visualize the logs.
