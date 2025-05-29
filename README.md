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

| Argument              | Alias | Type    | Description                                                      | Default Value      |
|-----------------------|-------|---------|------------------------------------------------------------------|--------------------|
| `--verbose`           | `-v`  | Flag    | Enables verbose logging                                          | false              |
| `--preview`           | `-p`  | Flag    | Enables preview mode, requires a display                         | false              |
| `--time`              | `-t`  | Int     | Sets execution time in seconds (must be positive)                | 0                  |
| `--save-frames`       | `-s`  | Flag    | Enables saving video frames                                      | false              |
| `--config`            | —     | String  | Path to configuration file                                       | camera_config.json |
| `--save-rate`         | —     | Int     | Save the frames that are multiples of this value                 | 1                  |
| `--contrast`          | —     | Double  | Image contrast adjustment                                        | camera default     |
| `--brightness`        | —     | Double  | Image brightness adjustment                                      | camera default     |
| `--exposure`          | —     | Int     | Exposure time                                                    | camera default     |
| `--fps`               | —     | Int     | Frame rate in frames per second                                  | 120                |
| `--stream`            | —     | Flag    | Enables video streaming                                          | false              |
| `--stream-port`       | —     | Int     | Port for video streaming                                         | 8080               |
| `--stream-type`       | —     | String  | Streaming protocol type (`http` or `udp`)                        | http               |


## Visualize Logs
Use https://github.com/Hamedamz/cam-pose-vis repository to visualize the logs.
