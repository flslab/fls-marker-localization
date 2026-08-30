#!/usr/bin/env python3
"""Run the Blender 2x2-grid video with a shared-memory test attitude."""

import argparse
import ctypes
import ctypes.util
import errno
import json
import math
import mmap
import os
import signal
import statistics
import struct
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent
LIGHTBENDER = ROOT.parent / "lightbender"
SHM_NAME = b"/pos_shared_mem"
SHARED_SIZE = 80
POSE_TIMESTAMP_OFFSET = 32
ATTITUDE_BEGIN_OFFSET = 40
ATTITUDE_PAYLOAD_OFFSET = 44
ATTITUDE_CHECKSUM_OFFSET = 76
FNV1A_OFFSET_BASIS = 2166136261
FNV1A_PRIME = 16777619
IDENTITY_QUATERNION_XYZW = (0.0, 0.0, 0.0, 1.0)
STARTUP_TIMEOUT_SECONDS = 10.0
RUN_TIMEOUT_SECONDS = 30.0
EXPECTED_FRAME_COUNT = 600
MIN_SUCCESSFUL_FRAMES = 200
EXPECTED_MARKER_IDS = {1, 7, 10, 12}
EXPECTED_MARKER_CELLS = {
    (1, 0, 0), (12, 0, 1), (10, 1, 0), (7, 1, 1)
}
EXPECTED_GLOBAL_POSITIONS = {
    1: (0.05, 0.05, 0.0),
    12: (0.05, -0.05, 0.0),
    10: (-0.05, 0.05, 0.0),
    7: (-0.05, -0.05, 0.0),
}
MAX_REPROJECTION_ERROR_PIXELS = 0.5


def posix_shm_library():
    candidates = [None]
    realtime_library = ctypes.util.find_library("rt")
    if realtime_library:
        candidates.append(realtime_library)
    for candidate in candidates:
        try:
            library = ctypes.CDLL(candidate, use_errno=True)
        except OSError:
            continue
        if hasattr(library, "shm_open") and hasattr(library, "shm_unlink"):
            # Darwin declares shm_open variadically. Keep only its fixed
            # arguments here and pass the creation mode as an explicit c_int.
            library.shm_open.argtypes = [ctypes.c_char_p, ctypes.c_int]
            library.shm_open.restype = ctypes.c_int
            library.shm_unlink.argtypes = [ctypes.c_char_p]
            library.shm_unlink.restype = ctypes.c_int
            return library
    raise RuntimeError("POSIX shared memory is unavailable")


def unlink_shared_memory(library):
    if library.shm_unlink(SHM_NAME) == 0:
        return
    error = ctypes.get_errno()
    if error != errno.ENOENT:
        raise OSError(error, os.strerror(error), SHM_NAME.decode())


def reserve_shared_memory(library, force):
    flags = os.O_CREAT | os.O_EXCL | os.O_RDWR
    descriptor = library.shm_open(SHM_NAME, flags, ctypes.c_int(0o600))
    if descriptor >= 0:
        os.close(descriptor)
        return
    error = ctypes.get_errno()
    if error != errno.EEXIST:
        raise OSError(error, os.strerror(error), SHM_NAME.decode())
    if not force:
        raise RuntimeError(
            "/pos_shared_mem already exists; stop the live eye/controller "
            "first, or use --force-shm only if the object is stale"
        )

    unlink_shared_memory(library)
    descriptor = library.shm_open(SHM_NAME, flags, ctypes.c_int(0o600))
    if descriptor >= 0:
        os.close(descriptor)
        return
    error = ctypes.get_errno()
    if error == errno.EEXIST:
        raise RuntimeError(
            "/pos_shared_mem was claimed by another process during startup"
        )
    raise OSError(error, os.strerror(error), SHM_NAME.decode())


def attach_shared_memory(library, process, timeout=STARTUP_TIMEOUT_SECONDS):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        descriptor = library.shm_open(SHM_NAME, os.O_RDWR, ctypes.c_int(0))
        if descriptor >= 0:
            try:
                if os.fstat(descriptor).st_size >= SHARED_SIZE:
                    return mmap.mmap(descriptor, SHARED_SIZE,
                                     access=mmap.ACCESS_WRITE)
            finally:
                os.close(descriptor)
        elif ctypes.get_errno() != errno.ENOENT:
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error), SHM_NAME.decode())

        return_code = process.poll()
        if return_code is not None:
            raise RuntimeError(
                f"eye exited before creating shared memory ({return_code})"
            )
        time.sleep(0.005)
    raise TimeoutError("timed out waiting for /pos_shared_mem")


def fnv1a_32(data):
    checksum = FNV1A_OFFSET_BASIS
    for byte in data:
        checksum = ((checksum ^ byte) * FNV1A_PRIME) & 0xFFFFFFFF
    return checksum


def write_attitude(shared, timestamp, quaternion, valid=True):
    sequence = struct.unpack_from(
        "<I", shared, ATTITUDE_BEGIN_OFFSET
    )[0]
    next_even = ((sequence & ~1) + 2) & 0xFFFFFFFF
    if next_even == 0:
        next_even = 2

    struct.pack_into("<I", shared, ATTITUDE_BEGIN_OFFSET, next_even - 1)
    payload = struct.pack(
        "<Id4fI", int(valid), timestamp, *quaternion, next_even
    )
    shared[ATTITUDE_PAYLOAD_OFFSET:ATTITUDE_CHECKSUM_OFFSET] = payload
    struct.pack_into(
        "<I", shared, ATTITUDE_CHECKSUM_OFFSET, fnv1a_32(payload)
    )
    struct.pack_into("<I", shared, ATTITUDE_BEGIN_OFFSET, next_even)


def stop_process(process):
    if process.poll() is not None:
        return
    process.send_signal(signal.SIGINT)
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def finite_number(value):
    return (isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value))


def verify_log(path, expected_distance):
    log = json.loads(path.read_text())
    frames = log.get("frames", [])
    if len(frames) != EXPECTED_FRAME_COUNT:
        raise RuntimeError(
            f"expected {EXPECTED_FRAME_COUNT} frames, got {len(frames)}"
        )

    localizations = [frame.get("blob_grid_localization", {})
                     for frame in frames]
    statuses = Counter(item.get("status", "missing")
                       for item in localizations)
    if statuses.get("normalization_failed", 0):
        raise RuntimeError(f"attitude normalization failed: {statuses}")

    saw_valid_attitude = False
    for index, localization in enumerate(localizations):
        attitude = localization.get("attitude", {})
        quaternion = attitude.get("quaternion_xyzw")
        age = attitude.get("age_seconds")
        sequence = attitude.get("sequence")
        if (not saw_valid_attitude
                and localization.get("status") == "no_detections"
                and attitude.get("status") == "absent"):
            continue
        if (attitude.get("source") != "shared_memory"
                or attitude.get("status") != "valid"
                or attitude.get("valid") is not True):
            raise RuntimeError(
                f"frame {index} did not use a valid shared-memory attitude"
            )
        saw_valid_attitude = True
        if (not isinstance(quaternion, list) or len(quaternion) != 4
                or not all(finite_number(value) for value in quaternion)
                or any(abs(actual - expected) > 1e-6
                       for actual, expected in zip(
                           quaternion, IDENTITY_QUATERNION_XYZW
                       ))):
            raise RuntimeError(
                f"frame {index} has unexpected XYZW quaternion: {quaternion}"
            )
        if (not finite_number(age) or age < 0.0 or age > 0.1):
            raise RuntimeError(
                f"frame {index} has invalid attitude age: {age}"
            )
        if (not isinstance(sequence, int) or sequence <= 0
                or sequence % 2 != 0):
            raise RuntimeError(
                f"frame {index} has invalid attitude sequence: {sequence}"
            )

    successes = [
        frame for frame in frames
        if frame.get("blob_grid_localization", {}).get("status") == "success"
    ]
    if len(successes) < MIN_SUCCESSFUL_FRAMES:
        raise RuntimeError(
            f"only {len(successes)} successful frames; expected at least "
            f"{MIN_SUCCESSFUL_FRAMES}: {statuses}"
        )

    next_distance = expected_distance
    for index, localization in enumerate(localizations):
        distance_used = localization.get("distance_used")
        if (not finite_number(distance_used)
                or abs(distance_used - next_distance) > 1e-9):
            raise RuntimeError(
                f"frame {index} used {distance_used}, expected retained "
                f"distance {next_distance}"
            )
        if localization.get("status") == "success":
            next_distance = localization.get("camera_to_plane_distance")
            if not finite_number(next_distance) or next_distance <= 0.0:
                raise RuntimeError(
                    f"frame {index} produced invalid next distance: "
                    f"{next_distance}"
                )

    translations = []
    for frame in successes:
        localization = frame["blob_grid_localization"]
        poses = frame.get("poses", [])
        if len(poses) != 1 or poses[0].get("source") != "blob_grid":
            raise RuntimeError(
                "a successful frame did not contain exactly one blob_grid pose"
            )
        pose = poses[0]
        marker_ids = pose.get("used_marker_ids", [])
        map_cells = pose.get("used_map_cells", [])
        marker_cells = {
            (marker_id, cell.get("row"), cell.get("col"))
            for marker_id, cell in zip(marker_ids, map_cells)
            if isinstance(cell, dict)
        }
        if (len(marker_ids) != 4 or set(marker_ids) != EXPECTED_MARKER_IDS
                or len(map_cells) != 4
                or marker_cells != EXPECTED_MARKER_CELLS):
            raise RuntimeError(
                f"unexpected marker/grid mapping: {sorted(marker_cells)}"
            )
        if (pose.get("pnp_solver") != "ap3p"
                or pose.get("markers_used") != 4
                or localization.get("pnp_solver") != "ap3p"):
            raise RuntimeError("successful pose did not use four-marker AP3P")

        matched_markers = localization.get("matched_markers", [])
        if len(matched_markers) != 4:
            raise RuntimeError("successful pose did not report four map matches")
        for marker in matched_markers:
            marker_id = marker.get("id")
            position = marker.get("global_position")
            expected_position = EXPECTED_GLOBAL_POSITIONS.get(marker_id)
            if (expected_position is None or not isinstance(position, list)
                    or len(position) != 3
                    or not all(finite_number(value) for value in position)
                    or any(abs(actual - expected) > 1e-6
                           for actual, expected in zip(
                               position, expected_position
                           ))):
                raise RuntimeError(
                    f"marker {marker_id} has wrong global position: {position}"
                )
        reprojection_error = pose.get("reprojection_error")
        if (not finite_number(reprojection_error)
                or reprojection_error < 0.0
                or reprojection_error > MAX_REPROJECTION_ERROR_PIXELS):
            raise RuntimeError(
                f"invalid reprojection error: {reprojection_error}"
            )
        translation = pose.get("marker_position")
        if (not isinstance(translation, list) or len(translation) != 3
                or not all(finite_number(value) for value in translation)):
            raise RuntimeError(f"invalid OpenCV tvec: {translation}")
        translations.append(translation)
        for field in ("camera_position", "drone_position"):
            position = pose.get(field)
            if (not isinstance(position, list) or len(position) != 3
                    or not all(finite_number(value) for value in position)):
                raise RuntimeError(f"invalid {field}: {position}")
        distance_used = localization.get("distance_used")
        updated_distance = localization.get("camera_to_plane_distance")
        if (not finite_number(distance_used) or distance_used <= 0.0
                or not finite_number(updated_distance)
                or updated_distance <= 0.0):
            raise RuntimeError(
                f"invalid dynamic distance: {distance_used}, {updated_distance}"
            )

    median_tvec = tuple(
        statistics.median(translation[axis] for translation in translations)
        for axis in range(3)
    )
    expected_tvec = (0.5, 0.2, expected_distance)
    if any(abs(actual - expected) > 0.03
           for actual, expected in zip(median_tvec, expected_tvec)):
        raise RuntimeError(
            f"median tvec {median_tvec} is not near {expected_tvec}"
        )
    print(
        f"PASS: {len(successes)}/{len(frames)} frames localized with valid "
        "shared attitude; median tvec="
        f"({median_tvec[0]:.3f}, {median_tvec[1]:.3f}, "
        f"{median_tvec[2]:.3f}) m"
    )
    print("statuses:", dict(statuses))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eye", type=Path, default=ROOT / "build_video/eye")
    parser.add_argument(
        "--video", type=Path,
        default=LIGHTBENDER / "orchestrator/render_lb1.mp4"
    )
    parser.add_argument(
        "--grid", type=Path,
        default=LIGHTBENDER / "orchestrator/marker_grid_2x2.json"
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "logs/render_lb1_grid_test.json"
    )
    parser.add_argument("--distance", type=float, default=1.0)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--force-shm", action="store_true",
        help="replace an existing /pos_shared_mem object (only if it is stale)"
    )
    return parser.parse_args()


def interrupt(_signal_number, _frame):
    raise KeyboardInterrupt


def main():
    args = parse_args()
    signal.signal(signal.SIGTERM, interrupt)
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, interrupt)
    args.eye = args.eye.expanduser().resolve()
    args.video = args.video.expanduser().resolve()
    args.grid = args.grid.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    for path, label in ((args.eye, "video-build executable"),
                        (args.video, "video"), (args.grid, "grid")):
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")
    if not math.isfinite(args.distance) or args.distance <= 0.0:
        raise ValueError("--distance must be positive and finite")
    if args.output in {args.eye, args.video, args.grid}:
        raise ValueError("--output must not overwrite the executable or inputs")
    # Blender camera: 2.85 mm lens, 2.4 mm vertical sensor, 640x400 render.
    # Therefore fx=fy=475 px, principal point=(320,200), no lens distortion.
    camera_config = {
        "camera_matrix": [[475.0, 0.0, 320.0],
                          [0.0, 475.0, 200.0],
                          [0.0, 0.0, 1.0]],
        "dist_coeffs": [0.0, 0.0, 0.0, 0.0, 0.0],
    }
    library = posix_shm_library()

    process = None
    shared = None
    owns_shared_memory = False
    try:
        reserve_shared_memory(library, args.force_shm)
        owns_shared_memory = True
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.unlink(missing_ok=True)
        with tempfile.TemporaryDirectory(prefix="fls-video-grid-") as directory:
            config_path = Path(directory) / "blender_camera.json"
            config_path.write_text(json.dumps(camera_config))
            command = [
                str(args.eye), "--video-input", str(args.video),
                "--grid-map", str(args.grid), "--config", str(config_path),
                "--distance", str(args.distance), "--window-size", "2",
                "--payload-size", "4", "--encoder-fps", "50",
                "--max-attitude-age", "0.1",
                "--dark-blob-intensity", "0.25", "--grid-center-ap3p",
                "--json-path", str(args.output),
            ]
            if args.preview:
                command.append("--preview")
            if args.verbose:
                command.append("--verbose")

            process = subprocess.Popen(command, cwd=directory)
            run_deadline = time.monotonic() + RUN_TIMEOUT_SECONDS
            shared = attach_shared_memory(library, process)
            last_timestamp = 0.0
            published_timestamp = None
            while process.poll() is None:
                if time.monotonic() >= run_deadline:
                    raise TimeoutError(
                        f"video test exceeded {RUN_TIMEOUT_SECONDS:.0f} seconds"
                    )
                timestamp = struct.unpack_from(
                    "<d", shared, POSE_TIMESTAMP_OFFSET
                )[0]
                if math.isfinite(timestamp) and timestamp >= 0.0:
                    last_timestamp = timestamp
                sequence = struct.unpack_from(
                    "<I", shared, ATTITUDE_BEGIN_OFFSET
                )[0]
                if published_timestamp != last_timestamp or sequence == 0:
                    write_attitude(
                        shared, last_timestamp, IDENTITY_QUATERNION_XYZW
                    )
                    published_timestamp = last_timestamp
                time.sleep(0.002)

            return_code = process.wait()
            if return_code != 0:
                raise RuntimeError(f"eye exited with status {return_code}")
    finally:
        try:
            if shared is not None:
                try:
                    write_attitude(
                        shared, 0.0, IDENTITY_QUATERNION_XYZW, valid=False
                    )
                finally:
                    shared.close()
        finally:
            try:
                if process is not None:
                    stop_process(process)
            finally:
                if owns_shared_memory:
                    unlink_shared_memory(library)

    verify_log(args.output, args.distance)
    print(f"log: {args.output}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        sys.exit(130)
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
