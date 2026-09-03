#!/usr/bin/env python3
"""Run a Blender marker-grid video with a shared-memory test attitude."""

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
RUN_TIMEOUT_SECONDS = 120.0
EXPECTED_FRAME_COUNT = 1950
MIN_SUCCESSFUL_FRAMES = 200
MIN_LIFECYCLE_PHASE_FRAMES = 5
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


def write_attitude(shared, timestamp, quaternion, valid=True,
                   use_short_range=False):
    sequence = struct.unpack_from(
        "<I", shared, ATTITUDE_BEGIN_OFFSET
    )[0]
    next_even = ((sequence & ~1) + 2) & 0xFFFFFFFF
    if next_even == 0:
        next_even = 2

    struct.pack_into("<I", shared, ATTITUDE_BEGIN_OFFSET, next_even - 1)
    payload = struct.pack(
        "<BB2xd4fI", int(valid), int(use_short_range), timestamp,
        *quaternion, next_even
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


def marker_id_sets(marker_map):
    main_ids = {
        marker_id
        for row in marker_map.get("grid", [])
        if isinstance(row, list)
        for marker_id in row
        if isinstance(marker_id, int) and not isinstance(marker_id, bool)
    }
    short_ids = {
        marker.get("id")
        for tile in marker_map.get("short_range", {}).get("tiles", [])
        if isinstance(tile, dict)
        for marker in tile.get("markers", [])
        if isinstance(marker, dict)
        and isinstance(marker.get("id"), int)
        and not isinstance(marker.get("id"), bool)
    }
    return main_ids, short_ids


def payload_size_for_map(marker_map):
    main_ids, short_ids = marker_id_sets(marker_map)
    marker_ids = main_ids | short_ids
    if not marker_ids or min(marker_ids) < 0:
        raise ValueError("grid map must contain non-negative marker IDs")
    payload_size = max(1, max(marker_ids).bit_length())
    if payload_size > 16:
        raise ValueError(
            f"marker ID {max(marker_ids)} exceeds the 16-bit decoder limit"
        )
    return payload_size


def lifecycle_phases(successes, minimum_phase_frames):
    phases = []
    for frame in successes:
        localization = frame.get("blob_grid_localization", {})
        grid_type = localization.get("grid_type")
        if grid_type not in {"main", "short_range"}:
            raise RuntimeError(
                "lifecycle validation requires explicit main/short_range "
                f"grid_type on every successful frame; got {grid_type!r}"
            )
        if not phases or phases[-1][0] != grid_type:
            phases.append([grid_type, 0])
        phases[-1][1] += 1
    phase_names = [phase[0] for phase in phases]
    expected = ["short_range", "main", "short_range"]
    if phase_names != expected:
        raise RuntimeError(
            f"expected takeoff/cruise/landing phases {expected}, got "
            f"{phase_names or 'no successful phase data'}"
        )
    short_phase_counts = [count for name, count in phases
                          if name == "short_range"]
    if any(count < minimum_phase_frames for _, count in phases):
        raise RuntimeError(
            f"lifecycle phase counts {dict(enumerate(phases))} do not each "
            f"contain at least {minimum_phase_frames} successful frames"
        )
    if len(short_phase_counts) != 2:
        raise RuntimeError("lifecycle must contain two short-range phases")
    return phases


def verify_lifecycle_frame(frame, marker_map, expected_tile):
    if not isinstance(marker_map, dict):
        raise RuntimeError("lifecycle validation requires the marker map")
    localization = frame["blob_grid_localization"]
    pose = frame["poses"][0]
    grid_type = localization["grid_type"]
    if pose.get("grid_type") != grid_type:
        raise RuntimeError(
            f"pose/localization grid mismatch: {pose.get('grid_type')!r} "
            f"versus {grid_type!r}"
        )

    main_ids, short_ids = marker_id_sets(marker_map)
    used_ids = pose.get("used_marker_ids", [])
    expected_ids = short_ids if grid_type == "short_range" else main_ids
    if not used_ids or any(marker_id not in expected_ids
                           for marker_id in used_ids):
        raise RuntimeError(
            f"{grid_type} pose used IDs outside its map: {used_ids}"
        )

    map_cells = pose.get("used_map_cells", [])
    if (len(map_cells) != len(used_ids)
            or not all(isinstance(cell, dict) for cell in map_cells)):
        raise RuntimeError("used_marker_ids/used_map_cells are inconsistent")
    if any(cell.get("grid_type") != grid_type for cell in map_cells):
        raise RuntimeError("used map cell grid_type does not match the pose")

    matched = localization.get("matched_markers", [])
    if (len(matched) < 4
            or not all(isinstance(marker, dict) for marker in matched)):
        raise RuntimeError(
            f"{grid_type} success reported only {len(matched)} map matches"
        )
    if any(marker.get("grid_type") != grid_type for marker in matched):
        raise RuntimeError("matched marker grid_type does not match the pose")
    for marker in matched:
        position = marker.get("global_position")
        if (not isinstance(position, list) or len(position) != 3
                or not all(finite_number(value) for value in position)):
            raise RuntimeError(
                f"marker {marker.get('id')} has invalid world position"
            )

    if grid_type != "short_range":
        return
    tile = localization.get("tile")
    if (not isinstance(tile, dict)
            or not isinstance(tile.get("i"), int)
            or not isinstance(tile.get("j"), int)):
        raise RuntimeError("short-range success is missing its tile coordinate")
    if pose.get("tile") != tile:
        raise RuntimeError("short-range pose/localization tile mismatch")
    if expected_tile is not None and (tile["i"], tile["j"]) != expected_tile:
        raise RuntimeError(
            f"short-range pose used tile {(tile['i'], tile['j'])}, "
            f"expected {expected_tile}"
        )
    for marker in matched:
        if ((marker.get("tile_i"), marker.get("tile_j"))
                != (tile["i"], tile["j"])
                or not isinstance(marker.get("local_i"), int)
                or not isinstance(marker.get("local_j"), int)):
            raise RuntimeError(
                "short-range match is missing matching tile/local coordinates"
            )
    for cell in map_cells:
        if ((cell.get("tile_i"), cell.get("tile_j"))
                != (tile["i"], tile["j"])
                or not isinstance(cell.get("local_i"), int)
                or not isinstance(cell.get("local_j"), int)):
            raise RuntimeError(
                "short-range used cell is missing matching tile/local coordinates"
            )


def verify_log(path, expected_distance, marker_map=None,
               expect_lifecycle=False, expected_frame_count=None,
               minimum_successful_frames=MIN_SUCCESSFUL_FRAMES,
               minimum_phase_frames=MIN_LIFECYCLE_PHASE_FRAMES,
               expected_tile=None):
    log = json.loads(path.read_text())
    frames = log.get("frames", [])
    if expected_frame_count is None and not expect_lifecycle:
        expected_frame_count = EXPECTED_FRAME_COUNT
    if (expected_frame_count is not None
            and len(frames) != expected_frame_count):
        raise RuntimeError(
            f"expected {expected_frame_count} frames, got {len(frames)}"
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
        use_short_range = attitude.get("use_short_range")
        selection = localization.get("grid_selection", {})
        expected_grid = "short_range" if use_short_range else "main"
        if (not isinstance(use_short_range, bool)
                or selection.get("source") != "shared_memory"
                or selection.get("selected_grid") != expected_grid):
            raise RuntimeError(
                f"frame {index} did not obey its shared grid flag"
            )

    successes = [
        frame for frame in frames
        if frame.get("blob_grid_localization", {}).get("status") == "success"
    ]
    required_successes = (3 * minimum_phase_frames if expect_lifecycle
                          else minimum_successful_frames)
    if len(successes) < required_successes:
        raise RuntimeError(
            f"only {len(successes)} successful frames; expected at least "
            f"{required_successes}: {statuses}"
        )
    phases = (lifecycle_phases(successes, minimum_phase_frames)
              if expect_lifecycle else None)

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
        if expect_lifecycle:
            verify_lifecycle_frame(frame, marker_map, expected_tile)
        else:
            marker_ids = pose.get("used_marker_ids", [])
            map_cells = pose.get("used_map_cells", [])
            marker_cells = {
                (marker_id, cell.get("row"), cell.get("col"))
                for marker_id, cell in zip(marker_ids, map_cells)
                if isinstance(cell, dict)
            }
            if (len(marker_ids) != 4
                    or set(marker_ids) != EXPECTED_MARKER_IDS
                    or len(map_cells) != 4
                    or marker_cells != EXPECTED_MARKER_CELLS):
                raise RuntimeError(
                    f"unexpected marker/grid mapping: {sorted(marker_cells)}"
                )
            if (pose.get("pnp_solver") != "ap3p"
                    or pose.get("markers_used") != 4
                    or localization.get("pnp_solver") != "ap3p"):
                raise RuntimeError(
                    "successful pose did not use four-marker AP3P"
                )

            matched_markers = localization.get("matched_markers", [])
            if len(matched_markers) != 4:
                raise RuntimeError(
                    "successful pose did not report four map matches"
                )
            for marker in matched_markers:
                marker_id = marker.get("id")
                position = marker.get("global_position")
                expected_position = EXPECTED_GLOBAL_POSITIONS.get(marker_id)
                if (expected_position is None
                        or not isinstance(position, list)
                        or len(position) != 3
                        or not all(finite_number(value) for value in position)
                        or any(abs(actual - expected) > 1e-6
                               for actual, expected in zip(
                                   position, expected_position
                               ))):
                    raise RuntimeError(
                        f"marker {marker_id} has wrong global position: "
                        f"{position}"
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
    if not expect_lifecycle:
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
    if phases:
        print("lifecycle phases:", ", ".join(
            f"{name}={count}" for name, count in phases
        ))
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
    parser.add_argument(
        "--expect-lifecycle", action="store_true",
        help=("require successful short_range -> main -> short_range "
              "takeoff/cruise/landing phases")
    )
    parser.add_argument(
        "--grid-mode", choices=("main", "short_range"), default="main",
        help="shared-memory grid selection outside lifecycle mode"
    )
    parser.add_argument(
        "--short-range-off-time", type=float, default=3.0,
        help="lifecycle time in seconds to switch from short range to main"
    )
    parser.add_argument(
        "--short-range-on-time", type=float, default=10.0,
        help="lifecycle time in seconds to switch back to short range"
    )
    parser.add_argument(
        "--expected-tile", type=int, nargs=2, metavar=("I", "J"),
        help="require both short-range phases to use this tile"
    )
    parser.add_argument(
        "--expected-frames", type=int,
        help=("expected output frame count (legacy test defaults to 600; "
              "lifecycle mode accepts the scheduled video's length)")
    )
    parser.add_argument(
        "--min-phase-frames", type=int,
        default=MIN_LIFECYCLE_PHASE_FRAMES,
        help="minimum successful frames in each lifecycle phase"
    )
    parser.add_argument(
        "--payload-size", type=int,
        help="decoder bits (default: derive from all main/short marker IDs)"
    )
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
    if args.expected_frames is not None and args.expected_frames <= 0:
        raise ValueError("--expected-frames must be positive")
    if args.min_phase_frames <= 0:
        raise ValueError("--min-phase-frames must be positive")
    if args.expected_tile is not None and not args.expect_lifecycle:
        raise ValueError("--expected-tile requires --expect-lifecycle")
    if (args.expect_lifecycle
            and (not math.isfinite(args.short_range_off_time)
            or not math.isfinite(args.short_range_on_time)
            or args.short_range_off_time < 0.0
            or args.short_range_on_time <= args.short_range_off_time)):
        raise ValueError(
            "short-range lifecycle times must satisfy 0 <= off < on"
        )
    if args.output in {args.eye, args.video, args.grid}:
        raise ValueError("--output must not overwrite the executable or inputs")
    marker_map = json.loads(args.grid.read_text())
    main_ids, short_ids = marker_id_sets(marker_map)
    if main_ids & short_ids:
        raise ValueError(
            "main and short-range marker ID sets must be disjoint: "
            f"{sorted(main_ids & short_ids)}"
        )
    if args.expect_lifecycle and not short_ids:
        raise ValueError(
            "--expect-lifecycle requires a grid map with short_range tiles"
        )
    if args.grid_mode == "short_range" and not short_ids:
        raise ValueError("--grid-mode short_range requires short-range tiles")
    payload_size = (args.payload_size if args.payload_size is not None
                    else payload_size_for_map(marker_map))
    if payload_size < 1 or payload_size > 16:
        raise ValueError("--payload-size must be between 1 and 16")
    required_payload_size = payload_size_for_map(marker_map)
    if payload_size < required_payload_size:
        raise ValueError(
            f"--payload-size {payload_size} cannot encode the map; need at "
            f"least {required_payload_size} bits"
        )
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
                "--payload-size", str(payload_size), "--encoder-fps", "50",
                "--max-attitude-age", "0.1",
                "--dark-blob-intensity", "0.25", "--grid-center-ap3p",
                "--json-path", str(args.output),
                "--save-video", "--video-path", str(args.output.with_suffix(".mp4"))
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
                    use_short_range = args.grid_mode == "short_range"
                    if args.expect_lifecycle:
                        use_short_range = (
                            last_timestamp < args.short_range_off_time
                            or last_timestamp >= args.short_range_on_time
                        )
                    write_attitude(
                        shared, last_timestamp, IDENTITY_QUATERNION_XYZW,
                        use_short_range=use_short_range,
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

    verify_log(
        args.output, args.distance, marker_map=marker_map,
        expect_lifecycle=args.expect_lifecycle,
        expected_frame_count=args.expected_frames,
        minimum_phase_frames=args.min_phase_frames,
        expected_tile=(tuple(args.expected_tile)
                       if args.expected_tile is not None else None),
    )
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
