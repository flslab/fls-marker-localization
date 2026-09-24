#!/usr/bin/env python3
"""Capture calibration images with the production high-rate camera settings."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import signal
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from calibration_lib import create_charuco_detector, write_json


@dataclass(frozen=True)
class ProductionSettings:
    width: int
    height: int
    frame_rate: float
    buffer_count: int
    pixel_format: str
    exposure_time_us: int
    analogue_gain: float
    brightness: float
    contrast: float
    camera_matrix: np.ndarray
    distortion: np.ndarray

    @classmethod
    def from_config(cls, root: dict[str, Any]) -> "ProductionSettings":
        camera = root["camera"]
        calibration = root["calibration"]
        settings = cls(
            width=int(camera["width"]),
            height=int(camera["height"]),
            frame_rate=float(camera["frame_rate"]),
            buffer_count=int(camera.get("buffer_count", 4)),
            pixel_format=str(camera.get("pixel_format", "YUV420")),
            exposure_time_us=int(camera["exposure_time_us"]),
            analogue_gain=float(camera["analogue_gain"]),
            brightness=float(camera.get("brightness", 0.0)),
            contrast=float(camera.get("contrast", 1.0)),
            camera_matrix=np.asarray(calibration["camera_matrix"], dtype=np.float64),
            distortion=np.asarray(calibration["distortion_coefficients"], dtype=np.float64),
        )
        if settings.width <= 0 or settings.height <= 0 or settings.frame_rate <= 0:
            raise ValueError("production camera dimensions and frame rate must be positive")
        if settings.pixel_format != "YUV420":
            raise ValueError("capture assistant currently requires the production YUV420 format")
        if settings.camera_matrix.shape != (3, 3):
            raise ValueError("production camera matrix must be 3x3")
        return settings

    def as_json(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "frame_rate": self.frame_rate,
            "buffer_count": self.buffer_count,
            "pixel_format": self.pixel_format,
            "exposure_time_us": self.exposure_time_us,
            "analogue_gain": self.analogue_gain,
            "brightness": self.brightness,
            "contrast": self.contrast,
            "camera_matrix": self.camera_matrix.tolist(),
            "distortion_coefficients": self.distortion.tolist(),
        }


@dataclass
class Frame:
    gray: np.ndarray
    timestamp_s: float
    metadata: dict[str, Any]


class Picamera2Source:
    """Picamera2/libcamera source configured to match LibcameraSource."""

    def __init__(self, settings: ProductionSettings):
        try:
            from picamera2 import Picamera2
        except ImportError as error:
            raise RuntimeError(
                "Picamera2 is required for production capture. Install the Raspberry Pi "
                "python3-picamera2 package and run this script with that Python environment."
            ) from error
        self.settings = settings
        self.camera = Picamera2()
        frame_duration_us = int(round(1_000_000.0 / settings.frame_rate))
        controls = {
            "FrameDurationLimits": (frame_duration_us, frame_duration_us),
            # "ExposureTime": settings.exposure_time_us,
            # "AnalogueGain": settings.analogue_gain,
            # "Brightness": settings.brightness,
            # "Contrast": settings.contrast,
        }
        # The C++ production source requests libcamera::StreamRole::Viewfinder;
        # Picamera2's preview configuration is the corresponding use case.
        configuration = self.camera.create_preview_configuration(
            main={"size": (settings.width, settings.height), "format": settings.pixel_format},
            buffer_count=settings.buffer_count,
            controls=controls,
        )
        self.camera.configure(configuration)
        self.camera.start()

    def read(self) -> Frame | None:
        request = self.camera.capture_request()
        try:
            array = request.make_array("main")
            metadata = dict(request.get_metadata())
            if array.ndim == 2:
                gray = np.asarray(
                    array[: self.settings.height, : self.settings.width], dtype=np.uint8
                ).copy()
            elif array.ndim == 3:
                gray = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
            else:
                raise RuntimeError(f"unexpected Picamera2 array shape {array.shape}")
            timestamp_ns = metadata.get("SensorTimestamp")
            timestamp_s = (
                float(timestamp_ns) * 1e-9 if timestamp_ns is not None else time.monotonic()
            )
            return Frame(gray=gray, timestamp_s=timestamp_s, metadata=metadata)
        finally:
            request.release()

    def close(self) -> None:
        self.camera.stop()
        self.camera.close()


class VideoSource:
    """Non-production source for UI rehearsal and automated tests."""

    def __init__(self, path: Path, settings: ProductionSettings, loop: bool):
        self.path = path
        self.settings = settings
        self.loop = loop
        self.capture = cv2.VideoCapture(str(path))
        if not self.capture.isOpened():
            raise RuntimeError(f"unable to open video {path}")

    def read(self) -> Frame | None:
        success, image = self.capture.read()
        if not success and self.loop:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            success, image = self.capture.read()
        if not success:
            return None
        if image.shape[1] != self.settings.width or image.shape[0] != self.settings.height:
            raise RuntimeError(
                f"video frame is {image.shape[1]}x{image.shape[0]}, production config is "
                f"{self.settings.width}x{self.settings.height}"
            )
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        return Frame(gray=gray, timestamp_s=time.monotonic(), metadata={"source": "video"})

    def close(self) -> None:
        self.capture.release()


@dataclass
class TargetObservation:
    valid: bool
    image_points: np.ndarray
    object_points: np.ndarray
    ids: np.ndarray
    sharpness: float | None
    coverage_fraction: float
    centroid: tuple[float, float] | None
    occupied_bins: list[tuple[int, int]]
    descriptor: np.ndarray | None
    approximate_rvec: np.ndarray | None
    approximate_tvec: np.ndarray | None


class TargetTracker:
    def __init__(
        self,
        pipeline_config: dict[str, Any],
        settings: ProductionSettings,
        coverage_shape: tuple[int, int],
    ):
        target = pipeline_config["target"]
        if str(target["type"]).lower() != "charuco":
            raise ValueError(
                "the guided capture assistant currently supports the recommended ChArUco target"
            )
        self.board, self.detector = create_charuco_detector(
            target, pipeline_config.get("detection", {})
        )
        if self.detector is None:
            raise RuntimeError("guided capture requires OpenCV CharucoDetector support")
        self.board_points = np.asarray(
            self.board.getChessboardCorners(), dtype=np.float64
        ).reshape(-1, 3)
        self.minimum_points = int(target.get("minimum_points", 12))
        self.minimum_sharpness = pipeline_config.get("detection", {}).get(
            "minimum_laplacian_variance"
        )
        self.settings = settings
        self.coverage_shape = coverage_shape

    def detect(self, gray: np.ndarray) -> TargetObservation:
        corners, ids, _, _ = self.detector.detectBoard(gray)
        if corners is None or ids is None:
            return self._empty()
        ids = np.asarray(ids).reshape(-1).astype(int)
        image_points = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
        object_points = self.board_points[ids]
        if not len(image_points):
            return self._empty()
        x, y, width, height = cv2.boundingRect(image_points.astype(np.float32))
        pad = 4
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1 = min(gray.shape[1], x + width + pad)
        y1 = min(gray.shape[0], y + height + pad)
        target_image = gray[y0:y1, x0:x1]
        sharpness = float(cv2.Laplacian(target_image, cv2.CV_64F).var())
        hull = cv2.convexHull(image_points.astype(np.float32))
        coverage_fraction = float(
            cv2.contourArea(hull) / (self.settings.width * self.settings.height)
        )
        centroid_array = np.mean(image_points, axis=0)
        centroid = (float(centroid_array[0]), float(centroid_array[1]))
        occupied_bins = sorted(
            {
                (
                    min(
                        self.coverage_shape[0] - 1,
                        max(0, int(point[0] / self.settings.width * self.coverage_shape[0])),
                    ),
                    min(
                        self.coverage_shape[1] - 1,
                        max(0, int(point[1] / self.settings.height * self.coverage_shape[1])),
                    ),
                )
                for point in image_points
            }
        )
        approximate_pose = self._solve_approximate_pose(object_points, image_points)
        rvec = approximate_pose[0] if approximate_pose else None
        tvec = approximate_pose[1] if approximate_pose else None
        if rvec is not None and tvec is not None and tvec[2] > 0:
            pose_terms = np.concatenate([rvec / 0.5, [math.log(max(float(tvec[2]), 1e-6))]])
        else:
            centered = image_points - centroid_array
            covariance = centered.T @ centered / max(1, len(centered))
            _, vectors = np.linalg.eigh(covariance)
            major = vectors[:, -1]
            angle = math.atan2(float(major[1]), float(major[0]))
            pose_terms = np.asarray(
                [math.log(max(math.sqrt(coverage_fraction), 1e-6)), math.cos(2 * angle), math.sin(2 * angle), 0.0]
            )
        descriptor = np.concatenate(
            [
                np.asarray(
                    [centroid[0] / self.settings.width, centroid[1] / self.settings.height]
                ),
                pose_terms,
            ]
        )
        valid = len(image_points) >= self.minimum_points and (
            self.minimum_sharpness is None or sharpness >= float(self.minimum_sharpness)
        )
        return TargetObservation(
            valid=valid,
            image_points=image_points,
            object_points=object_points,
            ids=ids,
            sharpness=sharpness,
            coverage_fraction=coverage_fraction,
            centroid=centroid,
            occupied_bins=occupied_bins,
            descriptor=descriptor,
            approximate_rvec=rvec,
            approximate_tvec=tvec,
        )

    def _solve_approximate_pose(
        self, object_points: np.ndarray, image_points: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray] | None:
        if len(object_points) < 4:
            return None
        try:
            count, rvecs, tvecs = cv2.solvePnPGeneric(
                object_points,
                image_points,
                self.settings.camera_matrix,
                self.settings.distortion,
                flags=cv2.SOLVEPNP_SQPNP,
            )[:3]
        except cv2.error:
            return None
        if count <= 0:
            return None
        best = None
        for rvec, tvec in zip(rvecs, tvecs):
            rvec = np.asarray(rvec, dtype=np.float64).reshape(3)
            tvec = np.asarray(tvec, dtype=np.float64).reshape(3)
            projected, _ = cv2.projectPoints(
                object_points,
                rvec,
                tvec,
                self.settings.camera_matrix,
                self.settings.distortion,
            )
            rms = float(
                np.sqrt(
                    np.mean(
                        np.sum(np.square(projected.reshape(-1, 2) - image_points), axis=1)
                    )
                )
            )
            if tvec[2] > 0 and (best is None or rms < best[0]):
                best = (rms, rvec, tvec)
        return None if best is None else (best[1], best[2])

    @staticmethod
    def _empty() -> TargetObservation:
        return TargetObservation(
            valid=False,
            image_points=np.empty((0, 2)),
            object_points=np.empty((0, 3)),
            ids=np.empty((0,), dtype=int),
            sharpness=None,
            coverage_fraction=0.0,
            centroid=None,
            occupied_bins=[],
            descriptor=None,
            approximate_rvec=None,
            approximate_tvec=None,
        )


class CapturePolicy:
    def __init__(self, capture_config: dict[str, Any], coverage_shape: tuple[int, int]):
        self.stable_frames_required = int(capture_config.get("stable_frames", 6))
        self.maximum_motion_px = float(capture_config.get("maximum_stable_motion_px", 1.5))
        self.minimum_interval_s = float(capture_config.get("minimum_seconds_between_captures", 1.0))
        self.minimum_descriptor_distance = float(
            capture_config.get("minimum_descriptor_distance", 0.35)
        )
        self.minimum_captures_per_bin = int(capture_config.get("minimum_captures_per_bin", 2))
        self.coverage = np.zeros((coverage_shape[1], coverage_shape[0]), dtype=np.int64)
        self.previous_points: dict[int, np.ndarray] = {}
        self.stable_frames = 0
        self.last_capture_time = -math.inf
        self.descriptors: list[np.ndarray] = []

    def update_stability(self, observation: TargetObservation) -> float | None:
        current = {
            int(identifier): point
            for identifier, point in zip(observation.ids, observation.image_points)
        }
        common = sorted(set(current) & set(self.previous_points))
        motion = None
        if len(common) >= 4:
            motion = float(
                np.median(
                    [np.linalg.norm(current[item] - self.previous_points[item]) for item in common]
                )
            )
            self.stable_frames = self.stable_frames + 1 if motion <= self.maximum_motion_px else 0
        else:
            self.stable_frames = 0
        self.previous_points = current
        return motion

    def decision(
        self,
        observation: TargetObservation,
        now: float,
        settings_ok: bool,
    ) -> tuple[bool, str, float | None]:
        if not observation.valid:
            return False, "target_or_sharpness_invalid", None
        if not settings_ok:
            return False, "camera_settings_not_settled", None
        if self.stable_frames < self.stable_frames_required:
            return False, "target_moving", None
        if now - self.last_capture_time < self.minimum_interval_s:
            return False, "cooldown", None
        undercovered = any(
            self.coverage[by, bx] < self.minimum_captures_per_bin
            for bx, by in observation.occupied_bins
        )
        minimum_distance = None
        if observation.descriptor is not None and self.descriptors:
            minimum_distance = min(
                float(np.linalg.norm(observation.descriptor - previous))
                for previous in self.descriptors
            )
        novel_pose = minimum_distance is None or minimum_distance >= self.minimum_descriptor_distance
        if not undercovered and not novel_pose:
            return False, "duplicate_pose", minimum_distance
        return True, "undercovered_region" if undercovered else "novel_pose", minimum_distance

    def accepted(self, observation: TargetObservation, now: float) -> None:
        for bx, by in observation.occupied_bins:
            self.coverage[by, bx] += 1
        if observation.descriptor is not None:
            self.descriptors.append(observation.descriptor.copy())
        self.last_capture_time = now


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _settings_match(metadata: dict[str, Any], settings: ProductionSettings) -> tuple[bool, dict[str, Any]]:
    if metadata.get("source") == "video":
        return True, {"source": "video; production controls not verifiable"}
    actual_exposure = metadata.get("ExposureTime")
    actual_gain = metadata.get("AnalogueGain")
    actual_duration = metadata.get("FrameDuration")
    requested_duration = 1_000_000.0 / settings.frame_rate
    checks = {
        "exposure": actual_exposure is not None
        and abs(float(actual_exposure) - settings.exposure_time_us)
        <= max(10.0, settings.exposure_time_us * 0.05),
        "gain": actual_gain is not None
        and abs(float(actual_gain) - settings.analogue_gain)
        <= max(0.02, settings.analogue_gain * 0.05),
        "frame_duration": actual_duration is not None
        and abs(float(actual_duration) - requested_duration) <= requested_duration * 0.05,
    }
    return all(checks.values()), {
        "requested_exposure_time_us": settings.exposure_time_us,
        "actual_exposure_time_us": actual_exposure,
        "requested_analogue_gain": settings.analogue_gain,
        "actual_analogue_gain": actual_gain,
        "requested_frame_duration_us": requested_duration,
        "actual_frame_duration_us": actual_duration,
        "checks": checks,
    }


def _draw_overlay(
    gray: np.ndarray,
    observation: TargetObservation,
    policy: CapturePolicy,
    message: str,
    saved: int,
    auto: bool,
    settings_ok: bool,
) -> np.ndarray:
    display = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if len(observation.image_points):
        cv2.aruco.drawDetectedCornersCharuco(
            display,
            observation.image_points.astype(np.float32).reshape(-1, 1, 2),
            observation.ids.astype(np.int32).reshape(-1, 1),
        )
    lines = [
        f"saved={saved} auto={'on' if auto else 'off'} settings={'ok' if settings_ok else 'WAIT'}",
        f"corners={len(observation.image_points)} sharpness={observation.sharpness if observation.sharpness is not None else 0:.1f} area={100*observation.coverage_fraction:.1f}%",
        f"stable={policy.stable_frames}/{policy.stable_frames_required} status={message}",
        "keys: SPACE save valid | f force raw | a auto | r reset coverage | q quit",
    ]
    for index, line in enumerate(lines):
        y = 22 + index * 22
        cv2.putText(display, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 3)
        cv2.putText(display, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)
    bins_y, bins_x = policy.coverage.shape
    cell_size = 28
    origin_x = display.shape[1] - bins_x * cell_size - 10
    origin_y = 10
    for by in range(bins_y):
        for bx in range(bins_x):
            count = int(policy.coverage[by, bx])
            complete = count >= policy.minimum_captures_per_bin
            color = (0, 180, 0) if complete else (0, 80, 220)
            top_left = (origin_x + bx * cell_size, origin_y + by * cell_size)
            bottom_right = (top_left[0] + cell_size - 2, top_left[1] + cell_size - 2)
            cv2.rectangle(display, top_left, bottom_right, color, 2)
            cv2.putText(
                display,
                str(count),
                (top_left[0] + 8, top_left[1] + 19),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
            )
    return display


def _capture_record(
    path: Path,
    frame: Frame,
    observation: TargetObservation,
    reason: str,
    forced: bool,
    settings_details: dict[str, Any],
) -> dict[str, Any]:
    return {
        "path": str(path),
        "sensor_timestamp_s": frame.timestamp_s,
        "saved_monotonic_s": time.monotonic(),
        "capture_reason": reason,
        "forced_without_valid_target": forced,
        "detected_corners": len(observation.image_points),
        "charuco_ids": observation.ids.tolist(),
        "sharpness_laplacian_variance": observation.sharpness,
        "target_coverage_fraction": observation.coverage_fraction,
        "target_centroid_px": list(observation.centroid) if observation.centroid else None,
        "occupied_coverage_bins": [list(item) for item in observation.occupied_bins],
        "novelty_descriptor": (
            observation.descriptor.tolist() if observation.descriptor is not None else None
        ),
        "approximate_target_rvec": (
            observation.approximate_rvec.tolist()
            if observation.approximate_rvec is not None
            else None
        ),
        "approximate_target_tvec_m": (
            observation.approximate_tvec.tolist()
            if observation.approximate_tvec is not None
            else None
        ),
        "camera_metadata": frame.metadata,
        "production_settings_check": settings_details,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--production-config", required=True, help="high-rate localizer JSON config")
    parser.add_argument("--pipeline-config", required=True, help="calibration pipeline JSON config")
    parser.add_argument("--session", required=True, help="independent capture-session name")
    parser.add_argument("--output-dir", required=True, help="parent directory for session folders")
    parser.add_argument("--auto", action="store_true", help="start automatic quality-gated capture")
    parser.add_argument("--headless", action="store_true", help="write latest_preview.jpg instead of a GUI")
    parser.add_argument("--duration", type=float, default=0.0, help="stop after this many seconds; zero disables")
    parser.add_argument("--max-images", type=int, default=0, help="stop after this many images; zero disables")
    parser.add_argument("--resume", action="store_true", help="continue an existing session directory")
    parser.add_argument("--video", help="rehearse UI using a video instead of the production camera")
    parser.add_argument("--loop-video", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.session):
        raise ValueError("session must contain only letters, digits, underscore, and hyphen")
    if args.headless and not args.auto:
        raise ValueError("headless capture requires --auto; stop it with Ctrl-C")
    production_path = Path(args.production_config).expanduser().resolve()
    pipeline_path = Path(args.pipeline_config).expanduser().resolve()
    production_root = json.loads(production_path.read_text())
    pipeline_root = json.loads(pipeline_path.read_text())
    settings = ProductionSettings.from_config(production_root)
    if tuple(int(item) for item in pipeline_root["image_size"]) != (
        settings.width,
        settings.height,
    ):
        raise ValueError("pipeline image_size does not match the production camera config")
    capture_config = pipeline_root.get("capture", {})
    coverage_shape = tuple(int(item) for item in capture_config.get("coverage_bins", [4, 3]))
    tracker = TargetTracker(pipeline_root, settings, coverage_shape)
    policy = CapturePolicy(capture_config, coverage_shape)

    session_dir = Path(args.output_dir).expanduser().resolve() / args.session
    existing_images = sorted(session_dir.glob("*.png")) if session_dir.exists() else []
    if existing_images and not args.resume:
        raise ValueError(
            f"session already contains {len(existing_images)} images; pass --resume or choose a new session"
        )
    session_dir.mkdir(parents=True, exist_ok=True)
    new_session_metadata = {
        "schema_version": 1,
        "session": args.session,
        "started_epoch_s": time.time(),
        "production_config": str(production_path),
        "production_config_sha256": _sha256(production_path),
        "pipeline_config": str(pipeline_path),
        "pipeline_config_sha256": _sha256(pipeline_path),
        "production_settings": settings.as_json(),
        "processing_crop_ignored_for_calibration_capture": production_root.get("processing_crop"),
        "capture_config": capture_config,
        "source": "video_rehearsal" if args.video else "picamera2_libcamera",
    }
    session_path = session_dir / "session.json"
    if args.resume and session_path.exists():
        session_metadata = json.loads(session_path.read_text())
        if session_metadata.get("production_config_sha256") != new_session_metadata[
            "production_config_sha256"
        ]:
            raise ValueError("cannot resume: production configuration hash changed")
        if session_metadata.get("pipeline_config_sha256") != new_session_metadata[
            "pipeline_config_sha256"
        ]:
            raise ValueError("cannot resume: pipeline configuration hash changed")
        session_metadata.setdefault("resumed_epoch_s", []).append(time.time())
    else:
        session_metadata = new_session_metadata
    write_json(session_path, session_metadata)
    manifest_path = session_dir / "manifest.jsonl"
    if args.resume and manifest_path.exists():
        with manifest_path.open() as stream:
            for line in stream:
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("forced_without_valid_target", False):
                    continue
                for bx, by in record.get("occupied_coverage_bins", []):
                    if 0 <= by < policy.coverage.shape[0] and 0 <= bx < policy.coverage.shape[1]:
                        policy.coverage[by, bx] += 1
                descriptor = record.get("novelty_descriptor")
                if descriptor is not None:
                    policy.descriptors.append(np.asarray(descriptor, dtype=np.float64))

    source = (
        VideoSource(Path(args.video).expanduser().resolve(), settings, args.loop_video)
        if args.video
        else Picamera2Source(settings)
    )
    running = True

    def stop(_signal=None, _frame=None):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    auto = bool(args.auto)
    saved = len(existing_images)
    existing_indices = []
    for path in existing_images:
        match = re.match(r"(?:frame|forced)_([0-9]+)_", path.name)
        if match:
            existing_indices.append(int(match.group(1)))
    next_index = max(existing_indices, default=0) + 1
    frame_count = 0
    settings_stable_frames = 0
    reasons: Counter[str] = Counter()
    started = time.monotonic()
    last_console = -math.inf
    last_preview = -math.inf
    message = "starting"

    def save(frame: Frame, observation: TargetObservation, reason: str, forced: bool) -> None:
        nonlocal saved, next_index
        saved += 1
        timestamp_ns = int(round(frame.timestamp_s * 1_000_000_000.0))
        prefix = "forced" if forced else "frame"
        image_path = session_dir / f"{prefix}_{next_index:04d}_{timestamp_ns}.png"
        next_index += 1
        if not cv2.imwrite(str(image_path), frame.gray, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
            raise RuntimeError(f"failed to write {image_path}")
        settings_ok, settings_details = _settings_match(frame.metadata, settings)
        record = _capture_record(
            image_path, frame, observation, reason, forced, settings_details
        )
        with manifest_path.open("a") as stream:
            stream.write(json.dumps(_json_safe(record), sort_keys=True) + "\n")
        if not forced:
            policy.accepted(observation, time.monotonic())
        print(f"\nsaved {image_path.name}: {reason}, corners={len(observation.image_points)}")

    try:
        while running:
            frame = source.read()
            if frame is None:
                break
            frame_count += 1
            observation = tracker.detect(frame.gray)
            motion = policy.update_stability(observation)
            settings_ok, settings_details = _settings_match(frame.metadata, settings)
            settings_stable_frames = settings_stable_frames + 1 if settings_ok else 0
            controls_ready = settings_stable_frames >= int(
                capture_config.get("settings_settled_frames", 5)
            )
            now = time.monotonic()
            should_capture, reason, novelty = policy.decision(
                observation, now, controls_ready
            )
            message = reason
            reasons[reason] += 1
            if auto and should_capture:
                save(frame, observation, reason, False)
            overlay = _draw_overlay(
                frame.gray, observation, policy, message, saved, auto, controls_ready
            )
            if args.headless:
                if now - last_preview >= float(
                    capture_config.get("headless_preview_interval_s", 0.5)
                ):
                    cv2.imwrite(str(session_dir / "latest_preview.jpg"), overlay)
                    last_preview = now
                key = -1
            else:
                scale = float(capture_config.get("preview_scale", 1.5))
                shown = (
                    cv2.resize(overlay, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
                    if scale != 1.0
                    else overlay
                )
                cv2.imshow("FLS calibration capture", shown)
                key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("a"):
                auto = not auto
            elif key == ord("r"):
                policy.coverage.fill(0)
                policy.descriptors.clear()
            elif key == ord(" ") and observation.valid and controls_ready:
                save(frame, observation, "manual_valid_target", False)
            elif key == ord("f"):
                save(frame, observation, "manual_forced_raw", True)

            if now - last_console >= 1.0:
                coverage_complete = int(
                    np.sum(policy.coverage >= policy.minimum_captures_per_bin)
                )
                coverage_total = int(policy.coverage.size)
                print(
                    "\rframes={frames} saved={saved} corners={corners} sharp={sharp:.1f} "
                    "motion={motion} coverage={done}/{total} status={status}    ".format(
                        frames=frame_count,
                        saved=saved,
                        corners=len(observation.image_points),
                        sharp=observation.sharpness or 0.0,
                        motion="—" if motion is None else f"{motion:.2f}px",
                        done=coverage_complete,
                        total=coverage_total,
                        status=message,
                    ),
                    end="",
                    flush=True,
                )
                last_console = now
            if args.duration > 0 and now - started >= args.duration:
                break
            if args.max_images > 0 and saved >= args.max_images:
                break
    finally:
        source.close()
        if not args.headless:
            cv2.destroyAllWindows()
        summary = {
            **session_metadata,
            "ended_epoch_s": time.time(),
            "duration_s": time.monotonic() - started,
            "frames_processed": frame_count,
            "images_saved": saved,
            "calibration_images_saved": len(list(session_dir.glob("frame_*.png"))),
            "forced_diagnostic_images_saved": len(list(session_dir.glob("forced_*.png"))),
            "coverage_counts": policy.coverage.tolist(),
            "minimum_captures_per_bin": policy.minimum_captures_per_bin,
            "coverage_complete": bool(
                np.all(policy.coverage >= policy.minimum_captures_per_bin)
            ),
            "frame_decision_counts": dict(reasons),
        }
        write_json(session_dir / "capture_summary.json", summary)
        print(f"\nsummary: {session_dir / 'capture_summary.json'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, OSError, cv2.error) as error:
        print(f"capture failed: {error}", file=sys.stderr)
        raise SystemExit(1)
