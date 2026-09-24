#!/usr/bin/env python3
"""Core routines for task-oriented camera calibration and validation.

The module deliberately has only NumPy and OpenCV as hard dependencies.  Plot
generation is optional and imported lazily by ``write_plots``.
"""

from __future__ import annotations

import csv
import glob
import json
import math
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np


SCHEMA_VERSION = 1


def _finite_float(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"expected a finite number, got {value!r}")
    return result


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_value(value), indent=2, sort_keys=True) + "\n")


def read_json(path: Path) -> Any:
    with path.open() as stream:
        return json.load(stream)


def resolve_path(value: str | None, base: Path) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


@dataclass
class Observation:
    view_id: str
    path: str
    group: str
    image_size: tuple[int, int]
    object_points: np.ndarray
    image_points: np.ndarray
    sharpness: float | None = None
    coverage_fraction: float | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "view_id": self.view_id,
            "path": self.path,
            "group": self.group,
            "image_size": list(self.image_size),
            "object_points": self.object_points.tolist(),
            "image_points": self.image_points.tolist(),
            "sharpness": self.sharpness,
            "coverage_fraction": self.coverage_fraction,
        }

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> "Observation":
        object_points = np.asarray(value["object_points"], dtype=np.float64).reshape(-1, 3)
        image_points = np.asarray(value["image_points"], dtype=np.float64).reshape(-1, 2)
        if len(object_points) != len(image_points):
            raise ValueError(f"observation {value.get('view_id')} has unequal point counts")
        return cls(
            view_id=str(value["view_id"]),
            path=str(value.get("path", "")),
            group=str(value.get("group", value["view_id"])),
            image_size=tuple(int(item) for item in value["image_size"]),
            object_points=object_points,
            image_points=image_points,
            sharpness=(None if value.get("sharpness") is None else _finite_float(value["sharpness"])),
            coverage_fraction=(
                None
                if value.get("coverage_fraction") is None
                else _finite_float(value["coverage_fraction"])
            ),
        )


@dataclass
class ModelFit:
    name: str
    model_type: str
    K: np.ndarray
    distortion: np.ndarray
    rms: float
    rvecs: list[np.ndarray]
    tvecs: list[np.ndarray]
    per_view_errors: list[float]
    intrinsic_stddev: list[float]
    flags: list[str]
    deployment_compatible: bool
    parameter_count: int

    def calibration_json(self) -> dict[str, Any]:
        return {
            "model": self.name,
            "model_type": self.model_type,
            "camera_matrix": self.K.tolist(),
            "distortion_coefficients": self.distortion.reshape(-1).tolist(),
            "training_rms_px": self.rms,
            "per_view_rms_px": self.per_view_errors,
            "intrinsic_stddev": self.intrinsic_stddev,
            "flags": self.flags,
            "parameter_count": self.parameter_count,
            "deployment_compatible": self.deployment_compatible,
        }


@dataclass
class ResidualRecord:
    view_id: str
    group: str
    observed_x: float
    observed_y: float
    dx: float
    dy: float
    norm: float


@dataclass
class PoseRecord:
    view_id: str
    group: str
    solved: bool
    retained_points: int
    camera_z_error_m: float | None = None
    camera_position_error_m: float | None = None
    rotation_error_deg: float | None = None
    reprojection_rms_px: float | None = None
    normal_matrix_condition: float | None = None
    tvec_z_sigma_mm_per_px: float | None = None


def series_stats(values: Iterable[float]) -> dict[str, Any]:
    array = np.asarray([float(value) for value in values if math.isfinite(float(value))])
    if array.size == 0:
        return {"samples": 0}
    return {
        "samples": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "standard_deviation": float(np.std(array)),
        "rms": float(np.sqrt(np.mean(np.square(array)))),
        "p05": float(np.quantile(array, 0.05)),
        "p95": float(np.quantile(array, 0.95)),
        "p99": float(np.quantile(array, 0.99)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def _coverage_fraction(points: np.ndarray, image_size: tuple[int, int]) -> float:
    if len(points) < 3:
        return 0.0
    hull = cv2.convexHull(points.astype(np.float32))
    width, height = image_size
    return float(cv2.contourArea(hull) / (width * height))


def _group_for_path(path: Path, regex: str | None) -> str:
    if not regex:
        return path.stem
    match = re.search(regex, str(path))
    if not match:
        return path.stem
    if match.groups():
        return match.group(1)
    return match.group(0)


def create_charuco_detector(target: dict[str, Any], detection: dict[str, Any]):
    aruco = cv2.aruco
    dictionary_name = str(target["dictionary"])
    if not hasattr(aruco, dictionary_name):
        raise ValueError(f"unknown ArUco dictionary {dictionary_name!r}")
    dictionary = aruco.getPredefinedDictionary(getattr(aruco, dictionary_name))
    board = aruco.CharucoBoard(
        (int(target["squares_x"]), int(target["squares_y"])),
        float(target["square_length_m"]),
        float(target["marker_length_m"]),
        dictionary,
    )
    detector_parameters = aruco.DetectorParameters()
    if hasattr(detector_parameters, "cornerRefinementMethod"):
        detector_parameters.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
    for name, value in detection.get("aruco_parameters", {}).items():
        if not hasattr(detector_parameters, name):
            raise ValueError(f"unknown ArUco detector parameter {name!r}")
        setattr(detector_parameters, name, value)
    if hasattr(aruco, "CharucoDetector"):
        return board, aruco.CharucoDetector(board, aruco.CharucoParameters(), detector_parameters)
    return board, None


def detect_observations(config: dict[str, Any], config_dir: Path) -> tuple[list[Observation], dict[str, Any]]:
    dataset = config["dataset"]
    pattern = str(resolve_path(dataset["images"], config_dir))
    paths = [Path(item).resolve() for item in sorted(glob.glob(pattern, recursive=True))]
    if not paths:
        raise ValueError(f"dataset image pattern matched no files: {pattern}")

    target = config["target"]
    detection = config.get("detection", {})
    target_type = str(target["type"]).lower()
    minimum_points = int(target.get("minimum_points", 12))
    minimum_sharpness = detection.get("minimum_laplacian_variance")
    expected_size = tuple(int(item) for item in config["image_size"])
    group_regex = dataset.get("group_regex")
    charuco_board = charuco_detector = None
    if target_type == "charuco":
        charuco_board, charuco_detector = create_charuco_detector(target, detection)

    accepted: list[Observation] = []
    rejected: list[dict[str, Any]] = []
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            rejected.append({"path": str(path), "reason": "unreadable_image"})
            continue
        image_size = (int(image.shape[1]), int(image.shape[0]))
        if image_size != expected_size:
            rejected.append(
                {
                    "path": str(path),
                    "reason": "image_size_mismatch",
                    "actual": list(image_size),
                    "expected": list(expected_size),
                }
            )
            continue
        sharpness = float(cv2.Laplacian(image, cv2.CV_64F).var())
        if minimum_sharpness is not None and sharpness < float(minimum_sharpness):
            rejected.append(
                {"path": str(path), "reason": "below_sharpness_threshold", "sharpness": sharpness}
            )
            continue

        image_points: np.ndarray | None = None
        object_points: np.ndarray | None = None
        if target_type == "charuco":
            assert charuco_board is not None
            if charuco_detector is not None:
                corners, ids, _, _ = charuco_detector.detectBoard(image)
            else:
                marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(
                    image, charuco_board.getDictionary()
                )
                if marker_ids is None:
                    corners, ids = None, None
                else:
                    _, corners, ids = cv2.aruco.interpolateCornersCharuco(
                        marker_corners, marker_ids, image, charuco_board
                    )
            if ids is not None and corners is not None:
                flat_ids = np.asarray(ids).reshape(-1).astype(int)
                image_points = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
                chessboard_points = np.asarray(
                    charuco_board.getChessboardCorners(), dtype=np.float64
                ).reshape(-1, 3)
                object_points = chessboard_points[flat_ids]
        elif target_type == "chessboard":
            columns = int(target["columns"])
            rows = int(target["rows"])
            flags = cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE
            found, corners = cv2.findChessboardCornersSB(image, (columns, rows), flags=flags)
            if found:
                image_points = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
                spacing = float(target["square_length_m"])
                object_points = np.zeros((rows * columns, 3), dtype=np.float64)
                object_points[:, :2] = (
                    np.mgrid[0:columns, 0:rows].T.reshape(-1, 2).astype(np.float64) * spacing
                )
        elif target_type in {"circles", "asymmetric_circles"}:
            columns = int(target["columns"])
            rows = int(target["rows"])
            flags = (
                cv2.CALIB_CB_ASYMMETRIC_GRID
                if target_type == "asymmetric_circles"
                else cv2.CALIB_CB_SYMMETRIC_GRID
            )
            found, centers = cv2.findCirclesGrid(image, (columns, rows), flags=flags)
            if found:
                image_points = np.asarray(centers, dtype=np.float64).reshape(-1, 2)
                spacing = float(target["spacing_m"])
                object_points = np.zeros((rows * columns, 3), dtype=np.float64)
                index = 0
                for row in range(rows):
                    for column in range(columns):
                        x = (2 * column + (row % 2)) if target_type == "asymmetric_circles" else column
                        object_points[index, :2] = (x * spacing, row * spacing)
                        index += 1
        else:
            raise ValueError(f"unsupported target type {target_type!r}")

        if image_points is None or object_points is None or len(image_points) < minimum_points:
            rejected.append(
                {
                    "path": str(path),
                    "reason": "target_not_detected_or_too_few_points",
                    "detected_points": 0 if image_points is None else len(image_points),
                    "minimum_points": minimum_points,
                }
            )
            continue
        accepted.append(
            Observation(
                view_id=path.stem,
                path=str(path),
                group=_group_for_path(path, group_regex),
                image_size=image_size,
                object_points=object_points,
                image_points=image_points,
                sharpness=sharpness,
                coverage_fraction=_coverage_fraction(image_points, image_size),
            )
        )
    return accepted, {
        "matched_images": len(paths),
        "accepted_views": len(accepted),
        "rejected_views": len(rejected),
        "rejections": rejected,
    }


def save_observations(path: Path, observations: Sequence[Observation]) -> None:
    if not observations:
        raise ValueError("cannot save an empty observation set")
    write_json(
        path,
        {
            "schema_version": SCHEMA_VERSION,
            "image_size": list(observations[0].image_size),
            "views": [observation.to_json() for observation in observations],
        },
    )


def load_observations(path: Path) -> list[Observation]:
    root = read_json(path)
    if int(root.get("schema_version", 0)) != SCHEMA_VERSION:
        raise ValueError(f"unsupported observation schema in {path}")
    observations = [Observation.from_json(value) for value in root["views"]]
    if not observations:
        raise ValueError(f"observation file is empty: {path}")
    expected = observations[0].image_size
    if any(observation.image_size != expected for observation in observations):
        raise ValueError("all observations must use the same image size")
    return observations


def split_observations(
    observations: Sequence[Observation], split: dict[str, Any]
) -> tuple[list[Observation], list[Observation], dict[str, str]]:
    """Coverage-stratified, group-safe train/validation split."""
    fraction = float(split.get("validation_fraction", 0.25))
    if not 0.0 < fraction < 1.0:
        raise ValueError("split.validation_fraction must be between zero and one")
    seed = int(split.get("seed", 0))
    bins_x, bins_y = (int(item) for item in split.get("spatial_bins", [4, 3]))
    width, height = observations[0].image_size

    by_group: dict[str, list[Observation]] = defaultdict(list)
    for observation in observations:
        by_group[observation.group].append(observation)
    if len(by_group) < 4:
        raise ValueError("at least four independent groups are required for a held-out split")

    strata: dict[tuple[int, int], list[str]] = defaultdict(list)
    for group, group_views in by_group.items():
        centers = [np.mean(view.image_points, axis=0) for view in group_views]
        center = np.mean(centers, axis=0)
        bx = min(bins_x - 1, max(0, int(center[0] / width * bins_x)))
        by = min(bins_y - 1, max(0, int(center[1] / height * bins_y)))
        strata[(bx, by)].append(group)

    rng = np.random.default_rng(seed)
    validation_groups: set[str] = set()
    for groups in strata.values():
        shuffled = list(groups)
        rng.shuffle(shuffled)
        count = int(round(len(shuffled) * fraction))
        if len(shuffled) >= 2:
            count = max(1, min(len(shuffled) - 1, count))
        else:
            count = 0
        validation_groups.update(shuffled[:count])

    target_count = max(1, int(round(len(by_group) * fraction)))
    all_groups = list(by_group)
    rng.shuffle(all_groups)
    for group in all_groups:
        if len(validation_groups) >= target_count:
            break
        validation_groups.add(group)
    if len(validation_groups) >= len(by_group):
        validation_groups.remove(sorted(validation_groups)[-1])

    assignment = {
        group: ("validation" if group in validation_groups else "training") for group in by_group
    }
    training = [item for item in observations if assignment[item.group] == "training"]
    validation = [item for item in observations if assignment[item.group] == "validation"]
    return training, validation, assignment


def _opencv_flags(names: Sequence[str]) -> int:
    mapping = {
        "rational": cv2.CALIB_RATIONAL_MODEL,
        "thin_prism": cv2.CALIB_THIN_PRISM_MODEL,
        "tilted_sensor": cv2.CALIB_TILTED_MODEL,
        "zero_tangent": cv2.CALIB_ZERO_TANGENT_DIST,
        "fix_k1": cv2.CALIB_FIX_K1,
        "fix_k2": cv2.CALIB_FIX_K2,
        "fix_k3": cv2.CALIB_FIX_K3,
        "fix_k4": cv2.CALIB_FIX_K4,
        "fix_k5": cv2.CALIB_FIX_K5,
        "fix_k6": cv2.CALIB_FIX_K6,
        "fix_principal_point": cv2.CALIB_FIX_PRINCIPAL_POINT,
        "fix_aspect_ratio": cv2.CALIB_FIX_ASPECT_RATIO,
    }
    flags = 0
    for name in names:
        if name not in mapping:
            raise ValueError(f"unknown OpenCV calibration flag {name!r}")
        flags |= mapping[name]
    return flags


def _distortion_length(flags: Sequence[str]) -> int:
    if "tilted_sensor" in flags:
        return 14
    if "thin_prism" in flags:
        return 12
    if "rational" in flags:
        return 8
    return 5


def calibrate_model(
    observations: Sequence[Observation], model: dict[str, Any]
) -> ModelFit:
    if len(observations) < int(model.get("minimum_views", 8)):
        raise ValueError(
            f"model {model['name']} needs at least {model.get('minimum_views', 8)} training views"
        )
    image_size = observations[0].image_size
    object_points = [item.object_points.astype(np.float32) for item in observations]
    image_points = [item.image_points.astype(np.float32) for item in observations]
    model_type = str(model["type"]).lower()
    flag_names = [str(item) for item in model.get("flags", [])]

    if model_type == "opencv":
        flags = _opencv_flags(flag_names)
        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT,
            int(model.get("maximum_iterations", 100)),
            float(model.get("epsilon", 1e-12)),
        )
        result = cv2.calibrateCameraExtended(
            object_points,
            image_points,
            image_size,
            None,
            None,
            flags=flags,
            criteria=criteria,
        )
        rms, K, distortion, rvecs, tvecs, std_intrinsics, _, per_view = result
        length = _distortion_length(flag_names)
        distortion = np.asarray(distortion, dtype=np.float64).reshape(-1)[:length]
        return ModelFit(
            name=str(model["name"]),
            model_type=model_type,
            K=np.asarray(K, dtype=np.float64),
            distortion=distortion,
            rms=float(rms),
            rvecs=[np.asarray(item, dtype=np.float64).reshape(3) for item in rvecs],
            tvecs=[np.asarray(item, dtype=np.float64).reshape(3) for item in tvecs],
            per_view_errors=np.asarray(per_view).reshape(-1).astype(float).tolist(),
            intrinsic_stddev=np.asarray(std_intrinsics).reshape(-1).astype(float).tolist(),
            flags=flag_names,
            deployment_compatible=bool(model.get("deployment_compatible", True)),
            parameter_count=4 + length,
        )

    if model_type == "fisheye":
        fisheye_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv2.fisheye.CALIB_CHECK_COND
        if model.get("fix_skew", True):
            fisheye_flags |= cv2.fisheye.CALIB_FIX_SKEW
        fisheye_object = [item.reshape(-1, 1, 3) for item in object_points]
        fisheye_image = [item.reshape(-1, 1, 2) for item in image_points]
        K = np.eye(3, dtype=np.float64)
        K[0, 0] = K[1, 1] = max(image_size)
        K[0, 2] = image_size[0] / 2.0
        K[1, 2] = image_size[1] / 2.0
        distortion = np.zeros((4, 1), dtype=np.float64)
        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT,
            int(model.get("maximum_iterations", 100)),
            float(model.get("epsilon", 1e-10)),
        )
        rms, K, distortion, rvecs, tvecs = cv2.fisheye.calibrate(
            fisheye_object,
            fisheye_image,
            image_size,
            K,
            distortion,
            flags=fisheye_flags,
            criteria=criteria,
        )
        errors = []
        for points_3d, points_2d, rvec, tvec in zip(
            fisheye_object, fisheye_image, rvecs, tvecs
        ):
            projected, _ = cv2.fisheye.projectPoints(points_3d, rvec, tvec, K, distortion)
            error = projected.reshape(-1, 2) - points_2d.reshape(-1, 2)
            errors.append(float(np.sqrt(np.mean(np.sum(np.square(error), axis=1)))))
        return ModelFit(
            name=str(model["name"]),
            model_type=model_type,
            K=np.asarray(K, dtype=np.float64),
            distortion=np.asarray(distortion, dtype=np.float64).reshape(-1),
            rms=float(rms),
            rvecs=[np.asarray(item, dtype=np.float64).reshape(3) for item in rvecs],
            tvecs=[np.asarray(item, dtype=np.float64).reshape(3) for item in tvecs],
            per_view_errors=errors,
            intrinsic_stddev=[],
            flags=flag_names,
            deployment_compatible=bool(model.get("deployment_compatible", False)),
            parameter_count=8,
        )
    raise ValueError(f"unknown model type {model_type!r}")


def project_points(
    object_points: np.ndarray, rvec: np.ndarray, tvec: np.ndarray, fit: ModelFit
) -> tuple[np.ndarray, np.ndarray | None]:
    if fit.model_type == "fisheye":
        projected, jacobian = cv2.fisheye.projectPoints(
            object_points.astype(np.float64).reshape(-1, 1, 3),
            np.asarray(rvec, dtype=np.float64).reshape(3, 1),
            np.asarray(tvec, dtype=np.float64).reshape(3, 1),
            fit.K,
            fit.distortion.reshape(-1, 1),
        )
    else:
        projected, jacobian = cv2.projectPoints(
            object_points.astype(np.float64),
            np.asarray(rvec, dtype=np.float64).reshape(3, 1),
            np.asarray(tvec, dtype=np.float64).reshape(3, 1),
            fit.K,
            fit.distortion,
        )
    return projected.reshape(-1, 2), jacobian


def undistort_points(image_points: np.ndarray, fit: ModelFit) -> np.ndarray:
    points = image_points.astype(np.float64).reshape(-1, 1, 2)
    if fit.model_type == "fisheye":
        result = cv2.fisheye.undistortPoints(points, fit.K, fit.distortion.reshape(-1, 1))
    else:
        result = cv2.undistortPoints(points, fit.K, fit.distortion)
    return result.reshape(-1, 2)


def solve_pose(
    object_points: np.ndarray,
    image_points: np.ndarray,
    fit: ModelFit,
    method: str = "sqpnp",
    refine_lm: bool = True,
) -> tuple[np.ndarray, np.ndarray] | None:
    if len(object_points) < 4:
        return None
    methods = {
        "sqpnp": cv2.SOLVEPNP_SQPNP,
        "iterative": cv2.SOLVEPNP_ITERATIVE,
        "epnp": cv2.SOLVEPNP_EPNP,
        "ippe": cv2.SOLVEPNP_IPPE,
        "ap3p": cv2.SOLVEPNP_AP3P,
    }
    method = method.lower()
    if method not in methods:
        raise ValueError(f"unsupported PnP method {method!r}")
    if method == "ap3p" and len(object_points) != 4:
        return None
    camera_matrix = fit.K
    distortion: np.ndarray | None = fit.distortion
    solve_image_points = image_points.astype(np.float64)
    try:
        if fit.model_type == "fisheye":
            solve_image_points = undistort_points(image_points, fit)
            camera_matrix = np.eye(3, dtype=np.float64)
            distortion = None
        generic = cv2.solvePnPGeneric(
            object_points.astype(np.float64),
            solve_image_points,
            camera_matrix,
            distortion,
            flags=methods[method],
        )
        solution_count, rvecs, tvecs = generic[:3]
    except cv2.error:
        return None
    if solution_count <= 0:
        return None
    best: tuple[float, np.ndarray, np.ndarray] | None = None
    for candidate_rvec, candidate_tvec in zip(rvecs, tvecs):
        rvec = np.asarray(candidate_rvec, dtype=np.float64).reshape(3, 1)
        tvec = np.asarray(candidate_tvec, dtype=np.float64).reshape(3, 1)
        try:
            if refine_lm:
                rvec, tvec = cv2.solvePnPRefineLM(
                    object_points.astype(np.float64),
                    solve_image_points,
                    camera_matrix,
                    distortion,
                    rvec,
                    tvec,
                )
            rotation, _ = cv2.Rodrigues(rvec)
            camera_points = (rotation @ object_points.astype(np.float64).T + tvec).T
            if not np.all(np.isfinite(camera_points)) or np.any(camera_points[:, 2] <= 0.0):
                continue
            if fit.model_type == "fisheye":
                projected, _ = cv2.projectPoints(
                    object_points.astype(np.float64),
                    rvec,
                    tvec,
                    camera_matrix,
                    None,
                )
            else:
                projected, _ = project_points(object_points, rvec, tvec, fit)
            error = solve_image_points - projected.reshape(-1, 2)
            rms = float(np.sqrt(np.mean(np.sum(np.square(error), axis=1))))
            if best is None or rms < best[0]:
                best = (rms, rvec.reshape(3), tvec.reshape(3))
        except cv2.error:
            continue
    if best is None:
        return None
    return best[1], best[2]


def _anchor_mask(
    observation: Observation, fit: ModelFit, validation: dict[str, Any]
) -> tuple[np.ndarray, bool]:
    anchor = validation.get("pose_anchor", {})
    if anchor.get("mode", "central_radius") == "all":
        return np.ones(len(observation.image_points), dtype=bool), False
    maximum_radius = float(anchor.get("maximum_normalized_radius", 0.45))
    normalized_pixel = np.column_stack(
        [
            (observation.image_points[:, 0] - fit.K[0, 2]) / fit.K[0, 0],
            (observation.image_points[:, 1] - fit.K[1, 2]) / fit.K[1, 1],
        ]
    )
    mask = np.linalg.norm(normalized_pixel, axis=1) <= maximum_radius
    minimum = int(anchor.get("minimum_points", 6))
    if int(np.sum(mask)) < minimum and anchor.get("fallback", "all") == "all":
        return np.ones(len(observation.image_points), dtype=bool), True
    return mask, False


def evaluate_model(
    observations: Sequence[Observation], fit: ModelFit, validation: dict[str, Any]
) -> tuple[dict[str, Any], list[ResidualRecord], dict[str, tuple[np.ndarray, np.ndarray]]]:
    records: list[ResidualRecord] = []
    reference_poses: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    per_view: list[dict[str, Any]] = []
    anchor_fallbacks = 0
    pose_method = str(validation.get("pose_solver", "sqpnp"))
    refine_lm = bool(validation.get("refine_lm", True))
    for observation in observations:
        mask, used_fallback = _anchor_mask(observation, fit, validation)
        if used_fallback:
            anchor_fallbacks += 1
        pose = solve_pose(
            observation.object_points[mask],
            observation.image_points[mask],
            fit,
            pose_method,
            refine_lm,
        )
        if pose is None:
            per_view.append({"view_id": observation.view_id, "solved": False})
            continue
        rvec, tvec = pose
        reference_poses[observation.view_id] = pose
        projected, _ = project_points(observation.object_points, rvec, tvec, fit)
        residual = observation.image_points - projected
        norms = np.linalg.norm(residual, axis=1)
        for point, error, norm in zip(observation.image_points, residual, norms):
            records.append(
                ResidualRecord(
                    view_id=observation.view_id,
                    group=observation.group,
                    observed_x=float(point[0]),
                    observed_y=float(point[1]),
                    dx=float(error[0]),
                    dy=float(error[1]),
                    norm=float(norm),
                )
            )
        per_view.append(
            {
                "view_id": observation.view_id,
                "solved": True,
                "anchor_points": int(np.sum(mask)),
                "total_points": len(mask),
                "residual_px": series_stats(norms),
            }
        )
    image_size = observations[0].image_size
    spatial = spatial_residual_map(records, image_size, validation)
    regions = horizontal_region_metrics(records, image_size, validation)
    return (
        {
            "views": len(observations),
            "solved_views": len(reference_poses),
            "pose_anchor_all_point_fallbacks": anchor_fallbacks,
            "residual_px": series_stats(record.norm for record in records),
            "signed_residual_dx_px": series_stats(record.dx for record in records),
            "signed_residual_dy_px": series_stats(record.dy for record in records),
            "spatial_bins": spatial,
            "horizontal_regions": regions,
            "per_view": per_view,
        },
        records,
        reference_poses,
    )


def spatial_residual_map(
    records: Sequence[ResidualRecord], image_size: tuple[int, int], validation: dict[str, Any]
) -> dict[str, Any]:
    bins_x, bins_y = (int(item) for item in validation.get("spatial_bins", [8, 5]))
    minimum = int(validation.get("minimum_points_per_spatial_bin", 20))
    width, height = image_size
    cells: list[dict[str, Any]] = []
    valid_p95: list[float] = []
    for by in range(bins_y):
        for bx in range(bins_x):
            x0, x1 = width * bx / bins_x, width * (bx + 1) / bins_x
            y0, y1 = height * by / bins_y, height * (by + 1) / bins_y
            selected = [
                item
                for item in records
                if x0 <= item.observed_x < x1 and y0 <= item.observed_y < y1
            ]
            cell = {
                "bin_x": bx,
                "bin_y": by,
                "bounds_px": [x0, y0, x1, y1],
                "samples": len(selected),
                "mean_vector_px": (
                    [float(np.mean([item.dx for item in selected])), float(np.mean([item.dy for item in selected]))]
                    if selected
                    else None
                ),
                "residual_px": series_stats(item.norm for item in selected),
                "eligible_for_worst_bin": len(selected) >= minimum,
            }
            if len(selected) >= minimum:
                valid_p95.append(cell["residual_px"]["p95"])
            cells.append(cell)
    return {
        "shape": [bins_x, bins_y],
        "minimum_points_per_bin": minimum,
        "eligible_bins": len(valid_p95),
        "eligible_bin_fraction": len(valid_p95) / (bins_x * bins_y),
        "worst_eligible_bin_p95_px": max(valid_p95) if valid_p95 else None,
        "cells": cells,
    }


def horizontal_region_metrics(
    records: Sequence[ResidualRecord], image_size: tuple[int, int], validation: dict[str, Any]
) -> dict[str, Any]:
    fraction = float(validation.get("horizontal_edge_fraction", 0.2))
    width, _ = image_size
    boundaries = [width * fraction, width * (1.0 - fraction)]
    regions = {
        "left": [item for item in records if item.observed_x < boundaries[0]],
        "center": [item for item in records if boundaries[0] <= item.observed_x < boundaries[1]],
        "right": [item for item in records if item.observed_x >= boundaries[1]],
    }
    return {
        "edge_fraction": fraction,
        "boundaries_px": boundaries,
        **{name: {"residual_px": series_stats(item.norm for item in values)} for name, values in regions.items()},
    }


def _pixel_grid(image_size: tuple[int, int], roi: Sequence[int] | None, shape: Sequence[int]) -> np.ndarray:
    width, height = image_size
    if roi is None:
        x0, y0, x1, y1 = 0.0, 0.0, width - 1.0, height - 1.0
    else:
        x, y, roi_width, roi_height = (int(item) for item in roi)
        x0, y0, x1, y1 = float(x), float(y), float(x + roi_width - 1), float(y + roi_height - 1)
    nx, ny = int(shape[0]), int(shape[1])
    xs = np.linspace(x0, x1, nx)
    ys = np.linspace(y0, y1, ny)
    return np.stack(np.meshgrid(xs, ys), axis=-1)


def mapping_validity(
    fit: ModelFit,
    image_size: tuple[int, int],
    analysis: dict[str, Any],
    roi: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Check that pixel-to-ray inversion is finite, accurate and orientation preserving."""
    shape = analysis.get("mapping_grid", [41, 27])
    grid = _pixel_grid(image_size, roi, shape)
    pixels = grid.reshape(-1, 2)
    try:
        rays = undistort_points(pixels, fit)
        object_points = np.column_stack([rays, np.ones(len(rays))])
        round_trip, _ = project_points(object_points, np.zeros(3), np.zeros(3), fit)
        error = np.linalg.norm(round_trip - pixels, axis=1)
    except (cv2.error, ValueError, FloatingPointError):
        return {"valid": False, "reason": "mapping_exception"}

    finite = bool(np.all(np.isfinite(rays)) and np.all(np.isfinite(error)))
    ny, nx = grid.shape[:2]
    ray_grid = rays.reshape(ny, nx, 2)
    du_dx = np.gradient(ray_grid[:, :, 0], axis=1)
    du_dy = np.gradient(ray_grid[:, :, 0], axis=0)
    dv_dx = np.gradient(ray_grid[:, :, 1], axis=1)
    dv_dy = np.gradient(ray_grid[:, :, 1], axis=0)
    determinant = du_dx * dv_dy - du_dy * dv_dx
    tolerance = float(analysis.get("maximum_round_trip_error_px", 0.05))
    minimum_determinant = float(analysis.get("minimum_inverse_jacobian_determinant", 1e-12))
    positive = bool(np.all(np.isfinite(determinant)) and np.min(determinant) > minimum_determinant)
    maximum_error = float(np.max(error)) if finite else None
    return {
        "valid": bool(finite and positive and maximum_error is not None and maximum_error <= tolerance),
        "finite": finite,
        "orientation_preserving": positive,
        "maximum_round_trip_error_px": maximum_error,
        "p95_round_trip_error_px": float(np.quantile(error, 0.95)) if finite else None,
        "minimum_inverse_jacobian_determinant": (
            float(np.min(determinant)) if np.all(np.isfinite(determinant)) else None
        ),
        "thresholds": {
            "maximum_round_trip_error_px": tolerance,
            "minimum_inverse_jacobian_determinant": minimum_determinant,
        },
        "grid_shape": [int(shape[0]), int(shape[1])],
        "roi": list(roi) if roi is not None else [0, 0, image_size[0], image_size[1]],
    }


def radial_monotonicity(fit: ModelFit, maximum_radius: float = 5.0) -> dict[str, Any]:
    if fit.model_type != "opencv":
        return {"applicable": False, "reason": "not_an_opencv_radial_model"}
    d = np.zeros(8, dtype=np.float64)
    flat = fit.distortion.reshape(-1)
    d[: min(len(flat), len(d))] = flat[:8]
    k1, k2, _, _, k3, k4, k5, k6 = d

    def value(radius: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        r2 = radius * radius
        numerator = 1.0 + k1 * r2 + k2 * r2**2 + k3 * r2**3
        denominator = 1.0 + k4 * r2 + k5 * r2**2 + k6 * r2**3
        nprime = 2 * k1 * radius + 4 * k2 * radius**3 + 6 * k3 * radius**5
        dprime = 2 * k4 * radius + 4 * k5 * radius**3 + 6 * k6 * radius**5
        radial = radius * numerator / denominator
        derivative = numerator / denominator + radius * (
            nprime * denominator - numerator * dprime
        ) / np.square(denominator)
        return radial, derivative

    radii = np.linspace(0.0, float(maximum_radius), 20001)
    mapped, derivative = value(radii)
    valid = np.isfinite(mapped) & np.isfinite(derivative) & (derivative > 0.0)
    first_bad = np.flatnonzero(~valid)
    if first_bad.size == 0:
        return {
            "applicable": True,
            "monotonic_to_test_limit": True,
            "tested_undistorted_radius": float(maximum_radius),
        }
    index = int(first_bad[0])
    low = radii[max(0, index - 1)]
    high = radii[index]
    for _ in range(60):
        middle = (low + high) / 2.0
        mapped_middle, derivative_middle = value(np.asarray([middle]))
        if math.isfinite(mapped_middle[0]) and math.isfinite(derivative_middle[0]) and derivative_middle[0] > 0:
            low = middle
        else:
            high = middle
    mapped_limit, _ = value(np.asarray([low]))
    return {
        "applicable": True,
        "monotonic_to_test_limit": False,
        "maximum_monotonic_undistorted_radius": float(low),
        "maximum_monotonic_distorted_radius": float(mapped_limit[0]),
    }


def bootstrap_parameter_stability(
    training: Sequence[Observation], model: dict[str, Any], bootstrap: dict[str, Any]
) -> dict[str, Any]:
    iterations = int(bootstrap.get("iterations", 0))
    if iterations <= 0:
        return {"iterations_requested": iterations, "iterations_succeeded": 0, "enabled": False}
    seed = int(bootstrap.get("seed", 0))
    rng = np.random.default_rng(seed)
    samples: list[np.ndarray] = []
    failures: list[str] = []
    groups: dict[str, list[Observation]] = defaultdict(list)
    for observation in training:
        groups[observation.group].append(observation)
    group_names = sorted(groups)
    for _ in range(iterations):
        selected_names = rng.choice(group_names, size=len(group_names), replace=True)
        selected = [view for name in selected_names for view in groups[str(name)]]
        try:
            fit = calibrate_model(selected, model)
            samples.append(
                np.concatenate(
                    [
                        np.asarray([fit.K[0, 0], fit.K[1, 1], fit.K[0, 2], fit.K[1, 2]]),
                        fit.distortion.reshape(-1),
                    ]
                )
            )
        except (cv2.error, ValueError) as error:
            failures.append(str(error))
    if not samples:
        return {
            "enabled": True,
            "iterations_requested": iterations,
            "iterations_succeeded": 0,
            "failure_count": len(failures),
            "failure_examples": failures[:3],
        }
    matrix = np.vstack(samples)
    names = ["fx", "fy", "cx", "cy"] + [f"d{index}" for index in range(matrix.shape[1] - 4)]
    parameters = {}
    for index, name in enumerate(names):
        column = matrix[:, index]
        parameters[name] = {
            "mean": float(np.mean(column)),
            "standard_deviation": float(np.std(column)),
            "p025": float(np.quantile(column, 0.025)),
            "p975": float(np.quantile(column, 0.975)),
        }
    return {
        "enabled": True,
        "iterations_requested": iterations,
        "iterations_succeeded": len(samples),
        "failure_count": len(failures),
        "parameters": parameters,
    }


def _camera_center(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    rotation, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    return (-rotation.T @ np.asarray(tvec, dtype=np.float64).reshape(3, 1)).reshape(3)


def _rotation_error_deg(first: np.ndarray, second: np.ndarray) -> float:
    first_rotation, _ = cv2.Rodrigues(np.asarray(first, dtype=np.float64).reshape(3, 1))
    second_rotation, _ = cv2.Rodrigues(np.asarray(second, dtype=np.float64).reshape(3, 1))
    relative = first_rotation @ second_rotation.T
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def _pose_conditioning(
    object_points: np.ndarray, rvec: np.ndarray, tvec: np.ndarray, fit: ModelFit
) -> tuple[float | None, float | None]:
    if fit.model_type != "opencv":
        return None, None
    _, jacobian = project_points(object_points, rvec, tvec, fit)
    if jacobian is None or jacobian.shape[1] < 6:
        return None, None
    pose_jacobian = np.asarray(jacobian[:, :6], dtype=np.float64)
    normal = pose_jacobian.T @ pose_jacobian
    try:
        condition = float(np.linalg.cond(normal))
        covariance = np.linalg.pinv(normal)
        sigma_tz_mm = float(math.sqrt(max(0.0, covariance[5, 5])) * 1000.0)
        return condition, sigma_tz_mm
    except np.linalg.LinAlgError:
        return None, None


def crop_candidates(config: dict[str, Any], image_size: tuple[int, int]) -> list[list[int]]:
    width, height = image_size
    crop = config.get("crop_sweep", {})
    candidates: list[list[int]] = []
    for item in crop.get("horizontal_trims_px", []):
        if isinstance(item, int):
            left = right = item
        else:
            left, right = (int(value) for value in item)
        roi_width = width - left - right
        if left < 0 or right < 0 or roi_width <= 0:
            raise ValueError(f"invalid horizontal trim candidate {item!r}")
        candidates.append([left, 0, roi_width, height])
    for roi in crop.get("explicit_rois", []):
        x, y, roi_width, roi_height = (int(value) for value in roi)
        if x < 0 or y < 0 or roi_width <= 0 or roi_height <= 0:
            raise ValueError(f"invalid ROI {roi!r}")
        if x + roi_width > width or y + roi_height > height:
            raise ValueError(f"ROI is outside the image: {roi!r}")
        candidates.append([x, y, roi_width, roi_height])
    if not candidates:
        candidates.append([0, 0, width, height])
    unique: list[list[int]] = []
    for roi in candidates:
        if roi not in unique:
            unique.append(roi)
    return unique


def _inside_roi(points: np.ndarray, roi: Sequence[int]) -> np.ndarray:
    x, y, width, height = (int(item) for item in roi)
    return (
        (points[:, 0] >= x)
        & (points[:, 0] < x + width)
        & (points[:, 1] >= y)
        & (points[:, 1] < y + height)
    )


def _pose_records_for_crop(
    observations: Sequence[Observation],
    fit: ModelFit,
    roi: Sequence[int],
    references: dict[str, tuple[np.ndarray, np.ndarray]],
    minimum_points: int,
    pose_method: str,
    refine_lm: bool,
) -> list[PoseRecord]:
    records: list[PoseRecord] = []
    for observation in observations:
        mask = _inside_roi(observation.image_points, roi)
        retained = int(np.sum(mask))
        if retained < minimum_points or observation.view_id not in references:
            records.append(
                PoseRecord(observation.view_id, observation.group, False, retained_points=retained)
            )
            continue
        pose = solve_pose(
            observation.object_points[mask],
            observation.image_points[mask],
            fit,
            pose_method,
            refine_lm,
        )
        if pose is None:
            records.append(
                PoseRecord(observation.view_id, observation.group, False, retained_points=retained)
            )
            continue
        rvec, tvec = pose
        reference_rvec, reference_tvec = references[observation.view_id]
        camera = _camera_center(rvec, tvec)
        reference_camera = _camera_center(reference_rvec, reference_tvec)
        projected, _ = project_points(observation.object_points[mask], rvec, tvec, fit)
        error = observation.image_points[mask] - projected
        condition, sigma_tz = _pose_conditioning(observation.object_points[mask], rvec, tvec, fit)
        records.append(
            PoseRecord(
                observation.view_id,
                observation.group,
                True,
                retained_points=retained,
                camera_z_error_m=float(camera[2] - reference_camera[2]),
                camera_position_error_m=float(np.linalg.norm(camera - reference_camera)),
                rotation_error_deg=_rotation_error_deg(rvec, reference_rvec),
                reprojection_rms_px=float(np.sqrt(np.mean(np.sum(np.square(error), axis=1)))),
                normal_matrix_condition=condition,
                tvec_z_sigma_mm_per_px=sigma_tz,
            )
        )
    return records


def load_operational_validation(path: Path, image_size: tuple[int, int]) -> list[dict[str, Any]]:
    root = read_json(path)
    if int(root.get("schema_version", 0)) != SCHEMA_VERSION:
        raise ValueError(f"unsupported operational validation schema in {path}")
    if tuple(int(item) for item in root["image_size"]) != image_size:
        raise ValueError("operational validation image size does not match calibration data")
    frames = []
    for value in root["frames"]:
        object_points = np.asarray(value["object_points_m"], dtype=np.float64).reshape(-1, 3)
        image_points = np.asarray(value["image_points_px"], dtype=np.float64).reshape(-1, 2)
        if len(object_points) != len(image_points):
            raise ValueError(f"operational frame {value.get('frame_id')} has unequal point counts")
        frames.append(
            {
                "frame_id": str(value["frame_id"]),
                "group": str(value.get("group", value["frame_id"])),
                "object_points": object_points,
                "image_points": image_points,
                "reference_rvec": np.asarray(value["reference_rvec"], dtype=np.float64).reshape(3),
                "reference_tvec_m": np.asarray(value["reference_tvec_m"], dtype=np.float64).reshape(3),
            }
        )
    return frames


def _operational_records_for_crop(
    frames: Sequence[dict[str, Any]],
    fit: ModelFit,
    roi: Sequence[int],
    minimum_points: int,
    pose_method: str,
    refine_lm: bool,
) -> list[PoseRecord]:
    records: list[PoseRecord] = []
    for frame in frames:
        mask = _inside_roi(frame["image_points"], roi)
        retained = int(np.sum(mask))
        if retained < minimum_points:
            records.append(PoseRecord(frame["frame_id"], frame["group"], False, retained))
            continue
        pose = solve_pose(
            frame["object_points"][mask],
            frame["image_points"][mask],
            fit,
            pose_method,
            refine_lm,
        )
        if pose is None:
            records.append(PoseRecord(frame["frame_id"], frame["group"], False, retained))
            continue
        rvec, tvec = pose
        camera = _camera_center(rvec, tvec)
        reference_camera = _camera_center(frame["reference_rvec"], frame["reference_tvec_m"])
        projected, _ = project_points(frame["object_points"][mask], rvec, tvec, fit)
        residual = frame["image_points"][mask] - projected
        condition, sigma_tz = _pose_conditioning(frame["object_points"][mask], rvec, tvec, fit)
        records.append(
            PoseRecord(
                frame["frame_id"],
                frame["group"],
                True,
                retained,
                camera_z_error_m=float(camera[2] - reference_camera[2]),
                camera_position_error_m=float(np.linalg.norm(camera - reference_camera)),
                rotation_error_deg=_rotation_error_deg(rvec, frame["reference_rvec"]),
                reprojection_rms_px=float(np.sqrt(np.mean(np.sum(np.square(residual), axis=1)))),
                normal_matrix_condition=condition,
                tvec_z_sigma_mm_per_px=sigma_tz,
            )
        )
    return records


def _block_bootstrap_bounds(
    values_by_group: dict[str, list[float]], metric: str, iterations: int, seed: int
) -> dict[str, Any] | None:
    if iterations <= 0 or not values_by_group:
        return None
    groups = sorted(values_by_group)
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(iterations):
        selected = rng.choice(groups, size=len(groups), replace=True)
        values = np.asarray([value for group in selected for value in values_by_group[str(group)]])
        if values.size == 0:
            continue
        if metric == "p95":
            samples.append(float(np.quantile(values, 0.95)))
        elif metric == "rmse":
            samples.append(float(np.sqrt(np.mean(np.square(values)))))
        else:
            raise ValueError(metric)
    if not samples:
        return None
    return {
        "lower95": float(np.quantile(samples, 0.025)),
        "upper95": float(np.quantile(samples, 0.975)),
        "iterations": len(samples),
    }


def _availability_bootstrap(
    records: Sequence[PoseRecord], iterations: int, seed: int
) -> dict[str, Any] | None:
    if iterations <= 0 or not records:
        return None
    by_group: dict[str, list[float]] = defaultdict(list)
    for item in records:
        by_group[item.group].append(1.0 if item.solved else 0.0)
    groups = sorted(by_group)
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(iterations):
        selected = rng.choice(groups, size=len(groups), replace=True)
        values = [value for group in selected for value in by_group[str(group)]]
        if values:
            samples.append(float(np.mean(values)))
    if not samples:
        return None
    return {
        "lower95": float(np.quantile(samples, 0.025)),
        "upper95": float(np.quantile(samples, 0.975)),
        "iterations": len(samples),
    }


def summarize_pose_records(
    records: Sequence[PoseRecord], confidence: dict[str, Any]
) -> dict[str, Any]:
    solved = [item for item in records if item.solved]
    total = len(records)
    availability = len(solved) / total if total else 0.0
    z_mm = [float(item.camera_z_error_m) * 1000.0 for item in solved]
    position_mm = [float(item.camera_position_error_m) * 1000.0 for item in solved]
    by_group: dict[str, list[float]] = defaultdict(list)
    for item in solved:
        by_group[item.group].append(abs(float(item.camera_z_error_m) * 1000.0))
    iterations = int(confidence.get("iterations", 0))
    seed = int(confidence.get("seed", 0))
    return {
        "frames": total,
        "solved_frames": len(solved),
        "availability_fraction": availability,
        "availability_bootstrap": _availability_bootstrap(records, iterations, seed + 3),
        "retained_points": series_stats(item.retained_points for item in records),
        "camera_z_error_mm": series_stats(z_mm),
        "absolute_camera_z_error_mm": series_stats(abs(value) for value in z_mm),
        "camera_position_error_mm": series_stats(position_mm),
        "rotation_error_deg": series_stats(float(item.rotation_error_deg) for item in solved),
        "reprojection_rms_px": series_stats(float(item.reprojection_rms_px) for item in solved),
        "normal_matrix_condition": series_stats(
            float(item.normal_matrix_condition)
            for item in solved
            if item.normal_matrix_condition is not None
        ),
        "tvec_z_sigma_mm_per_px": series_stats(
            float(item.tvec_z_sigma_mm_per_px)
            for item in solved
            if item.tvec_z_sigma_mm_per_px is not None
        ),
        "absolute_camera_z_p95_bootstrap": _block_bootstrap_bounds(
            by_group, "p95", iterations, seed
        ),
        "camera_z_rmse_bootstrap": _block_bootstrap_bounds(by_group, "rmse", iterations, seed + 1),
    }


def _residuals_in_roi(records: Sequence[ResidualRecord], roi: Sequence[int]) -> list[ResidualRecord]:
    x, y, width, height = (int(item) for item in roi)
    return [
        item
        for item in records
        if x <= item.observed_x < x + width and y <= item.observed_y < y + height
    ]


def residual_coverage_in_roi(
    records: Sequence[ResidualRecord], roi: Sequence[int], crop_config: dict[str, Any]
) -> dict[str, Any]:
    bins_x, bins_y = (int(item) for item in crop_config.get("spatial_bins", [8, 5]))
    minimum = int(crop_config.get("minimum_points_per_spatial_bin", 10))
    x0, y0, width, height = (int(item) for item in roi)
    counts = np.zeros((bins_y, bins_x), dtype=np.int64)
    for item in records:
        if not (x0 <= item.observed_x < x0 + width and y0 <= item.observed_y < y0 + height):
            continue
        bx = min(bins_x - 1, int((item.observed_x - x0) / width * bins_x))
        by = min(bins_y - 1, int((item.observed_y - y0) / height * bins_y))
        counts[by, bx] += 1
    eligible = counts >= minimum
    return {
        "shape": [bins_x, bins_y],
        "minimum_points_per_bin": minimum,
        "counts": counts.tolist(),
        "eligible_bins": int(np.sum(eligible)),
        "eligible_bin_fraction": float(np.mean(eligible)),
    }


def _residual_bootstrap(
    records: Sequence[ResidualRecord], confidence: dict[str, Any]
) -> dict[str, Any] | None:
    by_group: dict[str, list[float]] = defaultdict(list)
    for item in records:
        by_group[item.group].append(item.norm)
    return _block_bootstrap_bounds(
        by_group,
        "p95",
        int(confidence.get("iterations", 0)),
        int(confidence.get("seed", 0)) + 2,
    )


def passes_thresholds(
    candidate: dict[str, Any],
    thresholds: dict[str, Any],
    require_operational_validation: bool = True,
) -> tuple[bool, list[str]]:
    failures: list[str] = []
    if thresholds.get("require_mapping_valid", True) and not candidate["mapping_validity"]["valid"]:
        failures.append("mapping_invalid")

    residual_limit = thresholds.get("maximum_validation_p95_px")
    if residual_limit is not None:
        bounds = candidate.get("validation_residual_p95_bootstrap")
        value = bounds["upper95"] if bounds else candidate["validation_residual_px"].get("p95")
        if value is None or value > float(residual_limit):
            failures.append("validation_p95_px")

    availability_limit = thresholds.get("minimum_availability_fraction")
    pose = candidate.get("operational_pose") or candidate.get("calibration_pose_consistency")
    if availability_limit is not None:
        availability_bounds = pose.get("availability_bootstrap") if pose else None
        availability = (
            availability_bounds["lower95"]
            if availability_bounds
            else (pose["availability_fraction"] if pose else None)
        )
        if availability is None or availability < float(availability_limit):
            failures.append("availability_fraction")

    z_limit = thresholds.get("maximum_p95_absolute_z_error_mm")
    if z_limit is not None and require_operational_validation:
        if candidate.get("operational_pose") is None:
            failures.append("missing_operational_pose_validation")
        else:
            operational = candidate["operational_pose"]
            bounds = operational.get("absolute_camera_z_p95_bootstrap")
            value = (
                bounds["upper95"]
                if bounds
                else operational["absolute_camera_z_error_mm"].get("p95")
            )
            if value is None or value > float(z_limit):
                failures.append("p95_absolute_z_error_mm")

    condition_limit = thresholds.get("maximum_median_normal_matrix_condition")
    if condition_limit is not None:
        condition = pose["normal_matrix_condition"].get("median") if pose else None
        if condition is None or condition > float(condition_limit):
            failures.append("normal_matrix_condition")
    coverage_limit = thresholds.get("minimum_spatial_bin_coverage_fraction")
    if coverage_limit is not None:
        fraction = candidate["validation_spatial_coverage"]["eligible_bin_fraction"]
        if fraction < float(coverage_limit):
            failures.append("spatial_bin_coverage_fraction")
    return not failures, failures


def sweep_crops(
    observations: Sequence[Observation],
    fit: ModelFit,
    residuals: Sequence[ResidualRecord],
    reference_poses: dict[str, tuple[np.ndarray, np.ndarray]],
    config: dict[str, Any],
    operational_frames: Sequence[dict[str, Any]] | None,
    require_operational_validation: bool = True,
) -> list[dict[str, Any]]:
    image_size = observations[0].image_size
    crop_config = config.get("crop_sweep", {})
    minimum_points = int(crop_config.get("minimum_points_per_pose", 6))
    pose_method = str(crop_config.get("pose_solver", "sqpnp"))
    refine_lm = bool(crop_config.get("refine_lm", True))
    confidence = crop_config.get("confidence", {})
    thresholds = crop_config.get("thresholds", {})
    results = []
    for roi in crop_candidates(config, image_size):
        selected_residuals = _residuals_in_roi(residuals, roi)
        consistency_records = _pose_records_for_crop(
            observations,
            fit,
            roi,
            reference_poses,
            minimum_points,
            pose_method,
            refine_lm,
        )
        candidate = {
            "roi": roi,
            "retained_image_fraction": float(roi[2] * roi[3] / (image_size[0] * image_size[1])),
            "mapping_validity": mapping_validity(
                fit, image_size, config.get("model_analysis", {}), roi
            ),
            "validation_residual_px": series_stats(item.norm for item in selected_residuals),
            "validation_spatial_coverage": residual_coverage_in_roi(
                selected_residuals, roi, crop_config
            ),
            "validation_residual_p95_bootstrap": _residual_bootstrap(
                selected_residuals, confidence
            ),
            "calibration_pose_consistency": summarize_pose_records(
                consistency_records, confidence
            ),
            "operational_pose": None,
        }
        if operational_frames is not None:
            operational_records = _operational_records_for_crop(
                operational_frames,
                fit,
                roi,
                minimum_points,
                pose_method,
                refine_lm,
            )
            candidate["operational_pose"] = summarize_pose_records(
                operational_records, confidence
            )
        passed, failures = passes_thresholds(
            candidate,
            thresholds,
            require_operational_validation=require_operational_validation,
        )
        candidate["passes_thresholds"] = passed
        candidate["threshold_failures"] = failures
        results.append(candidate)
    return results


def choose_model(
    model_results: Sequence[dict[str, Any]], selection: dict[str, Any]
) -> dict[str, Any] | None:
    metric_name = str(selection.get("metric", "validation_p95_px"))
    tolerance = float(selection.get("simpler_model_tolerance_px", 0.02))
    require_mapping = bool(selection.get("require_full_frame_mapping_valid", True))
    minimum_coverage = float(selection.get("minimum_eligible_spatial_bin_fraction", 0.0))
    eligible = []
    for result in model_results:
        if result.get("status") != "ok":
            continue
        if require_mapping and not result["mapping_validity"]["valid"]:
            continue
        if (
            result["validation"]["spatial_bins"].get("eligible_bin_fraction", 0.0)
            < minimum_coverage
        ):
            continue
        if metric_name == "validation_p95_px":
            metric = result["validation"]["residual_px"].get("p95")
        elif metric_name == "worst_spatial_bin_p95_px":
            metric = result["validation"]["spatial_bins"].get("worst_eligible_bin_p95_px")
        else:
            raise ValueError(f"unknown selection metric {metric_name!r}")
        if metric is not None and math.isfinite(float(metric)):
            eligible.append((float(metric), int(result["calibration"]["parameter_count"]), result))
    if not eligible:
        return None
    eligible.sort(key=lambda item: (item[0], item[1]))
    best_metric = eligible[0][0]
    statistically_equivalent = [item for item in eligible if item[0] <= best_metric + tolerance]
    statistically_equivalent.sort(key=lambda item: (item[1], item[0]))
    metric, _, result = statistically_equivalent[0]
    return {
        "model": result["name"],
        "metric": metric_name,
        "metric_value_px": metric,
        "best_observed_metric_px": best_metric,
        "simpler_model_tolerance_px": tolerance,
        "reason": "simplest model within the configured tolerance of the best held-out metric",
    }


def choose_deployment(model_results: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = []
    for result in model_results:
        if result.get("status") != "ok" or not result["calibration"]["deployment_compatible"]:
            continue
        for crop in result.get("crop_sweep", []):
            if crop["passes_thresholds"]:
                residual = crop["validation_residual_px"].get("p95", math.inf)
                candidates.append(
                    (
                        -float(crop["retained_image_fraction"]),
                        float(residual),
                        int(result["calibration"]["parameter_count"]),
                        result,
                        crop,
                    )
                )
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[:3])
    _, _, _, result, crop = candidates[0]
    return {
        "model": result["name"],
        "roi": crop["roi"],
        "retained_image_fraction": crop["retained_image_fraction"],
        "validation_residual_p95_px": crop["validation_residual_px"].get("p95"),
        "operational_pose": crop.get("operational_pose"),
        "reason": "largest deployment-compatible ROI satisfying every configured threshold; ties use held-out accuracy and then model simplicity",
    }


def evaluate_operational_crop(
    frames: Sequence[dict[str, Any]],
    fit: ModelFit,
    roi: Sequence[int],
    minimum_points: int,
    confidence: dict[str, Any],
    pose_method: str = "sqpnp",
    refine_lm: bool = True,
) -> dict[str, Any]:
    return summarize_pose_records(
        _operational_records_for_crop(
            frames, fit, roi, minimum_points, pose_method, refine_lm
        ),
        confidence,
    )


def write_residual_csv(path: Path, records: Sequence[ResidualRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["view_id", "group", "observed_x", "observed_y", "dx", "dy", "norm"])
        for item in records:
            writer.writerow(
                [item.view_id, item.group, item.observed_x, item.observed_y, item.dx, item.dy, item.norm]
            )


def write_plots(
    output_dir: Path,
    observations: Sequence[Observation],
    model_results: Sequence[dict[str, Any]],
    residual_records: dict[str, Sequence[ResidualRecord]],
) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    output_dir.mkdir(parents=True, exist_ok=True)
    width, height = observations[0].image_size
    written: list[str] = []

    figure, axis = plt.subplots(figsize=(10, 6))
    for observation in observations:
        axis.scatter(observation.image_points[:, 0], observation.image_points[:, 1], s=2, alpha=0.18)
    axis.set_xlim(0, width)
    axis.set_ylim(height, 0)
    axis.set_aspect("equal")
    axis.set_title("Calibration observation coverage")
    axis.set_xlabel("image x [px]")
    axis.set_ylabel("image y [px]")
    path = output_dir / "observation_coverage.png"
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    written.append(str(path))

    for result in model_results:
        if result.get("status") != "ok":
            continue
        name = result["name"]
        records = residual_records.get(name, [])
        if not records:
            continue
        figure, axis = plt.subplots(figsize=(10, 6))
        x = np.asarray([item.observed_x for item in records])
        y = np.asarray([item.observed_y for item in records])
        norm = np.asarray([item.norm for item in records])
        scatter = axis.scatter(x, y, c=norm, s=7, cmap="viridis", vmin=0, vmax=max(0.25, np.quantile(norm, 0.99)))
        axis.set_xlim(0, width)
        axis.set_ylim(height, 0)
        axis.set_aspect("equal")
        axis.set_title(f"Held-out residual magnitude: {name}")
        axis.set_xlabel("image x [px]")
        axis.set_ylabel("image y [px]")
        figure.colorbar(scatter, ax=axis, label="residual [px]")
        path = output_dir / f"residuals_{name}.png"
        figure.tight_layout()
        figure.savefig(path, dpi=160)
        plt.close(figure)
        written.append(str(path))

        crops = result.get("crop_sweep", [])
        if crops:
            figure, primary = plt.subplots(figsize=(9, 5))
            trims = [crop["roi"][0] for crop in crops]
            p95 = [crop["validation_residual_px"].get("p95", np.nan) for crop in crops]
            availability = [
                (crop.get("operational_pose") or crop["calibration_pose_consistency"])[
                    "availability_fraction"
                ]
                for crop in crops
            ]
            primary.plot(trims, p95, marker="o", label="validation p95 residual")
            primary.set_xlabel("left trim [px]")
            primary.set_ylabel("p95 residual [px]")
            secondary = primary.twinx()
            secondary.plot(trims, availability, color="tab:orange", marker="s", label="availability")
            secondary.set_ylabel("availability")
            secondary.set_ylim(0, 1.05)
            primary.set_title(f"Crop trade-off: {name}")
            path = output_dir / f"crop_sweep_{name}.png"
            figure.tight_layout()
            figure.savefig(path, dpi=160)
            plt.close(figure)
            written.append(str(path))
    return written


def markdown_report(result: dict[str, Any]) -> str:
    operational = result["operational_validation"]
    lines = [
        "# Camera calibration report",
        "",
        f"Generated from `{result['config']}`.",
        "",
        "## Dataset",
        "",
        f"- Accepted observations: {result['dataset']['accepted_views']}",
        f"- Training observations: {result['dataset']['training_views']}",
        f"- Held-out observations: {result['dataset']['validation_views']}",
        f"- Image size: `{result['dataset']['image_size'][0]} x {result['dataset']['image_size'][1]}`",
        f"- Operational validation: {operational.get('mode', 'enabled')}",
        "",
        "## Model comparison",
        "",
        "| Model | Status | Train RMS px | Held-out p95 px | Worst bin p95 px | Full mapping valid | Deployable |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model in result["models"]:
        if model.get("status") != "ok":
            lines.append(f"| {model['name']} | failed | — | — | — | — | — |")
            continue
        lines.append(
            "| {name} | ok | {train:.4f} | {p95:.4f} | {worst} | {mapping} | {deployable} |".format(
                name=model["name"],
                train=model["calibration"]["training_rms_px"],
                p95=model["validation"]["residual_px"].get("p95", math.nan),
                worst=(
                    "—"
                    if model["validation"]["spatial_bins"]["worst_eligible_bin_p95_px"] is None
                    else f"{model['validation']['spatial_bins']['worst_eligible_bin_p95_px']:.4f}"
                ),
                mapping="yes" if model["mapping_validity"]["valid"] else "no",
                deployable="yes" if model["calibration"]["deployment_compatible"] else "no",
            )
        )
    lines.extend(["", "## Recommendations", ""])
    if result.get("model_recommendation"):
        recommendation = result["model_recommendation"]
        lines.append(
            f"Held-out model recommendation: **{recommendation['model']}** "
            f"({recommendation['metric']} = {recommendation['metric_value_px']:.4f} px)."
        )
    else:
        lines.append("No model met the configured full-frame eligibility rules.")
    lines.append("")
    if result.get("deployment_recommendation"):
        deployment = result["deployment_recommendation"]
        lines.append(
            f"Deployment recommendation: **{deployment['model']}**, ROI `{deployment['roi']}`, "
            f"retaining {100.0 * deployment['retained_image_fraction']:.1f}% of the image."
        )
    else:
        lines.append(
            "No model/ROI pair met every configured deployment threshold. Do not export a "
            "production calibration until the failed thresholds are addressed."
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Training RMS is diagnostic only. Selection uses held-out observations, spatial worst-bin behavior, "
            "mapping invertibility, crop availability, and—when supplied—independent operational reference poses.",
            "A calibration-board pose-consistency error is not a substitute for mocap or another independent pose reference.",
            "",
        ]
    )
    if operational.get("mode") == "skipped":
        lines.extend(
            [
                "**Operational validation was explicitly skipped.** The exported calibration passed the "
                "remaining geometric and held-out calibration-target checks, but its absolute task-space "
                "pose accuracy was not independently measured.",
                "",
            ]
        )
    return "\n".join(lines)
