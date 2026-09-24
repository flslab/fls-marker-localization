#!/usr/bin/env python3

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from calibration_lib import (  # noqa: E402
    ModelFit,
    Observation,
    calibrate_model,
    detect_observations,
    evaluate_model,
    mapping_validity,
    passes_thresholds,
    save_observations,
    split_observations,
)
from capture_session import CapturePolicy, ProductionSettings, TargetObservation  # noqa: E402


def make_fit(K: np.ndarray, distortion: np.ndarray) -> ModelFit:
    return ModelFit(
        name="test",
        model_type="opencv",
        K=K,
        distortion=distortion,
        rms=0.0,
        rvecs=[],
        tvecs=[],
        per_view_errors=[],
        intrinsic_stddev=[],
        flags=[],
        deployment_compatible=True,
        parameter_count=9,
    )


def synthetic_observations() -> tuple[list[Observation], np.ndarray]:
    rng = np.random.default_rng(8)
    K = np.asarray([[480.0, 0.0, 320.0], [0.0, 482.0, 200.0], [0.0, 0.0, 1.0]])
    distortion = np.asarray([0.08, -0.03, 0.001, -0.0005, 0.01])
    columns, rows = 9, 6
    points = np.zeros((columns * rows, 3), dtype=np.float64)
    points[:, :2] = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2) * 0.025
    points[:, :2] -= np.mean(points[:, :2], axis=0)
    observations = []
    for index in range(28):
        rvec = np.asarray(
            [
                rng.uniform(-0.35, 0.35),
                rng.uniform(-0.35, 0.35),
                rng.uniform(-0.12, 0.12),
            ]
        )
        tvec = np.asarray(
            [rng.uniform(-0.14, 0.14), rng.uniform(-0.08, 0.08), rng.uniform(0.38, 0.75)]
        )
        image, _ = cv2.projectPoints(points, rvec, tvec, K, distortion)
        image = image.reshape(-1, 2) + rng.normal(0.0, 0.08, size=(len(points), 2))
        visible = (
            (image[:, 0] >= 2)
            & (image[:, 0] < 638)
            & (image[:, 1] >= 2)
            & (image[:, 1] < 398)
        )
        if np.sum(visible) < 20:
            continue
        observations.append(
            Observation(
                view_id=f"view_{index}",
                path="",
                group=f"session_{index}",
                image_size=(640, 400),
                object_points=points[visible],
                image_points=image[visible],
            )
        )
    return observations, K


class CalibrationPipelineTests(unittest.TestCase):
    def test_operational_threshold_is_only_bypassed_in_explicit_skip_mode(self) -> None:
        candidate = {
            "mapping_validity": {"valid": True},
            "validation_residual_px": {"p95": 0.1},
            "validation_residual_p95_bootstrap": None,
            "validation_spatial_coverage": {"eligible_bin_fraction": 1.0},
            "calibration_pose_consistency": {"availability_fraction": 1.0},
            "operational_pose": None,
        }
        thresholds = {"maximum_p95_absolute_z_error_mm": 2.0}

        passed, failures = passes_thresholds(candidate, thresholds)
        self.assertFalse(passed)
        self.assertEqual(failures, ["missing_operational_pose_validation"])

        passed, failures = passes_thresholds(
            candidate,
            thresholds,
            require_operational_validation=False,
        )
        self.assertTrue(passed)
        self.assertEqual(failures, [])

    def test_capture_policy_prefers_uncovered_regions_then_rejects_duplicates(self) -> None:
        policy = CapturePolicy(
            {
                "stable_frames": 0,
                "minimum_seconds_between_captures": 0.0,
                "minimum_descriptor_distance": 0.3,
                "minimum_captures_per_bin": 1,
            },
            (2, 2),
        )
        observation = TargetObservation(
            valid=True,
            image_points=np.asarray([[10.0, 10.0], [20.0, 10.0], [20.0, 20.0], [10.0, 20.0]]),
            object_points=np.zeros((4, 3)),
            ids=np.arange(4),
            sharpness=100.0,
            coverage_fraction=0.1,
            centroid=(15.0, 15.0),
            occupied_bins=[(0, 0)],
            descriptor=np.asarray([0.1, 0.1, 0.0, 0.0, 0.0, 0.0]),
            approximate_rvec=None,
            approximate_tvec=None,
        )
        accepted, reason, _ = policy.decision(observation, now=1.0, settings_ok=True)
        self.assertTrue(accepted)
        self.assertEqual(reason, "undercovered_region")
        policy.accepted(observation, now=1.0)
        accepted, reason, _ = policy.decision(observation, now=2.0, settings_ok=True)
        self.assertFalse(accepted)
        self.assertEqual(reason, "duplicate_pose")

    def test_production_settings_read_exact_camera_controls(self) -> None:
        settings = ProductionSettings.from_config(
            {
                "camera": {
                    "width": 640,
                    "height": 400,
                    "frame_rate": 120.0,
                    "buffer_count": 4,
                    "pixel_format": "YUV420",
                    "exposure_time_us": 3000,
                    "analogue_gain": 1.0,
                    "brightness": 0.0,
                    "contrast": 1.0,
                },
                "calibration": {
                    "camera_matrix": [[478.0, 0.0, 320.0], [0.0, 478.0, 200.0], [0.0, 0.0, 1.0]],
                    "distortion_coefficients": [0.1, 0.0, 0.0, 0.0, 0.0],
                },
            }
        )
        self.assertEqual(settings.width, 640)
        self.assertEqual(settings.height, 400)
        self.assertEqual(settings.frame_rate, 120.0)
        self.assertEqual(settings.exposure_time_us, 3000)
        self.assertEqual(settings.pixel_format, "YUV420")

    def test_generated_charuco_board_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
            board = cv2.aruco.CharucoBoard((10, 7), 0.03, 0.022, dictionary)
            image = board.generateImage((1000, 700), marginSize=24, borderBits=1)
            image_path = root / "session_01_board.png"
            self.assertTrue(cv2.imwrite(str(image_path), image))
            config = {
                "image_size": [1000, 700],
                "dataset": {
                    "images": "*.png",
                    "group_regex": "(session_[0-9]+)",
                },
                "target": {
                    "type": "charuco",
                    "dictionary": "DICT_5X5_100",
                    "squares_x": 10,
                    "squares_y": 7,
                    "square_length_m": 0.03,
                    "marker_length_m": 0.022,
                    "minimum_points": 12,
                },
                "detection": {"minimum_laplacian_variance": 1.0},
            }
            observations, summary = detect_observations(config, root)
            self.assertEqual(summary["accepted_views"], 1)
            self.assertGreaterEqual(len(observations[0].image_points), 12)
            self.assertEqual(observations[0].group, "session_01")

    def test_mapping_check_rejects_current_full_frame_but_accepts_crop(self) -> None:
        K = np.asarray(
            [[478.11017984, 0.0, 322.59805209], [0.0, 478.29786406, 195.78709198], [0, 0, 1]],
            dtype=np.float64,
        )
        distortion = np.asarray(
            [0.159361045, 0.00175631861, -0.000966795628, 0.001165244, -1.18066737]
        )
        fit = make_fit(K, distortion)
        settings = {
            "mapping_grid": [41, 27],
            "maximum_round_trip_error_px": 0.05,
            "minimum_inverse_jacobian_determinant": 1e-12,
        }
        self.assertFalse(mapping_validity(fit, (640, 400), settings)["valid"])
        self.assertTrue(mapping_validity(fit, (640, 400), settings, [120, 0, 400, 400])["valid"])

    def test_group_split_never_leaks(self) -> None:
        observations = []
        for group_index in range(12):
            for frame_index in range(2):
                points = np.asarray([[100 + group_index, 100], [200, 100], [200, 200], [100, 200]])
                observations.append(
                    Observation(
                        view_id=f"g{group_index}_f{frame_index}",
                        path="",
                        group=f"g{group_index}",
                        image_size=(640, 400),
                        object_points=np.zeros((4, 3)),
                        image_points=points.astype(np.float64),
                    )
                )
        training, validation, _ = split_observations(
            observations, {"validation_fraction": 0.25, "seed": 4, "spatial_bins": [2, 2]}
        )
        self.assertTrue(training)
        self.assertTrue(validation)
        self.assertFalse({item.group for item in training} & {item.group for item in validation})

    def test_synthetic_calibration_has_small_held_out_error(self) -> None:
        observations, K = synthetic_observations()
        training, validation, _ = split_observations(
            observations,
            {"validation_fraction": 0.25, "seed": 7, "spatial_bins": [3, 2]},
        )
        fit = calibrate_model(
            training,
            {
                "name": "opencv5",
                "type": "opencv",
                "flags": [],
                "minimum_views": 8,
                "deployment_compatible": True,
            },
        )
        result, _, _ = evaluate_model(
            validation,
            fit,
            {
                "pose_anchor": {"mode": "all"},
                "spatial_bins": [4, 3],
                "minimum_points_per_spatial_bin": 5,
            },
        )
        self.assertLess(result["residual_px"]["p95"], 0.35)
        self.assertAlmostEqual(fit.K[0, 0], K[0, 0], delta=8.0)
        self.assertAlmostEqual(fit.K[1, 1], K[1, 1], delta=8.0)

    def test_end_to_end_cli_writes_a_final_refit_and_patch(self) -> None:
        observations, _ = synthetic_observations()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            observations_path = root / "observations.json"
            output_path = root / "output"
            config_path = root / "config.json"
            save_observations(observations_path, observations)
            config = {
                "image_size": [640, 400],
                "dataset": {"images": "unused", "observations_file": str(observations_path)},
                "target": {"type": "charuco"},
                "split": {"validation_fraction": 0.25, "seed": 7, "spatial_bins": [3, 2]},
                "models": [
                    {
                        "name": "opencv5",
                        "type": "opencv",
                        "flags": [],
                        "minimum_views": 8,
                        "deployment_compatible": True,
                    }
                ],
                "validation": {
                    "pose_anchor": {"mode": "all"},
                    "spatial_bins": [4, 3],
                    "minimum_points_per_spatial_bin": 5,
                },
                "model_analysis": {
                    "mapping_grid": [21, 15],
                    "maximum_round_trip_error_px": 0.05,
                    "minimum_inverse_jacobian_determinant": 1e-12,
                },
                "bootstrap": {"iterations": 0},
                "selection": {
                    "metric": "validation_p95_px",
                    "simpler_model_tolerance_px": 0.02,
                    "require_full_frame_mapping_valid": True,
                },
                "crop_sweep": {
                    "horizontal_trims_px": [0, 80],
                    "minimum_points_per_pose": 6,
                    "confidence": {"iterations": 0},
                    "thresholds": {
                        "require_mapping_valid": True,
                        "maximum_validation_p95_px": 0.5,
                        "minimum_availability_fraction": 0.5,
                        "maximum_p95_absolute_z_error_mm": 2.0,
                    },
                },
                "operational_validation": {"dataset": None},
                "output_dir": str(output_path),
            }
            config_path.write_text(json.dumps(config))
            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "run_calibration.py"),
                    "--config",
                    str(config_path),
                    "--no-plots",
                    "--skip-operational-validation",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr + completed.stdout)
            result = json.loads((output_path / "results.json").read_text())
            self.assertEqual(result["final_refit"]["status"], "ok")
            self.assertEqual(result["deployment_recommendation"]["model"], "opencv5")
            self.assertEqual(result["operational_validation"]["mode"], "skipped")
            self.assertFalse(result["operational_validation"]["required_for_acceptance"])
            self.assertEqual(
                result["operational_validation"]["skipped_operational_thresholds"],
                ["maximum_p95_absolute_z_error_mm"],
            )
            self.assertTrue((output_path / "localizer_calibration_patch.json").exists())
            patch = json.loads((output_path / "localizer_calibration_patch.json").read_text())
            self.assertFalse(patch["provenance"]["operational_validation_used"])
            self.assertEqual(patch["provenance"]["operational_validation_mode"], "skipped")

    def test_observation_round_trip(self) -> None:
        observation = Observation(
            view_id="one",
            path="image.png",
            group="session",
            image_size=(640, 400),
            object_points=np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            image_points=np.asarray([[100.0, 200.0], [300.0, 200.0]]),
            sharpness=123.0,
            coverage_fraction=0.1,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "observations.json"
            save_observations(path, [observation])
            root = json.loads(path.read_text())
            self.assertEqual(root["schema_version"], 1)
            self.assertEqual(root["views"][0]["group"], "session")


if __name__ == "__main__":
    unittest.main()
