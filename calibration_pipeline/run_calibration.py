#!/usr/bin/env python3
"""Run target detection, camera-model comparison, validation and crop selection."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import cv2
import numpy as np

from calibration_lib import (
    SCHEMA_VERSION,
    bootstrap_parameter_stability,
    calibrate_model,
    choose_deployment,
    choose_model,
    detect_observations,
    evaluate_operational_crop,
    evaluate_model,
    load_observations,
    load_operational_validation,
    mapping_validity,
    markdown_report,
    passes_thresholds,
    radial_monotonicity,
    resolve_path,
    save_observations,
    split_observations,
    sweep_crops,
    write_json,
    write_plots,
    write_residual_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Pipeline JSON configuration")
    parser.add_argument("--output", help="Override output directory")
    parser.add_argument(
        "--detect-only",
        action="store_true",
        help="Detect and save calibration observations without fitting models",
    )
    parser.add_argument("--no-plots", action="store_true", help="Skip optional matplotlib plots")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    with config_path.open() as stream:
        config = json.load(stream)
    config_dir = config_path.parent
    output_dir = (
        Path(args.output).expanduser().resolve()
        if args.output
        else resolve_path(config.get("output_dir", "output"), config_dir)
    )
    assert output_dir is not None
    output_dir.mkdir(parents=True, exist_ok=True)
    export_path = output_dir / "localizer_calibration_patch.json"
    if export_path.exists():
        export_path.unlink()
    write_json(output_dir / "config_snapshot.json", config)

    observations_path = resolve_path(config["dataset"].get("observations_file"), config_dir)
    if observations_path is not None and observations_path.exists():
        observations = load_observations(observations_path)
        detection_summary = {
            "source": "observations_file",
            "path": str(observations_path),
            "accepted_views": len(observations),
            "matched_images": None,
            "rejected_views": None,
            "rejections": [],
        }
    else:
        observations, detection_summary = detect_observations(config, config_dir)
        detection_summary["source"] = "images"
        observations_path = output_dir / "observations.json"
        save_observations(observations_path, observations)

    if not observations:
        raise ValueError("no usable calibration observations")
    expected_size = tuple(int(item) for item in config["image_size"])
    if observations[0].image_size != expected_size:
        raise ValueError(
            f"observations are {observations[0].image_size}, config image_size is {expected_size}"
        )
    if args.detect_only:
        detection_result = {
            "schema_version": SCHEMA_VERSION,
            "config": str(config_path),
            "opencv_version": cv2.__version__,
            "numpy_version": np.__version__,
            "python_version": platform.python_version(),
            "capture_metadata": config.get("capture_metadata", {}),
            "dataset": {
                **detection_summary,
                "observations_file": str(observations_path),
                "image_size": list(expected_size),
                "accepted_views": len(observations),
            },
        }
        if not args.no_plots:
            detection_result["plots"] = write_plots(output_dir, observations, [], {})
        write_json(output_dir / "detection_results.json", detection_result)
        print(f"wrote {observations_path}")
        print(f"wrote {output_dir / 'detection_results.json'}")
        return 0
    training, validation, assignments = split_observations(observations, config.get("split", {}))
    write_json(output_dir / "split_assignments.json", assignments)

    operational_frames = None
    operational_path = resolve_path(
        config.get("operational_validation", {}).get("dataset"), config_dir
    )
    if operational_path is not None:
        operational_frames = load_operational_validation(operational_path, expected_size)

    model_results = []
    residual_by_model = {}
    for model in config["models"]:
        if not model.get("enabled", True):
            continue
        name = str(model["name"])
        print(f"calibrating {name}...", flush=True)
        try:
            fit = calibrate_model(training, model)
            validation_result, residuals, reference_poses = evaluate_model(
                validation, fit, config.get("validation", {})
            )
            residual_by_model[name] = residuals
            write_residual_csv(output_dir / f"residuals_{name}.csv", residuals)
            analysis = config.get("model_analysis", {})
            model_result = {
                "name": name,
                "status": "ok",
                "calibration": fit.calibration_json(),
                "validation": validation_result,
                "mapping_validity": mapping_validity(fit, expected_size, analysis),
                "radial_monotonicity": radial_monotonicity(
                    fit, float(analysis.get("radial_test_limit", 5.0))
                ),
                "bootstrap_stability": bootstrap_parameter_stability(
                    training, model, config.get("bootstrap", {})
                ),
            }
            model_result["crop_sweep"] = sweep_crops(
                validation,
                fit,
                residuals,
                reference_poses,
                config,
                operational_frames,
            )
            model_results.append(model_result)
        except (cv2.error, ValueError, ArithmeticError) as error:
            model_results.append(
                {
                    "name": name,
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )

    model_recommendation = choose_model(model_results, config.get("selection", {}))
    deployment_recommendation = choose_deployment(model_results)
    final_refit = None
    deployment_fit = None
    if deployment_recommendation is not None:
        selected_name = deployment_recommendation["model"]
        selected_roi = deployment_recommendation["roi"]
        selected_config = next(item for item in config["models"] if item["name"] == selected_name)
        selected_result = next(item for item in model_results if item["name"] == selected_name)
        selected_crop = next(item for item in selected_result["crop_sweep"] if item["roi"] == selected_roi)
        try:
            deployment_fit = calibrate_model(observations, selected_config)
            final_operational = None
            if operational_frames is not None:
                final_operational = evaluate_operational_crop(
                    operational_frames,
                    deployment_fit,
                    selected_roi,
                    int(config.get("crop_sweep", {}).get("minimum_points_per_pose", 6)),
                    config.get("crop_sweep", {}).get("confidence", {}),
                    str(config.get("crop_sweep", {}).get("pose_solver", "sqpnp")),
                    bool(config.get("crop_sweep", {}).get("refine_lm", True)),
                )
            final_candidate = {
                "mapping_validity": mapping_validity(
                    deployment_fit,
                    expected_size,
                    config.get("model_analysis", {}),
                    selected_roi,
                ),
                "validation_residual_px": selected_crop["validation_residual_px"],
                "validation_spatial_coverage": selected_crop[
                    "validation_spatial_coverage"
                ],
                "validation_residual_p95_bootstrap": selected_crop[
                    "validation_residual_p95_bootstrap"
                ],
                "calibration_pose_consistency": selected_crop[
                    "calibration_pose_consistency"
                ],
                "operational_pose": final_operational,
            }
            final_passed, final_failures = passes_thresholds(
                final_candidate, config.get("crop_sweep", {}).get("thresholds", {})
            )
            final_refit = {
                "status": "ok" if final_passed else "rejected",
                "calibration": deployment_fit.calibration_json(),
                "roi": selected_roi,
                "full_frame_mapping_validity": mapping_validity(
                    deployment_fit, expected_size, config.get("model_analysis", {})
                ),
                "selected_roi_mapping_validity": final_candidate["mapping_validity"],
                "operational_pose": final_operational,
                "passes_thresholds": final_passed,
                "threshold_failures": final_failures,
                "note": (
                    "Model family and ROI were selected without held-out leakage; coefficients were then "
                    "refitted using all calibration observations and rechecked against independent criteria."
                ),
            }
            if not final_passed:
                deployment_recommendation = None
                deployment_fit = None
        except (cv2.error, ValueError, ArithmeticError) as error:
            final_refit = {
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
            }
            deployment_recommendation = None
            deployment_fit = None
    result = {
        "schema_version": SCHEMA_VERSION,
        "config": str(config_path),
        "opencv_version": cv2.__version__,
        "numpy_version": np.__version__,
        "python_version": platform.python_version(),
        "capture_metadata": config.get("capture_metadata", {}),
        "dataset": {
            **detection_summary,
            "observations_file": str(observations_path),
            "image_size": list(expected_size),
            "accepted_views": len(observations),
            "training_views": len(training),
            "validation_views": len(validation),
            "training_groups": len({item.group for item in training}),
            "validation_groups": len({item.group for item in validation}),
        },
        "operational_validation": {
            "dataset": str(operational_path) if operational_path else None,
            "frames": len(operational_frames) if operational_frames is not None else 0,
            "independent_pose_reference_available": operational_frames is not None,
        },
        "models": model_results,
        "model_recommendation": model_recommendation,
        "deployment_recommendation": deployment_recommendation,
        "final_refit": final_refit,
    }
    if not args.no_plots:
        result["plots"] = write_plots(output_dir, observations, model_results, residual_by_model)
    else:
        result["plots"] = []

    if deployment_recommendation is not None:
        assert deployment_fit is not None
        fit = deployment_fit
        roi = deployment_recommendation["roi"]
        deployment = {
            "calibration": {
                "camera_matrix": fit.K.tolist(),
                "distortion_coefficients": fit.distortion.reshape(-1).tolist(),
            },
            "processing_crop": {
                "enabled": roi != [0, 0, expected_size[0], expected_size[1]],
                "x": roi[0],
                "y": roi[1],
                "width": roi[2],
                "height": roi[3],
            },
            "provenance": {
                "pipeline_result": str(output_dir / "results.json"),
                "model": fit.name,
            },
        }
        write_json(export_path, deployment)

    write_json(output_dir / "results.json", result)
    (output_dir / "report.md").write_text(markdown_report(result))
    print(f"wrote {output_dir / 'results.json'}")
    print(f"wrote {output_dir / 'report.md'}")
    if deployment_recommendation is None:
        print("no deployment recommendation met every configured threshold", file=sys.stderr)
        return 2
    print(
        "recommended model={model} roi={roi}".format(
            model=deployment_recommendation["model"], roi=deployment_recommendation["roi"]
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
