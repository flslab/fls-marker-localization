#!/usr/bin/env python3
"""Generate a print-scale ChArUco SVG and a raster preview from pipeline config."""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Calibration pipeline JSON config")
    parser.add_argument("--output-dir", help="Output directory (default: beside config)")
    parser.add_argument("--dpi", type=int, default=600, help="Rasterization resolution")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    config = json.loads(config_path.read_text())
    target = config["target"]
    if str(target["type"]).lower() != "charuco":
        raise ValueError("board generation currently supports target.type=charuco")
    aruco = cv2.aruco
    dictionary_name = str(target["dictionary"])
    if not hasattr(aruco, dictionary_name):
        raise ValueError(f"unknown ArUco dictionary {dictionary_name!r}")
    dictionary = aruco.getPredefinedDictionary(getattr(aruco, dictionary_name))
    squares_x = int(target["squares_x"])
    squares_y = int(target["squares_y"])
    square_length_m = float(target["square_length_m"])
    marker_length_m = float(target["marker_length_m"])
    board = aruco.CharucoBoard(
        (squares_x, squares_y), square_length_m, marker_length_m, dictionary
    )

    width_mm = squares_x * square_length_m * 1000.0
    height_mm = squares_y * square_length_m * 1000.0
    pixels_per_mm = args.dpi / 25.4
    raster_width = max(1, int(round(width_mm * pixels_per_mm)))
    raster_height = max(1, int(round(height_mm * pixels_per_mm)))
    image = board.generateImage((raster_width, raster_height), marginSize=0, borderBits=1)
    success, encoded = cv2.imencode(".png", image)
    if not success:
        raise RuntimeError("OpenCV could not encode the generated board")

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else config_path.parent / "generated_board"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "charuco_board.png"
    png_path.write_bytes(encoded.tobytes())
    data = base64.b64encode(encoded.tobytes()).decode("ascii")
    svg = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"
     width="{width_mm:.6f}mm" height="{height_mm:.6f}mm"
     viewBox="0 0 {raster_width} {raster_height}">
  <image width="{raster_width}" height="{raster_height}"
         xlink:href="data:image/png;base64,{data}"/>
</svg>
'''
    svg_path = output_dir / "charuco_board_print_at_100_percent.svg"
    svg_path.write_text(svg)
    metadata = {
        "config": str(config_path),
        "dictionary": dictionary_name,
        "squares": [squares_x, squares_y],
        "configured_square_length_mm": square_length_m * 1000.0,
        "configured_marker_length_mm": marker_length_m * 1000.0,
        "expected_printed_board_size_mm": [width_mm, height_mm],
        "raster_dpi": args.dpi,
        "critical_instruction": (
            "Print the SVG at 100%/actual size with all fit-to-page scaling disabled, then measure "
            "several square spans with a calibrated ruler or caliper. Put the measured square length "
            "back into the calibration config before running calibration."
        ),
    }
    (output_dir / "board_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(svg_path)
    print(png_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

