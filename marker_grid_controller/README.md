# Marker-grid controller

This is a small Raspberry Pi Zero W controller based directly on
`fls-cf-offboard-controller/led.py`. It uses `board.SPI()` and
`neopixel_spi.NeoPixel_SPI` on GPIO 10 (physical pin 19).

Install:

```sh
python3 -m pip install -r marker_grid_controller/requirements.txt
```

Run:

```sh
python3 -m marker_grid_controller \
  high_rate_localizer/config/hypergrid-mygrid.json \
  --host 0.0.0.0 --port 5558 --gpio 10 \
  --initial-mode blink --mygrid-level 255 --hypergrid-level 255
```

HyperGrid LEDs are on for every tile by default. To enable them only on
specific tiles, pass their `[i, j]` coordinates to `--hypergrid-tiles` as JSON:

```sh
python3 -m marker_grid_controller \
  high_rate_localizer/config/hypergrid-mygrid.json \
  --hypergrid-tiles '[[0,0],[1,0]]'
```

Hardware test—all channels alternate between 0 and 255 every second and each
transition is printed:

```sh
python3 -m marker_grid_controller \
  high_rate_localizer/config/hypergrid-mygrid.json --gpio 10 --test
```

Tile `(i, j)` uses `i` for the x index and `j` for the y index. Tiles are wired
in serpentine row-major order: within the first fixed-x row, `j` runs from its
lowest to highest value; in the next x row, `j` runs from highest to lowest;
and so on. Thus `(-1, -1)` and `(-1, 0)` are adjacent along the y axis. Each
tile uses two WS2811 values:

- chip 1 R/G/B: the first three MyGrid patterns
- chip 2 R: the fourth MyGrid pattern
- chip 2 G/B: HyperGrid, on at `--hypergrid-level` for every tile by default,
  or only for tiles selected with `--hypergrid-tiles`

The UDP commands used by the orchestrator are:

```json
{"version":1,"request_id":"1","command":"set_mode","mode":"blink"}
```

Add `"tile":[i,j]` to change one tile. Modes are `off`, `static`, and
`blink`. Every successful response includes `grid_sha256`, the SHA-256 of the
loaded grid JSON, so camera pose consumers can reject coordinate-map skew.
