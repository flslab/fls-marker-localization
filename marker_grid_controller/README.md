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

Hardware test—all channels alternate between 0 and 255 every second and each
transition is printed:

```sh
python3 -m marker_grid_controller \
  high_rate_localizer/config/hypergrid-mygrid.json --gpio 10 --test
```

Each row-major tile uses two WS2811 values:

- chip 1 R/G/B: the first three MyGrid patterns
- chip 2 R: the fourth MyGrid pattern
- chip 2 G/B: HyperGrid, always on at `--hypergrid-level`

The UDP commands used by the orchestrator are:

```json
{"version":1,"request_id":"1","command":"set_mode","mode":"blink"}
```

Add `"tile":[i,j]` to change one tile. Modes are `off`, `static`, and
`blink`.
