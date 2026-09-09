# Marker-grid LED controller

This is the Raspberry Pi Zero W node for the physical MyGrid/HyperGrid. It is
Python because network commands are sparse and the Adafruit `NeoPixel_SPI`
stack handles the timing-sensitive GPIO 10 transfer. The hardware adapter uses
the same `board.SPI()`, GRB order, manual `show()`, and brightness settings as
the proven `fls-cf-offboard-controller/led.py` implementation.

The grid is controlled only by the swarm orchestrator. Drones report lifecycle
events through their existing orchestrator connection; they do not connect to
this node. This keeps tile ownership and retries in one place and avoids an
N-to-one peer protocol on the flight network.

## Physical chain

GPIO 10 (header pin 19 / SPI0 MOSI) connects to chip 1 DIN. Chip 1 DOUT connects
to chip 2 DIN, and chip 2 DOUT connects to the next tile. Tiles are ordered by
`(i, j)` from the grid JSON: row `i` first, then column `j`.

Each tile consumes two WS2811 values:

| WS2811 value | R output | G output | B output |
| --- | --- | --- | --- |
| chip 1 | MyGrid signature 0 | MyGrid signature 1 | MyGrid signature 2 |
| chip 2 | MyGrid signature 3 | two HyperGrid LEDs | two HyperGrid LEDs |

The four signature positions retain the JSON's
`clockwise_from_top_left` order. HyperGrid stays on in every MyGrid mode while
the controller is running; a clean process shutdown blanks the complete chain.

## Pi setup and run

Enable SPI, ensure the runtime user can open `/dev/spidev0.0`, and install the
single hardware dependency in the node's virtual environment:

```sh
sudo raspi-config nonint do_spi 0
python3 -m pip install -r marker_grid_controller/requirements.txt
python3 -m marker_grid_controller \
  high_rate_localizer/config/hypergrid-mygrid.json --check
```

For a no-GPIO integration test:

```sh
python3 -m marker_grid_controller \
  high_rate_localizer/config/hypergrid-mygrid.json --dry-run
```

The orchestrator starts the production process. Direct invocation is:

```sh
python3 -m marker_grid_controller GRID.json \
  --host 0.0.0.0 --port 5558 --allow-host ORCHESTRATOR_IP --gpio 10 \
  --initial-mode blink --mygrid-level 255 --hypergrid-level 255
```

## Zero-output and wire-rate check

`--mygrid-level` controls chip 1 R/G/B and chip 2 R. `--hypergrid-level`
controls chip 2 G/B. Setting only one level to zero intentionally leaves the
other group active. To transmit an unambiguous all-zero frame continuously,
set both levels to zero and start MyGrid in `off` mode:

```sh
python3 -m marker_grid_controller GRID.json --gpio 10 \
  --initial-mode off --mygrid-level 0 --hypergrid-level 0 --verbose
```

Every verbose `pixels=` tuple must be `(0, 0, 0)` in this test. If it is and
the LEDs remain illuminated, verify that the runtime has the updated adapter
and the same dependencies as `fls-cf-offboard-controller/led.py`.

## UDP protocol

Requests and acknowledgements are UTF-8 JSON datagrams. Reusing a
`request_id` returns the cached acknowledgement, which makes orchestrator
retries idempotent.

```json
{"version":1,"request_id":"01","command":"set_mode","tile":[0,0],"mode":"static"}
```

Omit `tile` to update every tile. Valid modes are `blink`, `static`, and `off`.
Status uses `{"version":1,"request_id":"02","command":"status"}`. A successful
response contains `ok`, `changed`, and the desired mode of every tile.
