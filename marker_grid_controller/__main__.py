import argparse
import json
import select
import signal
import socket
import time


def test_leds(pixels):
    value = 1
    print("LED test started; press Ctrl+C to stop", flush=True)
    try:
        while True:
            for index in range(len(pixels)):
                pixels[index] = (value, value, value)
            pixels.show()
            print(f"LED test: all channels = {value}", flush=True)
            value = 256 - value
            time.sleep(1)
    except KeyboardInterrupt:
        print("LED test stopped", flush=True)


def snake_order_tiles(tiles):
    """Order (i, j) = (x, y) tiles along fixed-x serpentine rows."""
    rows = sorted({coordinate[0] for coordinate, _ in tiles})
    row_number = {row: index for index, row in enumerate(rows)}
    return sorted(
        tiles,
        key=lambda tile: (
            tile[0][0],
            tile[0][1]
            if row_number[tile[0][0]] % 2 == 0
            else -tile[0][1],
        ),
    )


def load_grid(path):
    with open(path, encoding="utf-8") as file:
        grid = json.load(file)

    encoding = grid["encoding"]
    payload_bits = encoding["payload_bits"]
    delimiter = [int(bit) for bit in encoding["delimiter_pattern"]]
    bit_time = encoding["bit_duration_s"]
    msb_first = encoding.get("payload_bit_order") != "least_significant_first"

    tiles = []
    for tile in grid["mygrid"]["tiles"]:
        coordinate = (tile["i"], tile["j"])
        patterns = []
        for marker_id in tile["signature"]:
            payload = [
                (marker_id >> shift) & 1
                for shift in range(payload_bits - 1, -1, -1)
            ]
            if not msb_first:
                payload.reverse()
            patterns.append(payload + delimiter)
        tiles.append((coordinate, patterns))

    # Tile coordinates are (i, j) = (x, y). Follow the physical daisy chain
    # across each fixed-x row by changing j, then enter the next x row from
    # the same side. Alternating the y direction avoids a long wire from the
    # end of one row back to the start of the next.
    return snake_order_tiles(tiles), bit_time, payload_bits + len(delimiter)


def response(request, modes, changed=0):
    return {
        "version": 1,
        "request_id": request.get("request_id"),
        "ok": True,
        "changed": changed,
        "tiles": [
            {"tile": list(coordinate), "mode": mode}
            for coordinate, mode in modes.items()
        ],
    }


def hypergrid_level(coordinate, enabled_tiles, level):
    """Return the configured level when this tile's HyperGrid is enabled."""
    if enabled_tiles is None or coordinate in enabled_tiles:
        return level
    return 0


def parse_tiles(value):
    try:
        tiles = json.loads(value)
        if not isinstance(tiles, list):
            raise ValueError
        parsed = []
        for tile in tiles:
            if (
                not isinstance(tile, list)
                or len(tile) != 2
                or not all(
                    isinstance(part, int) and not isinstance(part, bool)
                    for part in tile
                )
            ):
                raise ValueError
            parsed.append(tuple(tile))
        return parsed
    except (json.JSONDecodeError, ValueError) as error:
        raise argparse.ArgumentTypeError(
            "tiles must be a JSON list of [i, j] integer pairs"
        ) from error


def main():
    import board
    import neopixel_spi as neopixel

    parser = argparse.ArgumentParser()
    parser.add_argument("grid")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5558)
    parser.add_argument("--gpio", type=int, default=10)
    parser.add_argument(
        "--initial-mode", choices=("off", "static", "blink"), default="blink"
    )
    parser.add_argument("--mygrid-level", type=int, default=255)
    parser.add_argument("--hypergrid-level", type=int, default=255)
    parser.add_argument(
        "--hypergrid-tiles",
        type=parse_tiles,
        metavar="JSON",
        help=(
            "enable HyperGrid LEDs only on these tiles, as a JSON list "
            "(default: enable every tile; an empty list disables all)"
        ),
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="alternate every LED between 0 and 255 once per second",
    )
    args = parser.parse_args()

    if args.gpio != 10:
        parser.error("board.SPI() uses GPIO 10 / physical pin 19")
    if not 0 <= args.mygrid_level <= 255:
        parser.error("--mygrid-level must be between 0 and 255")
    if not 0 <= args.hypergrid_level <= 255:
        parser.error("--hypergrid-level must be between 0 and 255")

    tiles, bit_time, packet_length = load_grid(args.grid)
    modes = {coordinate: args.initial_mode for coordinate, _ in tiles}
    enabled_hypergrid_tiles = (
        None
        if args.hypergrid_tiles is None
        else set(args.hypergrid_tiles)
    )
    unknown_hypergrid_tiles = (
        set() if enabled_hypergrid_tiles is None
        else enabled_hypergrid_tiles.difference(modes)
    )
    if unknown_hypergrid_tiles:
        parser.error(
            "--hypergrid-tiles contains unknown tile(s): "
            + ", ".join(str(tile) for tile in sorted(unknown_hypergrid_tiles))
        )

    # Binding first also prevents a test process and a running controller from
    # writing conflicting frames to the same SPI bus.
    udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp.bind((args.host, args.port))
    udp.setblocking(False)

    # Same setup as fls-cf-offboard-controller/led.py.
    pixels = neopixel.NeoPixel_SPI(
        board.SPI(),
        len(tiles) * 2,
        pixel_order=neopixel.RGB,
        auto_write=False,
        brightness=1.0,
    )

    if args.test:
        try:
            test_leds(pixels)
        finally:
            udp.close()
            for index in range(len(tiles) * 2):
                pixels[index] = (0, 0, 0)
            pixels.show()
        return

    def draw(bit):
        for index, (coordinate, patterns) in enumerate(tiles):
            mode = modes[coordinate]
            if mode == "off":
                levels = [0, 0, 0, 0]
            elif mode == "static":
                levels = [args.mygrid_level] * 4
            else:
                levels = [args.mygrid_level * pattern[bit] for pattern in patterns]

            tile_hypergrid_level = hypergrid_level(
                coordinate, enabled_hypergrid_tiles, args.hypergrid_level
            )
            pixels[index * 2] = tuple(levels[:3])
            pixels[index * 2 + 1] = (
                levels[3],
                tile_hypergrid_level,
                tile_hypergrid_level,
            )
        pixels.show()

    running = True

    def stop(_signal, _frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    bit = 0
    next_frame = time.monotonic()
    print(f"marker grid ready: {len(tiles)} tiles, UDP {args.host}:{args.port}")

    try:
        while running:
            timeout = max(0, next_frame - time.monotonic())
            readable, _, _ = select.select([udp], [], [], timeout)

            if readable:
                data, address = udp.recvfrom(4096)
                request = {}
                try:
                    request = json.loads(data)
                    changed = 0
                    if request["command"] == "set_mode":
                        mode = request["mode"]
                        if mode not in ("off", "static", "blink"):
                            raise ValueError("invalid mode")
                        targets = modes
                        if "tile" in request:
                            target = tuple(request["tile"])
                            if target not in modes:
                                raise ValueError("unknown tile")
                            targets = (target,)
                        for target in targets:
                            if modes[target] != mode:
                                modes[target] = mode
                                changed += 1
                    elif request["command"] != "status":
                        raise ValueError("invalid command")
                    reply = response(request, modes, changed)
                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                    reply = {
                        "version": 1,
                        "request_id": request.get("request_id"),
                        "ok": False,
                        "error": str(error),
                    }
                udp.sendto(json.dumps(reply).encode(), address)

            now = time.monotonic()
            if now >= next_frame:
                draw(bit)
                bit = (bit + 1) % packet_length
                next_frame = now + bit_time
    finally:
        udp.close()
        for index in range(len(tiles) * 2):
            pixels[index] = (0, 0, 0)
        pixels.show()


if __name__ == "__main__":
    main()
