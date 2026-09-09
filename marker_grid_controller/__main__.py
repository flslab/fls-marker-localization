import argparse
import json
import select
import signal
import socket
import time

import board
import neopixel_spi as neopixel


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

    tiles.sort(key=lambda tile: tile[0])
    return tiles, bit_time, payload_bits + len(delimiter)


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("grid")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5558)
    parser.add_argument("--allow-host")
    parser.add_argument("--gpio", type=int, default=10)
    parser.add_argument(
        "--initial-mode", choices=("off", "static", "blink"), default="blink"
    )
    parser.add_argument("--mygrid-level", type=int, default=255)
    parser.add_argument("--hypergrid-level", type=int, default=255)
    args = parser.parse_args()

    if args.gpio != 10:
        parser.error("board.SPI() uses GPIO 10 / physical pin 19")
    if not 0 <= args.mygrid_level <= 255:
        parser.error("--mygrid-level must be between 0 and 255")
    if not 0 <= args.hypergrid_level <= 255:
        parser.error("--hypergrid-level must be between 0 and 255")

    tiles, bit_time, packet_length = load_grid(args.grid)
    modes = {coordinate: args.initial_mode for coordinate, _ in tiles}

    # Same setup as fls-cf-offboard-controller/led.py.
    pixels = neopixel.NeoPixel_SPI(
        board.SPI(),
        len(tiles) * 2,
        pixel_order=neopixel.GRB,
        auto_write=False,
        brightness=1.0,
    )

    def draw(bit):
        for index, (coordinate, patterns) in enumerate(tiles):
            mode = modes[coordinate]
            if mode == "off":
                levels = [0, 0, 0, 0]
            elif mode == "static":
                levels = [args.mygrid_level] * 4
            else:
                levels = [args.mygrid_level * pattern[bit] for pattern in patterns]

            pixels[index * 2] = tuple(levels[:3])
            pixels[index * 2 + 1] = (
                levels[3],
                args.hypergrid_level,
                args.hypergrid_level,
            )
        pixels.show()

    allowed_ip = socket.gethostbyname(args.allow_host) if args.allow_host else None
    udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp.bind((args.host, args.port))
    udp.setblocking(False)

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
                    if allowed_ip and address[0] != allowed_ip:
                        continue
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
