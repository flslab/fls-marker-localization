"""Command-line entry point for the Raspberry Pi marker-grid node."""

from __future__ import annotations

import argparse
import json
import logging
import select
import signal
import socket
from pathlib import Path
from time import monotonic
from typing import Any, Dict, Optional

from .controller import (
    GridDefinition,
    MarkerGridController,
    MarkerGridProtocol,
    Mode,
)
from .hardware import DryRunOutput, Ws2811Output

LOGGER = logging.getLogger("marker-grid")


class UdpServer:
    def __init__(
        self,
        host: str,
        port: int,
        protocol: MarkerGridProtocol,
        allowed_host: Optional[str] = None,
    ):
        self.protocol = protocol
        self.allowed_ip = (
            None if allowed_host is None else socket.gethostbyname(allowed_host)
        )
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind((host, port))
        self.socket.setblocking(False)
        self.running = True

    def stop(self, _signal: Optional[int] = None, _frame: Any = None) -> None:
        self.running = False

    def serve(self) -> None:
        controller = self.protocol.controller
        controller.render(force=True)
        next_tick = monotonic() + controller.grid.bit_duration_s
        while self.running:
            timeout = max(0.0, min(0.25, next_tick - monotonic()))
            readable, _, _ = select.select((self.socket,), (), (), timeout)
            if readable:
                self._receive_one()

            now = monotonic()
            if now >= next_tick:
                controller.render(now)
                elapsed_ticks = int(
                    (now - controller.phase_start) / controller.grid.bit_duration_s
                )
                next_tick = (
                    controller.phase_start
                    + (elapsed_ticks + 1) * controller.grid.bit_duration_s
                )

    def close(self) -> None:
        self.socket.close()

    def _receive_one(self) -> None:
        payload, address = self.socket.recvfrom(65535)
        if self.allowed_ip is not None and address[0] != self.allowed_ip:
            LOGGER.warning(
                "ignored UDP command from non-orchestrator host %s", address[0]
            )
            return
        request_id = None
        try:
            request = json.loads(payload.decode("utf-8"))
            if isinstance(request, dict):
                request_id = request.get("request_id")
            response = self.protocol.handle(request)
        except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as error:
            response: Dict[str, Any] = {
                "version": MarkerGridProtocol.VERSION,
                "request_id": request_id,
                "ok": False,
                "error": str(error),
            }
        self.socket.sendto(
            json.dumps(response, separators=(",", ":")).encode("utf-8"), address
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Control a row-major chain of two-WS2811 marker-grid tiles"
    )
    parser.add_argument("grid", type=Path, help="fls-marker-grid JSON file")
    parser.add_argument("--host", default="0.0.0.0", help="UDP bind address")
    parser.add_argument("--port", type=int, default=5558, help="UDP control port")
    parser.add_argument(
        "--allow-host",
        help="accept commands only from this orchestrator host or IP",
    )
    parser.add_argument("--gpio", type=int, default=10, help="WS2811 data GPIO")
    parser.add_argument("--frequency-hz", type=int, default=800_000)
    parser.add_argument("--dma-channel", type=int, default=10)
    parser.add_argument("--mygrid-level", type=int, default=255)
    parser.add_argument("--hypergrid-level", type=int, default=255)
    parser.add_argument(
        "--initial-mode", choices=[mode.value for mode in Mode], default="blink"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="run networking and timing without GPIO"
    )
    parser.add_argument(
        "--check", action="store_true", help="validate the grid and print wiring order"
    )
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    if not 1 <= args.port <= 65535:
        raise SystemExit("--port must be in [1, 65535]")
    if args.frequency_hz <= 0:
        raise SystemExit("--frequency-hz must be positive")
    if args.gpio < 0 or args.dma_channel < 0:
        raise SystemExit("--gpio and --dma-channel must be non-negative")
    if (not 0 <= args.mygrid_level <= 255 or
            not 0 <= args.hypergrid_level <= 255):
        raise SystemExit("LED levels must be in [0, 255]")

    grid = GridDefinition.load(args.grid)
    pixel_count = len(grid.tiles) * 2
    wire_duration_s = pixel_count * 24 / args.frequency_hz
    if wire_duration_s >= grid.bit_duration_s:
        raise SystemExit(
            f"{pixel_count} WS2811 chips need at least {wire_duration_s:g} s "
            f"per update, longer than the {grid.bit_duration_s:g} s bit period"
        )
    if args.check:
        print(f"{len(grid.tiles)} tiles, {pixel_count} WS2811 chips")
        print(
            f"packet: {grid.payload_bits} payload + {len(grid.delimiter)} delimiter "
            f"bits at {grid.bit_duration_s:g} s/bit"
        )
        for physical_index, tile in enumerate(grid.tiles):
            print(
                f"tile {physical_index}: {tile.coordinate} -> "
                f"R1/G1/B1/R2 IDs {tile.signature}"
            )
        return 0

    output = (
        DryRunOutput(pixel_count)
        if args.dry_run
        else Ws2811Output(
            pixel_count,
            gpio=args.gpio,
            frequency_hz=args.frequency_hz,
            dma_channel=args.dma_channel,
        )
    )
    controller = MarkerGridController(
        grid,
        output,
        initial_mode=Mode(args.initial_mode),
        mygrid_level=args.mygrid_level,
        hypergrid_level=args.hypergrid_level,
    )
    server = UdpServer(
        args.host,
        args.port,
        MarkerGridProtocol(controller),
        allowed_host=args.allow_host,
    )
    signal.signal(signal.SIGINT, server.stop)
    signal.signal(signal.SIGTERM, server.stop)
    LOGGER.info(
        "ready on udp://%s:%d with %d row-major tiles (GPIO %d)",
        args.host,
        args.port,
        len(grid.tiles),
        args.gpio,
    )
    try:
        server.serve()
    finally:
        server.close()
        controller.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
