"""Hardware-independent marker-grid configuration and control logic."""

from __future__ import annotations

import json
import math
from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from time import monotonic
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

RGB = Tuple[int, int, int]
TileCoordinate = Tuple[int, int]


class PixelOutput:
    """The small interface implemented by the WS2811 and test outputs."""

    def write(self, pixels: Sequence[RGB]) -> None:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


class Mode(str, Enum):
    OFF = "off"
    STATIC = "static"
    BLINK = "blink"


@dataclass(frozen=True)
class TileDefinition:
    coordinate: TileCoordinate
    signature: Tuple[int, int, int, int]


@dataclass(frozen=True)
class GridDefinition:
    """The subset of an ``fls-marker-grid`` file needed by the LED node."""

    tiles: Tuple[TileDefinition, ...]
    payload_bits: int
    delimiter: Tuple[bool, ...]
    bit_duration_s: float
    payload_most_significant_first: bool

    @property
    def packet_bits(self) -> int:
        return self.payload_bits + len(self.delimiter)

    @classmethod
    def load(cls, path: Path) -> "GridDefinition":
        try:
            root = json.loads(path.read_text(encoding="utf-8"))
        except OSError as error:
            raise ValueError(f"unable to read grid file {path}: {error}") from error
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON in grid file {path}: {error}") from error

        if not isinstance(root, dict):
            raise ValueError("grid file root must be a JSON object")
        if root.get("schema") != "fls-marker-grid":
            raise ValueError("grid file must use the fls-marker-grid schema")
        if root.get("schema_version") != 1:
            raise ValueError("grid file must use fls-marker-grid schema version 1")

        encoding = root.get("encoding")
        if not isinstance(encoding, dict):
            raise ValueError("grid file is missing encoding")
        payload_bits = _positive_int(
            encoding.get("payload_bits"), "encoding.payload_bits"
        )
        if payload_bits > 16:
            raise ValueError("encoding.payload_bits must be at most 16")
        delimiter_pattern = encoding.get("delimiter_pattern")
        if (not isinstance(delimiter_pattern, str) or not delimiter_pattern or
                any(bit not in "01" for bit in delimiter_pattern)):
            raise ValueError(
                "encoding.delimiter_pattern must be a non-empty bit string"
            )
        bit_duration_s = encoding.get("bit_duration_s")
        if (isinstance(bit_duration_s, bool) or
                not isinstance(bit_duration_s, (int, float)) or
                not math.isfinite(bit_duration_s) or bit_duration_s <= 0):
            raise ValueError("encoding.bit_duration_s must be positive and finite")

        bit_order = encoding.get("payload_bit_order", "most_significant_first")
        if bit_order not in ("most_significant_first", "least_significant_first"):
            raise ValueError(
                "encoding.payload_bit_order must be most_significant_first or "
                "least_significant_first"
            )

        mygrid = root.get("mygrid")
        raw_tiles = mygrid.get("tiles") if isinstance(mygrid, dict) else None
        if not isinstance(raw_tiles, list) or not raw_tiles:
            raise ValueError("grid file must contain at least one mygrid tile")

        tiles: List[TileDefinition] = []
        seen = set()
        maximum_id = (1 << payload_bits) - 1
        for index, raw_tile in enumerate(raw_tiles):
            if not isinstance(raw_tile, dict):
                raise ValueError(f"mygrid.tiles[{index}] must be an object")
            i = _integer(raw_tile.get("i"), f"mygrid.tiles[{index}].i")
            j = _integer(raw_tile.get("j"), f"mygrid.tiles[{index}].j")
            coordinate = (i, j)
            if coordinate in seen:
                raise ValueError(f"duplicate MyGrid tile coordinate {coordinate}")
            seen.add(coordinate)

            raw_signature = raw_tile.get("signature")
            if not isinstance(raw_signature, list) or len(raw_signature) != 4:
                raise ValueError(
                    f"mygrid.tiles[{index}].signature must contain four IDs"
                )
            signature = tuple(
                _integer(value, f"mygrid.tiles[{index}].signature[{slot}]")
                for slot, value in enumerate(raw_signature)
            )
            if any(
                marker_id < 0 or marker_id > maximum_id
                for marker_id in signature
            ):
                raise ValueError(
                    f"mygrid.tiles[{index}].signature IDs must fit in "
                    f"{payload_bits} payload bits"
                )
            tiles.append(TileDefinition(coordinate, signature))

        # The physical chain is tile-row first, then tile-column.
        tiles.sort(key=lambda tile: tile.coordinate)
        return cls(
            tiles=tuple(tiles),
            payload_bits=payload_bits,
            delimiter=tuple(bit == "1" for bit in delimiter_pattern),
            bit_duration_s=float(bit_duration_s),
            payload_most_significant_first=bit_order == "most_significant_first",
        )

    def marker_bit(self, marker_id: int, packet_index: int) -> bool:
        packet_index %= self.packet_bits
        if packet_index >= self.payload_bits:
            return self.delimiter[packet_index - self.payload_bits]
        if self.payload_most_significant_first:
            shift = self.payload_bits - packet_index - 1
        else:
            shift = packet_index
        return bool((marker_id >> shift) & 1)


class MarkerGridController:
    """Turns desired tile modes into the two WS2811 values for each tile."""

    def __init__(
        self,
        grid: GridDefinition,
        output: PixelOutput,
        *,
        initial_mode: Mode = Mode.BLINK,
        mygrid_level: int = 255,
        hypergrid_level: int = 255,
        phase_start: Optional[float] = None,
    ) -> None:
        self.grid = grid
        self.output = output
        self.mygrid_level = _byte(mygrid_level, "mygrid_level")
        self.hypergrid_level = _byte(hypergrid_level, "hypergrid_level")
        self.phase_start = monotonic() if phase_start is None else phase_start
        self.modes: Dict[TileCoordinate, Mode] = {
            tile.coordinate: initial_mode for tile in grid.tiles
        }
        self._last_pixels: Optional[Tuple[RGB, ...]] = None

    def current_packet_index(self, now: Optional[float] = None) -> int:
        current = monotonic() if now is None else now
        elapsed = max(0.0, current - self.phase_start)
        return int(elapsed / self.grid.bit_duration_s) % self.grid.packet_bits

    def set_mode(
        self, mode: Mode, tile: Optional[TileCoordinate] = None
    ) -> int:
        targets: Iterable[TileCoordinate]
        if tile is None:
            targets = self.modes
        else:
            if tile not in self.modes:
                raise ValueError(f"unknown MyGrid tile {tile}")
            targets = (tile,)

        changed = 0
        for coordinate in targets:
            if self.modes[coordinate] != mode:
                self.modes[coordinate] = mode
                changed += 1
        return changed

    def pixels_for_packet_index(self, packet_index: int) -> Tuple[RGB, ...]:
        pixels: List[RGB] = []
        for tile in self.grid.tiles:
            mode = self.modes[tile.coordinate]
            if mode is Mode.OFF:
                marker_levels = (0, 0, 0, 0)
            elif mode is Mode.STATIC:
                marker_levels = (self.mygrid_level,) * 4
            else:
                marker_levels = tuple(
                    self.mygrid_level
                    if self.grid.marker_bit(marker_id, packet_index) else 0
                    for marker_id in tile.signature
                )

            # Tile wiring: chip 1 = M0/M1/M2, chip 2 = M3/H/H.
            pixels.append(marker_levels[:3])
            pixels.append(
                (marker_levels[3], self.hypergrid_level, self.hypergrid_level)
            )
        return tuple(pixels)

    def render(self, now: Optional[float] = None, *, force: bool = False) -> bool:
        pixels = self.pixels_for_packet_index(self.current_packet_index(now))
        if force or pixels != self._last_pixels:
            self.output.write(pixels)
            self._last_pixels = pixels
            return True
        return False

    def close(self) -> None:
        self.output.close()


class MarkerGridProtocol:
    """Idempotent JSON request handler used by the UDP server."""

    VERSION = 1

    def __init__(self, controller: MarkerGridController, cache_size: int = 256):
        self.controller = controller
        self.cache_size = cache_size
        self._responses: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()

    def handle(self, request: Mapping[str, Any]) -> Dict[str, Any]:
        if not isinstance(request, Mapping):
            raise ValueError("request must be a JSON object")
        request_id = request.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("request_id must be a non-empty string")
        if request_id in self._responses:
            self._responses.move_to_end(request_id)
            return self._responses[request_id]

        version = request.get("version", self.VERSION)
        if version != self.VERSION:
            raise ValueError(f"unsupported protocol version {version}")
        command = request.get("command")
        if command == "status":
            response = self._response(request_id, changed=0)
        elif command == "set_mode":
            try:
                mode = Mode(request.get("mode"))
            except ValueError as error:
                raise ValueError("mode must be off, static, or blink") from error
            raw_tile = request.get("tile")
            tile = None if raw_tile is None else _tile_coordinate(raw_tile)
            changed = self.controller.set_mode(mode, tile)
            self.controller.render()
            response = self._response(request_id, changed=changed)
        else:
            raise ValueError("command must be status or set_mode")

        self._responses[request_id] = response
        while len(self._responses) > self.cache_size:
            self._responses.popitem(last=False)
        return response

    def _response(self, request_id: str, changed: int) -> Dict[str, Any]:
        return {
            "version": self.VERSION,
            "request_id": request_id,
            "ok": True,
            "changed": changed,
            "tiles": [
                {
                    "tile": list(tile.coordinate),
                    "mode": self.controller.modes[tile.coordinate].value,
                }
                for tile in self.controller.grid.tiles
            ],
        }


def _integer(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    return value


def _positive_int(value: Any, name: str) -> int:
    parsed = _integer(value, name)
    if parsed <= 0:
        raise ValueError(f"{name} must be positive")
    return parsed


def _byte(value: Any, name: str) -> int:
    parsed = _integer(value, name)
    if not 0 <= parsed <= 255:
        raise ValueError(f"{name} must be in [0, 255]")
    return parsed


def _tile_coordinate(value: Any) -> TileCoordinate:
    if (not isinstance(value, list) or len(value) != 2 or
            any(not isinstance(part, int) or isinstance(part, bool)
                for part in value)):
        raise ValueError("tile must contain two integer coordinates")
    return value[0], value[1]
