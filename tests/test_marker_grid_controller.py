import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from marker_grid_controller import (
    GridDefinition,
    MarkerGridController,
    MarkerGridProtocol,
    Mode,
)
from marker_grid_controller.hardware import DryRunOutput, Ws2811Output


def grid_document():
    return {
        "schema": "fls-marker-grid",
        "schema_version": 1,
        "encoding": {
            "payload_bits": 4,
            "delimiter_pattern": "111110",
            "payload_bit_order": "most_significant_first",
            "bit_duration_s": 0.02,
        },
        "mygrid": {
            "tiles": [
                {"i": 1, "j": 0, "signature": [1, 2, 3, 4]},
                {"i": -1, "j": 2, "signature": [8, 4, 2, 1]},
            ]
        },
    }


class MarkerGridControllerTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        path = Path(self.temporary_directory.name) / "grid.json"
        path.write_text(json.dumps(grid_document()), encoding="utf-8")
        self.grid = GridDefinition.load(path)
        self.output = DryRunOutput(4)
        self.controller = MarkerGridController(
            self.grid,
            self.output,
            phase_start=100.0,
            mygrid_level=200,
            hypergrid_level=77,
        )

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_tiles_are_row_major_and_channels_match_the_two_chips(self):
        self.assertEqual(
            [tile.coordinate for tile in self.grid.tiles], [(-1, 2), (1, 0)]
        )
        # First MSB: [8,4,2,1] -> [1,0,0,0]. HyperGrid is always on.
        self.assertEqual(
            self.controller.pixels_for_packet_index(0),
            ((200, 0, 0), (0, 77, 77), (0, 0, 0), (0, 77, 77)),
        )

    def test_delimiter_and_modes(self):
        # First delimiter bit is 1 for every ID.
        self.assertEqual(
            self.controller.pixels_for_packet_index(4),
            ((200, 200, 200), (200, 77, 77), (200, 200, 200), (200, 77, 77)),
        )
        self.controller.set_mode(Mode.OFF, (-1, 2))
        self.controller.set_mode(Mode.STATIC, (1, 0))
        self.assertEqual(
            self.controller.pixels_for_packet_index(0),
            ((0, 0, 0), (0, 77, 77), (200, 200, 200), (200, 77, 77)),
        )

    def test_blink_frames_repeat_without_exceeding_requested_levels(self):
        first_packet = [
            self.controller.pixels_for_packet_index(index)
            for index in range(self.grid.packet_bits)
        ]
        three_packets = [
            self.controller.pixels_for_packet_index(index)
            for index in range(self.grid.packet_bits * 3)
        ]

        self.assertEqual(three_packets, first_packet * 3)
        for frame in three_packets:
            for red, green, blue in frame:
                self.assertLessEqual(red, 200)
                self.assertLessEqual(green, 200)
                self.assertLessEqual(blue, 200)

    def test_protocol_is_idempotent_and_can_target_all_tiles(self):
        protocol = MarkerGridProtocol(self.controller)
        request = {
            "version": 1,
            "request_id": "abc",
            "command": "set_mode",
            "mode": "static",
        }
        first = protocol.handle(request)
        self.assertEqual(first["changed"], 2)
        writes = self.output.write_count
        self.assertEqual(protocol.handle(request), first)
        self.assertEqual(self.output.write_count, writes)
        self.assertTrue(all(tile["mode"] == "static" for tile in first["tiles"]))

    def test_unknown_tile_is_rejected(self):
        protocol = MarkerGridProtocol(self.controller)
        with self.assertRaisesRegex(ValueError, "unknown MyGrid tile"):
            protocol.handle({
                "request_id": "bad",
                "command": "set_mode",
                "mode": "off",
                "tile": [99, 99],
            })

    def test_ws2811_adapter_passes_exact_rgb_levels(self):
        class FakeStrip:
            def __init__(self, **options):
                self.options = options
                self.pixels = {}
                self.shows = 0

            def begin(self):
                pass

            def setPixelColorRGB(self, index, red, green, blue):
                self.pixels[index] = (red, green, blue)

            def show(self):
                self.shows += 1

        fake_module = SimpleNamespace(
            PixelStrip=FakeStrip,
            WS2811_STRIP_RGB=0x00100800,
        )
        with patch.dict(sys.modules, {"rpi_ws281x": fake_module}):
            output = Ws2811Output(2, gpio=10)
            output.write(((31, 0, 7), (9, 47, 47)))
            self.assertEqual(
                output._strip.pixels,
                {0: (31, 0, 7), 1: (9, 47, 47)},
            )
            self.assertEqual(output._strip.options["brightness"], 255)
            output.close()
            self.assertEqual(output._strip.pixels, {0: (0, 0, 0), 1: (0, 0, 0)})


if __name__ == "__main__":
    unittest.main()
