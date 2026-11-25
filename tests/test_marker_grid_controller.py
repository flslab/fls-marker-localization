import unittest
import json
import tempfile
from pathlib import Path

from marker_grid_controller.__main__ import (
    hypergrid_level,
    parse_tiles,
    response,
    snake_order_tiles,
    load_grid,
    frame_values,
)


class MarkerGridControllerTest(unittest.TestCase):
    def test_sparse_chain_respects_explicit_slots_and_clears_inactive_chips(self):
        grid = dict(encoding=dict(payload_bits=4, delimiter_pattern='111110', bit_duration_s=.1),
                    mygrid=dict(tiles=[dict(i=0, j=0, signature=[0,0,0,1])]),
                    wiring=dict(physical_layout=[4,4], tile_order=[None]*16))
        grid['wiring']['tile_order'][6] = [0,0]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'grid.json'
            path.write_text(json.dumps(grid))
            slots, _, _ = load_grid(path)
            frame = frame_values(slots, {(0,0):'blink'}, 3, 70, 200)
            self.assertEqual(len(frame), 32)
            self.assertEqual(frame[12:14], [(0,0,0),(70,200,200)])
            self.assertTrue(all(value == (0,0,0) for i,value in enumerate(frame) if i not in (12,13)))
            for invalid in ([None]*16, [[0,0]]*16, [[8,8]]+[None]*15, [None]*15):
                grid['wiring']['tile_order'] = invalid
                path.write_text(json.dumps(grid))
                with self.assertRaises(ValueError):
                    load_grid(path)

    def test_status_response_reports_grid_provenance(self):
        digest = "a" * 64
        result = response(
            {"request_id": "request"},
            {(0, 0): "off"},
            grid_sha256=digest,
            physical_tile_count=16,
        )

        self.assertEqual(result["grid_sha256"], digest)
        self.assertEqual(result['physical_tile_count'], 16)

    def test_hypergrid_defaults_to_every_tile(self):
        self.assertEqual(hypergrid_level((4, -2), None, 173), 173)

    def test_hypergrid_only_enables_selected_tiles(self):
        enabled_tiles = {(0, 0), (2, -1)}

        self.assertEqual(hypergrid_level((2, -1), enabled_tiles, 173), 173)
        self.assertEqual(hypergrid_level((2, 0), enabled_tiles, 173), 0)

    def test_hypergrid_tiles_accept_negative_coordinates_and_empty_selection(self):
        self.assertEqual(parse_tiles("[[0,0],[-2,3]]"), [(0, 0), (-2, 3)])
        self.assertEqual(parse_tiles("[]"), [])

    def test_snake_order_uses_i_for_x_rows_and_j_for_y_columns(self):
        tiles = [
            ((1, 1), "e"),
            ((-1, 0), "b"),
            ((1, -1), "d"),
            ((-1, -1), "a"),
            ((-1, 1), "c"),
        ]

        ordered = snake_order_tiles(tiles)

        self.assertEqual(
            [coordinate for coordinate, _patterns in ordered],
            [(-1, -1), (-1, 0), (-1, 1), (1, 1), (1, -1)],
        )


if __name__ == "__main__":
    unittest.main()
