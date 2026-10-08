import unittest

from marker_grid_controller.__main__ import (
    hypergrid_level,
    parse_tiles,
    response,
    snake_order_tiles,
)


class MarkerGridControllerTest(unittest.TestCase):
    def test_status_response_reports_grid_provenance(self):
        digest = "a" * 64
        result = response(
            {"request_id": "request"},
            {(0, 0): "off"},
            grid_sha256=digest,
        )

        self.assertEqual(result["grid_sha256"], digest)

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
