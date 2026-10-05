import unittest

from marker_grid_controller.__main__ import snake_order_tiles


class MarkerGridControllerTest(unittest.TestCase):
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
