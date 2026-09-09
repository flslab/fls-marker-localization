"""WS2811 output using the proven offboard-controller NeoPixel SPI stack."""

from __future__ import annotations

from typing import Sequence

from .controller import RGB


class Ws2811Output:
    """Adapt marker-grid RGB frames to Adafruit ``NeoPixel_SPI``.

    This intentionally matches ``fls-cf-offboard-controller/led.py``: it uses
    ``board.SPI()``, GRB wire order, disabled auto-write, and full library
    brightness. MarkerGridController applies the independent MyGrid and
    HyperGrid levels to the individual channel values.
    """

    def __init__(
        self,
        pixel_count: int,
        *,
        gpio: int = 10,
        frequency_hz: int = 800_000,
        dma_channel: int = 10,
        invert: bool = False,
        channel: int = 0,
    ) -> None:
        if gpio != 10:
            raise ValueError("NeoPixel SPI output requires GPIO 10 / SPI0 MOSI")
        if invert:
            raise ValueError("NeoPixel SPI output does not support inversion")
        if channel != 0:
            raise ValueError("NeoPixel SPI output requires channel 0")
        # These arguments remain accepted so existing launch commands and
        # manifests do not break. NeoPixel_SPI owns the SPI waveform settings.
        del frequency_hz, dma_channel

        try:
            import board
            import neopixel_spi as neopixel
        except ImportError as error:
            raise RuntimeError(
                "adafruit-blinka and adafruit-circuitpython-neopixel-spi are "
                "required on the marker-grid Raspberry Pi"
            ) from error

        self._pixel_count = pixel_count
        self._pixels = neopixel.NeoPixel_SPI(
            board.SPI(),
            pixel_count,
            pixel_order=neopixel.GRB,
            auto_write=False,
            brightness=1.0,
        )

    def write(self, pixels: Sequence[RGB]) -> None:
        if len(pixels) != self._pixel_count:
            raise ValueError(
                f"expected {self._pixel_count} WS2811 values, got {len(pixels)}"
            )
        for index, color in enumerate(pixels):
            self._pixels[index] = tuple(int(component) for component in color)
        self._pixels.show()

    def close(self) -> None:
        for index in range(self._pixel_count):
            self._pixels[index] = (0, 0, 0)
        self._pixels.show()


class DryRunOutput:
    """No-hardware output used for validation and development."""

    def __init__(self, pixel_count: int) -> None:
        self.pixel_count = pixel_count
        self.pixels = tuple((0, 0, 0) for _ in range(pixel_count))
        self.write_count = 0

    def write(self, pixels: Sequence[RGB]) -> None:
        if len(pixels) != self.pixel_count:
            raise ValueError(
                f"expected {self.pixel_count} WS2811 values, got {len(pixels)}"
            )
        self.pixels = tuple(pixels)
        self.write_count += 1

    def close(self) -> None:
        self.pixels = tuple((0, 0, 0) for _ in range(self.pixel_count))
