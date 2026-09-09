"""WS2811 output adapter for Raspberry Pi GPIO 10 / SPI0 MOSI."""

from __future__ import annotations

from typing import Sequence

from .controller import RGB


class Ws2811Output:
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
        try:
            import rpi_ws281x as ws
        except ImportError as error:
            raise RuntimeError(
                "rpi_ws281x is required on the marker-grid Raspberry Pi"
            ) from error

        # RGB is intentional: these are bare WS2811 R/G/B outputs, not a GRB
        # packaged LED. Global brightness remains 255; the two logical output
        # levels are applied per channel by MarkerGridController.
        self._strip = ws.PixelStrip(
            num=pixel_count,
            pin=gpio,
            freq_hz=frequency_hz,
            dma=dma_channel,
            invert=invert,
            brightness=255,
            channel=channel,
            strip_type=ws.WS2811_STRIP_RGB,
        )
        self._strip.begin()
        self._pixel_count = pixel_count

    def write(self, pixels: Sequence[RGB]) -> None:
        if len(pixels) != self._pixel_count:
            raise ValueError(
                f"expected {self._pixel_count} WS2811 values, got {len(pixels)}"
            )
        for index, (red, green, blue) in enumerate(pixels):
            self._strip.setPixelColorRGB(index, red, green, blue)
        self._strip.show()

    def close(self) -> None:
        for index in range(self._pixel_count):
            self._strip.setPixelColorRGB(index, 0, 0, 0)
        self._strip.show()


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
