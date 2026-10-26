"""A pg-widgets widget for displaying a single image.

pg-widgets does not ship an image widget, but ``UIElement`` is designed to be
subclassed: implementing ``render`` to return a ``pygame.Surface`` is enough.
This widget scales the image to fit the available space and centre it, so the
original aspect ratio is preserved.
"""

import pygame as pg

import pg_widgets as pw


class ImageView(pw.UIElement):
    """Display a ``pygame.Surface`` with letterboxing."""

    def __init__(self, pos, size=(1.0, 1.0)):
        super().__init__(pos, size)
        self._image: pg.Surface | None = None

    def setImage(self, image: pg.Surface | None) -> None:
        """Set the image to show. Pass ``None`` to clear the view."""
        self._image = image
        self._updateSurf = True

    def getImage(self) -> pg.Surface | None:
        """Return the currently displayed image, if any."""
        return self._image

    def _getFittedSize(self) -> tuple[int, int] | None:
        """Return the image size after fitting it inside the widget."""
        if self._image is None:
            return None

        width, height = self.getSize()
        imageWidth, imageHeight = self._image.get_size()
        if width <= 0 or height <= 0 or imageWidth <= 0 or imageHeight <= 0:
            return None

        scale = min(width / imageWidth, height / imageHeight)
        fittedWidth = max(1, int(round(imageWidth * scale)))
        fittedHeight = max(1, int(round(imageHeight * scale)))

        # Rounding can make a fitted dimension one pixel larger than the widget.
        fittedWidth = min(fittedWidth, max(1, int(width)))
        fittedHeight = min(fittedHeight, max(1, int(height)))
        return fittedWidth, fittedHeight

    def render(self, bgColor=(0, 0, 0)):
        if self._updateSurfBase:
            self._updateSurfBase = False
            width, height = self.getSize()
            self._surfBase = pg.Surface(
                (max(1, int(width)), max(1, int(height)))
            )
            self._surfBase.fill(bgColor)

        if self._updateSurf:
            self._updateSurf = False
            self._surf = self._surfBase.copy()

            fittedSize = self._getFittedSize()
            if fittedSize is not None:
                if fittedSize == self._image.get_size():
                    scaled = self._image
                else:
                    scaled = pg.transform.smoothscale(self._image, fittedSize)

                x = (self._surf.get_width() - fittedSize[0]) // 2
                y = (self._surf.get_height() - fittedSize[1]) // 2
                self._surf.blit(scaled, (x, y))

        return self._surf
