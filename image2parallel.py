from typing import Any
from PIL import ImageColor
import svgwrite
import numpy as np


def fade_color(color: tuple, fade_amount: int) -> tuple:
    r = max(0, color[0] - fade_amount)
    g = max(0, color[1] - fade_amount)
    b = max(0, color[2] - fade_amount)
    return r, g, b, color[3]


def get_rgba_tuple(color: Any) -> tuple:
    """

    :param color:
    :return: (R, G, B, A) tuple
    """
    if isinstance(color, tuple):
        rgba = color
    elif isinstance(color, int):
        rgba = (color >> 16 & 0xff, color >> 8 & 0xff, color & 0xff, color >> 24 & 0xff)
    else:
        rgba = ImageColor.getrgb(color)

    if len(rgba) == 3:
        rgba = (rgba[0], rgba[1], rgba[2], 255)
    return rgba


class RectShape:
    def __init__(self) -> None:
        self.x1: int = 0
        self.x2: int = 0
        self.y1: int = 0
        self.y2: int = 0
        self._fill: Any = None
        self._outline: Any = None

    @property
    def fill(self):
        return self._fill

    @property
    def outline(self):
        return self._outline

    @fill.setter
    def fill(self, v):
        self._fill = get_rgba_tuple(v)

    @outline.setter
    def outline(self, v):
        self._outline = get_rgba_tuple(v)


class SVGPixel(RectShape):
    
    def __init__(self) -> None:
        super().__init__()
        self.de: int = 0
        self.shade: int = 0

    def draw_line(self, draw: svgwrite.Drawing, coords, pen):
        draw.add(draw.line(coords[:2], coords[2:], stroke=pen))

    def draw_polygon(self, draw: svgwrite.Drawing, coords, pen, brush):
        draw.add(draw.polygon(points=coords, stroke=pen, fill=brush))

    def draw_rect(self, draw: svgwrite.Drawing, coords, pen, brush):
        draw.add(draw.rect(insert=coords[:2], size=(coords[2] - coords[0], coords[3] - coords[1]), stroke=pen, fill=brush))

    def draw(self, draw):

        brush = svgwrite.rgb(*self.fill[:3], 'RGB')
        pen = svgwrite.rgb(*self.outline[:3], 'RGB')

        if hasattr(self, 'de') and self.de > 0:            
            brush_s1 = svgwrite.rgb(*fade_color(self.fill, self.shade)[:3], 'RGB')
            brush_s2 = svgwrite.rgb(*fade_color(self.fill, 2 * self.shade)[:3], 'RGB')

            self.draw_line(draw, [self.x1 + self.de, self.y1 - self.de, self.x1 + self.de, self.y2 - self.de], pen)
            self.draw_line(draw, [self.x1 + self.de, self.y2 - self.de, self.x1, self.y2], pen)
            self.draw_line(draw, [self.x1 + self.de, self.y2 - self.de, self.x2 + self.de, self.y2 - self.de], pen)

            self.draw_polygon(draw, [(self.x1, self.y1),
                            (self.x1 + self.de, self.y1 - self.de),
                            (self.x2 + self.de, self.y1 - self.de),
                            (self.x2, self.y1)
                            ], pen, brush_s1)

            self.draw_polygon(draw, [(self.x2 + self.de, self.y1 - self.de),
                            (self.x2, self.y1),
                            (self.x2, self.y2),
                            (self.x2 + self.de, self.y2 - self.de)
                            ], pen, brush_s2)

        self.draw_rect(draw, [self.x1, self.y1, self.x2, self.y2], pen, brush)


def compute_viewbox(drawing):
    min_x = 0
    max_x = 0
    min_y = 0
    max_y = 0

    for elem in drawing.elements:
        if elem.elementname == 'rect':
            min_x = min(min_x, elem['x'])
            max_x = max(max_x, elem['x'] + elem['width'])
            min_y = min(min_y, elem['y'])
            max_y = max(max_y, elem['y'] + elem['height'])
        elif elem.elementname == 'line':
            min_x = min(min_x, elem['x1'], elem['x2'])
            max_x = max(max_x, elem['x1'], elem['x2'])
            min_y = min(min_y, elem['y1'], elem['y2'])
            max_y = max(max_y, elem['y1'], elem['y2'])
        elif elem.elementname == 'polygon':
            for x, y in elem.points:
                min_x = min(min_x, x)
                max_x = max(max_x, x)
                min_y = min(min_y, y)
                max_y = max(max_y, y)
        elif elem.elementname == 'defs':
            continue
        else:
            raise ValueError(f"Unsupported element {elem.elementname}")
    
    return min_x, min_y, max_x, max_y


def draw_parallel(drawing, 
                  img: np.array, 
                  s: int =50, 
                  use_rgb: bool = False, 
                  cmap=None, 
                  vmin=None, 
                  vmax=None, 
                  outline_color=(100, 100, 100, 100), 
                  margin=0, 
                  draw_hidden=False):

    padding_x = 0
    padding_y = 0    

    # add a channel dimension for 2D images
    if len(img.shape) == 2:
        img = np.expand_dims(img, axis=2)

    if use_rgb:
        assert len(img.shape) == 3 and img.shape[2] == 3, "Input must be in RGB (in WHC order) for use_rgb"
    else:
        assert cmap is not None, "cmap must be provided if not use_rgb"
        if vmin is None:
            vmin = img.min()
        if vmax is None:
            vmax = img.max()
        img = (img - vmin) / (vmax - vmin)

    # we need to draw bottom to top!, left to right, front to back

    for k in range(img.shape[1]): # depth
        for j in range(img.shape[2]): # width
            for i in range(img.shape[0]): # height
                if not draw_hidden:
                    if j < img.shape[2] - 1 and k != (img.shape[1] - 1) and i != (img.shape[0] - 1):
                        continue

                box = SVGPixel()
                box.outline = outline_color

                if use_rgb:
                    color = img[img.shape[0] - i - 1, img.shape[1] - k - 1]
                    if j == 2:
                        color = (color[0], 0, 0, 255)
                    elif j == 1:
                        color = (0, color[1], 0, 255)
                    elif j == 0:
                        color = (0, 0, color[2], 255)
                else:  # if not use_rgb, we need to get the value and project it to the colormap
                    value = img[img.shape[0] - i - 1, img.shape[1] - k - 1, img.shape[2] - j - 1]
                    color = (np.asarray(cmap(value)) * 255).round().astype(np.uint8)

                box.fill = tuple(color)

                box.x1 = padding_x + s * j - (k * s) / 2
                box.x2 = box.x1 + s
                box.y1 = padding_y + (img.shape[1] * s) - s * i + (k * s) / 2
                box.y2 = box.y1 + s
                box.de = s / 2

                box.shade = 0
                box.draw(drawing)

    # fit the viewbox
    min_x, min_y, max_x, max_y = compute_viewbox(drawing)
    min_x -= margin
    min_y -= margin
    max_x += margin
    max_y += margin
    drawing.viewbox(min_x, min_y, max_x - min_x, max_y - min_y)
