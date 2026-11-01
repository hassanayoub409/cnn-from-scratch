"""Turn arrays into RGB images using the project colour palette (no matplotlib needed).

Colour rules used everywhere in the app:
  green  = "present / strong" (ink in a digit, a filter finding its pattern)
  gold   = "wants blank" (negative weight in a filter pattern)
  red    = only for the current window outline
  grey   = not computed yet
"""
import numpy as np

WHITE = (255, 255, 255)
GREEN = (0x4F, 0x7D, 0x32)
AMBER = (0xF5, 0xB5, 0x2E)
RED = (0xD9, 0x36, 0x1E)
GREY = (232, 232, 232)


def ramp(t, start, end):
    """Blend from `start` (t=0) to `end` (t=1)."""
    t = np.clip(np.asarray(t, dtype=float), 0.0, 1.0)[..., None]
    return (np.array(start) * (1 - t) + np.array(end) * t).astype(np.uint8)


def intensity(m, vmax=None):
    """Any non-negative map (a digit, a feature map): white = nothing, dark green = strong."""
    m = np.asarray(m, dtype=float)
    vmax = vmax if vmax else max(float(m.max()), 1e-9)
    return ramp(m / vmax, WHITE, GREEN)


def filter_image(w, vmax=None):
    """A filter pattern: green = wants ink here (positive weight), gold = wants blank here (negative)."""
    w = np.asarray(w, dtype=float)
    vmax = vmax if vmax else max(float(np.abs(w).max()), 1e-9)
    t = w / vmax
    return np.where((t >= 0)[..., None], ramp(t, WHITE, GREEN), ramp(-t, WHITE, AMBER))


def partial_map(values, filled, vmax):
    """Like intensity(), but cells that are not computed yet (filled=False) are grey."""
    rgb = intensity(np.maximum(values, 0), vmax)
    rgb[~filled] = GREY
    return rgb


def upscale(rgb, k):
    return np.repeat(np.repeat(rgb, k, axis=0), k, axis=1)


def draw_box(rgb_up, row, col, size, k, color=RED, thickness=3):
    """Outline a size x size block of cells (cell coordinates) on an upscaled image."""
    out = rgb_up.copy()
    y0, x0 = row * k, col * k
    y1, x1 = min((row + size) * k, out.shape[0]), min((col + size) * k, out.shape[1])
    out[y0:y0 + thickness, x0:x1] = color
    out[y1 - thickness:y1, x0:x1] = color
    out[y0:y1, x0:x0 + thickness] = color
    out[y0:y1, x1 - thickness:x1] = color
    return out
