"""
Shared image preparation. The SAME code prepares training images and the drawings
that come from the web page, so the CNN always sees the same kind of input.
"""
import numpy as np
from PIL import Image

CLASSES = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "+", "-", "×", "÷", "="]


def normalize_char(ink):
    """ink: 2D uint8 array (255 = pen, 0 = background), any size.
    Crop to the pen, scale the longest side to 20 px (keep shape), and center it
    in a 28x28 image by its center of mass (the MNIST way). Returns float32 0..1."""
    ys, xs = np.where(ink > 40)
    if len(ys) == 0:
        return np.zeros((28, 28), np.float32)
    crop = ink[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = crop.shape
    s = 20.0 / max(h, w)
    nh, nw = max(1, int(round(h * s))), max(1, int(round(w * s)))
    small = np.asarray(Image.fromarray(crop).resize((nw, nh), Image.LANCZOS), dtype=np.float32) / 255.0
    canvas = np.zeros((28, 28), np.float32)
    cy, cx = (small.sum(1) @ np.arange(nh)) / max(small.sum(), 1e-6), (small.sum(0) @ np.arange(nw)) / max(small.sum(), 1e-6)
    oy = int(np.clip(round(14 - cy), 0, 28 - nh))
    ox = int(np.clip(round(14 - cx), 0, 28 - nw))
    canvas[oy:oy + nh, ox:ox + nw] = small
    return canvas


def split_characters(ink, min_gap=None):
    """Cut a wide drawing into separate characters wherever there is an empty gap
    between them (looking at the pen column by column). Returns a list of (x0, x1) ranges."""
    mask = ink > 60
    cols = mask.any(axis=0)
    gap = min_gap or max(8, int(0.012 * ink.shape[1]))
    runs, start = [], None
    for x, on in enumerate(cols):
        if on and start is None:
            start = x
        if not on and start is not None:
            runs.append([start, x - 1]); start = None
    if start is not None:
        runs.append([start, len(cols) - 1])
    merged = []
    for r in runs:
        if merged and r[0] - merged[-1][1] < gap:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    return [(a, b) for a, b in merged if mask[:, a:b + 1].sum() >= 30]
