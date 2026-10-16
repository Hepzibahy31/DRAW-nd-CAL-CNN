"""
Generates the training images: 10 digits + 5 operators = 15 classes.
Each character is drawn with a virtual pen (strokes with a thick round brush),
then randomly changed (size, tilt, slant, wobble, pen thickness) so that every
sample looks different, like real handwriting drawn with a mouse.
Real handwritten digits (scikit-learn) are added as well, see train.py.
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from preprocess import normalize_char, CLASSES


def arc(cx, cy, rx, ry, a0, a1, n=18):
    t = np.radians(np.linspace(a0, a1, n))
    return [(cx + rx * np.cos(a), cy + ry * np.sin(a)) for a in t]


# strokes live in a 0..1 box, y goes down. Several ways to write each character.
TEMPLATES = {
    "0": [[arc(.5, .5, .32, .5, 0, 360, 28)], [arc(.5, .5, .24, .5, 250, 625, 28)]],
    "1": [[[(.5, 0), (.5, 1)]], [[(.28, .22), (.5, 0), (.5, 1)]], [[(.3, .2), (.5, 0), (.5, 1)], [(.3, 1), (.7, 1)]]],
    "2": [[arc(.5, .28, .3, .26, 180, 400) + [(.2, 1), (.82, 1)]], [arc(.5, .28, .3, .26, 190, 410) + [(.18, .98), (.85, .98)]]],
    "3": [[arc(.48, .26, .3, .24, 200, 450) + arc(.48, .74, .32, .26, 270, 520)],
          [[(.2, .05), (.8, .05), (.45, .45)] + arc(.48, .72, .32, .27, 250, 520)]],
    "4": [[[(.65, 0), (.15, .65), (.88, .65)], [(.65, .25), (.65, 1)]], [[(.25, 0), (.25, .6), (.8, .6)], [(.7, .15), (.7, 1)]]],
    "5": [[[(.78, 0), (.3, 0), (.27, .46)] + arc(.46, .70, .34, .28, 250, 500)],
          [[(.8, 0), (.3, 0)], [(.3, 0), (.27, .46)] + arc(.46, .70, .34, .28, 250, 500)]],
    "6": [[[(.68, .02), (.38, .28), (.2, .65)] + arc(.5, .68, .3, .3, 180, 540)], [arc(.5, .68, .3, .3, 0, 360), [(.7, .05), (.3, .45)]]],
    "7": [[[(.15, .02), (.85, .02), (.4, 1)]], [[(.15, .02), (.85, .02), (.4, 1)], [(.3, .55), (.7, .55)]]],
    "8": [[arc(.5, .26, .26, .25, 90, 450) + arc(.5, .74, .32, .26, 270, 630)], [arc(.5, .26, .26, .25, 0, 360), arc(.5, .74, .32, .26, 0, 360)]],
    "9": [[arc(.5, .3, .28, .28, 0, 360), [(.78, .3), (.7, .7), (.42, 1)]], [arc(.5, .3, .28, .28, 0, 360), [(.78, .3), (.75, 1)]]],
    "+": [[[(.1, .5), (.9, .5)], [(.5, .1), (.5, .9)]]],
    "-": [[[(.1, .5), (.9, .5)]], [[(.1, .55), (.5, .47), (.9, .5)]]],
    "×": [[[(.15, .15), (.85, .85)], [(.85, .15), (.15, .85)]]],
    "÷": [[[(.1, .5), (.9, .5)], [(.5, .1), (.5, .14)], [(.5, .86), (.5, .9)]]],
    "=": [[[(.12, .35), (.88, .35)], [(.12, .65), (.88, .65)]]],
}


def wobble(points, rng, amount):
    """Densify a stroke and add a slow random wobble, like a shaky hand."""
    pts = np.array(points, dtype=np.float32)
    dense = [pts[0]]
    for a, b in zip(pts[:-1], pts[1:]):
        k = max(2, int(np.hypot(*(b - a)) * 22))
        dense += [a + (b - a) * (i / k) for i in range(1, k + 1)]
    dense = np.array(dense)
    noise = np.cumsum(rng.normal(0, amount, dense.shape), axis=0)
    noise -= np.linspace(noise[0], noise[-1], len(dense))        # keep start and end in place
    return dense + noise * 0.35 + rng.normal(0, amount * 0.4, dense.shape)


def draw_char(label, rng, size=112):
    strokes = TEMPLATES[label][rng.integers(len(TEMPLATES[label]))]
    ang = np.radians(rng.uniform(-13, 13)); shear = rng.uniform(-0.22, 0.22)
    sx, sy = rng.uniform(0.7, 1.25), rng.uniform(0.7, 1.25)
    M = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]]) @ np.array([[1, shear], [0, 1]]) @ np.diag([sx, sy])
    width = max(3, int(rng.uniform(3.5, 11)))
    img = Image.new("L", (size, size), 0)
    d = ImageDraw.Draw(img)
    for st in strokes:
        p = wobble(st, rng, 0.012) - 0.5
        p = (p @ M.T) * (size * 0.62) + size / 2
        xy = [tuple(q) for q in p]
        d.line(xy, fill=255, width=width, joint="curve")
        for (x, y) in (xy[0], xy[-1]):
            d.ellipse([x - width / 2, y - width / 2, x + width / 2, y + width / 2], fill=255)
    return np.asarray(img)


def make_char_image(label, rng):
    ink = draw_char(label, rng)
    x = normalize_char(ink)
    # light camera-like changes: blur, noise, 1-pixel shift
    if rng.random() < 0.4:
        x = np.asarray(Image.fromarray((x * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 0.8))), np.float32) / 255
    x = np.clip(x + rng.normal(0, 0.02, x.shape) * (rng.random() < 0.5), 0, 1)
    return x.astype(np.float32)


def build(per_class=700, seed=3):
    rng = np.random.default_rng(seed)
    X, y = [], []
    for ci, c in enumerate(CLASSES):
        for _ in range(per_class):
            X.append(make_char_image(c, rng)); y.append(ci)
    return np.array(X, np.float32)[:, None], np.array(y, np.int64)


if __name__ == "__main__":
    X, y = build(40)
    print("Generated", X.shape[0], "sample images of shape", X.shape[1:])
