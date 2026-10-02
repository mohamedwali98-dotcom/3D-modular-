"""Synthetic sheets for tests: white page, pen lines, dashed lines, arrows, dimension text."""
from __future__ import annotations

import cv2
import numpy as np

from s2c.reading import ReaderResult

INK = (25, 25, 25)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _i(p):
    return round(p[0]), round(p[1])


class Sheet:
    def __init__(self, w: int = 1600, h: int = 1131):
        self.img = np.full((h, w, 3), 250, np.uint8)
        self.texts: list[tuple[str, tuple[int, int, int, int]]] = []
        self.dims: list[tuple[tuple[float, float], tuple[float, float]]] = []  # arrow tips of each dimension

    def line(self, p, q, t=3):
        cv2.line(self.img, _i(p), _i(q), INK, t, cv2.LINE_AA)

    def dashed(self, p, q, dash=14, gap=9, t=2):
        p, q = np.float64(p), np.float64(q)
        length = np.linalg.norm(q - p)
        d = (q - p) / length
        s = 0.0
        while s < length:
            e = min(s + dash, length)
            self.line(p + d * s, p + d * e, t)
            s = e + gap

    def circle(self, c, r, t=3):
        cv2.circle(self.img, _i(c), round(r), INK, t, cv2.LINE_AA)

    def arrowhead(self, tip, direction, length=16, half=5):
        tip, d = np.float64(tip), np.float64(direction)
        d /= np.linalg.norm(d)
        n = np.array([-d[1], d[0]])
        base = tip - d * length
        cv2.fillPoly(self.img, [np.int32([tip, base + n * half, base - n * half])], INK,
                     cv2.LINE_AA)

    def text(self, s, centre, scale=0.8, t=2, rotate=False):
        """Draw `s` centred on `centre`. Ø is drawn as a slashed circle (Hershey fonts have none)."""
        parts = s.split("Ø")
        (_, th), base = cv2.getTextSize("0", FONT, scale, t)
        widths = [cv2.getTextSize(p, FONT, scale, t)[0][0] if p else 0 for p in parts]
        glyph = int(th * 1.1)
        W = sum(widths) + glyph * (len(parts) - 1) + 8
        H = th + base + 8
        tile = np.full((H, W, 3), 250, np.uint8)
        x = 4
        for k, part in enumerate(parts):
            if part:
                cv2.putText(tile, part, (x, th + 4), FONT, scale, INK, t, cv2.LINE_AA)
                x += widths[k]
            if k < len(parts) - 1:
                c, r = (x + glyph // 2, 4 + th // 2), int(th * 0.45)
                cv2.circle(tile, c, r, INK, t, cv2.LINE_AA)
                cv2.line(tile, (c[0] - r, c[1] + r), (c[0] + r, c[1] - r), INK, t, cv2.LINE_AA)
                x += glyph
        if rotate:
            tile = cv2.rotate(tile, cv2.ROTATE_90_COUNTERCLOCKWISE)
        h, w = tile.shape[:2]
        x0, y0 = int(centre[0] - w / 2), int(centre[1] - h / 2)
        region = self.img[y0:y0 + h, x0:x0 + w]
        np.minimum(region, tile, out=region)
        self.texts.append((s, (x0, y0, w, h)))

    def hdim(self, x1, x2, y_obj, y_line, text, scale=0.8):
        sgn = 1 if y_line > y_obj else -1
        for x in (x1, x2):
            self.line((x, y_obj + sgn * 6), (x, y_line + sgn * 8), 1)
        self.line((x1, y_line), (x2, y_line), 1)
        self.dims.append(((x1, y_line), (x2, y_line)))
        self.arrowhead((x1, y_line), (-1, 0))
        self.arrowhead((x2, y_line), (1, 0))
        self.text(text, ((x1 + x2) / 2, y_line - 16), scale=scale)

    def vdim(self, y1, y2, x_obj, x_line, text, rotate=False):
        sgn = 1 if x_line > x_obj else -1
        for y in (y1, y2):
            self.line((x_obj + sgn * 6, y), (x_line + sgn * 8, y), 1)
        self.line((x_line, y1), (x_line, y2), 1)
        self.dims.append(((x_line, y1), (x_line, y2)))
        self.arrowhead((x_line, y1), (0, -1))
        self.arrowhead((x_line, y2), (0, 1))
        self.text(text, (x_line + sgn * (20 if rotate else 36), (y1 + y2) / 2), rotate=rotate)

    def leader(self, tip, tail, text, scale=0.8):
        self.line(tail, tip, 1)
        self.arrowhead(tip, np.float64(tip) - np.float64(tail))
        self.text(text, (tail[0], tail[1] - 16), scale=scale)

    def bgr(self):
        return self.img.copy()

    def ink(self):
        return (cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY) < 128).astype(np.uint8) * 255


S = 4.0  # px per mm


def F(x, y):
    return 200 + S * x, 900 - S * y


def T(x, b):  # TOP: b is the distance from the front face (the view's bottom edge)
    return 200 + S * x, 520 - S * b


def R(a, y):  # SIDE: a is the distance from the front face (the view's left edge)
    return 800 + S * a, 900 - S * y


FRONT_OUTLINE = [(0, 0), (37.5, 0), (37.5, 12.5), (62.5, 12.5), (62.5, 0), (100, 0),
                 (100, 12.5), (75, 12.5), (75, 50), (62.5, 50), (62.5, 25), (37.5, 25),
                 (37.5, 50), (25, 50), (25, 12.5), (0, 12.5)]


def bridge_block(sh: Sheet, labels: bool = True) -> Sheet:
    # FRONT
    for p, q in zip(FRONT_OUTLINE, FRONT_OUTLINE[1:] + FRONT_OUTLINE[:1]):
        sh.line(F(*p), F(*q))
    for x in (6.25, 18.75, 81.25, 93.75):
        sh.dashed(F(x, 0), F(x, 12.5))
    for x0, x1 in ((25, 37.5), (62.5, 75)):
        for y in (31.25, 43.75):
            sh.dashed(F(x0, y), F(x1, y))
    bottom, top, left, right = F(0, 0)[1], F(0, 50)[1], F(0, 0)[0], F(100, 0)[0]
    sh.hdim(F(0, 0)[0], F(100, 0)[0], bottom, bottom + 90, "100")
    sh.hdim(F(0, 0)[0], F(37.5, 0)[0], bottom, bottom + 45, "37.5")
    sh.hdim(F(62.5, 0)[0], F(100, 0)[0], bottom, bottom + 45, "37.5")
    for x0, x1, t in ((25, 37.5, "12.5"), (37.5, 62.5, "25"), (62.5, 75, "12.5")):
        sh.hdim(F(x0, 0)[0], F(x1, 0)[0], top, top - 40, t)
    sh.vdim(F(0, 50)[1], F(0, 12.5)[1], left, left - 50, "37.5")
    for y0, y1, t in ((50, 25, "25"), (25, 12.5, "12.5"), (12.5, 0, "12.5")):
        sh.vdim(F(0, y0)[1], F(0, y1)[1], right, right + 50, t)
    # TOP
    corners = [T(0, 0), T(100, 0), T(100, 25), T(0, 25)]
    for p, q in zip(corners, corners[1:] + corners[:1]):
        sh.line(p, q)
    for x in (25, 37.5, 62.5, 75):
        sh.line(T(x, 0), T(x, 25))
    for x in (12.5, 87.5):
        sh.circle(T(x, 12.5), S * 6.25)
    for x0, x1 in ((25, 37.5), (62.5, 75)):
        for b in (6.25, 18.75):
            sh.dashed(T(x0, b), T(x1, b))
    tip = np.float64(T(87.5, 12.5)) + S * 6.25 * np.array([0.707, -0.707])
    sh.leader(tip, (tip[0] + 45, tip[1] - 45), "2xØ12.5")
    sh.hdim(T(0, 0)[0], T(12.5, 0)[0], T(0, 25)[1], T(0, 25)[1] - 35, "12.5")
    sh.vdim(T(0, 25)[1], T(0, 12.5)[1], T(0, 0)[0], T(0, 0)[0] - 50, "12.5")
    # SIDE
    corners = [R(0, 0), R(25, 0), R(25, 50), R(0, 50)]
    for p, q in zip(corners, corners[1:] + corners[:1]):
        sh.line(p, q)
    sh.line(R(0, 12.5), R(25, 12.5))
    sh.dashed(R(0, 25), R(25, 25))
    for a in (6.25, 18.75):
        sh.dashed(R(a, 0), R(a, 12.5))
    sh.circle(R(12.5, 37.5), S * 6.25)
    tip = np.float64(R(12.5, 37.5)) + S * 6.25 * np.array([-0.707, -0.707])
    sh.leader(tip, (tip[0] - 64, tip[1] - 60), "Ø12.5", scale=0.6)
    stop, sright = R(0, 50)[1], R(25, 0)[0]
    for a0, a1 in ((0, 12.5), (12.5, 25)):
        sh.hdim(R(a0, 0)[0], R(a1, 0)[0], stop, stop - 40, "12.5", scale=0.6)
    sh.vdim(R(0, 50)[1], R(0, 37.5)[1], sright, sright + 50, "12.5")
    sh.vdim(R(0, 50)[1], R(0, 0)[1], sright, sright + 110, "50")
    if labels:
        sh.text("TOP", (400, 575), scale=1.0)
        sh.text("FRONT", (400, 1060), scale=1.0)
        sh.text("SIDE", (850, 1060), scale=1.0)
    return sh


def boxes_overlap(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    return ix * iy / max(bw * bh, 1)


class TruthReader:
    """Reads a crop by looking up the synthetic text whose box it overlaps most.
    `swap` maps a true text to what this reader "sees"; `upright_only` reads only crops that are
    wider than tall, like a reader that cannot handle sideways text."""

    def __init__(self, texts, name="truth", confidence=0.95, swap=None, upright_only=False):
        self.texts, self.name, self.confidence = texts, name, confidence
        self.swap, self.upright_only = swap or {}, upright_only

    def read(self, crops):
        out = []
        for c in crops:
            best, score = "", 0.0
            for s, b in self.texts:
                ov = boxes_overlap(c.box, b)
                if ov > score:
                    best, score = s, ov
            h, w = c.image.shape[:2]
            if score < 0.3 or (self.upright_only and h > w):
                best = ""
            best = self.swap.get(best, best)
            out.append(ReaderResult(text=best, confidence=self.confidence if best else 0.0))
        return out
