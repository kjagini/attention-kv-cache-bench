"""Generate the animated GIFs and charts for the attention article.
Run:  python make_gifs.py   (outputs next to this file)
Priority is clarity over polish: boxes, arrows, and dots that flow along the arrows.
"""
import math
import os

from PIL import Image, ImageDraw, ImageFont

OUT = os.path.dirname(os.path.abspath(__file__))
W, H = 960, 560
BG = (255, 255, 255)
INK = (30, 30, 30)
BLUE = (66, 133, 244)
ORANGE = (251, 140, 0)
GREEN = (46, 160, 90)
GREY = (150, 150, 150)
RED = (220, 50, 47)


def font(size, bold=False):
    for name in ("arialbd.ttf" if bold else "arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


F_T, F_M, F_S = font(30, True), font(20), font(16)


def text(d, xy, s, f=F_M, fill=INK, anchor="mm"):
    d.text(xy, s, font=f, fill=fill, anchor=anchor)


def box(d, cx, cy, w, h, fill, label="", dashed=False, f=F_S, outline=INK):
    x0, y0, x1, y1 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
    if dashed:
        d.rectangle([x0, y0, x1, y1], fill=(245, 245, 245))
        dash(d, (x0, y0), (x1, y0)); dash(d, (x1, y0), (x1, y1))
        dash(d, (x1, y1), (x0, y1)); dash(d, (x0, y1), (x0, y0))
    else:
        d.rectangle([x0, y0, x1, y1], fill=fill, outline=outline, width=2)
    if label:
        text(d, (cx, cy), label, f)


def dash(d, a, b, on=6, off=5, fill=GREY, width=2):
    L = math.dist(a, b)
    n = int(L // (on + off)) + 1
    for i in range(n):
        s = i * (on + off) / L
        e = min((i * (on + off) + on) / L, 1)
        d.line([a[0] + (b[0] - a[0]) * s, a[1] + (b[1] - a[1]) * s,
                a[0] + (b[0] - a[0]) * e, a[1] + (b[1] - a[1]) * e], fill=fill, width=width)


def arrow(d, a, b, fill=GREY, head=10):
    d.line([a, b], fill=fill, width=2)
    ang = math.atan2(b[1] - a[1], b[0] - a[0])
    for s in (-0.45, 0.45):
        d.line([b, (b[0] - head * math.cos(ang + s), b[1] - head * math.sin(ang + s))], fill=fill, width=2)


def dots(d, a, b, t, n=3, r=6, fill=RED):
    for k in range(n):
        p = (t + k / n) % 1
        x, y = a[0] + (b[0] - a[0]) * p, a[1] + (b[1] - a[1]) * p
        d.ellipse([x - r, y - r, x + r, y + r], fill=fill)


def save(frames, name, ms=70):
    frames[0].save(os.path.join(OUT, name), save_all=True, append_images=frames[1:],
                   duration=ms, loop=0, optimize=True)
    print("wrote", name, len(frames), "frames")


# --------------------------------------------------------------------------
# 1) One GIF per attention method: who reads which cache entry
# --------------------------------------------------------------------------
QX = [180, 380, 580, 780]


def attention_gif(name, title, subtitle, groups, cache_label, bar_frac, bar_text, latent=False):
    frames, N = [], 24
    for f in range(N):
        t = f / N
        im = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(im)
        text(d, (W / 2, 32), title, F_T)
        text(d, (W / 2, 68), subtitle, F_M, GREY)
        qy = 120
        for i, x in enumerate(QX):
            d.ellipse([x - 38, qy - 38, x + 38, qy + 38], fill=BLUE, outline=INK, width=2)
            text(d, (x, qy), f"Q{i+1}", F_M, (255, 255, 255))
        text(d, (60, qy), "query\nheads", F_S, GREY)
        if not latent:
            cy = 350
            for gi, qs in enumerate(groups):
                cx = sum(QX[q] for q in qs) / len(qs)
                box(d, cx, cy, 120, 56, ORANGE, f"K,V #{gi+1}")
                for q in qs:
                    a, b = (cx, cy - 28), (QX[q], qy + 40)
                    arrow(d, a, b); dots(d, a, b, t)
            text(d, (W / 2, 420), cache_label, F_S, GREY)
        else:
            ey, ly = 245, 400
            for i, x in enumerate(QX):
                box(d, x, ey, 110, 46, None, "K,V (rebuilt)", dashed=True)
                a, b = (x, ey - 23), (x, qy + 40)
                arrow(d, a, b); dots(d, a, b, t)
                a2, b2 = (W / 2, ly - 25), (x, ey + 24)
                arrow(d, a2, b2); dots(d, a2, b2, t, n=2)
            box(d, W / 2, ly, 150, 46, GREEN, "latent c  (small)")
            text(d, (W / 2, 450), cache_label, F_S, GREY)
            text(d, (800, 400), "dashed boxes are rebuilt\nevery step, never stored", F_S, GREY)
        # size bar
        by = 500
        text(d, (60, by), "cache\nsize", F_S, GREY)
        d.rectangle([110, by - 14, 110 + 560, by + 14], outline=INK, width=2)
        d.rectangle([110, by - 14, 110 + 560 * bar_frac, by + 14], fill=ORANGE if not latent else GREEN)
        text(d, (690, by), bar_text, F_M, INK, "lm")
        frames.append(im)
    save(frames, name)


# --------------------------------------------------------------------------
# 2) Prefill vs decode
# --------------------------------------------------------------------------
def prefill_decode_gif():
    frames, N = [], 48
    for f in range(N):
        im = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(im)
        d.line([W / 2, 20, W / 2, H - 20], fill=GREY, width=2)
        text(d, (240, 30), "PREFILL", F_T, BLUE)
        text(d, (240, 62), "whole prompt, all at once", F_S, GREY)
        text(d, (720, 30), "DECODE", F_T, RED)
        text(d, (720, 62), "one new token per step", F_S, GREY)

        # ---- prefill (loops twice) ----
        t = (f % 24) / 24
        xs = [90 + i * 75 for i in range(5)]
        for i, x in enumerate(xs):
            box(d, x, 120, 56, 40, BLUE, f"t{i+1}")
            box(d, x, 440, 56, 40, ORANGE, "K,V")
            a, b = (x, 142), (240, 220)
            arrow(d, a, b); dots(d, a, b, t, n=2)
            a, b = (240, 300), (x, 418)
            arrow(d, a, b); dots(d, a, b, t, n=2)
        box(d, 240, 260, 260, 80, (235, 235, 235), "model: one big\nmatrix multiply")
        text(d, (240, 500), "compute-bound: GPU is busy", F_M, INK)

        # ---- decode: 4 steps x 12 frames ----
        s, ph = divmod(f, 12)
        xs2 = [525 + i * 50 for i in range(8)]
        filled = 5 + s
        for i, x in enumerate(xs2):
            if i < filled:
                box(d, x, 440, 42, 40, ORANGE, "K,V", f=font(12))
            else:
                box(d, x, 440, 42, 40, None, dashed=True)
        box(d, 720, 260, 260, 80, (235, 235, 235), "model: tiny\nmatrix-vector")
        read = ph < 7
        for i in range(filled):
            a, b = (xs2[i], 418), (720, 300)
            arrow(d, a, b)
            if read:
                dots(d, a, b, ph / 7, n=1)
        out_x = 720
        box(d, out_x, 120, 90, 40, RED, f"new token {s+1}") if ph >= 7 else box(d, out_x, 120, 90, 40, None, dashed=True)
        a, b = (720, 220), (720, 142)
        arrow(d, a, b)
        if ph >= 7:
            dots(d, a, b, (ph - 7) / 5, n=1)
        text(d, (720, 500), "memory-bound: waits on the cache", F_M, INK)
        text(d, (720, 530), "every step re-reads ALL past K,V", F_S, GREY)
        frames.append(im)
    save(frames, "00_prefill_vs_decode.gif", ms=90)


if __name__ == "__main__":
    prefill_decode_gif()
    attention_gif("01_mha.gif", "MHA: every head has its own K,V",
                  "4 query heads -> 4 separate cache entries (schematic)",
                  [[0], [1], [2], [3]], "each decode step reads ALL of these", 1.0, "1x (baseline)")
    attention_gif("02_mqa.gif", "MQA: all heads share ONE K,V",
                  "4 query heads -> 1 cache entry (schematic)",
                  [[0, 1, 2, 3]], "each decode step reads just this one", 1 / 32, "1/32 (our big model)")
    attention_gif("03_gqa.gif", "GQA: heads share K,V in groups",
                  "4 query heads -> 2 groups (schematic)",
                  [[0, 1], [2, 3]], "a middle ground between MHA and MQA", 1 / 4, "1/4 (our big model)")
    attention_gif("04_mla.gif", "MLA: store a tiny latent, rebuild K,V",
                  "cache holds a compressed vector; K,V are re-expanded each step",
                  [], "trade: smaller cache, extra compute every step", 1 / 16, "1/16 (our big model)",
                  latent=True)
