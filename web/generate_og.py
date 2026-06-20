#!/usr/bin/env python3
"""Render public/og.png — the social share card for ttperf.

A 1200x630 image in the site's instrument identity: cool ink ground,
faint profiler grid, monospace type, and the device-kernel-duration
readout in amber. Real PNG (not SVG) so social crawlers render it.

Run from the repo root:
    python3 web/generate_og.py
"""

import json
import os

from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO_ROOT, "public", "og.png")
OPS_JSON = os.path.join(REPO_ROOT, "public", "ops.json")

W, H = 1200, 630

# identity palette (mirrors style.css)
INK = (11, 14, 17)
INK_2 = (17, 22, 27)
LINE = (35, 44, 53)
LINE_2 = (46, 58, 69)
FG = (230, 237, 243)
FG_DIM = (147, 161, 175)
FG_FAINT = (92, 107, 121)
AMBER = (255, 192, 97)
PASS = (92, 214, 138)

MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
MONO_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"


def font(path, size):
    return ImageFont.truetype(path, size)


def op_count():
    try:
        with open(OPS_JSON) as f:
            return json.load(f)["count"]
    except Exception:
        return 263


def main():
    img = Image.new("RGB", (W, H), INK)
    d = ImageDraw.Draw(img)

    # faint profiler grid
    for x in range(0, W, 48):
        d.line([(x, 0), (x, H)], fill=INK_2, width=1)
    for y in range(0, H, 48):
        d.line([(0, y), (W, y)], fill=INK_2, width=1)

    # amber baseline = the "time axis" the tool measures along
    base_y = 470
    d.line([(80, base_y), (W - 80, base_y)], fill=LINE_2, width=2)
    for i, x in enumerate(range(80, W - 79, (W - 160) // 6)):
        d.line([(x, base_y - 6), (x, base_y + 6)], fill=FG_FAINT, width=2)

    f_word = font(MONO_BOLD, 30)
    f_h1 = font(MONO_BOLD, 76)
    f_sub = font(MONO, 27)
    f_label = font(MONO, 22)
    f_ns = font(MONO_BOLD, 58)
    f_meta = font(MONO, 22)

    # wordmark
    d.text((80, 70), ">_", font=f_word, fill=AMBER)
    d.text((132, 70), "ttperf", font=f_word, fill=FG)

    # eyebrow
    d.text((80, 132), "TT-METAL KERNEL PROFILER · CLI", font=f_label, fill=FG_FAINT)

    # headline (two lines)
    d.text((78, 182), "Profile a kernel", font=f_h1, fill=FG)
    d.text((78, 262), "in ", font=f_h1, fill=FG)
    one_w = d.textlength("in ", font=f_h1)
    d.text((78 + one_w, 262), "one command.", font=f_h1, fill=AMBER)

    # the readout — the number you came for
    d.text((80, 372), "DEVICE KERNEL DURATION [ns] total", font=f_sub, fill=FG_DIM)
    ns_text = "1,234,567.89"
    d.text((80, 402), ns_text, font=f_ns, fill=AMBER)
    ns_w = d.textlength(ns_text, font=f_ns)
    d.text((80 + ns_w + 16, 426), "ns", font=f_sub, fill=FG_DIM)

    # meta row under the axis
    meta = f"{op_count()} operations   ·   one command   ·   pip install ttperf"
    d.text((80, base_y + 40), meta, font=f_meta, fill=FG_FAINT)
    # domain, right-aligned
    dom = "ttperf.aswincloud.com"
    dom_w = d.textlength(dom, font=f_meta)
    d.text((W - 80 - dom_w, base_y + 40), dom, font=f_meta, fill=AMBER)

    img.save(OUT, "PNG")
    print(f"Wrote {os.path.relpath(OUT, REPO_ROOT)}  ({W}x{H})")


if __name__ == "__main__":
    main()
