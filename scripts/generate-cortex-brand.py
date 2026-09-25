#!/usr/bin/env python3
"""Generate the CORTEX brand assets shipped by the web app and the Android app.

The mark is a hexagonal "cortex" network: six outer nodes wired to a hub. It is
drawn deterministically from the palette below so every asset (web favicon, PWA
icons, splash screens, Android launcher icons) stays visually consistent.

Run from the repository root:

    python3 scripts/generate-cortex-brand.py

Only the files listed at the bottom are written.
"""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent

# CORTEX palette. The gradient is the primary brand surface.
BLUE = (37, 99, 235)
VIOLET = (124, 58, 237)
CYAN = (34, 211, 238)
INK = (8, 12, 24)
PAPER = (250, 250, 252)

WEB_STATIC_DIRS = [ROOT / 'static' / 'static', ROOT / 'backend' / 'open_webui' / 'static']
ANDROID_ASSETS_DIR = ROOT / 'assets'
UPSCALE = 4  # supersampling factor, keeps the thin spokes clean when downscaled


def gradient(size: int) -> Image.Image:
    """Diagonal blue -> violet gradient used as the mark's fill."""
    img = Image.new('RGBA', (size, size))
    px = img.load()
    for y in range(size):
        for x in range(size):
            t = (x + y) / (2 * (size - 1))
            px[x, y] = (
                round(BLUE[0] + (VIOLET[0] - BLUE[0]) * t),
                round(BLUE[1] + (VIOLET[1] - BLUE[1]) * t),
                round(BLUE[2] + (CYAN[2] - BLUE[2]) * t),
                255,
            )
    return img


def hexagon(cx: float, cy: float, r: float) -> list[tuple[float, float]]:
    """Pointy-top hexagon vertices."""
    return [
        (cx + r * math.cos(math.radians(-90 + 60 * i)), cy + r * math.sin(math.radians(-90 + 60 * i)))
        for i in range(6)
    ]


def mark_mask(size: int, scale: float = 0.40, alpha: int = 255) -> Image.Image:
    """White-on-transparent glyph mask, drawn oversized then downsampled."""
    big = size * UPSCALE
    mask = Image.new('L', (big, big), 0)
    d = ImageDraw.Draw(mask)
    cx = cy = big / 2
    r = big * scale
    vertices = hexagon(cx, cy, r)
    hub_r = big * 0.062
    node_r = big * 0.052

    d.line(vertices + [vertices[0]], fill=alpha, width=round(big * 0.030), joint='curve')
    for vx, vy in vertices:
        d.line([(cx, cy), (vx, vy)], fill=alpha, width=round(big * 0.018))
    d.ellipse([cx - hub_r, cy - hub_r, cx + hub_r, cy + hub_r], fill=alpha)
    for vx, vy in vertices:
        d.ellipse([vx - node_r, vy - node_r, vx + node_r, vy + node_r], fill=alpha)

    return mask.resize((size, size), Image.LANCZOS)


def render_mark(size: int, scale: float = 0.40) -> Image.Image:
    """Gradient mark on a transparent background."""
    out = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    out.paste(gradient(size), (0, 0), mark_mask(size, scale=scale))
    return out


def render_badge(size: int, bg: tuple[int, int, int] = INK, scale: float = 0.32) -> Image.Image:
    """Solid-background variant used for legacy/maskable launcher icons."""
    out = Image.new('RGBA', (size, size), (*bg, 255))
    out.paste(gradient(size), (0, 0), mark_mask(size, scale=scale))
    return out


def render_splash(size: int, background: tuple[int, int, int], wordmark: bool = True) -> Image.Image:
    """Splash screen: mark plus an optional CORTEX wordmark, centred."""
    out = Image.new('RGBA', (size, size), (*background, 255))
    mark_size = round(size * 0.34)
    mark = render_mark(mark_size)
    out.paste(mark, ((size - mark_size) // 2, round(size * (0.30 if wordmark else 0.33))), mark)

    if wordmark:
        d = ImageDraw.Draw(out)
        text = 'CORTEX'
        font = _wordmark_font(round(size * 0.072))
        box = d.textbbox((0, 0), text, font=font)
        d.text(
            ((size - (box[2] - box[0])) / 2 - box[0], size * 0.66),
            text,
            font=font,
            fill=(*PAPER, 235),
        )
    return out


def _wordmark_font(px: int):
    for path in (
        '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, px)
    return ImageFont.load_default()


def write(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    print(f'wrote {path.relative_to(ROOT)} ({image.width}x{image.height})')


def main() -> None:
    web_assets = {
        'logo.png': render_mark(500, scale=0.42),
        'favicon.png': render_mark(512, scale=0.44),
        'favicon-96x96.png': render_mark(96, scale=0.44),
        'apple-touch-icon.png': render_badge(180),
        'web-app-manifest-192x192.png': render_badge(192),
        'web-app-manifest-512x512.png': render_badge(512),
        'splash.png': render_splash(500, PAPER),
        'splash-dark.png': render_splash(500, INK),
    }

    for directory in WEB_STATIC_DIRS:
        for name, image in web_assets.items():
            write(image, directory / name)
        # Multi-resolution .ico keeps the browser tab crisp at 16/32/48px.
        render_mark(256, scale=0.46).save(
            directory / 'favicon.ico', sizes=[(16, 16), (32, 32), (48, 48), (64, 64), (256, 256)]
        )
        print(f'wrote {(directory / "favicon.ico").relative_to(ROOT)}')

    # android/ is produced by `npx cap add android`; keep the source assets at
    # the repository root so @capacitor/assets can regenerate the native icons.
    android_assets = {
        'icon-only.png': render_badge(1024),
        'icon-foreground.png': render_mark(1024, scale=0.30),
        'icon-background.png': Image.new('RGBA', (1024, 1024), (*INK, 255)),
        'splash.png': render_splash(2732, PAPER),
        'splash-dark.png': render_splash(2732, INK),
    }
    for name, image in android_assets.items():
        write(image, ANDROID_ASSETS_DIR / name)


if __name__ == '__main__':
    main()
