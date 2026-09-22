"""Generate the synthetic placeholder image used by examples/manifest.json.

The public repository ships no photographic assets, so the smoke example uses a
generated product-like still life instead. Re-run this script to recreate it:

    python examples/make_placeholder.py
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter


def build(size: int = 768) -> Image.Image:
    image = Image.new("RGB", (size, size), (232, 231, 228))
    draw = ImageDraw.Draw(image)

    # Soft vertical studio gradient for the backdrop.
    for y in range(size):
        shade = 236 - int(46 * y / size)
        draw.line([(0, y), (size, y)], fill=(shade, shade - 2, shade - 6))

    # Table plane.
    draw.rectangle([0, int(size * 0.72), size, size], fill=(198, 192, 186))

    # Contact shadow, blurred separately so the edges stay soft.
    shadow = Image.new("L", (size, size), 0)
    ImageDraw.Draw(shadow).ellipse(
        [int(size * 0.24), int(size * 0.68), int(size * 0.78), int(size * 0.79)], fill=110
    )
    shadow = shadow.filter(ImageFilter.GaussianBlur(size / 55))
    image.paste((150, 145, 140), (0, 0), shadow)

    # A simple bottle-like product silhouette.
    body = [int(size * 0.36), int(size * 0.30), int(size * 0.64), int(size * 0.74)]
    draw.rounded_rectangle(body, radius=int(size * 0.045), fill=(64, 96, 128))
    draw.rectangle(
        [int(size * 0.46), int(size * 0.20), int(size * 0.54), int(size * 0.31)], fill=(52, 78, 104)
    )
    draw.rounded_rectangle(
        [int(size * 0.445), int(size * 0.17), int(size * 0.555), int(size * 0.21)],
        radius=int(size * 0.012), fill=(196, 172, 96),
    )
    # Label band and a specular highlight.
    draw.rectangle(
        [int(size * 0.36), int(size * 0.46), int(size * 0.64), int(size * 0.58)], fill=(238, 236, 230)
    )
    draw.line(
        [(int(size * 0.40), int(size * 0.33)), (int(size * 0.40), int(size * 0.71))],
        fill=(120, 150, 178), width=max(2, size // 96),
    )
    return image


def main() -> None:
    target = Path(__file__).resolve().parent / "images" / "input.jpg"
    target.parent.mkdir(parents=True, exist_ok=True)
    build().save(target, quality=92)
    print(f"wrote {target}")


if __name__ == "__main__":
    main()
