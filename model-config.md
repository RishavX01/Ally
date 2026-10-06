
---

## File 3: `scripts/preprocess.py`

```python
#!/usr/bin/env python3
"""Ally image preprocessing: resize, tile, and prepare inputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

MAX_EDGE = 2048
TILE_OVERLAP = 64
TILE_MIN_ASPECT = 3.0


def load_image(path: Path) -> Image.Image:
    img = Image.open(path)
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    return img


def downscale(img: Image.Image, max_edge: int = MAX_EDGE) -> Image.Image:
    w, h = img.size
    longest = max(w, h)
    if longest <= max_edge:
        return img
    scale = max_edge / longest
    return img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)


def should_tile(img: Image.Image) -> bool:
    w, h = img.size
    aspect = max(w, h) / max(1, min(w, h))
    return aspect >= TILE_MIN_ASPECT


def tile(img: Image.Image) -> list[Image.Image]:
    w, h = img.size
    if w >= h:
        tile_size = h
        step = tile_size - TILE_OVERLAP
        return [
            img.crop((x, 0, min(x + tile_size, w), h))
            for x in range(0, w, step)
        ]
    else:
        tile_size = w
        step = tile_size - TILE_OVERLAP
        return [
            img.crop((0, y, w, min(y + tile_size, h)))
            for y in range(0, h, step)
        ]


def process(input_path: Path, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    original = Image.open(input_path)
    img = downscale(load_image(input_path))

    if should_tile(img):
        tiles = tile(img)
    else:
        tiles = [img]

    paths = []
    for i, t in enumerate(tiles):
        p = out_dir / f"tile_{i}.png"
        t.save(p, "PNG")
        paths.append(str(p))

    meta = {
        "source": str(input_path),
        "original_size": list(original.size),
        "processed_size": list(img.size),
        "tile_count": len(paths),
        "tiles": paths,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def main() -> int:
    ap = argparse.ArgumentParser(description="Ally preprocessor")
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    if not args.input.exists():
        print(f"error: input not found: {args.input}", file=sys.stderr)
        return 1

    meta = process(args.input, args.out)
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())