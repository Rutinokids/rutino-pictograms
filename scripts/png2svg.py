#!/usr/bin/env python3
"""Convert PNG(s) to SVG using potrace for bitmap-to-vector tracing.

Supports color icons by quantizing colors and tracing each layer separately.
Requires: potrace CLI (brew install potrace), Pillow, numpy.

Usage:
    python scripts/png2svg.py                          # batch: PNG/*.png -> SVG/*.svg
    python scripts/png2svg.py PNG/apple.png            # single file -> SVG/apple.svg
    python scripts/png2svg.py PNG/apple.png -o out.svg # single file -> custom output
"""
import argparse
import os
import subprocess
import sys
import tempfile
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_DIR = Path(__file__).parent.parent
PNG_DIR = PROJECT_DIR / "PNG"
SVG_DIR = PROJECT_DIR / "SVG"


def _quantize_image(img: Image.Image, colors: int = 16, merge_dist: float = 20) -> Image.Image:
    """Quantize to N colors, preserving alpha. Merges similar colors via nearest-color."""
    arr = np.array(img.convert("RGBA"))

    # Flatten semi-transparent pixels onto white background
    semi = (arr[:,:,3] > 0) & (arr[:,:,3] < 255)
    if semi.any():
        alpha = arr[semi, 3:4].astype(np.float32) / 255.0
        arr[semi, :3] = (arr[semi, :3].astype(np.float32) * alpha + 255 * (1 - alpha)).astype(np.uint8)
        arr[semi, 3] = 255

    # Mark near-white pixels before quantization
    r, g, b, a = arr[:,:,0], arr[:,:,1], arr[:,:,2], arr[:,:,3]
    near_white = (r > 225) & (g > 225) & (b > 225) & (a > 0)
    white_mask = near_white.copy()

    # Quantize
    rgb = Image.fromarray(arr[:, :, :3], 'RGB')
    quantized = rgb.quantize(colors=colors, method=Image.Quantize.MEDIANCUT).convert("RGB")
    result = np.dstack([np.array(quantized), arr[:, :, 3]])

    # Force white on near-white pixels
    result[white_mask] = [255, 255, 255, 255]

    # Merge similar colors
    opaque_mask = result[:,:,3] > 0
    unique_q = np.unique(result[opaque_mask][:,:3], axis=0)

    merged = []
    used = set()
    for i, c1 in enumerate(unique_q):
        if i in used: continue
        group = [c1]
        for j, c2 in enumerate(unique_q):
            if j <= i or j in used: continue
            dist = np.sqrt(np.sum((c1.astype(int) - c2.astype(int))**2))
            if dist < merge_dist:
                group.append(c2)
                used.add(j)
        merged.append(np.mean(group, axis=0).astype(np.uint8))
        used.add(i)

    merged = np.array(merged)

    # Remap using nearest color
    out = result.copy()
    pixels = out[opaque_mask][:,:3].astype(np.float32)
    diffs = pixels[:, np.newaxis, :] - merged[np.newaxis, :, :].astype(np.float32)
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    nearest = np.argmin(dists, axis=1)
    out[opaque_mask, 0] = merged[nearest, 0]
    out[opaque_mask, 1] = merged[nearest, 1]
    out[opaque_mask, 2] = merged[nearest, 2]

    return Image.fromarray(out)


def _extract_layers(img: Image.Image, min_area: int = 20) -> list[tuple[tuple[int, int, int, int], np.ndarray]]:
    """Extract color regions as binary masks using numpy. Filters out tiny regions."""
    arr = np.array(img.convert("RGBA"))
    h, w = arr.shape[:2]

    # Build color -> mask mapping (skip fully transparent)
    opaque = arr[:, :, 3] > 0
    rgb = arr[:, :, :3]

    # Find unique colors among opaque pixels
    opaque_pixels = rgb[opaque]
    if len(opaque_pixels) == 0:
        return []

    # Use view to make rows hashable for unique
    opaque_flat = opaque_pixels.view(np.dtype((np.void, opaque_pixels.dtype.itemsize * 3)))
    unique_colors = np.unique(opaque_flat).view(opaque_pixels.dtype).reshape(-1, 3)

    layers = []
    for color in unique_colors:
        r, g, b = int(color[0]), int(color[1]), int(color[2])
        # Create mask: True where this color matches (with small tolerance for anti-aliasing)
        match = opaque & (rgb[:, :, 0] == r) & (rgb[:, :, 1] == g) & (rgb[:, :, 2] == b)
        area = int(match.sum())
        if area < min_area:
            continue
        # Binary mask: 0 = foreground (black), 255 = background (white) for potrace
        mask = np.where(match, 0, 255).astype(np.uint8)
        # Get alpha for this color from original
        alpha_vals = arr[:, :, 3][match]
        avg_alpha = int(alpha_vals.mean()) if len(alpha_vals) > 0 else 255
        layers.append(((r, g, b, avg_alpha), mask, area))

    # Sort by area descending (background first)
    layers.sort(key=lambda x: -x[2])
    return layers


def _trace_mask_to_svg_group(mask: np.ndarray) -> str:
    """Trace a binary mask with potrace, return the full <g> group with transform."""
    with tempfile.TemporaryDirectory() as tmpdir:
        mask_path = os.path.join(tmpdir, "mask.pgm")
        svg_path = os.path.join(tmpdir, "out.svg")

        img = Image.fromarray(mask, mode="L")
        img.save(mask_path)

        try:
            subprocess.run(
                ["potrace", "-s", "-o", svg_path, mask_path],
                check=True,
                capture_output=True,
                timeout=30,
            )
            with open(svg_path) as f:
                content = f.read()
            # Extract the <g transform="..."> ... </g> block
            start = content.find("<g ")
            if start == -1:
                return ""
            end = content.find("</g>", start)
            if end == -1:
                return ""
            return content[start : end + 4]  # include </g>
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return ""


def _build_svg(width: int, height: int, layers: list) -> str:
    """Build SVG string from traced layers."""
    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}">',
    ]
    for (r, g, b, a), _mask, _area, g_group in layers:
        if not g_group:
            continue
        hex_color = f"#{r:02x}{g:02x}{b:02x}"
        opacity = f' opacity="{a / 255:.2f}"' if a < 255 else ""
        colored = g_group.replace('fill="#000000"', f'fill="{hex_color}"{opacity}', 1)
        parts.append(f"  {colored}")
    parts.append("</svg>")
    return "\n".join(parts)


def _png_to_svg(png_path: str, num_colors: int = 16, min_area: int = 20, target_size: int = 0) -> str | None:
    """Convert a single PNG file to SVG string."""
    img = Image.open(png_path).convert("RGBA")

    # Downscale before tracing if target_size is set
    if target_size:
        w, h = img.size
        scale = target_size / max(w, h)
        new_w, new_h = round(w * scale), round(h * scale)
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    w, h = img.size
    quantized = _quantize_image(img, colors=num_colors)
    layers_raw = _extract_layers(quantized, min_area=min_area)

    if not layers_raw:
        return None

    layers = []
    for color, mask, area in layers_raw:
        g_group = _trace_mask_to_svg_group(mask)
        layers.append((color, mask, area, g_group))

    return _build_svg(w, h, layers)


def convert_one(args: tuple) -> tuple[str, bool, str]:
    png_path, out_path, num_colors, min_area, target_size = args
    name = os.path.basename(png_path)
    try:
        svg_content = _png_to_svg(png_path, num_colors, min_area, target_size)
        if svg_content is None:
            return (name, False, "empty result")
        with open(out_path, "w") as f:
            f.write(svg_content)
        return (name, True, "")
    except Exception as e:
        return (name, False, str(e))


def main():
    parser = argparse.ArgumentParser(description="Convert PNG files to SVG using potrace.")
    parser.add_argument("input", nargs="?", default=None, help="Input PNG file (omit for batch mode)")
    parser.add_argument("-o", "--output", default=None, help="Output SVG path (single-file mode only)")
    parser.add_argument("-j", "--jobs", type=int, default=8, help="Parallel workers for batch mode (default: 8)")
    parser.add_argument("-c", "--colors", type=int, default=16,
                        help="Number of colors to quantize to (default: 16)")
    parser.add_argument("--min-area", type=int, default=20,
                        help="Min pixel area for a color region to be traced (default: 20)")
    parser.add_argument("-s", "--size", type=int, default=0,
                        help="Output SVG size in px (scales to fit, 0 = keep original)")
    args = parser.parse_args()

    # Single-file mode
    if args.input:
        png_path = Path(args.input)
        if not png_path.is_file():
            print(f"Error: file not found: {png_path}", file=sys.stderr)
            sys.exit(1)

        if args.output:
            out_path = Path(args.output)
        else:
            SVG_DIR.mkdir(exist_ok=True)
            out_path = SVG_DIR / (png_path.stem + ".svg")

        name, ok, err = convert_one((str(png_path), str(out_path), args.colors, args.min_area, args.size))
        if ok:
            print(f"OK: {name} -> {out_path}")
        else:
            print(f"FAIL: {name}: {err}", file=sys.stderr)
            sys.exit(1)
        return

    # Batch mode
    SVG_DIR.mkdir(exist_ok=True)
    pngs = sorted(str(p) for p in PNG_DIR.glob("*.png"))
    if not pngs:
        print("No PNG files found!")
        sys.exit(1)

    tasks = [(p, str(SVG_DIR / (Path(p).stem + ".svg")), args.colors, args.min_area, args.size) for p in pngs]

    print(f"Converting {len(tasks)} PNGs to SVGs (quantize to {args.colors} colors, min_area={args.min_area})...")
    ok = 0
    fail = 0
    with Pool(args.jobs) as pool:
        for i, (name, success, err) in enumerate(pool.imap_unordered(convert_one, tasks), 1):
            if success:
                ok += 1
            else:
                fail += 1
                print(f"  FAIL: {name}: {err}")
            if i % 500 == 0:
                print(f"  Progress: {i}/{len(tasks)} ({ok} ok, {fail} fail)")

    print(f"\nDone! {ok} converted, {fail} failed")
    print(f"SVGs in: {SVG_DIR}")


if __name__ == "__main__":
    main()
