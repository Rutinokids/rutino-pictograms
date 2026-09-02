#!/usr/bin/env python3
"""Convert PNG(s) to SVG using autotrace for native multi-color vector tracing.

Requires: autotrace CLI (brew install autotrace)

Usage:
    python scripts/png2svg-v2.py                          # batch: PNG/*.png -> SVG/*.svg
    python scripts/png2svg-v2.py PNG/apple.png            # single file -> SVG/apple.svg
    python scripts/png2svg-v2.py PNG/apple.png -o out.svg # single file -> custom output
"""
import argparse
import os
import re
import subprocess
import sys
import tempfile
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from sklearn.cluster import MiniBatchKMeans

PROJECT_DIR = Path(__file__).parent.parent
PNG_DIR = PROJECT_DIR / "PNG"
SVG_DIR = PROJECT_DIR / "SVG"

DEFAULT_COLORS = 16
DEFAULT_DESPECKLE = 12
DEFAULT_PRE_COLORS = 14
DEFAULT_OUTLINE_THRESHOLD = 40  # RGB color-distance radius around outline_color (conservative after mean-shift)
DEFAULT_OUTLINE_COLOR = (6, 38, 81)  # #062651

# Chroma-key color used as background for transparent areas; removed from SVG after tracing.
# Magenta is never a natural pictogram color, so it's safe to use as a marker.
_CHROMA = (255, 0, 255)


def _parse_hex_color(hex_str: str) -> tuple[int, int, int]:
    hex_str = hex_str.lstrip("#")
    r, g, b = int(hex_str[0:2], 16), int(hex_str[2:4], 16), int(hex_str[4:6], 16)
    return (r, g, b)


def _near_outline_mask(arr: np.ndarray, outline_color: tuple[int, int, int], threshold: int) -> np.ndarray:
    """True where a pixel's RGB distance to outline_color is within threshold."""
    diff = arr - np.array(outline_color, dtype=np.float32)
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    return dist < threshold


def _prepare_png(
    png_path: str,
    target_size: int,
    pre_colors: int,
    outline_threshold: int,
    outline_color: tuple[int, int, int],
) -> str:
    img = Image.open(png_path).convert("RGBA")
    alpha = np.array(img.split()[3])
    transparent = alpha < 128  # pixels that were originally transparent

    # Use white for compositing so mean-shift works well on pictogram content
    bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
    bg.paste(img, mask=img.split()[3])
    arr = np.array(bg.convert("RGB"), dtype=np.float32)

    # Mean-shift filter: absorbs anti-aliased border pixels into their dominant neighbor region
    arr_u8 = arr.astype(np.uint8)
    arr_bgr = cv2.cvtColor(arr_u8, cv2.COLOR_RGB2BGR)
    filtered_bgr = cv2.pyrMeanShiftFiltering(arr_bgr, sp=10, sr=50)
    # Median blur eliminates isolated pixel-level islands left by mean-shift (causes white specks)
    filtered_bgr = cv2.medianBlur(filtered_bgr, 5)
    arr = cv2.cvtColor(filtered_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)

    # Conservative outline snap - only pixels very close to #062651 (not dark fills)
    near = _near_outline_mask(arr, outline_color, outline_threshold)
    arr[near] = outline_color

    # Stamp chroma-key on transparent pixels so we can remove them from the SVG later
    arr[transparent] = _CHROMA

    img = Image.fromarray(arr.astype(np.uint8), "RGB")

    if target_size:
        w, h = img.size
        if max(w, h) > target_size:
            scale = target_size / max(w, h)
            img = img.resize((round(w * scale), round(h * scale)), Image.Resampling.NEAREST)

    if pre_colors:
        # K-means: each pixel gets assigned to the nearest cluster centroid
        pixels = arr.reshape(-1, 3)
        km = MiniBatchKMeans(n_clusters=pre_colors, n_init=3, random_state=0, batch_size=2048)
        labels = km.fit_predict(pixels)
        centroids = km.cluster_centers_.astype(np.uint8)
        quantized = centroids[labels].reshape(arr.shape).astype(np.float32)
        # Re-apply snaps after k-means
        quantized[near] = outline_color
        quantized[transparent] = _CHROMA  # ensure chroma survives quantization
        img = Image.fromarray(quantized.astype(np.uint8), "RGB")

    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    img.save(tmp.name)
    return tmp.name


def _fill_svg_gaps(svg_path: str) -> None:
    """Add thin stroke matching each path's fill to close sub-pixel gaps, then remove chroma background."""
    with open(svg_path) as f:
        content = f.read()
    content = re.sub(
        r'fill:(#[0-9a-fA-F]{6}); stroke:none;',
        r'fill:\1; stroke:\1; stroke-width:1;',
        content,
    )
    # Remove chroma-key paths (magenta = transparent background marker)
    content = re.sub(
        r'<path style="fill:#ff00ff;[^"]*"[^/]*/>\n?',
        '',
        content,
        flags=re.IGNORECASE,
    )
    with open(svg_path, "w") as f:
        f.write(content)


def _run_autotrace(input_path: str, output_path: str, colors: int, despeckle: int) -> None:
    cmd = [
        "autotrace",
        "--color-count", str(colors),
        "--despeckle-level", str(despeckle),
        "--output-format", "svg",
        "--output-file", output_path,
        input_path,
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=60)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.decode().strip() or f"autotrace exited with code {result.returncode}")


def convert_one(args: tuple) -> tuple[str, bool, str]:
    png_path, svg_path, colors, despeckle, target_size, pre_colors, outline_threshold, outline_color = args
    name = os.path.basename(png_path)
    tmp_path = None
    try:
        tmp_path = _prepare_png(png_path, target_size, pre_colors, outline_threshold, outline_color)
        _run_autotrace(tmp_path, svg_path, colors if not pre_colors else 0, despeckle)
        _fill_svg_gaps(svg_path)
        return (name, True, "")
    except Exception as e:
        return (name, False, str(e))
    finally:
        if tmp_path:
            os.unlink(tmp_path)


def main():
    parser = argparse.ArgumentParser(description="Convert PNG files to SVG using autotrace.")
    parser.add_argument("input", nargs="?", default=None, help="Input PNG file (omit for batch mode)")
    parser.add_argument("-o", "--output", default=None, help="Output SVG path (single-file mode only)")
    parser.add_argument("-j", "--jobs", type=int, default=8, help="Parallel workers for batch mode (default: 8)")
    parser.add_argument("--input-dir", default=None, help="Directory with PNGs for batch mode (default: PNG/ next to project root)")
    parser.add_argument("--output-dir", default=None, help="Directory for SVG output in batch mode (default: SVG/ next to project root)")
    parser.add_argument(
        "-c", "--colors", type=int, default=DEFAULT_COLORS,
        help=f"Number of colors to reduce to (default: {DEFAULT_COLORS}, 0 = no reduction)",
    )
    parser.add_argument(
        "-d", "--despeckle", type=int, default=DEFAULT_DESPECKLE,
        help=f"Despeckle level 0-20: removes small noise regions (default: {DEFAULT_DESPECKLE})",
    )
    parser.add_argument(
        "-s", "--size", type=int, default=0,
        help="Resize PNG to this max dimension before tracing (0 = keep original)",
    )
    parser.add_argument(
        "-p", "--pre-colors", type=int, default=DEFAULT_PRE_COLORS,
        help=f"Pre-quantize to N colors before tracing to unify similar colors (default: {DEFAULT_PRE_COLORS}, 0 = skip)",
    )
    parser.add_argument(
        "-t", "--outline-threshold", type=int, default=DEFAULT_OUTLINE_THRESHOLD,
        help=f"RGB distance radius: pixels within this distance of outline-color are snapped to it (default: {DEFAULT_OUTLINE_THRESHOLD})",
    )
    default_hex = "#{:02x}{:02x}{:02x}".format(*DEFAULT_OUTLINE_COLOR)
    parser.add_argument(
        "--outline-color", type=str, default=default_hex,
        help=f"Hex color for outlines, e.g. #062651 (default: {default_hex})",
    )
    args = parser.parse_args()

    outline_color = _parse_hex_color(args.outline_color)
    task_args = (args.colors, args.despeckle, args.size, args.pre_colors, args.outline_threshold, outline_color)

    if args.input:
        png_path = Path(args.input)
        if not png_path.is_file():
            print(f"Error: file not found: {png_path}", file=sys.stderr)
            sys.exit(1)
        out_path = Path(args.output) if args.output else SVG_DIR / (png_path.stem + ".svg")
        SVG_DIR.mkdir(exist_ok=True)

        name, ok, err = convert_one((str(png_path), str(out_path), *task_args))
        if ok:
            print(f"OK: {name} -> {out_path}")
        else:
            print(f"FAIL: {name}: {err}", file=sys.stderr)
            sys.exit(1)
        return

    png_dir = Path(args.input_dir) if args.input_dir else PNG_DIR
    svg_dir = Path(args.output_dir) if args.output_dir else SVG_DIR

    svg_dir.mkdir(parents=True, exist_ok=True)
    pngs = sorted(str(p) for p in png_dir.glob("*.png"))
    if not pngs:
        print(f"No PNG files found in {png_dir}")
        sys.exit(1)

    tasks = [(p, str(svg_dir / (Path(p).stem + ".svg")), *task_args) for p in pngs]
    print(
        f"Converting {len(tasks)} PNGs to SVGs "
        f"(pre_colors={args.pre_colors}, outline={args.outline_color} threshold={args.outline_threshold}, "
        f"despeckle={args.despeckle}, jobs={args.jobs})..."
    )

    ok_count = fail_count = 0
    with Pool(args.jobs) as pool:
        for i, (name, success, err) in enumerate(pool.imap_unordered(convert_one, tasks), 1):
            if success:
                ok_count += 1
            else:
                fail_count += 1
                print(f"  FAIL: {name}: {err}")
            if i % 500 == 0:
                print(f"  Progress: {i}/{len(tasks)} ({ok_count} ok, {fail_count} fail)")

    print(f"\nDone! {ok_count} converted, {fail_count} failed")
    print(f"SVGs in: {svg_dir}")


if __name__ == "__main__":
    main()
