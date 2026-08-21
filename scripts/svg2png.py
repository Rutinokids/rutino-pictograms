#!/usr/bin/env python3
"""Convert all SVGs in SVG/ to PNGs in PNG/ (200x200, transparent background)."""
import os
import sys
import cairosvg
from pathlib import Path
from multiprocessing import Pool

SVG_DIR = Path(__file__).parent.parent / "SVG"
PNG_DIR = Path(__file__).parent.parent / "PNG"
SIZE = 200  # pixels


def convert_one(svg_file: str) -> tuple[str, bool, str]:
    name = os.path.basename(svg_file)
    png_path = PNG_DIR / name.replace(".svg", ".png")
    try:
        cairosvg.svg2png(
            url=svg_file,
            write_to=str(png_path),
            output_width=SIZE,
            output_height=SIZE,
        )
        return (name, True, "")
    except Exception as e:
        return (name, False, str(e))


def main():
    PNG_DIR.mkdir(exist_ok=True)

    svgs = sorted(str(p) for p in SVG_DIR.glob("*.svg"))
    if not svgs:
        print("No SVG files found!")
        sys.exit(1)

    print(f"Converting {len(svgs)} SVGs to {SIZE}x{SIZE} PNGs...")

    ok = 0
    fail = 0
    with Pool(8) as pool:
        for i, (name, success, err) in enumerate(pool.imap_unordered(convert_one, svgs), 1):
            if success:
                ok += 1
            else:
                fail += 1
                print(f"  FAIL: {name}: {err}")
            if i % 500 == 0:
                print(f"  Progress: {i}/{len(svgs)} ({ok} ok, {fail} fail)")

    print(f"\nDone! {ok} converted, {fail} failed")
    print(f"PNGs in: {PNG_DIR}")


if __name__ == "__main__":
    main()
