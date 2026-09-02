#!/usr/bin/env python3
"""Resize PNG(s) to 200x200 for the firmware display.

Maintains aspect ratio and centers the image on a transparent background.

Usage:
    python scripts/resizer.py                    # batch: resize all non-200x200 PNGs in PNG/
    python scripts/resizer.py PNG/apple.png      # single file
    python scripts/resizer.py PNG/apple.png -o out.png  # single file -> custom output
"""
import argparse
import os
import sys
from multiprocessing import Pool
from pathlib import Path

from PIL import Image

PROJECT_DIR = Path(__file__).parent.parent
PNG_DIR = PROJECT_DIR / "PNG"
TARGET_SIZE = 200


def resize_one(args: tuple) -> tuple[str, bool, str]:
    png_path, out_path, target = args
    name = os.path.basename(png_path)
    try:
        img = Image.open(png_path).convert("RGBA")
        w, h = img.size

        if w == target and h == target:
            return (name, True, "already correct")

        # Scale to fit within target_size, maintaining aspect ratio
        scale = target / max(w, h)
        new_w, new_h = round(w * scale), round(h * scale)
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

        # Center on transparent canvas
        canvas = Image.new("RGBA", (target, target), (0, 0, 0, 0))
        offset_x = (target - new_w) // 2
        offset_y = (target - new_h) // 2
        canvas.paste(img, (offset_x, offset_y))

        canvas.save(out_path)
        return (name, True, f"{w}x{h} -> {target}x{target}")
    except Exception as e:
        return (name, False, str(e))


def main():
    parser = argparse.ArgumentParser(description="Resize PNGs to 200x200.")
    parser.add_argument("input", nargs="?", default=None, help="Input PNG file (omit for batch mode)")
    parser.add_argument("-o", "--output", default=None, help="Output PNG path (single-file mode only)")
    parser.add_argument("-j", "--jobs", type=int, default=8, help="Parallel workers for batch mode (default: 8)")
    parser.add_argument("-s", "--size", type=int, default=TARGET_SIZE, help=f"Target size in px (default: {TARGET_SIZE})")
    args = parser.parse_args()

    if args.input:
        png_path = Path(args.input)
        if not png_path.is_file():
            print(f"Error: file not found: {png_path}", file=sys.stderr)
            sys.exit(1)

        out_path = Path(args.output) if args.output else png_path
        name, ok, msg = resize_one((str(png_path), str(out_path), args.size))
        if ok:
            print(f"OK: {name} ({msg})")
        else:
            print(f"FAIL: {name}: {msg}", file=sys.stderr)
            sys.exit(1)
        return

    # Batch mode: find all PNGs that are not target_size x target_size
    pngs = sorted(str(p) for p in PNG_DIR.glob("*.png"))
    if not pngs:
        print("No PNG files found!")
        sys.exit(1)

    to_resize = []
    already_ok = 0
    for p in pngs:
        try:
            img = Image.open(p)
            w, h = img.size
            if w != args.size or h != args.size:
                to_resize.append(p)
            else:
                already_ok += 1
        except Exception:
            pass

    if not to_resize:
        print(f"All {len(pngs)} images are already {args.size}x{args.size}. Nothing to do.")
        return

    tasks = [(p, p, args.size) for p in to_resize]
    print(f"Resizing {len(to_resize)} images to {args.size}x{args.size} ({already_ok} already correct)...")

    ok = 0
    fail = 0
    with Pool(args.jobs) as pool:
        for i, (name, success, msg) in enumerate(pool.imap_unordered(resize_one, tasks), 1):
            if success:
                ok += 1
                if msg != "already correct":
                    print(f"  {name}: {msg}")
            else:
                fail += 1
                print(f"  FAIL: {name}: {msg}")

    print(f"\nDone! {ok} resized, {fail} failed")


if __name__ == "__main__":
    main()
