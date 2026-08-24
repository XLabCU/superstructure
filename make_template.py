#!/usr/bin/env python3
"""
make_template.py
────────────────
Converts the archaeological Context Record PDF into a Supernote-compatible
PNG template file.

Usage
-----
    python make_template.py context-sheet.pdf

Output
------
    context_record_template.png   (place in Supernote's MyStyle/ folder)

Supernote template requirements
--------------------------------
  • Format : PNG (recommended), JPG, JPEG or WEBP
  • Size   : 1404 × 1872 px  for A5 X / A6 X / A6 X2 Nomad
             1920 × 2560 px  for A5 X2 Manta
  • Filename: no special characters  (  / : * ? " < > |)

On-device setup
---------------
1. Connect Supernote via USB.
2. Copy context_record_template.png  →  Supernote/MyStyle/
3. Open or create a notebook on the device.
4. Tap the template icon → "My Style" → select context_record_template.
5. Write!
"""

import sys
import os
from pathlib import Path
from pdf2image import convert_from_path
from PIL import Image

# ── Device profiles ──────────────────────────────────────────────────────────
PROFILES = {
    "nomad":  (1404, 1872),   # A5 X, A6 X, A6 X2 Nomad  (default)
    "manta":  (1920, 2560),   # A5 X2 Manta
}

def make_template(
    pdf_path: str,
    device: str = "nomad",
    page_index: int = 0,
    output_path: str | None = None,
    dpi: int = 300,
) -> Path:
    """
    Render one page of a PDF at high DPI, then resize/pad to exact Supernote
    canvas dimensions, preserving the form's aspect ratio.

    Parameters
    ----------
    pdf_path   : path to the source PDF
    device     : "nomad" or "manta"
    page_index : 0-based page number to convert (default: first page)
    output_path: override the output filename
    dpi        : render DPI before resizing (300 recommended for crisp lines)
    """
    if device not in PROFILES:
        raise ValueError(f"Unknown device '{device}'. Choose from: {list(PROFILES)}")

    target_w, target_h = PROFILES[device]
    pdf_path = Path(pdf_path).resolve()

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    print(f"[1/4] Rendering page {page_index + 1} of '{pdf_path.name}' at {dpi} DPI …")
    pages = convert_from_path(str(pdf_path), dpi=dpi, first_page=page_index + 1,
                               last_page=page_index + 1)
    if not pages:
        raise RuntimeError("pdf2image returned no pages. Check poppler is installed.")

    img: Image.Image = pages[0].convert("RGB")
    print(f"    Rendered size: {img.width} × {img.height} px")

    # ── Resize to fit inside the target canvas, maintaining aspect ratio ──────
    print(f"[2/4] Resizing to fit {target_w} × {target_h} (device: {device}) …")
    img.thumbnail((target_w, target_h), Image.LANCZOS)

    # ── Pad to exact canvas size with white background ────────────────────────
    print("[3/4] Padding to exact canvas dimensions …")
    canvas = Image.new("RGB", (target_w, target_h), color=(255, 255, 255))
    x_off = (target_w - img.width) // 2
    y_off = (target_h - img.height) // 2
    canvas.paste(img, (x_off, y_off))

    # ── Save ──────────────────────────────────────────────────────────────────
    if output_path is None:
        stem = pdf_path.stem.replace(" ", "_")
        output_path = pdf_path.parent / f"{stem}_template_{device}.png"
    output_path = Path(output_path)
    canvas.save(str(output_path), format="PNG", optimize=True)
    print(f"[4/4] Template saved → {output_path}")
    print()
    print("Next steps:")
    print(f"  1. Connect your Supernote via USB.")
    print(f"  2. Copy  '{output_path.name}'  →  Supernote/MyStyle/")
    print(f"  3. Open a notebook → tap the template icon → My Style → select this template.")
    print(f"  4. Fill in the form with your stylus, then sync or copy the .note file to Mac.")
    print(f"  5. Run  parse_context_record.py  on the .note file to extract structured data.")
    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert archaeological context-sheet PDF to Supernote template PNG."
    )
    parser.add_argument("pdf", help="Path to the source PDF file")
    parser.add_argument(
        "--device", choices=list(PROFILES), default="nomad",
        help="Target Supernote device (default: nomad = A5 X / A6 X / A6 X2 Nomad)"
    )
    parser.add_argument(
        "--page", type=int, default=1,
        help="1-based page number to extract (default: 1)"
    )
    parser.add_argument("--output", default=None, help="Override output file path")
    parser.add_argument("--dpi", type=int, default=300, help="Render DPI (default: 300)")
    args = parser.parse_args()

    make_template(
        pdf_path=args.pdf,
        device=args.device,
        page_index=args.page - 1,
        output_path=args.output,
        dpi=args.dpi,
    )
