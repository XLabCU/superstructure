#!/usr/bin/env python3
"""
visualise_rois.py
─────────────────
Draws every field region from FIELD_REGIONS onto the template PNG so you can
verify alignment before running OCR.

Output: a single annotated PNG — open it in Preview/any image viewer.

Usage
-----
    python visualise_rois.py context_record_template_nomad.png
    python visualise_rois.py context_record_template_nomad.png --out check.png
    
    # NEW: Run the interactive GUI to draw boxes and generate dict entries
    python visualise_rois.py context_record_template_nomad.png --gui

Dependencies
------------
    pip install pillow
    (tkinter is used for the GUI, which is included in standard Python)
"""

import argparse
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# ── Copy of FIELD_REGIONS from parse_context_record.py ───────────────────────
FIELD_REGIONS = {
    "site_code":    {"label": "Site Code",                    "roi": (0.167, 0.077, 0.256, 0.092), "type": "text"},
    "area":         {"label": "Area",                         "roi": (0.301, 0.077, 0.384, 0.092), "type": "text"},
    "trench":       {"label": "Trench",                       "roi": (0.449, 0.077, 0.608, 0.092), "type": "text"},
    "context":      {"label": "Context",                      "roi": (0.684, 0.077, 0.949, 0.092), "type": "text"},
    "date":         {"label": "Date",                         "roi": (0.117, 0.105, 0.380, 0.129), "type": "text"},
    "recorded_by":  {"label": "Recorded by",                  "roi": (0.504, 0.105, 0.948, 0.129), "type": "text"},
    "feature_type": {"label": "Feature Type",                 "roi": (0.281, 0.145, 0.744, 0.163), "type": "checkbox_group",
                     "options": ["Deposit", "Cut", "Fill", "Structural"]},
    "description":  {"label": "Description",                  "roi": (0.070, 0.172, 0.952, 0.496), "type": "text"},
    "above":        {"label": "Above",                        "roi": (0.105, 0.501, 0.910, 0.525), "type": "text"},
    "below":        {"label": "Below",                        "roi": (0.105, 0.568, 0.910, 0.593), "type": "text"},
    "comments":     {"label": "Comments",                     "roi": (0.058, 0.604, 0.951, 0.793), "type": "text"},
    "finds":        {"label": "Finds",                        "roi": (0.173, 0.807, 0.571, 0.827), "type": "checkbox_group",
                     "options": ["Pot", "Lithic", "Bone", "Metal", "Other"]},
    "small_finds":  {"label": "Small Finds",                  "roi": (0.171, 0.835, 0.934, 0.882), "type": "text"},
    "samples":      {"label": "Samples",                      "roi": (0.152, 0.888, 0.947, 0.918), "type": "text"},
    "plan":         {"label": "Plan",                         "roi": (0.115, 0.922, 0.239, 0.953), "type": "text"},
    "section":      {"label": "Section",                      "roi": (0.312, 0.925, 0.501, 0.952), "type": "text"},
    "photo":        {"label": "Photo",                        "roi": (0.559, 0.920, 0.946, 0.960), "type": "text"},
}

CANVAS = {
    "nomad": (1404, 1872),
    "manta": (1920, 2560),
}

COLOURS = {
    "text":           {"box": (220,  60,  60, 120), "label": (180,  30,  30, 255)},
    "checkbox_group": {"box": ( 30, 120, 220, 120), "label": ( 20,  80, 180, 255)},
}

# ─────────────────────────────────────────────────────────────────────────────

def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibri.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except (IOError, OSError):
            continue
    return ImageFont.load_default()


def visualise(
    template_path: Path,
    device: str = "nomad",
    out_path: Path | None = None,
    opacity: int = 120,
) -> Path:
    if device not in CANVAS:
        raise ValueError(f"Unknown device '{device}'. Choose from: {list(CANVAS)}")

    img = Image.open(str(template_path)).convert("RGBA")
    target_w, target_h = CANVAS[device]
    if img.size != (target_w, target_h):
        print(f"  Resizing {img.size} → {(target_w, target_h)} …")
        img = img.resize((target_w, target_h), Image.LANCZOS)

    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    font_label = load_font(max(18, target_h // 110))
    font_coord = load_font(max(14, target_h // 145))

    for field_key, meta in FIELD_REGIONS.items():
        x0f, y0f, x1f, y1f = meta["roi"]
        ftype = meta["type"]
        label = meta["label"]
        colour = COLOURS.get(ftype, COLOURS["text"])

        x0, y0 = int(x0f * target_w), int(y0f * target_h)
        x1, y1 = int(x1f * target_w), int(y1f * target_h)

        fill_colour = colour["box"][:3] + (opacity,)
        draw.rectangle([x0, y0, x1, y1], fill=fill_colour)

        border_colour = colour["label"]
        draw.rectangle([x0, y0, x1, y1], outline=border_colour, width=3)

        label_text = f"{label}  [{field_key}]"
        coord_text = f"({x0f:.3f},{y0f:.3f})→({x1f:.3f},{y1f:.3f})"
        tx, ty = x0 + 6, y0 + 4

        for dx, dy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
            draw.text((tx + dx, ty + dy), label_text, font=font_label, fill=(255, 255, 255, 230))
        draw.text((tx, ty), label_text, font=font_label, fill=border_colour)

        ty2 = ty + font_label.size + 4 if hasattr(font_label, "size") else ty + 22
        for dx, dy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
            draw.text((tx + dx, ty2 + dy), coord_text, font=font_coord, fill=(255, 255, 255, 210))
        draw.text((tx, ty2), coord_text, font=font_coord, fill=border_colour)

    result = Image.alpha_composite(img, overlay).convert("RGB")
    if out_path is None:
        out_path = template_path.parent / f"{template_path.stem}_roi_check.png"

    result.save(str(out_path), format="PNG")
    print(f"Saved → {out_path}")
    return out_path


def print_roi_table() -> None:
    print(f"\n{'Key':<16} {'Label':<30} {'x0':>6} {'y0':>6} {'x1':>6} {'y1':>6}  Type")
    print("─" * 82)
    for key, meta in FIELD_REGIONS.items():
        x0, y0, x1, y1 = meta["roi"]
        print(f"{key:<16} {meta['label']:<30} {x0:>6.3f} {y0:>6.3f} {x1:>6.3f} {y1:>6.3f}  {meta['type']}")
    print()


def gui_converter(template_path: Path, device: str = "nomad") -> None:
    """
    Interactive UI that lets you click and drag to define fields,
    automatically generating the JSON/Dict lines to be pasted.
    """
    try:
        import tkinter as tk
        from tkinter import simpledialog
        from PIL import ImageTk
    except ImportError:
        print("ERROR: tkinter is required for the GUI. It is usually included with Python.")
        return

    w, h = CANVAS[device]

    # Initialize Tkinter
    root = tk.Tk()
    root.title(f"ROI Annotator - {template_path.name}")

    # Load and scale down image for display (max height ~850px)
    img = Image.open(template_path).convert("RGB")
    if img.size != (w, h):
        img = img.resize((w, h), Image.LANCZOS)
    
    scale = min(1.0, 850.0 / h)
    disp_w, disp_h = int(w * scale), int(h * scale)
    disp_img = img.resize((disp_w, disp_h), Image.LANCZOS)
    
    photo = ImageTk.PhotoImage(disp_img)
    
    canvas = tk.Canvas(root, width=disp_w, height=disp_h, cursor="cross")
    canvas.pack(fill=tk.BOTH, expand=True)
    canvas.create_image(0, 0, anchor=tk.NW, image=photo)

    # Draw existing regions for context
    for key, meta in FIELD_REGIONS.items():
        x0f, y0f, x1f, y1f = meta["roi"]
        canvas.create_rectangle(
            x0f * disp_w, y0f * disp_h, x1f * disp_w, y1f * disp_h,
            outline="red", width=2, dash=(4, 4)
        )
        canvas.create_text(
            x0f * disp_w + 4, y0f * disp_h + 4,
            text=key, fill="red", anchor=tk.NW
        )

    # Variables for interactive drawing
    start_x = start_y = 0
    current_rect = None

    def on_press(event):
        nonlocal start_x, start_y, current_rect
        start_x, start_y = event.x, event.y
        current_rect = canvas.create_rectangle(
            start_x, start_y, start_x, start_y, 
            outline="blue", width=2
        )

    def on_drag(event):
        canvas.coords(current_rect, start_x, start_y, event.x, event.y)

    def on_release(event):
        end_x, end_y = event.x, event.y
        
        # Calculate fractions based on display size
        x0f = min(start_x, end_x) / disp_w
        y0f = min(start_y, end_y) / disp_h
        x1f = max(start_x, end_x) / disp_w
        y1f = max(start_y, end_y) / disp_h
        
        # Ignore tiny accidental clicks
        if (x1f - x0f < 0.005) or (y1f - y0f < 0.005):
            canvas.delete(current_rect)
            return

        # Prompt user
        field_key = simpledialog.askstring("Field Key", "Enter field key (e.g., 'site_code'):", parent=root)
        if not field_key:
            canvas.delete(current_rect)
            return
            
        field_label = simpledialog.askstring("Label", "Enter display label (e.g., 'Site Code'):", parent=root)
        if not field_label:
            field_label = field_key.replace("_", " ").title()

        # Output the required dictionary snippet
        print("\n  ── Paste this into FIELD_REGIONS ──────────────────────────")
        print(f'    "{field_key}": {{')
        print(f'        "label": "{field_label}",')
        print(f'        "roi": ({x0f:.3f}, {y0f:.3f}, {x1f:.3f}, {y1f:.3f}),')
        print(f'        "type": "text",')
        print(f'    }},')
        print("  ───────────────────────────────────────────────────────────\n")

        # Update visual to show it was successfully captured
        canvas.itemconfig(current_rect, outline="green")
        canvas.create_text(
            min(start_x, end_x) + 4, min(start_y, end_y) + 4,
            text=field_key, fill="green", anchor=tk.NW
        )

    canvas.bind("<ButtonPress-1>", on_press)
    canvas.bind("<B1-Motion>", on_drag)
    canvas.bind("<ButtonRelease-1>", on_release)

    print("\n── Interactive ROI Builder Started ──")
    print("1. Click and drag on the image to draw a region.")
    print("2. A popup will ask for the Field Key and Label.")
    print("3. The fractional coordinates will be printed directly to this console.")
    print("Close the window when finished.\n")

    # Bring window to front
    root.lift()
    root.attributes('-topmost', True)
    root.after_idle(root.attributes, '-topmost', False)
    root.mainloop()


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Draw field ROIs over the template PNG to verify alignment.\n"
            "Red boxes = text fields.  Blue boxes = checkbox fields.\n\n"
            "Use --gui for an interactive drag-and-drop tool to create fields."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "template",
        nargs="?",
        help="Path to the template PNG.",
    )
    parser.add_argument(
        "--device",
        choices=list(CANVAS),
        default="nomad",
        help="Supernote device model. Default: nomad.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output file path. Default: <template_stem>_roi_check.png.",
    )
    parser.add_argument(
        "--opacity",
        type=int,
        default=120,
        metavar="0-255",
        help="Fill opacity of the coloured overlays (default: 120).",
    )
    parser.add_argument(
        "--table",
        action="store_true",
        help="Print a plain-text table of all current ROI coordinates.",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Run the interactive Tkinter GUI to visually draw and generate new ROIs.",
    )
    args = parser.parse_args()

    if args.table:
        print_roi_table()

    if not args.template:
        parser.print_help()
        print("\nERROR: Please provide a template PNG path.")
        sys.exit(0)

    template_path = Path(args.template)
    if not template_path.exists():
        print(f"ERROR: File not found: {template_path}", file=sys.stderr)
        sys.exit(1)

    if args.gui:
        gui_converter(template_path=template_path, device=args.device)
        return

    visualise(
        template_path=template_path,
        device=args.device,
        out_path=Path(args.out) if args.out else None,
        opacity=max(0, min(255, args.opacity)),
    )

    print()
    print("Legend:  Red = text field   Blue = checkbox_group")
    print()
    print("To adjust or create a box interactively:")
    print("  1. Run:  python visualise_rois.py <your_template.png> --gui")
    print("  2. Click and drag the box where you want it.")
    print("  3. Paste the printed dictionary into FIELD_REGIONS in your scripts.")
    print("  4. Re-run this script (without --gui) to render a static verification image.")


if __name__ == "__main__":
    main()