#!/usr/bin/env python3
"""
parse_context_record.py
───────────────────────
Parses a completed Supernote .note file of the archaeological Context Record
form. Rasterises the note to a PNG, crops each form field's region of
interest, runs OCR on each crop, and outputs structured data as JSON and CSV.

Supported OCR engines  (select with --engine)
---------------------------------------------
  ocrmac     Apple Vision framework. macOS only. Best choice on Mac.
             pip install ocrmac

  rapidocr   RapidOCR via ONNX Runtime. Cross-platform (Linux/Windows/Mac).
             Same model quality as PaddleOCR without requiring PaddlePaddle.
             pip install rapidocr onnxruntime

  qwen       Qwen2.5-VL vision-language model running fully locally.
             Best handwriting accuracy; requires ~8 GB RAM for the 3B model,
             ~16 GB for 7B. Model is downloaded from HuggingFace on first run
             and cached locally — no data leaves the machine after that.
             pip install git+https://github.com/huggingface/transformers
             pip install torch accelerate qwen-vl-utils

Platform guidance
-----------------
  macOS (Apple Silicon)  →  --engine ocrmac   (default if ocrmac importable)
                             --engine qwen     (better accuracy, slower)
  Linux / Windows        →  --engine rapidocr (default on non-Mac)
                             --engine qwen     (better accuracy, needs RAM)

Usage
-----
  python parse_context_record.py context_001.note
  python parse_context_record.py context_001.note --engine qwen
  python parse_context_record.py context_001.note --engine qwen --qwen-model 7B
  python parse_context_record.py context_001.note --engine rapidocr
  python parse_context_record.py --batch ./field_notes/ --engine qwen
  python parse_context_record.py context_001.note --format json
  python parse_context_record.py context_001.note --debug-crops

Output
------
  context_001.json            one JSON object per form
  context_records_batch.csv   appended CSV row (shared across batch runs)

Notes on field regions
----------------------
All ROI coordinates are fractional (0.0–1.0) of the canvas:
  Nomad: 1404 × 1872 px    Manta: 1920 × 2560 px
Tune FIELD_REGIONS below if crops are misaligned; use --debug-crops to inspect.
"""

import argparse
import csv
import json
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

from PIL import Image

# ── Supernote canvas dimensions ───────────────────────────────────────────────
CANVAS = {
    "nomad": (1404, 1872),
    "manta": (1920, 2560),
}

# ── Available engine names ────────────────────────────────────────────────────
ENGINES = ["ocrmac", "rapidocr", "qwen"]

# ── Qwen model size → HuggingFace repo ───────────────────────────────────────
QWEN_MODELS = {
    "3B": "Qwen/Qwen2.5-VL-3B-Instruct",
    "7B": "Qwen/Qwen2.5-VL-7B-Instruct",
    "72B": "Qwen/Qwen2.5-VL-72B-Instruct",
}

# ── Form field regions (fractional x0, y0, x1, y1) ───────────────────────────
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


# ─────────────────────────────────────────────────────────────────────────────
# Rasterisation helpers
# ─────────────────────────────────────────────────────────────────────────────

def rasterise_note(note_path: Path, device: str = "nomad") -> Image.Image:
    """Convert a .note file to a PIL Image via supernote-tool."""
    try:
        import supernotelib  # noqa: F401
    except ImportError:
        raise ImportError("Run:  pip install supernotelib")

    with tempfile.TemporaryDirectory() as tmp:
        out_png = Path(tmp) / "page.png"
        for cmd in (
            ["supernote-tool", "convert", str(note_path), str(out_png)],
            [sys.executable, "-m", "supernotelib.cli", "convert", str(note_path), str(out_png)],
        ):
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode == 0 and out_png.exists():
                break
        else:
            raise RuntimeError(
                f"supernote-tool failed.\nSTDERR: {r.stderr}\n"
                "Make sure supernotelib is installed:  pip install supernotelib"
            )
        img = Image.open(str(out_png)).convert("RGB")

    tw, th = CANVAS[device]
    if img.size != (tw, th):
        img = img.resize((tw, th), Image.LANCZOS)
    return img


def load_image(path: Path, device: str = "nomad") -> Image.Image:
    """Load a .note, .png, .jpg, etc. and return a PIL Image at canvas size."""
    if path.suffix.lower() == ".note":
        return rasterise_note(path, device)
    img = Image.open(str(path)).convert("RGB")
    target = CANVAS[device]
    if img.size != target:
        img = img.resize(target, Image.LANCZOS)
    return img


def crop_roi(img: Image.Image, roi: tuple, device: str = "nomad") -> Image.Image:
    """Crop a fractional ROI from the full canvas image."""
    w, h = CANVAS[device]
    x0, y0, x1, y1 = roi
    return img.crop((int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)))


# ─────────────────────────────────────────────────────────────────────────────
# Engine: ocrmac  (macOS / Apple Vision)
# ─────────────────────────────────────────────────────────────────────────────

def _ocr_ocrmac(crop: Image.Image) -> str:
    """OCR via Apple Vision framework (macOS only). pip install ocrmac"""
    try:
        from ocrmac import ocrmac as _om
    except ImportError:
        raise ImportError(
            "ocrmac not installed.\nRun:  pip install ocrmac\n"
            "(macOS only — use --engine rapidocr or --engine qwen on other platforms)"
        )
    annotations = _om.OCR(
        crop,
        recognition_level="accurate",
        language_preference=["en-US"],
    ).recognize()
    return " ".join(
        t.strip() for t, conf, _ in annotations
        if t.strip() and conf > 0.3
    )


# ─────────────────────────────────────────────────────────────────────────────
# Engine: rapidocr  (cross-platform ONNX, same models as PaddleOCR)
# ─────────────────────────────────────────────────────────────────────────────

_RAPIDOCR_ENGINE = None

def _ocr_rapidocr(crop: Image.Image) -> str:
    """OCR via RapidOCR + ONNX Runtime. pip install rapidocr onnxruntime"""
    global _RAPIDOCR_ENGINE
    if _RAPIDOCR_ENGINE is None:
        try:
            from rapidocr import RapidOCR
        except ImportError:
            raise ImportError(
                "RapidOCR not installed.\n"
                "Run:  pip install rapidocr onnxruntime"
            )
        _RAPIDOCR_ENGINE = RapidOCR()

    import io
    import numpy as np

    # Round-trip through PNG to get a clean contiguous array
    buf = io.BytesIO()
    crop.convert("RGB").save(buf, format="PNG")
    buf.seek(0)
    arr = np.array(Image.open(buf).copy())

    result = _RAPIDOCR_ENGINE(arr)
    if result is None or result.txts is None:
        return ""
    return " ".join(
        t for t, score in zip(result.txts, result.scores)
        if t.strip() and score > 0.3
    )


# ─────────────────────────────────────────────────────────────────────────────
# Engine: qwen  (Qwen2.5-VL, fully local)
# ─────────────────────────────────────────────────────────────────────────────

_QWEN_MODEL = None
_QWEN_PROCESSOR = None
_QWEN_DEVICE = None

OCR_PROMPT = (
    "This is a cropped region of a handwritten archaeological context record form. "
    "Transcribe every handwritten word exactly as written. "
    "Return only the transcribed text — no commentary, no labels, no formatting."
)

def _ensure_qwen(model_size: str = "3B") -> None:
    """
    Download (first run) and load Qwen2.5-VL into memory.

    Models are cached by HuggingFace in ~/.cache/huggingface/ after the first
    download — no data leaves the machine during subsequent runs.

    Model sizes and approximate requirements:
      3B  — ~6 GB download,  ~8 GB RAM/VRAM   (good for laptops)
      7B  — ~14 GB download, ~16 GB RAM/VRAM  (good for workstations)
      72B — ~144 GB download, needs multiple GPUs or extreme RAM
    """
    global _QWEN_MODEL, _QWEN_PROCESSOR, _QWEN_DEVICE

    if _QWEN_MODEL is not None:
        return

    repo = QWEN_MODELS.get(model_size)
    if repo is None:
        raise ValueError(
            f"Unknown Qwen model size '{model_size}'. "
            f"Choose from: {list(QWEN_MODELS)}"
        )

    try:
        import torch
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    except ImportError:
        raise ImportError(
            "Qwen dependencies not installed. Run:\n"
            "  pip install git+https://github.com/huggingface/transformers\n"
            "  pip install torch accelerate qwen-vl-utils"
        )

    # Detect best available device
    if torch.cuda.is_available():
        _QWEN_DEVICE = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        _QWEN_DEVICE = "mps"
    else:
        _QWEN_DEVICE = "cpu"

    dtype = torch.bfloat16 if _QWEN_DEVICE != "cpu" else torch.float32

    print(f"[qwen] Loading {repo} on {_QWEN_DEVICE} ({dtype}) …")
    print(f"[qwen] First run downloads the model to ~/.cache/huggingface/")

    _QWEN_MODEL = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        repo,
        torch_dtype=dtype,
        device_map="auto",
    )
    _QWEN_PROCESSOR = AutoProcessor.from_pretrained(repo)
    print(f"[qwen] Model ready.")


def _ocr_qwen(crop: Image.Image, model_size: str = "3B") -> str:
    """
    OCR via Qwen2.5-VL running fully locally.

    The entire page crop is sent as a single image with a transcription prompt.
    The model generates the handwritten text directly — no region detection
    step needed since the VLM understands layout context.
    """
    import torch
    from qwen_vl_utils import process_vision_info

    _ensure_qwen(model_size)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": crop},
                {"type": "text",  "text": OCR_PROMPT},
            ],
        }
    ]

    text_input = _QWEN_PROCESSOR.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = _QWEN_PROCESSOR(
        text=[text_input],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(_QWEN_DEVICE)

    with torch.inference_mode():
        generated_ids = _QWEN_MODEL.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
        )

    # Strip the input tokens from the output
    trimmed = [
        out[len(inp):]
        for inp, out in zip(inputs.input_ids, generated_ids)
    ]
    decoded = _QWEN_PROCESSOR.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return decoded[0].strip() if decoded else ""


# ─────────────────────────────────────────────────────────────────────────────
# Unified OCR dispatch
# ─────────────────────────────────────────────────────────────────────────────

def ocr_crop(crop: Image.Image, engine: str, qwen_model: str = "3B") -> str:
    """Route a crop to the selected OCR engine and return recognised text."""
    if engine == "ocrmac":
        return _ocr_ocrmac(crop)
    elif engine == "rapidocr":
        return _ocr_rapidocr(crop)
    elif engine == "qwen":
        return _ocr_qwen(crop, model_size=qwen_model)
    else:
        raise ValueError(f"Unknown engine '{engine}'. Choose from: {ENGINES}")


def detect_checkboxes(
    crop: Image.Image,
    options: list,
    engine: str,
    qwen_model: str = "3B",
) -> list:
    """
    Identify ticked checkboxes in a crop.

    For ocrmac/rapidocr: heuristic scan for tick characters near option labels.
    For qwen: ask the model directly which boxes are ticked — more reliable.
    """
    if engine == "qwen":
        import torch
        from qwen_vl_utils import process_vision_info

        _ensure_qwen(qwen_model)
        opts_str = ", ".join(options)
        prompt = (
            f"This image shows a row of checkboxes from an archaeological form. "
            f"The options are: {opts_str}. "
            f"List only the options that have a tick, cross, or mark inside their box. "
            f"Return a comma-separated list, or the word NONE if nothing is ticked."
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": crop},
                    {"type": "text",  "text": prompt},
                ],
            }
        ]
        text_input = _QWEN_PROCESSOR.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = _QWEN_PROCESSOR(
            text=[text_input], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        ).to(_QWEN_DEVICE)
        with torch.inference_mode():
            generated_ids = _QWEN_MODEL.generate(**inputs, max_new_tokens=64, do_sample=False)
        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
        response = _QWEN_PROCESSOR.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()

        if response.upper() == "NONE" or not response:
            return []
        # Match response words against known options (case-insensitive)
        response_lower = response.lower()
        return [opt for opt in options if opt.lower() in response_lower]

    else:
        # Heuristic for ocrmac / rapidocr
        raw = ocr_crop(crop, engine, qwen_model)
        raw_lower = raw.lower()
        tick_chars = {"✓", "✗", "x", "☑", "■", "✔", "v", "[x]", "[v]"}
        ticked = []
        for opt in options:
            if opt.lower() in raw_lower:
                idx = raw_lower.find(opt.lower())
                window = raw_lower[max(0, idx - 10): idx + len(opt) + 10]
                if any(t in window for t in tick_chars):
                    ticked.append(opt)
        return ticked if ticked else (["RAW: " + raw] if raw else [])


# ─────────────────────────────────────────────────────────────────────────────
# Main parsing routine
# ─────────────────────────────────────────────────────────────────────────────

def parse_record(
    note_path: Path,
    device: str = "nomad",
    engine: str = "ocrmac",
    qwen_model: str = "3B",
    debug_crops: bool = False,
) -> dict:
    """Parse one completed context record form and return a structured dict."""
    print(f"[parse] Loading '{note_path.name}' …")
    img = load_image(note_path, device)
    print(f"[parse] Canvas size: {img.size}  |  engine: {engine}")

    if engine == "qwen":
        print(f"[parse] Qwen model: {QWEN_MODELS[qwen_model]}")
        _ensure_qwen(qwen_model)   # load once before the field loop

    debug_dir = note_path.parent / f"{note_path.stem}_debug_crops"
    if debug_crops:
        debug_dir.mkdir(exist_ok=True)
        print(f"[parse] Debug crops → {debug_dir}/")

    record = {
        "_source_file": str(note_path.name),
        "_device": device,
        "_engine": engine,
    }

    for field_key, meta in FIELD_REGIONS.items():
        roi   = meta["roi"]
        ftype = meta["type"]
        crop  = crop_roi(img, roi, device)

        if debug_crops:
            crop.save(str(debug_dir / f"{field_key}.png"))

        print(f"  [{field_key}] …", end=" ", flush=True)

        if ftype == "text":
            value = ocr_crop(crop, engine, qwen_model)
        elif ftype == "checkbox_group":
            value = detect_checkboxes(crop, meta.get("options", []), engine, qwen_model)
        else:
            value = ocr_crop(crop, engine, qwen_model)

        record[field_key] = value
        print(f"→ {str(value)[:60].replace(chr(10), ' ')!r}")

    return record


# ─────────────────────────────────────────────────────────────────────────────
# Output serialisers
# ─────────────────────────────────────────────────────────────────────────────

def save_json(record: dict, out_path: Path) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
    print(f"[output] JSON → {out_path}")


def save_csv(record: dict, out_path: Path) -> None:
    headers  = ["Source File", "Device", "Engine"] + [m["label"] for m in FIELD_REGIONS.values()]
    row_keys = ["_source_file", "_device", "_engine"] + list(FIELD_REGIONS.keys())
    row = []
    for k in row_keys:
        v = record.get(k, "")
        if isinstance(v, list):
            v = "; ".join(v)
        row.append(v)
    file_exists = out_path.exists()
    with open(out_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(headers)
        writer.writerow(row)
    print(f"[output] CSV  → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _default_engine() -> str:
    """Pick a sensible default engine based on platform and what's installed."""
    if platform.system() == "Darwin":
        try:
            from ocrmac import ocrmac  # noqa: F401
            return "ocrmac"
        except ImportError:
            pass
    try:
        from rapidocr import RapidOCR  # noqa: F401
        return "rapidocr"
    except ImportError:
        pass
    return "ocrmac"  # let the import error surface with a helpful message


def main():
    parser = argparse.ArgumentParser(
        description="Parse completed Supernote context-record .note files to JSON/CSV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Engine install commands
-----------------------
  ocrmac    (macOS only)   pip install ocrmac
  rapidocr  (cross-platform)  pip install rapidocr onnxruntime
  qwen      (all platforms, best accuracy, needs RAM)
            pip install git+https://github.com/huggingface/transformers
            pip install torch accelerate qwen-vl-utils

Qwen model sizes
----------------
  3B  — ~6 GB download,  ~8 GB RAM   (default; good for laptops)
  7B  — ~14 GB download, ~16 GB RAM  (better accuracy)
  72B — research/server use only
""",
    )
    parser.add_argument(
        "input",
        help="Path to a .note file, a rasterised PNG, or a folder (with --batch).",
    )
    parser.add_argument(
        "--engine",
        choices=ENGINES,
        default=None,
        help=(
            "OCR engine to use. "
            "Auto-detected if omitted (ocrmac on Mac, rapidocr elsewhere)."
        ),
    )
    parser.add_argument(
        "--qwen-model",
        choices=list(QWEN_MODELS),
        default="3B",
        metavar="SIZE",
        help="Qwen model size (3B / 7B / 72B). Only used with --engine qwen. Default: 3B.",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Process all .note files found in the input folder.",
    )
    parser.add_argument(
        "--device",
        choices=list(CANVAS),
        default="nomad",
        help="Supernote device model (affects canvas dimensions). Default: nomad.",
    )
    parser.add_argument(
        "--format",
        choices=["json", "csv", "both"],
        default="both",
        help="Output format. Default: both.",
    )
    parser.add_argument(
        "--outdir",
        default=None,
        help="Directory to write output files (default: same as input).",
    )
    parser.add_argument(
        "--debug-crops",
        action="store_true",
        help="Save each field's cropped image for ROI debugging.",
    )
    args = parser.parse_args()

    engine     = args.engine or _default_engine()
    qwen_model = args.qwen_model
    input_path = Path(args.input)
    outdir     = Path(args.outdir) if args.outdir else None

    print(f"Engine: {engine}" + (f" ({QWEN_MODELS[qwen_model]})" if engine == "qwen" else ""))

    def run_one(note_path: Path, batch_csv: Path) -> None:
        record = parse_record(
            note_path,
            device=args.device,
            engine=engine,
            qwen_model=qwen_model,
            debug_crops=args.debug_crops,
        )
        base_dir = outdir or note_path.parent
        base_dir.mkdir(parents=True, exist_ok=True)
        if args.format in ("json", "both"):
            save_json(record, base_dir / f"{note_path.stem}.json")
        if args.format in ("csv", "both"):
            save_csv(record, batch_csv)

    if args.batch:
        if not input_path.is_dir():
            print(f"ERROR: --batch requires a directory, got: {input_path}", file=sys.stderr)
            sys.exit(1)
        note_files = sorted(input_path.glob("*.note"))
        if not note_files:
            print(f"No .note files found in '{input_path}'.", file=sys.stderr)
            sys.exit(1)
        print(f"Batch mode: {len(note_files)} file(s).")
        batch_csv = (outdir or input_path) / "context_records_batch.csv"
        for nf in note_files:
            run_one(nf, batch_csv)
    else:
        if not input_path.exists():
            print(f"ERROR: File not found: {input_path}", file=sys.stderr)
            sys.exit(1)
        batch_csv = (outdir or input_path.parent) / "context_records_batch.csv"
        run_one(input_path, batch_csv)

    print("\nDone.")


if __name__ == "__main__":
    main()