#!/usr/bin/env python3
"""
parse_hybrid_record.py
──────────────────────
A Hybrid AI parser for Supernote RTR context records on Apple Silicon.

Uses `ocrmac` for spatial layout detection and checkbox heuristics, 
and `mlx-lm` (Apple MLX framework) to correct the OCR spelling using the 
highly accurate Supernote Real-Time Recognition text dump.

Dependencies (macOS only):
    pip install supernotelib pillow ocrmac mlx-lm

Usage:
    python parse_hybrid_record.py context_001.note
"""

import argparse
import json
import sys
from pathlib import Path
from PIL import Image

try:
    import supernotelib as sn
    from supernotelib.converter import TextConverter, ImageConverter
except ImportError:
    print("ERROR: supernotelib or pillow is missing. Run: pip install supernotelib pillow")
    sys.exit(1)

try:
    from ocrmac import ocrmac
except ImportError:
    print("ERROR: ocrmac is missing. Run: pip install ocrmac")
    sys.exit(1)

try:
    from mlx_lm import load, generate
except ImportError:
    print("ERROR: mlx-lm is missing. Run: pip install mlx-lm")
    sys.exit(1)

# Supernote canvas dimensions (Nomad default)
CANVAS_W, CANVAS_H = 1404, 1872

# The exact field schema
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

def extract_rtr_data(note_path: Path):
    """Extract Real-Time Recognition text and the visual image of the page."""
    try:
        if hasattr(sn, 'load_notebook'):
            notebook = sn.load_notebook(str(note_path))
        elif hasattr(sn, 'load'):
            notebook = sn.load(str(note_path))
        else:
            notebook = sn.notebook.load(str(note_path))

        # 1. Extract RTR Text
        txt_converter = TextConverter(notebook)
        texts = []
        for i in range(notebook.get_total_pages()):
            res = txt_converter.convert(i)
            if isinstance(res, bytes):
                res = res.decode("utf-8", errors="ignore")
            if res and isinstance(res, str):
                texts.append(res.strip())
        raw_text = "\n\n".join(texts).strip()

        # 2. Extract Image
        img_converter = ImageConverter(notebook)
        img = img_converter.convert(0).convert("RGB")
        if img.size != (CANVAS_W, CANVAS_H):
            img = img.resize((CANVAS_W, CANVAS_H), Image.LANCZOS)

        return raw_text, img

    except Exception as e:
        print(f"ERROR: Could not parse note data. Details: {e}")
        sys.exit(1)


def get_rough_spatial_data(img: Image.Image) -> dict:
    """Use ocrmac to generate a rough layout map of the text and checkboxes."""
    print("[ocrmac] Mapping spatial regions and heuristics...")
    rough_data = {}

    for field_key, meta in FIELD_REGIONS.items():
        x0, y0, x1, y1 = meta["roi"]
        crop = img.crop((int(x0 * CANVAS_W), int(y0 * CANVAS_H), int(x1 * CANVAS_W), int(y1 * CANVAS_H)))
        
        # Run Apple Vision OCR on the crop
        annotations = ocrmac.OCR(crop, recognition_level="accurate", language_preference=["en-US"]).recognize()
        raw_text = " ".join([t for t, conf, _ in annotations if conf > 0.3])
        
        if meta["type"] == "text":
            rough_data[field_key] = raw_text.strip()
            
        elif meta["type"] == "checkbox_group":
            # Heuristic for checkboxes
            options = meta["options"]
            raw_lower = raw_text.lower()
            tick_chars = {"✓", "✗", "x", "☑", "■", "✔", "v", "[x]", "[v]"}
            ticked = []
            
            for opt in options:
                if opt.lower() in raw_lower:
                    idx = raw_lower.find(opt.lower())
                    # Look for a tick mark within 10 characters of the option label
                    window = raw_lower[max(0, idx - 10): idx + len(opt) + 10]
                    if any(t in window for t in tick_chars):
                        ticked.append(opt)
            
            rough_data[field_key] = ticked if ticked else (["RAW: " + raw_text] if raw_text else [])

    return rough_data


def refine_with_mlx(rough_data: dict, rtr_text: str, model_id: str) -> dict:
    """Use Apple MLX text LLM to merge rough spatial data with perfect RTR spelling."""
    import re
    from mlx_lm import load, generate
    
    print(f"[mlx-lm] Loading model {model_id} into unified memory...")
    model, tokenizer = load(model_id)

    prompt = f"""
    You are a backend data extraction script. Your ONLY purpose is to output a raw JSON object.
    Do NOT output any conversational text. Do NOT explain your reasoning.

    1. ROUGH SPATIAL DATA (from OCR):
    {json.dumps(rough_data, indent=2)}

    2. PERFECT TEXT DUMP (from Stylus):
    {rtr_text}

    Merge the spatial mapping from 1 with the perfect spelling from 2.
    Return ONLY a valid JSON object matching the exact keys from the ROUGH SPATIAL DATA.
    """

    messages = [
        {"role": "system", "content": "You are a robotic JSON API. You only output raw JSON."},
        {"role": "user", "content": prompt}
    ]
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    print("[mlx-lm] Reasoning and generating final JSON...")
    
    response = generate(
        model, 
        tokenizer, 
        prompt=formatted_prompt, 
        max_tokens=1500, 
        verbose=False
    )

    # Hard-truncate the response at Llama 3's internal stop tokens before regex processing
    truncated_response = response.split("<|eot_id|>")[0].split("<|end_of_text|>")[0]
    
    # Extract the JSON payload from the sanitized string
    json_match = re.search(r'\{.*\}', truncated_response, re.DOTALL)
    
    if json_match:
        clean_response = json_match.group(0)
    else:
        clean_response = truncated_response.strip().strip("`").replace("json\n", "", 1)

    try:
        return json.loads(clean_response)
    except json.JSONDecodeError:
        print("\nERROR: Model did not return valid JSON. Raw output:")
        print(response)
        sys.exit(1)
def main():
    if len(sys.argv) == 1:
        print("\n=== Supernote Hybrid Parser (ocrmac + mlx-lm) ===")
        print("Usage: python parse_hybrid_record.py <path_to_your_note.note>")
        sys.exit(0)

    parser = argparse.ArgumentParser(description="Parse Supernote RTR notes using ocrmac and mlx-lm.")
    parser.add_argument("input", help="Path to a Real-Time Recognition .note file.")
    # We use a 4-bit quantized Llama 3 8B. It uses ~5GB RAM and runs incredibly fast on Mac.
    parser.add_argument(
        "--model", 
        default="mlx-community/Meta-Llama-3-8B-Instruct-4bit", 
        help="HuggingFace model ID for mlx-lm."
    )
    args = parser.parse_args()

    note_path = Path(args.input)
    if not note_path.exists():
        print(f"ERROR: File not found: {note_path}")
        sys.exit(1)
    
    # 1. Get raw text and image
    rtr_text, img = extract_rtr_data(note_path)
    print("\n--- Text from Supernote ---")
    print(json.dumps(rtr_text, indent=2))
    print("-----------------------------------\n")
    if not rtr_text:
        print("WARNING: No RTR text found. Ensure this note has Real-Time Recognition enabled.")
    
    # 2. Get rough spatial mapping via Apple Vision
    rough_data = get_rough_spatial_data(img)

    print("\n--- Rough Spatial Data (ocrmac) ---")
    print(json.dumps(rough_data, indent=2))
    print("-----------------------------------\n")

    # 3. Clean up the mapping using the text LLM
    structured_data = refine_with_mlx(rough_data, rtr_text, args.model)
    structured_data["_source_file"] = note_path.name
    structured_data["_engine"] = "ocrmac + mlx-lm"
    
    # 4. Output
    out_path = note_path.parent / f"{note_path.stem}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(structured_data, f, indent=2, ensure_ascii=False)
        
    print(f"\n[success] Data mapped and saved to {out_path}")

if __name__ == "__main__":
    main()