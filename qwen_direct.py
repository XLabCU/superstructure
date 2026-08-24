#!/usr/bin/env python3
"""
parse_rtr_record.py
───────────────────
Parses a Supernote Real-Time Recognition (RTR) .note file using Multimodal AI.
1. Extracts highly-accurate recognized text (RTR) via supernotelib.
2. Extracts the visual image of the page.
3. Feeds the Image, the RTR Text, and the spatial FIELD_REGIONS to Qwen-VL.
   (This allows Qwen to use the text for perfect spelling, and the image for 
   checkboxes and spatial field boundaries).

Dependencies
------------
    pip install supernotelib pillow torch accelerate qwen-vl-utils transformers

Usage
-----
    python parse_rtr_record.py context_001.note
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
    print("\nERROR: supernotelib is not installed.")
    print("Please run: pip install supernotelib pillow\n")
    sys.exit(1)


# The exact field schema, providing the LLM with types, options, and spatial ROIs
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

QWEN_MODELS = {
    "3B": "Qwen/Qwen2.5-VL-3B-Instruct",
    "7B": "Qwen/Qwen2.5-VL-7B-Instruct",
}

def extract_rtr_data(note_path: Path):
    """Extract both the Real-Time Recognition text and the visual image of the note."""
    print(f"[extract] Processing {note_path.name}...")
    
    try:
        if hasattr(sn, 'load_notebook'):
            notebook = sn.load_notebook(str(note_path))
        elif hasattr(sn, 'load'):
            notebook = sn.load(str(note_path))
        elif hasattr(sn, 'notebook') and hasattr(sn.notebook, 'load'):
            notebook = sn.notebook.load(str(note_path))
        else:
            raise RuntimeError("Could not find a valid load() function in supernotelib.")

        # 1. Extract Text
        txt_converter = TextConverter(notebook)
        texts = []
        for i in range(notebook.get_total_pages()):
            res = txt_converter.convert(i)
            if isinstance(res, bytes):
                res = res.decode("utf-8", errors="ignore")
            if res and isinstance(res, str):
                texts.append(res.strip())
        raw_text = "\n\n".join(texts).strip()

        # 2. Extract Image (Page 0)
        try:
            img_converter = ImageConverter(notebook)
            img = img_converter.convert(0)
            if img.mode != "RGB":
                img = img.convert("RGB")
        except Exception as e:
            print(f"[warn] Could not extract image, falling back to text-only: {e}")
            img = None

        return raw_text, img

    except Exception as e:
        print(f"\nERROR: Could not parse note data.")
        print(f"Details: {e}\n")
        sys.exit(1)


def map_to_json_with_qwen(raw_text: str, img: Image.Image, model_size: str = "3B") -> dict:
    """Use local Qwen-VL to semantically merge the RTR text and the spatial image."""
    import torch
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info

    repo = QWEN_MODELS[model_size]
    
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
        
    dtype = torch.bfloat16 if device != "cpu" else torch.float32

    print(f"[qwen] Loading {repo} on {device}...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(repo, torch_dtype=dtype, device_map="auto")
    processor = AutoProcessor.from_pretrained(repo)

    prompt = f"""
    You are an expert archaeological data parser analyzing a handwritten field record form.
    
    To assist you, here is the exact layout SCHEMA of the form. It includes:
    - The keys to use in your JSON output.
    - The expected data type ('text' vs 'checkbox_group').
    - The fractional 'roi' (Region of Interest) coordinates (x0, y0, x1, y1) defining where the field is located on the page.
    - For checkbox groups, the valid 'options'.

    SCHEMA:
    {json.dumps(FIELD_REGIONS, indent=2)}

    I am also providing the raw text extracted by the device's Real-Time Recognition engine. 
    This text is highly accurate for spelling but lacks spatial formatting and does not capture checkboxes well.

    RAW TEXT DUMP:
    ---
    {raw_text}
    ---

    Your task:
    1. Cross-reference the visual image (if provided) and the raw text to reconstruct the form data.
    2. For "text" fields, use the raw text for perfect spelling, relying on the schema's 'roi' locations and the image to know which text belongs to which field.
    3. For "checkbox_group" fields, rely primarily on the visual image to determine which options are ticked, marked, or crossed out.
    4. Return ONLY a valid JSON object matching the keys in the schema. Do not wrap it in markdown blockquotes (no ```json).
    """

    # Build the multimodal content payload
    content = []
    if img:
        content.append({"type": "image", "image": img})
    content.append({"type": "text", "text": prompt})

    messages = [{"role": "user", "content": content}]
    
    text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text_input], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt"
    ).to(device)

    print("[qwen] Structuring data...")
    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=1500, do_sample=False)
        
    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
    response = processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()

    # Clean up formatting if the model replies with ```json wrappers
    if response.startswith("```"):
        response = response.strip("`").replace("json\n", "", 1).strip()

    try:
        return json.loads(response)
    except json.JSONDecodeError:
        print("\nERROR: Model did not return valid JSON. Raw output:")
        print(response)
        sys.exit(1)


def main():
    if len(sys.argv) == 1:
        print("\n=== Supernote Multimodal RTR Parser ===")
        print("Usage: python parse_rtr_record.py <path_to_your_note.note>")
        sys.exit(0)

    parser = argparse.ArgumentParser(description="Parse Supernote RTR notes using Multimodal LLM mapping.")
    parser.add_argument("input", help="Path to a Real-Time Recognition .note file.")
    parser.add_argument("--qwen-model", choices=list(QWEN_MODELS), default="3B", help="Model size.")
    args = parser.parse_args()

    note_path = Path(args.input)
    if not note_path.exists():
        print(f"ERROR: File not found: {note_path}")
        sys.exit(1)
    
    # 1. Get raw RTR text AND the Image
    raw_text, img = extract_rtr_data(note_path)
    
    if not raw_text and not img:
        print("\nNo data found in the note.")
        return
        
    if raw_text:
        print("\n--- Raw Text Dump Extracted ---")
        print(raw_text[:200] + "...\n-------------------------------")

    # 2. Use Qwen to combine both streams of information contextually
    structured_data = map_to_json_with_qwen(raw_text, img, args.qwen_model)
    structured_data["_source_file"] = note_path.name
    
    # 3. Output
    out_path = note_path.parent / f"{note_path.stem}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(structured_data, f, indent=2, ensure_ascii=False)
        
    print(f"\n[success] Data mapped and saved to {out_path}")

if __name__ == "__main__":
    main()