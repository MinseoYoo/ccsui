import os
import argparse
import glob
import traceback
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

# local modules
from visual_score_cnn import visual_score
from semantic_score import semantic_score
from layout_llm_score import llm_score


def read_rgb(path):
    img = Image.open(path).convert("RGB")
    return np.array(img)


def normalize_weights(style: float, semantic: float, llm: float):
    total = style + semantic + llm
    if total <= 0:
        return 0.0, 0.0, 0.0
    return style / total, semantic / total, llm / total


def iter_files(directory, exts=("*.png", "*.jpg", "*.jpeg")):
    paths = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(directory, ext)))
    paths.sort()
    return paths


def compute_scores(ui_path: str, elem_path: str, weights):
    ui_rgb = read_rgb(ui_path)
    elem_rgb = read_rgb(elem_path)

    # 1) Style
    s_style, style_raw, s_hist = visual_score(ui_rgb, elem_rgb)

    # 2) Semantic (extract OCR text from UI and element)
    ui_text = ""
    elem_text = ""
    try:
        from semantic_score import ocr_text
        ui_text = ocr_text(ui_path)
        elem_text = ocr_text(elem_path)
    except Exception:
        # OCR 실패 시 빈 문자열 사용 (fallback to image-only comparison)
        pass
    ui_text_concat = f"{ui_text}\n{elem_text}".strip()
    s_sem, sem_break = semantic_score(ui_rgb, elem_rgb, ui_text_concat)

    # 3) Layout score
    llm = None
    explanation = ""
    try:
        llm_result = llm_score(ui_path, elem_path)
        llm = llm_result.get("score")
        explanation = llm_result.get("explanation", "")
    except Exception:
        # keep None and explanation empty on failure
        pass

    w_style, w_sem, w_llm = weights
    s_llm = float(llm) / 10.0 if isinstance(llm, (int, float)) else 0.0

    uiccs = w_style * float(s_style) + w_sem * float(s_sem) + w_llm * float(s_llm)

    return {
        "ui_path": ui_path,
        "element_path": elem_path,
        "S_style": float(s_style),
        "S_semantic": float(s_sem),
        "S_llm": float(s_llm),
        "UICCS": float(uiccs),
        "detail_style_raw": float(style_raw),
        "detail_style_hist": float(s_hist),
        "llm_explanation": explanation,
    }


def main():
    parser = argparse.ArgumentParser(description="Batch UICCS scoring over raw_ui and element_crop")
    parser.add_argument("--mode", choices=["mismatch", "right"], default=None,
                        help="mismatch: use mismatch_pairs.csv with mismatch_ui directory, right: use right_pairs.csv with raw_ui directory")
    parser.add_argument("--raw_ui_dir", default=None, help="Directory containing UI images (overrides mode default)")
    parser.add_argument("--element_dir", default=None, help="Directory containing element images (overrides mode default)")
    parser.add_argument("--out", default=None, help="Output CSV path (defaults based on mode)")
    parser.add_argument("--csv_file", type=str, default=None,
                        help="CSV file containing pairs (overrides mode default). CSV should have 'raw_ui' and 'element_crop' columns")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--no_reuse", action="store_true",
                        help="If set and enough elements exist, avoid reusing elements in random_mismatch mode")

    parser.add_argument("--limit_ui", type=int, default=0, help="Limit number of UI images (0 = no limit)")
    parser.add_argument("--limit_element", type=int, default=0, help="Limit number of element images (0 = no limit)")
    parser.add_argument("--style_w", type=float, default=0.3, help="Style weight")
    parser.add_argument("--semantic_w", type=float, default=0.3, help="Semantic weight")
    parser.add_argument("--llm_w", type=float, default=0.4, help="LLM weight")

    args = parser.parse_args()
    weights = normalize_weights(args.style_w, args.semantic_w, args.llm_w)

    rows = []
    num_errors = 0
 
    # Read pairs from CSV
    df = pd.read_csv(args.csv_file)
        
    # Check required columns
    if "raw_ui" not in df.columns or "element_crop" not in df.columns:
            raise ValueError(f"CSV file must contain 'raw_ui' and 'element_crop' columns. Found columns: {df.columns.tolist()}")
        
    pairs = []
    for _, row in df.iterrows():
        ui_filename = str(row["raw_ui"]).strip()
        elem_filename = str(row["element_crop"]).strip()
            
        if not ui_filename or not elem_filename or ui_filename == "nan" or elem_filename == "nan":
            continue
            
        ui_path = "dataset/right_pair"
        elem_path = "dataset/element_crop"
        pairs.append((ui_path, elem_path))
        
    if not pairs:
        raise ValueError(f"No valid pairs found in CSV file: {args.csv_file}")
        
    print(f"Loaded {len(pairs)} pairs from {args.csv_file}")
        
    # Process pairs
    for ui_path, elem_path in tqdm(pairs):
        try:
            rows.append(compute_scores(ui_path, elem_path, weights))
        except Exception:
            num_errors += 1
            traceback.print_exc()

    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"Saved {len(rows)} results to {args.out}. Errors: {num_errors}")
    
    # Calculate and print average UICCS score
    if rows:
        avg_uiccs = sum(row["UICCS"] for row in rows) / len(rows)
        print(f"Average UICCS score: {avg_uiccs:.4f}")
    else:
        print("No results to calculate average UICCS score.")


if __name__ == "__main__":
    main()


