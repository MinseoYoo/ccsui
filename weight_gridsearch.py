import os
import random
import argparse
import glob
import traceback
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from itertools import product
from pathlib import Path

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

    # 2) Semantic (skip OCR text to avoid heavy downloads; empty string allowed)
    s_sem, sem_break = semantic_score(ui_rgb, elem_rgb, "")

    # 3) LLM Overall (use UI image only). Normalize 1–10 -> 0–1
    llm = None
    explanation = ""
    try:
        llm_result = llm_score(ui_path)
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


def load_dataset(csv_path, ui_dir=None, element_dir=None):
    """
    CSV 파일에서 데이터셋 로드 (정답값 포함)
    CSV 구조:
    - rico_id 또는 id: UI 이미지 ID
    - score 또는 full_score: 정답 UICCS 점수
    
    파일명 매핑:
    - UI 이미지: {ui_dir}/{rico_id}.jpg
    - Element 이미지: {element_dir}/{rico_id}.png
    """
    df = pd.read_csv(csv_path, encoding='cp949')
    
    # 컬럼명 확인
    rico_id_col = None
    score_col = None
    
    # rico_id 컬럼 찾기
    for col in df.columns:
        if 'rico_id' in col.lower() or (col.lower() == 'id' and 'rico' not in csv_path.lower()):
            rico_id_col = col
            break
    
    # score 컬럼 찾기
    for col in df.columns:
        if 'new_score' in col.lower():
            score_col = col
            break
    
    if rico_id_col is None:
        raise ValueError("CSV에 rico_id 또는 id 컬럼을 찾을 수 없습니다.")
    if score_col is None:
        raise ValueError("CSV에 score 컬럼을 찾을 수 없습니다.")
    
    # 데이터셋 구성
    dataset = []
    skipped = 0
    
    for idx, row in df.iterrows():
        try:
            entry = {}
            
            # rico_id 읽기
            rico_id = str(row[rico_id_col]).strip()
            if not rico_id or rico_id == 'nan':
                skipped += 1
                continue
            
            # UI 이미지 경로
            if ui_dir:
                ui_path = Path(ui_dir) / f"{rico_id}.jpg"
            else:
                ui_path = Path(f"{rico_id}.jpg")
            
            # 파일 존재 확인
            if not ui_path.exists():
                for ext in ['.jpg', '.jpeg', '.png']:
                    alt_path = Path(ui_dir) / f"{rico_id}{ext}" if ui_dir else Path(f"{rico_id}{ext}")
                    if alt_path.exists():
                        ui_path = alt_path
                        break
                else:
                    skipped += 1
                    continue
            
            entry['ui_path'] = str(ui_path)
            
            # Element 이미지 경로
            if element_dir:
                element_path = Path(element_dir) / f"{rico_id}.png"
            else:
                element_path = Path(f"{rico_id}.png")
            
            # 파일 존재 확인
            if not element_path.exists():
                patterns = [f"{rico_id}.png", f"{rico_id}.jpg", f"{rico_id}.jpeg"]
                found = False
                for pattern in patterns:
                    alt_path = Path(element_dir) / pattern if element_dir else Path(pattern)
                    if alt_path.exists():
                        element_path = alt_path
                        found = True
                        break
                
                if not found:
                    skipped += 1
                    continue
            
            entry['element_path'] = str(element_path)
            
            # 정답 점수
            try:
                entry['ground_truth'] = float(row[score_col])
            except (ValueError, TypeError):
                skipped += 1
                continue
            
            dataset.append(entry)
            
        except Exception as e:
            print(f"Row {idx} 처리 중 오류: {str(e)}")
            skipped += 1
            continue
    
    if skipped > 0:
        print(f"경고: {skipped}개의 행을 건너뛰었습니다.")
    
    return dataset


def evaluate_weights(dataset, weights, error_metric='mse'):
    """
    주어진 가중치로 데이터셋 평가 (정답값과 비교)
    """
    predictions = []
    ground_truths = []
    
    for entry in dataset:
        try:
            result = compute_scores(entry['ui_path'], entry['element_path'], weights)
            predictions.append(result['UICCS'])
            ground_truths.append(entry['ground_truth'])
        except Exception as e:
            print(f"오류 발생 (UI: {entry.get('ui_path', 'unknown')}): {str(e)}")
            continue
    
    if len(predictions) == 0:
        return float('inf'), None
    
    predictions = np.array(predictions)
    ground_truths = np.array(ground_truths)
    
    # 에러 메트릭 계산
    if error_metric == 'mse':
        error = np.mean((predictions - ground_truths) ** 2)
    elif error_metric == 'mae':
        error = np.mean(np.abs(predictions - ground_truths))
    elif error_metric == 'rmse':
        error = np.sqrt(np.mean((predictions - ground_truths) ** 2))
    else:
        error = np.mean((predictions - ground_truths) ** 2)
    
    # 상관계수 계산
    correlation = np.corrcoef(predictions, ground_truths)[0, 1] if len(predictions) > 1 else 0.0
    
    return error, {
        'error': error,
        'correlation': correlation,
        'predictions': predictions.tolist(),
        'ground_truths': ground_truths.tolist()
    }


def grid_search_weights(csv_path: str, ui_dir: str = None, element_dir: str = None, 
                        output_file: str = "gridsearch_results.csv", 
                        error_metric: str = 'mse'):
    """
    세 가중치(w_style, w_sem, w_llm)에 대한 grid search 수행 (정답값과 비교)
    범위: 0.1부터 0.5까지 0.05단위
    """
    # 데이터셋 로드
    print("데이터셋 로드 중...")
    dataset = load_dataset(csv_path, ui_dir, element_dir)
    print(f"{len(dataset)}개의 샘플을 로드했습니다.")
    
    if len(dataset) == 0:
        raise ValueError("로드된 데이터셋이 비어있습니다.")
    
    # 가중치 범위 생성: 0.1부터 0.5까지 0.05단위
    weight_range = np.arange(0.15, 0.55, 0.2) 

    # 모든 가중치 조합 생성
    weight_combinations = []
    for w_style_raw, w_sem_raw, w_llm_raw in product(weight_range, weight_range, weight_range):
        # 가중치 정규화
        weights = normalize_weights(w_style_raw, w_sem_raw, w_llm_raw)
        weight_combinations.append((w_style_raw, w_sem_raw, w_llm_raw, weights))
    
    print(f"총 {len(weight_combinations)}개의 가중치 조합을 테스트합니다.")
    
    # Grid search 수행
    results = []
    best_error = float('inf')
    best_weights = None
    best_result = None
    
    for w_style_raw, w_sem_raw, w_llm_raw, weights in tqdm(weight_combinations, desc="Grid Search 진행 중"):
        error, result = evaluate_weights(dataset, weights, error_metric)
        
        w_style, w_sem, w_llm = weights
        results.append({
            'w_style_raw': w_style_raw,
            'w_sem_raw': w_sem_raw,
            'w_llm_raw': w_llm_raw,
            'w_style_norm': w_style,
            'w_sem_norm': w_sem,
            'w_llm_norm': w_llm,
            'error': error,
            'correlation': result['correlation'] if result else None
        })
        
        if error < best_error:
            best_error = error
            best_weights = weights
            best_result = result
    
    # 결과 정리
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('error')
    
    # 결과 저장
    results_df.to_csv(output_file, index=False)
    
    print("\n" + "="*50)
    print("Grid Search Results")
    print("="*50)
    print(f"최적 가중치 (w_style, w_sem, w_llm): {best_weights}")
    print(f"최적 에러 ({error_metric}): {best_error:.6f}")
    if best_result:
        print(f"상관계수: {best_result['correlation']:.6f}")
    print(f"\n결과가 {output_file}에 저장되었습니다.")
    print("\n상위 10개 가중치 조합:")
    print(results_df.head(10).to_string(index=False))
    
    return results_df


def main():
    parser = argparse.ArgumentParser(description="가중치 Grid Search (정답값과 비교)")
    parser.add_argument("--csv", type=str, required=True,
                       help="정답 UICCS 점수가 포함된 CSV 파일 경로 (rico_id/id, score 컬럼 필요)")
    parser.add_argument("--ui_dir", type=str, default="raw_ui", 
                       help="UI 이미지 디렉토리")
    parser.add_argument("--element_dir", type=str, default="element_crop", 
                       help="Element 이미지 디렉토리")
    parser.add_argument("--output", type=str, default="gridsearch_results.csv",
                       help="결과 저장 파일 경로")
    parser.add_argument("--error_metric", type=str, default="mse", 
                       choices=['mse', 'mae', 'rmse'],
                       help="에러 메트릭 (mse, mae, rmse)")
    
    args = parser.parse_args()
    
    grid_search_weights(args.csv, args.ui_dir, args.element_dir, args.output, args.error_metric)


if __name__ == "__main__":
    main()