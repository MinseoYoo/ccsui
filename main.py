'''
# Streamlit 모드 (기본)
python main.py
# 또는
python main.py --mode streamlit

# Grid search 모드
python main.py --mode grid_search --csv ground_truth_scores.csv --ui_dir raw_ui --element_dir element_crop

# Grid search 모드 (범위 지정)
python main.py --mode grid_search --csv scores.csv --ui_dir raw_ui --element_dir element_crop \
    --style_range "0.1,0.2,0.3,0.4" --semantic_range "0.1,0.2,0.3,0.4" --llm_range "0.1,0.2,0.3,0.4" \
    --error_metric mse --output results.csv
'''

import cv2
import argparse
import json
from pathlib import Path
from itertools import product
from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm

# score functions
from visual_score_cnn import visual_score
from semantic_score import semantic_score, ocr_text
from layout_llm_score import llm_score

def read_rgb(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def crop(img, bbox):
    x,y,w,h = bbox
    H,W = img.shape[:2]
    x2, y2 = min(x+w, W), min(y+h, H)
    return img[max(0,y):y2, max(0,x):x2]

def pad_context(img, bbox, pad_ratio=0.75):
    x, y, w, h = bbox
    H, W = img.shape[:2]
    px, py = int(w*pad_ratio), int(h*pad_ratio)
    x1, y1 = max(0, x-px), max(0, y-py)
    x2, y2 = min(W, x+w+px), min(H, y+h+py)
    return img[y1:y2, x1:x2]

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def calculate_uiccs_score(ui_rgb, element_rgb, bbox, weights, pad_ratio):
    """
    UICCS 점수를 계산하는 함수
    """
    try:
        # 컨텍스트 패치 추출 (bbox 주변)
        context = pad_context(ui_rgb, bbox, pad_ratio=pad_ratio)
        
        # 1) Style 점수 계산
        S_style, style_raw, S_hist = visual_score(context, element_rgb)
        
        # 2) Semantic 점수 계산 
        ocr_ui = ocr_text(ui_rgb) + "\n" + ocr_text(element_rgb)
        S_sem, sem_break = semantic_score(context, element_rgb, ocr_ui)
        
        # 3) LLM Overall 점수 계산
        S_llm = llm_score(context, element_rgb)
        
        # 최종 UICCS 점수 계산
        wv, ws, wl = weights
        UICCS = wv*S_style + ws*S_sem + wl*S_llm
        
        return {
            "S_style": float(S_style),
            "S_semantic": float(S_sem),
            "S_llm": float(S_llm),
            "UICCS": float(UICCS),
            "detail": {
                "bbox": bbox,
                "style_raw_distance": float(style_raw),
                "style_hist_similarity": float(S_hist),
                "semantic_breakdown": sem_break,
                "ocr_ui_sample": ocr_ui[:200]  # 미리보기
            },
            "weights": {"style": wv, "semantic": ws, "LLM": wl}
        }
    except Exception as e:
        print(f"점수 계산 중 오류가 발생했습니다: {str(e)}")
        return None

# ============================================================================
# Grid Search Functions
# ============================================================================

def load_dataset(csv_path, ui_dir=None, element_dir=None):
    """
    CSV 파일에서 데이터셋 로드
    CSV 구조 (answer_by_uicrit_cleaned.csv):
    - rico_id: UI 이미지 ID
    - full_score: 정답 점수
    - comments: 코멘트 (bbox 정보 포함 가능, 선택적)
    
    파일명 매핑:
    - UI 이미지: {ui_dir}/{rico_id}.jpg
    - Element 이미지: {element_dir}/{rico_id}.png
    """
    df = pd.read_csv(csv_path)
    
    # 컬럼명 확인
    rico_id_col = None
    score_col = None
    
    # rico_id 컬럼 찾기
    for col in df.columns:
        if 'rico_id' in col.lower() or 'id' in col.lower():
            rico_id_col = col
            break
    
    # score 컬럼 찾기
    for col in df.columns:
        if 'score' in col.lower() or 'full_score' in col.lower():
            score_col = col
            break
    
    if rico_id_col is None:
        raise ValueError("CSV에 rico_id 컬럼을 찾을 수 없습니다.")
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
            
            # UI 이미지 경로: {ui_dir}/{rico_id}.jpg
            if ui_dir:
                ui_path = Path(ui_dir) / f"{rico_id}.jpg"
            else:
                ui_path = Path(f"{rico_id}.jpg")
            
            # 파일 존재 확인
            if not ui_path.exists():
                # .jpg가 없으면 다른 확장자 시도
                for ext in ['.jpg', '.jpeg', '.png']:
                    alt_path = Path(ui_dir) / f"{rico_id}{ext}" if ui_dir else Path(f"{rico_id}{ext}")
                    if alt_path.exists():
                        ui_path = alt_path
                        break
                else:
                    skipped += 1
                    continue
            
            entry['ui_path'] = str(ui_path)
            
            # Element 이미지 경로: {element_dir}/{rico_id}.png
            if element_dir:
                element_path = Path(element_dir) / f"{rico_id}.png"
            else:
                element_path = Path(f"{rico_id}.png")
            
            # 파일 존재 확인 (여러 패턴 시도)
            if not element_path.exists():
                # 다른 패턴 시도
                patterns = [
                    f"{rico_id}.png",
                    f"{rico_id}.jpg",
                    f"{rico_id}.jpeg",
                ]
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
            
            # Bbox 정보: element 이미지 크기로 추정하거나 None
            # (comments에서 파싱은 복잡하므로 일단 None으로 설정)
            entry['bbox'] = None
            
            dataset.append(entry)
            
        except Exception as e:
            print(f"Row {idx} 처리 중 오류: {str(e)}")
            skipped += 1
            continue
    
    if skipped > 0:
        print(f"경고: {skipped}개의 행을 건너뛰었습니다.")
    
    return dataset

def evaluate_weights(dataset, weights, pad_ratio=0.75, error_metric='mse'):
    """
    주어진 가중치로 데이터셋 평가
    """
    predictions = []
    ground_truths = []
    
    for entry in tqdm(dataset, desc="Evaluating weights"):
        try:
            # 이미지 로드
            ui_rgb = read_rgb(entry['ui_path'])
            element_rgb = read_rgb(entry['element_path'])
            
            # Bbox 설정
            if entry['bbox'] is None:
                # 전체 이미지 사용
                H, W = ui_rgb.shape[:2]
                bbox = (0, 0, W, H)
            else:
                bbox = entry['bbox']
            
            # 점수 계산
            result = calculate_uiccs_score(ui_rgb, element_rgb, bbox, weights, pad_ratio)
            
            if result is not None:
                predictions.append(result['UICCS'])
                ground_truths.append(entry['ground_truth'])
        except Exception as e:
            print(f"Error processing {entry.get('ui_path', 'unknown')}: {str(e)}")
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

def grid_search_weights(csv_path, ui_dir=None, element_dir=None, 
                        style_range=None, semantic_range=None, llm_range=None,
                        pad_ratio=0.75, error_metric='mse'):
    """
    Grid search를 사용하여 최적 가중치 찾기
    """
    # 데이터셋 로드
    print("Loading dataset...")
    dataset = load_dataset(csv_path, ui_dir, element_dir)
    print(f"Loaded {len(dataset)} samples")
    
    # 기본 탐색 범위 설정
    if style_range is None:
        style_range = np.arange(0.1, 0.5, 0.05)
    if semantic_range is None:
        semantic_range = np.arange(0.1, 0.5, 0.05)
    if llm_range is None:
        llm_range = np.arange(0.1, 0.5, 0.05)
    # 모든 가중치 조합 생성 (합이 1이 되도록 정규화)
    weight_combinations = []
    for wv, ws, wl in product(style_range, semantic_range, llm_range):
        total = wv + ws + wl
        if total > 0:
            # 정규화
            wv_norm = wv / total
            ws_norm = ws / total
            wl_norm = wl / total
            weight_combinations.append((wv_norm, ws_norm, wl_norm))
    
    print(f"Testing {len(weight_combinations)} weight combinations...")
    
    # Grid search 수행
    results = []
    best_error = float('inf')
    best_weights = None
    best_result = None
    
    for weights in tqdm(weight_combinations, desc="Grid search"):
        error, result = evaluate_weights(dataset, weights, pad_ratio, error_metric)
        
        results.append({
            'weights': weights,
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
    
    print("\n" + "="*50)
    print("Grid Search Results")
    print("="*50)
    print(f"Best weights (wv, ws, wl): {best_weights}")
    print(f"Best error ({error_metric}): {best_error:.6f}")
    if best_result:
        print(f"Correlation: {best_result['correlation']:.6f}")
    print("\nTop 10 combinations:")
    print(results_df.head(10).to_string(index=False))
    
    return {
        'best_weights': best_weights,
        'best_error': best_error,
        'best_result': best_result,
        'all_results': results_df
    }

def run_grid_search(args):
    """Grid search 모드 실행"""
    # 범위 파싱
    def parse_range(range_str):
        if range_str is None:
            return None
        if '...' in range_str:
            parts = range_str.split('...')
            start = float(parts[0])
            end = float(parts[-1])
            step = float(parts[1]) if len(parts) > 2 else 0.1
            return np.arange(start, end + step, step)
        else:
            return [float(x) for x in range_str.split(',')]
    
    style_range = parse_range(args.style_range)
    semantic_range = parse_range(args.semantic_range)
    llm_range = parse_range(args.llm_range)
    
    # Grid search 실행
    results = grid_search_weights(
        csv_path=args.csv,
        ui_dir=args.ui_dir,
        element_dir=args.element_dir,
        style_range=style_range,
        semantic_range=semantic_range,
        llm_range=llm_range,
        pad_ratio=args.pad_ratio,
        error_metric=args.error_metric
    )
    
    # 결과 저장
    results['all_results'].to_csv(args.output, index=False)
    print(f"\nResults saved to {args.output}")
    
    # 최적 가중치 저장
    best_weights_file = args.output.replace('.csv', '_best_weights.json')
    with open(best_weights_file, 'w') as f:
        json.dump({
            'best_weights': {
                'style_weight': results['best_weights'][0],
                'semantic_weight': results['best_weights'][1],
                'llm_weight': results['best_weights'][2]
            },
            'best_error': results['best_error'],
            'correlation': results['best_result']['correlation'] if results['best_result'] else None
        }, f, indent=2)
    print(f"Best weights saved to {best_weights_file}")

def run_streamlit():
    """Streamlit 모드 실행"""
    import streamlit as st
    from streamlit_drawable_canvas import st_canvas
    import io
    
    st.title("UI Element Compatibility Score (UICCS)")
    st.markdown("UI에 추가할 요소의 호환성을 평가합니다.")
    
    # 사이드바에서 가중치 조정
    st.sidebar.header("가중치 설정")
    style_weight = st.sidebar.slider("Visual Style Weight", 0.0, 1.0, 0.30, 0.05)
    semantic_weight = st.sidebar.slider("Semantic Weight", 0.0, 1.0, 0.20, 0.05)
    llm_weight = st.sidebar.slider("Overall LLM Weight", 0.0, 1.0, 0.40, 0.05)
    
    # 가중치 정규화
    total_weight = style_weight + semantic_weight + llm_weight
    if total_weight > 0:
        style_weight /= total_weight
        semantic_weight /= total_weight
    
    weights = (style_weight, semantic_weight, llm_weight)
    
    # 패딩 비율 설정
    pad_ratio = st.sidebar.slider("Context Padding Ratio", 0.1, 2.0, 0.75, 0.05)
    
    # 이미지 업로드 섹션
    st.subheader("1. 이미지 업로드")
    
    # UI 이미지 업로드
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**UI 이미지 업로드**")
        ui_uploaded_file = st.file_uploader(
            "UI 이미지를 업로드하세요", 
            type=['png', 'jpg', 'jpeg'],
            key="ui_upload"
        )
        
        if ui_uploaded_file is not None:
            base_img = Image.open(ui_uploaded_file).convert("RGBA")
            img_w, img_h = base_img.size
            st.image(base_img, caption="업로드된 UI 이미지", use_container_width=True)
        else:
            st.error("기본 이미지를 찾을 수 없습니다. UI 이미지를 업로드해주세요.")
            
    
    with col2:
        st.markdown("**추가할 요소 이미지 업로드**")
        element_uploaded_file = st.file_uploader(
            "추가할 요소 이미지를 업로드하세요", 
            type=['png', 'jpg', 'jpeg'],
            key="element_upload"
        )
        
        if element_uploaded_file is not None:
            element_img = Image.open(element_uploaded_file).convert("RGBA")
            st.image(element_img, caption="업로드된 요소 이미지", use_container_width=True)
        else:
            st.info("요소 이미지를 업로드해주세요.")
    
    # UI 이미지가 없으면 중단
    if ui_uploaded_file is None and 'base_img' not in locals():
        st.stop()
    
    # 그리기 캔버스
    st.subheader("2. 평가할 영역 선택")
    st.markdown("UI 이미지에서 평가하고 싶은 영역을 빨간 사각형으로 그려주세요")
    
    result = st_canvas(
        background_image=base_img,
        drawing_mode="rect",
        height=img_h,                 
        width=img_w,                  
        stroke_width=2,
        stroke_color="#EE0000",
        fill_color="rgba(0,0,0,0.2)",
        update_streamlit=True,
        display_toolbar=True,
        key="canvas",
    )
    
    # 그려진 사각형이 있고 요소 이미지가 업로드된 경우 처리
    if result.json_data is not None and element_uploaded_file is not None:
        df = pd.json_normalize(result.json_data["objects"])
        if not df.empty:
            # bbox 좌표 추출
            left = int(df["left"].iloc[0])
            top = int(df["top"].iloc[0])
            width = int(df["width"].iloc[0])
            height = int(df["height"].iloc[0])
            
            # 좌표 클램핑
            x1 = clamp(left, 0, img_w)
            y1 = clamp(top, 0, img_h)
            x2 = clamp(left + width, 0, img_w)
            y2 = clamp(top + height, 0, img_h)
            
            # bbox 형식으로 변환 (x, y, w, h)
            bbox = (x1, y1, x2-x1, y2-y1)
            
            # 크롭된 요소 추출
            cropped = base_img.crop((x1, y1, x2, y2))
            
            # 3. 비교 영역 표시
            st.subheader("3. 비교 영역")
            
            # 2컬럼으로 cropped와 업로드한 이미지 비교
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**UI에서 크롭된 영역**")
                st.image(cropped, caption=f"선택 영역 크롭\n좌표: ({x1}, {y1}, {x2}, {y2})", use_container_width=True)
                
                # 컨텍스트 영역도 표시
                st.markdown("**컨텍스트 영역**")
                context = pad_context(np.array(base_img.convert("RGB")), bbox, pad_ratio=pad_ratio)
                st.image(context, caption=f"컨텍스트 영역 (패딩 비율: {pad_ratio})", use_container_width=True)
            
            with col2:
                st.markdown("**업로드한 요소 이미지**")
                st.image(element_img, caption="업로드된 요소 이미지", use_container_width=True)
                
                # 요소 이미지 크기 정보
                elem_w, elem_h = element_img.size
                st.info(f"요소 이미지 크기: {elem_w} x {elem_h}")
            
            # 4. 점수 계산
            st.subheader("4. UICCS 점수 계산")
            
            # 점수 계산 버튼
            if st.button("UICCS 점수 계산", type="primary"):
                # PIL Image를 numpy array로 변환
                ui_rgb = np.array(base_img.convert("RGB"))
                element_rgb = np.array(element_img.convert("RGB"))
                
                # 점수 계산
                with st.spinner("점수를 계산하는 중..."):
                    score_result = calculate_uiccs_score(ui_rgb, element_rgb, bbox, weights, pad_ratio)
                
                if score_result:
                    # 결과 표시
                    st.subheader("UICCS 점수 결과")
                    
                    # 메인 점수 카드
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Style", f"{score_result['S_style']:.3f}")
                    with col2:
                        st.metric("Semantic", f"{score_result['S_semantic']:.3f}")
                    with col3:
                        st.metric("LLM", f"{score_result['S_llm']:.3f}")
                    with col4:
                        st.metric("**UICCS**", f"**{score_result['UICCS']:.3f}**", 
                                delta=f"{score_result['UICCS']:.1%} 호환성")
                    
                    # 상세 정보
                    with st.expander("상세 정보"):
                        detail = score_result['detail']
                        st.write(f"**Bbox 좌표:** {detail['bbox']}")
                        st.write(f"**Style Raw Distance:** {detail['style_raw_distance']:.3f}")
                        st.write(f"**Style Histogram Similarity:** {detail['style_hist_similarity']:.3f}")
                        
                        st.write("**Semantic Breakdown:**")
                        for key, value in detail['semantic_breakdown'].items():
                            st.write(f"  - {key}: {value:.3f}")
                        
                        st.write("**OCR 텍스트 샘플:**")
                        st.text(detail['ocr_ui_sample'])
                    
                    # 가중치 정보
                    with st.expander("사용된 가중치"):
                        weights_info = score_result['weights']
                        for key, value in weights_info.items():
                            st.write(f"**{key.title()}:** {value:.3f}")
                    
                    # === 3-b. 요소 붙여넣기 미리보기 ===
                    st.subheader("3-b. 요소 붙여넣기 미리보기")
                    
                    # 미리보기 옵션
                    col_prev1, col_prev2, col_prev3 = st.columns(3)
                    with col_prev1:
                        keep_ratio = st.checkbox("비율 유지(권장)", value=True)
                    with col_prev2:
                        fit_mode = st.selectbox("리사이즈 방식", ["fit(내부맞춤)", "cover(잘라서 채움)"], index=0)
                    with col_prev3:
                        alpha = st.slider("요소 투명도", 0.0, 1.0, 1.0, 0.05)
                    
                    # bbox 크기
                    bbox_w, bbox_h = bbox[2], bbox[3]
                    
                    # 요소 리사이즈 함수
                    def resize_element_to_bbox(elem: Image.Image, bw: int, bh: int, keep_ratio: bool=True, mode: str="fit") -> Image.Image:
                        ew, eh = elem.size
                        if not keep_ratio:
                            # 비율 무시하고 딱 맞춤
                            return elem.resize((max(1, bw), max(1, bh)), Image.LANCZOS)
                        
                        # 비율 유지
                        if mode.startswith("fit"):
                            # 요소가 bbox 안에 '전부' 들어오도록 축소/확대
                            scale = min(bw / ew, bh / eh) if ew and eh else 1.0
                            new_w, new_h = max(1, int(ew * scale)), max(1, int(eh * scale))
                            return elem.resize((new_w, new_h), Image.LANCZOS)
                        else:
                            # cover: bbox를 가득 채우되 넘치는 부분은 잘림 (중앙 기준)
                            scale = max(bw / ew, bh / eh) if ew and eh else 1.0
                            new_w, new_h = max(1, int(ew * scale)), max(1, int(eh * scale))
                            resized = elem.resize((new_w, new_h), Image.LANCZOS)
                            # 중앙 크롭
                            left = max(0, (new_w - bw) // 2)
                            top = max(0, (new_h - bh) // 2)
                            right = left + bw
                            bottom = top + bh
                            return resized.crop((left, top, right, bottom))
                    
                    # RGBA 보장
                    base_rgba = base_img.convert("RGBA")
                    elem_rgba = element_img.convert("RGBA")
                    
                    # 리사이즈
                    elem_fitted = resize_element_to_bbox(elem_rgba, bbox_w, bbox_h, keep_ratio=keep_ratio, mode=fit_mode)
                    
                    # 투명도 적용
                    if alpha < 1.0:
                        # 요소 알파 채널에 곱하기
                        r, g, b, a = elem_fitted.split()
                        a = a.point(lambda v: int(v * alpha))
                        elem_fitted = Image.merge("RGBA", (r, g, b, a))
                    
                    # 붙여넣기 좌표(중앙 정렬)
                    paste_x = bbox[0] + (bbox_w - elem_fitted.size[0]) // 2
                    paste_y = bbox[1] + (bbox_h - elem_fitted.size[1]) // 2
                    
                    # 합성
                    preview = base_rgba.copy()
                    layer = Image.new("RGBA", preview.size, (0, 0, 0, 0))
                    layer.paste(elem_fitted, (paste_x, paste_y), elem_fitted)
                    preview = Image.alpha_composite(preview, layer)
                    
                    # 표시
                    st.image(preview, caption=f"미리보기: 요소를 bbox에 붙여넣은 예시\n붙여넣기 좌표: ({paste_x}, {paste_y}), 크기: {elem_fitted.size}", use_container_width=True)
                    
                    # 다운로드 버튼
                    buf = io.BytesIO()
                    preview.save(buf, format="PNG")
                    st.download_button(
                        label="미리보기 이미지 다운로드 (PNG)",
                        data=buf.getvalue(),
                        file_name="uiccs_preview.png",
                        mime="image/png"
                    )

def main():
    parser = argparse.ArgumentParser(description='UICCS: UI Element Compatibility Score')
    parser.add_argument('--mode', type=str, choices=['streamlit', 'grid_search'], default='streamlit',
                       help='실행 모드: streamlit (웹 UI) 또는 grid_search (가중치 최적화)')
    
    # Grid search 관련 인자들
    parser.add_argument('--csv', type=str, default=None, 
                       help='Grid search 모드: 정답 점수가 포함된 CSV 파일 경로')
    parser.add_argument('--ui_dir', type=str, default=None, 
                       help='Grid search 모드: UI 이미지 디렉토리 경로')
    parser.add_argument('--element_dir', type=str, default=None,
                       help='Grid search 모드: Element 이미지 디렉토리 경로')
    parser.add_argument('--pad_ratio', type=float, default=0.75, 
                       help='Context padding ratio')
    parser.add_argument('--error_metric', type=str, default='mse', choices=['mse', 'mae', 'rmse'], 
                       help='Grid search 모드: 최소화할 에러 메트릭')
    parser.add_argument('--style_range', type=str, default=None, 
                       help='Grid search 모드: Style 가중치 범위 (예: "0.0,0.1,0.2,...,1.0" 또는 "0.0...1.0")')
    parser.add_argument('--semantic_range', type=str, default=None,
                       help='Grid search 모드: Semantic 가중치 범위')
    parser.add_argument('--llm_range', type=str, default=None,
                       help='Grid search 모드: LLM 가중치 범위')
    parser.add_argument('--output', type=str, default='grid_search_results.csv',
                       help='Grid search 모드: 결과 저장 파일 경로')
    
    args = parser.parse_args()
    
    if args.mode == 'grid_search':
        if args.csv is None:
            parser.error("--csv is required for grid_search mode")
        run_grid_search(args)
    else:
        # Streamlit 모드
        run_streamlit()

if __name__ == "__main__":
    main()


