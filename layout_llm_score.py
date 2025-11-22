import os
import base64
from openai import OpenAI
from typing import Dict, Union
from pathlib import Path
import re, json
import numpy as np
from PIL import Image
import tempfile
import io


from dotenv import load_dotenv
import httpx, certifi
load_dotenv()
http_client = httpx.Client(verify=certifi.where())

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), http_client=http_client)

# gpt prompt
try:
    with open('llm_prompt.txt', 'r', encoding='utf-8') as f:
        design_principle = f.read()
except Exception:
    design_principle = ""

def encode_image_gpt(image_input):
    """
    이미지 경로 또는 numpy array를 base64로 인코딩
    
    Args:
        image_input: 파일 경로(str) 또는 numpy array
    """
    try:
        # numpy array인 경우
        if isinstance(image_input, np.ndarray):
            # 값 범위 확인 및 정규화
            if image_input.dtype != np.uint8:
                # 0-255 범위로 정규화
                if image_input.max() <= 1.0:
                    image_input = (image_input * 255).astype(np.uint8)
                else:
                    image_input = np.clip(image_input, 0, 255).astype(np.uint8)
            
            # PIL Image로 변환
            if len(image_input.shape) == 3:
                # RGB 형식으로 변환 (numpy array는 RGB 형식이어야 함)
                if image_input.shape[2] == 3:
                    img = Image.fromarray(image_input, mode='RGB')
                elif image_input.shape[2] == 4:
                    img = Image.fromarray(image_input, mode='RGBA').convert('RGB')
                else:
                    raise ValueError(f"Unsupported image shape: {image_input.shape}")
            else:
                raise ValueError(f"Unsupported image shape: {image_input.shape}")
            
            # 메모리 버퍼에 저장하여 base64 인코딩
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            buffer.seek(0)
            result = base64.b64encode(buffer.read()).decode('utf-8')
            buffer.close()
            
            return result
        else:
            # 파일 경로인 경우 (기존 코드)
            if isinstance(image_input, str) or isinstance(image_input, (Path, os.PathLike)):
                with open(image_input, 'rb') as image_file:
                    return base64.b64encode(image_file.read()).decode('utf-8')
            else:
                raise TypeError(f"Unsupported image_input type: {type(image_input)}")
    except Exception as e:
        print(f"Error encoding image: {e}")
        import traceback
        traceback.print_exc()
        return None
    
prompt_single = f'''You are an autonomous AI Assistant who aids designers by providing insightful, objective, and constructive critiques of graphic design projects. Your goals are:
"Deliver comprehensive and unbiased evaluations of graphic designs based on the following design principles."
Grade seriously. The range of scores is from 1 to 10. A flawless design can earn 10 points, a mediocre design can only earn 7 points, a design with obvious shortcomings can only earn 4 points, and a very poor design can only earn 1-2 points.

{design_principle}

If the output is too long, it will be truncated. Only respond in JSON format, no other information. Example of output for a better graphic design:
{{"score": 6, explanation: "(Please concisely explain the reason of the score.)"}}

Please score the following image. [image]'''

prompt_pair = f'''You are an autonomous AI Assistant who aids designers by providing insightful, objective, and constructive critiques of graphic design projects. Your goals are:
"Deliver comprehensive and unbiased evaluations of graphic designs based on the following design principles."
Grade seriously. The range of scores is from 1 to 10. A flawless design can earn 10 points, a mediocre design can only earn 7 points, a design with obvious shortcomings can only earn 4 points, and a very poor design can only earn 1-2 points.

{design_principle}

**IMPORTANT**: You are evaluating how well a UI element (second image) harmonizes with the existing UI design (first image). Consider:
1. Visual consistency: Does the element match the style, colors, and design language of the UI?
2. Integration quality: Does the element fit naturally into the UI layout?
3. Overall coherence: Does adding this element improve or degrade the design?

If the output is too long, it will be truncated. Only respond in JSON format, no other information. Example of output:
{{"score": 6, explanation: "(Please concisely explain the reason of the score, focusing on element-UI harmony.)"}}

Please score how well the UI element (second image) harmonizes with the UI design (first image). [ui_image] [element_image]'''

MODEL = 'gpt-4o'

def parser(text: str) -> Dict[str, Union[int, str, None]]:
    """
    LLM 응답에서 {"score": <int>, "explanation": <str>}를 파싱합니다.
    - 코드펜스 제거, 첫 번째 JSON 오브젝트 추출
    - 단일 인용부호/키 따옴표 보정 시도
    - 실패 시 score=None, explanation=원문 반환
    """
    if text is None:
        return {"score": None, "explanation": ""}

    s = str(text).strip()
    # 코드펜스 제거
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)

    # 첫 JSON 객체 후보 추출
    m = re.search(r"\{.*?\}", s, flags=re.DOTALL)
    candidate = m.group(0) if m else s

    data = None
    try:
        data = json.loads(candidate)
    except Exception:
        # 단일 인용부호 -> 이중 인용부호, 키 따옴표 보정
        candidate2 = re.sub(r"(?<!\\)'", '"', candidate)
        candidate2 = re.sub(r'(\b[a-zA-Z_]\w*\b)\s*:', r'"\1":', candidate2)
        try:
            data = json.loads(candidate2)
        except Exception:
            return {"score": None, "explanation": s}

    score = data.get("score") if isinstance(data, dict) else None
    if isinstance(score, str):
        mnum = re.search(r"\d+", score)
        score = int(mnum.group(0)) if mnum else None
    elif isinstance(score, (int, float)):
        score = int(score)
    else:
        score = None

    explanation = None
    if isinstance(data, dict):
        explanation = data.get("explanation") or data.get("reason") or data.get("rationale")
    if isinstance(explanation, (int, float)):
        explanation = str(explanation)
    if explanation is None:
        explanation = ""

    return {"score": score, "explanation": explanation}

def llm_score(ui_input, element_input=None) -> Dict[str, Union[str, int, None]]:
    """
    UI 이미지와 element 이미지를 받아 LLM 점수 계산
    
    Args:
        ui_input: UI 이미지 경로(str) 또는 numpy array
        element_input: Element 이미지 경로(str) 또는 numpy array (None이면 UI만 평가)
    """
    base64_ui = encode_image_gpt(ui_input)
    if base64_ui is None:
        return {"score": None, "explanation": "Failed to encode UI image"}
    
    # Element가 제공된 경우 두 이미지를 함께 평가
    if element_input is not None:
        base64_element = encode_image_gpt(element_input)
        if base64_element is None:
            return {"score": None, "explanation": "Failed to encode element image"}
        
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt_pair
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{base64_ui}"}
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{base64_element}"}
                        }
                    ]
                }
            ],
            max_tokens=200,
            temperature=0
        )
    else:
        # Element가 없는 경우 UI만 평가 (하위 호환성)
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt_single
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{base64_ui}"}
                        }
                    ]
                }
            ],
            max_tokens=150,
            temperature=0
        )

    text = response.choices[0].message.content.strip()
    return parser(text)