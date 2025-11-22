################### Semantic Score Calculation ###################
import open_clip
import numpy as np
import torch, torch.nn.functional as F
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, AutoTokenizer
from sentence_transformers import SentenceTransformer
import unicodedata

def load_clip():
    try:
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k"
        )
        tokenizer = open_clip.get_tokenizer("ViT-B-32")
        return model.eval(), preprocess, tokenizer, True
    except Exception:
        import clip
        model, preprocess = clip.load("ViT-B/32")
        return model.eval(), preprocess, None, False

clip_model, clip_preprocess, clip_tokenizer, is_open_clip = load_clip()
st_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def clip_img_emb(img_rgb):
    img = Image.fromarray(img_rgb)
    x = clip_preprocess(img).unsqueeze(0)
    with torch.no_grad():
        feats = clip_model.encode_image(x)
        feats = F.normalize(feats, dim=-1)
    return feats.cpu().numpy()[0]

def clip_txt_emb(text):
    with torch.no_grad():
        if is_open_clip:
            tok = clip_tokenizer([text])
            feats = clip_model.encode_text(tok)
        else:
            import clip
            tok = clip.tokenize([text], truncate=True)
            feats = clip_model.encode_text(tok)
        feats = F.normalize(feats, dim=-1)
    return feats.cpu().numpy()[0]

def ocr_text(image):
    processor = TrOCRProcessor.from_pretrained("ddobokki/ko-trocr") 
    model = VisionEncoderDecoderModel.from_pretrained("ddobokki/ko-trocr")
    tokenizer = AutoTokenizer.from_pretrained("ddobokki/ko-trocr")
    img = Image.open(image)

    pixel_values = processor(img, return_tensors="pt").pixel_values 
    generated_ids = model.generate(pixel_values, max_length=64)
    generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    generated_text = unicodedata.normalize("NFC", generated_text)
    return generated_text

def sent_emb(text):
    if not text.strip():
        return np.zeros((384,), dtype=np.float32)
    return st_model.encode([text], normalize_embeddings=True)[0]

def cosine_similarity(a,b):
    a = a/(np.linalg.norm(a)+1e-8)
    b = b/(np.linalg.norm(b)+1e-8)
    return float(np.clip(np.dot(a,b), -1, 1))


def semantic_score(context_rgb, element_rgb, ui_text_concat):
    e_img = clip_img_emb(element_rgb)  
    c_img = clip_img_emb(context_rgb)  
    s_imgimg = (cosine_similarity(e_img, c_img)+1)/2  # 기존 UI와 추가하는 요소 간 i-i cos_sim

    # 텍스트가 있는 경우에만 img_txt 점수 계산
    has_text = ui_text_concat and ui_text_concat.strip()
    if has_text:
        try:
            c_txt = clip_txt_emb(ui_text_concat)
            s_imgtxt = (cosine_similarity(e_img, c_txt)+1)/2  # 추가하는 요소 - 기존 UI에 있는 텍스트 간 mm cos_sim
            # 텍스트가 있을 때: img_img에 더 높은 가중치 (이미지-이미지가 더 신뢰할 수 있음)
            s = 0.6*s_imgimg + 0.4*s_imgtxt
        except Exception:
            # 텍스트 임베딩 실패 시 이미지-이미지만 사용
            s = s_imgimg
            s_imgtxt = s_imgimg
    else:
        # 텍스트가 없을 경우 이미지-이미지만 사용
        s = s_imgimg
        s_imgtxt = s_imgimg
    
    return float(s), {"img_img": float(s_imgimg), "txt_txt": 0.0, "img_txt": float(s_imgtxt)}
