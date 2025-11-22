################### Visuality Score Calculation ###################
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import models, transforms

'''
Style Transfer에서 VGG Feature 가지고 Gram matrices 
Gatys, L. A., Ecker, A. S., & Bethge, M. (2016). Image style transfer using convolutional neural networks. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 2414-2423).
https://bkshin.tistory.com/entry/%EC%BB%B4%ED%93%A8%ED%84%B0-%EB%B9%84%EC%A0%84-14-%EC%8A%A4%ED%83%80%EC%9D%BC-%EC%A0%84%EC%9D%B4Style-Transfer
'''
vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
vgg.to(device)

def hist_score(a, b, bins=32):
    Ha = cv2.calcHist([a],[0,1,2],None,[bins,bins,bins],[0,256]*3); Ha=cv2.normalize(Ha,Ha).flatten()
    Hb = cv2.calcHist([b],[0,1,2],None,[bins,bins,bins],[0,256]*3); Hb=cv2.normalize(Hb,Hb).flatten()
    bc = np.sum(np.sqrt(Ha*Hb))
    return float(np.clip(bc, 0, 1))

# Preprocessing for VGG
to_vgg = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

@torch.no_grad()
def gram_feats(img_pil, layers=(1, 6, 11, 20)):
    x = to_vgg(img_pil).unsqueeze(0).to(device)
    feats = {}
    cur = x
    for i, layer in enumerate(vgg):
        cur = layer(cur)
        if i in layers:
            B, C, H, W = cur.shape
            Fm = cur.view(B, C, H * W)
            G = torch.bmm(Fm, Fm.transpose(1, 2)) / (C * H * W)
            feats[i] = G
    return feats

@torch.no_grad()
def style_distance(context_rgb, element_rgb):
    A = Image.fromarray(context_rgb)
    B = Image.fromarray(element_rgb)
    GA = gram_feats(A)
    GB = gram_feats(B)
    d = 0.0
    for k in GA:
        diff = (GA[k] - GB[k]).pow(2).sum().sqrt().item()
        d += diff
    return d / len(GA)

# 정규화 범위 (데이터셋 기반으로 업데이트됨)
# 기본값은 데이터셋 분석 결과를 기반으로 설정 (percentile 5% ~ 95%)
_style_norm_min = 2.0
_style_norm_max = 7.0

def set_style_normalization_range(sd_min=None, sd_max=None):
    """
    Style distance 정규화 범위를 설정합니다.
    데이터셋 분석 후 호출하여 최적 범위로 업데이트할 수 있습니다.
    
    Args:
        sd_min: 최소값 (None이면 현재값 유지)
        sd_max: 최대값 (None이면 현재값 유지)
    """
    global _style_norm_min, _style_norm_max
    if sd_min is not None:
        _style_norm_min = sd_min
    if sd_max is not None:
        _style_norm_max = sd_max

def visual_score(context_rgb, element_rgb, sd_min=None, sd_max=None):
    """
    Visual style 점수를 계산합니다.
    
    Args:
        context_rgb: UI 이미지 (numpy array)
        element_rgb: Element 이미지 (numpy array)
        sd_min: 정규화 최소값 (None이면 전역 설정값 사용)
        sd_max: 정규화 최대값 (None이면 전역 설정값 사용)
    """
    raw = style_distance(context_rgb, element_rgb)

    # normalize based on dataset range (동적 또는 전역 설정값 사용)
    if sd_min is None:
        sd_min = _style_norm_min
    if sd_max is None:
        sd_max = _style_norm_max
    
    # 범위가 유효한지 확인
    if sd_max <= sd_min:
        sd_max = sd_min + 1.0  # fallback
    
    s = np.clip(1.0 - (raw - sd_min) / (sd_max - sd_min), 0, 1)

    s_hist = hist_score(context_rgb, element_rgb)
    return float(0.7 * s + 0.3 * s_hist), float(raw), float(s_hist)