"""
این ماژول مسئول «بهبود نرم تصاویر» بر اساس ویژگی‌های بصری است تا
 خروجی برای انتشار در اینستاگرام جذاب‌تر شود و از اغراق اجتناب شده

"""

import cv2
import numpy as np
from ultralytics import YOLO
from features import (
    sharpness_laplacian,
    exposure_quality,
    rms_contrast,
    colorfulness,
    saturation_mean,
    subject_total_score,
)
import os

# -----------------------
# مدل YOLOv8-seg
# -----------------------
yolo_model = YOLO("yolov8s-seg.pt")
"""
برای فهمیدن اینکه 'سوژه اصلی' کجاست و ایجاد ماسک نسبتا دقیق دور آن،
 از مدل سگمنتیشن یولو استفاده می‌شود

"""
# -----------------------
# ذخیره عکس ادیت‌شده
# -----------------------
def save_enhanced_photo(enhanced_img, original_path):
    folder = "enhanced_photos"
    os.makedirs(folder, exist_ok=True)
    base_name = os.path.basename(original_path)
    name, ext = os.path.splitext(base_name)
    save_path = os.path.join(folder, f"{name}_enhanced{ext}")
    cv2.imwrite(save_path, enhanced_img)
    return save_path

"""
 توابع ادیت نرم (Soft Enhancements)

"""

# فقط کمی وضوح اضافه می‌شود 
# تا عکس «بیش‌ازحد شارپ و مصنوعی» نشود

def increase_sharpness_soft(img, alpha=0.6):
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(img, -1, kernel)
    return cv2.addWeighted(img, 1 - alpha, sharpened, alpha, 0)

# افزایش یا کاهش خطی شدت روشنایی پیکسل‌ها

def adjust_contrast_exposure_soft(img, contrast=1.05, brightness=5):
    return cv2.convertScaleAbs(img, alpha=contrast, beta=brightness)

# در فضای HSV 
# کمی اشباع بیشتر → رنگ‌های کمی بی‌روح

def adjust_saturation_soft(img, factor=1.1):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    img_hsv[..., 1] = np.clip(img_hsv[..., 1] * factor, 0, 255)
    return cv2.cvtColor(img_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

# -----------------------
# ماسک YOLOv8-seg
# -----------------------
def get_best_mask(img):
    results = yolo_model(img, verbose=False)[0]
    mask = None
    best_score = 0
    subject_cls = "none"

    if results.masks is not None:
        for m, conf, cls in zip(results.masks.data, results.boxes.conf, results.boxes.cls):
            m_np = m.cpu().numpy()
            area = np.sum(m_np > 0)
            score = float(conf) * area
            if score > best_score:
                best_score = score
                mask = m_np
                subject_cls = yolo_model.names[int(cls)]

    return mask, subject_cls

# تبدیل ماسک به جعبه

def mask_to_box(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return xs.min(), ys.min(), xs.max(), ys.max()

def enhance_subject_pop_seg(img, mask, blur_ksize=21):
    if mask is None:
        return img

    mask = cv2.resize(mask.astype(np.float32), (img.shape[1], img.shape[0]))
    mask = np.clip(mask, 0, 1)

    blurred = cv2.GaussianBlur(img, (blur_ksize, blur_ksize), 0)
    out = img.astype(np.float32) * mask[..., None] + blurred.astype(np.float32) * (1 - mask[..., None])
    return np.clip(out, 0, 255).astype(np.uint8)

# -----------------------
# استخراج ویژگی‌ها
# -----------------------
def extract_feats_for_enhance(img, mask=None, subject_cls="none"):
    feats = {
        "sharpness": sharpness_laplacian(img),
        "exposure": exposure_quality(img),
        "contrast": rms_contrast(img),
        "colorfulness": colorfulness(img),
        "saturation": saturation_mean(img),
        "subject_cls": subject_cls,
        "subject": 0.0  # ← اضافه شد: ترکیب وضوح + موقعیت
    }

    if mask is not None:
        box = mask_to_box(mask)
        if box:
            feats["subject"] = subject_total_score(img, box) 

    return feats


# -----------------------
# تابع اصلی ادیت
# -----------------------

# اگر کیفیت خوب باشد، تغییری اعمال نمی‌شود
 
def enhance_image(img, max_iter=2, blur_ksize=15, enable_subject_pop=True):
    enhanced = img.copy()
    mask, subject_cls = get_best_mask(enhanced)

    for _ in range(max_iter):
        feats = extract_feats_for_enhance(enhanced, mask, subject_cls)

        # Sharpness (کنترل‌شده)
        if feats["sharpness"] < 45:
            enhanced = increase_sharpness_soft(enhanced, alpha=0.4)

        # Exposure کنترل‌شده
        if feats["exposure"] < 40:
            enhanced = adjust_contrast_exposure_soft(enhanced, contrast=1.05, brightness=8)
        elif feats["exposure"] > 80:
            enhanced = adjust_contrast_exposure_soft(enhanced, contrast=0.97, brightness=-6)

        # Saturation
        if feats["saturation"] < 45:
            enhanced = adjust_saturation_soft(enhanced, factor=1.08)

        # Subject pop (اختیاری)
        if enable_subject_pop and mask is not None:
            enhanced = enhance_subject_pop_seg(enhanced, mask, blur_ksize)

    return enhanced
