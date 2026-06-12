"""
این فایل ابتدا خروجی خام ویژگی‌ها رو نرمالایز می‌کنه 
سپس با وزن‌دهی، یک نمره نهایی «جذابیت اینستاگرامی» تولید می‌کنه

"""
import math

# -----------------------------
# Sigmoid normalization (0..100)
# -----------------------------
def normalize_sigmoid(x, mid, scale):
    return int(100 * (1.0 / (1.0 + math.exp(-(x - mid) / scale))))

def normalize(x, key):
    """
    Normalize raw feature values into 0..100 range
    """
    if key == "sharpness":
         # لگاریتم برای کاهش اثر مقادیر بزرگ
         # وضوح خیلی بالا نباید امتیاز را غیرواقعی کند
        x_log = math.log(1 + x)
        return normalize_sigmoid(x_log, mid=6, scale=1.5)
    
    if key == "exposure":
        # exposure_quality is already 0..1 (higher is better)
        return int(max(0, min(1, x)) * 100)

    if key == "contrast":
        return normalize_sigmoid(x, mid=55, scale=15)

    if key == "colorfulness":
        return normalize_sigmoid(x, mid=40, scale=12)

    if key == "saturation":
        return normalize_sigmoid(x * 100, mid=35, scale=12)


    if key == "subject":
        # چون امتیاز برجستگی سوژه معمولاً کوچک است، تقویت می‌شود 
        # تا نقش واقعی خود را در امتیاز نهایی داشته باشد
        return normalize_sigmoid(x * 100, mid=40, scale=15)


    if key == "shakeless":
        # x is already stability score (higher is better)
        return normalize_sigmoid(x * 100, mid=60, scale=15)

    if key == "jumpcut":
        return normalize_sigmoid(x * 100, mid=50, scale=15)

    # fallback
    return int(max(0, min(100, x)))

# -----------------------------
# Final weighted score
# -----------------------------
WEIGHTS_IMAGE = {
    "sharpness": 0.22,
    "exposure": 0.16,
    "contrast": 0.10,
    "colorfulness": 0.14,
    "saturation": 0.08,
    "subject": 0.30,   # جمع وزن قبلی subject + subject_pop
}

WEIGHTS_VIDEO = {
    "sharpness": 0.1714,
    "exposure": 0.1333,
    "contrast": 0.0762,
    "colorfulness": 0.1143,
    "saturation": 0.0571,
    "subject": 0.2572,   # جمع قبلی subject + subject_pop
    "shakeless": 0.0952,
    "jumpcut": 0.0952,
}

def final_score(norm_feats, subject_cls=None, is_video=False):
    """
    ضرب هر ویژگی نرمال‌شده در وزن آن
     و جمع همه برای نمره نهایی 100
     
    """
    weights = WEIGHTS_VIDEO if is_video else WEIGHTS_IMAGE

    score = 0.0
    breakdown = {}

    for k, w in weights.items():
        if k in norm_feats:
            part = norm_feats[k] * w
            breakdown[k] = int(norm_feats[k])
            score += part

    final_s = int(round(score))

    return final_s, breakdown
