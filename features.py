"""
هدف این فایل استخراج ویژگی‌های بصری  
 از تصویر و ویدئو هست
 
"""
import cv2
import numpy as np

# ---------- Image Features ----------

#سنجش وضوح
#اگر عکس تار باشد، معمولاً جذابیت بصری آن کم می‌شود

def sharpness_laplacian(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

# کیفیت نوردهی
# عکس‌هایی که خیلی تیره یا خیلی روشن هستند معمولاً جذاب نیستند

def exposure_quality(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    total = gray.size
    under = np.sum(gray < 30) / total
    over = np.sum(gray > 225) / total
    return float(1.0 - (under + over))  # هرچی بزرگ‌تر، بهتر

# کنتراست 
# عکس‌های با کنتراست پایین تخت و بی‌روح دیده می‌شوند

def rms_contrast(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(np.std(gray))

# رنگ‌ پذیری
# عکس‌های رنگی‌تر معمولاً جذاب‌ترند (به‌خصوص در اینستاگرام)

def colorfulness(img):
    (B, G, R) = cv2.split(img.astype("float"))
    rg = np.abs(R - G)
    yb = np.abs(0.5 * (R + G) - B)
    std_rg, std_yb = np.std(rg), np.std(yb)
    mean_rg, mean_yb = np.mean(rg), np.mean(yb)
    return float(np.sqrt(std_rg**2 + std_yb**2) + 0.3 * np.sqrt(mean_rg**2 + mean_yb**2))

# اشباع رنگ ناحیه‌های روشن
#سچوریشن فقط در بخش‌های قابل دید اهمیت دارد (نه سایه‌های تیره)

def saturation_mean(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1].astype(np.float32) / 255.0
    v = hsv[:, :, 2].astype(np.float32) / 255.0

    mask = v > 0.2
    if np.sum(mask) == 0:
        return 0.0

    return float(np.percentile(s[mask], 75))

# برجستگی سوژه
# اگر سوژه نسبت به کل تصویر واضح‌تر باشد، توجه بیننده را بهتر جلب می‌کند

def subject_pop(img, box):
    x1,y1,x2,y2 = map(int, box)
    roi = img[y1:y2, x1:x2]
    s_roi = sharpness_laplacian(roi)
    s_all = sharpness_laplacian(img) + 1e-6
    return s_roi / s_all

# جایگاه سوژه 
# قانون یک‌سوم (Rule of Thirds)

def subject_position_score(box, img_shape):
    x1, y1, x2, y2 = box
    cx = (x1 + x2)/2
    cy = (y1 + y2)/2
    h, w = img_shape[:2]

    # مختصات نرمال (0..1)
    nx = cx / w
    ny = cy / h

    # فاصله تا خطوط thirds
    thirds_x = np.array([1/3, 2/3])
    thirds_y = np.array([1/3, 2/3])

    dx = np.min(np.abs(nx - thirds_x))
    dy = np.min(np.abs(ny - thirds_y))

    # فاصله نرمالیزه شده
    dist = np.sqrt(dx**2 + dy**2)
    score = 1.0 - dist  # هرچی نزدیک خطوط یک سوم باشه، بهتر
    return float(np.clip(score, 0, 1))

def subject_total_score(img, box):
    """
    ترکیب وضوح سوژه و موقعیت سوژه برای امتیاز کلی 
    """
    pop_score = subject_pop(img, box) 
    pos_score = subject_position_score(box, img.shape)
    
    # ترکیب با وزن دلخواه
    return float(np.clip(0.35 * pop_score + 0.15 * pos_score, 0, 1))



# ---------- Video Features (CPU-friendly) ----------

# امتیاز لرزش ویدئو
# حرکت نرم دوربین بد نیست، لرزش‌های ریز و ناگهانی بد هستند

def video_shake_score(frames, max_corners=150):
    if len(frames) < 3:
        return 1.0

    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=max_corners,
                                       qualityLevel=0.01, minDistance=7)

    if prev_pts is None:
        return 0.5

    jitters = []
    mean_motions = []

    for f in frames[1:]:
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None)

        if next_pts is None:
            continue

        good_prev = prev_pts[status == 1]
        good_next = next_pts[status == 1]
        if len(good_prev) < 10:
            continue

        flow = good_next - good_prev
        mean_motion = np.mean(flow, axis=0)
        residual = flow - mean_motion

        jitter = np.linalg.norm(residual, axis=1).mean()
        jitters.append(jitter)
        mean_motions.append(np.linalg.norm(mean_motion))

        prev_gray = gray
        prev_pts = good_next.reshape(-1, 1, 2)

    if len(jitters) < 3:
        return 0.5

    # حساس به لرزش ناگهانی
    jitter_std = np.std(jitters)
    jitter_mean = np.mean(jitters)

    # ترکیب شدت لرزش + نوسان
    raw_shake = 0.6 * jitter_mean + 0.4 * jitter_std
    raw_norm = np.clip((raw_shake - 40.0) / 70.0, 0, 1)
    score = 1.0 - raw_norm
    return float(score)


# تشخیص کات‌های ادیت

def jump_cut_score(frames):
    """
    Detects editing jump cuts (intentional cuts) as a positive stylistic feature.

    """
    if len(frames) < 3:
        return 0.0

    diffs = []
    prev = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)

    for f in frames[1:]:
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(prev, gray)
        diffs.append(np.mean(diff))
        prev = gray

    diffs = np.array(diffs)

    # Normalize diffs
    mean_d = np.mean(diffs)
    std_d = np.std(diffs) + 1e-6

    # Spikes: large sudden changes (jump cuts)
    spikes = diffs > (mean_d + 2.0 * std_d)

    spike_ratio = np.sum(spikes) / len(diffs)

    # Heuristic: some jump cuts are good, too many is chaotic
    # sweet spot around 5%..20%
    score = np.exp(-((spike_ratio - 0.15) ** 2) / (2 * 0.10 ** 2))
    return float(np.clip(score, 0, 1))

