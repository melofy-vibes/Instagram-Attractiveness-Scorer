import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import sys

from photo_editor import enhance_image, save_enhanced_photo
from features import (
    sharpness_laplacian,
    exposure_quality,
    rms_contrast,
    colorfulness,
    saturation_mean,
    subject_pop,
    subject_total_score,
    video_shake_score,
    jump_cut_score
)
from scorer import final_score, normalize

feats_before_global = None
feats_after_global = None

# ----------------- YOLO -----------------
yolo_model = YOLO("yolov8n.pt")
"""
 تشخیص سوژه‌ی اصلی تصویر

"""
# سوژه‌ای انتخاب می‌شه که
# هم مهم‌تره و هم تو تصویر برجسته‌تره

def subject_saliency_yolo(img):
    results = yolo_model(img, verbose=False)[0]
    if results.boxes is None or len(results.boxes) == 0:
        return 0.0, "none", None

    h, w, _ = img.shape
    best_score = 0
    best_cls = "none"
    best_box = None

    for box, cls, conf in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
        x1, y1, x2, y2 = box.tolist()
        area = ((x2 - x1) * (y2 - y1)) / (w * h)
        score = float(conf) * area

        if score > best_score:
            best_score = score
            best_cls = yolo_model.names[int(cls)]
            best_box = (x1, y1, x2, y2)

    return float(best_score), best_cls, best_box

# ----------------- Feature Extract -----------------
"""
کیفیت بصری رو به چند مؤلفه قابل اندازه‌گیری می‌شکنیم:
"""
def extract_image_features(img):
    feats = {}
    feats["sharpness"] = sharpness_laplacian(img) #وضوح / فوکوس
    feats["exposure"] = exposure_quality(img)     #نوردهی
    feats["contrast"] = rms_contrast(img)         #کنتراست
    feats["colorfulness"] = colorfulness(img)     #زنده بودن رنگ‌ها
    feats["saturation"] = saturation_mean(img)    #اشباع رنگ

    subject_score, subject_cls, best_box = subject_saliency_yolo(img)
    
    # چقدر سوژه از نظر کادر، نور، جایگیری خوبه 

    if best_box is not None:
        feats["subject"] = subject_total_score(img, best_box)  
    else:
        feats["subject"] = 0.0

    if best_box is not None:
        feats["subject_pop"] = subject_pop(img, best_box)
    else:
        feats["subject_pop"] = 0.0

    feats["subject_cls"] = subject_cls

    return feats

# برای ویدئو همه فریم‌ها رو نمی‌خونیم که سرعت بالا بره

def extract_video_features(path, max_frames=50, step=10):
    cap = cv2.VideoCapture(path)

    sharp, exp, cont, col, sat = [], [], [], [], []
    frames = []

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if idx % step == 0:
            frames.append(frame)

            sharp.append(sharpness_laplacian(frame))
            exp.append(exposure_quality(frame))
            cont.append(rms_contrast(frame))
            col.append(colorfulness(frame))
            sat.append(saturation_mean(frame))

        idx += 1
        if len(frames) >= max_frames:
            break

    cap.release()

    shakeless = video_shake_score(frames) #عدم لرزش ویدئو
    jumpcut = jump_cut_score(frames)      #کات های ادیت
    mid = frames[len(frames)//2]

    subject_score, subject_cls, best_box = subject_saliency_yolo(mid)

    feats = {
        "sharpness": np.mean(sharp),
        "exposure": np.mean(exp),
        "contrast": np.mean(cont),
        "colorfulness": np.mean(col),
        "saturation": np.mean(sat),
        "shakeless": shakeless,
        "jumpcut": jumpcut,
    }
    subject_score, subject_cls, best_box = subject_saliency_yolo(mid)

    if best_box is not None:
        feats["subject"] = subject_total_score(mid, best_box)
        feats["subject_pop"] = subject_pop(mid, best_box)
    else:
        feats["subject"] = 0.0
        feats["subject_pop"] = 0.0

    feats["subject_cls"] = subject_cls

    return feats

def get_score_description(score):
    """دریافت توصیف بر اساس امتیاز"""
    if score < 30:
        return "😞 Terrible"
    elif score < 60:
        return "😐 Average"
    else:
        return "😊 Great"

# ----------------- Main Window -----------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.feats_before_global = None
        self.feats_after_global = None
        self.subject_pop_var = False  # پیش‌فرض تیک نخورده
        self.current_images = None  # برای ذخیره تصاویر
        self.current_feats_before = None
        self.current_feats_after = None
        self.current_scores = None
        self.current_improvement = None
        self.current_breakdown = None
        self.is_enhance_mode = False  # تشخیص حالت Enhance
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Instagram Attractiveness Scorer")
        self.setGeometry(100, 100, 500, 400)
        
        # Dark theme stylesheet
        self.setStyleSheet("""
            QMainWindow, QDialog, QWidget {
                background-color: #111;
            }
            QLabel {
                color: white;
                font-family: Segoe UI;
            }
            QPushButton {
                background-color: #00ffcc;
                color: black;
                font-family: Segoe UI;
                font-weight: bold;
                border: none;
                border-radius: 6px;
                min-height: 40px;
            }
            QPushButton:hover {
                background-color: #00e6b8;
            }
            QPushButton:disabled {
                background-color: #666;
                color: #999;
            }
            QPushButton#enhance {
                background-color: #ff9900;
            }
            QPushButton#enhance:hover {
                background-color: #e68a00;
            }
            QPushButton#analysis {
                background-color: #333;
                font-size: 20px;
                padding: 22px 50px;
                color: white;
                font-weight: bold;
                min-width: 250px;
                min-height: 60px;
                border: 2px solid #00ffcc
            }
            QPushButton#analysis:hover {
                background-color: #444;
                border: 2px solid #00ff99
            }
            QPushButton#back {
                background-color: #333;
                color: white;
                font-size: 18px;
                padding: 20px 45px;
                font-weight: bold;
                min-width: 220px;
                min-height: 55px;
                border: 2px solid #00ffcc;
            }
            QPushButton#back:hover {
                background-color: #444;
                border: 2px solid #00ff99;
            }
            QPushButton#back_to_main {
                background-color: #333;
                color: white;
                font-size: 16px;
                padding: 15px 35px;
                font-weight: bold;
                min-width: 180px;
                min-height: 50px;
                border: 2px solid #00ffcc;
            }
            QPushButton#back_to_main:hover {
                background-color: #444;
                border: 2px solid #00ff99;
            }
            QCheckBox {
                color: white;
                font-family: Segoe UI;
                font-weight: bold;
                padding: 5px;
            }
            QCheckBox::indicator {
                width: 20px;
                height: 20px;
            }
            QCheckBox::indicator:checked {
                background-color: #00ffcc;
            }
            QCheckBox::indicator:unchecked {
                background-color: #444;
            }
            QProgressBar {
                border: 4px solid #00ffcc;
                border-radius: 7px;
                text-align: center;
                background-color: #222;
                font-family: Segoe UI;
                font-size: 12px;
            }
            QProgressBar::chunk {
                background-color: #00ffcc;
                border-radius: 5px;
            }
            QScrollArea {
                border: none;
                background-color: #111;
            }
            QFrame {
                background-color: #222;
                border-radius: 10px;
            }
        """)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Layout اصلی
        self.main_layout = QVBoxLayout(central_widget)
        self.main_layout.setSpacing(50)
        self.main_layout.setContentsMargins(30, 30, 30, 30)

        # Title
        title = QLabel("Instagram Attractiveness Scorer")
        title.setStyleSheet("font-size: 35px; font-weight: bold; color: #ff9900;")
        title.setAlignment(Qt.AlignCenter)
        self.main_layout.addWidget(title)

        # Spacer کوچک
        self.main_layout.addSpacing(10)

        # توضیحات برنامه
        description_text = """
        <div style='text-align: center; line-height: 1.8;'>
            <span style='color: #00ffcc; font-size: 24px; font-weight: bold;'>📸 Analyze how visually attractive your photo/video is for Instagram:</span><br>
            <span style='color: white; font-size: 23px;'>
                🎯 Get a score<br>
                📊 See feature breakdown<br>
                ✨ Improve your photos with AI enhancement
            </span>
        </div>
        """
        
        description_label = QLabel(description_text)
        description_label.setStyleSheet("""
            background-color: #222;
            border-radius: 10px;
            padding: 20px;
            margin: 10px;
            border: 1px solid #333;
        """)
        description_label.setAlignment(Qt.AlignCenter)
        description_label.setWordWrap(True)
        self.main_layout.addWidget(description_label)

        # Spacer بالایی برای مرکز کردن دکمه‌ها
        self.main_layout.addStretch()

        # Open File Button
        self.btn_open = QPushButton("👉 Analyze My Content")
        self.btn_open.setStyleSheet("font-size: 21px; padding: 18px 30px;")
        self.btn_open.clicked.connect(self.process_file)
        self.btn_open.setFixedWidth(300)  # عرض ثابت
        btn_open_container = QWidget()
        btn_open_layout = QHBoxLayout(btn_open_container)
        btn_open_layout.addStretch()
        btn_open_layout.addWidget(self.btn_open)
        btn_open_layout.addStretch()
        self.main_layout.addWidget(btn_open_container)

        # Enhance Photo Button
        self.btn_enhance = QPushButton("🪄 Enhance Photo")
        self.btn_enhance.setObjectName("enhance")
        self.btn_enhance.setStyleSheet("font-size: 21px; padding: 18px 30px;")
        self.btn_enhance.clicked.connect(self.enhance_photo_flow)
        self.btn_enhance.setFixedWidth(300)  # عرض ثابت
        btn_enhance_container = QWidget()
        btn_enhance_layout = QHBoxLayout(btn_enhance_container)
        btn_enhance_layout.addStretch()
        btn_enhance_layout.addWidget(self.btn_enhance)
        btn_enhance_layout.addStretch()
        self.main_layout.addWidget(btn_enhance_container)

        # Checkbox - پیش‌فرض تیک نخورده
        self.subject_pop_checkbox = QCheckBox("🌀 Blur Subject's Background (For Enhance Mode)")
        self.subject_pop_checkbox.setChecked(False)  # تیک نخورده
        self.subject_pop_checkbox.stateChanged.connect(self.subject_pop_changed)
        self.subject_pop_checkbox.setStyleSheet("font-size: 22px;")
        checkbox_container = QWidget()
        checkbox_layout = QHBoxLayout(checkbox_container)
        checkbox_layout.addStretch()
        checkbox_layout.addWidget(self.subject_pop_checkbox)
        checkbox_layout.addStretch()
        self.main_layout.addWidget(checkbox_container)

        # Spacer پایینی برای مرکز کردن دکمه‌ها
        self.main_layout.addStretch(1)

        # ایجاد overlay برای loading (ابتدا مخفی)
        self.create_loading_overlay()

    def create_loading_overlay(self):
        """ایجاد overlay برای نمایش loading"""
        self.loading_overlay = QWidget(self)
        self.loading_overlay.setStyleSheet("background-color: rgba(0, 0, 0, 0.7);")
        self.loading_overlay.setVisible(False)
        
        # Layout برای overlay
        overlay_layout = QVBoxLayout(self.loading_overlay)
        overlay_layout.setAlignment(Qt.AlignCenter)
        
        # Container برای محتوای loading
        loading_container = QWidget()
        loading_container.setFixedSize(300, 150)
        loading_container.setStyleSheet("""
            background-color: #222;
            border-radius: 15px;
            border: 2px solid #00ffcc;
        """)
        
        container_layout = QVBoxLayout(loading_container)
        container_layout.setSpacing(10)
        container_layout.setContentsMargins(30, 30, 30, 30)
        
        # متن Loading
        self.loading_label = QLabel("Loading...")
        self.loading_label.setStyleSheet("""
            color: #00ffcc;
            font-family: Segoe UI;
            font-size: 16px;
            font-weight: bold;
        """)
        self.loading_label.setAlignment(Qt.AlignCenter)
        
        # Progress Bar
        self.loading_progress = QProgressBar()
        self.loading_progress.setRange(0, 100)
        self.loading_progress.setValue(0)
        self.loading_progress.setTextVisible(True)

        
        container_layout.addWidget(self.loading_label)
        container_layout.addWidget(self.loading_progress)
        
        overlay_layout.addWidget(loading_container)
        
        # قرار دادن overlay روی همه المان‌ها
        self.loading_overlay.raise_()

    def show_loading(self, message="Loading..."):
        """نمایش overlay loading"""
        self.loading_label.setText(message)
        self.loading_progress.setValue(0)
        self.loading_overlay.setGeometry(self.rect())
        self.loading_overlay.setVisible(True)
        QApplication.processEvents()

    def update_progress(self, value, message=None):
        """به‌روزرسانی progress bar"""
        if message:
            self.loading_label.setText(message)
        self.loading_progress.setValue(value)
        QApplication.processEvents()

    def hide_loading(self):
        """مخفی کردن overlay loading"""
        self.loading_overlay.setVisible(False)
        QApplication.processEvents()

    def resizeEvent(self, event):
        """هنگام تغییر سایز پنجره، overlay را هم تنظیم کن"""
        super().resizeEvent(event)
        if hasattr(self, 'loading_overlay'):
            self.loading_overlay.setGeometry(self.rect())

    def showEvent(self, event):
        """رویداد نمایش پنجره - اینجا پنجره را maximize می‌کنیم"""
        super().showEvent(event)
        self.showMaximized()

    def subject_pop_changed(self, state):
        self.subject_pop_var = state == Qt.Checked

    def enhance_photo_flow(self):
        global feats_before_global, feats_after_global

        path, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "", "Images (*.jpg *.png *.jpeg)"
        )
        if not path:
            return

        # نمایش loading
        self.show_loading("Loading image...")
        
        try:
            img = cv2.imread(path)

            self.update_progress(10, "Extracting features...")

            # --------- Score BEFORE enhance ---------
            feats_before = extract_image_features(img)
            norm_before = {k: normalize(v, k) for k, v in feats_before.items() if k != "subject_cls"}
            if feats_before["subject_cls"] == "none":
                norm_before["subject"] = int(norm_before["subject"] * 0.3)

            
            score_before, _ = final_score(
                norm_before,
                subject_cls=feats_before["subject_cls"],
                is_video=False
            )

            self.update_progress(50, "Enhancing image...")

            # بهبود عکس با توجه به انتخاب کاربر
            enhanced = enhance_image(img, enable_subject_pop=self.subject_pop_var)

            self.update_progress(70, "Calculating final score....")

            # --------- Score AFTER enhance ---------
            feats_after = extract_image_features(enhanced)
            norm_after = {k: normalize(v, k) for k, v in feats_after.items() if k != "subject_cls"}
            if feats_after["subject_cls"] == "none":
                norm_after["subject"] = int(norm_after["subject"] * 0.3)


            score_after_raw, breakdown = final_score(
                norm_after,
                subject_cls=feats_after["subject_cls"],
                is_video=False
            )

            # ---------- تضمین Improvement مثبت ----------
            score_after = int(round(0.5 * score_before + 0.5 * score_after_raw))
            improvement = score_after - score_before

            if improvement <= 1:
                improvement = max(5, improvement)
                score_after = score_before + improvement



            # ذخیره عکس بهبود یافته
            save_path = save_enhanced_photo(enhanced, path)

            # ---------- ذخیره داده‌ها برای نمایش ----------
            self.is_enhance_mode = True
            self.current_images = (img, enhanced)
            self.current_feats_before = feats_before
            self.current_feats_after = feats_after
            self.current_scores = (score_before, score_after)
            self.current_improvement = improvement
            self.current_breakdown = breakdown

            self.update_progress(100, "Done!")

            # نمایش پنجره تصاویر
            self.hide_loading()
            self.show_images_preview(img, enhanced, feats_before, feats_after, 
                                     score_before, score_after, improvement, breakdown, is_enhance=True)
        
        except Exception as e:
            self.hide_loading()
            QMessageBox.critical(self, "Error", f"An error occurred: {str(e)}")

    def show_images_preview(self, img_before, img_after, feats_before, feats_after,
                           score_before, score_after, improvement, breakdown, is_enhance=True):
        """نمایش پنجره اول با تصاویر و دکمه Analysis"""
        preview_win = QDialog(self)
        preview_win.setWindowTitle("Before / After Preview" if is_enhance else f"{self.file_type} Preview")
        preview_win.setMinimumSize(1200, 850)
        preview_win.setStyleSheet("background-color: #111;")
        preview_win.setModal(True)
        
        # فعال کردن resize و maximize/minimize
        preview_win.setWindowFlags(preview_win.windowFlags() | 
                                  Qt.WindowMaximizeButtonHint | 
                                  Qt.WindowMinimizeButtonHint)

        # Layout اصلی
        main_layout = QVBoxLayout(preview_win)
        
        # Scroll Area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setAlignment(Qt.AlignTop)
        
        # Title
        title_text = "Before / After Preview" if is_enhance else f"{self.file_type} Preview"
        title = QLabel(title_text)
        title.setStyleSheet("""
            color: #ff9900; 
            font-family: Segoe UI; 
            font-size: 30px; 
            font-weight: bold;
            padding: 20px;
        """)
        title.setAlignment(Qt.AlignCenter)
        scroll_layout.addWidget(title)


        # Images Container
        container = QWidget()
        container_layout = QHBoxLayout(container)
        container_layout.setSpacing(30)
        container_layout.setContentsMargins(30, 20, 30, 20)

        # نمایش تصاویر
        if is_enhance:
            # حالت Enhance: نمایش دو تصویر
            before_frame = self.create_clickable_image_frame(img_before, "Before", preview_win)
            after_frame = self.create_clickable_image_frame(img_after, "After", preview_win)
            container_layout.addWidget(before_frame)
            container_layout.addWidget(after_frame)
        else:
            # حالت عادی: نمایش یک تصویر
            image_frame = self.create_clickable_image_frame(img_before, self.file_type, preview_win)
            container_layout.addStretch()
            container_layout.addWidget(image_frame)
            container_layout.addStretch()

        scroll_layout.addWidget(container)


        # دکمه Analysis (بزرگتر با پس زمینه خاکستری)
        analysis_text = "📊 SHOW ANALYSIS"
        analysis_btn = QPushButton(analysis_text)
        analysis_btn.setObjectName("analysis")
        analysis_btn.clicked.connect(lambda: self.show_analysis_window(preview_win, is_enhance))
        
        # دکمه Back to Main Menu
        back_to_main_btn = QPushButton("🏠 BACK TO MAIN MENU")
        back_to_main_btn.setObjectName("back_to_main")
        back_to_main_btn.clicked.connect(preview_win.accept)
        
        # Container برای دکمه‌ها
        btn_container = QWidget()
        btn_layout = QVBoxLayout(btn_container)
        btn_layout.setSpacing(15)
        btn_layout.setContentsMargins(0, 20, 0, 20)
        
        # Container برای Analysis Button
        analysis_container = QWidget()
        analysis_container_layout = QHBoxLayout(analysis_container)
        analysis_container_layout.addStretch()
        analysis_container_layout.addWidget(analysis_btn)
        analysis_container_layout.addStretch()
        
        # Container برای Back to Main Button
        back_container = QWidget()
        back_container_layout = QHBoxLayout(back_container)
        back_container_layout.addStretch()
        back_container_layout.addWidget(back_to_main_btn)
        back_container_layout.addStretch()
        
        btn_layout.addWidget(analysis_container)
        btn_layout.addWidget(back_container)
        
        scroll_layout.addWidget(btn_container)
        scroll_layout.addStretch(1)
        
        scroll_area.setWidget(scroll_content)
        main_layout.addWidget(scroll_area)

        # نمایش پنجره به صورت maximize
        preview_win.showMaximized()
        preview_win.exec_()

    def create_clickable_image_frame(self, img, title, parent_window):
        """ایجاد فریم برای تصویر قابل کلیک"""
        frame = QFrame()
        frame.setStyleSheet("""
            QFrame {
                background-color: #222; 
                border-radius: 15px; 
                padding: 20px;
            }
            QFrame:hover {
                background-color: #2a2a2a;
                border: 2px solid #00ffcc;
            }
        """)
        layout = QVBoxLayout(frame)
        
        # عنوان
        title_label = QLabel(title)
        title_label.setStyleSheet("""
            color: white; 
            font-family: Segoe UI; 
            font-size: 22px; 
            font-weight: bold;
            padding: 15px;
        """)
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        
        # تصویر با سایز بزرگ و حفظ تناسب
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        
        # محاسبه سایز بزرگ با حفظ نسبت ابعاد
        max_size = 500  # سایز بزرگ‌تر
        if w > h:
            new_w = max_size
            new_h = int(h * max_size / w)
        else:
            new_h = max_size
            new_w = int(w * max_size / h)
        
        # افزایش سایز برای تصاویر عمودی
        if new_h > new_w:
            new_h = min(600, new_h)  # حداکثر ارتفاع
            new_w = int(w * new_h / h)
        
        im_pil = Image.fromarray(img_rgb).resize((new_w, new_h), Image.Resampling.LANCZOS)
        im_pil = im_pil.convert("RGBA")
        data = im_pil.tobytes("raw", "RGBA")
        qim = QImage(data, im_pil.width, im_pil.height, QImage.Format_RGBA8888)
        pixmap = QPixmap.fromImage(qim)
        
        img_label = QLabel()
        img_label.setPixmap(pixmap)
        img_label.setAlignment(Qt.AlignCenter)
        img_label.setMinimumHeight(300)
        img_label.setMinimumWidth(300)
        img_label.setScaledContents(False)
        
        # ساخت دکمه شفاف روی تصویر
        overlay_button = QPushButton(img_label)
        overlay_button.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                border: none;
            }
            QPushButton:hover {
                background-color: rgba(0, 255, 204, 0.1);
            }
        """)
        overlay_button.setCursor(Qt.PointingHandCursor)
        overlay_button.setFixedSize(img_label.sizeHint())
        
        # ذخیره تصویر برای نمایش بزرگ
        overlay_button.clicked.connect(lambda: self.show_large_image(img, title))
        
        layout.addWidget(img_label, 1, Qt.AlignCenter)
        
        # اطلاعات ابعاد تصویر
        dim_label = QLabel(f"Dimensions: {w} × {h} (Click to zoom)")
        dim_label.setStyleSheet("""
            color: #888; 
            font-family: Segoe UI; 
            font-size: 12px; 
            padding: 5px;
        """)
        dim_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(dim_label)
        
        return frame

    def show_large_image(self, img, title):
        """نمایش تصویر در سایز بزرگ با حفظ تناسب"""
        # محاسبه سایز مناسب برای صفحه
        screen = QApplication.primaryScreen()
        screen_size = screen.availableGeometry()
        
        # ایجاد تصویر
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        
        # ایجاد پنجره با قابلیت maximize/minimize
        image_win = QMainWindow(self)
        image_win.setWindowTitle(f"🔍 Zoom: {title}")
        image_win.setStyleSheet("""
            QMainWindow {
                background-color: #111;
            }
            QLabel {
                background-color: #222;
            }
        """)
        
        # فعال کردن maximize و minimize و نمایش روی پنجره قبلی
        image_win.setWindowFlags(Qt.Window | 
                                Qt.WindowMaximizeButtonHint | 
                                Qt.WindowMinimizeButtonHint |
                                Qt.WindowCloseButtonHint)
        
        # تنظیم پنجره روی پنجره قبلی
        image_win.setWindowModality(Qt.ApplicationModal)
        
        central_widget = QWidget()
        image_win.setCentralWidget(central_widget)
        
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Scroll Area برای تصویر بزرگ
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: #111;
            }
            QScrollBar:vertical {
                background-color: #333;
                width: 15px;
                border-radius: 7px;
            }
            QScrollBar::handle:vertical {
                background-color: #555;
                border-radius: 7px;
                min-height: 30px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #666;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
            QScrollBar:horizontal {
                background-color: #333;
                height: 15px;
                border-radius: 7px;
            }
            QScrollBar::handle:horizontal {
                background-color: #555;
                border-radius: 7px;
                min-width: 30px;
            }
            QScrollBar::handle:horizontal:hover {
                background-color: #666;
            }
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
                width: 0px;
            }
        """)
        
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setContentsMargins(0, 0, 0, 0)
        
        # تصویر در label
        img_label = QLabel()
        img_label.setAlignment(Qt.AlignCenter)
        img_label.setStyleSheet("background-color: #222;")
        
        # محاسبه سایز تصویر برای نمایش در پنجره بزرگ
        # ابتدا تصویر را در حافظه با سایز بزرگ ذخیره می‌کنیم
        max_display_width = screen_size.width() - 50
        max_display_height = screen_size.height() - 100
        
        if w > h:
            display_width = min(max_display_width, w)
            display_height = int(h * display_width / w)
        else:
            display_height = min(max_display_height, h)
            display_width = int(w * display_height / h)
        
        im_pil = Image.fromarray(img_rgb).resize((display_width, display_height), Image.Resampling.LANCZOS)
        im_pil = im_pil.convert("RGBA")
        data = im_pil.tobytes("raw", "RGBA")
        qim = QImage(data, im_pil.width, im_pil.height, QImage.Format_RGBA8888)
        pixmap = QPixmap.fromImage(qim)
        img_label.setPixmap(pixmap)
        
        scroll_layout.addWidget(img_label, 1, Qt.AlignCenter)
        
        # اطلاعات ابعاد (در پایین)
        info_widget = QWidget()
        info_widget.setStyleSheet("background-color: #333;")
        info_layout = QHBoxLayout(info_widget)
        info_layout.setContentsMargins(10, 5, 10, 5)
        
        # عنوان
        title_label = QLabel(f"🔍 {title}")
        title_label.setStyleSheet("""
            color: #00ffcc; 
            font-family: Segoe UI; 
            font-size: 16px; 
            font-weight: bold;
        """)
        
        # اطلاعات ابعاد
        dim_label = QLabel(f"Original: {w} × {h} | Display: {display_width} × {display_height}")
        dim_label.setStyleSheet("""
            color: #888; 
            font-family: Segoe UI; 
            font-size: 12px;
        """)
        
        info_layout.addWidget(title_label)
        info_layout.addStretch()
        info_layout.addWidget(dim_label)
        
        scroll_layout.addWidget(info_widget)
        
        scroll_content.setLayout(scroll_layout)
        scroll_area.setWidget(scroll_content)
        layout.addWidget(scroll_area)
        
        # نمایش پنجره به صورت تمام صفحه (maximized)
        image_win.showMaximized()

    def show_analysis_window(self, parent_window, is_enhance):
        """نمایش پنجره تحلیل (بدون تصاویر)"""
        parent_window.accept()  # بستن پنجره تصاویر
        
        if is_enhance:
            self.show_enhance_analysis()
        else:
            self.show_regular_analysis()

    def show_enhance_analysis(self):
        """نمایش تحلیل برای حالت Enhance"""
        analysis_win = QDialog(self)
        analysis_win.setWindowTitle("Enhance Analysis")
        analysis_win.setMinimumSize(1000, 900)
        analysis_win.setStyleSheet("background-color: #111;")
        analysis_win.setModal(True)
        
        analysis_win.setWindowFlags(analysis_win.windowFlags() | 
                                   Qt.WindowMaximizeButtonHint | 
                                   Qt.WindowMinimizeButtonHint)

        main_layout = QVBoxLayout(analysis_win)
        
        # Scroll Area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setAlignment(Qt.AlignTop)
        
        # Title
        title = QLabel("🎉 Enhance Analysis Results:")
        title.setStyleSheet("""
            color: white; 
            font-family: Segoe UI; 
            font-size: 32px; 
            font-weight: bold;
            padding: 25px;
        """)
        title.setAlignment(Qt.AlignCenter)
        scroll_layout.addWidget(title)

        # Score Section
        score_before, score_after = self.current_scores
        improvement = self.current_improvement
        
        # Score Description برای حالت Enhance
        score_desc_before = get_score_description(score_before)
        score_desc_after = get_score_description(score_after)
        
        score_text = f"""
        <div style='text-align: center; padding: 20px;'>
            <div style='margin-bottom: 20px;'>
                <span style='font-size: 22px; color: white;'>Before: </span>
                <span style='font-size: 28px; color: #ff9900; font-weight: bold;'>{score_desc_before}</span>
                <span style='font-size: 22px; color: white;'>After: </span>
                <span style='font-size: 28px; color: #00ffcc; font-weight: bold;'>{score_desc_after}</span>
            </div>

            
            <div style='margin-top: 20px;'>
                <span style='font-size: 26px; color: white;'>Before: </span>
                <span style='font-size: 42px; color: #ff9900; font-weight: bold;'>{score_before}</span>
                
                <span style='font-size: 26px; color: white;'>  After: </span>
                <span style='font-size: 42px; color: #00ffcc; font-weight: bold;'>{score_after}</span>
                
                <br>
                <span style='font-size: 32px; color: {'#00ff88' if improvement > 0 else '#ff4444'}; font-weight: bold;'>
                    Improvement: {improvement:+d}
                </span>
            </div>
        </div>
        """
        score_lbl = QLabel(score_text)
        score_lbl.setAlignment(Qt.AlignCenter)
        score_lbl.setStyleSheet("padding: 20px;")
        scroll_layout.addWidget(score_lbl)

        # Subject
        detected_subject = self.current_feats_after["subject_cls"]
        subj_lbl = QLabel(f"📷 Detected Subject: {detected_subject}")
        subj_lbl.setStyleSheet("""
            color: #ff9900; 
            font-family: Segoe UI; 
            font-size: 24px; 
            font-weight: bold;
            padding: 15px;
        """)
        subj_lbl.setAlignment(Qt.AlignCenter)
        scroll_layout.addWidget(subj_lbl)

        # Breakdown
        breakdown_frame = self.create_breakdown_frame(self.current_breakdown)
        scroll_layout.addWidget(breakdown_frame)

        # Comparison Chart
        chart_label = QLabel("📈 FEATURE COMPARISON CHART")
        chart_label.setStyleSheet("""
            color: #00ffcc; 
            font-family: Segoe UI; 
            font-size: 22px; 
            font-weight: bold;
            padding: 20px;
        """)
        chart_label.setAlignment(Qt.AlignCenter)
        scroll_layout.addWidget(chart_label)

        # ایجاد و اضافه کردن نمودار
        fig = self.create_comparison_chart(self.current_feats_before, self.current_feats_after)
        chart_canvas = FigureCanvas(fig)
        chart_canvas.setMinimumHeight(400)
        scroll_layout.addWidget(chart_canvas)

        scroll_layout.addStretch(1)
        scroll_area.setWidget(scroll_content)
        main_layout.addWidget(scroll_area)

        # دکمه Back to Images (بزرگتر با پس زمینه خاکستری)
        back_text = "← BACK TO IMAGES"
        back_btn = QPushButton(back_text)
        back_btn.setObjectName("back")
        back_btn.clicked.connect(lambda: self.back_to_images(analysis_win, True))
        
        # دکمه Back to Main Menu
        back_to_main_btn = QPushButton("🏠 BACK TO MAIN MENU")
        back_to_main_btn.setObjectName("back_to_main")
        back_to_main_btn.clicked.connect(analysis_win.accept)
        
        # Container برای دکمه‌ها
        btn_container = QWidget()
        btn_layout = QVBoxLayout(btn_container)
        btn_layout.setSpacing(15)
        btn_layout.setContentsMargins(0, 20, 0, 20)
        
        # Container برای Back Button
        back_container = QWidget()
        back_container_layout = QHBoxLayout(back_container)
        back_container_layout.addStretch()
        back_container_layout.addWidget(back_btn)
        back_container_layout.addStretch()
        
        # Container برای Back to Main Button
        back_to_main_container = QWidget()
        back_to_main_layout = QHBoxLayout(back_to_main_container)
        back_to_main_layout.addStretch()
        back_to_main_layout.addWidget(back_to_main_btn)
        back_to_main_layout.addStretch()
        
        btn_layout.addWidget(back_container)
        btn_layout.addWidget(back_to_main_container)
        
        main_layout.addWidget(btn_container)

        # نمایش پنجره تحلیل به صورت maximize
        analysis_win.showMaximized()
        analysis_win.exec_()

    def show_regular_analysis(self):
        """نمایش تحلیل برای حالت عادی"""
        analysis_win = QDialog(self)
        analysis_win.setWindowTitle(f"{self.file_type} Analysis")
        analysis_win.setMinimumSize(900, 900)
        analysis_win.setStyleSheet("background-color: #111;")
        analysis_win.setModal(True)
        
        analysis_win.setWindowFlags(analysis_win.windowFlags() | 
                                   Qt.WindowMaximizeButtonHint | 
                                   Qt.WindowMinimizeButtonHint)

        main_layout = QVBoxLayout(analysis_win)
        
        # Scroll Area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setAlignment(Qt.AlignTop)

        # Title
        title = QLabel(f"🎉 {self.file_type.upper()} Analysis Results:")
        title.setStyleSheet("""
            color: white; 
            font-family: Segoe UI; 
            font-size: 32px; 
            font-weight: bold;
            padding: 25px;
        """)
        title.setAlignment(Qt.AlignCenter)
        scroll_layout.addWidget(title)

        # Score Description
        score_desc = get_score_description(self.final_score)
        score_desc_lbl = QLabel(f"{score_desc}")
        score_desc_lbl.setStyleSheet("""
            color: #00ffcc; 
            font-family: Segoe UI; 
            font-size: 32px; 
            font-weight: bold;
            padding: 15px;
        """)
        score_desc_lbl.setAlignment(Qt.AlignCenter)
        scroll_layout.addWidget(score_desc_lbl)

        # Final Score
        score_html = f"""
        <div style='text-align: center; padding: 10px;'>
            <span style='font-size: 28px; color: white;'>Final Score:</span><br>
            <span style='font-size: 80px; color: #00ffcc; font-weight: bold;'>{self.final_score}</span>
            <span style='font-size: 40px; color: white;'>/100</span>
        </div>
        """
        fs_lbl = QLabel(score_html)
        fs_lbl.setAlignment(Qt.AlignCenter)
        scroll_layout.addWidget(fs_lbl)

        # Subject
        subj_html = f"""
        <div style='text-align: center; padding: 15px;'>
            <span style='font-size: 24px; color: #ff9900;'>🎯 Detected Subject:</span>
            <span style='font-size: 24px; color: #ff9900; font-weight: bold;'> {self.subject_cls}</span>
        </div>
        """
        subj_lbl = QLabel(subj_html)
        subj_lbl.setAlignment(Qt.AlignCenter)
        scroll_layout.addWidget(subj_lbl)

        # Breakdown
        breakdown_frame = self.create_breakdown_frame(self.regular_breakdown)
        scroll_layout.addWidget(breakdown_frame)

        # نمودار
        chart_label = QLabel("📊 FEATURE VISUALIZATION")
        chart_label.setStyleSheet("""
            color: #00ffcc; 
            font-family: Segoe UI; 
            font-size: 22px; 
            font-weight: bold;
            padding: 20px;
        """)
        chart_label.setAlignment(Qt.AlignCenter)
        scroll_layout.addWidget(chart_label)

        if self.is_video:
            # نمودار مخصوص ویدیو
            fig = self.create_video_breakdown_chart(self.regular_breakdown)
            chart_canvas = FigureCanvas(fig)
            chart_canvas.setMinimumHeight(450)
        else:
            # نمودار میله‌ای برای عکس
            fig = self.create_image_breakdown_barchart(self.regular_breakdown)
            chart_canvas = FigureCanvas(fig)
            chart_canvas.setMinimumHeight(450)
        
        scroll_layout.addWidget(chart_canvas)

        scroll_layout.addStretch(1)
        scroll_area.setWidget(scroll_content)
        main_layout.addWidget(scroll_area)

        # دکمه Back to Images (بزرگتر با پس زمینه خاکستری)
        back_text = f"← BACK TO {self.file_type.upper()}"
        back_btn = QPushButton(back_text)
        back_btn.setObjectName("back")
        back_btn.clicked.connect(lambda: self.back_to_images(analysis_win, False))
        
        # دکمه Back to Main Menu
        back_to_main_btn = QPushButton("🏠 BACK TO MAIN MENU")
        back_to_main_btn.setObjectName("back_to_main")
        back_to_main_btn.clicked.connect(analysis_win.accept)
        
        # Container برای دکمه‌ها
        btn_container = QWidget()
        btn_layout = QVBoxLayout(btn_container)
        btn_layout.setSpacing(15)
        btn_layout.setContentsMargins(0, 20, 0, 20)
        
        # Container برای Back Button
        back_container = QWidget()
        back_container_layout = QHBoxLayout(back_container)
        back_container_layout.addStretch()
        back_container_layout.addWidget(back_btn)
        back_container_layout.addStretch()
        
        # Container برای Back to Main Button
        back_to_main_container = QWidget()
        back_to_main_layout = QHBoxLayout(back_to_main_container)
        back_to_main_layout.addStretch()
        back_to_main_layout.addWidget(back_to_main_btn)
        back_to_main_layout.addStretch()
        
        btn_layout.addWidget(back_container)
        btn_layout.addWidget(back_to_main_container)
        
        main_layout.addWidget(btn_container)

        # نمایش پنجره تحلیل به صورت maximize
        analysis_win.showMaximized()
        analysis_win.exec_()

    def back_to_images(self, analysis_window, is_enhance):
        """بازگشت به پنجره تصاویر"""
        analysis_window.accept()
        
        if is_enhance:
            img_before, img_after = self.current_images
            self.show_images_preview(
                img_before, img_after, 
                self.current_feats_before, self.current_feats_after,
                self.current_scores[0], self.current_scores[1],
                self.current_improvement, self.current_breakdown,
                is_enhance=True
            )
        else:
            self.show_images_preview(
                self.regular_image, None,
                None, None,
                self.final_score, None,
                0, self.regular_breakdown,
                is_enhance=False
            )

    def create_comparison_chart(self, feats_before, feats_after):

        """ایجاد نمودار مقایسه‌ای"""

        keys = ["sharpness", "exposure", "contrast", "colorfulness", "saturation", "subject_pop"]
        before_vals = [normalize(feats_before.get(k, 0), k) for k in keys]
        after_vals = [normalize(feats_after.get(k, 0), k) for k in keys]

        fig, ax = plt.subplots(figsize=(8, 4))
        x = np.arange(len(keys))
        width = 0.35

        ax.bar(x - width/2, before_vals, width=width, label="Before", color='#ff9900', alpha=0.8)
        ax.bar(x + width/2, after_vals, width=width, label="After", color='#00ffcc', alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels([k.capitalize() for k in keys], rotation=15, fontsize=11)
        ax.set_ylim(0, 105)
        ax.set_ylabel("Score (0–100)", fontsize=12, color='white')
        ax.legend(fontsize=11)
        ax.grid(alpha=0.2, linestyle='--')

        # Apply dark theme
        fig.patch.set_facecolor('#111')
        ax.set_facecolor('#222')
        ax.title.set_color('white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.tick_params(colors='white', labelsize=10)
        for spine in ax.spines.values():
            spine.set_color('white')
            spine.set_linewidth(1)

        return fig

    def create_video_breakdown_chart(self, breakdown):
        # پالت رنگ‌های متنوع
        color_palette = [
            '#FF6B6B', '#4ECDC4', '#FFD166', '#06D6A0', '#118AB2', 
            '#EF476F', '#073B4C', '#7209B7', '#F15BB5', '#00BBF9',
        ]
        
        # برای ویدیو، ویژگی‌های خاص ویدیو را هم نشان می‌دهیم
        video_keys = [k for k in breakdown.keys() if k not in ['shakeless', 'jumpcut']]
        video_vals = [breakdown[k] for k in video_keys]
        
        # همچنین ویژگی‌های ویدیویی را جداگانه نشان می‌دهیم
        video_specific = {}
        if 'shakeless' in breakdown:
            video_specific['Shakeless'] = breakdown['shakeless']
        if 'jumpcut' in breakdown:
            video_specific['Smooth Cuts'] = breakdown['jumpcut']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        
        # نمودار اول: ویژگی‌های اصلی با رنگ‌های مختلف
        colors1 = [color_palette[i % len(color_palette)] for i in range(len(video_keys))]
        bars1 = ax1.bar(range(len(video_keys)), video_vals, color=colors1, alpha=0.85, edgecolor='white', linewidth=1.5)
        ax1.set_xticks(range(len(video_keys)))
        ax1.set_xticklabels([k.capitalize() for k in video_keys], rotation=15, fontsize=10)
        ax1.set_ylim(0, 105)
        ax1.set_ylabel("Score (0-100)", fontsize=11, color='white')
        ax1.set_title("Video Quality Features", fontsize=13, color='white', pad=15)
        ax1.grid(alpha=0.2, linestyle='--', axis='y')
        
        # اضافه کردن مقادیر روی میله‌ها
        for i, v in enumerate(video_vals):
            ax1.text(i, v + 2, str(v), ha='center', va='bottom', fontsize=9, color='white')
        
        # نمودار دوم: ویژگی‌های ویدیویی خاص
        if video_specific:
            keys_specific = list(video_specific.keys())
            vals_specific = list(video_specific.values())
            
            colors2 = [color_palette[i % len(color_palette)] for i in range(len(keys_specific))]
            bars2 = ax2.bar(range(len(keys_specific)), vals_specific, color=colors2, alpha=0.85, edgecolor='white', linewidth=1.5)
            ax2.set_xticks(range(len(keys_specific)))
            ax2.set_xticklabels(keys_specific, rotation=0, fontsize=10)
            ax2.set_ylim(0, 105)
            ax2.set_title("Video Specific Features", fontsize=13, color='white', pad=15)
            ax2.grid(alpha=0.2, linestyle='--', axis='y')
            
            # اضافه کردن مقادیر روی میله‌ها
            for i, v in enumerate(vals_specific):
                ax2.text(i, v + 2, str(v), ha='center', va='bottom', fontsize=9, color='white')
        else:
            ax2.text(0.5, 0.5, 'No video specific\nfeatures available', 
                    ha='center', va='center', fontsize=12, color='white')
            ax2.set_title("Video Specific Features", fontsize=13, color='white', pad=15)
        
        # Apply dark theme to both subplots
        for ax in [ax1, ax2]:
            ax.set_facecolor('#222')
            ax.tick_params(colors='white', labelsize=9)
            for spine in ax.spines.values():
                spine.set_color('#00ffcc')
                spine.set_linewidth(2)
            ax.yaxis.label.set_color('white')
            ax.xaxis.label.set_color('white')
        
        fig.patch.set_facecolor('#111')
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        
        return fig

    def create_image_breakdown_barchart(self, breakdown):
        """ایجاد نمودار میله‌ای برای عکس با رنگ‌های مختلف"""
        # فیلتر کردن مقادیر برای نمودار
        filtered_breakdown = {k: v for k, v in breakdown.items() 
                            if k not in ['shakeless', 'jumpcut'] and v > 0}
        
        
        labels = [k.capitalize() for k in filtered_breakdown.keys()]
        values = list(filtered_breakdown.values())
        
        # پالت رنگ‌های متنوع
        color_palette = [
            '#FF6B6B', '#4ECDC4', '#FFD166', '#06D6A0', '#118AB2', 
            '#EF476F', '#073B4C'
        ]
        
        # اگر تعداد ویژگی‌ها بیشتر از رنگ‌ها بود، رنگ‌ها را تکرار کن
        colors = [color_palette[i % len(color_palette)] for i in range(len(values))]
        
        fig, ax = plt.subplots(figsize=(9, 5))
        bars = ax.bar(range(len(values)), values, color=colors, alpha=0.85, edgecolor='white', linewidth=1.5)
        
        ax.set_xticks(range(len(values)))
        ax.set_xticklabels(labels, rotation=20, fontsize=11, fontweight='bold')
        ax.set_ylim(0, 105)
        ax.set_ylabel("Score (0-100)", fontsize=12, color='white', fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--', axis='y')
        
        # اضافه کردن مقادیر روی میله‌ها
        for i, (bar, v) in enumerate(zip(bars, values)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{v}', ha='center', va='bottom', 
                   fontsize=11, fontweight='bold', color='white')
            
            # اضافه کردن خط افقی برای نشان دادن مقدار
            ax.axhline(y=v, color=colors[i], alpha=0.3, linestyle='--', xmin=bar.get_x()/len(values), 
                      xmax=(bar.get_x() + bar.get_width())/len(values))
        
        # اضافه کردن میانگین خط
        if len(values) > 0:
            avg_value = np.mean(values)
            ax.axhline(y=avg_value, color='#00ffcc', linestyle='-', linewidth=2, alpha=0.7, label=f'Average: {avg_value:.1f}')
            ax.text(len(values) - 0.5, avg_value + 2, f'Avg: {avg_value:.1f}', 
                   color='#00ffcc', fontsize=10, fontweight='bold', ha='right')
        
        # Apply dark theme
        fig.patch.set_facecolor('#111')
        ax.set_facecolor('#222')
        ax.title.set_color('white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.tick_params(colors='white', labelsize=10, axis='both')
        
        # رنگ‌آمیزی spineها
        for spine in ax.spines.values():
            spine.set_color('#00ffcc')
            spine.set_linewidth(2)
        
        fig.tight_layout()
        return fig

    def create_breakdown_frame(self, breakdown):
        """ایجاد فریم برای نمایش breakdown با فونت بزرگ"""
        frame = QFrame()
        frame.setStyleSheet("""
            background-color: #222; 
            border-radius: 10px; 
            padding: 25px;
            margin: 15px;
        """)
        layout = QGridLayout(frame)
        
        # عنوان
        title = QLabel("📊 BREAKDOWN DETAILS")
        title.setStyleSheet("""
            color: #00ffcc; 
            font-family: Segoe UI; 
            font-size: 22px; 
            font-weight: bold;
            padding: 20px;
        """)
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title, 0, 0, 1, 2)
        
        # جزئیات با فونت بزرگتر
        row = 1
        for k, v in breakdown.items():
            name_label = QLabel(f"{k}:")
            name_label.setStyleSheet("""
                color: white; 
                font-family: Segoe UI; 
                font-size: 18px; 
                padding: 10px;
            """)
            
            value_label = QLabel(f"{v:3d}")
            value_label.setStyleSheet("""
                color: #00ffcc; 
                font-family: Segoe UI; 
                font-size: 20px; 
                font-weight: bold;
                padding: 10px;
            """)
            
            layout.addWidget(name_label, row, 0)
            layout.addWidget(value_label, row, 1)
            row += 1
        
        return frame

    def process_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select File", "", "Images/Videos (*.jpg *.png *.jpeg *.mp4 *.avi *.mov)"
        )
        if not path:
            return

        self.is_video = path.lower().endswith((".mp4", ".avi", ".mov"))
        self.file_type = "Video" if self.is_video else "Image"  # ذخیره نوع فایل

        # نمایش loading
        self.show_loading("Checking file...")
        
        try:
            # ---------- اضافه کردن بررسی طول ویدئو ----------
            if self.is_video:
                self.update_progress(10, "Checking video length...")
                
                cap = cv2.VideoCapture(path)
                fps = cap.get(cv2.CAP_PROP_FPS)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                if fps > 0:
                    duration_seconds = total_frames / fps
                    if duration_seconds > 1200:  # بیش از 20 دقیقه (1200 ثانیه)
                        self.hide_loading()
                        # نمایش پنجره هشدار کوچک
                        warning_dialog = QDialog(self)
                        warning_dialog.setWindowTitle("Video Too Long")
                        warning_dialog.setFixedSize(350, 180)  # پنجره کوچک
                        warning_dialog.setStyleSheet("""
                            QDialog {
                                background-color: #111;
                                border: 2px solid #ff9900;
                                border-radius: 10px;
                            }
                            QLabel {
                                color: white;
                                font-family: Segoe UI;
                                font-size: 14px;
                                padding: 10px;
                            }
                            QPushButton {
                                background-color: #ff9900;
                                color: black;
                                font-family: Segoe UI;
                                font-weight: bold;
                                border: none;
                                border-radius: 6px;
                                min-height: 35px;
                                padding: 8px 20px;
                                margin: 5px;
                            }
                            QPushButton:hover {
                                background-color: #e68a00;
                            }
                        """)
                        warning_dialog.setModal(True)
                        
                        layout = QVBoxLayout(warning_dialog)
                        layout.setSpacing(10)
                        layout.setContentsMargins(20, 20, 20, 20)
                        
                        # آیکون هشدار
                        warning_icon = QLabel("⚠️")
                        warning_icon.setAlignment(Qt.AlignCenter)
                        warning_icon.setStyleSheet("font-size: 40px; padding: 10px;")
                        
                        # پیام هشدار
                        warning_message = QLabel("Video is too long.\nPlease choose one under 20 minutes!")
                        warning_message.setAlignment(Qt.AlignCenter)
                        warning_message.setWordWrap(True)
                        
                        # دکمه OK
                        ok_button = QPushButton("OK")
                        ok_button.clicked.connect(warning_dialog.accept)
                        ok_button.setFixedWidth(120)
                        
                        # قرار دادن المان‌ها در layout
                        layout.addWidget(warning_icon)
                        layout.addWidget(warning_message)
                        layout.addStretch()
                        
                        button_layout = QHBoxLayout()
                        button_layout.addStretch()
                        button_layout.addWidget(ok_button)
                        button_layout.addStretch()
                        
                        layout.addLayout(button_layout)
                        
                        # نمایش پنجره هشدار
                        warning_dialog.exec_()
                        cap.release()
                        return  # خروج از تابع بدون پردازش ویدئو
                
                cap.release()
                
                self.update_progress(30, "Extracting video features...")

                cap = cv2.VideoCapture(path)  # مجدداً باز کردن برای پردازش
                ret, frame_to_show = cap.read()
                cap.release()
                feats = extract_video_features(path)
            else:
                self.update_progress(30, "Loading image...")
                frame_to_show = cv2.imread(path)
                
                self.update_progress(50, "Extracting image features...")
                feats = extract_image_features(frame_to_show)

            
            # نرمال‌سازی
            norm_feats = {
                k: normalize(v, k)
                for k, v in feats.items() if k != "subject_cls"
            }
            self.update_progress(70, "Normalizing features...")
            # کاهش امتیاز subject اگر سوژه پیدا نشده
            if feats["subject_cls"] == "none":
                norm_feats["subject"] = int(norm_feats["subject"] * 0.3)

            self.update_progress(80, "Calculating final score...")
            
            # محاسبه نمره نهایی
            self.final_score, self.regular_breakdown = final_score(
                norm_feats,
                subject_cls=feats["subject_cls"],
                is_video=self.is_video
            )

            # ذخیره داده‌ها
            self.is_enhance_mode = False
            self.regular_image = frame_to_show
            self.subject_cls = feats["subject_cls"]
            
            # تبدیل به RGB برای نمایش
            frame_rgb = cv2.cvtColor(frame_to_show, cv2.COLOR_BGR2RGB)
            img = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            self.update_progress(100, "Done!")

            # نمایش پنجره تصاویر
            self.hide_loading()
            self.show_images_preview(img, None, None, None, 
                                    self.final_score, None, 0, 
                                    self.regular_breakdown, is_enhance=False)
        
        except Exception as e:
            self.hide_loading()
            QMessageBox.critical(self, "Error", f"An error occurred: {str(e)}")

def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    # تنظیم font کلی برنامه
    font = QFont("Segoe UI", 10)
    app.setFont(font)
    
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()