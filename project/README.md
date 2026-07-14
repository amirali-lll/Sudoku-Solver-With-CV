
# پروژه سیستم هوشمند تشخیص و حل سودوکو


هدف این پروژه طراحی یک سیستم بینایی کامپیوتر برای تشخیص و حل خودکار سودوکو از روی تصویر است. این سیستم با دریافت تصویر خام، جدول را استخراج کرده، ارقام (اعم از انگلیسی و فارسی) را به کمک یک شبکه عصبی تشخیص داده و در نهایت پاسخ را روی تصویر اولیه نمایش می‌دهد. تمرکز اصلی این توسعه بر روی بخش بینایی کامپیوتر بوده است.

![تصویر اصلی و خروجی نهایی سیستم](outputs/solved/image-6-solved.jpg)

---

## ویژگی‌ها و فازهای پیاده‌سازی

این سیستم در ۴ فاز اصلی پیاده‌سازی شده است:

### ۱. فاز ۱: استخراج جدول سودوکو
* تبدیل تصویر ورودی به حالت Grayscale و اعمال فیلترهای نویزگیری.
* پیدا کردن کانتورها (Contours) و اعمال Perspective Transform برای استخراج دقیق و تخت‌کردن جدول.
* تقسیم جدول استخراج‌شده به ۸۱ سلول و حذف حاشیه‌های خطوط جدول برای رسیدن به تصویر واضح از ارقام.

![تصویر اصلی](outputs/debug/image-6-test/01_original.jpg)
![استخراج کانتور ها ](outputs/debug/image-6-test/02_contours_preprocessed.png)


![ جدول تشخیص داده شده ](outputs/debug/image-6-test/03_detected_board_contour.jpg)
![جدول تبدیل شده به زاویه بالا](outputs/debug/image-6-test/warped_board.jpg)
![تقسیم جدول به ۸۱ سلول](outputs/debug/image-6-test/06_raw_cells_grid.jpg)
![تمام سلول های پردازش شده قبل از تبدیل به عدد](outputs/debug/image-6-test/07_processed_digits_grid.jpg)


![تبدیل سلول ها به عدد به همراه میزان اطمینان مدل](outputs/debug/image-6-test/05_predictions_with_confidence.jpg)


### ۲. فاز ۲: تشخیص ارقام
* طراحی یک مدل شبکه کانولوشنی (CNN) با خروجی ۱۰ کلاسه (اعداد ۱ تا ۹ + کلاس خانه خالی).
* آموزش مدل با استفاده از ترکیب دیتاست‌های استاندارد **MNIST**، **Hoda Dataset** (برای ارقام فارسی) و تولید داده‌های مصنوعی (Synthetic Data).
* اعمال Data Augmentation و Normalization جهت افزایش پایداری (Robustness) شبکه.

![نمودار آموزش شبکه عصبی و Loss/Accuracy](outputs/digit_report/training_curves.png)
![ماتریکس درهم‌ریختگی روی داده‌های ارزیابی](outputs/digit_report/validation_confusion_matrix.png)

### ۳. فاز ۳: حل سودوکو
* تبدیل مقادیر تشخیص داده شده از تصاویر سلول‌ها به یک ماتریس ۹×۹.
* استفاده از الگوریتم **Backtracking** جهت بررسی اعتبار اعداد (سطر، ستون، بلوک ۳×۳) و حل کامل جدول.

### ۴. فاز ۴: سیستم نهایی (Pipeline)
* اتصال تمامی اجزا از استخراج جدول، تشخیص ارقام تا حلگر در یک Pipeline یکپارچه.
* محاسبه پرسپکتیو معکوس و چاپ اعدادِ حل‌شده با حفظ زاویه و پرسپکتیو دقیق روی تصویر ورودی اصلی.

---

## ساختار پروژه

تمامی بخش‌های منطقی سیستم به صورت اسکریپت‌های ماژولار `.py` نوشته شده‌اند:

```text
.
├── data/                 # نگهداری دیتاست‌های MNIST و Hoda
├── fonts/                # فونت‌های استفاده شده برای تولید داده‌های مصنوعی ارقام
├── outputs/              # مسیر ذخیره خروجی‌ها (مدل‌ها، نمودارها، تصاویر حل شده و دیباگ)
├── raw_suduku/           # تصاویر خام سودوکو برای تست
├── scripts/              # اسکریپت‌های اجرایی آموزش، ارزیابی و استنتاج
│   ├── evaluate_digits.py
│   ├── run_pipeline.py
│   ├── run_pipeline_batch.py
│   └── train_digits.py
├── src/sudoku_cv/        # کدهای اصلی پکیج شامل Pipeline، مدل دیجیت و عملیات پردازش تصویر
├── requirements.txt      # لیست پیش‌نیازهای پروژه
└── README.md

```

---

## پیش‌نیازها و نصب

جهت راه‌اندازی پروژه محیط مجازی خود را ایجاد کرده و وابستگی‌ها را نصب کنید:

```bash
pip install -r requirements.txt

```

---

نحوه استفاده و اجرای دستورات 

پروژه به گونه‌ای طراحی شده که می‌توانید تمامی مراحل از آموزش تا استنتاج را از طریق خط فرمان (CLI) به راحتی اجرا کنید.

### آموزش مدل تشخیص ارقام

برای آموزش مدل روی ترکیب دیتاست‌های در دسترس و استخراج نتایج ارزیابی:

```bash
python scripts/train_digits.py \
    --hoda-path data/hoda/Data_hoda_full.mat \
    --epochs 8 \
    --save-path outputs/digit_model_v2.pt \
    --report-dir outputs/digit_report \
    --synthetic-samples-per-class 10000 

```

### ارزیابی مدل

برای تست دقت مدل آموزش دیده بر روی دیتاست‌های سه‌گانه و دریافت خروجی‌های بصری ارزیابی:

```bash
python scripts/evaluate_digits.py \
  --model outputs/digit_model_v2.pt \
  --hoda-path data/hoda/Data_hoda_full.mat \
  --synthetic-samples-per-class 10000 \
  --report-dir outputs/digit_evaluation

```

### حل تصویر تک (Inference)

برای وارد کردن یک تصویر خام سودوکو و دریافت خروجی نهایی حل شده به همراه ذخیره تمام مراحل دیباگ (نمایش پرسپکتیو، برش اعداد، و جدول باینری):

```bash
python scripts/run_pipeline.py \
    --image raw_suduku/images-6.jpeg \
    --model outputs/digit_model.pt \
    --save-overlay outputs/solved/image-6-solved.jpg \
    --debug-dir outputs/debug/image-6-test \
    --debug

```

*(تمامی آرتیفکت‌های دیباگ از جمله برش‌های هر سلول و ماتریکس تحلیلی در پوشه `debug-dir` ذخیره می‌شوند).*

### پردازش گروهی تصاویر (Batch Pipeline)

برای حل خودکار تمامی تصاویر خام موجود در یک پوشه:

```bash
python scripts/run_pipeline_batch.py \
    --input-dir raw_suduku \
    --model outputs/digit_model.pt \
    --overlay-dir outputs/solved \
    --debug-root outputs/debug \
    --debug

```
