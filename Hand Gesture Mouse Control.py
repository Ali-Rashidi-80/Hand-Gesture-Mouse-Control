import cv2  # وارد کردن OpenCV برای پردازش ویدیو
import mediapipe as mp  # وارد کردن MediaPipe جهت ردیابی دست
import numpy as np  # وارد کردن NumPy جهت محاسبات عددی
import pyautogui  # وارد کردن PyAutoGUI جهت کنترل موس
import time  # وارد کردن ماژول time جهت سنجش زمان

# ---------------- تنظیمات اولیه MediaPipe ----------------
mp_hands = mp.solutions.hands  # انتخاب ماژول دست از MediaPipe
hands_detector = mp_hands.Hands(  # ایجاد شیء برای تشخیص دست در حالت ویدیو
    static_image_mode=False,      # حالت ویدیو (تصاویر متوالی)
    max_num_hands=1,              # پردازش حداکثر یک دست
    min_detection_confidence=0.9, # آستانه حداقل اطمینان تشخیص
    min_tracking_confidence=0.9   # آستانه حداقل اطمینان پیگیری
)
mp_drawing = mp.solutions.drawing_utils  # ابزار رسم جهت نمایش نقاط دست

# ---------------- دریافت ابعاد صفحه نمایش ----------------
screen_width, screen_height = pyautogui.size()  # دریافت عرض و ارتفاع صفحه نمایش

# ---------------- تنظیمات حرکتی ----------------
SENSITIVITY = 1.8         # حساسیت حرکت موس نسبت به تغییرات دست
SMOOTHING_FACTOR = 0.99    # فاکتور نرم‌سازی جهت هموارسازی حرکت موس
SAFE_MARGIN = 10          # حاشیه ایمن جهت جلوگیری از ورود موس به گوشه‌های صفحه

# ---------------- آستانه‌های تشخیص ژست‌ها ----------------
LEFT_CLICK_THRESHOLD = 0.04   # آستانه برای کلیک چپ (فاصله بین نوک انگشت اشاره و شست)
RIGHT_CLICK_THRESHOLD = 0.04  # آستانه برای کلیک راست (فاصله بین نوک شست و نوک انگشت وسط)
DRAG_THRESHOLD = 0.05         # آستانه شروع حالت درگ (هم‌چسب شدن تمامی نوک‌ها)
DRAG_RELEASE_FACTOR = 1.5     # ضریب هیسترزیس برای خاتمه درگ (درگ ادامه دارد تا فاصله به DRAG_THRESHOLD * 1.5 برسد)
SCROLL_THRESHOLD = 0.04       # آستانه تشخیص اسکرول (فاصله بین نوک انگشت اشاره و وسط)
SCROLL_SENSITIVITY = 50       # میزان اسکرول در هر فراخوانی

# ---------------- متغیرهای کنترل وضعیت ----------------
prev_mouse_x = screen_width // 2  # مختصات اولیه موس (عرض) در مرکز صفحه
prev_mouse_y = screen_height // 2  # مختصات اولیه موس (ارتفاع) در مرکز صفحه
last_click_time = 0                # زمان آخرین کلیک جهت تشخیص دابل کلیک
double_click_count = 0             # شمارنده کلیک جهت تشخیص دابل کلیک
drag_mode = False                  # وضعیت فعال بودن حالت درگ (False به معنی عدم فعال بودن)
drag_offset_x = 0                  # اختلاف بین موقعیت موس و دست در لحظه شروع درگ
drag_offset_y = 0                  # اختلاف بین موقعیت موس و دست در لحظه شروع درگ
left_click_prev = False            # وضعیت قبلی کلیک چپ جهت تشخیص لبه (transition)
right_click_prev = False           # وضعیت قبلی کلیک راست جهت تشخیص لبه (transition)
prev_scroll_avg_y = None           # مقدار قبلی میانگین محور y برای تشخیص اسکرول

# ---------------- راه‌اندازی ضبط ویدیو از دوربین ----------------
cap = cv2.VideoCapture(0)  # شروع ضبط ویدیو از دوربین پیش‌فرض

# ---------------- تابع محاسبه فاصله اقلیدسی ----------------
def calc_distance(p1, p2):  # تابع با ورودی دو نقطه
    return np.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)  # محاسبه و بازگرداندن فاصله

# ---------------- حلقه اصلی پردازش فریم‌های ویدیو ----------------
while True:
    ret, frame = cap.read()  # خواندن یک فریم از ویدیو
    if not ret:  # در صورت عدم دریافت فریم
        break    # خروج از حلقه
    frame = cv2.flip(frame, 1)  # معکوس کردن فریم به صورت افقی (حالت آینه‌ای)
    h, w, _ = frame.shape  # دریافت ابعاد فریم
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # تبدیل فریم از BGR به RGB
    results = hands_detector.process(frame_rgb)  # پردازش فریم جهت تشخیص دست

    # مقداردهی اولیه موقعیت هدف موس بر اساس آخرین مختصات
    target_mouse_x = prev_mouse_x  
    target_mouse_y = prev_mouse_y  

    # اگر دست شناسایی شده باشد
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            lm = hand_landmarks.landmark  # استخراج نقاط کلیدی دست

            # ---------------- استخراج نقاط کلیدی ----------------
            thumb_tip = lm[4]    # نوک انگشت شست
            index_tip = lm[8]    # نوک انگشت اشاره
            middle_tip = lm[12]  # نوک انگشت وسط
            ring_tip = lm[16]    # نوک انگشت حلقه
            pinky_tip = lm[20]   # نوک انگشت کوچک

            # ---------------- تعیین موقعیت دست بر اساس نوک انگشت اشاره ----------------
            delta_x = index_tip.x - 0.5  # اختلاف نسبت به مرکز محور x
            delta_y = index_tip.y - 0.5  # اختلاف نسبت به مرکز محور y
            hand_screen_x = int((screen_width / 2) + SENSITIVITY * delta_x * screen_width)  # تبدیل به مختصات صفحه
            hand_screen_y = int((screen_height / 2) + SENSITIVITY * delta_y * screen_height)  # تبدیل به مختصات صفحه

            # به طور پیش‌فرض، اگر هیچ ژستی غیر از درگ نباشد، موس موقعیت دست را دنبال می‌کند
            target_mouse_x = hand_screen_x  
            target_mouse_y = hand_screen_y  

            current_time = time.time()  # دریافت زمان فعلی جهت تشخیص فاصله‌های زمانی بین کلیک‌ها

            # ---------------- محاسبه فواصل بین تمامی نوک‌ها جهت تشخیص حالت درگ ----------------
            distances = [
                calc_distance(thumb_tip, index_tip),   # فاصله شست - اشاره
                calc_distance(thumb_tip, middle_tip),    # فاصله شست - وسط
                calc_distance(thumb_tip, ring_tip),      # فاصله شست - حلقه
                calc_distance(thumb_tip, pinky_tip),     # فاصله شست - کوچک
                calc_distance(index_tip, middle_tip),    # فاصله اشاره - وسط
                calc_distance(index_tip, ring_tip),      # فاصله اشاره - حلقه
                calc_distance(index_tip, pinky_tip),     # فاصله اشاره - کوچک
                calc_distance(middle_tip, ring_tip),     # فاصله وسط - حلقه
                calc_distance(middle_tip, pinky_tip),    # فاصله وسط - کوچک
                calc_distance(ring_tip, pinky_tip)       # فاصله حلقه - کوچک
            ]

            # ---------------- حالت درگ و دراپ ----------------
            # اگر حالت درگ فعال نباشد و تمامی فاصله‌ها کمتر از آستانه درگ باشند، حالت درگ آغاز می‌شود.
            if not drag_mode and all(d < DRAG_THRESHOLD for d in distances):
                drag_mode = True  # فعال شدن حالت درگ
                # ثبت موقعیت دست در لحظه شروع درگ (با استفاده از نوک انگشت اشاره)
                initial_hand_x = hand_screen_x  
                initial_hand_y = hand_screen_y  
                # محاسبه اختلاف (offset) بین موقعیت موس فعلی و موقعیت دست در آغاز درگ
                drag_offset_x = prev_mouse_x - initial_hand_x  
                drag_offset_y = prev_mouse_y - initial_hand_y  
                pyautogui.mouseDown()  # فشار دادن دکمه موس (شروع درگ)
            # اگر حالت درگ فعال است و انگشتان همچنان به هم نزدیک هستند (با ضریب رهایی افزوده)
            elif drag_mode and all(d < DRAG_THRESHOLD * DRAG_RELEASE_FACTOR for d in distances):
                # به‌روزرسانی موقعیت دست در فریم فعلی
                current_hand_x = hand_screen_x  
                current_hand_y = hand_screen_y  
                # موقعیت جدید موس = موقعیت دست به علاوه offset ثبت‌شده
                target_mouse_x = current_hand_x + drag_offset_x  
                target_mouse_y = current_hand_y + drag_offset_y  
                cv2.putText(frame, "Drag Mode", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            else:
                # اگر حالت درگ فعال بوده و انگشتان باز شده‌اند، حالت درگ خاتمه می‌یابد
                if drag_mode:
                    drag_mode = False  
                    pyautogui.mouseUp()  # رهاسازی دکمه موس (پایان درگ)
                    drag_offset_x = 0  
                    drag_offset_y = 0

                # ---------------- کلیک راست (Right Click) ----------------
                # اگر فاصله بین نوک شست و نوک انگشت وسط کمتر از آستانه باشد
                if calc_distance(thumb_tip, middle_tip) < RIGHT_CLICK_THRESHOLD:
                    # تنها در زمان انتقال از غیرفعال به فعال، کلیک راست انجام شود
                    if not right_click_prev:
                        pyautogui.rightClick()  
                    right_click_prev = True  
                    cv2.putText(frame, "Right Click", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    # در این حالت، موقعیت موس ثابت نگه داشته می‌شود
                    target_mouse_x = prev_mouse_x  
                    target_mouse_y = prev_mouse_y  
                else:
                    right_click_prev = False

                # ---------------- کلیک چپ / دابل کلیک (Left Click / Double Click) ----------------
                # اگر فاصله بین نوک انگشت اشاره و شست کمتر از آستانه باشد
                if calc_distance(index_tip, thumb_tip) < LEFT_CLICK_THRESHOLD:
                    # تنها در زمان گذارشی (transition) کلیک ثبت شود
                    if not left_click_prev:
                        if current_time - last_click_time < 0.6:  # اگر فاصله زمانی از کلیک قبلی کمتر از 0.3 ثانیه باشد
                            double_click_count += 1  
                        else:
                            double_click_count = 1  
                        last_click_time = current_time  
                        if double_click_count == 2:  # اگر دابل کلیک تشخیص داده شود
                            pyautogui.doubleClick()  
                            double_click_count = 0  
                            cv2.putText(frame, "Double Click", (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        else:
                            pyautogui.click()  
                            cv2.putText(frame, "Left Click", (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    left_click_prev = True  
                else:
                    left_click_prev = False  

                # ---------------- اسکرول (Scroll Up/Down) ----------------
                if calc_distance(index_tip, middle_tip) < SCROLL_THRESHOLD:
                    scroll_avg_y = (index_tip.y + middle_tip.y) / 2  # میانگین محور y دو انگشت
                    if prev_scroll_avg_y is not None:
                        # در مختصات تصویر محور y به سمت پایین افزایش می‌یابد
                        if scroll_avg_y < prev_scroll_avg_y - 0.005:  # حرکت به بالا
                            pyautogui.scroll(SCROLL_SENSITIVITY)  
                            cv2.putText(frame, "Scroll Up", (10, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                        elif scroll_avg_y > prev_scroll_avg_y + 0.005:  # حرکت به پایین
                            pyautogui.scroll(-SCROLL_SENSITIVITY)  
                            cv2.putText(frame, "Scroll Down", (10, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    prev_scroll_avg_y = scroll_avg_y  
                else:
                    prev_scroll_avg_y = None  

                # ---------------- دکمه اسکرول (Scroll Button) ----------------
                if calc_distance(thumb_tip, pinky_tip) < SCROLL_THRESHOLD:
                    pyautogui.mouseDown(button='middle')  # فشار دادن دکمه میانی موس
                    cv2.putText(frame, "Scroll Button", (10, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                else:
                    pyautogui.mouseUp(button='middle')  # رهاسازی دکمه میانی موس

            # ---------------- رسم نقاط دست روی فریم (اختیاری) ----------------
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # ---------------- به‌روزرسانی تدریجی موقعیت موس با نرم‌سازی ----------------
    prev_mouse_x = int(prev_mouse_x + SMOOTHING_FACTOR * (target_mouse_x - prev_mouse_x))
    prev_mouse_y = int(prev_mouse_y + SMOOTHING_FACTOR * (target_mouse_y - prev_mouse_y))

    # ---------------- محدودسازی مختصات موس (برای جلوگیری از FailSafe) ----------------
    clamped_mouse_x = max(SAFE_MARGIN, min(prev_mouse_x, screen_width - SAFE_MARGIN))
    clamped_mouse_y = max(SAFE_MARGIN, min(prev_mouse_y, screen_height - SAFE_MARGIN))
    pyautogui.moveTo(clamped_mouse_x, clamped_mouse_y, duration=0.01)  # انتقال موس به موقعیت محدود شده

    # ---------------- نمایش اطلاعات موس و دستور خروج روی فریم ----------------
    cv2.putText(frame, f"Mouse: ({clamped_mouse_x}, {clamped_mouse_y})", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(frame, "Press 'q' to quit.", (10, 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow("Hand Mouse Control", frame)  # نمایش فریم نهایی
    key = cv2.waitKey(1) & 0xFF  # خواندن ورودی کاربر به مدت 1 میلی‌ثانیه
    if key == ord('q'):  # در صورت فشردن کلید 'q'
        break  # خروج از حلقه اصلی

cap.release()  # آزادسازی منبع ویدیو
cv2.destroyAllWindows()  # بستن تمامی پنجره‌های OpenCV
