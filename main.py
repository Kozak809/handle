# hand_mouse.py
# Щепотка < 0.5s -> клик. Дольше -> удержание ЛКМ (drag).
# Edge push: доставать края экрана легче — внутреннюю область камеры растягиваем на весь экран.
# Клавиши: e — toggle edge push, [ / ] — изменить поле k, g — гайды, q/Esc — выход.

import cv2
import math
import time
import pyautogui
import mediapipe as mp

pyautogui.FAILSAFE = True

# --- Параметры ---
CAM_W, CAM_H = 1280, 720
PINCH_ON  = 0.05
PINCH_OFF = 0.06
SMOOTH_ALPHA = 1
FOLLOW_ALWAYS = True

HOLD_DELAY = 0.5  # < этого — клик, >= — удержание
SHOW_GUIDES = True
WINDOW_NAME = "Hand Mouse (pinch: click <0.5s, hold otherwise)"

# ---------- EDGE PUSH ----------
EDGE_PUSH_ENABLED = True   # вкл/выкл фичу
EDGE_PUSH_K = 0.06         # доля поля с каждой стороны, которая будет «растянута»
EDGE_STEP = 0.02           # шаг клавиш [ / ]
# -------------------------------

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=1,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.7
)

SCREEN_W, SCREEN_H = pyautogui.size()

pinch_active = False
last_pinched = False
pinch_start_time = None   # monotonic()
is_pressing = False       # держим ли сейчас mouseDown
prev_cursor = None

def clamp(x, a, b): 
    return max(a, min(b, x))

def lowpass(prev, new, a):
    return new if prev is None else (prev[0]*(1-a)+new[0]*a, prev[1]*(1-a)+new[1]*a)

def edge_map(t, k):
    """
    Линеарно отображаем [k .. 1-k] -> [0 .. 1]; вне — насыщаем к 0/1.
    Так «почти у края камеры» даёт «впритык к краю экрана».
    """
    k = clamp(k, 0.0, 0.49)
    if k <= 1e-6:
        return clamp(t, 0.0, 1.0)
    return clamp((t - k) / max(1e-6, 1 - 2*k), 0.0, 1.0)

def norm_to_screen(nx, ny):
    x = int(clamp(nx, 0.0, 1.0) * SCREEN_W)
    y = int(clamp(ny, 0.0, 1.0) * SCREEN_H)
    return (x, y)

def draw_guides(frame, w, h, t, i, mid, pinched, elapsed, is_pressing):
    # fingertips + линия
    tx, ty = int(t.x * w), int(t.y * h)
    ix, iy = int(i.x * w), int(i.y * h)
    cv2.circle(frame, (tx, ty), 6, (200, 200, 255), -1)
    cv2.circle(frame, (ix, iy), 6, (200, 200, 255), -1)
    cv2.line(frame, (tx, ty), (ix, iy), (180, 200, 255), 2)

    # точка щепотки (в координатах кадра)
    mx, my = int(mid[0] * w), int(mid[1] * h)
    cv2.circle(frame, (mx, my), 7, (60, 220, 120) if pinched else (210, 210, 210), -1)

    # EDGE PUSH рамка
    if EDGE_PUSH_ENABLED:
        kx = int(EDGE_PUSH_K * w)
        ky = int(EDGE_PUSH_K * h)
        cv2.rectangle(frame, (kx, ky), (w - kx, h - ky), (90, 130, 255), 1, cv2.LINE_AA)

    # статусная строка
    if pinched:
        if not is_pressing:
            txt = f"CLICK if release < {HOLD_DELAY:.1f}s"
            color = (220, 220, 220)
        else:
            txt = "HOLDING (mouse down)"
            color = (60, 220, 120)
    else:
        txt = "Idle"
        color = (220, 220, 220)

    cv2.putText(frame, txt, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

    ep_txt = f"edge push: {'ON' if EDGE_PUSH_ENABLED else 'OFF'}  k={EDGE_PUSH_K:.2f}   g:guides  e:toggle  [/]:k  q/Esc:quit"
    cv2.putText(frame, ep_txt, (12, h-14), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv2.LINE_AA)

try:
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 960, 540)

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Не удалось прочитать кадр с камеры.")
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks and len(results.multi_hand_landmarks) > 0:
            lm = results.multi_hand_landmarks[0].landmark
            t = lm[4]; i = lm[8]

            dx = (t.x - i.x); dy = (t.y - i.y)
            dist = math.hypot(dx, dy)

            # Гистерезис
            if not pinch_active and dist < PINCH_ON:
                pinch_active = True
            elif pinch_active and dist > PINCH_OFF:
                pinch_active = False

            # Средняя точка щепотки (0..1)
            mid_nx = (t.x + i.x) * 0.5
            mid_ny = (t.y + i.y) * 0.5

            # ---------- EDGE PUSH применяем к норм. координатам ----------
            if EDGE_PUSH_ENABLED:
                mid_nx = edge_map(mid_nx, EDGE_PUSH_K)
                mid_ny = edge_map(mid_ny, EDGE_PUSH_K)
            # -------------------------------------------------------------

            target_px = norm_to_screen(mid_nx, mid_ny)
            prev_cursor = lowpass(prev_cursor, target_px, SMOOTH_ALPHA)
            cx, cy = int(prev_cursor[0]), int(prev_cursor[1])

            # Двигаем курсор (без нажатия, пока не решим: клик или hold)
            if FOLLOW_ALWAYS or pinch_active:
                pyautogui.moveTo(cx, cy, duration=0)

            now = time.monotonic()

            # Переходы состояний (клик/удержание)
            if pinch_active and not last_pinched:
                pinch_start_time = now
                is_pressing = False

            if pinch_active:
                if not is_pressing and pinch_start_time is not None and (now - pinch_start_time) >= HOLD_DELAY:
                    pyautogui.mouseDown()
                    is_pressing = True

            if not pinch_active and last_pinched:
                elapsed = (now - pinch_start_time) if pinch_start_time is not None else None
                if is_pressing:
                    try: pyautogui.mouseUp()
                    except Exception: pass
                else:
                    if elapsed is not None and elapsed < HOLD_DELAY:
                        pyautogui.click()
                pinch_start_time = None
                is_pressing = False

            last_pinched = pinch_active

            if SHOW_GUIDES:
                elapsed = (time.monotonic() - pinch_start_time) if (pinch_start_time is not None and pinch_active) else 0.0
                draw_guides(frame, w, h, t, i, ( (t.x+i.x)*0.5, (t.y+i.y)*0.5 ), pinch_active, elapsed, is_pressing)
        else:
            # Рука пропала — снимем удержание для безопасности
            if is_pressing:
                try: pyautogui.mouseUp()
                except Exception: pass
            pinch_active = False
            last_pinched = False
            pinch_start_time = None
            is_pressing = False
            prev_cursor = None
            if SHOW_GUIDES:
                cv2.putText(frame, "Show your hand in frame", (12, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (220, 220, 220), 2, cv2.LINE_AA)

        cv2.imshow(WINDOW_NAME, frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            break
        elif key == ord('g'):
            SHOW_GUIDES = not SHOW_GUIDES
        elif key == ord('e'):
            EDGE_PUSH_ENABLED = not EDGE_PUSH_ENABLED
        elif key == ord('['):
            EDGE_PUSH_K = clamp(EDGE_PUSH_K - EDGE_STEP, 0.0, 0.49)
        elif key == ord(']'):
            EDGE_PUSH_K = clamp(EDGE_PUSH_K + EDGE_STEP, 0.0, 0.49)

except pyautogui.FailSafeException:
    print("PyAutoGUI failsafe: курсор в верхнем левом углу — скрипт остановлен.")
finally:
    try:
        pyautogui.mouseUp()
    except Exception:
        pass
    hands.close()
    cap.release()
    cv2.destroyAllWindows()
