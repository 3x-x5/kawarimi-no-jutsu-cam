"""
Naruto Substitution Jutsu - Toggle Invisibility (v2)
----------------------------------------------------
- Press 'b' once (while out of frame) to capture a clean background.
- Do the seal to VANISH: poof + log, then stay invisible.
- Do the seal again to REAPPEAR: just poof, no log.
- Silhouette is filled with the captured background (or smoke-gray fallback).
"""

import os
import sys
import time

import cv2
import mediapipe as mp
import numpy as np
import pyvirtualcam
import sounddevice as sd
import soundfile as sf

# ============================================================
# CONFIG
# ============================================================
W, H, FPS = 1280, 720, 30

LOG_PATH   = "log.png"
SOUND_PATH = "poof.wav"

GESTURE_HOLD_FRAMES = 5
TRANSITION_TIME     = 1.2
COOLDOWN            = 0.6
SHOW_PREVIEW        = True

# Detection thresholds
THRESH_WRIST       = 0.35
ROBUST_LANDMARKS   = [0, 1, 5, 9, 13, 17]
PAIR_MATCH_DIST    = 0.18
MIN_MATCHING_PAIRS = 4

# Auto-remove baked checkerboard from log.png
REMOVE_CHECKERBOARD = True
CHECKER_TOLERANCE   = 25

# Selfie segmentation
SEG_MODEL = 1

# Smoke during transition
SMOKE_STRENGTH = 0.95
SMOKE_COLOR    = (235, 235, 235)   # BGR

# Fallback fill if no background captured yet
FALLBACK_COLOR = (240, 240, 240)   # BGR, near-white smoke look

# ============================================================
# FILE CHECKS
# ============================================================
if not os.path.exists(LOG_PATH):
    print(f"[!] '{LOG_PATH}' not found.")
    sys.exit(1)
if not os.path.exists(SOUND_PATH):
    print(f"[!] '{SOUND_PATH}' not found.")
    sys.exit(1)

# ============================================================
# LOAD LOG
# ============================================================
log_img = cv2.imread(LOG_PATH, cv2.IMREAD_UNCHANGED)
if log_img is None:
    print("[!] Failed to load log.png")
    sys.exit(1)

if len(log_img.shape) == 2:
    log_img = cv2.cvtColor(log_img, cv2.COLOR_GRAY2BGRA)
elif log_img.shape[2] == 3:
    log_img = cv2.cvtColor(log_img, cv2.COLOR_BGR2BGRA)

if REMOVE_CHECKERBOARD:
    h, w = log_img.shape[:2]
    corners = np.array([
        log_img[5, 5, :3],
        log_img[5, w - 5, :3],
        log_img[h - 5, 5, :3],
        log_img[h - 5, w - 5, :3],
    ], dtype=np.int16)

    pixels = log_img[:, :, :3].astype(np.int16)
    mask = np.zeros((h, w), dtype=bool)
    for c in corners:
        dist = np.linalg.norm(pixels - c, axis=2)
        mask |= (dist < CHECKER_TOLERANCE)

    log_img[mask, 3] = 0
    print(f"[+] Removed {mask.sum()} checkerboard pixels "
          f"({100 * mask.sum() / mask.size:.1f}% of image)")

# ============================================================
# SOUND
# ============================================================
sound_data, sound_sr = sf.read(SOUND_PATH, dtype="float32")
print(f"[+] Loaded log {log_img.shape[1]}x{log_img.shape[0]}, "
      f"sound {sound_data.shape} @ {sound_sr} Hz")

# ============================================================
# MEDIAPIPE
# ============================================================
mp_hands  = mp.solutions.hands
mp_draw   = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles
mp_selfie = mp.solutions.selfie_segmentation

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)
segmenter = mp_selfie.SelfieSegmentation(model_selection=SEG_MODEL)

# ============================================================
# WEBCAM
# ============================================================
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
cap.set(cv2.CAP_PROP_FPS, FPS)
if not cap.isOpened():
    print("[!] Could not open webcam.")
    sys.exit(1)

# ============================================================
# HELPERS
# ============================================================
def dist_2d(a, b):
    return float(np.linalg.norm(np.array(a) - np.array(b)))

def is_naruto_seal(hand_landmarks):
    if len(hand_landmarks) < 2:
        return False, {"hands": len(hand_landmarks), "reason": "need 2 hands"}

    h1 = hand_landmarks[0].landmark
    h2 = hand_landmarks[1].landmark

    d_wrist = dist_2d([h1[0].x, h1[0].y], [h2[0].x, h2[0].y])

    matching_pairs = 0
    pair_dists = []
    for idx in ROBUST_LANDMARKS:
        d = dist_2d([h1[idx].x, h1[idx].y], [h2[idx].x, h2[idx].y])
        pair_dists.append(round(d, 3))
        if d < PAIR_MATCH_DIST:
            matching_pairs += 1

    matched = (d_wrist < THRESH_WRIST and
               matching_pairs >= MIN_MATCHING_PAIRS)

    return matched, {
        "hands": 2,
        "d_wrist": round(d_wrist, 3),
        "matching": matching_pairs,
        "pairs": pair_dists,
    }

def overlay_png(bg, fg, x, y):
    h, w = fg.shape[:2]
    bh, bw = bg.shape[:2]
    x1, y1 = max(x, 0), max(y, 0)
    x2, y2 = min(x + w, bw), min(y + h, bh)
    if x1 >= x2 or y1 >= y2:
        return bg
    fx1, fy1 = x1 - x, y1 - y
    fx2, fy2 = fx1 + (x2 - x1), fy1 + (y2 - y1)
    fg_crop = fg[fy1:fy2, fx1:fx2]
    bg_crop = bg[y1:y2, x1:x2]
    alpha = fg_crop[:, :, 3:4].astype(np.float32) / 255.0
    blended = (1.0 - alpha) * bg_crop.astype(np.float32) + \
              alpha * fg_crop[:, :, :3].astype(np.float32)
    bg[y1:y2, x1:x2] = blended.astype(np.uint8)
    return bg

def get_mask(rgb_frame):
    seg = segmenter.process(rgb_frame)
    mask = seg.segmentation_mask
    if mask.shape[:2] != (H, W):
        mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_LINEAR)
    mask = cv2.GaussianBlur(mask, (21, 21), 0)
    return np.clip(mask, 0.0, 1.0)

def apply_smoke(frame, mask):
    mask_3ch = np.stack([mask] * 3, axis=2)
    smoke = np.full_like(frame, SMOKE_COLOR, dtype=np.float32)
    base = frame.astype(np.float32)
    blended = base * (1.0 - mask_3ch * SMOKE_STRENGTH) + \
              smoke * (mask_3ch * SMOKE_STRENGTH)
    return np.clip(blended, 0, 255).astype(np.uint8)

def draw_log(frame, mask):
    binary = (mask > 0.5).astype(np.uint8)
    ys, xs = np.where(binary > 0)
    if len(xs) > 0:
        cx = int((xs.min() + xs.max()) / 2)
        cy = int((ys.min() + ys.max()) / 2)
    else:
        cx, cy = W // 2, H // 2

    target_h = int(H * 0.75)
    scale = target_h / log_img.shape[0]
    new_w = int(log_img.shape[1] * scale)
    new_h = target_h
    log_resized = cv2.resize(log_img, (new_w, new_h),
                             interpolation=cv2.INTER_AREA)
    return overlay_png(frame, log_resized,
                       cx - new_w // 2, cy - new_h // 2)

def apply_invisibility(frame, mask, background_frame):
    """Fill person silhouette with the captured background (or fallback)."""
    mask_3ch = np.stack([mask] * 3, axis=2).astype(np.float32)

    if background_frame is not None:
        # Build a soft "background glow" using the current frame's
        # far-away regions rather than a rigid snapshot.
        # Simpler: just use the captured background directly.
        fill = background_frame.astype(np.float32)
        out = frame.astype(np.float32) * (1.0 - mask_3ch) + fill * mask_3ch
    else:
        # No background captured: solid smoke-gray fill
        fill = np.full_like(frame, FALLBACK_COLOR, dtype=np.float32)
        out = frame.astype(np.float32) * (1.0 - mask_3ch) + fill * mask_3ch

    return np.clip(out, 0, 255).astype(np.uint8)

# ============================================================
# STATE
# ============================================================
visible          = True
transitioning    = False
transition_start = 0.0
transition_dir   = None      # "vanish" or "reappear"
last_trigger     = 0.0
hold_count       = 0
background_frame = None      # captured clean background

# ============================================================
# START VIRTUAL CAMERA
# ============================================================
print("[+] Starting virtual camera...")
try:
    cam = pyvirtualcam.Camera(width=W, height=H, fps=FPS,
                              fmt=pyvirtualcam.PixelFormat.BGR)
except Exception as e:
    print(f"[!] Could not start virtual camera: {e}")
    print("    Open OBS Studio and click 'Start Virtual Camera' once, then retry.")
    cap.release()
    hands.close()
    segmenter.close()
    sys.exit(1)

print(f"[+] Virtual camera ready: {cam.device}")
print()
print("    HOW TO USE:")
print("      1. Step OUT of the camera's view.")
print("      2. Press 'b' in the preview window to capture your empty room.")
print("      3. Step back in.")
print("      4. Do the seal to vanish. Do it again to return.")
print()
print("[+] Press 'q' to quit.")

# ============================================================
# MAIN LOOP
# ============================================================
try:
    with cam:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False

            # --- GESTURE ---
            result = hands.process(rgb)
            seal = False
            debug = {"hands": 0}
            if result.multi_hand_landmarks:
                seal, debug = is_naruto_seal(result.multi_hand_landmarks)

            hold_count = hold_count + 1 if seal else 0

            now = time.time()
            can_trigger = (now - last_trigger) > (TRANSITION_TIME + COOLDOWN)

            # --- TOGGLE ---
            if (hold_count >= GESTURE_HOLD_FRAMES
                    and not transitioning
                    and can_trigger):
                transitioning = True
                transition_start = now
                last_trigger = now
                visible = not visible
                transition_dir = "vanish" if not visible else "reappear"
                print(f"[+] Jutsu! -> {'INVISIBLE' if not visible else 'VISIBLE'}")
                try:
                    sd.play(sound_data, sound_sr)
                except Exception as e:
                    print(f"[!] Sound playback failed: {e}")

            if transitioning and (now - transition_start) > TRANSITION_TIME:
                transitioning = False
                transition_dir = None

            # --- RENDER ---
            if transitioning:
                mask = get_mask(rgb)
                out = apply_smoke(frame, mask)
                # Log only when vanishing, never when reappearing
                if transition_dir == "vanish":
                    out = draw_log(out, mask)

            elif not visible:
                mask = get_mask(rgb)
                out = apply_invisibility(frame, mask, background_frame)

            else:
                out = frame.copy()

            cam.send(out)
            cam.sleep_until_next_frame()

            # --- PREVIEW ---
            if SHOW_PREVIEW:
                preview = out.copy()

                if result.multi_hand_landmarks:
                    for hand_lms in result.multi_hand_landmarks:
                        mp_draw.draw_landmarks(
                            preview, hand_lms, mp_hands.HAND_CONNECTIONS,
                            mp_styles.get_default_hand_landmarks_style(),
                            mp_styles.get_default_hand_connections_style(),
                        )

                state_txt = "INVISIBLE" if not visible else "VISIBLE"
                bg_txt = "yes" if background_frame is not None else "NO (press b)"

                lines = [
                    f"state: {state_txt}",
                    f"background captured: {bg_txt}",
                    f"hands: {debug.get('hands', 0)}",
                    f"matching pairs: {debug.get('matching', '-')}/{len(ROBUST_LANDMARKS)}",
                    f"hold: {hold_count}/{GESTURE_HOLD_FRAMES}",
                ]
                y = 25
                for line in lines:
                    cv2.putText(preview, line, (10, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                                (0, 255, 255), 1)
                    y += 22

                if transitioning:
                    txt = "VANISH!" if transition_dir == "vanish" else "RETURN!"
                    cv2.putText(preview, txt, (30, 220),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.6,
                                (0, 0, 255), 3)
                elif not visible:
                    cv2.putText(preview, "INVISIBLE", (30, 220),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.4,
                                (255, 100, 255), 3)

                cv2.imshow("Naruto Substitution - press q to quit", preview)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('b'):
                    background_frame = frame.copy()
                    print("[+] Background captured.")

finally:
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    segmenter.close()
    print("[+] Closed.")
