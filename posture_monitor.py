import math
import os
import time
import json
from collections import deque
from typing import Optional, Dict, Any

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision


ANALYSIS_INTERVAL_SECS = float(os.environ.get("POSTURE_ANALYSIS_INTERVAL_SECS", "2.0"))
MAX_ANALYSIS_WIDTH = int(os.environ.get("POSTURE_MAX_ANALYSIS_WIDTH", "640"))
SMOOTHING_WINDOW = int(os.environ.get("POSTURE_SMOOTHING_WINDOW", "4"))
SLOUCH_CONFIDENCE_THRESHOLD = float(os.environ.get("POSTURE_SLOUCH_CONF_THRESHOLD", "0.6"))

BaseOptions = mp_python.BaseOptions
VisionRunningMode = mp_vision.RunningMode

POSE_MODEL_PATH = os.path.join(os.path.dirname(__file__), "pose_landmarker_full.task")
POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/1/pose_landmarker_full.task"
)

_pose_landmarker = None


class PostureState(Dict[str, Any]):
    """Dictionary-like container for posture state."""


def ensure_pose_model() -> None:
    """Download the pose model file if it is not present."""
    if os.path.exists(POSE_MODEL_PATH):
        return
    try:
        import requests
    except ImportError:
        raise RuntimeError(
            "The 'requests' package is required to download the pose model. "
            "Install it with 'pip install requests' in your virtual environment."
        )

    print("Downloading pose model for posture detection...")
    resp = requests.get(POSE_MODEL_URL, timeout=60)
    resp.raise_for_status()
    with open(POSE_MODEL_PATH, "wb") as f:
        f.write(resp.content)
    print("Pose model downloaded.")


def get_pose_landmarker() -> mp_vision.PoseLandmarker:
    """Create or return a singleton PoseLandmarker instance."""
    global _pose_landmarker
    if _pose_landmarker is not None:
        return _pose_landmarker

    ensure_pose_model()
    options = mp_vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=POSE_MODEL_PATH),
        running_mode=VisionRunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    _pose_landmarker = mp_vision.PoseLandmarker.create_from_options(options)
    return _pose_landmarker


def resize_for_analysis(frame):
    """Resize frame to a reasonable width while keeping aspect ratio."""
    h, w = frame.shape[:2]
    if w <= MAX_ANALYSIS_WIDTH:
        return frame
    scale = MAX_ANALYSIS_WIDTH / float(w)
    new_size = (int(w * scale), int(h * scale))
    return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)


def estimate_posture_from_pose(frame) -> PostureState:
    """
    Use a local pose model to estimate whether the user is upright or slouching.

    A front-facing webcam cannot see depth directly, so we rely on signals that
    are visible in 2D:
      1. Nose elevation above shoulders (normalised by shoulder width).
         When upright the head sits well above the shoulder line; slouching
         brings the nose down toward the shoulders.
      2. Estimated Z depth from MediaPipe (head-forward displacement).
         The pose model estimates depth even from a 2-D image.  When the head
         juts forward the nose Z becomes more negative (closer to camera) than
         the shoulder Z.
      3. Ear elevation (when ears are visible) as a corroborating signal.
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    landmarker = get_pose_landmarker()
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_image)

    if not result.pose_landmarks:
        return PostureState(
            posture="unknown",
            confidence=0.0,
            advice="Make sure your upper body is fully visible to the camera.",
        )

    lm = result.pose_landmarks[0]

    NOSE = 0
    LEFT_EAR = 7
    RIGHT_EAR = 8
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12

    nose = lm[NOSE]
    left_ear = lm[LEFT_EAR]
    right_ear = lm[RIGHT_EAR]
    ls = lm[LEFT_SHOULDER]
    rs = lm[RIGHT_SHOULDER]

    # Need at least nose and both shoulders to be clearly visible.
    if any(p.visibility < 0.5 for p in [nose, ls, rs]):
        return PostureState(
            posture="unknown",
            confidence=0.0,
            advice="Make sure your head and shoulders are clearly visible and well lit.",
        )

    shoulder_mid_x = (ls.x + rs.x) / 2.0
    shoulder_mid_y = (ls.y + rs.y) / 2.0
    shoulder_mid_z = (ls.z + rs.z) / 2.0

    # Shoulder width in normalised image coords – used as a scale reference so
    # that the metrics don't depend on how close the person is to the camera.
    shoulder_width = abs(ls.x - rs.x)
    if shoulder_width < 0.05:
        return PostureState(
            posture="unknown",
            confidence=0.0,
            advice="Step back a little so your shoulders are both fully in frame.",
        )

    # ── Signal 1: nose elevation ──────────────────────────────────────────────
    # In image coordinates y increases downward, so a high nose has a SMALLER y
    # value than the shoulders.  nose_elevation > 0 means nose is above shoulders.
    nose_elevation = (shoulder_mid_y - nose.y) / shoulder_width
    # Typical ranges:  upright ≈ 0.7 – 1.4,  slouching ≈ 0.2 – 0.55

    # ── Signal 2: head-forward depth (Z axis) ────────────────────────────────
    # MediaPipe estimates depth with hip-midpoint as origin; nose should be
    # roughly level with shoulders when upright.  A larger positive value means
    # the nose is significantly in front of (closer to the camera than) the
    # shoulder plane → head-forward posture.
    nose_z_forward = (shoulder_mid_z - nose.z) / shoulder_width
    # Typical ranges:  upright ≈ -0.1 – 0.2,  slouching ≈ 0.4 – 1.0

    # ── Signal 3: ear elevation (optional) ───────────────────────────────────
    ears_visible = left_ear.visibility >= 0.5 and right_ear.visibility >= 0.5
    if ears_visible:
        ear_mid_y = (left_ear.y + right_ear.y) / 2.0
        ear_elevation = (shoulder_mid_y - ear_mid_y) / shoulder_width
    else:
        ear_elevation = None
    # Typical ranges: upright ≈ 0.4 – 0.9,  slouching ≈ 0.0 – 0.35

    # ── Combine into a single slouch score ───────────────────────────────────
    # Each sub-score is 0 (good) → 1 (clearly slouching).
    #
    # NOTE: The estimated Z-depth from MediaPipe always reads as heavily
    # head-forward (z_fwd ≈ 1.7–2.4) for a laptop webcam regardless of actual
    # posture, so it is omitted here — it would only add a constant penalty.

    # Primary: nose elevation above shoulders (in units of shoulder width).
    # Calibrated from observed data:
    #   upright  → elev 0.97–1.09  (score should be 0)
    #   slouching → elev 0.63–0.85 (score should be > 0)
    # Penalty curve starts at 0.95 so upright readings produce 0 penalty.
    elev_score = max(0.0, min(1.0, (0.95 - nose_elevation) / 0.35))

    # Secondary: ear elevation above shoulders (in units of shoulder width).
    # Calibrated:  upright → ear 1.03–1.14,  slouching → ear 0.83–1.00
    # Penalty curve starts at 1.00.
    if ear_elevation is not None:
        ear_score = max(0.0, min(1.0, (1.00 - ear_elevation) / 0.35))
    else:
        ear_score = elev_score * 0.8

    slouch_score = 0.60 * elev_score + 0.40 * ear_score

    # ── Classify ─────────────────────────────────────────────────────────────
    # Gap between clear upright (score ≈ 0.00) and lowest slouch (score ≈ 0.20)
    # is large enough to use a tight threshold pair.
    if slouch_score < 0.12:
        posture = "upright"
        confidence = 1.0 - slouch_score
        advice = "Great posture! Keep your chest open and your ears over your shoulders."
    elif slouch_score > 0.18:
        posture = "slouching"
        confidence = min(1.0, slouch_score)
        advice = (
            "You're slouching—lift your chest, pull your head back so your ears sit over "
            "your shoulders, and lengthen your spine."
        )
    else:
        posture = "unknown"
        confidence = 0.5
        advice = (
            "Borderline posture—try sitting a little taller and drawing your head back "
            "until your ears are directly above your shoulders."
        )

    print(
        f"elev={nose_elevation:.3f}  ear={round(ear_elevation,3) if ear_elevation is not None else 'N/A':<6}  "
        f"score={slouch_score:.3f}  → {posture.upper()}",
        flush=True,
    )

    return PostureState(
        posture=posture,
        confidence=confidence,
        advice=advice,
        slouch_score=slouch_score,
        landmarks={
            "nose":           (nose.x,      nose.y),
            "left_ear":       (left_ear.x,  left_ear.y)  if ears_visible else None,
            "right_ear":      (right_ear.x, right_ear.y) if ears_visible else None,
            "left_shoulder":  (ls.x,        ls.y),
            "right_shoulder": (rs.x,        rs.y),
        },
    )


def get_smoothed_state(history: "deque[PostureState]") -> Optional[PostureState]:
    """
    Require 2 consecutive readings to agree before switching posture label.
    This prevents single-frame flickers without diluting the signal across
    many frames the way an average would.
    """
    if not history:
        return None

    # Always use the most recent state as the base for display values.
    current = history[-1].copy()

    if len(history) < 2:
        return current

    prev_posture = history[-2].get("posture", "unknown")
    curr_posture = current.get("posture", "unknown")

    # If the last two readings agree, show that posture immediately.
    if prev_posture == curr_posture:
        return current

    # Readings disagree — keep showing the previous confirmed posture to
    # avoid a single-frame flip, but update the advice/score from the latest.
    current["posture"] = prev_posture
    return current


def _posture_color(posture: str):
    """Return a BGR color for the given posture label."""
    if posture == "upright":
        return (100, 235, 100)
    if posture == "slouching":
        return (70, 70, 245)
    return (0, 210, 230)


def _lm_px(nx: float, ny: float, w: int, h: int):
    return (int(nx * w), int(ny * h))


def _blend_rect(frame, x1, y1, x2, y2, bgr, alpha):
    """Draw a filled semi-transparent rectangle."""
    ov = frame.copy()
    cv2.rectangle(ov, (x1, y1), (x2, y2), bgr, -1)
    cv2.addWeighted(ov, alpha, frame, 1 - alpha, 0, frame)


def draw_skeleton(frame, state: Optional[PostureState]) -> None:
    """Draw key-point dots and connecting lines with a soft glow effect."""
    if state is None:
        return
    lms = state.get("landmarks")
    if not lms:
        return

    h, w = frame.shape[:2]
    color   = _posture_color(state.get("posture", "unknown"))
    dim_col = tuple(int(c * 0.45) for c in color)   # dimmer version for glow

    def px(key):
        pt = lms.get(key)
        return _lm_px(pt[0], pt[1], w, h) if pt else None

    nose  = px("nose")
    l_ear = px("left_ear")
    r_ear = px("right_ear")
    l_sh  = px("left_shoulder")
    r_sh  = px("right_shoulder")

    def line(a, b, thickness=2):
        if a and b:
            cv2.line(frame, a, b, dim_col, thickness + 2, cv2.LINE_AA)  # glow
            cv2.line(frame, a, b, color,   thickness,     cv2.LINE_AA)  # core

    line(l_sh, r_sh, 3)
    line(l_sh, l_ear)
    line(r_sh, r_ear)
    line(l_ear, nose)
    line(r_ear, nose)

    if l_sh and r_sh and nose:
        mid = ((l_sh[0] + r_sh[0]) // 2, (l_sh[1] + r_sh[1]) // 2)
        cv2.line(frame, mid, nose, dim_col, 3, cv2.LINE_AA)
        cv2.line(frame, mid, nose, color,   1, cv2.LINE_AA)

    for pt in [p for p in [l_sh, r_sh] if p]:
        cv2.circle(frame, pt, 18, dim_col,         -1, cv2.LINE_AA)   # outer glow
        cv2.circle(frame, pt, 12, (255, 255, 255),  2, cv2.LINE_AA)   # white ring
        cv2.circle(frame, pt, 10, color,            -1, cv2.LINE_AA)  # core

    for pt in [p for p in [nose, l_ear, r_ear] if p]:
        cv2.circle(frame, pt, 13, dim_col,        -1, cv2.LINE_AA)
        cv2.circle(frame, pt,  9, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(frame, pt,  6, color,          -1, cv2.LINE_AA)


def draw_posture_bar(frame, state: Optional[PostureState],
                     bar_x: int, bar_y: int, bar_w: int) -> None:
    """
    Draw a wide horizontal posture bar.
    Left end = GOOD (green), right end = BAD (red).
    The coloured fill shows how far toward 'bad' the current score is.
    """
    if state is None:
        return

    font    = cv2.FONT_HERSHEY_SIMPLEX
    score   = float(state.get("slouch_score", 0.0))
    posture = state.get("posture", "unknown")
    color   = _posture_color(posture)
    bar_h   = 18
    clamped = min(score / 0.8, 1.0)

    # Background track
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                  (50, 50, 50), -1, cv2.LINE_AA)

    # Coloured fill
    fill_w = int(bar_w * clamped)
    if fill_w > 2:
        cv2.rectangle(frame, (bar_x, bar_y),
                      (bar_x + fill_w, bar_y + bar_h), color, -1)

    # Thin border
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                  (120, 120, 120), 1, cv2.LINE_AA)

    # "GOOD" / "BAD" end labels
    cv2.putText(frame, "GOOD", (bar_x, bar_y + bar_h + 18),
                font, 0.45, (100, 235, 100), 1, cv2.LINE_AA)
    bad_w = cv2.getTextSize("BAD", font, 0.45, 1)[0][0]
    cv2.putText(frame, "BAD", (bar_x + bar_w - bad_w, bar_y + bar_h + 18),
                font, 0.45, (70, 70, 245), 1, cv2.LINE_AA)


def draw_corner_brackets(frame, posture: str) -> None:
    """Draw L-shaped corner brackets (more minimal than a full rectangle border)."""
    h, w = frame.shape[:2]
    margin = 14
    length = int(min(w, h) * 0.09)

    if posture == "slouching":
        pulse     = 0.55 + 0.45 * math.sin(time.time() * 3.5)
        color     = (60, 60, int(240 * pulse))
        thickness = max(3, int(7 * pulse))
    elif posture == "upright":
        color     = (100, 235, 100)
        thickness = 3
    else:
        color     = (80, 80, 80)
        thickness = 2

    corners = [
        # top-left
        [(margin, margin), (margin + length, margin)],
        [(margin, margin), (margin, margin + length)],
        # top-right
        [(w - margin, margin), (w - margin - length, margin)],
        [(w - margin, margin), (w - margin, margin + length)],
        # bottom-left
        [(margin, h - margin), (margin + length, h - margin)],
        [(margin, h - margin), (margin, h - margin - length)],
        # bottom-right
        [(w - margin, h - margin), (w - margin - length, h - margin)],
        [(w - margin, h - margin), (w - margin, h - margin - length)],
    ]
    for a, b in corners:
        cv2.line(frame, tuple(a), tuple(b), color, thickness, cv2.LINE_AA)


def draw_overlay(frame, state: Optional[PostureState],
                 state_duration: float = 0.0) -> None:
    """Composite all visual elements onto the frame."""
    if state is None:
        return

    h, w    = frame.shape[:2]
    posture = state.get("posture", "unknown")
    advice  = state.get("advice", "")
    color   = _posture_color(posture)
    font    = cv2.FONT_HERSHEY_SIMPLEX

    # ── 1. Corner bracket border ─────────────────────────────────────────────
    draw_corner_brackets(frame, posture)

    # ── 2. Skeleton with glow ────────────────────────────────────────────────
    draw_skeleton(frame, state)

    # ── 3. Top HUD strip ─────────────────────────────────────────────────────
    # Tall enough to hold the status label + posture bar + end labels.
    hud_h = int(h * 0.20)

    # Status label  (large, horizontally centred)
    label_map = {
        "upright":   "UPRIGHT",
        "slouching": "SLOUCHING",
        "unknown":   "CALIBRATING...",
    }
    status_txt  = label_map.get(posture, posture.upper())
    stat_scale  = 1.3
    stat_thick  = 3
    (stw, sth), _ = cv2.getTextSize(status_txt, font, stat_scale, stat_thick)

    # Indent to clear the corner bracket (margin=14, length≈9% of short side)
    bracket_end = 14 + int(min(w, h) * 0.09) + 12
    stat_x = bracket_end
    stat_y = int(hud_h * 0.55)

    cv2.putText(frame, status_txt, (stat_x, stat_y),
                font, stat_scale, color, stat_thick, cv2.LINE_AA)

    # Duration timer  (right of status label)
    if state_duration > 1.0:
        mins, secs = divmod(int(state_duration), 60)
        dur_txt = f"{mins}m {secs:02d}s" if mins else f"{secs}s"
        cv2.putText(frame, dur_txt, (stat_x + stw + 14, stat_y),
                    font, 0.85, (155, 155, 155), 2, cv2.LINE_AA)

    # Posture bar  (left-aligned under the status label, capped at 400px)
    bar_w = min(400, w - stat_x - 22)
    bar_y = stat_y + 10
    draw_posture_bar(frame, state, stat_x, bar_y, bar_w)

    # ── 4. Bottom advice strip ───────────────────────────────────────────────
    if advice:
        strip_h = 58
        _blend_rect(frame, 0, h - strip_h, w, h, (15, 15, 15), 0.80)
        cv2.rectangle(frame, (0, h - strip_h), (6, h), color, -1)  # accent bar

        # Wrap advice across up to two lines
        adv_scale = 0.70
        adv_thick = 1
        max_w     = w - 30
        words     = advice.split()
        lines: list = []
        row: list   = []
        for word in words:
            test = " ".join(row + [word])
            if cv2.getTextSize(test, font, adv_scale, adv_thick)[0][0] <= max_w:
                row.append(word)
            else:
                if row:
                    lines.append(" ".join(row))
                row = [word]
            if len(lines) == 1:   # cap at 2 lines total
                break
        if row:
            lines.append(" ".join(row))

        line_h = int(cv2.getTextSize("A", font, adv_scale, adv_thick)[0][1] * 1.6)
        start_y = h - strip_h + int((strip_h - line_h * len(lines)) // 2) + line_h - 4
        for i, ln in enumerate(lines):
            cv2.putText(frame, ln, (16, start_y + i * line_h),
                        font, adv_scale, (220, 220, 220), adv_thick, cv2.LINE_AA)


def run_posture_monitor(camera_index: int = 0) -> None:
    """
    Open the default webcam, run local posture estimation,
    and overlay posture feedback in real time.
    """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Error: unable to open camera index {camera_index}.")
        print("Check macOS camera permissions and that the camera is not in use by another app.")
        return

    last_state: Optional[PostureState] = None
    smoothed_state: Optional[PostureState] = None
    history: "deque[PostureState]" = deque(maxlen=max(1, SMOOTHING_WINDOW))
    last_analysis_time = 0.0
    current_posture    = "unknown"
    posture_since      = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)

            now = time.time()
            if now - last_analysis_time >= ANALYSIS_INTERVAL_SECS:
                resized = resize_for_analysis(frame)
                try:
                    last_state = estimate_posture_from_pose(resized)
                    last_state["timestamp"] = now
                    history.append(last_state)
                    smoothed_state = get_smoothed_state(history)
                    # Track how long the user has held the current posture.
                    new_posture = (smoothed_state or last_state).get("posture", "unknown")
                    if new_posture != current_posture:
                        current_posture = new_posture
                        posture_since   = now
                except Exception as e:
                    if smoothed_state is None:
                        smoothed_state = PostureState(
                            posture="unknown", confidence=0.0,
                            advice=str(e), timestamp=now,
                        )
                    else:
                        smoothed_state["timestamp"] = now
                last_analysis_time = now

            state_duration = time.time() - posture_since
            draw_overlay(frame, smoothed_state or last_state, state_duration)

            cv2.imshow("Posture Monitor", frame)
            # Quit on 'q'
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_posture_monitor()

